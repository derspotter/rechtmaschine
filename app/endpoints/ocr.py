import os
import tempfile
from datetime import datetime
from typing import Optional
import uuid

import httpx
import pikepdf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from shared import (
    AnonymizationResult,
    DocumentCategory,
    broadcast_documents_snapshot,
    limiter,
    load_document_text,
    store_document_text,
)
from database import get_db
from models import Document

router = APIRouter()


def get_anonymization_service_settings():
    return (
        os.environ.get("ANONYMIZATION_SERVICE_URL"),
        os.environ.get("ANONYMIZATION_API_KEY"),
    )


def get_ocr_service_settings():
    return (
        os.environ.get("OCR_SERVICE_URL"),
        os.environ.get("OCR_API_KEY"),
    )


def extract_pdf_text(pdf_path: str, max_pages: int = 5) -> str:
    """Extract text from first few pages of PDF."""
    try:
        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = min(total_pages, max_pages)

            text_parts = []
            for i in range(pages_to_read):
                page = pdf.pages[i]
                if hasattr(page, "extract_text"):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"--- Page {i+1} ---\n{text}")

            return "\n\n".join(text_parts) if text_parts else ""
    except Exception as exc:
        raise Exception(f"Failed to extract text from PDF: {exc}")


async def perform_ocr_on_pdf(pdf_path: str) -> Optional[str]:
    """Perform OCR on a PDF file using the configured OCR service."""
    ocr_service_url, ocr_api_key = get_ocr_service_settings()

    if not ocr_service_url:
        print("[WARNING] OCR_SERVICE_URL not configured")
        return None

    try:
        headers = {}
        if ocr_api_key:
            headers["X-API-Key"] = ocr_api_key

        with open(pdf_path, "rb") as handle:
            pdf_content = handle.read()

        print(f"[INFO] Sending PDF to OCR service (size: {len(pdf_content)} bytes)")

        async with httpx.AsyncClient(timeout=180.0) as client:
            files = {
                "file": (os.path.basename(pdf_path), pdf_content, "application/pdf"),
            }
            response = await client.post(
                f"{ocr_service_url}/ocr",
                files=files,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            text = data.get("full_text", "")
            confidence = data.get("avg_confidence", 0.0)
            page_count = data.get("page_count", 0)

            print(
                "[SUCCESS] OCR completed: "
                f"{len(text)} characters, {page_count} pages, confidence: {confidence:.2f}"
            )
            return text

    except httpx.TimeoutException:
        print("[ERROR] OCR service timeout (>180s)")
        return None
    except httpx.HTTPStatusError as exc:
        print(f"[ERROR] OCR service HTTP error: {exc.response.status_code}")
        return None
    except Exception as exc:
        print(f"[ERROR] OCR service error: {exc}")
        return None


async def anonymize_document_text(text: str, document_type: str) -> Optional[AnonymizationResult]:
    """Call anonymization service."""
    anonymization_service_url, anonymization_api_key = get_anonymization_service_settings()

    if not anonymization_service_url:
        print("[WARNING] ANONYMIZATION_SERVICE_URL not configured")
        return None

    try:
        headers = {}
        if anonymization_api_key:
            headers["X-API-Key"] = anonymization_api_key

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{anonymization_service_url}/anonymize",
                json={"text": text, "document_type": document_type},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            return AnonymizationResult(
                anonymized_text=data["anonymized_text"],
                plaintiff_names=data["plaintiff_names"],
                confidence=data["confidence"],
                original_text=text,
                processed_characters=len(text),
            )

    except httpx.TimeoutException:
        print("[ERROR] Anonymization service timeout (>120s)")
        raise HTTPException(
            status_code=504,
            detail="Anonymization service timeout (>120s). Please retry once the model is ready.",
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        message = None
        try:
            payload = exc.response.json()
            if isinstance(payload, dict):
                message = payload.get("detail") or payload.get("message")
            else:
                message = str(payload)
        except Exception:
            message = exc.response.text or str(exc)

        detail_message = message or f"HTTP {status} from anonymization service"
        print(f"[ERROR] Anonymization service HTTP error: {status} – {detail_message}")
        raise HTTPException(status_code=status, detail=detail_message)
    except Exception as exc:
        print(f"[ERROR] Anonymization service error: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Anonymization service error: {exc}",
        )


def stitch_anonymized_text(
    extracted_text: str,
    anonymized_result: AnonymizationResult,
) -> tuple[str, int, int]:
    """Merge the anonymized excerpt with the remaining original text."""
    total_length = len(extracted_text)
    processed_chars = anonymized_result.processed_characters or total_length
    processed_chars = max(0, min(processed_chars, total_length))
    remaining_text = extracted_text[processed_chars:]

    anonymized_section = anonymized_result.anonymized_text or extracted_text[:processed_chars]

    if remaining_text:
        if anonymized_section and not anonymized_section.endswith("\n"):
            anonymized_section = f"{anonymized_section}\n\n{remaining_text}"
        else:
            anonymized_section = f"{anonymized_section}{remaining_text}"

    return anonymized_section, processed_chars, len(remaining_text)


@router.post("/documents/{document_id}/ocr")
@limiter.limit(
    "10/hour")
async def run_document_ocr(request: Request, document_id: str, db: Session = Depends(get_db)):
    """Run OCR for a document and cache the extracted text."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_path = document.file_path
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on server")

    print(f"[INFO] Manual OCR triggered for {document.filename}")
    text = await perform_ocr_on_pdf(pdf_path)
    if not text:
        raise HTTPException(
            status_code=503,
            detail="OCR service unavailable. Please ensure home PC OCR service is running.",
        )

    normalized_text = text.strip()
    if len(normalized_text) < 50:
        raise HTTPException(
            status_code=422,
            detail="OCR completed but returned insufficient text. The document may have very low quality.",
        )

    metadata = {
        "ocr_text_length": len(normalized_text),
        "processed_at": datetime.utcnow().isoformat(),
        "preview": normalized_text[:300],
    }

    store_document_text(document, normalized_text)
    document.ocr_applied = True
    document.needs_ocr = False
    document.processing_status = "ocr_ready"
    document.anonymization_metadata = metadata

    db.commit()
    broadcast_documents_snapshot(db, "ocr_completed", {"filename": document.filename})

    return {
        "status": "success",
        "document_id": document_id,
        "text_length": len(normalized_text),
        "preview": normalized_text[:200],
        "extracted_text": normalized_text,
        "message": "OCR completed successfully and cached for anonymization.",
    }


@router.post("/documents/{document_id}/anonymize")
@limiter.limit(
    "5/hour")
async def anonymize_document_endpoint(request: Request, document_id: str, db: Session = Depends(get_db)):
    """Anonymize a classified document (Anhörung or Bescheid only)."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.category not in ["Anhörung", "Bescheid"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Document type '{document.category}' does not support anonymization. "
                "Only 'Anhörung' and 'Bescheid' can be anonymized."
            ),
        )

    if document.is_anonymized and document.anonymization_metadata:
        return {
            "status": "success",
            "anonymized_text": document.anonymization_metadata.get("anonymized_text", ""),
            "plaintiff_names": document.anonymization_metadata.get("plaintiff_names", []),
            "confidence": document.anonymization_metadata.get("confidence", 0.0),
            "cached": True,
        }

    pdf_path = document.file_path
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on server")

    extracted_text = None
    ocr_used = False

    cached_text = load_document_text(document)
    if document.ocr_applied and cached_text:
        extracted_text = cached_text
        ocr_used = True
        print(f"[INFO] Using cached OCR text for {document.filename}: {len(extracted_text)} characters")

    if not extracted_text:
        try:
            extracted_text = extract_pdf_text(pdf_path, max_pages=50)
            if extracted_text and len(extracted_text.strip()) >= 100:
                print(f"[INFO] Direct text extraction successful: {len(extracted_text)} characters")
            else:
                print("[INFO] Direct extraction insufficient, trying OCR...")
                extracted_text = None
        except Exception as exc:
            print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
            extracted_text = None

    if not extracted_text:
        extracted_text = await perform_ocr_on_pdf(pdf_path)
        if extracted_text:
            ocr_used = True
            print(f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters")
        else:
            raise HTTPException(
                status_code=503,
                detail="Could not extract text from PDF. OCR service unavailable. Please ensure home PC OCR service is running.",
            )

    if not extracted_text or len(extracted_text.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail="Insufficient text extracted from PDF. The document may be empty or corrupted.",
        )

    text_for_anonymization = extracted_text[:10000]
    print(f"[INFO] Sending {len(text_for_anonymization)} characters to anonymization service")

    result = await anonymize_document_text(text_for_anonymization, document.category)

    anonymized_full_text, processed_chars, remaining_chars = stitch_anonymized_text(
        extracted_text,
        result,
    )

    store_document_text(document, extracted_text)
    document.is_anonymized = True
    document.ocr_applied = ocr_used
    document.anonymization_metadata = {
        "plaintiff_names": result.plaintiff_names,
        "confidence": result.confidence,
        "anonymized_at": datetime.utcnow().isoformat(),
        "anonymized_text": anonymized_full_text,
        "anonymized_excerpt": result.anonymized_text,
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "ocr_used": ocr_used,
    }
    document.processing_status = "completed"
    db.commit()

    broadcast_documents_snapshot(db, "anonymize", {"document_id": document_id})

    return {
        "status": "success",
        "anonymized_text": anonymized_full_text,
        "plaintiff_names": result.plaintiff_names,
        "confidence": result.confidence,
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "ocr_used": ocr_used,
        "cached": False,
    }


@router.post("/anonymize-file")
@limiter.limit(
    "5/hour")
async def anonymize_uploaded_file(request: Request, 
    document_type: str = Form(...),
    file: UploadFile = File(...),
):
    """Anonymize an uploaded PDF without storing it in the database."""
    sanitized_type = document_type.strip()
    valid_types = {DocumentCategory.ANHOERUNG.value, DocumentCategory.BESCHEID.value}
    if sanitized_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type '{document_type}'. Allowed: {', '.join(valid_types)}",
        )

    filename = (file.filename or "upload.pdf").strip()
    _, ext = os.path.splitext(filename.lower())
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmpf.write(chunk)
        tmp_path = tmpf.name

    try:
        extracted_text = None
        ocr_used = False

        try:
            extracted_text = extract_pdf_text(tmp_path, max_pages=50)
            if extracted_text and len(extracted_text.strip()) >= 100:
                print(f"[INFO] Direct text extraction successful: {len(extracted_text)} characters")
            else:
                print("[INFO] Direct extraction insufficient, trying OCR...")
                extracted_text = None
        except Exception as exc:
            print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
            extracted_text = None

        if not extracted_text:
            extracted_text = await perform_ocr_on_pdf(tmp_path)
            if extracted_text:
                ocr_used = True
                print(f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters")
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Could not extract text from PDF. OCR service unavailable. Please ensure home PC OCR service is running.",
                )

        if not extracted_text or len(extracted_text.strip()) < 100:
            raise HTTPException(
                status_code=422,
                detail="Insufficient text extracted from PDF. The document may be empty or corrupted.",
            )

        text_for_anonymization = extracted_text[:10000]
        print(f"[INFO] Sending uploaded PDF ({len(text_for_anonymization)} chars) to anonymization service")

        result = await anonymize_document_text(text_for_anonymization, sanitized_type)

        anonymized_full_text, processed_chars, remaining_chars = stitch_anonymized_text(
            extracted_text,
            result,
        )

        return {
            "status": "success",
            "filename": filename,
            "anonymized_text": anonymized_full_text,
            "plaintiff_names": result.plaintiff_names,
            "confidence": result.confidence,
            "processed_characters": processed_chars,
            "remaining_characters": remaining_chars,
            "ocr_used": ocr_used,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


__all__ = ["router"]

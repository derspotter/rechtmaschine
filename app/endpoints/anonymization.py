import os
import re
import tempfile
from datetime import datetime
from typing import Optional
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from shared import (
    AnonymizationResult,
    DocumentCategory,
    broadcast_documents_snapshot,
    ensure_service_manager_ready,
    limiter,
    load_document_text,
    store_document_text,
    ANONYMIZED_TEXT_DIR,
)
from auth import get_current_active_user
from database import get_db
from models import Document, User
from .ocr import extract_pdf_text, perform_ocr_on_file, check_pdf_needs_ocr

router = APIRouter()


def get_anonymization_service_settings():
    return (
        os.environ.get("ANONYMIZATION_SERVICE_URL"),
        os.environ.get("ANONYMIZATION_API_KEY"),
    )


async def anonymize_document_text(
    text: str, document_type: str
) -> Optional[AnonymizationResult]:
    """Call anonymization service."""
    anonymization_service_url, anonymization_api_key = (
        get_anonymization_service_settings()
    )

    if not anonymization_service_url:
        print("[WARNING] ANONYMIZATION_SERVICE_URL not configured")
        return None

    await ensure_service_manager_ready()

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
                plaintiff_names=data.get("plaintiff_names", []),
                birth_dates=data.get("birth_dates", []),
                addresses=data.get("addresses", []),
                confidence=data["confidence"],
                original_text=text,
                processed_characters=len(text),
            )

    except HTTPException:
        raise
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


def apply_regex_anonymization(
    text: str, names: list[str], birth_dates: list[str], addresses: list[str]
) -> str:
    """Apply regex-based anonymization using extracted names, DOBs, and addresses."""
    result = text

    # Replace names with [PERSON_X] placeholders
    for i, name in enumerate(names, 1):
        if name and len(name) >= 2:  # Skip very short names
            # Escape special regex characters and create case-insensitive pattern
            pattern = re.escape(name)
            result = re.sub(pattern, f"[PERSON_{i}]", result, flags=re.IGNORECASE)

    # Replace DOBs with [GEBURTSDATUM]
    for birth_date in birth_dates:
        if birth_date and len(birth_date) >= 6:
            pattern = re.escape(birth_date)
            result = re.sub(pattern, "[GEBURTSDATUM]", result, flags=re.IGNORECASE)

    # Replace addresses with [ADRESSE] placeholder
    for address in addresses:
        if address and len(address) >= 5:  # Skip very short addresses
            pattern = re.escape(address)
            result = re.sub(pattern, "[ADRESSE]", result, flags=re.IGNORECASE)

    return result


def stitch_anonymized_text(
    extracted_text: str,
    anonymized_result: AnonymizationResult,
) -> tuple[str, int, int]:
    """Merge the anonymized excerpt with the remaining original text, applying regex anonymization to remainder."""
    total_length = len(extracted_text)
    processed_chars = anonymized_result.processed_characters or total_length
    processed_chars = max(0, min(processed_chars, total_length))
    remaining_text = extracted_text[processed_chars:]

    anonymized_section = (
        anonymized_result.anonymized_text or extracted_text[:processed_chars]
    )

    if remaining_text:
        # Apply regex-based anonymization to remaining text using extracted names/addresses
        remaining_anonymized = apply_regex_anonymization(
            remaining_text,
            anonymized_result.plaintiff_names,
            anonymized_result.birth_dates,
            anonymized_result.addresses,
        )

        if anonymized_section and not anonymized_section.endswith("\n"):
            anonymized_section = f"{anonymized_section}\n\n{remaining_anonymized}"
        else:
            anonymized_section = f"{anonymized_section}{remaining_anonymized}"

    return anonymized_section, processed_chars, len(remaining_text)


@router.post("/documents/{document_id}/anonymize")
@limiter.limit("5/hour")
async def anonymize_document_endpoint(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Anonymize a classified document (Anhörung, Bescheid, or Sonstige gespeicherte Quellen)."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = (
        db.query(Document)
        .filter(Document.id == doc_uuid, Document.owner_id == current_user.id)
        .first()
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # All document types can be anonymized (names and addresses are extracted)

    if document.is_anonymized and document.anonymization_metadata:
        # Only path-based anonymized text is supported
        anonymized_text = ""
        path = document.anonymization_metadata.get("anonymized_text_path")
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    anonymized_text = f.read()
            except Exception as e:
                print(f"[ERROR] Failed to read anonymized text from file: {e}")
        else:
            print(f"[WARN] Missing anonymized_text_path for {document.filename}; reprocessing.")
            anonymized_text = ""

        if anonymized_text:
            processed_chars = document.anonymization_metadata.get(
                "processed_characters"
            )
            remaining_chars = document.anonymization_metadata.get(
                "remaining_characters"
            )
            if processed_chars is not None and remaining_chars is not None:
                input_characters = processed_chars + remaining_chars
            else:
                input_characters = len(anonymized_text)

            return {
                "status": "success",
                "anonymized_text": anonymized_text,
                "plaintiff_names": document.anonymization_metadata.get(
                    "plaintiff_names", []
                ),
                "addresses": document.anonymization_metadata.get("addresses", []),
                "confidence": document.anonymization_metadata.get("confidence", 0.0),
                "input_characters": input_characters,
                "processed_characters": processed_chars,
                "remaining_characters": remaining_chars,
                "cached": True,
            }

    pdf_path = document.file_path
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    extracted_text = None
    ocr_used = False

    cached_text = load_document_text(document)
    if document.ocr_applied and cached_text:
        extracted_text = cached_text
        ocr_used = True
        print(
            f"[INFO] Using cached OCR text for {document.filename}: {len(extracted_text)} characters"
        )

    if not extracted_text:
        # Check if we need OCR using shared logic (consistent with classification)
        # We respect the DB flag if it's True, otherwise we verify with the shared check.
        should_use_ocr = document.needs_ocr or check_pdf_needs_ocr(pdf_path)

        if should_use_ocr:
            print(
                f"[INFO] Document needs OCR (flag={document.needs_ocr}). Skipping direct extraction."
            )
            extracted_text = None
        else:
            try:
                extracted_text = extract_pdf_text(pdf_path, max_pages=50)
                # Final sanity check: even if check passed, maybe extraction failed or yielded garbage
                if extracted_text and len(extracted_text.strip()) >= 500:
                    print(
                        f"[INFO] Direct text extraction successful: {len(extracted_text)} characters"
                    )
                else:
                    print(
                        f"[INFO] Direct extraction insufficient ({len(extracted_text) if extracted_text else 0} chars), trying OCR..."
                    )
                    extracted_text = None
            except Exception as exc:
                print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
                extracted_text = None

    if not extracted_text:
        extracted_text = await perform_ocr_on_file(pdf_path)
        if extracted_text:
            ocr_used = True
            print(
                f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters"
            )
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

    text_for_anonymization = extracted_text
    print(
        f"[INFO] Sending {len(text_for_anonymization)} characters to anonymization service"
    )

    document_type = document.category
    if document_type == "Sonstiges":
        document_type = DocumentCategory.SONSTIGES.value

    result = await anonymize_document_text(text_for_anonymization, document_type)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Anonymization service unavailable. Please ensure it is running.",
        )

    anonymized_full_text, processed_chars, remaining_chars = stitch_anonymized_text(
        extracted_text,
        result,
    )

    # Save anonymized text to file
    anonymized_filename = f"{document.id}.txt"
    anonymized_path = ANONYMIZED_TEXT_DIR / anonymized_filename
    try:
        with open(anonymized_path, "w", encoding="utf-8") as f:
            f.write(anonymized_full_text)
    except Exception as e:
        print(f"[ERROR] Failed to write anonymized text to file: {e}")
        # Fallback? Or raise? For now, we proceed, but metadata might be incomplete if we rely on path.
        # Actually, if write fails, we should probably fail the request or fallback to DB storage.
        # Let's fallback to DB storage implicitly if path is missing, but here we want to enforce file.
        raise HTTPException(
            status_code=500, detail=f"Failed to save anonymized text: {e}"
        )

    store_document_text(document, extracted_text)
    document.is_anonymized = True
    document.ocr_applied = ocr_used
    document.anonymization_metadata = {
        "plaintiff_names": result.plaintiff_names,
        "birth_dates": result.birth_dates,
        "addresses": result.addresses,
        "confidence": result.confidence,
        "anonymized_at": datetime.utcnow().isoformat(),
        "anonymized_text_path": str(anonymized_path),
        "anonymized_excerpt": result.anonymized_text,
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "input_characters": len(extracted_text),
        "ocr_used": ocr_used,
    }
    document.processing_status = "completed"
    db.commit()

    broadcast_documents_snapshot(db, "anonymize", {"document_id": document_id})

    return {
        "status": "success",
        "anonymized_text": anonymized_full_text,
        "plaintiff_names": result.plaintiff_names,
        "birth_dates": result.birth_dates,
        "addresses": result.addresses,
        "confidence": result.confidence,
        "input_characters": len(extracted_text),
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "ocr_used": ocr_used,
        "cached": False,
    }


@router.post("/anonymize-file")
@limiter.limit("5/hour")
async def anonymize_uploaded_file(
    request: Request,
    document_type: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
):
    """Anonymize an uploaded PDF without storing it in the database."""
    sanitized_type = document_type.strip() or "Sonstiges"

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

        # Check if we need OCR using shared logic
        should_use_ocr = check_pdf_needs_ocr(tmp_path)

        if not should_use_ocr:
            try:
                extracted_text = extract_pdf_text(tmp_path, max_pages=50)
                # Final sanity check: even if check passed, maybe extraction failed or yielded garbage
                if extracted_text and len(extracted_text.strip()) >= 500:
                    print(
                        f"[INFO] Direct text extraction successful: {len(extracted_text)} characters"
                    )
                else:
                    print(
                        f"[INFO] Direct extraction insufficient ({len(extracted_text) if extracted_text else 0} chars), trying OCR..."
                    )
                    extracted_text = None
            except Exception as exc:
                print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
                extracted_text = None

        if not extracted_text:
            extracted_text = await perform_ocr_on_file(tmp_path)
            if extracted_text:
                ocr_used = True
                print(
                    f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters"
                )
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

        text_for_anonymization = extracted_text
        print(
            f"[INFO] Sending uploaded PDF ({len(text_for_anonymization)} chars) to anonymization service"
        )

        result = await anonymize_document_text(text_for_anonymization, sanitized_type)
        if result is None:
            raise HTTPException(
                status_code=503,
                detail="Anonymization service unavailable. Please ensure it is running.",
            )

        anonymized_full_text, processed_chars, remaining_chars = stitch_anonymized_text(
            extracted_text,
            result,
        )

        return {
            "status": "success",
            "filename": filename,
            "anonymized_text": anonymized_full_text,
            "plaintiff_names": result.plaintiff_names,
            "birth_dates": result.birth_dates,
            "addresses": result.addresses,
            "confidence": result.confidence,
            "input_characters": len(extracted_text),
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

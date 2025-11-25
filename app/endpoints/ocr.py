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
    DocumentCategory,
    broadcast_documents_snapshot,
    limiter,
    load_document_text,
    store_document_text,
)
from database import get_db
from models import Document

router = APIRouter()


def get_ocr_service_settings():
    return (
        os.environ.get("OCR_SERVICE_URL"),
        os.environ.get("OCR_API_KEY"),
    )


def extract_pdf_text(pdf_path: str, max_pages: int = 5) -> str:
    """Extract text from first few pages of PDF."""
    try:
        import fitz  # pymupdf

        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            pages_to_read = min(total_pages, max_pages)

            text_parts = []
            for i in range(pages_to_read):
                page = doc[i]
                text = page.get_text()
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


__all__ = ["router", "extract_pdf_text", "perform_ocr_on_pdf"]

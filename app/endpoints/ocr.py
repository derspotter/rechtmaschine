import mimetypes
import os
import tempfile
from datetime import datetime
from typing import Any, Optional
import uuid

import httpx
import pikepdf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from shared import (
    DocumentCategory,
    ensure_ocr_service_ready,
    broadcast_documents_snapshot,
    limiter,
    load_document_text,
    store_document_text,
)
from auth import get_current_active_user
from database import get_db
from models import Document, User

router = APIRouter()


def get_ocr_service_settings():
    return (
        os.environ.get("OCR_SERVICE_URL"),
        os.environ.get("OCR_API_KEY"),
    )


def extract_pdf_text(
    pdf_path: str,
    max_pages: Optional[int] = 5,
    include_page_headers: bool = True,
) -> str:
    """Extract text from first few pages of PDF."""
    try:
        import fitz  # pymupdf

        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            pages_to_read = total_pages if max_pages is None else min(total_pages, max_pages)

            text_parts = []
            for i in range(pages_to_read):
                page = doc[i]
                text = page.get_text()
                if text:
                    if include_page_headers:
                        text_parts.append(f"--- Page {i+1} ---\n{text}")
                    else:
                        text_parts.append(text)

            if not text_parts:
                return ""
            if include_page_headers:
                return "\n\n".join(text_parts)
            return "".join(text_parts)
    except Exception as exc:
        raise Exception(f"Failed to extract text from PDF: {exc}")



def check_pdf_needs_ocr(pdf_path: str, max_pages: int = 1, min_chars_per_page: int = 500) -> bool:
    """
    Check if a PDF needs OCR by attempting to extract text with pymupdf.
    """
    try:
        import fitz  # pymupdf

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_check = min(total_pages, max_pages)

        total_chars = 0
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text()
            meaningful_chars = sum(1 for c in text if c.isalnum())
            total_chars += meaningful_chars

        doc.close()

        threshold = min_chars_per_page * pages_to_check
        needs_ocr = total_chars < threshold

        print(
            f"[OCR CHECK] {pdf_path}: {total_chars} chars in {pages_to_check} pages "
            f"(threshold: {threshold}) -> needs_ocr={needs_ocr}"
        )
        return needs_ocr

    except Exception as exc:
        print(f"[OCR CHECK ERROR] Failed to check {pdf_path}: {exc}")
        return False


def _guess_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def _format_ocr_pages(pages: Any, fallback_text: str) -> str:
    """Preserve OCR page boundaries for citation verification."""
    if not isinstance(pages, list):
        return fallback_text

    page_entries = []
    for fallback_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue

        page_index = page.get("page_index")
        if not isinstance(page_index, int) or page_index < 1:
            page_index = fallback_index

        lines = page.get("lines")
        if not isinstance(lines, list):
            lines = []

        page_text = "\n".join(str(line).strip() for line in lines if str(line).strip()).strip()
        if not page_text:
            continue

        page_entries.append((page_index, page_text))

    if not page_entries:
        return fallback_text

    page_entries.sort(key=lambda item: item[0])
    page_blocks = [
        f"--- Seite {sequential_index} ---\n{page_text}"
        for sequential_index, (_, page_text) in enumerate(page_entries, start=1)
    ]
    return "\n\n\f\n\n".join(page_blocks)


async def perform_ocr_on_file(file_path: str) -> Optional[str]:
    """Perform OCR on a PDF or image file using the configured OCR service."""
    ocr_service_url, ocr_api_key = get_ocr_service_settings()

    if not ocr_service_url:
        print("[WARNING] OCR_SERVICE_URL not configured")
        return None

    await ensure_ocr_service_ready()

    try:
        headers = {}
        if ocr_api_key:
            headers["X-API-Key"] = ocr_api_key

        with open(file_path, "rb") as handle:
            file_content = handle.read()

        filename = os.path.basename(file_path)
        mime_type = _guess_mime_type(file_path)
        print(f"[INFO] Sending file to OCR service (size: {len(file_content)} bytes)")

        async with httpx.AsyncClient(timeout=180.0) as client:
            files = {
                "file": (filename, file_content, mime_type),
            }
            response = await client.post(
                f"{ocr_service_url}/ocr",
                files=files,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            text = _format_ocr_pages(data.get("pages"), data.get("full_text", ""))
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
async def run_document_ocr(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Run OCR for a document and cache the extracted text."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = db.query(Document).filter(
        Document.id == doc_uuid,
        Document.owner_id == current_user.id,
        Document.case_id == current_user.active_case_id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = document.file_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    print(f"[INFO] Manual OCR triggered for {document.filename}")
    text = await perform_ocr_on_file(file_path)
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


__all__ = ["router", "extract_pdf_text", "perform_ocr_on_file", "check_pdf_needs_ocr"]

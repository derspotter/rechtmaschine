import base64
import mimetypes
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, Optional
import uuid

import httpx
import pikepdf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from shared import (
    DocumentCategory,
    ensure_ocr_service_ready,
    broadcast_documents_snapshot,
    get_owned_document,
    limiter,
    load_document_text,
    should_auto_anonymize_category,
    store_document_text,
)
from auth import get_current_active_user
from database import get_db
from models import Document, OcrJob, User

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


OCR_EMBED_PDF_ENABLED = (
    os.getenv("OCR_EMBED_PDF_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
)
OCR_EMBED_PDF_TIMEOUT_SEC = float(
    (os.getenv("OCR_EMBED_PDF_TIMEOUT_SEC", "1500") or "1500").strip()
)


async def perform_ocr_pdf_embed(file_path: str) -> Optional[str]:
    """OCR a scanned PDF via /ocr-pdf and replace it in place with the
    searchable version (text layer embedded). Returns the extracted text,
    or None so the caller can fall back to plain /ocr."""
    ocr_service_url, ocr_api_key = get_ocr_service_settings()
    if not ocr_service_url:
        return None

    headers = {}
    if ocr_api_key:
        headers["X-API-Key"] = ocr_api_key

    try:
        with open(file_path, "rb") as handle:
            file_content = handle.read()
        filename = os.path.basename(file_path)
        print(f"[INFO] Sending PDF to OCR-embed service (size: {len(file_content)} bytes)")

        async with httpx.AsyncClient(timeout=OCR_EMBED_PDF_TIMEOUT_SEC) as client:
            response = await client.post(
                f"{ocr_service_url}/ocr-pdf",
                files={"file": (filename, file_content, "application/pdf")},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        print(f"[WARN] OCR-embed failed, falling back to plain OCR: {exc}")
        return None

    page_texts = data.get("page_texts") or []
    page_blocks = [
        f"--- Seite {index} ---\n{page.strip()}"
        for index, page in enumerate(page_texts, start=1)
        if str(page or "").strip()
    ]
    text = "\n\n".join(page_blocks)

    pdf_b64 = data.get("pdf_base64") or ""
    try:
        pdf_bytes = base64.b64decode(pdf_b64) if pdf_b64 else b""
    except Exception:
        pdf_bytes = b""

    if pdf_bytes.startswith(b"%PDF") and len(pdf_bytes) > 1000:
        try:
            backup_path = f"{file_path}.orig.pdf"
            if not os.path.exists(backup_path):
                shutil.copy2(file_path, backup_path)
            tmp_path = f"{file_path}.tmp"
            with open(tmp_path, "wb") as handle:
                handle.write(pdf_bytes)
            os.replace(tmp_path, file_path)
            print(
                f"[SUCCESS] Embedded OCR text layer into {os.path.basename(file_path)} "
                f"({data.get('page_count', 0)} pages)"
            )
        except Exception as exc:
            print(f"[WARN] Could not replace PDF with searchable version: {exc}")
    else:
        print("[WARN] OCR-embed returned no usable PDF; keeping original file")

    return text or None


async def perform_ocr_on_file(file_path: str, text_only: bool = False) -> Optional[str]:
    """Perform OCR on a PDF or image file using the configured OCR service.

    text_only=True forces the lightweight raw /ocr path (returns text only).
    Use it when the file is read once and discarded (e.g. memory extraction):
    the /ocr-pdf embed pipeline renders + OCRs + rewrites a searchable PDF,
    which is far slower per page and pointless when the PDF is thrown away."""
    ocr_service_url, ocr_api_key = get_ocr_service_settings()

    if not ocr_service_url:
        print("[WARNING] OCR_SERVICE_URL not configured")
        return None

    await ensure_ocr_service_ready()

    # Preferred path for PDFs: OCR + embed the text layer into the PDF itself,
    # so later AI uploads and downloads get a searchable document.
    if not text_only and OCR_EMBED_PDF_ENABLED and file_path.lower().endswith(".pdf"):
        text = await perform_ocr_pdf_embed(file_path)
        if text:
            return text

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


class OcrJobRequest(BaseModel):
    document_id: str


async def _execute_ocr_request(body: OcrJobRequest, db: Session, user: User) -> dict:
    """OCR a document and cache the extracted text.

    Shared core for the synchronous endpoint and the background OcrJob. Raises
    HTTPException on user-addressable failures; the job worker unwraps .detail
    via the JobSpec error_formatter."""
    document = get_owned_document(db, user, body.document_id)

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

    # Merge OCR stats into the EXISTING anonymization_metadata dict instead of
    # replacing it. Replacing it used to silently drop anonymized_text_path
    # while is_anonymized stayed True, so get_document_for_upload later fell
    # back to raw (un-anonymized) client text for cloud LLM calls.
    metadata = dict(document.anonymization_metadata or {})
    metadata["ocr_text_length"] = len(normalized_text)
    metadata["ocr_processed_at"] = datetime.utcnow().isoformat()
    metadata["ocr_preview"] = normalized_text[:300]
    stale_anonymized_path = metadata.pop("anonymized_text_path", None)

    was_anonymized = bool(document.is_anonymized)

    store_document_text(document, normalized_text)
    document.ocr_applied = True
    document.needs_ocr = False
    document.processing_status = "ocr_ready"
    if was_anonymized:
        # The underlying text just changed via manual OCR, so the previous
        # anonymized version no longer matches it. Never leave
        # is_anonymized=True pointing at a stale/missing anonymized file.
        document.is_anonymized = False
    document.anonymization_metadata = metadata

    db.commit()
    broadcast_documents_snapshot(db, "ocr_completed", {"filename": document.filename})

    if stale_anonymized_path and os.path.exists(stale_anonymized_path):
        try:
            os.remove(stale_anonymized_path)
        except Exception as exc:
            print(f"[WARN] Konnte alte anonymisierte Textdatei nicht löschen ({stale_anonymized_path}): {exc}")

    if should_auto_anonymize_category(document.category):
        # Same scheduling pattern as the auto-pipeline after classification
        # (see schedule_auto_anonymization in classification.py). Deferred
        # import avoids a circular import (classification.py imports ocr.py).
        from .classification import schedule_auto_anonymization

        schedule_auto_anonymization(document.id, document.owner_id, document.case_id)

    return {
        "status": "success",
        "document_id": body.document_id,
        "text_length": len(normalized_text),
        "preview": normalized_text[:200],
        "message": "OCR completed successfully and cached for anonymization.",
    }


@router.post("/documents/{document_id}/ocr")
@limiter.limit(
    "10/hour")
async def run_document_ocr(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Run OCR for a document and cache the extracted text (synchronous path)."""
    result = await _execute_ocr_request(OcrJobRequest(document_id=document_id), db, current_user)
    text = load_document_text(
        get_owned_document(db, current_user, document_id)
    )
    return {**result, "extracted_text": (text or "").strip()}


@router.post("/documents/ocr-jobs", status_code=202)
@limiter.limit("30/hour")
async def create_ocr_job(
    request: Request,
    body: OcrJobRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a background OCR job (runs in the job worker, survives app reloads)."""
    document = get_owned_document(db, current_user, body.document_id)
    job = OcrJob(
        owner_id=current_user.id,
        case_id=document.case_id,
        status="queued",
        request_payload={"document_id": str(document.id)},
        result_payload={},
        updated_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job.to_dict()


@router.get("/documents/ocr-jobs/{job_id}")
@limiter.limit("240/hour")
async def get_ocr_job(
    request: Request,
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid OCR job id format")
    job = (
        db.query(OcrJob)
        .filter(OcrJob.id == job_uuid, OcrJob.owner_id == current_user.id)
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="OCR job not found")
    return job.to_dict()


__all__ = ["router", "extract_pdf_text", "perform_ocr_on_file", "check_pdf_needs_ocr"]

"""Logical document segmentation using page-preserved OCR text."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import pikepdf
from sqlalchemy.orm import Session

from citation_qwen import (
    CITATION_QWEN_MODEL,
    call_qwen_json,
)
from citation_verifier import _page_texts_from_entry
from models import Document, DocumentSegment, User
from shared import ensure_anonymization_service_ready, store_document_text


DOCUMENT_SEGMENTATION_ENABLED = (
    os.getenv("DOCUMENT_SEGMENTATION_ENABLED", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)
DOCUMENT_SEGMENTATION_MODEL = (
    os.getenv("DOCUMENT_SEGMENTATION_MODEL", CITATION_QWEN_MODEL).strip()
    or CITATION_QWEN_MODEL
)
DOCUMENT_SEGMENTATION_PAGE_CHAR_LIMIT = int(
    (os.getenv("DOCUMENT_SEGMENTATION_PAGE_CHAR_LIMIT", "26000") or "26000").strip()
)
DOCUMENT_SEGMENTATION_NUM_PREDICT = int(
    (os.getenv("DOCUMENT_SEGMENTATION_NUM_PREDICT", "3600") or "3600").strip()
)


def load_document_pages(document: Any) -> Dict[int, str]:
    entry = {
        "id": str(getattr(document, "id", "") or ""),
        "filename": getattr(document, "filename", None),
        "file_path": getattr(document, "file_path", None),
        "extracted_text_path": getattr(document, "extracted_text_path", None),
    }
    return _page_texts_from_entry(entry)


def should_segment_document(document: Any, pages: Dict[int, str]) -> bool:
    if not pages or len(pages) <= 1:
        return False
    category = str(getattr(document, "category", "") or "").lower()
    if "rechtsprechung" in category:
        return False
    return True


def _build_ocr_text(pages: Dict[int, str]) -> str:
    return "\n\n".join(
        f"--- Seite {page} ---\n{text}"
        for page, text in sorted(pages.items())
        if str(text).strip()
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_segment(raw: Dict[str, Any], page_count: int) -> Optional[Dict[str, Any]]:
    start_page = _safe_int(raw.get("start_page"))
    end_page = _safe_int(raw.get("end_page"))
    if start_page <= 0 or end_page <= 0 or start_page > end_page:
        return None
    if start_page > page_count:
        return None
    end_page = min(end_page, page_count)
    return {
        "start_page": start_page,
        "end_page": end_page,
        "document_type": str(raw.get("document_type") or "other")[:50],
        "title": str(raw.get("title") or "").strip()[:500],
        "date": str(raw.get("date") or "").strip()[:32] or None,
        "sender_or_authority": str(raw.get("sender_or_authority") or "").strip()[:500] or None,
        "addressee": str(raw.get("addressee") or "").strip()[:500] or None,
        "topic": str(raw.get("topic") or "").strip()[:1000] or None,
        "confidence": _safe_float(raw.get("confidence")),
        "boundary_reason": str(raw.get("boundary_reason") or "").strip()[:1000] or None,
    }


def _segment_filename_slug(value: str, max_length: int = 72) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_value).strip("_")
    return (slug[:max_length].strip("_") or "segment")


def _segment_pdf_filename(document: Document, segment: DocumentSegment) -> str:
    source_stem = _segment_filename_slug(Path(document.filename or "document").stem, 96)
    title = segment.title or segment.document_type or "segment"
    title_slug = _segment_filename_slug(title, 72)
    page_label = f"p{int(segment.start_page or 0)}-{int(segment.end_page or 0)}"
    return f"{source_stem}_{title_slug}_{page_label}.pdf"


def _segment_text_from_pages(
    pages: Dict[int, str],
    start_page: int,
    end_page: int,
) -> str:
    parts = []
    for relative_page, source_page in enumerate(range(start_page, end_page + 1), start=1):
        text = (pages.get(source_page) or "").strip()
        if text:
            parts.append(f"--- Seite {relative_page} ---\n{text}")
    return "\n\n".join(parts).strip()


def _segment_explanation(document: Document, segment: DocumentSegment) -> str:
    doc_type = segment.document_type or "Dokument"
    title = segment.title or "ohne Titel"
    return (
        f"Segment ({doc_type}) aus Dokument {document.filename}, "
        f"Seiten {segment.start_page}-{segment.end_page}, Titel: {title}"
    )


def ensure_physical_document_segments(
    document: Document,
    segments: List[DocumentSegment],
    db: Session,
    current_user: User,
) -> List[Dict[str, Any]]:
    """Cut stored logical segments into child PDFs and document rows."""
    if not segments:
        return []
    if not document.file_path:
        print(f"[SEGMENTATION WARN] {document.filename}: no source file path")
        return []

    source_path = Path(document.file_path)
    if not source_path.exists() or source_path.suffix.lower() != ".pdf":
        print(f"[SEGMENTATION WARN] {document.filename}: source PDF unavailable")
        return []

    pages = load_document_pages(document)
    segment_dir = source_path.parent / f"{source_path.stem}_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    created_documents: List[Dict[str, Any]] = []

    try:
        with pikepdf.Pdf.open(str(source_path)) as source_pdf:
            total_pages = len(source_pdf.pages)

            for segment in segments:
                start_page = int(segment.start_page or 0)
                end_page = int(segment.end_page or 0)
                if start_page <= 0 or end_page < start_page or start_page > total_pages:
                    print(
                        f"[SEGMENTATION WARN] {document.filename}: invalid segment "
                        f"{start_page}-{end_page}"
                    )
                    continue
                end_page = min(end_page, total_pages)
                output_filename = _segment_pdf_filename(document, segment)
                output_path = segment_dir / output_filename

                child_pdf = pikepdf.Pdf.new()
                for page_idx in range(start_page - 1, end_page):
                    child_pdf.pages.append(source_pdf.pages[page_idx])
                child_pdf.save(str(output_path))

                segment_explanation = _segment_explanation(document, segment)
                existing_doc = (
                    db.query(Document)
                    .filter(
                        Document.filename == output_filename,
                        Document.owner_id == current_user.id,
                        Document.case_id == current_user.active_case_id,
                    )
                    .first()
                )
                if existing_doc:
                    child_doc = existing_doc
                    child_doc.category = document.category
                    child_doc.confidence = segment.confidence or document.confidence or 1.0
                    child_doc.explanation = segment_explanation
                    child_doc.file_path = str(output_path)
                    child_doc.outline_title = segment.title
                else:
                    child_doc = Document(
                        filename=output_filename,
                        category=document.category,
                        confidence=segment.confidence or document.confidence or 1.0,
                        explanation=segment_explanation,
                        file_path=str(output_path),
                        owner_id=current_user.id,
                        case_id=current_user.active_case_id,
                        outline_title=segment.title,
                        processing_status="pending",
                    )
                    db.add(child_doc)

                db.flush()
                segment_text = _segment_text_from_pages(pages, start_page, end_page)
                if segment_text:
                    store_document_text(child_doc, segment_text)
                    child_doc.ocr_applied = True
                    child_doc.needs_ocr = False
                    child_doc.processing_status = "ocr_ready"
                else:
                    child_doc.needs_ocr = True

                metadata = dict(segment.metadata_ or {})
                metadata["created_document_id"] = str(child_doc.id)
                metadata["created_filename"] = output_filename
                segment.metadata_ = metadata

                created_documents.append(
                    {
                        "id": str(child_doc.id),
                        "filename": child_doc.filename,
                        "start_page": start_page,
                        "end_page": end_page,
                        "ocr_text_cached": bool(segment_text),
                    }
                )
    except Exception as exc:
        print(f"[SEGMENTATION ERROR] Failed to cut {document.filename}: {exc}")
        raise RuntimeError(f"PDF segmentation failed: {exc}") from exc

    db.commit()
    return created_documents


async def segment_document_with_qwen(document: Any) -> Dict[str, Any]:
    if not DOCUMENT_SEGMENTATION_ENABLED:
        return {
            "segments": [],
            "skipped": True,
            "reason": "Document segmentation disabled.",
            "model": DOCUMENT_SEGMENTATION_MODEL,
        }

    pages = load_document_pages(document)
    if not should_segment_document(document, pages):
        return {
            "segments": [],
            "skipped": True,
            "reason": "Document does not need segmentation.",
            "page_count": len(pages),
            "model": DOCUMENT_SEGMENTATION_MODEL,
        }

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL", "").strip()
    if not service_url:
        raise RuntimeError("ANONYMIZATION_SERVICE_URL is not configured")
    await ensure_anonymization_service_ready()
    ocr_text = _build_ocr_text(pages)
    prompt = f"""/no_think
Segmentiere den OCR-Text anhand der Seitenmarker in logische Dokumente/Einheiten.
Beispiele: Brief, Formular, Bescheinigung, E-Mail-Thread, Informationsblatt, Anhang.
Mehrere Seiten können zu einer Einheit gehören. Trenne nur bei einem echten neuen Dokument.
Gib nur JSON zurück.

JSON-Schema:
{{
  "segments": [
    {{
      "start_page": 1,
      "end_page": 2,
      "document_type": "email_thread|letter|form|certificate|info_sheet|other",
      "title": "kurzer Titel",
      "date": "YYYY-MM-DD oder null",
      "sender_or_authority": "kurz oder null",
      "addressee": "kurz oder null",
      "topic": "kurze Inhaltsangabe",
      "confidence": 0.0,
      "boundary_reason": "warum diese Seiten zusammengehören"
    }}
  ]
}}

DOKUMENT:
{getattr(document, "filename", "")}

OCR-TEXT:
{ocr_text[:DOCUMENT_SEGMENTATION_PAGE_CHAR_LIMIT]}
"""
    parsed = await call_qwen_json(
        service_url,
        prompt,
        num_predict=DOCUMENT_SEGMENTATION_NUM_PREDICT,
        temperature=0.0,
        model=DOCUMENT_SEGMENTATION_MODEL,
    )
    raw_segments = parsed.get("segments") or []
    if not isinstance(raw_segments, list):
        raw_segments = []
    segments = [
        segment
        for segment in (_normalize_segment(raw, len(pages)) for raw in raw_segments if isinstance(raw, dict))
        if segment is not None
    ]
    return {
        "segments": segments,
        "skipped": False,
        "page_count": len(pages),
        "model": DOCUMENT_SEGMENTATION_MODEL,
        "raw_segment_count": len(raw_segments),
    }

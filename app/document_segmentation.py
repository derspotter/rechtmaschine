"""Logical document segmentation using page-preserved OCR text."""

from __future__ import annotations

import asyncio
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pikepdf
from sqlalchemy.orm import Session

from citation_qwen import (
    CITATION_QWEN_MODEL,
    call_qwen_json,
    parse_qwen_json_response,
)
from citation_verifier import _page_texts_from_entry
from database import SessionLocal
from models import Document, DocumentSegment, User
from shared import (
    broadcast_documents_snapshot,
    ensure_anonymization_service_ready,
    store_document_text,
)


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
DOCUMENT_SEGMENTATION_VISION_ENABLED = (
    os.getenv("DOCUMENT_SEGMENTATION_VISION_ENABLED", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)
DOCUMENT_SEGMENTATION_VISION_MAX_PAGES = int(
    (os.getenv("DOCUMENT_SEGMENTATION_VISION_MAX_PAGES", "24") or "24").strip()
)
DOCUMENT_SEGMENTATION_VISION_ZOOM = float(
    (os.getenv("DOCUMENT_SEGMENTATION_VISION_ZOOM", "1.0") or "1.0").strip()
)
SEGMENT_CHILD_AUTO_PROCESS_ENABLED = (
    os.getenv("SEGMENT_CHILD_AUTO_PROCESS_ENABLED", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)
SEGMENT_CHILD_AUTO_OCR_ENABLED = (
    os.getenv("SEGMENT_CHILD_AUTO_OCR_ENABLED", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)
SEGMENT_CHILD_AUTO_ANON_ENABLED = (
    os.getenv("SEGMENT_CHILD_AUTO_ANON_ENABLED", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)
NON_TEXT_SEGMENT_TYPES = {"photo_evidence"}
PRIVACY_SENSITIVE_SEGMENT_TYPES = {
    "medical_document",
    "identity_document",
    "certificate",
    "form",
    "letter",
    "email_thread",
}
MANDANTENUNTERLAGEN_CATEGORY = "Mandantenunterlagen"
SONSTIGE_CATEGORY = "Sonstige gespeicherte Quellen"


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
    return _category_allows_segmentation(document)


def _category_allows_segmentation(document: Any) -> bool:
    category = str(getattr(document, "category", "") or "").lower()
    if "rechtsprechung" in category:
        return False
    return True


def _pdf_page_count(document: Any) -> int:
    file_path = getattr(document, "file_path", None)
    if not file_path:
        return 0
    path = Path(file_path)
    if path.suffix.lower() != ".pdf" or not path.exists():
        return 0
    try:
        with pikepdf.Pdf.open(str(path)) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


async def _call_qwen_vision_pdf_endpoint(
    service_url: str,
    document: Any,
    prompt: str,
    *,
    num_predict: int,
    temperature: float,
    model: str,
    max_pages: int,
) -> tuple[Dict[str, Any], int, int]:
    file_path = getattr(document, "file_path", None)
    if not file_path:
        raise RuntimeError("Document has no file_path")
    path = Path(file_path)
    if path.suffix.lower() != ".pdf" or not path.exists():
        raise RuntimeError("Document is not an available PDF")

    data = {
        "prompt": prompt,
        "model": model,
        "max_pages": str(max_pages),
        "zoom": str(DOCUMENT_SEGMENTATION_VISION_ZOOM),
        "num_predict": str(num_predict),
        "temperature": str(temperature),
    }
    async with httpx.AsyncClient(timeout=300) as client:
        with open(path, "rb") as fh:
            response = await client.post(
                f"{service_url.rstrip('/')}/qwen-vision-segment-pdf",
                data=data,
                files={"file": (path.name, fh, "application/pdf")},
            )
        response.raise_for_status()
        raw = response.json()
    return (
        parse_qwen_json_response(raw),
        int(raw.get("page_count") or 0),
        int(raw.get("image_count") or 0),
    )


def _build_ocr_text(pages: Dict[int, str]) -> str:
    return "\n\n".join(
        f"--- Seite {page} ---\n{text}"
        for page, text in sorted(pages.items())
        if str(text).strip()
    )


def _build_vision_segmentation_prompt(document: Any, page_count: int, image_count: int) -> str:
    return f"""/no_think
Du bekommst die Seiten eines PDF-Dokuments als Bilder in der richtigen Reihenfolge.
Segmentiere diese Seiten in logische Dokumente/Einheiten und gib ihnen kurze, sachliche Titel.

Wichtig:
- Bild 1 entspricht Seite 1, Bild 2 entspricht Seite 2 usw.
- Trenne nur bei einem echten neuen Dokument oder eigenständigen Beleg.
- Mehrere Seiten können zu einer Einheit gehören.
- Wenn eine Seite nur ein Foto/Beweisbild ohne nennenswerten Text ist, klassifiziere sie als "photo_evidence".
- Wenn die Seite ein fotografiertes Textdokument ist, klassifiziere sie nach dem Dokumenttyp, nicht als Foto.
- Erfinde keine Namen oder Daten. Nutze nur, was visuell erkennbar ist.
- Wenn nicht alle PDF-Seiten angehängt sind, segmentiere nur die angehängten Seiten.
- Gib ausschließlich JSON zurück.

JSON-Schema:
{{
  "segments": [
    {{
      "start_page": 1,
      "end_page": 2,
      "document_type": "email_thread|letter|form|certificate|medical_document|identity_document|photo_evidence|info_sheet|other",
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

DOKUMENTNAME:
{getattr(document, "filename", "")}

PDF-SEITEN GESAMT: {page_count}
ANGEHÄNGTE BILDER: {image_count}
"""


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


def _normalized_segment_type(segment: DocumentSegment | Any) -> str:
    return str(getattr(segment, "document_type", "") or "").strip().lower()


def should_ocr_segment_child(segment: DocumentSegment | Any) -> bool:
    """Return whether a segmented child is likely text-like enough to OCR."""
    return _normalized_segment_type(segment) not in NON_TEXT_SEGMENT_TYPES


def should_anonymize_segment_child(document: Document, segment: DocumentSegment | Any) -> bool:
    """Return whether a segmented child is likely to contain personal data worth anonymizing."""
    if not SEGMENT_CHILD_AUTO_ANON_ENABLED:
        return False
    category = str(getattr(document, "category", "") or "")
    if category == MANDANTENUNTERLAGEN_CATEGORY:
        return True
    if category == SONSTIGE_CATEGORY and _normalized_segment_type(segment) in PRIVACY_SENSITIVE_SEGMENT_TYPES:
        return True
    return False


def _broadcast_documents_snapshot_safe(db: Session, event_type: str, payload: Dict[str, Any]) -> None:
    try:
        broadcast_documents_snapshot(db, event_type, payload)
    except Exception as exc:
        print(f"[SEGMENT POSTPROCESS WARN] Failed to broadcast {event_type}: {exc}")


async def segment_document_with_qwen_vision(document: Any) -> Dict[str, Any]:
    if not DOCUMENT_SEGMENTATION_VISION_ENABLED:
        return {
            "segments": [],
            "skipped": True,
            "reason": "Vision segmentation disabled.",
            "model": DOCUMENT_SEGMENTATION_MODEL,
        }
    if not _category_allows_segmentation(document):
        return {
            "segments": [],
            "skipped": True,
            "reason": "Document category should not be segmented.",
            "model": DOCUMENT_SEGMENTATION_MODEL,
        }

    page_count = _pdf_page_count(document)
    if page_count <= 1:
        return {
            "segments": [],
            "skipped": True,
            "reason": "Document does not need vision segmentation.",
            "page_count": page_count,
            "model": DOCUMENT_SEGMENTATION_MODEL,
        }

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL", "").strip()
    if not service_url:
        raise RuntimeError("ANONYMIZATION_SERVICE_URL is not configured")
    await ensure_anonymization_service_ready()
    max_pages = min(page_count, DOCUMENT_SEGMENTATION_VISION_MAX_PAGES)
    prompt = _build_vision_segmentation_prompt(document, page_count, max_pages)
    try:
        parsed, service_page_count, image_count = await _call_qwen_vision_pdf_endpoint(
            service_url,
            document,
            prompt,
            num_predict=DOCUMENT_SEGMENTATION_NUM_PREDICT,
            temperature=0.0,
            model=DOCUMENT_SEGMENTATION_MODEL,
            max_pages=max_pages,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {404, 405}:
            raise RuntimeError(
                "Desktop service manager does not expose /qwen-vision-segment-pdf yet. "
                "Pull and restart the desktop service manager."
            ) from exc
        raise
    except httpx.RequestError as exc:
        raise RuntimeError(f"Qwen vision segmentation service is unreachable: {exc}") from exc

    page_count = service_page_count or page_count
    image_count = image_count or max_pages
    raw_segments = parsed.get("segments") or []
    if not isinstance(raw_segments, list):
        raw_segments = []
    segments = [
        segment
        for segment in (_normalize_segment(raw, image_count) for raw in raw_segments if isinstance(raw, dict))
        if segment is not None
    ]
    return {
        "segments": segments,
        "skipped": False,
        "page_count": page_count,
        "image_count": image_count,
        "model": DOCUMENT_SEGMENTATION_MODEL,
        "raw_segment_count": len(raw_segments),
        "segmentation_source": "vision",
    }


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
                auto_ocr_recommended = should_ocr_segment_child(segment)
                auto_anonymize_recommended = should_anonymize_segment_child(child_doc, segment)
                already_ocr_ready = bool(child_doc.ocr_applied and not child_doc.needs_ocr)
                already_anonymized = bool(child_doc.is_anonymized)
                segment_text = _segment_text_from_pages(pages, start_page, end_page)
                if already_anonymized:
                    auto_anonymize_recommended = False
                    auto_ocr_recommended = False
                elif already_ocr_ready:
                    auto_ocr_recommended = False
                    if auto_anonymize_recommended:
                        child_doc.processing_status = "anon_pending"
                elif segment_text:
                    store_document_text(child_doc, segment_text)
                    child_doc.ocr_applied = True
                    child_doc.needs_ocr = False
                    child_doc.processing_status = (
                        "anon_pending" if auto_anonymize_recommended else "ocr_ready"
                    )
                else:
                    child_doc.ocr_applied = False
                    child_doc.needs_ocr = bool(auto_ocr_recommended)
                    if auto_anonymize_recommended:
                        child_doc.processing_status = "anon_pending"
                    elif auto_ocr_recommended:
                        child_doc.processing_status = "ocr_pending"
                    else:
                        child_doc.processing_status = "completed"

                metadata = dict(segment.metadata_ or {})
                metadata["created_document_id"] = str(child_doc.id)
                metadata["created_filename"] = output_filename
                metadata["auto_ocr_recommended"] = auto_ocr_recommended
                metadata["auto_anonymize_recommended"] = auto_anonymize_recommended
                segment.metadata_ = metadata

                created_documents.append(
                    {
                        "id": str(child_doc.id),
                        "filename": child_doc.filename,
                        "start_page": start_page,
                        "end_page": end_page,
                        "ocr_text_cached": bool(segment_text),
                        "auto_ocr_recommended": auto_ocr_recommended,
                        "auto_anonymize_recommended": auto_anonymize_recommended,
                        "document_type": segment.document_type,
                    }
                )
    except Exception as exc:
        print(f"[SEGMENTATION ERROR] Failed to cut {document.filename}: {exc}")
        raise RuntimeError(f"PDF segmentation failed: {exc}") from exc

    db.commit()
    return created_documents


async def _ocr_segment_child_document(db: Session, document: Document) -> None:
    from endpoints.ocr import perform_ocr_on_file

    if document.ocr_applied and not document.needs_ocr:
        return
    if not document.file_path or not Path(document.file_path).exists():
        raise RuntimeError("Segment child file is missing")

    document.processing_status = "ocr_processing"
    db.commit()
    _broadcast_documents_snapshot_safe(db, "segment_child_ocr_started", {"document_id": str(document.id)})

    text = await perform_ocr_on_file(document.file_path)
    normalized_text = (text or "").strip()
    if len(normalized_text) < 50:
        document.processing_status = "ocr_failed"
        metadata = dict(document.anonymization_metadata or {})
        metadata["auto_ocr_error"] = "OCR returned insufficient text"
        metadata["auto_ocr_failed_at"] = datetime.utcnow().isoformat()
        metadata["ocr_text_length"] = len(normalized_text)
        document.anonymization_metadata = metadata
        db.commit()
        _broadcast_documents_snapshot_safe(db, "segment_child_ocr_failed", {"document_id": str(document.id)})
        return

    store_document_text(document, normalized_text)
    document.ocr_applied = True
    document.needs_ocr = False
    document.processing_status = "ocr_ready"
    metadata = dict(document.anonymization_metadata or {})
    metadata["ocr_text_length"] = len(normalized_text)
    metadata["ocr_processed_at"] = datetime.utcnow().isoformat()
    metadata["ocr_source"] = "segment_child_auto_processing"
    document.anonymization_metadata = metadata
    db.commit()
    _broadcast_documents_snapshot_safe(db, "segment_child_ocr_completed", {"document_id": str(document.id)})


async def process_segment_child_document_bg(
    document_id: str,
    owner_id: str,
    case_id: Optional[str],
    *,
    auto_ocr: bool,
    auto_anonymize: bool,
) -> None:
    """Run recommended OCR/anonymization for one segmented child document."""

    if not SEGMENT_CHILD_AUTO_PROCESS_ENABLED:
        return

    db = SessionLocal()
    try:
        document = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.owner_id == owner_id,
                Document.case_id == case_id,
            )
            .first()
        )
        if not document:
            print(f"[SEGMENT POSTPROCESS] Document not found: {document_id}")
            return

        if auto_anonymize:
            from endpoints.anonymization import anonymize_document_record

            if document.is_anonymized:
                return
            document.processing_status = "anonymizing"
            db.commit()
            _broadcast_documents_snapshot_safe(
                db,
                "segment_child_anonymize_started",
                {"document_id": str(document.id)},
            )
            await anonymize_document_record(db, document)
            return

        if auto_ocr and SEGMENT_CHILD_AUTO_OCR_ENABLED:
            await _ocr_segment_child_document(db, document)
    except Exception as exc:
        db.rollback()
        print(f"[SEGMENT POSTPROCESS ERROR] document_id={document_id}: {exc}")
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            if auto_anonymize and document.is_anonymized:
                print(
                    f"[SEGMENT POSTPROCESS WARN] document_id={document_id}: "
                    "anonymization completed but post-commit notification failed"
                )
                return
            document.processing_status = "anon_failed" if auto_anonymize else "ocr_failed"
            metadata = dict(document.anonymization_metadata or {})
            metadata["segment_child_postprocess_error"] = str(exc)
            metadata["segment_child_postprocess_failed_at"] = datetime.utcnow().isoformat()
            document.anonymization_metadata = metadata
            db.commit()
            _broadcast_documents_snapshot_safe(
                db,
                "segment_child_postprocess_failed",
                {"document_id": str(document.id)},
            )
    finally:
        db.close()


def schedule_segment_child_post_processing(
    created_documents: List[Dict[str, Any]],
    current_user: User,
) -> None:
    """Schedule sequential OCR/anonymization for newly cut segment children."""

    if not SEGMENT_CHILD_AUTO_PROCESS_ENABLED or not created_documents:
        return
    work_items = [
        item
        for item in created_documents
        if item.get("auto_anonymize_recommended")
        or (SEGMENT_CHILD_AUTO_OCR_ENABLED and item.get("auto_ocr_recommended"))
    ]
    if not work_items:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def runner() -> None:
        for item in work_items:
            await process_segment_child_document_bg(
                str(item.get("id")),
                str(current_user.id),
                str(current_user.active_case_id) if current_user.active_case_id else None,
                auto_ocr=bool(item.get("auto_ocr_recommended")),
                auto_anonymize=bool(item.get("auto_anonymize_recommended")),
            )

    loop.create_task(runner())


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
        vision_result = await segment_document_with_qwen_vision(document)
        if not vision_result.get("skipped"):
            return vision_result
        return {
            "segments": [],
            "skipped": True,
            "reason": vision_result.get("reason") or "Document does not need segmentation.",
            "page_count": vision_result.get("page_count", len(pages)),
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
        "segmentation_source": "ocr_text",
    }

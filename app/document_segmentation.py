"""Logical document segmentation using page-preserved OCR text."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from citation_qwen import (
    CITATION_QWEN_MODEL,
    call_qwen_json,
)
from citation_verifier import _page_texts_from_entry
from shared import ensure_anonymization_service_ready


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

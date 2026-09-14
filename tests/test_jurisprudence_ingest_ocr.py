"""Scanned asyl.net PDFs (no text layer) must fall back to the OCR service
instead of being dropped as SHORT — 15 decisions were lost that way between
June and September 2026 and aged out of the 45-day refresh window.

Run: .venv/bin/python -m pytest tests/test_jurisprudence_ingest_ocr.py -q
"""
import fitz
import httpx
import pytest

import jurisprudence_ingest as ji

LONG = "Das Verwaltungsgericht hat die Klage abgewiesen. " * 30


def _pdf(text: str | None) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text[:80])
        page.insert_text((72, 100), text[80:160])
        for i in range(2, 20):
            page.insert_text((72, 72 + 14 * i), text[80 * i : 80 * (i + 1)])
    else:
        # image-only stand-in: a drawn box, no text objects at all
        page.draw_rect(fitz.Rect(50, 50, 300, 300), fill=(0, 0, 0))
    data = doc.tobytes()
    doc.close()
    return data


def test_text_layer_pdf_is_used_without_ocr(monkeypatch):
    monkeypatch.setattr(ji, "ocr_pdf_bytes", lambda data: pytest.fail("OCR must not run"))
    text, method = ji.pdf_text_with_ocr_fallback(_pdf(LONG))
    assert method == "text_layer"
    assert "Verwaltungsgericht" in text


def test_image_only_pdf_falls_back_to_ocr(monkeypatch):
    monkeypatch.setattr(ji, "ocr_pdf_bytes", lambda data: LONG)
    text, method = ji.pdf_text_with_ocr_fallback(_pdf(None))
    assert method == "ocr"
    assert text == LONG


def test_ocr_failure_yields_short_text_not_exception(monkeypatch):
    def boom(data):
        raise httpx.ConnectError("ocr host asleep")

    monkeypatch.setattr(ji, "ocr_pdf_bytes", boom)
    text, method = ji.pdf_text_with_ocr_fallback(_pdf(None))
    assert method == "ocr_failed"
    assert len(text) < 400


def test_ocr_pdf_bytes_posts_pdf_to_ocr_service_and_keeps_page_breaks(monkeypatch):
    monkeypatch.setenv("OCR_SERVICE_URL", "http://ocr.test:8004")
    monkeypatch.setenv("OCR_API_KEY", "geheim")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        seen["multipart"] = b'name="file"' in request.content and b"%PDF" in request.content
        return httpx.Response(
            200,
            json={
                "pages": [
                    {"page_index": 1, "lines": ["Zeile 1", "Zeile 2"]},
                    {"page_index": 2, "lines": ["Zeile 3"]},
                ],
                "full_text": "Zeile 1\nZeile 2\nZeile 3",
                "page_count": 2,
                "avg_confidence": 0.97,
            },
        )

    text = ji.ocr_pdf_bytes(b"%PDF-1.4 fake", transport=httpx.MockTransport(handler))
    assert seen == {"url": "http://ocr.test:8004/ocr", "key": "geheim", "multipart": True}
    assert "Zeile 1" in text and "Zeile 3" in text
    assert "--- Seite 2 ---" in text

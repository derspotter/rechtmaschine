"""ocrmypdf plugin that uses the local PaddleOCR HPI service as the OCR engine.

Instead of loading a second PaddleOCR instance (as ocrmypdf-paddleocr does),
this plugin posts each rasterized page to the already-running OCR service
(`ocr/ocr_service.py`, managed by service_manager.py) and converts the returned
lines/boxes into hOCR. The embedded text layer is therefore identical to the
text the Rechtmaschine app stores as sidecar.

Invocation (used by service_manager.py's /ocr-pdf endpoint):

    ocrmypdf --plugin /path/to/ocrmypdf_paddle_plugin.py \
        --pdf-renderer hocr --output-type pdf --jobs 1 \
        --sidecar out.txt input.pdf output.pdf

Environment:
    PADDLE_OCR_SERVICE_URL  base URL of the OCR service (default http://127.0.0.1:9003)
    PADDLE_OCR_TIMEOUT      per-page request timeout in seconds (default 300)
"""

from __future__ import annotations

import html
import logging
import os
from pathlib import Path

import requests
from PIL import Image

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

log = logging.getLogger(__name__)

SERVICE_URL = os.getenv("PADDLE_OCR_SERVICE_URL", "http://127.0.0.1:9003").rstrip("/")
TIMEOUT = float(os.getenv("PADDLE_OCR_TIMEOUT", "300"))

HOCR_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
 <head>
  <title></title>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
  <meta name="ocr-system" content="paddleocr-hpi-service"/>
  <meta name="ocr-capabilities" content="ocr_page ocr_line ocrx_word"/>
 </head>
 <body>
"""
HOCR_FOOT = " </body>\n</html>\n"


def _bbox(value) -> tuple[int, int, int, int] | None:
    """Normalize a 4-point polygon or flat [x0,y0,x1,y1] box to a bbox tuple."""
    try:
        if value is None or len(value) == 0:
            return None
        if isinstance(value[0], (list, tuple)):
            xs = [float(p[0]) for p in value]
            ys = [float(p[1]) for p in value]
        elif len(value) == 4:
            x0, y0, x1, y1 = (float(v) for v in value)
            xs, ys = [x0, x1], [y0, y1]
        else:
            return None
        box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return box
    except Exception:
        return None


def _clamp(box, width, height):
    x0, y0, x1, y1 = box
    return (
        max(0, min(x0, width - 1)),
        max(0, min(y0, height - 1)),
        max(1, min(x1, width)),
        max(1, min(y1, height)),
    )


def _line_words(line_idx: int, page: dict) -> list[tuple[str, tuple[int, int, int, int]]] | None:
    """Word-level (text, bbox) pairs for one line, if the service provided them.

    Word boxes can be missing or malformed on near-blank pages (known PaddleOCR
    quirk) — return None to fall back to line-level output.
    """
    words = page.get("text_word")
    boxes = page.get("word_boxes")
    if not isinstance(words, list) or not isinstance(boxes, list):
        return None
    if line_idx >= len(words) or line_idx >= len(boxes):
        return None
    line_words = words[line_idx]
    line_boxes = boxes[line_idx]
    if not isinstance(line_words, list) or not isinstance(line_boxes, list):
        return None
    if not line_words or len(line_words) != len(line_boxes):
        return None
    result = []
    for word, raw_box in zip(line_words, line_boxes):
        text = str(word or "").strip()
        box = _bbox(raw_box)
        if not text or not box:
            return None
        result.append((text, box))
    return result or None


def _page_to_hocr(page: dict, width: int, height: int, page_no: int) -> str:
    lines = page.get("lines") or []
    boxes = page.get("boxes") or []
    confidences = page.get("confidences") or []

    parts = [
        f"  <div class='ocr_page' id='page_{page_no}' "
        f"title='image; bbox 0 0 {width} {height}; ppageno {page_no - 1}'>\n"
    ]
    for idx, line_text in enumerate(lines):
        text = str(line_text or "").strip()
        if not text:
            continue
        line_box = _bbox(boxes[idx]) if idx < len(boxes) else None
        if not line_box:
            continue
        line_box = _clamp(line_box, width, height)
        try:
            conf = int(round(float(confidences[idx]) * 100)) if idx < len(confidences) else 90
        except Exception:
            conf = 90
        conf = max(0, min(conf, 100))

        x0, y0, x1, y1 = line_box
        parts.append(
            f"   <span class='ocr_line' id='line_{page_no}_{idx}' "
            f"title='bbox {x0} {y0} {x1} {y1}'>"
        )
        words = _line_words(idx, page)
        if words:
            for w_idx, (word, w_box) in enumerate(words):
                wx0, wy0, wx1, wy1 = _clamp(w_box, width, height)
                parts.append(
                    f"<span class='ocrx_word' id='word_{page_no}_{idx}_{w_idx}' "
                    f"title='bbox {wx0} {wy0} {wx1} {wy1}; x_wconf {conf}'>"
                    f"{html.escape(word)}</span> "
                )
        else:
            parts.append(
                f"<span class='ocrx_word' id='word_{page_no}_{idx}_0' "
                f"title='bbox {x0} {y0} {x1} {y1}; x_wconf {conf}'>"
                f"{html.escape(text)}</span>"
            )
        parts.append("</span>\n")
    parts.append("  </div>\n")
    return "".join(parts)


class PaddleServiceEngine(OcrEngine):
    """OcrEngine backed by the local PaddleOCR HPI HTTP service."""

    @staticmethod
    def version() -> str:
        return "1.0"

    @staticmethod
    def creator_tag(options) -> str:
        return "PaddleOCR-HPI-service"

    def __str__(self):
        return f"PaddleOCR service at {SERVICE_URL}"

    @staticmethod
    def languages(options):
        # The PaddleOCR latin/german models handle our documents; accept the
        # common codes so ocrmypdf's validation passes.
        return {"deu", "eng", "fra", "rus", "ara", "tur", "fas"}

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        # Orientation is handled inside the PaddleOCR pipeline itself.
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options) -> None:
        with Image.open(input_file) as img:
            width, height = img.size

        with open(input_file, "rb") as handle:
            response = requests.post(
                f"{SERVICE_URL}/ocr",
                files={"file": (input_file.name, handle, "image/png")},
                timeout=TIMEOUT,
            )
        response.raise_for_status()
        data = response.json()
        pages = data.get("pages") or []
        page = pages[0] if pages else {}

        body = _page_to_hocr(page, width, height, page_no=1)
        output_hocr.write_text(HOCR_HEAD + body + HOCR_FOOT, encoding="utf-8")

        lines = [str(line).strip() for line in (page.get("lines") or []) if str(line).strip()]
        output_text.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options) -> None:
        raise NotImplementedError("Use --pdf-renderer hocr with this plugin")


@hookimpl
def get_ocr_engine():
    return PaddleServiceEngine()

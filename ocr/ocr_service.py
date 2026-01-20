#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal OCR HTTP service using PaddleOCR (German-friendly).
- Accepts PDF or image uploads
- Renders PDFs to images (300 DPI) and OCRs each page
- Returns per-page lines, confidences, boxes, plus aggregated text
Test quickly with:
  curl -F "file=@/path/to/doc.pdf" http://localhost:8003/ocr
"""

import io
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from paddleocr import PaddleOCR  # PaddleOCR API
from PIL import Image

import paddle

app = FastAPI(title="Rechtmaschine OCR Service", version="1.0.0")

# --------- Configure OCR (init once) ----------
# For German legal text you can use:
#   lang="german"  -> recognizer with German alphabet (ß, umlauts)
# or a multilingual Latin model if preferred.
# PaddleOCR returns: results = [ [ [box], (text, score) ], ... ] one list per page.
# https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


USE_DOC_ORIENTATION = _env_flag("OCR_USE_DOC_ORIENTATION", True)
USE_DOC_UNWARPING = _env_flag("OCR_USE_UNWARPING", False)
USE_TEXTLINE_ORIENTATION = _env_flag("OCR_USE_TEXTLINE_ORIENTATION", True)
ENABLE_HPI = _env_flag("OCR_ENABLE_HPI", True)
RETURN_WORD_BOX = _env_flag("OCR_RETURN_WORD_BOX", False)
ENABLE_PDF_REPAIR = _env_flag("OCR_ENABLE_PDF_REPAIR", True)
PDF_REPAIR_TIMEOUT = int(os.getenv("OCR_PDF_REPAIR_TIMEOUT", "30"))

OCR_ENGINE = PaddleOCR(
    use_doc_orientation_classify=USE_DOC_ORIENTATION,
    use_doc_unwarping=USE_DOC_UNWARPING,
    use_textline_orientation=USE_TEXTLINE_ORIENTATION,
    enable_hpi=ENABLE_HPI,
    return_word_box=RETURN_WORD_BOX,
)

print("[INFO] OCR Engine initialized (German text support via latin model)")
print(
    "[INFO] OCR config: "
    f"doc_orientation={USE_DOC_ORIENTATION}, "
    f"textline_orientation={USE_TEXTLINE_ORIENTATION}, "
    f"unwarping={USE_DOC_UNWARPING}, "
    f"hpi={ENABLE_HPI}, "
    f"return_word_box={RETURN_WORD_BOX}"
)


def _ocr_file_path(file_path: str):
    """
    Run OCR on a file (image or PDF).
    Uses .predict() which returns a list of OCRResult objects (one per page).
    """
    print(f"[INFO] Running OCR on: {file_path}")
    try:
        result = OCR_ENGINE.predict(input=file_path)
    except Exception as exc:
        print(f"[WARNING] OCR failed for {file_path}: {exc}")
        if file_path.lower().endswith(".pdf"):
            repaired_path = _repair_pdf(file_path)
            if repaired_path:
                try:
                    print("[INFO] Retrying OCR with repaired PDF...")
                    result = OCR_ENGINE.predict(input=repaired_path)
                finally:
                    try:
                        os.unlink(repaired_path)
                    except FileNotFoundError:
                        pass
            else:
                raise
        else:
            raise

    page_results = []

    if not result:
        return page_results

    for page_idx, page_result in enumerate(result, 1):
        # Extract data from OCRResult.json
        page_data = page_result.json["res"]

        lines = page_data.get("rec_texts", [])
        scores = page_data.get("rec_scores", [])
        boxes = page_data.get("rec_polys", [])
        text_word = page_data.get("text_word")
        word_boxes = page_data.get("text_word_boxes")

        page_results.append(
            {
                "page_index": page_idx,
                "lines": lines,
                "confidences": scores,
                "boxes": boxes,
                "text_word": text_word,
                "word_boxes": word_boxes,
            }
        )
        print(f"[INFO] Page {page_idx}: extracted {len(lines)} lines")

    return page_results


def _repair_pdf(file_path: str) -> Optional[str]:
    if not ENABLE_PDF_REPAIR:
        return None

    qpdf_path = shutil.which("qpdf")
    if not qpdf_path:
        print("[WARNING] qpdf not found; skipping PDF repair pass.")
        return None

    repaired_handle, repaired_path = tempfile.mkstemp(suffix=".pdf")
    os.close(repaired_handle)

    try:
        result = subprocess.run(
            [qpdf_path, file_path, repaired_path],
            capture_output=True,
            text=True,
            timeout=PDF_REPAIR_TIMEOUT,
        )
    except Exception as exc:
        print(f"[WARNING] PDF repair failed: {exc}")
        try:
            os.unlink(repaired_path)
        except FileNotFoundError:
            pass
        return None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        print(f"[WARNING] PDF repair failed (exit {result.returncode}): {stderr}")
        try:
            os.unlink(repaired_path)
        except FileNotFoundError:
            pass
        return None

    print("[INFO] PDF repair completed successfully.")
    return repaired_path


@app.get("/health")
def health():
    return {
        "service": app.title,
        "version": app.version,
        "paddle_version": paddle.__version__,
        "gpu_available": paddle.is_compiled_with_cuda(),
    }


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    # Basic validation by extension; robust code should also sniff MIME.
    filename = (file.filename or "").lower()
    if not filename:
        raise HTTPException(400, "No filename provided.")
    _, ext = os.path.splitext(filename)
    if ext not in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"]:
        raise HTTPException(400, f"Unsupported extension: {ext}")

    # Stream upload to a temp file to avoid big RAM spikes.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmpf:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmpf.write(chunk)
        tmp_path = tmpf.name

    try:
        # PaddleOCR handles both images and PDFs directly
        page_results = _ocr_file_path(tmp_path)

        # Aggregate
        all_lines: List[str] = []
        all_scores: List[float] = []
        for p in page_results:
            all_lines.extend(p["lines"])
            all_scores.extend(p["confidences"])

        full_text = "\n".join(all_lines)
        avg_conf = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

        return JSONResponse(
            {
                "filename": filename,
                "page_count": len(page_results),
                "avg_confidence": round(avg_conf, 4),
                "full_text": full_text,
                "pages": page_results,
            }
        )

    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    # Bind to all interfaces; change as you wish.
    uvicorn.run("ocr_service:app", host="0.0.0.0", port=9003, reload=False)

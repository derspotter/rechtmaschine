#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR HTTP service using PaddleOCR with lazy load + hibernate endpoints.
- Accepts PDF or image uploads
- Loads OCR engine on-demand
- /load and /unload endpoints to move weights in/out of VRAM
- Optional model dir overrides for RAM-disk use
"""

import gc
import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from paddleocr import PaddleOCR
import paddle

app = FastAPI(title="Rechtmaschine OCR Service (Hibernate)", version="1.1.0")


# --------- Configure OCR (lazy init) ----------

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


USE_DOC_ORIENTATION = _env_flag("OCR_USE_DOC_ORIENTATION", True)
USE_DOC_UNWARPING = _env_flag("OCR_USE_UNWARPING", False)
USE_TEXTLINE_ORIENTATION = _env_flag("OCR_USE_TEXTLINE_ORIENTATION", True)
ENABLE_HPI = _env_flag("OCR_ENABLE_HPI", True)
RETURN_WORD_BOX = _env_flag("OCR_RETURN_WORD_BOX", False)
ENABLE_PDF_REPAIR = _env_flag("OCR_ENABLE_PDF_REPAIR", True)
PDF_REPAIR_TIMEOUT = int(os.getenv("OCR_PDF_REPAIR_TIMEOUT", "30"))

MODEL_BASE_DIR = _env_path("OCR_MODEL_BASE_DIR")
DET_MODEL_DIR = _env_path("OCR_DET_MODEL_DIR")
REC_MODEL_DIR = _env_path("OCR_REC_MODEL_DIR")
CLS_MODEL_DIR = _env_path("OCR_CLS_MODEL_DIR")


def _resolve_model_dir(
    explicit: Optional[str], base_dir: Optional[str], subdir: str
) -> Optional[str]:
    if explicit:
        return explicit
    if not base_dir:
        return None
    candidate = os.path.join(base_dir, subdir)
    if os.path.isdir(candidate):
        return candidate
    print(f"[WARNING] OCR_MODEL_BASE_DIR set but missing '{subdir}' in {base_dir}")
    return None


DET_MODEL_DIR = _resolve_model_dir(DET_MODEL_DIR, MODEL_BASE_DIR, "det")
REC_MODEL_DIR = _resolve_model_dir(REC_MODEL_DIR, MODEL_BASE_DIR, "rec")
CLS_MODEL_DIR = _resolve_model_dir(CLS_MODEL_DIR, MODEL_BASE_DIR, "cls")

OCR_ENGINE: Optional[PaddleOCR] = None
OCR_ENGINE_LOCK = threading.Lock()
OCR_IN_FLIGHT = 0
OCR_IN_FLIGHT_LOCK = threading.Lock()
OCR_LAST_LOAD_SECONDS: Optional[float] = None
OCR_LAST_UNLOAD_AT: Optional[float] = None

print("[INFO] OCR Engine configured (lazy load enabled)")
print(
    "[INFO] OCR config: "
    f"doc_orientation={USE_DOC_ORIENTATION}, "
    f"textline_orientation={USE_TEXTLINE_ORIENTATION}, "
    f"unwarping={USE_DOC_UNWARPING}, "
    f"hpi={ENABLE_HPI}, "
    f"return_word_box={RETURN_WORD_BOX}"
)
if MODEL_BASE_DIR or DET_MODEL_DIR or REC_MODEL_DIR or CLS_MODEL_DIR:
    print(
        "[INFO] OCR model dirs: "
        f"base={MODEL_BASE_DIR or 'default'}, "
        f"det={DET_MODEL_DIR or 'default'}, "
        f"rec={REC_MODEL_DIR or 'default'}, "
        f"cls={CLS_MODEL_DIR or 'default'}"
    )


def load_engine() -> PaddleOCR:
    global OCR_ENGINE, OCR_LAST_LOAD_SECONDS
    if OCR_ENGINE is not None:
        return OCR_ENGINE

    with OCR_ENGINE_LOCK:
        if OCR_ENGINE is not None:
            return OCR_ENGINE

        print("[INFO] Loading OCR Engine into VRAM...")
        start_time = time.time()

        model_kwargs: Dict[str, Any] = {}
        if DET_MODEL_DIR:
            model_kwargs["det_model_dir"] = DET_MODEL_DIR
        if REC_MODEL_DIR:
            model_kwargs["rec_model_dir"] = REC_MODEL_DIR
        if CLS_MODEL_DIR:
            model_kwargs["cls_model_dir"] = CLS_MODEL_DIR

        OCR_ENGINE = PaddleOCR(
            use_doc_orientation_classify=USE_DOC_ORIENTATION,
            use_doc_unwarping=USE_DOC_UNWARPING,
            use_textline_orientation=USE_TEXTLINE_ORIENTATION,
            enable_hpi=ENABLE_HPI,
            return_word_box=RETURN_WORD_BOX,
            **model_kwargs,
        )

        OCR_LAST_LOAD_SECONDS = time.time() - start_time
        print(f"[INFO] OCR Engine initialized in {OCR_LAST_LOAD_SECONDS:.2f}s")

    return OCR_ENGINE


def _ocr_file_path(file_path: str):
    """
    Run OCR on a file (image or PDF).
    Uses .predict() which returns a list of OCRResult objects (one per page).
    """
    print(f"[INFO] Running OCR on: {file_path}")
    engine = load_engine()
    try:
        result = engine.predict(input=file_path)
    except Exception as exc:
        print(f"[WARNING] OCR failed for {file_path}: {exc}")
        if file_path.lower().endswith(".pdf"):
            repaired_path = _repair_pdf(file_path)
            if repaired_path:
                try:
                    print("[INFO] Retrying OCR with repaired PDF...")
                    result = engine.predict(input=repaired_path)
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
        "engine_loaded": OCR_ENGINE is not None,
        "last_load_seconds": OCR_LAST_LOAD_SECONDS,
        "model_dirs": {
            "base": MODEL_BASE_DIR,
            "det": DET_MODEL_DIR,
            "rec": REC_MODEL_DIR,
            "cls": CLS_MODEL_DIR,
        },
    }


@app.post("/load")
def load():
    engine = load_engine()
    return {
        "status": "loaded",
        "engine_loaded": engine is not None,
        "last_load_seconds": OCR_LAST_LOAD_SECONDS,
    }


@app.post("/unload")
def unload():
    global OCR_ENGINE, OCR_LAST_UNLOAD_AT

    with OCR_IN_FLIGHT_LOCK:
        if OCR_IN_FLIGHT > 0:
            raise HTTPException(
                status_code=409,
                detail=f"OCR busy ({OCR_IN_FLIGHT} in-flight request(s))",
            )

    with OCR_ENGINE_LOCK:
        if OCR_ENGINE is None:
            return {"status": "already_unloaded", "engine_loaded": False}
        OCR_ENGINE = None

    gc.collect()
    if paddle.is_compiled_with_cuda():
        try:
            paddle.device.cuda.empty_cache()
        except Exception as exc:
            print(f"[WARNING] Failed to clear CUDA cache: {exc}")

    OCR_LAST_UNLOAD_AT = time.time()
    print("[INFO] OCR Engine unloaded from VRAM")
    return {"status": "unloaded", "engine_loaded": False}


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

    global OCR_IN_FLIGHT
    try:
        with OCR_IN_FLIGHT_LOCK:
            OCR_IN_FLIGHT += 1

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
        with OCR_IN_FLIGHT_LOCK:
            OCR_IN_FLIGHT -= 1
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    # Bind to all interfaces; change as you wish.
    uvicorn.run("ocr_service_hibernate:app", host="0.0.0.0", port=9003, reload=False)

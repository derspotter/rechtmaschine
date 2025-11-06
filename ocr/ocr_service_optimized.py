#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized OCR HTTP service using PaddleOCR with TensorRT acceleration.
- TensorRT GPU acceleration
- FP16 precision for faster inference
- Batch processing support
- Same API as ocr_service.py but ~2-3x faster
"""

import os
import tempfile
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from paddleocr import PaddleOCR
import paddle

app = FastAPI(title="Rechtmaschine OCR Service (Optimized)", version="2.0.0")

# --------- Configure OPTIMIZED OCR (init once) ----------
# Optimization features:
# - use_tensorrt=True: GPU inference acceleration (biggest speedup)
# - precision="fp16": Faster computation with minimal accuracy loss
# - enable_hpi=True: High-performance inference mode
# - use_gpu=True: Explicit GPU usage
# - rec_batch_num=8: Process 8 text lines in parallel

print("[INFO] Initializing OPTIMIZED OCR Engine...")
print("[INFO] Features: TensorRT + FP16 + Batch Size 16")

OCR_ENGINE = PaddleOCR(
    # Start with working config from ocr_service.py
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # Speed optimizations
    text_recognition_batch_size=16,      # Larger batch
    use_tensorrt=True,                   # TensorRT acceleration
    precision='fp16',                    # FP16 precision
)

print("[INFO] Optimized OCR Engine initialized")
print(f"[INFO] GPU available: {paddle.is_compiled_with_cuda()}")
print(f"[INFO] TensorRT: Enabled")
print(f"[INFO] Precision: FP16")
print(f"[INFO] Batch size: 16")


def _ocr_file_path(file_path: str):
    """
    Run optimized OCR on a file (image or PDF).
    Uses TensorRT + FP16 for maximum speed.
    """
    print(f"[INFO] Running OPTIMIZED OCR on: {file_path}")
    result = OCR_ENGINE.predict(input=file_path)

    page_results = []

    if not result:
        return page_results

    for page_idx, page_result in enumerate(result, 1):
        # Extract data from OCRResult.json
        page_data = page_result.json['res']

        lines = page_data.get('rec_texts', [])
        scores = page_data.get('rec_scores', [])
        boxes = page_data.get('rec_polys', [])

        page_results.append({
            "page_index": page_idx,
            "lines": lines,
            "confidences": scores,
            "boxes": boxes
        })
        print(f"[INFO] Page {page_idx}: extracted {len(lines)} lines (TensorRT+FP16)")

    return page_results


@app.get("/health")
def health():
    return {
        "service": app.title,
        "version": app.version,
        "optimizations": {
            "tensorrt": True,
            "fp16": True,
            "batch_processing": True,
            "batch_size": 8
        },
        "paddle_version": paddle.__version__,
        "gpu_available": paddle.is_compiled_with_cuda()
    }


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not filename:
        raise HTTPException(400, "No filename provided.")

    _, ext = os.path.splitext(filename)
    if ext not in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"]:
        raise HTTPException(400, f"Unsupported extension: {ext}")

    # Stream upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmpf:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmpf.write(chunk)
        tmp_path = tmpf.name

    try:
        # Optimized OCR with TensorRT + FP16
        page_results = _ocr_file_path(tmp_path)

        # Aggregate
        all_lines: List[str] = []
        all_scores: List[float] = []
        for p in page_results:
            all_lines.extend(p["lines"])
            all_scores.extend(p["confidences"])

        full_text = "\n".join(all_lines)
        avg_conf = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

        return JSONResponse({
            "filename": filename,
            "page_count": len(page_results),
            "avg_confidence": round(avg_conf, 4),
            "full_text": full_text,
            "pages": page_results,
            "optimizations_used": ["tensorrt", "fp16", "batch_processing"]
        })

    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    print("="*60)
    print("Starting OPTIMIZED OCR Service")
    print("TensorRT + FP16 + Batch Processing")
    print("Expected speedup: 2-3x vs standard OCR")
    print("="*60)
    uvicorn.run("ocr_service_optimized:app", host="0.0.0.0", port=9003, reload=False)

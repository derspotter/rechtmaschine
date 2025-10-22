#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PaddleOCR-VL HTTP service using vision-language model.
- More advanced than regular OCR (handles tables, formulas, charts)
- 0.9B parameter model with 109 language support
- Returns structured document parsing results
"""

import os
import tempfile
import shutil
import json
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from paddleocr import PaddleOCRVL
import paddle
import numpy as np

app = FastAPI(title="Rechtmaschine OCR-VL Service", version="1.0.0")

# --------- Configure OCR-VL (init once) ----------
print("[INFO] Initializing PaddleOCR-VL engine...")
OCR_VL_ENGINE = PaddleOCRVL()
print("[INFO] OCR-VL Engine initialized (Vision-Language Model)")


@app.get("/health")
def health():
    return {
        "service": app.title,
        "version": app.version,
        "model": "PaddleOCR-VL (0.9B parameters)",
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
        print(f"[INFO] Running OCR-VL on: {tmp_path}")

        # PaddleOCR-VL prediction - returns result objects
        results = OCR_VL_ENGINE.predict(tmp_path)

        full_text_parts = []
        all_structured_results = []

        for res in results:
            # Access the JSON data directly from result object
            json_data = res.json
            all_structured_results.append(json_data)

            # Extract text from parsing results
            if 'res' in json_data and 'parsing_res_list' in json_data['res']:
                parsing_results = json_data['res']['parsing_res_list']

                for block in parsing_results:
                    # Extract block content (the actual text)
                    block_content = block.get('block_content', '')
                    block_label = block.get('block_label', '')

                    if block_content:
                        # Add label prefix for context
                        full_text_parts.append(f"[{block_label}] {block_content}")

        full_text = "\n\n".join(full_text_parts)

        return JSONResponse({
            "filename": filename,
            "model": "PaddleOCR-VL",
            "full_text": full_text,
            "structured_output": all_structured_results,
            "page_count": len(all_structured_results),
            "block_count": len(full_text_parts)
        })

    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    # Pass app object directly to avoid double-loading the model
    uvicorn.run(app, host="0.0.0.0", port=9005, reload=False)

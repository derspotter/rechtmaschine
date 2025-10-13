"""
Rechtmaschine OCR Service

This service runs on the home PC with GPU and provides OCR (Optical Character Recognition)
for scanned PDF documents using PaddleOCR.

Architecture:
- Runs on home PC with 12GB VRAM (RTX 3060)
- Accessed via Tailscale mesh network from production server
- Uses PaddleOCR with PaddlePaddle GPU acceleration
- Provides REST API on port 8003

Usage:
    python ocr_service.py

Environment Variables:
    OCR_API_KEY: API key for authentication (optional)
    OCR_PORT: Port to listen on (default: 8003)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from pydantic import BaseModel
import os
import tempfile
from typing import Optional, List
import paddle
from paddleocr import PaddleOCR

app = FastAPI(
    title="Rechtmaschine OCR Service",
    description="OCR service for scanned PDF documents using PaddleOCR",
    version="1.0.0"
)

# Configuration
OCR_API_KEY = os.getenv("OCR_API_KEY")
OCR_PORT = int(os.getenv("OCR_PORT", "8003"))

# Initialize PaddleOCR
# use_angle_cls=True enables text orientation detection
# lang='en' for English, 'german' for German (if available)
# use_gpu=True to use GPU acceleration
ocr_engine = None

def get_ocr_engine():
    """Lazy initialization of OCR engine"""
    global ocr_engine
    if ocr_engine is None:
        print("[INFO] Initializing PaddleOCR engine...")
        ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='en',  # PaddleOCR supports: en, ch, german, french, etc.
            use_gpu=True,
            show_log=False
        )
        print("[INFO] PaddleOCR engine initialized")
    return ocr_engine


class OCRResponse(BaseModel):
    text: str
    confidence: float
    page_count: int
    language: str = "en"


@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None)
):
    """
    Perform OCR on an uploaded PDF or image file.

    Supported formats: PDF, PNG, JPG, JPEG, BMP, TIFF

    Returns extracted text with confidence score.
    """

    # API key authentication (if configured)
    if OCR_API_KEY:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key required")
        if x_api_key != OCR_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Validate file type
    allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}"
        )

    print(f"[INFO] Processing OCR request for file: {file.filename}")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        print(f"[INFO] File saved to: {tmp_file_path}")
        print(f"[INFO] File size: {len(content)} bytes")

        # Get OCR engine
        ocr = get_ocr_engine()

        # Perform OCR
        print("[INFO] Running OCR...")
        result = ocr.ocr(tmp_file_path, cls=True)

        # Extract text and confidence scores
        extracted_text = []
        confidence_scores = []
        page_count = len(result) if result else 0

        for page_idx, page_result in enumerate(result):
            if page_result is None:
                continue

            print(f"[INFO] Processing page {page_idx + 1}/{page_count}")

            for line in page_result:
                if line is None:
                    continue

                # Each line is: [bbox, (text, confidence)]
                text_info = line[1]
                text = text_info[0]
                confidence = text_info[1]

                extracted_text.append(text)
                confidence_scores.append(confidence)

        # Combine all text
        full_text = "\n".join(extracted_text)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        print(f"[SUCCESS] OCR completed")
        print(f"[INFO] Extracted {len(extracted_text)} text lines")
        print(f"[INFO] Average confidence: {avg_confidence:.2f}")
        print(f"[INFO] Total characters: {len(full_text)}")

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return OCRResponse(
            text=full_text,
            confidence=avg_confidence,
            page_count=page_count,
            language="en"
        )

    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        # Clean up on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns service status and GPU information.
    """
    gpu_available = paddle.device.is_compiled_with_cuda()

    # Get GPU info if available
    gpu_info = "Not available"
    if gpu_available:
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
        except Exception as e:
            gpu_info = f"Error: {e}"

    return {
        "status": "healthy",
        "paddle_version": paddle.__version__,
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "ocr_engine": "PaddleOCR"
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Rechtmaschine OCR Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ocr": "/ocr (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine OCR Service")
    print("=" * 60)
    print(f"PaddlePaddle Version: {paddle.__version__}")
    print(f"GPU Available: {paddle.device.is_compiled_with_cuda()}")
    print(f"API Key Auth: {'Enabled' if OCR_API_KEY else 'Disabled'}")
    print(f"Listening on: 0.0.0.0:{OCR_PORT}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=OCR_PORT)

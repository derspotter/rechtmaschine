#!/usr/bin/env python3
"""
Smart Service Manager for OCR and Anonymization
Automatically loads/unloads services to optimize VRAM usage
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from pydantic import BaseModel
import httpx
import subprocess
import time
import psutil
import os

app = FastAPI(title="Rechtmaschine Service Manager")

# Get base directory (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Service configurations
SERVICES = {
    "ocr": {
        "port": 9003,
        "url": "http://localhost:9003",
        "process_name": "ocr_service.py",
        "start_cmd": ["python3", "ocr_service.py"],
        "cwd": os.path.join(BASE_DIR, "ocr"),
        "venv": os.path.join(BASE_DIR, "ocr", ".venv", "bin", "python3")
    },
    "anon": {
        "port": 9002,
        "url": "http://localhost:9002",
        "process_name": "anonymization_service.py",
        "start_cmd": ["python3", "anonymization_service.py"],
        "cwd": os.path.join(BASE_DIR, "anon"),
        "venv": os.path.join(BASE_DIR, "anon", ".venv", "bin", "python3")
    }
}

class AnonymizationRequest(BaseModel):
    text: str
    document_type: str

def is_service_running(service_name: str) -> bool:
    """Check if a service process is running"""
    process_name = SERVICES[service_name]["process_name"]
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and any(process_name in cmd for cmd in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False

def kill_service(service_name: str):
    """Kill a service to free VRAM"""
    process_name = SERVICES[service_name]["process_name"]
    print(f"[Manager] Killing {service_name} service to free VRAM...")
    subprocess.run(["pkill", "-f", process_name])

    # For anon service, also kill Ollama runner to actually free VRAM
    if service_name == "anon":
        print(f"[Manager] Killing Ollama runner to free model from VRAM...")
        subprocess.run(["pkill", "-f", "ollama runner"])

    time.sleep(2)

def start_service(service_name: str, timeout: int = 60):
    """Start a service and wait until it's ready"""
    config = SERVICES[service_name]
    print(f"[Manager] Starting {service_name} service...")

    # Use venv python if available
    cmd = [config["venv"]] + config["start_cmd"][1:]

    # Temporarily log output to file for debugging
    log_file = open(f"/tmp/{service_name}_service.log", "w")
    subprocess.Popen(
        cmd,
        cwd=config["cwd"],
        stdout=log_file,
        stderr=log_file
    )

    # Poll health endpoint until ready
    print(f"[Manager] Waiting for {service_name} to become ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{config['url']}/health", timeout=2.0)
            if response.status_code == 200:
                elapsed = int(time.time() - start_time)
                print(f"[Manager] {service_name} ready after {elapsed}s")
                return
        except:
            pass

        time.sleep(1)

    raise Exception(f"{service_name} failed to start within {timeout}s")

def ensure_service_ready(service_name: str, unload_other: bool = True):
    """Ensure target service is ready, optionally unload the other"""
    other_service = "anon" if service_name == "ocr" else "ocr"

    # Kill other service if it's running and we need VRAM
    if unload_other and is_service_running(other_service):
        kill_service(other_service)

    # Start target service if not running (polls until ready)
    if not is_service_running(service_name):
        try:
            start_service(service_name)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to start {service_name}: {str(e)}"
            )
    else:
        # Service already running, verify it's responsive
        config = SERVICES[service_name]
        try:
            response = httpx.get(f"{config['url']}/health", timeout=5.0)
            response.raise_for_status()
            print(f"[Manager] {service_name} service already ready")
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"{service_name} not responding: {str(e)}"
            )

@app.post("/ocr")
async def ocr_document(file: UploadFile = File(...)):
    """OCR endpoint - automatically manages service state"""
    print(f"[Manager] OCR request received for: {file.filename}")

    # Ensure OCR is ready (and unload anon if needed)
    ensure_service_ready("ocr", unload_other=True)

    # Forward request to OCR service
    async with httpx.AsyncClient(timeout=300.0) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        response = await client.post(
            f"{SERVICES['ocr']['url']}/ocr",
            files=files
        )
        if response.status_code != 200:
            error_detail = response.text
            print(f"[Manager] OCR error: {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=error_detail)
        return response.json()

@app.post("/anonymize")
async def anonymize_document(
    request: AnonymizationRequest,
    x_api_key: str = Header(None)
):
    """Anonymization endpoint - automatically manages service state"""
    print(f"[Manager] Anonymization request received ({len(request.text)} chars)")

    # Ensure anon is ready (and unload OCR if needed)
    ensure_service_ready("anon", unload_other=True)

    # Forward request to anonymization service
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{SERVICES['anon']['url']}/anonymize",
            json=request.model_dump(),
            headers={"X-API-Key": x_api_key} if x_api_key else {}
        )
        if response.status_code != 200:
            error_detail = response.text
            print(f"[Manager] Anonymization error: {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=error_detail)
        return response.json()

@app.get("/status")
async def get_status():
    """Get current service status"""
    return {
        "ocr_running": is_service_running("ocr"),
        "anon_running": is_service_running("anon"),
        "services": {
            "ocr": SERVICES["ocr"]["url"],
            "anon": SERVICES["anon"]["url"]
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "manager": "active",
        "ocr_loaded": is_service_running("ocr"),
        "anon_loaded": is_service_running("anon")
    }

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine Service Manager")
    print("=" * 60)
    print("Listening on: http://0.0.0.0:8004")
    print("OCR endpoint: http://0.0.0.0:8004/ocr (forwards to :9003)")
    print("Anon endpoint: http://0.0.0.0:8004/anonymize (forwards to :9002)")
    print("Status: http://0.0.0.0:8004/status")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8004)

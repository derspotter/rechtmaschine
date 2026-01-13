#!/usr/bin/env python3
"""
Smart Service Manager for OCR and Anonymization
Automatically loads/unloads services to optimize VRAM usage
Uses a service-aware queue to batch requests and minimize service switches
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from pydantic import BaseModel
import httpx
import subprocess
import time
import psutil
import os
import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Awaitable
from collections import deque

app = FastAPI(title="Rechtmaschine Service Manager")

def log(message: str):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

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
        "venv": os.path.join(BASE_DIR, "ocr", ".venv", "bin", "python3"),
        "load_time": 11  # seconds - OCR takes longer due to CUDA init
    },
    "anon": {
        "port": 9002,
        "url": "http://localhost:9002",
        "process_name": "anonymization_service.py",
        "start_cmd": ["python3", "anonymization_service.py"],
        "cwd": os.path.join(BASE_DIR, "anon"),
        "venv": os.path.join(BASE_DIR, "anon", ".venv", "bin", "python3"),
        "load_time": 1  # seconds - anon loads fast
    }
}

class AnonymizationRequest(BaseModel):
    text: str
    document_type: str


# =============================================================================
# Service-Aware Queue Implementation
# =============================================================================

@dataclass
class QueuedRequest:
    """A request waiting to be processed"""
    service: str
    handler: Callable[[], Awaitable[Any]]
    future: asyncio.Future
    queued_at: float


class ServiceQueue:
    """
    Queue that batches requests by service type to minimize expensive service switches.

    Strategy:
    1. If current service has pending requests, process those first
    2. Only switch services when current service queue is empty
    3. This minimizes OCR loads (11s) vs anon loads (1s)
    """

    def __init__(self):
        self.queues: dict[str, deque[QueuedRequest]] = {
            "ocr": deque(),
            "anon": deque()
        }
        self.current_service: str | None = None
        self.processing = False
        self.lock = asyncio.Lock()
        self.stats = {"ocr": 0, "anon": 0, "switches": 0}

    async def enqueue(self, service: str, handler: Callable[[], Awaitable[Any]]) -> Any:
        """Add request to queue and wait for result"""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request = QueuedRequest(
            service=service,
            handler=handler,
            future=future,
            queued_at=time.time()
        )

        async with self.lock:
            self.queues[service].append(request)
            queue_pos = len(self.queues[service])
            other_service = "anon" if service == "ocr" else "ocr"
            other_pending = len(self.queues[other_service])

            log(f"[Queue] {service.upper()} request queued (position {queue_pos}, "
                f"{other_pending} {other_service} pending)")

            if not self.processing:
                self.processing = True
                asyncio.create_task(self._process_queue())

        # Wait for our request to be processed
        return await future

    async def _process_queue(self):
        """Process queued requests, batching by service type"""
        log("[Queue] Processor started")

        while True:
            async with self.lock:
                # Decide which service to process next
                # PRIORITY: Currently loaded service (to avoid switches)
                service = self._select_next_service()

                if service is None:
                    self.processing = False
                    log(f"[Queue] Processor stopped (stats: {self.stats})")
                    return

                request = self.queues[service].popleft()
                remaining = len(self.queues[service])
                wait_time = time.time() - request.queued_at

            log(f"[Queue] Processing {service.upper()} (waited {wait_time:.1f}s, "
                f"{remaining} more {service} queued)")

            # Switch service if needed (blocking - run in thread)
            if service != self.current_service:
                try:
                    self.stats["switches"] += 1
                    old = self.current_service or "none"
                    log(f"[Queue] Service switch #{self.stats['switches']}: {old} -> {service}")
                    await asyncio.to_thread(self._ensure_service_ready, service)
                    self.current_service = service
                except Exception as e:
                    log(f"[Queue] Service switch failed: {e}")
                    request.future.set_exception(
                        HTTPException(status_code=503, detail=f"Failed to start {service}: {e}")
                    )
                    continue

            # Process the request
            try:
                self.stats[service] += 1
                result = await request.handler()
                request.future.set_result(result)
            except Exception as e:
                log(f"[Queue] Request failed: {e}")
                request.future.set_exception(e)

    def _select_next_service(self) -> str | None:
        """Select which service queue to process next"""
        ocr_count = len(self.queues["ocr"])
        anon_count = len(self.queues["anon"])

        if ocr_count == 0 and anon_count == 0:
            return None

        # Prefer currently loaded service to avoid expensive switches
        if self.current_service and self.queues[self.current_service]:
            return self.current_service

        # If current service queue is empty, switch to the other
        if ocr_count > 0:
            return "ocr"
        return "anon"

    def _ensure_service_ready(self, service_name: str):
        """Ensure target service is ready (runs in thread pool)"""
        other_service = "anon" if service_name == "ocr" else "ocr"

        # Kill other service to free VRAM
        if is_service_running(other_service):
            kill_service(other_service)

        # When starting OCR, always try to unload Ollama model (might be loaded from previous session)
        if service_name == "ocr":
            unload_ollama_model()

        # Start target service if not running
        if not is_service_running(service_name):
            start_service(service_name)
        else:
            # Verify it's responsive
            config = SERVICES[service_name]
            try:
                response = httpx.get(f"{config['url']}/health", timeout=5.0)
                response.raise_for_status()
                log(f"[Manager] {service_name} already running and healthy")
            except Exception as e:
                raise Exception(f"{service_name} not responding: {e}")

    def get_status(self) -> dict:
        """Get current queue status"""
        return {
            "current_service": self.current_service,
            "processing": self.processing,
            "queued": {
                "ocr": len(self.queues["ocr"]),
                "anon": len(self.queues["anon"])
            },
            "stats": self.stats
        }


# Global queue instance
service_queue = ServiceQueue()


# =============================================================================
# Service Management Functions
# =============================================================================

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


def unload_ollama_model():
    """Unload Ollama model from VRAM"""
    log(f"[Manager] Unloading Ollama model from VRAM...")
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11435")
        model = os.getenv("OLLAMA_MODEL", "qwen3:14b")

        with httpx.Client(timeout=5.0) as client:
            client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "keep_alive": 0
                }
            )
        log(f"[Manager] Ollama model unloaded successfully")
        time.sleep(3)  # Wait for VRAM to be freed
    except Exception as e:
        log(f"[Manager] Warning: Failed to unload Ollama model: {e}")


def kill_service(service_name: str):
    """Kill a service to free VRAM"""
    process_name = SERVICES[service_name]["process_name"]
    log(f"[Manager] Killing {service_name} service to free VRAM...")
    subprocess.run(["pkill", "-f", process_name])

    # For anon service, unload Ollama model to actually free VRAM
    if service_name == "anon":
        unload_ollama_model()

    # Wait for GPU memory to be freed
    time.sleep(2)


def start_service(service_name: str, timeout: int = 60):
    """Start a service and wait until it's ready"""
    config = SERVICES[service_name]
    log(f"[Manager] Starting {service_name} service...")

    # Use venv python if available
    cmd = [config["venv"]] + config["start_cmd"][1:]

    # Log output to file for debugging
    log_file = open(f"/tmp/{service_name}_service.log", "w")
    subprocess.Popen(
        cmd,
        cwd=config["cwd"],
        stdout=log_file,
        stderr=log_file
    )

    # Poll health endpoint until ready
    log(f"[Manager] Waiting for {service_name} to become ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{config['url']}/health", timeout=2.0)
            if response.status_code == 200:
                elapsed = int(time.time() - start_time)
                log(f"[Manager] {service_name} ready after {elapsed}s")
                return
        except:
            pass

        time.sleep(1)

    raise Exception(f"{service_name} failed to start within {timeout}s")


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/ocr")
async def ocr_document(file: UploadFile = File(...)):
    """OCR endpoint - queued for efficient service management"""
    request_start = time.time()
    filename = file.filename
    file_content = await file.read()
    content_type = file.content_type

    log(f"[API] OCR request received for: {filename}")

    async def do_ocr():
        """Actual OCR work - called when service is ready"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            files = {"file": (filename, file_content, content_type)}
            response = await client.post(
                f"{SERVICES['ocr']['url']}/ocr",
                files=files
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            total_elapsed = time.time() - request_start
            log(f"[API] OCR completed for {filename} (total: {total_elapsed:.2f}s)")
            return response.json()

    return await service_queue.enqueue("ocr", do_ocr)


@app.post("/anonymize")
async def anonymize_document(
    request: AnonymizationRequest,
    x_api_key: str = Header(None)
):
    """Anonymization endpoint - queued for efficient service management"""
    request_start = time.time()
    text_len = len(request.text)
    request_data = request.model_dump()

    log(f"[API] Anonymization request received ({text_len} chars)")

    async def do_anonymize():
        """Actual anonymization work - called when service is ready"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{SERVICES['anon']['url']}/anonymize",
                json=request_data,
                headers={"X-API-Key": x_api_key} if x_api_key else {}
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            total_elapsed = time.time() - request_start
            log(f"[API] Anonymization completed ({text_len} chars, total: {total_elapsed:.2f}s)")
            return response.json()

    return await service_queue.enqueue("anon", do_anonymize)


@app.get("/status")
async def get_status():
    """Get current service and queue status"""
    return {
        "ocr_running": is_service_running("ocr"),
        "anon_running": is_service_running("anon"),
        "services": {
            "ocr": SERVICES["ocr"]["url"],
            "anon": SERVICES["anon"]["url"]
        },
        "queue": service_queue.get_status()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    queue_status = service_queue.get_status()
    return {
        "status": "healthy",
        "manager": "active",
        "ocr_loaded": is_service_running("ocr"),
        "anon_loaded": is_service_running("anon"),
        "queue": {
            "ocr_pending": queue_status["queued"]["ocr"],
            "anon_pending": queue_status["queued"]["anon"]
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine Service Manager (with Smart Queue)")
    print("=" * 60)
    print("Listening on: http://0.0.0.0:8004")
    print("OCR endpoint: http://0.0.0.0:8004/ocr (forwards to :9003)")
    print("Anon endpoint: http://0.0.0.0:8004/anonymize (forwards to :9002)")
    print("Status: http://0.0.0.0:8004/status")
    print("=" * 60)
    print("Queue Strategy:")
    print("  - Batches requests by service type")
    print("  - Minimizes expensive service switches")
    print("  - OCR loads in ~11s, Anon loads in ~1s")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8004)

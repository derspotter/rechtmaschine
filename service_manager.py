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
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional
from collections import deque
import uuid

app = FastAPI(title="Rechtmaschine Service Manager")


def log(message: str):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


# Get base directory (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ANON_BACKEND = os.getenv("ANON_BACKEND", "flair").strip().lower()
ANON_BACKENDS = {
    "flair": {
        "kind": "process",
        "port": 9002,
        "url": "http://localhost:9002",
        "process_name": "anonymization_service_flair.py",
        "start_cmd": ["python3", "anonymization_service_flair.py"],
        "cwd": os.path.join(BASE_DIR, "anon"),
        "venv": os.path.join(BASE_DIR, "anon", ".venv", "bin", "python3"),
        "load_time": 1,
    },
    "qwen": {
        "kind": "process",
        "port": 9002,
        "url": "http://localhost:9002",
        "process_name": "anonymization_service.py",
        "start_cmd": ["python3", "anonymization_service.py"],
        "cwd": os.path.join(BASE_DIR, "anon"),
        "venv": os.path.join(BASE_DIR, "anon", ".venv", "bin", "python3"),
        "load_time": 1,
    },
}
if ANON_BACKEND not in ANON_BACKENDS:
    log(f"[Manager] Unknown ANON_BACKEND '{ANON_BACKEND}', defaulting to 'flair'")
    ANON_BACKEND = "flair"

# Service configurations
SERVICES = {
    "ocr": {
        "kind": "docker",
        "container": "paddlex-ocr-hpi",
        "compose_file": os.path.join(BASE_DIR, "docker-compose.ocr-hpi.yml"),
        "docker_env": {
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "ORT_LOG_SEVERITY_LEVEL": "3",
        },
        "env": {
            "OCR_RETURN_WORD_BOX": "1",
            "OCR_USE_DOC_ORIENTATION": "1",
            "OCR_USE_TEXTLINE_ORIENTATION": "1",
            "OCR_USE_UNWARPING": "0",
            "OCR_ENABLE_HPI": "1",
        },
        "load_time": 11,  # seconds - OCR takes longer due to CUDA init
        "fallback": "ocr_legacy",
    },
    "ocr_legacy": {
        "kind": "process",
        "port": 9003,
        "url": "http://localhost:9003",
        "process_name": "ocr_service.py",
        "start_cmd": ["python3", "ocr_service.py"],
        "cwd": os.path.join(BASE_DIR, "ocr"),
        "venv": os.path.join(BASE_DIR, "ocr", ".venv", "bin", "python3"),
        "env": {
            "OCR_RETURN_WORD_BOX": "1",
            "OCR_USE_DOC_ORIENTATION": "1",
            "OCR_USE_TEXTLINE_ORIENTATION": "1",
            "OCR_USE_UNWARPING": "0",
            "OCR_ENABLE_HPI": "0",
        },
        "load_time": 11,
    },
    "anon": {**ANON_BACKENDS[ANON_BACKEND], "load_time": 1},
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
        self.queues: dict[str, deque[QueuedRequest]] = {"ocr": deque(), "anon": deque()}
        self.current_service: str | None = None
        self.processing = False
        self.lock = asyncio.Lock()
        self.stats = {"ocr": 0, "anon": 0, "switches": 0}

        if ANON_BACKEND != "flair":
            log(f"[Manager] Anonymization backend: {ANON_BACKEND}")

    async def enqueue(self, service: str, handler: Callable[[], Awaitable[Any]]) -> Any:
        """Add request to queue and wait for result"""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request = QueuedRequest(
            service=service, handler=handler, future=future, queued_at=time.time()
        )

        async with self.lock:
            self.queues[service].append(request)
            queue_pos = len(self.queues[service])
            other_service = "anon" if service == "ocr" else "ocr"
            other_pending = len(self.queues[other_service])

            log(
                f"[Queue] {service.upper()} request queued (position {queue_pos}, "
                f"{other_pending} {other_service} pending)"
            )

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

            log(
                f"[Queue] Processing {service.upper()} (waited {wait_time:.1f}s, "
                f"{remaining} more {service} queued)"
            )

            # Switch service if needed (blocking - run in thread)
            if service != self.current_service:
                try:
                    self.stats["switches"] += 1
                    old = self.current_service or "none"
                    log(
                        f"[Queue] Service switch #{self.stats['switches']}: {old} -> {service}"
                    )
                    await asyncio.to_thread(self._ensure_service_ready, service)
                    self.current_service = service
                except Exception as e:
                    log(f"[Queue] Service switch failed: {e}")
                    request.future.set_exception(
                        HTTPException(
                            status_code=503, detail=f"Failed to start {service}: {e}"
                        )
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
                service_url = config.get("url")
                if not service_url:
                    log(f"[Manager] {service_name} already running (no health check)")
                    return
                response = httpx.get(f"{service_url}/health", timeout=5.0)
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
                "anon": len(self.queues["anon"]),
            },
            "stats": self.stats,
        }


# Global queue instance
service_queue = ServiceQueue()


# =============================================================================
# Service Management Functions
# =============================================================================


def is_container_running(container: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    return result.stdout.strip().lower() == "true"


def is_service_running(service_name: str) -> bool:
    """Check if a service process or container is running"""
    config = SERVICES[service_name]
    kind = config.get("kind", "process")

    if kind == "docker":
        container = config.get("container")
        if container and is_container_running(container):
            return True
        fallback = config.get("fallback")
        if fallback:
            return is_service_running(fallback)
        return False

    process_name = config["process_name"]
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline and any(process_name in cmd for cmd in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def get_active_ocr_backend() -> str:
    """Return which OCR backend is active"""
    config = SERVICES["ocr"]
    container = config.get("container")
    if container and is_container_running(container):
        return "docker"
    fallback = config.get("fallback")
    if fallback and is_service_running(fallback):
        return "legacy"
    return "unavailable"


def get_legacy_ocr_config() -> dict | None:
    fallback = SERVICES["ocr"].get("fallback")
    if not fallback:
        return None
    return SERVICES.get(fallback)


def ensure_legacy_ocr_running() -> dict:
    legacy_config = get_legacy_ocr_config()
    if not legacy_config:
        raise HTTPException(status_code=500, detail="Legacy OCR backend not configured")
    if not is_service_running("ocr_legacy"):
        start_service("ocr_legacy")
    return legacy_config


async def run_legacy_ocr(filename: str, file_content: bytes) -> dict:
    legacy_config = ensure_legacy_ocr_running()
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{legacy_config['url']}/ocr",
            files={"file": (filename, file_content, "application/octet-stream")},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


def _repair_pdf(pdf_path: Path) -> Optional[Path]:
    if not pdf_path.exists():
        return None
    try:
        import pikepdf
    except Exception as exc:
        log(f"[WARNING] PDF repair skipped (pikepdf unavailable): {exc}")
        return None

    repaired_path = pdf_path.with_name(f"{pdf_path.stem}_repaired.pdf")
    try:
        with pikepdf.open(pdf_path) as pdf:
            pdf.save(repaired_path)
        log(f"[INFO] Repaired PDF before OCR: {pdf_path.name}")
        return repaired_path
    except Exception as exc:
        log(f"[WARNING] PDF repair failed for {pdf_path.name}: {exc}")
        try:
            if repaired_path.exists():
                repaired_path.unlink()
        except Exception:
            pass
        return None


def unload_ollama_model():
    """Unload Ollama model from VRAM"""
    log(f"[Manager] Unloading Ollama model from VRAM...")
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11435")
        model = os.getenv("OLLAMA_MODEL", "qwen3:14b")

        with httpx.Client(timeout=5.0) as client:
            client.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
            )
        log(f"[Manager] Ollama model unloaded successfully")
        time.sleep(3)  # Wait for VRAM to be freed
    except Exception as e:
        log(f"[Manager] Warning: Failed to unload Ollama model: {e}")


def kill_service(service_name: str):
    """Kill a service to free VRAM"""
    config = SERVICES[service_name]
    kind = config.get("kind", "process")

    if kind == "docker":
        log(f"[Manager] Skipping stop for {service_name} container")
        return

    process_name = config["process_name"]
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
    kind = config.get("kind", "process")
    log(f"[Manager] Starting {service_name} service...")

    if kind == "docker":
        container = config.get("container")
        compose_file = config.get("compose_file")
        if container and not is_container_running(container):
            started = False
            start_result = subprocess.run(["docker", "start", container])
            if start_result.returncode == 0:
                started = True
            elif compose_file:
                subprocess.run(
                    ["docker", "compose", "-f", compose_file, "up", "-d"], check=True
                )
                started = True
            if not started and config.get("fallback"):
                log(f"[Manager] Falling back to {config['fallback']}")
                start_service(config["fallback"], timeout=timeout)
                return
            if not started:
                raise Exception(f"Failed to start container {container}")
        time.sleep(2)
        log(f"[Manager] {service_name} container ready")
        return

    # Use venv python if available
    cmd = [config["venv"]] + config["start_cmd"][1:]

    env = os.environ.copy()
    service_env = config.get("env")
    if service_env:
        for key, value in service_env.items():
            env.setdefault(key, value)

    # Log output to file for debugging
    log_file = open(f"/tmp/{service_name}_service.log", "w")
    subprocess.Popen(cmd, cwd=config["cwd"], env=env, stdout=log_file, stderr=log_file)

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
        except Exception:
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
    filename = file.filename or "uploaded_file"
    file_content = await file.read()
    log(f"[API] OCR request received for: {filename}")

    async def do_ocr():
        """Actual OCR work - called when service is ready"""
        config = SERVICES["ocr"]
        container = config.get("container")
        fallback = config.get("fallback")
        if not container and not fallback:
            raise HTTPException(status_code=500, detail="OCR backend not configured")

        _, ext = os.path.splitext(filename or "")
        if ext.lower() not in [
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".pdf",
        ]:
            raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")

        tmp_dir = Path("/tmp/ocr_service_manager")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{uuid.uuid4().hex}{ext}"
        tmp_path.write_bytes(file_content)
        repaired_path = None
        if ext.lower() == ".pdf":
            repaired_path = _repair_pdf(tmp_path)
        ocr_input_path = repaired_path or tmp_path

        output_dir = tmp_dir / f"ocr_{uuid.uuid4().hex}"
        output_dir.mkdir(parents=True, exist_ok=True)

        env_flags = config.get("env", {})
        docker_env = config.get("docker_env", {})

        def _flag(name: str, default: str) -> str:
            value = env_flags.get(name)
            if value is None:
                return default
            return (
                "True"
                if value.strip().lower() in {"1", "true", "yes", "on"}
                else "False"
            )

        def run_ocr(return_word_box: bool) -> None:
            cmd = ["docker", "exec"]
            for key, value in docker_env.items():
                if value is None:
                    continue
                cmd.extend(["-e", f"{key}={value}"])
            if not container:
                raise HTTPException(
                    status_code=500, detail="OCR container not configured"
                )
            cmd.extend(
                [
                    container,
                    "paddleocr",
                    "ocr",
                    "-i",
                    str(ocr_input_path),
                    "--enable_hpi",
                    _flag("OCR_ENABLE_HPI", "True"),
                    "--use_doc_orientation_classify",
                    _flag("OCR_USE_DOC_ORIENTATION", "True"),
                    "--use_textline_orientation",
                    _flag("OCR_USE_TEXTLINE_ORIENTATION", "True"),
                    "--use_doc_unwarping",
                    _flag("OCR_USE_UNWARPING", "False"),
                    "--return_word_box",
                    "True" if return_word_box else "False",
                    "--device",
                    "gpu:0",
                    "--save_path",
                    str(output_dir),
                ]
            )
            subprocess.run(cmd, check=True)

        try:
            return_word_box = _flag("OCR_RETURN_WORD_BOX", "False") == "True"
            try:
                run_ocr(return_word_box)
            except subprocess.CalledProcessError:
                if return_word_box:
                    log("[API] OCR failed with word boxes, retrying without word boxes")
                    run_ocr(False)
                else:
                    raise

            page_results = []
            json_files = sorted(output_dir.glob("*_res.json"))
            for file_path in json_files:
                raw_page = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(raw_page, dict) and "res" in raw_page:
                    page_data = raw_page.get("res") or {}
                elif isinstance(raw_page, dict):
                    page_data = raw_page
                else:
                    page_data = {}
                page_index = page_data.get("page_index")
                if isinstance(page_index, int):
                    page_index = page_index + 1
                else:
                    page_index = len(page_results) + 1

                lines = page_data.get("rec_texts", [])
                text_word = page_data.get("text_word")
                if text_word:
                    lines = ["".join(tokens).strip() for tokens in text_word]

                page_results.append(
                    {
                        "page_index": page_index,
                        "lines": lines,
                        "confidences": page_data.get("rec_scores", []),
                        "boxes": page_data.get("rec_polys", []),
                        "text_word": page_data.get("text_word"),
                        "word_boxes": page_data.get("text_word_boxes"),
                    }
                )

            all_lines: list[str] = []
            all_scores: list[float] = []
            for page in page_results:
                all_lines.extend(page["lines"])
                all_scores.extend(page["confidences"] or [])

            full_text = "\n".join(all_lines)
            avg_conf = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

            total_elapsed = time.time() - request_start
            log(f"[API] OCR completed for {filename} (total: {total_elapsed:.2f}s)")

            return {
                "filename": filename,
                "page_count": len(page_results),
                "avg_confidence": round(avg_conf, 4),
                "full_text": full_text,
                "pages": page_results,
            }
        except HTTPException:
            raise
        except Exception as exc:
            if not fallback:
                raise
            log(f"[API] OCR container failed, using legacy backend: {exc}")
            legacy_result = await run_legacy_ocr(filename, file_content)
            total_elapsed = time.time() - request_start
            log(
                f"[API] Legacy OCR completed for {filename} (total: {total_elapsed:.2f}s)"
            )
            return legacy_result
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                if repaired_path and repaired_path.exists():
                    repaired_path.unlink()
                if output_dir.exists():
                    for file in output_dir.glob("*"):
                        file.unlink(missing_ok=True)
                    output_dir.rmdir()
            except Exception as exc:
                log(f"[API] OCR cleanup warning: {exc}")

    return await service_queue.enqueue("ocr", do_ocr)


@app.post("/anonymize")
async def anonymize_document(
    request: AnonymizationRequest, x_api_key: str = Header(None)
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
                headers={"X-API-Key": x_api_key} if x_api_key else {},
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=response.text
                )

            total_elapsed = time.time() - request_start
            log(
                f"[API] Anonymization completed ({text_len} chars, total: {total_elapsed:.2f}s)"
            )
            return response.json()

    return await service_queue.enqueue("anon", do_anonymize)


@app.get("/status")
async def get_status():
    """Get current service and queue status"""
    return {
        "ocr_running": is_service_running("ocr"),
        "anon_running": is_service_running("anon"),
        "services": {
            "ocr": SERVICES["ocr"].get("container", "docker"),
            "ocr_backend": get_active_ocr_backend(),
            "anon": SERVICES["anon"].get("url"),
            "anon_backend": ANON_BACKEND,
        },
        "queue": service_queue.get_status(),
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
            "anon_pending": queue_status["queued"]["anon"],
        },
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine Service Manager (with Smart Queue)")
    print("=" * 60)
    print("Listening on: http://0.0.0.0:8004")
    print("OCR endpoint: http://0.0.0.0:8004/ocr (paddlex-ocr-hpi container)")
    print(
        f"Anon endpoint: http://0.0.0.0:8004/anonymize (forwards to :9002, backend={ANON_BACKEND})"
    )
    print("Status: http://0.0.0.0:8004/status")
    print("=" * 60)
    print("Queue Strategy:")
    print("  - Batches requests by service type")
    print("  - Minimizes expensive service switches")
    print("  - OCR loads in ~11s, Anon loads in ~1s")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8004)

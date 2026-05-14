#!/usr/bin/env python3
"""
Smart Service Manager for OCR and Anonymization
Automatically loads/unloads services to optimize VRAM usage
Uses a service-aware queue to batch requests and minimize service switches
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from pydantic import BaseModel
import httpx
import subprocess
import time
import psutil
import os
import asyncio
import json
import hashlib
import shlex
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional
from collections import deque
import uuid
from urllib.parse import urlparse

app = FastAPI(title="Rechtmaschine Service Manager")


def log(message: str):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_command(command: str | None) -> list[str] | None:
    if not command:
        return None
    try:
        parsed = shlex.split(command)
    except Exception:
        return None
    return parsed if parsed else None


def _as_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _as_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _normalize_ollama_base_url(raw_url: str | None) -> str:
    raw = (raw_url or "http://localhost:11434").strip()
    if not raw:
        return "http://localhost:11434"

    if raw.endswith("/api/generate"):
        raw = raw[: -len("/api/generate")]

    return raw.rstrip("/")


def _ollama_base_url() -> str:
    return _normalize_ollama_base_url(os.getenv("OLLAMA_URL", "http://localhost:11434"))


def _ollama_generate_url() -> str:
    return f"{_ollama_base_url()}/api/generate"


# Optional OCR overrides (RAM-disk / alternate service file).
OCR_SERVICE_FILE = os.getenv("OCR_SERVICE_FILE", "ocr_service_hibernate.py")
OCR_HPI_BACKEND = os.getenv("OCR_HPI_BACKEND", "paddle")
OCR_MODEL_ENV: dict[str, str] = {}
for _name in [
    "OCR_MODEL_BASE_DIR",
    "OCR_DET_MODEL_DIR",
    "OCR_REC_MODEL_DIR",
    "OCR_CLS_MODEL_DIR",
]:
    _value = os.getenv(_name)
    if _value:
        OCR_MODEL_ENV[_name] = _value

# Get base directory (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Anonymization service
ANON_MODEL_DEFAULTS = {
    "qwen": "qwen3:8b",
    "gemma": "gemma3:12b",
    "gemma4": "gemma4:e4b",
    "gemma4_openrouter": "google/gemma-4-31b-it",
}
ANON_BACKEND_ALIASES = {
    "qwen3": "qwen",
    "qwen3:8b": "qwen",
    "gemma3": "gemma",
    "gemma3:12b": "gemma",
    "gemma4-smart": "gemma4",
    "gemma4_hybrid": "gemma4",
    "gemma4-4b": "gemma4",
    "gemma4_4b": "gemma4",
    "gemma4:e4b": "gemma4",
    "gemma4-openrouter": "gemma4_openrouter",
    "gemma4-31b-openrouter": "gemma4_openrouter",
    "openrouter-gemma4-31b": "gemma4_openrouter",
    "google/gemma-4-31b-it": "gemma4_openrouter",
}
ANON_OLLAMA_URL = _ollama_generate_url()
ANON_BACKEND = os.getenv("ANON_BACKEND", "gemma").strip().lower()
KEEP_SERVICES_RUNNING = _env_flag("KEEP_SERVICES_RUNNING", False)
FLAIR_FORCE_CPU = _env_flag("FLAIR_FORCE_CPU", True)
ANON_BACKEND = ANON_BACKEND_ALIASES.get(ANON_BACKEND, ANON_BACKEND)
ANON_MODEL = ANON_MODEL_DEFAULTS.get(ANON_BACKEND, "")
SERVICE_MANAGER_ROLE = os.getenv("SERVICE_MANAGER_ROLE", "all").strip().lower()
SERVICE_MANAGER_ROLES = {
    role.strip()
    for role in SERVICE_MANAGER_ROLE.replace(",", " ").split()
    if role.strip()
}
if not SERVICE_MANAGER_ROLES:
    SERVICE_MANAGER_ROLES = {"all"}
SERVICE_MANAGER_HOST = os.getenv("SERVICE_MANAGER_HOST", "0.0.0.0")
SERVICE_MANAGER_PORT = _as_int_env("SERVICE_MANAGER_PORT", 8004)


# RAG defaults
RAG_EMBED_ENABLED = _env_flag("RAG_EMBED_ENABLED", True)
RAG_RERANK_ENABLED = _env_flag("RAG_RERANK_ENABLED", False)
RAG_EMBED_URL = os.getenv("RAG_EMBED_URL", "http://127.0.0.1:8085").rstrip("/")
RAG_RERANK_URL = os.getenv("RAG_RERANK_URL", "http://127.0.0.1:8086").rstrip("/")
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
RAG_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RAG_EMBED_CONTAINER_NAME = os.getenv("RAG_EMBED_CONTAINER_NAME", "bge-m3")
RAG_RERANK_CONTAINER_NAME = os.getenv("RAG_RERANK_CONTAINER_NAME", "bge-reranker")
RAG_EMBED_HEALTH_URL = os.getenv("RAG_EMBED_HEALTH_URL")
RAG_RERANK_HEALTH_URL = os.getenv("RAG_RERANK_HEALTH_URL")
RAG_EMBED_REQUEST_PATH = os.getenv("RAG_EMBED_REQUEST_PATH", "/embed").strip() or "/embed"
RAG_RERANK_REQUEST_PATH = os.getenv("RAG_RERANK_REQUEST_PATH", "/rerank").strip() or "/rerank"
RAG_EMBED_HEALTH_PATH = os.getenv("RAG_EMBED_HEALTH_PATH", "/health").strip() or "/health"
RAG_RERANK_HEALTH_PATH = os.getenv("RAG_RERANK_HEALTH_PATH", "/health").strip() or "/health"
RAG_EMBED_KIND = os.getenv("RAG_EMBED_KIND", "external").strip().lower() or "external"
RAG_RERANK_KIND = os.getenv("RAG_RERANK_KIND", "external").strip().lower() or "external"
RAG_EMBED_KIND = (
    RAG_EMBED_KIND if RAG_EMBED_ENABLED else "external_disabled"
)
RAG_RERANK_KIND = (
    RAG_RERANK_KIND if RAG_RERANK_ENABLED else "external_disabled"
)
RAG_EMBED_PROCESS_NAME = os.getenv("RAG_EMBED_PROCESS_NAME", "text-embeddings-inference")
RAG_RERANK_PROCESS_NAME = os.getenv("RAG_RERANK_PROCESS_NAME", "reranker_service.py")
RAG_EMBED_START_CMD = _parse_command(os.getenv("RAG_EMBED_START_CMD"))
RAG_RERANK_START_CMD = _parse_command(os.getenv("RAG_RERANK_START_CMD"))
RAG_EMBED_PORT = _as_int_env("RAG_EMBED_PORT", 8085)
RAG_RERANK_PORT = _as_int_env("RAG_RERANK_PORT", 8086)
RAG_EMBED_LOAD_TIME = _as_float_env("RAG_EMBED_LOAD_TIME", 1.0)
RAG_RERANK_LOAD_TIME = _as_float_env("RAG_RERANK_LOAD_TIME", 1.0)
RAG_SERVICE_TIMEOUT_SEC = _as_float_env("RAG_SERVICE_TIMEOUT_SEC", 120.0)
RAG_EMBED_QUERY_TIMEOUT = _as_float_env("RAG_EMBED_QUERY_TIMEOUT", 20.0)
RAG_EMBED_BATCH_TIMEOUT = _as_float_env("RAG_EMBED_BATCH_TIMEOUT", 120.0)
RAG_RERANK_TIMEOUT = _as_float_env("RAG_RERANK_TIMEOUT", 30.0)
RAG_SERVICE_API_KEYS = [
    key.strip() for key in os.getenv("RAG_SERVICE_API_KEYS", "").split(",") if key.strip()
]
if not RAG_SERVICE_API_KEYS:
    fallback_key = os.getenv("RAG_SERVICE_API_KEY", "").strip()
    if fallback_key:
        RAG_SERVICE_API_KEYS = [fallback_key]
RAG_REQUEST_ID_HEADER = "X-Request-ID"


RAG_REQUEUEABLE_SERVICES = ("ocr", "anon", "embed", "rerank")


def _service_role_enabled(service_name: str) -> bool:
    if "all" in SERVICE_MANAGER_ROLES:
        return True
    if service_name in {"ocr", "ocr_legacy"}:
        return "ocr" in SERVICE_MANAGER_ROLES
    if service_name in {"anon", "anon_legacy"}:
        return bool({"anon", "anonymization", "llm", "qwen"} & SERVICE_MANAGER_ROLES)
    if service_name in {"embed", "rerank"}:
        return bool({"rag", service_name} & SERVICE_MANAGER_ROLES)
    return True


def _build_anon_ollama_backend(
    model_name: str,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = {
        "OLLAMA_URL": ANON_OLLAMA_URL,
        "OLLAMA_MODEL": model_name,
    }
    if extra_env:
        env.update(extra_env)
    return {
        "kind": "process",
        "port": 9002,
        "url": "http://localhost:9002",
        "process_name": "anonymization_service.py",
        "start_cmd": ["python3", "anonymization_service.py"],
        "cwd": os.path.join(BASE_DIR, "anon"),
        "venv": os.path.join(BASE_DIR, "anon", ".venv", "bin", "python3"),
        "env": env,
        "model": model_name,
        "load_time": 1,
    }


def _build_anon_openrouter_backend(model_name: str) -> dict[str, Any]:
    return {
        "kind": "process",
        "port": 9002,
        "url": "http://localhost:9002",
        "process_name": "anonymization_service.py",
        "start_cmd": ["python3", "anonymization_service.py"],
        "cwd": os.path.join(BASE_DIR, "anon"),
        "venv": os.path.join(BASE_DIR, "anon", ".venv", "bin", "python3"),
        "env": {
            "ANON_PROVIDER": "openrouter",
            "OPENROUTER_MODEL": model_name,
            "OPENROUTER_URL": os.getenv(
                "OPENROUTER_URL",
                "https://openrouter.ai/api/v1/chat/completions",
            ),
            "OPENROUTER_SITE_URL": os.getenv("OPENROUTER_SITE_URL", "https://rechtmaschine.de"),
            "OPENROUTER_APP_NAME": os.getenv("OPENROUTER_APP_NAME", "Rechtmaschine"),
        },
        "model": model_name,
        "load_time": 1,
    }


ANON_BACKENDS = {
    "flair": {
        "kind": "process",
        "port": 9002,
        "url": "http://localhost:9002",
        "process_name": "anonymization_service_flair.py",
        "start_cmd": ["python3", "anonymization_service_flair.py"],
        "cwd": os.path.join(BASE_DIR, "anon"),
        "venv": os.path.join(BASE_DIR, "anon", ".venv", "bin", "python3"),
        "env": {"CUDA_VISIBLE_DEVICES": ""} if FLAIR_FORCE_CPU else {},
        "load_time": 1,
    },
    "qwen": _build_anon_ollama_backend(ANON_MODEL_DEFAULTS["qwen"]),
    "gemma": _build_anon_ollama_backend(ANON_MODEL_DEFAULTS["gemma"]),
    "gemma4": _build_anon_ollama_backend(
        ANON_MODEL_DEFAULTS["gemma4"],
        extra_env={
            "ANON_ENABLE_PAGE_CHUNKING": "1",
            "ANON_CHUNK_PAGES": "10",
            "ANON_CHUNK_MIN_PAGES": "11",
            "ANON_ENABLE_FLAIR_AUGMENT": "1",
            "FLAIR_FORCE_CPU": "1",
        },
    ),
    "gemma4_openrouter": _build_anon_openrouter_backend(ANON_MODEL_DEFAULTS["gemma4_openrouter"]),
}
if ANON_BACKEND not in ANON_BACKENDS:
    log(f"[Manager] Unknown ANON_BACKEND '{ANON_BACKEND}', defaulting to 'gemma'")
    ANON_BACKEND = "gemma"

ANON_MODEL_OVERRIDE = os.getenv("OLLAMA_MODEL", "").strip()
if ANON_MODEL_OVERRIDE and ANON_BACKEND in {"qwen", "gemma", "gemma4"}:
    ANON_BACKENDS[ANON_BACKEND]["env"]["OLLAMA_MODEL"] = ANON_MODEL_OVERRIDE
    ANON_BACKENDS[ANON_BACKEND]["model"] = ANON_MODEL_OVERRIDE

ANON_MODEL = ANON_BACKENDS.get(ANON_BACKEND, {}).get("model", ANON_MODEL_DEFAULTS["gemma"])

# Service configurations
SERVICES = {
    "ocr": {
        "kind": "process",
        "port": 9003,
        "url": "http://127.0.0.1:9003",
        "use_http_service": True,
        "supports_hibernate": True,
        "process_name": OCR_SERVICE_FILE,
        "process_match": os.path.join("ocr", ".venv_hpi", "bin", "python"),
        "start_cmd": ["bash", "run_hpi_service.sh"],
        "cwd": os.path.join(BASE_DIR, "ocr"),
        "venv": "/bin/bash",
        "env": {
            "OCR_SERVICE_FILE": OCR_SERVICE_FILE,
            "OCR_RETURN_WORD_BOX": "1",
            "OCR_USE_DOC_ORIENTATION": "1",
            "OCR_USE_TEXTLINE_ORIENTATION": "1",
            "OCR_USE_UNWARPING": "0",
            "OCR_ENABLE_HPI": "1",
            "OCR_HPI_BACKEND": OCR_HPI_BACKEND,
            "DISABLE_MODEL_SOURCE_CHECK": "True",
            **OCR_MODEL_ENV,
        },
        "load_time": 11,  # seconds - OCR takes longer due to CUDA init
        "fallback": "ocr_legacy",
    },
    "ocr_legacy": {
        "kind": "process",
        "port": 9003,
        "url": "http://127.0.0.1:9003",
        "process_name": "ocr_service.py",
        "process_match": os.path.join("ocr", ".venv", "bin", "python"),
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
    "anon": {"kind": "ollama", "load_time": 1},
    "anon_legacy": ANON_BACKENDS[ANON_BACKEND],
}

if RAG_EMBED_ENABLED:
    SERVICES["embed"] = {
        "kind": RAG_EMBED_KIND,
        "api_key": os.getenv("RAG_EMBED_API_KEY", ""),
        "container": RAG_EMBED_CONTAINER_NAME,
        "port": RAG_EMBED_PORT,
        "url": RAG_EMBED_URL,
        "use_http_service": True,
        "health_url": (
            RAG_EMBED_HEALTH_URL
            if RAG_EMBED_HEALTH_URL
            else f"{RAG_EMBED_URL}{RAG_EMBED_HEALTH_PATH}"
        ),
        "request_path": RAG_EMBED_REQUEST_PATH,
        "start_cmd": RAG_EMBED_START_CMD,
        "cwd": os.path.join(BASE_DIR, "rag"),
        "venv": os.path.join(BASE_DIR, "rag", ".venv", "bin", "python3"),
        "process_name": RAG_EMBED_PROCESS_NAME,
        "load_time": RAG_EMBED_LOAD_TIME,
        "model": RAG_EMBED_MODEL,
    }

if RAG_RERANK_ENABLED:
    SERVICES["rerank"] = {
        "kind": RAG_RERANK_KIND,
        "api_key": os.getenv("RAG_RERANK_API_KEY", ""),
        "container": RAG_RERANK_CONTAINER_NAME,
        "port": RAG_RERANK_PORT,
        "url": RAG_RERANK_URL,
        "use_http_service": True,
        "health_url": (
            RAG_RERANK_HEALTH_URL
            if RAG_RERANK_HEALTH_URL
            else f"{RAG_RERANK_URL}{RAG_RERANK_HEALTH_PATH}"
        ),
        "request_path": RAG_RERANK_REQUEST_PATH,
        "start_cmd": RAG_RERANK_START_CMD,
        "cwd": os.path.join(BASE_DIR, "rag"),
        "venv": os.path.join(BASE_DIR, "rag", ".venv", "bin", "python3"),
        "process_name": RAG_RERANK_PROCESS_NAME,
        "load_time": RAG_RERANK_LOAD_TIME,
        "model": RAG_RERANK_MODEL,
    }


SERVICES = {
    service_name: config
    for service_name, config in SERVICES.items()
    if _service_role_enabled(service_name)
}


class AnonymizationRequest(BaseModel):
    text: str
    document_type: str


class OllamaJsonRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    think: Optional[bool] = None
    format: dict | str = "json"
    options: dict = {}


EXTRACTION_ENTITY_KEYS = [
    "names",
    "birth_dates",
    "birth_places",
    "streets",
    "postal_codes",
    "cities",
    "azr_numbers",
    "aufenthaltsgestattung_ids",
    "case_numbers",
]


def _sanitize_model_response_preview(value: Any, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    compact = " ".join(value.split())
    return compact[:limit]


def _strip_model_response_wrappers(raw_response: str) -> str:
    cleaned = (raw_response or "").strip()
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"(?is)^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"(?is)\s*```$", "", cleaned).strip()
    return cleaned


def _extract_first_json_object(text: str) -> Optional[str]:
    start = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _loads_model_json(raw_response: str) -> Any:
    cleaned = _strip_model_response_wrappers(raw_response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        json_object = _extract_first_json_object(cleaned)
        if json_object:
            return json.loads(json_object)
        raise first_error


def _parse_model_json_payload(data: dict[str, Any], context: str) -> tuple[Any, str]:
    fields = [
        ("parsed_payload", data.get("parsed_payload")),
        ("response", data.get("response")),
        ("thinking", data.get("thinking")),
    ]
    last_error: Optional[json.JSONDecodeError] = None

    for field_name, raw_value in fields:
        if isinstance(raw_value, (dict, list)):
            return raw_value, field_name
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        try:
            return _loads_model_json(raw_value), field_name
        except json.JSONDecodeError as exc:
            last_error = exc
            preview = _sanitize_model_response_preview(raw_value)
            log(
                f"[WARN] Failed to parse {context} JSON from {field_name}: "
                f"{exc}; prefix={preview!r}"
            )

    response_preview = _sanitize_model_response_preview(data.get("response"))
    thinking_preview = _sanitize_model_response_preview(data.get("thinking"))
    detail = (
        f"No parseable {context} JSON "
        f"(response_prefix={response_preview!r}, thinking_prefix={thinking_preview!r})"
    )
    if last_error is not None:
        detail += f": {last_error}"
    raise ValueError(detail)


def _normalize_extraction_entities(payload: Any) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {key: [] for key in EXTRACTION_ENTITY_KEYS}
    if not isinstance(payload, dict):
        return normalized

    for key in EXTRACTION_ENTITY_KEYS:
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            clean = value.strip()
            if not clean:
                continue
            marker = clean.casefold()
            if marker in seen:
                continue
            seen.add(marker)
            out.append(clean)
        normalized[key] = out
    return normalized


class EmbedRequest(BaseModel):
    text: str
    with_sparse: bool = True
    input_type: str = "query"


class EmbedBatchItem(BaseModel):
    id: str
    text: str
    input_type: str = "passage"


class EmbedBatchRequest(BaseModel):
    items: list[EmbedBatchItem]
    with_sparse: bool = True


class RerankRequest(BaseModel):
    query: str
    documents: list[dict]
    top_k: int = 8


SERVICE_REQUEST_TYPES = tuple(
    service_name
    for service_name in RAG_REQUEUEABLE_SERVICES
    if service_name in SERVICES
)


def _services_to_clear_for_switch(service_name: str) -> tuple[str, ...]:
    """Return services that must be unloaded before activating the target service."""
    if service_name == "anon":
        return tuple(
            service
            for service in SERVICE_REQUEST_TYPES
            if service != "anon"
        )

    if service_name in {"embed", "rerank", "ocr"}:
        return ("anon",) if "anon" in SERVICE_REQUEST_TYPES else tuple()

    return tuple(service for service in SERVICE_REQUEST_TYPES if service != service_name)


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
            name: deque() for name in SERVICE_REQUEST_TYPES
        }
        self.current_service: str | None = None
        self.processing = False
        self.lock = asyncio.Lock()
        self.stats: dict[str, int] = {"switches": 0}
        for name in SERVICE_REQUEST_TYPES:
            self.stats[name] = 0

        log(
            f"[Manager] Anonymization backend: {ANON_BACKEND} "
            f"(model={ANON_MODEL or 'n/a'}, keep_services_running={KEEP_SERVICES_RUNNING})"
        )

    async def enqueue(self, service: str, handler: Callable[[], Awaitable[Any]]) -> Any:
        """Add request to queue and wait for result"""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request = QueuedRequest(
            service=service, handler=handler, future=future, queued_at=time.time()
        )

        async with self.lock:
            if service not in self.queues:
                raise HTTPException(status_code=400, detail=f"Unsupported service '{service}'")
            self.queues[service].append(request)
            queue_pos = len(self.queues[service])
            other_pending = sum(
                len(items) for other_service, items in self.queues.items()
                if other_service != service
            )

            log(
                f"[Queue] {service.upper()} request queued (position {queue_pos}, "
                f"{other_pending} other pending)"
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
        if not any(self.queues.values()):
            return None

        # Prefer currently loaded service to avoid expensive switches
        if self.current_service and self.queues[self.current_service]:
            return self.current_service

        for name in SERVICE_REQUEST_TYPES:
            if self.queues[name]:
                return name
        return None

    def _ensure_service_ready(self, service_name: str):
        """Ensure target service is ready (runs in thread pool)"""
        t0 = time.time()

        if not KEEP_SERVICES_RUNNING:
            target_kind = SERVICES[service_name].get("kind", "process")
            clear_services = _services_to_clear_for_switch(service_name)
            if clear_services:
                log(
                    f"[Manager] Clearing services before {service_name} "
                    f"(target kind={target_kind}): {', '.join(clear_services)}"
                )

            # Enforce VRAM policy before loading target service.
            for other_service in clear_services:
                if other_service == service_name:
                    continue
                if is_service_running(other_service):
                    kill_service(other_service)

            # Unload any resident Ollama model when switching TO a non-ollama
            # service (e.g. OCR needs the VRAM). Skip if the target is ollama.
            if target_kind != "ollama":
                unload_ollama_model()
        else:
            log("[Manager] KEEP_SERVICES_RUNNING enabled; skipping service stop/unload")

        # Start target service if not running
        if not is_service_running(service_name):
            start_service(service_name)
        else:
            # Verify it's responsive
            config = SERVICES[service_name]
            try:
                service_health_url = config.get("health_url")
                if not service_health_url and config.get("url"):
                    service_health_url = f"{config['url'].rstrip('/')}/health"
                if not service_health_url:
                    log(f"[Manager] {service_name} already running (no health check, {time.time() - t0:.1f}s)")
                    return
                response = httpx.get(service_health_url, timeout=5.0)
                response.raise_for_status()
                log(f"[Manager] {service_name} already running and healthy ({time.time() - t0:.1f}s)")
            except Exception as e:
                raise Exception(f"{service_name} not responding: {e}")

        if service_name == "ocr" and SERVICES["ocr"].get("supports_hibernate"):
            if not load_ocr_service():
                raise Exception("OCR /load failed")
            log(f"[Manager] OCR loaded and ready ({time.time() - t0:.1f}s total)")

    def get_status(self) -> dict:
        """Get current queue status"""
        return {
            "current_service": self.current_service,
            "processing": self.processing,
            "queued": {name: len(queue) for name, queue in self.queues.items()},
            "stats": self.stats,
        }


def _service_error(
    code: str, message: str, retryable: bool = False, details: Optional[dict] = None
) -> dict[str, dict]:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": details or {},
        }
    }


def _service_auth_keys(service_name: str) -> list[str]:
    config = SERVICES.get(service_name, {})
    service_key = str(config.get("api_key", "")).strip()
    keys = []
    if service_key:
        keys.append(service_key)
    for key in RAG_SERVICE_API_KEYS:
        if key and key not in keys:
            keys.append(key)
    return keys


def _validate_service_key(service_name: str, request: Request):
    keys = _service_auth_keys(service_name)
    if not keys:
        return

    provided = request.headers.get("X-API-Key")
    if provided and provided in keys:
        return

    raise HTTPException(
        status_code=401,
        detail=_service_error(
            "unauthorized",
            "Missing or invalid X-API-Key",
            retryable=False,
            details={"service": service_name},
        ),
    )


def _forward_headers(service_name: str, request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    service_key = str(SERVICES.get(service_name, {}).get("api_key", "")).strip()
    provided = request.headers.get("X-API-Key")
    if provided:
        headers["X-API-Key"] = provided
    elif service_key:
        headers["X-API-Key"] = service_key

    request_id = request.headers.get(RAG_REQUEST_ID_HEADER)
    if request_id:
        headers[RAG_REQUEST_ID_HEADER] = request_id
    return headers


def _service_url(service_name: str) -> Optional[str]:
    return SERVICES[service_name].get("url")


def _service_health_url(service_name: str) -> Optional[str]:
    config = SERVICES[service_name]
    explicit = config.get("health_url")
    if explicit:
        return explicit
    base_url = config.get("url")
    if base_url:
        return f"{base_url.rstrip('/')}/health"
    return None


def _service_request_path(service_name: str) -> str:
    return SERVICES[service_name].get("request_path", "/").strip() or "/"


def _wait_for_service_health_url(
    service_name: str, service_url: str, timeout: int
) -> None:
    log(f"[Manager] Waiting for {service_name} health at {service_url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(service_url, timeout=2.0)
            if response.status_code == 200:
                elapsed = int(time.time() - start_time)
                log(f"[Manager] {service_name} ready after {elapsed}s")
                return
        except Exception:
            pass
        time.sleep(1)

    raise Exception(f"{service_name} failed to start within {timeout}s")


def _is_openai_style_embedding(service_name: str) -> bool:
    path = _service_request_path(service_name).lower()
    return path.endswith("/embeddings") or path.endswith("/v1/embeddings")


def _build_embed_request_payload(service_name: str, body: EmbedRequest) -> dict[str, Any]:
    if _is_openai_style_embedding(service_name):
        return {
            "model": SERVICES[service_name].get("model"),
            "input": body.text,
        }
    return {
        "inputs": [body.text],
    }


def _build_batch_embed_request_payload(
    service_name: str,
    body: EmbedBatchRequest,
) -> dict[str, Any]:
    texts = [item.text for item in body.items]
    if _is_openai_style_embedding(service_name):
        return {
            "model": SERVICES[service_name].get("model"),
            "input": texts,
        }
    return {"inputs": texts}


def _build_rerank_request_payload(service_name: str, body: RerankRequest) -> dict[str, Any]:
    request_path = _service_request_path(service_name).lower()
    texts = []
    for document in body.documents:
        if isinstance(document, str):
            texts.append(document)
            continue
        if isinstance(document, dict):
            document_text = document.get("text")
            if isinstance(document_text, str):
                texts.append(document_text)
                continue
        texts.append(str(document))

    if request_path.endswith("/rerank"):
        return {"query": body.query, "texts": texts, "top_n": body.top_k}

    return body.model_dump()


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

    if kind == "external_disabled":
        return False

    if kind in {"external", "external_api", "external_http"}:
        health_url = _service_health_url(service_name)
        if not health_url:
            return False
        try:
            response = httpx.get(health_url, timeout=3.0)
            return response.status_code == 200
        except Exception:
            return False

    if kind == "ollama":
        ollama_url = _ollama_base_url()
        try:
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    if kind == "docker":
        container = config.get("container")
        if container and is_container_running(container):
            return True
        fallback = config.get("fallback")
        if fallback:
            return is_service_running(fallback)
        return False

    process_name = config["process_name"]
    process_match = config.get("process_match")
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if not cmdline:
                continue
            full_cmd = " ".join(cmdline)
            if process_name in full_cmd:
                if process_match and process_match not in full_cmd:
                    continue
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def _call_service_endpoint(service_name: str, path: str, timeout: float = 10.0) -> bool:
    config = SERVICES.get(service_name, {})
    if not config.get("use_http_service"):
        return False
    service_url = config.get("url")
    if not service_url:
        return False
    try:
        response = httpx.post(f"{service_url}{path}", timeout=timeout)
        if response.status_code == 409:
            log(f"[Manager] {service_name} endpoint {path} busy (409)")
            return False
        response.raise_for_status()
        return True
    except Exception as exc:
        log(f"[Manager] Failed to call {service_name}{path}: {exc}")
        return False


def load_ocr_service() -> bool:
    if not SERVICES["ocr"].get("supports_hibernate"):
        return False
    return _call_service_endpoint("ocr", "/load", timeout=120.0)


def unload_ocr_service() -> bool:
    if not SERVICES["ocr"].get("supports_hibernate"):
        return False
    return _call_service_endpoint("ocr", "/unload", timeout=10.0)


def get_active_ocr_backend() -> str:
    """Return which OCR backend is active"""
    ocr_match = SERVICES["ocr"].get("process_match")
    legacy_match = SERVICES.get("ocr_legacy", {}).get("process_match")

    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if not cmdline:
                continue
            full_cmd = " ".join(cmdline)
            if ocr_match and ocr_match in full_cmd:
                return "host_hpi"
            if legacy_match and legacy_match in full_cmd:
                return "legacy"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

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


async def run_http_ocr(service_url: str, filename: str, file_content: bytes) -> dict:
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{service_url}/ocr",
            files={"file": (filename, file_content, "application/octet-stream")},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


def _coerce_list_of_numbers(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return None
    return result


def _extract_dense(values: Any) -> list[float] | None:
    if isinstance(values, dict):
        if "embedding" in values:
            return _coerce_list_of_numbers(values["embedding"])
        if "dense" in values:
            return _coerce_list_of_numbers(values["dense"])
    if isinstance(values, list):
        return _coerce_list_of_numbers(values)
    return None


def _extract_dense_from_payload(values: Any) -> list[float] | None:
    """Extract the first dense vector from a response payload."""
    dense = _extract_dense(values)
    if dense is not None:
        return dense

    if isinstance(values, dict):
        embeddings = values.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, list):
                return _coerce_list_of_numbers(first)
            if isinstance(first, dict):
                dense = _extract_dense(first)
                if dense is not None:
                    return dense
    return None


def _extract_sparse(values: Any) -> dict[str, list[float] | list[int]] | None:
    if not isinstance(values, dict):
        return None
    sparse_payload = values.get("sparse")
    if isinstance(sparse_payload, dict):
        values = sparse_payload

    indices = values.get("indices")
    coeffs = values.get("values")
    if indices is not None and coeffs is not None:
        if not isinstance(indices, list) or not isinstance(coeffs, list):
            return None
        return {"indices": indices, "values": coeffs}

    indices = values.get("sparse_indices")
    coeffs = values.get("sparse_values")
    if indices is not None and coeffs is not None:
        if not isinstance(indices, list) or not isinstance(coeffs, list):
            return None
        return {"indices": indices, "values": coeffs}

    return None


def _extract_sparse_from_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    sparse = item.get("sparse")
    if isinstance(sparse, dict):
        return _extract_sparse(sparse)
    return _extract_sparse(item)
  
def _extract_dense_and_sparse(values: Any) -> tuple[list[float] | None, dict[str, Any] | None]:
    dense = _extract_dense(values)
    sparse = _extract_sparse_from_item(values)
    return dense, sparse


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_error_payload(message: str, status_code: int, details: Optional[Any] = None):
    return _service_error(
        "upstream_error",
        message,
        retryable=status_code >= 500,
        details=details or {},
    )


def _normalize_embed_response(
    service_name: str,
    request: EmbedRequest,
    response_payload: Any,
    elapsed_ms: float,
) -> dict[str, Any]:
    model = SERVICES[service_name].get("model", "")
    if isinstance(response_payload, dict):
        model = str(response_payload.get("model") or model)

    if isinstance(response_payload, list):
        if not response_payload:
            raise HTTPException(
                status_code=502,
                detail=_to_error_payload(
                    "Invalid embed response payload: missing dense vector",
                    502,
                ),
            )

        first_item = response_payload[0]
        if isinstance(first_item, list):
            dense = _coerce_list_of_numbers(first_item)
        else:
            dense = _extract_dense(first_item) if isinstance(first_item, dict) else None

        if dense is None:
            raise HTTPException(
                status_code=502,
                detail=_to_error_payload(
                    "Invalid embed response payload: missing dense vector",
                    502,
                ),
            )

        result = {
            "model": str(model),
            "dense": dense,
            "dimension": len(dense),
            "elapsed_ms": int(elapsed_ms),
        }
        if request.with_sparse:
            sparse = _extract_sparse_from_item(first_item)
            if sparse is not None:
                result["sparse"] = sparse
        return result

    dense = _extract_dense_from_payload(response_payload)
    if not dense and isinstance(response_payload.get("data"), list):
        dense = _extract_dense(response_payload["data"][0])

    if not dense:
        raise HTTPException(
            status_code=502,
            detail=_to_error_payload(
                "Invalid embed response payload: missing dense vector",
                502,
            ),
        )

    result = {
        "model": model,
        "dense": dense,
        "dimension": int(response_payload.get("dimension") or len(dense)),
        "elapsed_ms": int(elapsed_ms),
    }
    if request.with_sparse:
        sparse = _extract_sparse_from_item(response_payload)
        if sparse is not None:
            result["sparse"] = sparse
    return result


def _normalize_embed_batch_response(
    service_name: str,
    request: EmbedBatchRequest,
    response_payload: Any,
    elapsed_ms: float,
) -> dict[str, Any]:
    model = SERVICES[service_name].get("model", "")
    if isinstance(response_payload, dict):
        model = str(response_payload.get("model") or model)
    requested_items = request.items

    if isinstance(response_payload, list):
        raw_results = response_payload
    elif isinstance(response_payload.get("results"), list):
        raw_results = response_payload.get("results", [])
    elif isinstance(response_payload.get("data"), list):
        raw_results = response_payload.get("data", [])
    elif isinstance(response_payload.get("embeddings"), list):
        raw_results = response_payload.get("embeddings", [])
    else:
        raw_results = []

    normalized_raw = []
    for index, value in enumerate(raw_results):
        if isinstance(value, list):
            normalized_raw.append(
                {
                    "index": index,
                    "id": str(index),
                    "dense": _coerce_list_of_numbers(value),
                }
            )
            continue
        if isinstance(value, dict):
            norm_item = dict(value)
            if "id" not in norm_item:
                norm_item["id"] = str(index)
            if "index" not in norm_item:
                norm_item["index"] = index
            if "dense" not in norm_item:
                norm_item["dense"] = _extract_dense_from_payload(value)
            normalized_raw.append(norm_item)
            continue

        normalized_raw.append({"index": index, "id": str(index)})

    raw_by_index: dict[int, dict[str, Any]] = {}
    raw_by_id: dict[str, dict[str, Any]] = {}
    for item in normalized_raw:
        index_hint = item.get("index")
        if isinstance(index_hint, int):
            raw_by_index[index_hint] = item
        elif isinstance(index_hint, str) and index_hint.isdigit():
            raw_by_index[int(index_hint)] = item

        item_id = item.get("id")
        if isinstance(item_id, str):
            raw_by_id[item_id] = item
        elif isinstance(item_id, int):
            raw_by_id[str(item_id)] = item

    results = []
    failed = []

    for idx, item in enumerate(requested_items):
        id_str = item.id
        raw_item: dict[str, Any] | None = raw_by_id.get(id_str)
        if raw_item is None and idx in raw_by_index:
            raw_item = raw_by_index[idx]
        if raw_item is None and idx < len(normalized_raw):
            raw_item = normalized_raw[idx]
        if raw_item is None:
            raw_item = {}

        if not isinstance(raw_item, dict):
            raw_item = {}

        dense = _extract_dense_from_payload(raw_item)
        if dense is None:
            failed.append({"id": id_str, "error": "missing embedding vector"})
            continue

        result_item = {
            "id": id_str,
            "dense": dense,
        }
        if request.with_sparse:
            sparse = _extract_sparse(raw_item)
            if sparse is not None:
                result_item["sparse"] = sparse

        results.append(result_item)

    if not results and not failed:
        raise HTTPException(
            status_code=502,
            detail=_to_error_payload(
                "Invalid embed-batch response payload",
                502,
            ),
        )

    dimension = 0
    if results:
        first_dense = results[0].get("dense")
        if isinstance(first_dense, list):
            dimension = len(first_dense)

    return {
        "model": model,
        "dimension": dimension,
        "results": results,
        "failed": failed,
        "elapsed_ms": int(elapsed_ms),
    }


def _normalize_rerank_response(
    service_name: str,
    request: RerankRequest,
    response_payload: Any,
    elapsed_ms: float,
) -> dict[str, Any]:
    model = SERVICES[service_name].get("model", "")
    if isinstance(response_payload, dict):
        model = str(response_payload.get("model") or model)
    raw_results = []
    if isinstance(response_payload, list):
        raw_results = response_payload
    elif isinstance(response_payload.get("results"), list):
        raw_results = response_payload.get("results", [])
    elif isinstance(response_payload.get("data"), list):
        raw_results = response_payload.get("data", [])

    results = []
    for idx, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue

        result_id = item.get("id")
        if not result_id:
            if item.get("index") is not None:
                index_for_id = item.get("index")
                if isinstance(index_for_id, int):
                    idx = index_for_id
            document = request.documents[idx] if idx < len(request.documents) else {}
            if isinstance(document, dict):
                result_id = document.get("id", f"doc_{idx}")
            else:
                result_id = f"doc_{idx}"

        score = item.get("score", item.get("relevance", item.get("logits", 0.0)))
        rank = _coerce_int(item.get("rank")) if isinstance(item, dict) else None
        if rank is None:
            rank = idx + 1
        score_float = _coerce_float(score)
        score = score_float if score_float is not None else 0.0
        results.append(
            {
                "id": result_id,
                "score": score,
                "rank": rank,
            }
        )

    if len(results) > request.top_k:
        results = sorted(results, key=lambda item: item["rank"])[:request.top_k]

    if not results:
        # Keep response shape stable even for empty results.
        results = []

    return {
        "model": model,
        "results": results,
        "elapsed_ms": int(elapsed_ms),
    }


async def _post_to_rag_service(
    service_name: str,
    payload: dict[str, Any],
    request: Request,
    timeout: float = RAG_SERVICE_TIMEOUT_SEC,
) -> Any:
    service_url = _service_url(service_name)
    request_path = _service_request_path(service_name)
    if not request_path.startswith("/"):
        request_path = f"/{request_path}"
    if not service_url:
        raise HTTPException(
            status_code=503,
            detail=_to_error_payload(
                f"{service_name} service is not configured",
                503,
            ),
        )

    url = f"{service_url.rstrip('/')}{request_path}"
    headers = _forward_headers(service_name, request)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code < 200 or response.status_code >= 300:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = response.text
                raise HTTPException(
                    status_code=response.status_code,
                    detail=_to_error_payload(
                        "RAG target returned error",
                        response.status_code,
                        error_payload,
                    ),
                )
            try:
                return response.json()
            except ValueError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=_to_error_payload(
                        f"RAG target returned invalid JSON: {exc}",
                        502,
                    ),
                )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=_to_error_payload(
                f"{service_name} request timed out after {timeout}s",
                504,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=_to_error_payload(f"{service_name} request failed: {exc}", 502),
        )


def _ensure_rag_service_enabled(service_name: str):
    if service_name not in SERVICES:
        raise HTTPException(
            status_code=404,
            detail=_to_error_payload(
                f"{service_name} service not configured",
                404,
            ),
        )


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


def _pdf_is_readable(pdf_path: Path) -> bool:
    if not pdf_path.exists():
        return False
    try:
        import pikepdf

        with pikepdf.open(pdf_path):
            return True
    except Exception as exc:
        log(f"[WARNING] PDF read failed for {pdf_path.name}: {exc}")

    try:
        import fitz  # type: ignore

        with fitz.open(pdf_path) as doc:
            return len(doc) > 0
    except Exception as exc:
        log(f"[WARNING] PDF read failed with pymupdf for {pdf_path.name}: {exc}")

    return False


def unload_ollama_model():
    """Unload Ollama model from VRAM"""
    t0 = time.time()
    log("[Manager] Unloading Ollama model(s) from VRAM...")
    try:
        ollama_url = os.getenv("OLLAMA_URL", ANON_OLLAMA_URL)
        parsed = urlparse(_normalize_ollama_base_url(ollama_url))
        if parsed.scheme and parsed.netloc:
            ollama_base_url = f"{parsed.scheme}://{parsed.netloc}"
        else:
            ollama_base_url = "http://localhost:11434"

        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{ollama_base_url}/api/ps")
            response.raise_for_status()
            models = response.json().get("models", [])

            if not models:
                log(f"[Manager] No Ollama models loaded ({time.time() - t0:.1f}s)")
                return

            unloaded_models = []
            for model_info in models:
                model_name = model_info.get("name")
                if not model_name:
                    continue
                client.post(
                    f"{ollama_base_url}/api/generate",
                    json={"model": model_name, "prompt": "", "keep_alive": 0},
                ).raise_for_status()
                unloaded_models.append(model_name)

        if unloaded_models:
            log(f"[Manager] Ollama unloaded: {', '.join(unloaded_models)} ({time.time() - t0:.1f}s)")
        time.sleep(3)  # Wait for VRAM to be freed
        log(f"[Manager] VRAM cooldown done ({time.time() - t0:.1f}s total)")
    except Exception as e:
        log(f"[Manager] Warning: Failed to unload Ollama model: {e}")


def kill_service(service_name: str):
    """Kill a service to free VRAM"""
    config = SERVICES[service_name]
    kind = config.get("kind", "process")

    if kind in {"external", "external_disabled", "external_api", "external_http"}:
        container = config.get("container")
        if container:
            log(f"[Manager] Stopping container for {service_name}: {container}")
            subprocess.run(["docker", "stop", container], check=False)
            time.sleep(2)
            return

        process_match = config.get("process_match")
        process_name = config.get("process_name")
        target = process_match or process_name
        if target:
            log(f"[Manager] Killing {service_name} process using pattern: {target}")
            subprocess.run(["pkill", "-f", target])
            time.sleep(2)
            return

        log(f"[Manager] Skipping stop for {service_name} ({kind})")
        return

    if kind == "ollama":
        unload_ollama_model()
        return

    if kind == "docker":
        log(f"[Manager] Skipping stop for {service_name} container")
        return

    if service_name == "ocr" and config.get("supports_hibernate"):
        if unload_ocr_service():
            log("[Manager] OCR engine unloaded (hibernate mode)")
            return

    process_name = config["process_name"]
    process_match = config.get("process_match")
    log(f"[Manager] Killing {service_name} service to free VRAM...")
    if process_match:
        subprocess.run(["pkill", "-f", process_match])
    else:
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

    if kind == "ollama":
        ollama_url = _ollama_base_url()
        try:
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            log(f"[Manager] Ollama reachable at {ollama_url}")
        except Exception as e:
            raise Exception(f"Ollama not reachable at {ollama_url}: {e}")
        return

    if kind in {"external", "external_api", "external_http"}:
        service_url = _service_health_url(service_name)
        if not service_url:
            if config.get("url"):
                raise Exception(f"{service_name} has no health endpoint configured")
            raise Exception(f"{service_name} not running and not startable")

        start_cmd = config.get("start_cmd")
        if start_cmd:
            env = os.environ.copy()
            service_env = config.get("env")
            if service_env:
                for key, value in service_env.items():
                    env.setdefault(key, value)

            venv = config.get("venv")
            if venv:
                cmd = [venv] + start_cmd[1:]
            else:
                cmd = start_cmd

            log_file = open(f"/tmp/{service_name}_service.log", "a")
            subprocess.Popen(
                cmd,
                cwd=config.get("cwd"),
                env=env,
                stdout=log_file,
                stderr=log_file,
            )
            _wait_for_service_health_url(
                service_name=service_name,
                service_url=service_url,
                timeout=timeout,
            )
            return

        _wait_for_service_health_url(
            service_name=service_name,
            service_url=service_url,
            timeout=timeout,
        )
        return

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
        service_url = config.get("url")
        if service_url:
            _wait_for_service_health_url(
                service_name=service_name,
                service_url=f"{service_url.rstrip('/')}/health",
                timeout=timeout,
            )
        else:
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

    # Log output to file for debugging.
    # Use append mode so logs survive service restarts (useful for post-mortems).
    log_file = open(f"/tmp/{service_name}_service.log", "a")
    subprocess.Popen(cmd, cwd=config["cwd"], env=env, stdout=log_file, stderr=log_file)

    # Poll health endpoint until ready
    service_url = config.get("url")
    if not service_url:
        raise Exception(f"{service_name} has no configured URL")
    _wait_for_service_health_url(
        service_name=service_name,
        service_url=f"{service_url.rstrip('/')}/health",
        timeout=timeout,
    )


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
        ext_lower = ext.lower()
        if ext_lower not in [
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".pdf",
        ]:
            raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")

        tmp_path = None
        repaired_path = None
        output_dir = None

        tmp_dir = Path("/tmp/ocr_service_manager")
        tmp_dir.mkdir(parents=True, exist_ok=True)

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

        def run_ocr(input_path: Path, return_word_box: bool) -> None:
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
                    str(input_path),
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
            if config.get("use_http_service") and config.get("url"):
                result = await run_http_ocr(
                    config["url"], filename, file_content
                )
                total_elapsed = time.time() - request_start
                log(f"[API] OCR completed for {filename} (total: {total_elapsed:.2f}s)")
                return result

            tmp_path = tmp_dir / f"{uuid.uuid4().hex}{ext}"
            tmp_path.write_bytes(file_content)
            ocr_input_path = tmp_path
            if ext_lower == ".pdf" and not _pdf_is_readable(tmp_path):
                repaired_path = _repair_pdf(tmp_path)
                if repaired_path:
                    ocr_input_path = repaired_path

            output_dir = tmp_dir / f"ocr_{uuid.uuid4().hex}"
            output_dir.mkdir(parents=True, exist_ok=True)

            return_word_box = _flag("OCR_RETURN_WORD_BOX", "False") == "True"
            try:
                run_ocr(ocr_input_path, return_word_box)
            except subprocess.CalledProcessError:
                if return_word_box:
                    log("[API] OCR failed with word boxes, retrying without word boxes")
                    run_ocr(ocr_input_path, False)
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
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()
                if repaired_path and repaired_path.exists():
                    repaired_path.unlink()
                if output_dir and output_dir.exists():
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

    try:
        text = request.text or ""
        words = len(text.split())
        lines = text.count("\n") + (1 if text else 0)
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        nulls = text.count("\x00")
        sha256 = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        doc_type = request.document_type
        log(
            "[API] Anonymization payload stats "
            f"(type={doc_type!r}): "
            f"chars={text_len}, words={words}, lines={lines}, "
            f"non_ascii={non_ascii}, nulls={nulls}, sha256={sha256}"
        )
    except Exception as exc:
        log(f"[API] Anonymization payload stats failed: {exc}")

    log(f"[API] Anonymization request received ({text_len} chars)")

    async def do_anonymize():
        """Actual anonymization work - called when service is ready"""
        legacy_key = "anon_legacy"
        legacy_config = SERVICES.get(legacy_key, {})
        legacy_url = legacy_config.get("url")

        if not legacy_url:
            raise HTTPException(
                status_code=500,
                detail=f"Anonymization backend '{ANON_BACKEND}' has no configured URL",
            )

        if legacy_config.get("kind", "process") != "ollama" and not is_service_running(
            legacy_key
        ):
            start_service(legacy_key)

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{legacy_url}/anonymize",
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


async def _run_ollama_json_request(request: OllamaJsonRequest, purpose: str = "Ollama JSON"):
    """Queued direct Ollama generate call for structured JSON-producing prompts."""
    request_start = time.time()
    prompt_len = len(request.prompt)
    log(f"[API] {purpose} request received (model={request.model}, prompt_len={prompt_len})")

    payload = request.model_dump()

    async def do_extract():
        ollama_url = _ollama_base_url()
        ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "300"))
        async with httpx.AsyncClient(timeout=ollama_timeout) as client:
            try:
                response = await client.post(
                    f"{ollama_url}/api/generate",
                    json=payload,
                )
            except httpx.ReadTimeout:
                elapsed = time.time() - request_start
                log(f"[API] {purpose} TIMEOUT after {elapsed:.0f}s (prompt_len={prompt_len}, model={request.model})")
                raise HTTPException(
                    status_code=504,
                    detail=f"Ollama inference timeout after {elapsed:.0f}s (prompt_len={prompt_len})",
                )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=response.text
                )

            total_elapsed = time.time() - request_start
            log(f"[API] {purpose} completed (total: {total_elapsed:.2f}s)")
            data = response.json()
            if isinstance(data, dict):
                data.pop("context", None)
            return data

    return await service_queue.enqueue("anon", do_extract)


@app.post("/ollama-json")
async def ollama_json(request: OllamaJsonRequest):
    """Generic structured Ollama JSON endpoint for citation checks, segmentation, and other non-entity tasks."""
    return await _run_ollama_json_request(request, "Ollama JSON")


@app.post("/extract-entities")
async def extract_entities(request: OllamaJsonRequest):
    """Entity extraction endpoint with parsed JSON normalized at the service boundary."""
    data = await _run_ollama_json_request(request, "Entity extraction")
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned a non-object payload for entity extraction",
        )

    try:
        parsed_payload, parsed_from = _parse_model_json_payload(
            data,
            "entity extraction",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Entity extraction response was not parseable JSON: {exc}",
        )

    data["parse_ok"] = True
    data["parsed_from"] = parsed_from
    data["normalized_entities"] = _normalize_extraction_entities(parsed_payload)
    response_payload = {
        "parse_ok": True,
        "parsed_from": parsed_from,
        "normalized_entities": data["normalized_entities"],
    }
    for key in (
        "model",
        "created_at",
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        if key in data:
            response_payload[key] = data[key]
    return response_payload


@app.get("/v1/health")
async def v1_health():
    embed_enabled = "embed" in SERVICES
    rerank_enabled = "rerank" in SERVICES
    embed_running = is_service_running("embed") if embed_enabled else False
    rerank_running = is_service_running("rerank") if rerank_enabled else False
    status = "healthy"
    if (embed_enabled and not embed_running) or (rerank_enabled and not rerank_running):
        status = "degraded"
    return {
        "status": status,
        "service": "rag-gpu-gateway",
        "embed": {
            "enabled": embed_enabled,
            "model": RAG_EMBED_MODEL,
            "running": embed_running,
        },
        "rerank": {
            "enabled": rerank_enabled,
            "model": RAG_RERANK_MODEL,
            "running": rerank_running,
        },
    }


@app.get("/v1/status")
async def v1_status():
    queue_state = service_queue.get_status()
    return {
        "current_service": queue_state["current_service"],
        "queued": queue_state["queued"],
        "keep_services_running": KEEP_SERVICES_RUNNING,
        "service_running": {
            service: is_service_running(service)
            for service in SERVICE_REQUEST_TYPES
        },
        "processing": queue_state["processing"],
    }


@app.post("/v1/embed")
async def embed_text(request: Request, body: EmbedRequest):
    _ensure_rag_service_enabled("embed")
    _validate_service_key("embed", request)

    async def do_embed():
        t0 = time.time()
        payload = _build_embed_request_payload("embed", body)
        raw = await _post_to_rag_service(
            "embed",
            payload,
            request=request,
            timeout=RAG_EMBED_QUERY_TIMEOUT,
        )
        return _normalize_embed_response("embed", body, raw, (time.time() - t0) * 1000.0)

    return await service_queue.enqueue("embed", do_embed)


@app.post("/v1/embed-batch")
async def embed_text_batch(request: Request, body: EmbedBatchRequest):
    _ensure_rag_service_enabled("embed")
    _validate_service_key("embed", request)
    if not body.items:
        raise HTTPException(
            status_code=400,
            detail=_to_error_payload(
                "Embed batch requires at least one item",
                400,
            ),
        )

    async def do_embed_batch():
        t0 = time.time()
        payload = _build_batch_embed_request_payload("embed", body)
        raw = await _post_to_rag_service(
            "embed",
            payload,
            request=request,
            timeout=RAG_EMBED_BATCH_TIMEOUT,
        )
        return _normalize_embed_batch_response(
            "embed",
            body,
            raw,
            (time.time() - t0) * 1000.0,
        )

    return await service_queue.enqueue("embed", do_embed_batch)


@app.post("/v1/rerank")
async def rerank_documents(request: Request, body: RerankRequest):
    _ensure_rag_service_enabled("rerank")
    _validate_service_key("rerank", request)

    async def do_rerank():
        t0 = time.time()
        payload = _build_rerank_request_payload("rerank", body)
        raw = await _post_to_rag_service(
            "rerank",
            payload,
            request=request,
            timeout=RAG_RERANK_TIMEOUT,
        )
        return _normalize_rerank_response(
            "rerank",
            body,
            raw,
            (time.time() - t0) * 1000.0,
        )

    return await service_queue.enqueue("rerank", do_rerank)


@app.get("/status")
async def get_status():
    """Get current service and queue status"""
    queue_state = service_queue.get_status()
    return {
        "role": sorted(SERVICE_MANAGER_ROLES),
        "ocr_running": is_service_running("ocr") if "ocr" in SERVICES else False,
        "anon_running": is_service_running("anon") if "anon" in SERVICES else False,
        "services": {
            "enabled": sorted(SERVICES.keys()),
            "ocr": "host_hpi" if "ocr" in SERVICES else "disabled",
            "ocr_backend": get_active_ocr_backend() if "ocr" in SERVICES else "disabled",
            "anon": "ollama" if "anon" in SERVICES else "disabled",
            "anon_backend": ANON_BACKEND,
            "anon_model": ANON_MODEL,
        },
        "queue": queue_state,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    queue_status = service_queue.get_status()
    return {
        "status": "healthy",
        "manager": "active",
        "role": sorted(SERVICE_MANAGER_ROLES),
        "enabled_services": sorted(SERVICES.keys()),
        "ocr_loaded": is_service_running("ocr") if "ocr" in SERVICES else False,
        "anon_loaded": is_service_running("anon") if "anon" in SERVICES else False,
        "anon_backend": ANON_BACKEND,
        "anon_model": ANON_MODEL,
        "queue": queue_status["queued"],
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine Service Manager (with Smart Queue)")
    print("=" * 60)
    print(f"Role: {', '.join(sorted(SERVICE_MANAGER_ROLES))}")
    print(f"Enabled services: {', '.join(sorted(SERVICES.keys())) or 'none'}")
    base_url = f"http://{SERVICE_MANAGER_HOST}:{SERVICE_MANAGER_PORT}"
    print(f"Listening on: {base_url}")
    if "ocr" in SERVICES:
        print(f"OCR endpoint: {base_url}/ocr (host HPI)")
    if "anon" in SERVICES:
        print(f"Ollama JSON endpoint: {base_url}/ollama-json (queued direct Ollama)")
        print(f"Entity extraction endpoint: {base_url}/extract-entities (deprecated alias)")
        print(
            f"Anon endpoint: {base_url}/anonymize "
            f"(legacy, backend={ANON_BACKEND}, model={ANON_MODEL or 'n/a'})"
        )
    print(f"Status: {base_url}/status")
    print("=" * 60)
    print("Queue Strategy:")
    print("  - Batches requests by service type")
    print("  - Minimizes expensive service switches")
    print("  - OCR loads in ~11s, Anon loads in ~1s")
    print("=" * 60)

    uvicorn.run(app, host=SERVICE_MANAGER_HOST, port=SERVICE_MANAGER_PORT)

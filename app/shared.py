from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import desc
from sqlalchemy.orm import Session

from database import DATABASE_URL
from events import DOCUMENTS_CHANNEL, SOURCES_CHANNEL, notify_postgres
from models import Document, ResearchSource


# ---------------------------------------------------------------------------
# FastAPI application registration & global limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])

DOWNLOADS_DIR = Path("/app/downloaded_sources")
UPLOADS_DIR = Path("/app/uploads")
OCR_TEXT_DIR = Path("/app/ocr_text")
ANONYMIZED_TEXT_DIR = Path("/app/anonymized_text")

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"
TEMPLATES_DIR = APP_ROOT / "templates"

WOL_MAC = os.getenv("WOL_MAC")
WOL_SSH_HOST = os.getenv("WOL_SSH_HOST", "osmc")
WOL_SSH_USER = os.getenv("WOL_SSH_USER")
WOL_COMMAND = os.getenv("WOL_COMMAND", "wakeonlan {mac}")

SERVICE_MANAGER_HEALTH_URL = os.getenv("SERVICE_MANAGER_HEALTH_URL")
SERVICE_MANAGER_SSH_HOST = os.getenv("SERVICE_MANAGER_SSH_HOST", "desktop")
SERVICE_MANAGER_SSH_USER = os.getenv("SERVICE_MANAGER_SSH_USER")
SERVICE_MANAGER_START_CMD = os.getenv("SERVICE_MANAGER_START_CMD")
SERVICE_MANAGER_START_TIMEOUT_SEC = int(
    os.getenv("SERVICE_MANAGER_START_TIMEOUT_SEC", "180")
)
SERVICE_MANAGER_POLL_INTERVAL_SEC = float(
    os.getenv("SERVICE_MANAGER_POLL_INTERVAL_SEC", "5")
)
SERVICE_MANAGER_SSH_TIMEOUT_SEC = int(
    os.getenv("SERVICE_MANAGER_SSH_TIMEOUT_SEC", "20")
)


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to ensure directory {path}: {exc}")


def clear_directory_contents(directory: Path) -> int:
    """
    Remove all files and subdirectories within the provided directory.
    Returns the number of filesystem entries removed.
    """
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"Error ensuring directory {directory}: {exc}")
        return 0

    removed = 0
    for item in directory.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed += 1
        except Exception as exc:
            print(f"Error removing {item}: {exc}")

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error re-creating directory {directory}: {exc}")

    return removed


for _path in (
    DOWNLOADS_DIR,
    UPLOADS_DIR,
    OCR_TEXT_DIR,
    ANONYMIZED_TEXT_DIR,
    STATIC_DIR,
):
    _ensure_directory(_path)

_fastapi_app: Optional[FastAPI] = None


def register_fastapi_app(app: FastAPI) -> None:
    global _fastapi_app
    _fastapi_app = app


def _require_app() -> FastAPI:
    if _fastapi_app is None:
        raise RuntimeError("FastAPI application not registered in shared module")
    return _fastapi_app


def _build_tailscale_target(host: Optional[str], user: Optional[str]) -> Optional[str]:
    if not host:
        return None
    if user:
        return f"{user}@{host}"
    return host


def _format_wol_command() -> Optional[str]:
    if not WOL_COMMAND:
        return None
    if "{mac}" in WOL_COMMAND:
        if not WOL_MAC:
            print("[WARN] WOL_COMMAND requires {mac} but WOL_MAC is not set")
            return None
        return WOL_COMMAND.format(mac=WOL_MAC)
    return WOL_COMMAND


async def _run_ssh(
    host: Optional[str], user: Optional[str], command: Optional[str]
) -> bool:
    target = _build_tailscale_target(host, user)
    if not target or not command:
        return False

    command_args = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={SERVICE_MANAGER_SSH_TIMEOUT_SEC}",
        target,
        command,
    ]

    def _run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command_args,
            capture_output=True,
            text=True,
            timeout=SERVICE_MANAGER_SSH_TIMEOUT_SEC,
        )

    try:
        result = await asyncio.to_thread(_run)
        if result.returncode != 0:
            print(f"[WARN] ssh failed ({result.returncode}): {result.stderr}")
            return False
        return True
    except Exception as exc:
        print(f"[WARN] ssh error: {exc}")
        return False


async def _is_service_manager_healthy(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            return response.status_code < 500
    except Exception as exc:
        print(f"[INFO] Service manager health check failed: {exc}")
        return False


async def ensure_service_manager_ready() -> None:
    if not SERVICE_MANAGER_HEALTH_URL:
        return

    if await _is_service_manager_healthy(SERVICE_MANAGER_HEALTH_URL):
        return

    wol_command = _format_wol_command()
    if wol_command and WOL_SSH_HOST:
        print("[INFO] Attempting Wake-on-LAN via OSMC...")
        await _run_ssh(WOL_SSH_HOST, WOL_SSH_USER, wol_command)
    else:
        print("[INFO] WOL not configured or WOL host missing; skipping wake")

    if SERVICE_MANAGER_START_CMD and SERVICE_MANAGER_SSH_HOST:
        print("[INFO] Attempting to start service_manager on desktop...")
        await _run_ssh(
            SERVICE_MANAGER_SSH_HOST,
            SERVICE_MANAGER_SSH_USER,
            SERVICE_MANAGER_START_CMD,
        )
    else:
        print("[INFO] Service manager start command not configured")

    start_time = time.monotonic()
    while time.monotonic() - start_time < SERVICE_MANAGER_START_TIMEOUT_SEC:
        if await _is_service_manager_healthy(SERVICE_MANAGER_HEALTH_URL):
            return
        await asyncio.sleep(SERVICE_MANAGER_POLL_INTERVAL_SEC)

    raise HTTPException(
        status_code=503,
        detail="Service manager not ready. Please try again shortly.",
    )


def _text_path_for_document(document_id: uuid.UUID) -> Path:
    return OCR_TEXT_DIR / f"{document_id}.txt"


def store_document_text(document: Document, text: str) -> Optional[str]:
    if not text:
        return document.extracted_text_path

    _ensure_directory(OCR_TEXT_DIR)
    target_path = _text_path_for_document(document.id)
    try:
        target_path.write_text(text, encoding="utf-8")
        document.extracted_text_path = str(target_path)
        return document.extracted_text_path
    except Exception as exc:
        print(f"[ERROR] Failed to write OCR text cache for {document.filename}: {exc}")
        return document.extracted_text_path


def load_document_text(document: Document) -> Optional[str]:
    candidate_paths: List[Path] = []
    if document.extracted_text_path:
        candidate_paths.append(Path(document.extracted_text_path))
    candidate_paths.append(_text_path_for_document(document.id))

    for path_obj in candidate_paths:
        if not path_obj or not path_obj.exists():
            continue
        try:
            text = path_obj.read_text(encoding="utf-8")
            if text and not document.extracted_text_path:
                document.extracted_text_path = str(path_obj)
            return text
        except Exception as exc:
            print(
                f"[WARN] Failed to read cached OCR text for {document.filename}: {exc}"
            )
            continue

    return None


def delete_document_text(document: Document) -> None:
    path = document.extracted_text_path
    if not path:
        return
    try:
        path_obj = Path(path)
        if path_obj.exists():
            path_obj.unlink()
    except Exception as exc:
        print(f"[WARN] Failed to delete OCR text cache for {document.filename}: {exc}")


def get_document_for_upload(entry: Dict[str, Optional[str]]) -> tuple[str, str, bool]:
    """
    Get the appropriate file for uploading a document.
    Prefers OCR text (from disk) when available for better accuracy.

    Returns:
        Tuple of (file_path, mime_type, needs_cleanup)
        - file_path: Path to the file to upload
        - mime_type: MIME type of the file
        - needs_cleanup: Always False (no temporary files used)
    """
    # 1. Prefer Anonymized Text if available (path-only; embedded text deprecated)
    anonymization_metadata = entry.get("anonymization_metadata")
    if entry.get("is_anonymized") and anonymization_metadata:
        anonymized_path = anonymization_metadata.get("anonymized_text_path")
        if anonymized_path:
            path_obj = Path(anonymized_path)
            if path_obj.exists():
                return (str(path_obj), "text/plain", False)

    # 2. Prefer OCR text file if available on disk
    text_path = entry.get("extracted_text_path")
    if text_path:
        path_obj = Path(text_path)
        if path_obj.exists():
            return (str(path_obj), "text/plain", False)

    # 3. Fall back to original PDF
    file_path = entry.get("file_path")
    if not file_path:
        raise ValueError(f"No file_path or extracted_text_path available for document")

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return (str(path_obj), "application/pdf", False)

def ensure_document_on_gemini(document: Any, db: Session) -> Optional[Any]:
    """
    Ensures the document/source file is uploaded to Gemini and returns the file object.
    Reuses existing URI if valid, otherwise uploads and updates DB.
    Handles both Document and ResearchSource models.
    """
    client = get_gemini_client()

    # 1. Check existing URI (skip reuse if anonymized text exists)
    force_refresh = False
    if getattr(document, "is_anonymized", False):
        anon_meta = getattr(document, "anonymization_metadata", None) or {}
        if anon_meta.get("anonymized_text_path"):
            force_refresh = True

    if document.gemini_file_uri and not force_refresh:
        try:
            existing_file = client.files.get(name=document.gemini_file_uri)
            # print(f"[INFO] Reuse Gemini File {document.gemini_file_uri} ({existing_file.state})")
            return existing_file
        except Exception as e:
            print(f"[WARN] URI {document.gemini_file_uri} expired/invalid: {e}")
            document.gemini_file_uri = None
    elif force_refresh and document.gemini_file_uri:
        document.gemini_file_uri = None

    # 2. Upload preferred file if available locally
    # Handle different field names for Document vs ResearchSource
    filename = getattr(document, "filename", None) or getattr(
        document, "title", "document"
    )
    file_path = getattr(document, "file_path", None) or getattr(
        document, "download_path", None
    )

    upload_entry = {
        "filename": filename,
        "file_path": file_path,
        "extracted_text_path": getattr(document, "extracted_text_path", None),
        "anonymization_metadata": getattr(document, "anonymization_metadata", None),
        "is_anonymized": getattr(document, "is_anonymized", False),
    }

    try:
        selected_path, mime_type, _ = get_document_for_upload(upload_entry)
    except Exception as exc:
        print(f"[WARN] No uploadable file for {filename}: {exc}")
        return None

    if selected_path and os.path.exists(selected_path):
        try:
            display_name = filename
            if mime_type == "text/plain" and not display_name.lower().endswith(".txt"):
                display_name = f"{display_name}.txt"
            print(f"[INFO] Uploading {display_name} to Gemini")
            with open(selected_path, "rb") as f:
                uploaded_file = client.files.upload(
                    file=f,
                    config={
                        "mime_type": mime_type,
                        "display_name": display_name,
                    },
                )

            # 3. Persist new URI
            document.gemini_file_uri = uploaded_file.name
            db.add(document)
            db.commit()
            return uploaded_file

        except Exception as e:
            print(f"[ERROR] Upload failed for {filename}: {e}")
            return None

    return None


# ---------------------------------------------------------------------------
# Shared enums and models
# ---------------------------------------------------------------------------


class DocumentCategory(str, Enum):
    ANHOERUNG = "Anhörung"
    BESCHEID = "Bescheid"
    RECHTSPRECHUNG = "Rechtsprechung"
    VORINSTANZ = "Vorinstanz"
    SONSTIGES = "Sonstige gespeicherte Quellen"
    AKTE = "Akte"


class ClassificationResult(BaseModel):
    category: DocumentCategory
    confidence: float
    explanation: str
    filename: str
    gemini_file_uri: Optional[str] = None


class GeminiClassification(BaseModel):
    category: DocumentCategory
    confidence: float
    explanation: str


class ResearchRequest(BaseModel):
    query: Optional[str] = None
    primary_bescheid: Optional[str] = None
    reference_document_id: Optional[str] = None
    selected_documents: Optional[SelectedDocuments] = None
    search_engine: Literal["gemini", "meta", "grok-4-1-fast"] = "meta"
    asylnet_keywords: Optional[str] = None


class ResearchResult(BaseModel):
    query: str
    summary: str
    sources: List[Dict[str, Any]] = []
    suggestions: List[str] = []


class SavedSource(BaseModel):
    id: str
    url: str
    title: str
    description: Optional[str] = None
    document_type: str
    pdf_url: Optional[str] = None
    download_path: Optional[str] = None
    download_status: str = "pending"
    research_query: str
    timestamp: str


class AddSourceRequest(BaseModel):
    title: str
    url: str
    description: Optional[str] = None
    pdf_url: Optional[str] = None
    document_type: str = "Rechtsprechung"
    research_query: Optional[str] = None
    auto_download: bool = False


class RechtsprechungEntryBase(BaseModel):
    document_id: Optional[str] = None
    country: str
    tags: List[str] = []
    court: Optional[str] = None
    court_level: Optional[str] = None
    decision_date: Optional[str] = None
    aktenzeichen: Optional[str] = None
    outcome: Optional[str] = None
    key_facts: List[str] = []
    key_holdings: List[str] = []
    argument_patterns: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    summary: Optional[str] = None
    confidence: Optional[float] = None
    warnings: List[str] = []
    is_active: bool = True


class RechtsprechungEntryCreate(BaseModel):
    document_id: str


class RechtsprechungEntryUpdate(BaseModel):
    country: Optional[str] = None
    tags: Optional[List[str]] = None
    court: Optional[str] = None
    court_level: Optional[str] = None
    decision_date: Optional[str] = None
    aktenzeichen: Optional[str] = None
    outcome: Optional[str] = None
    key_facts: Optional[List[str]] = None
    key_holdings: Optional[List[str]] = None
    argument_patterns: Optional[List[Dict[str, Any]]] = None
    citations: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    warnings: Optional[List[str]] = None
    is_active: Optional[bool] = None


class RechtsprechungEntryResponse(RechtsprechungEntryBase):
    id: str
    extracted_at: Optional[str] = None
    model: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AddDocumentFromUrlRequest(BaseModel):
    title: str
    url: str
    category: DocumentCategory = DocumentCategory.RECHTSPRECHUNG
    auto_download: bool = True


class AnonymizationRequest(BaseModel):
    text: str
    document_type: str


class AnonymizationResult(BaseModel):
    anonymized_text: str
    plaintiff_names: List[str]
    birth_dates: List[str] = []
    addresses: List[str] = []
    confidence: float
    original_text: str
    processed_characters: int


class BescheidSelection(BaseModel):
    primary: Optional[str] = None
    others: List[str] = []


class VorinstanzSelection(BaseModel):
    primary: Optional[str] = None
    others: List[str] = []


class SelectedDocuments(BaseModel):
    anhoerung: List[str] = []
    bescheid: BescheidSelection
    vorinstanz: VorinstanzSelection
    rechtsprechung: List[str] = []
    saved_sources: List[str] = []
    sonstiges: List[str] = []
    akte: List[str] = []


class TokenUsage(BaseModel):
    """Detailed token usage and cost tracking"""

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0  # Claude extended thinking
    cache_read_tokens: int = 0  # Prompt caching
    cache_write_tokens: int = 0
    total_tokens: int = 0

    # Cost in USD (calculated based on model pricing)
    cost_usd: Optional[float] = None
    model: Optional[str] = None


class GenerationMetadata(BaseModel):
    documents_used: Dict[str, int]
    citations_found: int = 0
    missing_citations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    word_count: int = 0
    token_count: Optional[int] = None
    token_usage: Optional[TokenUsage] = None  # Detailed token breakdown


class GenerationResponse(BaseModel):
    success: bool = True
    document_type: str
    user_prompt: str
    generated_text: str
    thinking_text: Optional[str] = None  # Claude extended thinking output
    used_documents: List[Dict[str, str]] = []
    metadata: GenerationMetadata


class GenerationRequest(BaseModel):
    document_type: Literal["Klagebegründung", "Schriftsatz"]
    user_prompt: str
    legal_area: Literal["migrationsrecht", "sozialrecht"] = "migrationsrecht"
    selected_documents: SelectedDocuments
    model: Literal[
        "claude-opus-4-5", "gpt-5.2", "gemini-3-pro-preview", "multi-step-expert"
    ] = "claude-opus-4-5"
    verbosity: Literal["low", "medium", "high"] = "high"
    chat_history: List[Dict[str, str]] = []


class JLawyerSendRequest(BaseModel):
    case_id: str
    template_name: str
    file_name: str
    generated_text: str
    template_folder: Optional[str] = None


class JLawyerResponse(BaseModel):
    success: bool
    message: str


class JLawyerTemplatesResponse(BaseModel):
    templates: List[str]
    folder: str


try:
    GenerationRequest.model_rebuild()
    GenerationResponse.model_rebuild()
except AttributeError:  # pragma: no cover - Legacy Pydantic v1 support
    GenerationRequest.update_forward_refs(
        SelectedDocuments=SelectedDocuments,
        BescheidSelection=BescheidSelection,
    )
    GenerationResponse.update_forward_refs(
        GenerationMetadata=GenerationMetadata,
    )


# ---------------------------------------------------------------------------
# External client factories
# ---------------------------------------------------------------------------


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key, timeout=3600.0)


def get_xai_client() -> OpenAI:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


# ---------------------------------------------------------------------------
# Broadcast utilities
# ---------------------------------------------------------------------------


def _emit_event(
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
    channel: str = DOCUMENTS_CHANNEL,
) -> None:
    app = _require_app()
    hub: Optional[Any] = getattr(app.state, "document_hub", None)
    if not hub:
        return

    message: Dict[str, Any] = {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if payload:
        message.update(payload)

    data = json.dumps(message, ensure_ascii=False)
    try:
        hub.publish(data)
    except Exception as exc:  # pragma: no cover - logging only
        print(f"Broadcast hub publish failed: {exc}")

    try:
        notify_postgres(DATABASE_URL, data, channel)
    except Exception as exc:
        print(f"Postgres NOTIFY failed: {exc}")


def emit_documents_event(
    event_type: str, payload: Optional[Dict[str, Any]] = None
) -> None:
    _emit_event(event_type, payload, DOCUMENTS_CHANNEL)


def emit_sources_event(
    event_type: str, payload: Optional[Dict[str, Any]] = None
) -> None:
    _emit_event(event_type, payload, SOURCES_CHANNEL)


def group_documents(
    documents: List[Document],
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    grouped: Dict[str, List[Dict[str, Optional[str]]]] = {
        "Anhörung": [],
        "Bescheid": [],
        "Vorinstanz": [],
        "Rechtsprechung": [],
        "Akte": [],
        "Sonstige gespeicherte Quellen": [],
    }

    for doc in documents:
        category = doc.category
        if category == "Sonstiges":
            category = "Sonstige gespeicherte Quellen"

        category = category if category in grouped else "Sonstige gespeicherte Quellen"
        grouped[category].append(doc.to_dict())

    return grouped


def build_documents_snapshot(
    db: Session,
    owner_id: Optional[uuid.UUID] = None,
    case_id: Optional[uuid.UUID] = None,
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    query = db.query(Document)
    if owner_id:
        query = query.filter(Document.owner_id == owner_id)
    if case_id:
        query = query.filter(Document.case_id == case_id)
    documents = query.order_by(desc(Document.created_at)).all()
    return group_documents(documents)


def broadcast_documents_snapshot(
    db: Session,
    reason: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    db.rollback()
    payload: Dict[str, Any] = {"reason": reason}
    if extra:
        payload.update(extra)

    # SECURE: Do not broadcast full data in multi-user mode.
    # payload["documents"] = build_documents_snapshot(db)
    emit_documents_event("documents_snapshot", payload)


def build_sources_snapshot(
    db: Session,
    owner_id: Optional[uuid.UUID] = None,
    case_id: Optional[uuid.UUID] = None,
) -> List[Dict[str, Optional[str]]]:
    query = db.query(ResearchSource)
    if owner_id:
        query = query.filter(ResearchSource.owner_id == owner_id)
    if case_id:
        query = query.filter(ResearchSource.case_id == case_id)
    sources = query.order_by(desc(ResearchSource.created_at)).all()
    return [source.to_dict() for source in sources]


def broadcast_sources_snapshot(
    db: Session,
    reason: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {"reason": reason}
    if extra:
        payload.update(extra)

    # SECURE: Do not broadcast full data in multi-user mode.
    # payload["sources"] = build_sources_snapshot(db)
    emit_sources_event("sources_snapshot", payload)


__all__ = [
    "limiter",
    "DOWNLOADS_DIR",
    "UPLOADS_DIR",
    "OCR_TEXT_DIR",
    "APP_ROOT",
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "_ensure_directory",
    "clear_directory_contents",
    "store_document_text",
    "load_document_text",
    "delete_document_text",
    "register_fastapi_app",
    "DocumentCategory",
    "ClassificationResult",
    "GeminiClassification",
    "ResearchRequest",
    "ResearchResult",
    "SavedSource",
    "AddSourceRequest",
    "RechtsprechungEntryBase",
    "RechtsprechungEntryCreate",
    "RechtsprechungEntryUpdate",
    "RechtsprechungEntryResponse",
    "AddDocumentFromUrlRequest",
    "AnonymizationRequest",
    "AnonymizationResult",
    "SelectedDocuments",
    "BescheidSelection",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationMetadata",
    "JLawyerSendRequest",
    "JLawyerResponse",
    "JLawyerTemplatesResponse",
    "get_openai_client",
    "get_gemini_client",
    "get_anthropic_client",
    "get_xai_client",
    "emit_documents_event",
    "emit_sources_event",
    "group_documents",
    "build_documents_snapshot",
    "broadcast_documents_snapshot",
    "build_sources_snapshot",
    "broadcast_sources_snapshot",
    "ensure_document_on_gemini",
]

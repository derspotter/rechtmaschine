from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
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
from models import Case, Document, ResearchSource, User


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


def _role_service_manager_env(role: str, name: str, default: Optional[str] = None) -> Optional[str]:
    if role and role != "default":
        role_prefix = f"{role.upper()}_SERVICE_MANAGER_"
        role_key = f"{role_prefix}{name}"
        if role_key in os.environ:
            return os.getenv(role_key, "")
        if f"{role_prefix}HEALTH_URL" in os.environ:
            return ""
    return os.getenv(f"SERVICE_MANAGER_{name}", default or "")


def _role_service_manager_int(role: str, name: str, default: int) -> int:
    value = _role_service_manager_env(role, name, str(default))
    try:
        return int(value or default)
    except ValueError:
        return default


def _role_service_manager_float(role: str, name: str, default: float) -> float:
    value = _role_service_manager_env(role, name, str(default))
    try:
        return float(value or default)
    except ValueError:
        return default


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


def resolve_case_uuid_for_request(
    db: Session,
    current_user: User,
    requested_case_id: Optional[str] = None,
) -> Optional[uuid.UUID]:
    """Resolve a request-scoped case id, falling back to the user's active case.

    Raises ``HTTPException`` if a requested case id is invalid or not owned by the user.
    """
    raw_case_id = (requested_case_id or "").strip()
    if not raw_case_id:
        return current_user.active_case_id

    try:
        case_uuid = uuid.UUID(raw_case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")

    case = (
        db.query(Case)
        .filter(Case.id == case_uuid, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return case_uuid


def _append_identifier(values: List[str], raw: Optional[str]) -> None:
    value = str(raw or "").strip()
    if value:
        values.append(value)


def collect_selected_document_identifiers(selection: Optional["SelectedDocuments"]) -> List[str]:
    """Collect filename and id selectors from a SelectedDocuments payload."""
    if not selection:
        return []

    identifiers: List[str] = []

    for value in selection.anhoerung:
        _append_identifier(identifiers, value)
    for value in getattr(selection, "anhoerung_ids", []):
        _append_identifier(identifiers, value)

    _append_identifier(identifiers, selection.bescheid.primary)
    _append_identifier(identifiers, getattr(selection.bescheid, "primary_id", None))
    for value in selection.bescheid.others:
        _append_identifier(identifiers, value)
    for value in getattr(selection.bescheid, "other_ids", []):
        _append_identifier(identifiers, value)

    _append_identifier(identifiers, selection.vorinstanz.primary)
    _append_identifier(identifiers, getattr(selection.vorinstanz, "primary_id", None))
    for value in selection.vorinstanz.others:
        _append_identifier(identifiers, value)
    for value in getattr(selection.vorinstanz, "other_ids", []):
        _append_identifier(identifiers, value)

    for value in selection.rechtsprechung:
        _append_identifier(identifiers, value)
    for value in getattr(selection, "rechtsprechung_ids", []):
        _append_identifier(identifiers, value)

    for value in selection.akte:
        _append_identifier(identifiers, value)
    for value in getattr(selection, "akte_ids", []):
        _append_identifier(identifiers, value)

    for value in selection.sonstiges:
        _append_identifier(identifiers, value)
    for value in getattr(selection, "sonstiges_ids", []):
        _append_identifier(identifiers, value)

    return identifiers


def resolve_document_identifier(
    db: Session,
    current_user: User,
    case_id: Optional[uuid.UUID],
    identifier: str,
) -> Optional[Document]:
    normalized = str(identifier or "").strip()
    if not normalized:
        return None

    try:
        identifier_uuid = uuid.UUID(normalized)
        return (
            db.query(Document)
            .filter(
                Document.id == identifier_uuid,
                Document.owner_id == current_user.id,
                Document.case_id == case_id,
            )
            .first()
        )
    except ValueError:
        return (
            db.query(Document)
            .filter(
                Document.filename == normalized,
                Document.owner_id == current_user.id,
                Document.case_id == case_id,
            )
            .first()
        )


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


async def ensure_service_manager_ready(role: str = "default") -> None:
    health_url = _role_service_manager_env(role, "HEALTH_URL", SERVICE_MANAGER_HEALTH_URL)
    ssh_host = _role_service_manager_env(role, "SSH_HOST", SERVICE_MANAGER_SSH_HOST)
    ssh_user = _role_service_manager_env(role, "SSH_USER", SERVICE_MANAGER_SSH_USER)
    start_cmd = _role_service_manager_env(role, "START_CMD", SERVICE_MANAGER_START_CMD)
    start_timeout = _role_service_manager_int(
        role, "START_TIMEOUT_SEC", SERVICE_MANAGER_START_TIMEOUT_SEC
    )
    poll_interval = _role_service_manager_float(
        role, "POLL_INTERVAL_SEC", SERVICE_MANAGER_POLL_INTERVAL_SEC
    )
    role_label = role if role != "default" else "service"

    if not health_url:
        return

    if await _is_service_manager_healthy(health_url):
        return

    wol_command = _format_wol_command()
    if wol_command and WOL_SSH_HOST:
        print(f"[INFO] Attempting Wake-on-LAN for {role_label} service_manager via OSMC...")
        await _run_ssh(WOL_SSH_HOST, WOL_SSH_USER, wol_command)
    else:
        print("[INFO] WOL not configured or WOL host missing; skipping wake")

    if start_cmd and ssh_host:
        print(f"[INFO] Attempting to start {role_label} service_manager...")
        await _run_ssh(
            ssh_host,
            ssh_user,
            start_cmd,
        )
    else:
        print(f"[INFO] {role_label} service_manager start command not configured")

    start_time = time.monotonic()
    while time.monotonic() - start_time < start_timeout:
        if await _is_service_manager_healthy(health_url):
            return
        await asyncio.sleep(poll_interval)

    raise HTTPException(
        status_code=503,
        detail=f"{role_label.capitalize()} service manager not ready. Please try again shortly.",
    )


async def ensure_ocr_service_ready() -> None:
    await ensure_service_manager_ready("ocr")


async def ensure_anonymization_service_ready() -> None:
    await ensure_service_manager_ready("anonymization")


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

    if force_refresh and document.gemini_file_uri:
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

    guessed_mime_type, _ = mimetypes.guess_type(selected_path)
    if guessed_mime_type and guessed_mime_type.startswith("image/"):
        mime_type = guessed_mime_type

    if document.gemini_file_uri and not force_refresh:
        try:
            existing_file = client.files.get(name=document.gemini_file_uri)
            existing_mime_type = getattr(existing_file, "mime_type", None)
            if existing_mime_type and mime_type and existing_mime_type != mime_type:
                print(
                    f"[INFO] Refreshing Gemini file for {filename} due to MIME mismatch "
                    f"({existing_mime_type} -> {mime_type})"
                )
                document.gemini_file_uri = None
            else:
                return existing_file
        except Exception as e:
            print(f"[WARN] URI {document.gemini_file_uri} expired/invalid: {e}")
            document.gemini_file_uri = None

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
    case_id: Optional[str] = None
    primary_bescheid: Optional[str] = None
    reference_document_id: Optional[str] = None
    selected_documents: Optional[SelectedDocuments] = None
    search_engine: Literal["gemini", "meta", "chatgpt-search", "grok-4.3"] = "meta"
    asylnet_keywords: Optional[str] = None
    search_mode: Literal["fast", "balanced", "deep"] = "balanced"
    max_sources: int = Field(default=12, ge=1, le=40)
    domain_policy: Literal["legal_strict", "legal_balanced", "broad"] = "legal_balanced"
    jurisdiction_focus: Literal["de", "de_eu", "eu", "global"] = "de_eu"
    recency_years: int = Field(default=6, ge=1, le=20)


PreferredSourceType = Literal[
    "court_decision",
    "higher_court_decision",
    "constitutional_court_decision",
    "supranational_court_decision",
    "administrative_decision",
    "official_country_report",
    "curated_legal_database_entry",
]


PreferredOutcome = Literal[
    "claimant_positive",
    "negative_if_authoritative",
    "mixed",
    "landmark_regardless_of_outcome",
]


class ResearchEvidenceItem(BaseModel):
    field: str = ""
    value: str = ""
    document: str = ""
    page_hint: str = ""
    quote: str = ""


class ResearchCountriesRelevant(BaseModel):
    origin_country: str = ""
    other_relevant_countries: List[str] = Field(default_factory=list)


class ResearchCaseFingerprint(BaseModel):
    procedure_type: str = ""
    countries_relevant: ResearchCountriesRelevant = Field(default_factory=ResearchCountriesRelevant)
    document_types_seen: List[str] = Field(default_factory=list)
    core_legal_questions: List[str] = Field(default_factory=list)
    core_fact_patterns: List[str] = Field(default_factory=list)
    relevant_actors: List[str] = Field(default_factory=list)
    relationship_patterns: List[str] = Field(default_factory=list)
    risk_mechanisms: List[str] = Field(default_factory=list)
    decision_match_requirements: List[str] = Field(default_factory=list)
    decision_mismatch_filters: List[str] = Field(default_factory=list)
    evidence: List[ResearchEvidenceItem] = Field(default_factory=list)
    confidence: float = 0.0


class ResearchSearchPlan(BaseModel):
    search_objective: str = ""
    search_queries: List[str] = Field(default_factory=list)
    must_cover: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    preferred_source_types: List[PreferredSourceType] = Field(default_factory=list)
    preferred_outcomes: List[PreferredOutcome] = Field(default_factory=list)
    preferred_recency_years: int = 0


class ResearchRankingProfile(BaseModel):
    primary_match_dimensions: List[str] = Field(default_factory=list)
    downgrade_if: List[str] = Field(default_factory=list)
    prefer_if: List[str] = Field(default_factory=list)


class ResearchCaseProfile(BaseModel):
    case_fingerprint: ResearchCaseFingerprint = Field(default_factory=ResearchCaseFingerprint)
    search_plan: ResearchSearchPlan = Field(default_factory=ResearchSearchPlan)
    ranking_profile: ResearchRankingProfile = Field(default_factory=ResearchRankingProfile)


class ResearchResult(BaseModel):
    query: str
    summary: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    discarded_sources: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    research_run_id: Optional[str] = None
    document_contexts: List[Dict[str, Any]] = Field(default_factory=list)
    seed_query: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    case_profile: Optional[Dict[str, Any]] = None


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
    case_id: Optional[str] = None
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
    case_id: Optional[str] = None
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
    extraction_prompt_tokens: Optional[int] = None
    extraction_completion_tokens: Optional[int] = None
    extraction_total_duration_ns: Optional[int] = None
    extraction_inference_params: Optional[Dict[str, Any]] = None


class BescheidSelection(BaseModel):
    primary: Optional[str] = None
    primary_id: Optional[str] = None
    others: List[str] = Field(default_factory=list)
    other_ids: List[str] = Field(default_factory=list)


class VorinstanzSelection(BaseModel):
    primary: Optional[str] = None
    primary_id: Optional[str] = None
    others: List[str] = Field(default_factory=list)
    other_ids: List[str] = Field(default_factory=list)


class SelectedDocuments(BaseModel):
    anhoerung: List[str] = Field(default_factory=list)
    anhoerung_ids: List[str] = Field(default_factory=list)
    bescheid: BescheidSelection
    vorinstanz: VorinstanzSelection
    rechtsprechung: List[str] = Field(default_factory=list)
    rechtsprechung_ids: List[str] = Field(default_factory=list)
    saved_sources: List[str] = Field(default_factory=list)
    sonstiges: List[str] = Field(default_factory=list)
    sonstiges_ids: List[str] = Field(default_factory=list)
    akte: List[str] = Field(default_factory=list)
    akte_ids: List[str] = Field(default_factory=list)


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
    resolved_legal_area: Optional[str] = None
    citations_found: int = 0
    missing_citations: List[str] = Field(default_factory=list)
    pinpoint_missing: List[str] = Field(default_factory=list)
    citation_checks: Dict[str, Any] = Field(default_factory=dict)
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


class GenerationJobResponse(BaseModel):
    id: str
    status: str
    case_id: Optional[str] = None
    draft_id: Optional[str] = None
    error_message: Optional[str] = None
    claimed_by: Optional[str] = None
    claimed_at: Optional[str] = None
    heartbeat_at: Optional[str] = None
    available_at: Optional[str] = None
    attempt_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_payload: Optional[Dict[str, Any]] = None


class QueryJobResponse(BaseModel):
    id: str
    status: str
    case_id: Optional[str] = None
    error_message: Optional[str] = None
    claimed_by: Optional[str] = None
    claimed_at: Optional[str] = None
    heartbeat_at: Optional[str] = None
    available_at: Optional[str] = None
    attempt_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_payload: Optional[Dict[str, Any]] = None


class ResearchJobResponse(BaseModel):
    id: str
    status: str
    case_id: Optional[str] = None
    research_run_id: Optional[str] = None
    error_message: Optional[str] = None
    claimed_by: Optional[str] = None
    claimed_at: Optional[str] = None
    heartbeat_at: Optional[str] = None
    available_at: Optional[str] = None
    attempt_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_payload: Optional[Dict[str, Any]] = None


class WorkflowJLawyerCaseResolveResponse(BaseModel):
    requested_reference: str
    resolved_case_id: str


class WorkflowInventoryResponse(BaseModel):
    case_id: Optional[str] = None
    documents: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    drafts: List[Dict[str, Any]] = Field(default_factory=list)


class JLawyerSendDraftRequest(BaseModel):
    draft_id: str
    case_reference: str
    template_name: str
    file_name: str
    template_folder: Optional[str] = None


class ApiTokenCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


class ApiTokenResponse(BaseModel):
    id: str
    name: str
    token_prefix: str
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    revoked_at: Optional[str] = None


class ApiTokenCreateResponse(ApiTokenResponse):
    token: str


class UploadDirectResponse(BaseModel):
    success: bool = True
    document_id: str
    original_filename: str
    filename: str
    category: str
    case_id: Optional[str] = None
    processing_status: Optional[str] = None
    needs_ocr: bool = False
    ocr_applied: bool = False
    message: str


class GenerationRequest(BaseModel):
    document_type: Literal[
        "Klagebegründung",
        "Antrag auf Zulassung der Berufung (AZB)",
        "Schriftsatz",
    ]
    user_prompt: str
    case_id: Optional[str] = None
    legal_area: Literal["migrationsrecht", "sozialrecht", "zivilrecht"] = "migrationsrecht"
    selected_documents: SelectedDocuments
    model: Literal[
        "claude-opus-4-7",
        "gpt-5.5",
        "gemini-3-pro-preview",
        "gemini-3.1-pro-preview",
        "two-step-expert",
        "multi-step-expert",
    ] = "claude-opus-4-7"
    verbosity: Literal["low", "medium", "high"] = "high"
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


class JLawyerSendRequest(BaseModel):
    case_id: str
    template_name: str
    file_name: str
    generated_text: str
    template_folder: Optional[str] = None


class JLawyerResponse(BaseModel):
    success: bool
    message: str
    requested_case_reference: Optional[str] = None
    resolved_case_id: Optional[str] = None
    template_folder: Optional[str] = None
    template_name: Optional[str] = None
    file_name: Optional[str] = None
    created_document_id: Optional[str] = None
    jlawyer_response: Optional[Dict[str, Any]] = None


class JLawyerTemplatesResponse(BaseModel):
    templates: List[str]
    folder: str


class RagHealthResponse(BaseModel):
    status: str
    qdrant_ok: bool
    desktop_embedder_ok: bool
    details: Optional[Dict[str, Any]] = None


class RagUpsertChunk(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)
    context_header: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance: List[str] = Field(default_factory=list)
    dense: List[float] = Field(default_factory=list)
    sparse: Optional[Dict[str, Any]] = None


class RagUpsertRequest(BaseModel):
    chunks: List[RagUpsertChunk] = Field(min_items=1, max_items=128)
    collection: str = "rag_chunks"


class RagUpsertResponse(BaseModel):
    upserted: int
    collection: str
    warnings: List[str] = Field(default_factory=list)


class RagFilters(BaseModel):
    section_type: Optional[List[str]] = None
    statute: Optional[str] = None
    paragraph: Optional[str] = None
    applicant_origin: Optional[str] = None
    court: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    citations: Optional[List[str]] = None


class RagRetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=12)
    dense_top_k: int = Field(default=50, ge=1, le=200)
    sparse_top_k: int = Field(default=50, ge=1, le=200)
    use_reranker: bool = False
    filters: Optional[RagFilters] = None


class RagRetrieveChunk(BaseModel):
    chunk_id: str
    score: float
    text: str
    context_header: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance: List[str] = Field(default_factory=list)


class RagRetrieveMetadata(BaseModel):
    fusion: str = "rrf"
    dense_top_k: int
    sparse_top_k: int
    limit: int
    reranker_applied: bool


class RagRetrieveResponse(BaseModel):
    query: str
    retrieval: RagRetrieveMetadata
    chunks: List[RagRetrieveChunk] = Field(default_factory=list)


MemoryTargetType = Literal["case_brief", "case_strategy"]
MemoryProposalStatus = Literal["pending", "accepted", "rejected", "superseded"]
MemorySourceType = Literal[
    "document",
    "draft",
    "chat",
    "research_run",
    "rechtsprechung_entry",
    "user_instruction",
]
MemoryPatchOpName = Literal["set", "append", "remove"]


class CaseBriefContent(BaseModel):
    beteiligte: List[Dict[str, Any]] = Field(default_factory=list)
    verfahrensstand: List[str] = Field(default_factory=list)
    sachverhalt: List[str] = Field(default_factory=list)
    antraege_ziele: List[str] = Field(default_factory=list)
    streitige_punkte: List[str] = Field(default_factory=list)
    beweismittel: List[str] = Field(default_factory=list)
    risiken: List[str] = Field(default_factory=list)
    offene_fragen: List[str] = Field(default_factory=list)
    notizen: str = ""


class CaseStrategyContent(BaseModel):
    kernstrategie: str = ""
    argumentationslinien: List[str] = Field(default_factory=list)
    rechtliche_ansatzpunkte: List[str] = Field(default_factory=list)
    beweisstrategie: List[str] = Field(default_factory=list)
    prozessuale_schritte: List[str] = Field(default_factory=list)
    vergleich_oder_taktik: List[str] = Field(default_factory=list)
    risiken_und_gegenargumente: List[str] = Field(default_factory=list)
    offene_fragen: List[str] = Field(default_factory=list)
    notizen: str = ""


class MemorySourceRef(BaseModel):
    source_type: MemorySourceType
    source_id: Optional[str] = None
    label: str = ""
    excerpt: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryPatchOperation(BaseModel):
    op: MemoryPatchOpName
    path: str
    value: Optional[Any] = None


class CaseMemoryBaseResponse(BaseModel):
    id: str
    owner_id: str
    case_id: str
    content_json: Dict[str, Any]
    search_text: str = ""
    version: int = 1
    last_reflected_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CaseBriefResponse(CaseMemoryBaseResponse):
    rendered: Optional[str] = None


class CaseStrategyResponse(CaseMemoryBaseResponse):
    rendered: Optional[str] = None


class CaseMemoryUpdateRequest(BaseModel):
    content_json: Dict[str, Any]
    expected_version: Optional[int] = None
    source_refs: List[MemorySourceRef] = Field(default_factory=list)
    actor: str = "user"


class CaseMemoryRenderResponse(BaseModel):
    target_type: MemoryTargetType
    target_id: str
    version: int
    rendered: str


class CaseMemoryRevisionResponse(BaseModel):
    id: str
    target_type: MemoryTargetType
    target_id: str
    previous_content_json: Dict[str, Any]
    new_content_json: Dict[str, Any]
    source_refs: List[Dict[str, Any]] = Field(default_factory=list)
    actor: str = ""
    created_at: Optional[str] = None


class CaseDocumentExtractionCreateRequest(BaseModel):
    document_id: str
    extraction_json: Dict[str, Any]
    source_refs: List[MemorySourceRef] = Field(default_factory=list)
    model: Optional[str] = None
    confidence: Optional[float] = None


class CaseDocumentExtractionResponse(BaseModel):
    id: str
    owner_id: str
    case_id: str
    document_id: str
    extraction_json: Dict[str, Any]
    source_refs: List[Dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[str] = None


class MemoryUpdateProposalCreateRequest(BaseModel):
    target_type: MemoryTargetType
    case_id: Optional[str] = None
    target_id: Optional[str] = None
    expected_version: int
    ops: List[MemoryPatchOperation] = Field(min_items=1)
    source_refs: List[MemorySourceRef] = Field(min_items=1)
    confidence: Optional[float] = None
    model: Optional[str] = None


class MemoryUpdateProposalReviewRequest(BaseModel):
    actor: str = "user"


class MemoryUpdateProposalResponse(BaseModel):
    id: str
    owner_id: str
    case_id: str
    target_type: MemoryTargetType
    target_id: str
    status: MemoryProposalStatus
    expected_version: Optional[int] = None
    ops: List[Dict[str, Any]] = Field(default_factory=list)
    source_refs: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None
    model: Optional[str] = None
    created_at: Optional[str] = None
    reviewed_at: Optional[str] = None


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
    provider = (os.environ.get("OPENAI_PROVIDER") or "openai").strip().lower()
    if provider in {"azure", "azure_openai", "azure-openai"}:
        api_key = (
            os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("AZURE_OPENAI_KEY1")
            or os.environ.get("AZURE_OPENAI_KEY")
        )
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")
        base_url = (
            os.environ.get("AZURE_OPENAI_BASE_URL")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
            or "https://rechtmaschine.openai.azure.com/openai/v1/"
        ).strip()
        if not base_url:
            raise ValueError("AZURE_OPENAI_BASE_URL environment variable not set")
        if not base_url.rstrip("/").endswith("/openai/v1"):
            base_url = base_url.rstrip("/") + "/openai/v1/"
        else:
            base_url = base_url.rstrip("/") + "/"
        return OpenAI(api_key=api_key, base_url=base_url)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def get_native_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def is_azure_openai_enabled() -> bool:
    provider = (os.environ.get("OPENAI_PROVIDER") or "openai").strip().lower()
    return provider in {"azure", "azure_openai", "azure-openai"}


def openai_file_uploads_enabled() -> bool:
    if not is_azure_openai_enabled():
        return True
    return (os.environ.get("AZURE_OPENAI_ENABLE_FILE_UPLOADS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_openai_model(model: str) -> str:
    """Resolve a public OpenAI model id to an Azure deployment name when needed."""
    model_name = (model or "").strip()
    if not model_name or not is_azure_openai_enabled():
        return model_name

    model_map_raw = (os.environ.get("AZURE_OPENAI_MODEL_MAP") or "").strip()
    if model_map_raw:
        try:
            model_map = json.loads(model_map_raw)
            if isinstance(model_map, dict):
                mapped = str(model_map.get(model_name) or "").strip()
                if mapped:
                    return mapped
        except Exception as exc:
            print(f"[WARN] Invalid AZURE_OPENAI_MODEL_MAP JSON: {exc}")

    env_key = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").upper()
    for key in (
        f"AZURE_OPENAI_DEPLOYMENT_{env_key}",
        f"AZURE_OPENAI_MODEL_{env_key}",
    ):
        mapped = (os.environ.get(key) or "").strip()
        if mapped:
            return mapped

    default_deployment = (os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "").strip()
    return default_deployment or model_name


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
    "GenerationJobResponse",
    "QueryJobResponse",
    "ResearchJobResponse",
    "ApiTokenCreateRequest",
    "ApiTokenResponse",
    "ApiTokenCreateResponse",
    "UploadDirectResponse",
    "GenerationMetadata",
    "JLawyerSendRequest",
    "JLawyerResponse",
    "JLawyerTemplatesResponse",
    "RagHealthResponse",
    "RagUpsertChunk",
    "RagUpsertRequest",
    "RagUpsertResponse",
    "RagFilters",
    "RagRetrieveRequest",
    "RagRetrieveChunk",
    "RagRetrieveMetadata",
    "RagRetrieveResponse",
    "get_openai_client",
    "get_native_openai_client",
    "is_azure_openai_enabled",
    "openai_file_uploads_enabled",
    "resolve_openai_model",
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

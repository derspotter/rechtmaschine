"""
Rechtmaschine - Document Classifier
Simplified document classification system for German asylum law documents
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response, Form
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from typing import Optional, List, Dict, Any
import tempfile
import pikepdf
import fitz  # PyMuPDF for text extraction
import markdown
import re
from openai import OpenAI
from google.genai import types
import httpx
import anthropic
import uuid
import unicodedata
from urllib.parse import quote_plus, urljoin, urlparse, quote
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session
from sqlalchemy import desc, text
from fastapi import Depends

# Database imports
from database import get_db, engine, Base, DATABASE_URL, SessionLocal
from models import Document, ResearchSource
from events import BroadcastHub, PostgresListener, DOCUMENTS_CHANNEL, SOURCES_CHANNEL
from endpoints import (
    classification as classification_endpoints,
    cases as cases_endpoints,
    documents as documents_endpoints,
    ocr as ocr_endpoints,
    research_sources as research_endpoints,
    generation as generation_endpoints,
    rechtsprechung_playbook as rechtsprechung_playbook_endpoints,
    root as root_endpoints,
    system as system_endpoints,
    anonymization as anonymization_endpoints,
    auth as auth_endpoints,
    agent_memory as agent_memory_endpoints,
    drafts as drafts_endpoints,
    query as query_endpoints,
    workflow as workflow_endpoints,
)
from shared import (
    limiter,
    DOWNLOADS_DIR,
    UPLOADS_DIR,
    OCR_TEXT_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    _ensure_directory,
    clear_directory_contents,
    DocumentCategory,
    ClassificationResult,
    GeminiClassification,
    ResearchRequest,
    ResearchResult,
    SavedSource,
    AddSourceRequest,
    AnonymizationRequest,
    AnonymizationResult,
    SelectedDocuments,
    BescheidSelection,
    GenerationRequest,
    GenerationResponse,
    GenerationMetadata,
    JLawyerSendRequest,
    JLawyerResponse,
    JLawyerTemplatesResponse,
    store_document_text,
    load_document_text,
    delete_document_text,
    get_openai_client,
    get_gemini_client,
    get_anthropic_client,
    get_xai_client,
    register_fastapi_app,
    build_documents_snapshot,
    build_sources_snapshot,
    broadcast_documents_snapshot,
    broadcast_sources_snapshot,
)
from endpoints.segmentation import (
    _infer_document_type_from_title,
    _infer_hearing_subtype_from_title,
    collect_outline_items,
)

ENABLE_API_DOCS = os.getenv("ENABLE_API_DOCS", "false").strip().lower() in {"1", "true", "yes"}
TRUSTED_HOSTS = [
    host.strip()
    for host in os.getenv(
        "TRUSTED_HOSTS",
        "rechtmaschine.de,www.rechtmaschine.de,localhost,127.0.0.1,rechtmaschine-app",
    ).split(",")
    if host.strip()
]

app = FastAPI(
    title="Rechtmaschine Document Classifier",
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
)
register_fastapi_app(app)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.source_subscribers = set()
app.state.document_hub = None
app.state.document_listener = None

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(auth_endpoints.router)
app.include_router(agent_memory_endpoints.router)
app.include_router(cases_endpoints.router)
app.include_router(classification_endpoints.router)
app.include_router(documents_endpoints.router)
app.include_router(ocr_endpoints.router)
app.include_router(anonymization_endpoints.router)
app.include_router(research_endpoints.router)
app.include_router(generation_endpoints.router)
app.include_router(rechtsprechung_playbook_endpoints.router)
app.include_router(drafts_endpoints.router)
app.include_router(root_endpoints.router)
app.include_router(query_endpoints.router)
app.include_router(workflow_endpoints.router)
app.include_router(system_endpoints.router)


BOOTSTRAP_ADMIN_EMAIL = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "der_spotter").strip() or "der_spotter"
BOOTSTRAP_ADMIN_PASSWORD_HASH = (
    os.getenv(
        "BOOTSTRAP_ADMIN_PASSWORD_HASH",
        "$2b$12$xgxihm..lboIYMthS68onuPNTmPr8batS8JhCSDMcmr3mXAH20jGG"
    ).strip()
)


def _sql_literal(value: str) -> str:
    """Escape single quotes for embedding values into SQL string literals."""
    return str(value).replace("'", "''")


MIGRATIONS: List[tuple[str, List[str]]] = [
    (
        "2025-11-04_drop_extracted_text",
        [
            """
            ALTER TABLE documents
                DROP COLUMN IF EXISTS extracted_text,
                ADD COLUMN IF NOT EXISTS extracted_text_path VARCHAR(512),
                ADD COLUMN IF NOT EXISTS is_anonymized BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS ocr_applied BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS needs_ocr BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS anonymization_metadata JSONB,
                ADD COLUMN IF NOT EXISTS processing_status VARCHAR(20) DEFAULT 'pending'
            """,
            """
            DO $$
            BEGIN
                IF to_regclass('public.processed_documents') IS NOT NULL THEN
                    UPDATE documents d
                    SET
                        is_anonymized = COALESCE(p.is_anonymized, FALSE),
                        ocr_applied = COALESCE(p.ocr_applied, FALSE),
                        needs_ocr = COALESCE(p.needs_ocr, FALSE),
                        anonymization_metadata = p.anonymization_metadata,
                        processing_status = COALESCE(p.processing_status, 'pending')
                    FROM (
                        SELECT DISTINCT ON (document_id)
                            document_id,
                            is_anonymized,
                            ocr_applied,
                            needs_ocr,
                            anonymization_metadata,
                            processing_status
                        FROM processed_documents
                        ORDER BY document_id, created_at DESC
                    ) p
                    WHERE d.id = p.document_id;

                    DROP TABLE IF EXISTS processed_documents CASCADE;
                END IF;
            END$$;
            """,
            """
            CREATE INDEX IF NOT EXISTS ix_documents_needs_ocr
            ON documents(needs_ocr) WHERE needs_ocr = TRUE
            """,
            """
            CREATE INDEX IF NOT EXISTS ix_documents_is_anonymized
            ON documents(is_anonymized) WHERE is_anonymized = TRUE
            """,
        ],
    ),
    (
        "2025-12-03_add_users_and_owner_id",
        [
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            INSERT INTO users (id, email, hashed_password, is_active, created_at)
            VALUES (
                '00000000-0000-0000-0000-000000000000',
                '{admin_email}',
                '{password_hash}',
                TRUE,
                NOW()
            )
            ON CONFLICT (email) DO NOTHING
            """.format(admin_email=_sql_literal(BOOTSTRAP_ADMIN_EMAIL), password_hash=_sql_literal(BOOTSTRAP_ADMIN_PASSWORD_HASH)),
            """
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS owner_id UUID
            """,
            """
            UPDATE documents
            SET owner_id = '00000000-0000-0000-0000-000000000000'
            WHERE owner_id IS NULL
            """,
            """
            ALTER TABLE research_sources
            ADD COLUMN IF NOT EXISTS owner_id UUID
            """,
            """
            UPDATE research_sources
            SET owner_id = '00000000-0000-0000-0000-000000000000'
            WHERE owner_id IS NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS ix_documents_owner_id ON documents(owner_id);
            CREATE INDEX IF NOT EXISTS ix_research_sources_owner_id ON research_sources(owner_id);
            """
        ]
    ),
    (
        "2026-01-28_rechtsprechung_entries",
        [
            """
            CREATE TABLE IF NOT EXISTS rechtsprechung_entries (
                id UUID PRIMARY KEY,
                document_id UUID NULL REFERENCES documents(id) ON DELETE SET NULL,
                country VARCHAR(100) NOT NULL,
                tags JSONB,
                court VARCHAR(255),
                court_level VARCHAR(50),
                decision_date DATE,
                aktenzeichen VARCHAR(100),
                outcome VARCHAR(50),
                key_facts JSONB,
                key_holdings JSONB,
                argument_patterns JSONB,
                citations JSONB,
                summary TEXT,
                extracted_at TIMESTAMP,
                model VARCHAR(50),
                confidence FLOAT,
                warnings JSONB,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS rechtsprechung_entries_country_idx ON rechtsprechung_entries (country)",
            "CREATE INDEX IF NOT EXISTS rechtsprechung_entries_decision_date_idx ON rechtsprechung_entries (decision_date)"
        ]
    ),
    (
        "2026-02-06_cases",
        [
            """
            CREATE TABLE IF NOT EXISTS cases (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                name TEXT,
                state JSONB,
                archived BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_cases_owner_id ON cases(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_cases_updated_at ON cases(updated_at)",
            """
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS active_case_id UUID
            """,
            """
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS case_id UUID
            """,
            "CREATE INDEX IF NOT EXISTS ix_documents_case_id ON documents(case_id)",
            # Allow the same filename to exist in different cases (or users).
            "ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_filename_key",
            "DROP INDEX IF EXISTS ix_documents_filename",
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_owner_case_filename ON documents(owner_id, case_id, filename)",
            """
            ALTER TABLE research_sources
            ADD COLUMN IF NOT EXISTS case_id UUID
            """,
            "CREATE INDEX IF NOT EXISTS ix_research_sources_case_id ON research_sources(case_id)",
            """
            ALTER TABLE generated_drafts
            ADD COLUMN IF NOT EXISTS case_id UUID
            """,
            "CREATE INDEX IF NOT EXISTS ix_generated_drafts_case_id ON generated_drafts(case_id)",
        ],
    ),
    (
        "2026-02-06_documents_filename_per_case",
        [
            # Old schema enforced filename uniqueness globally, which breaks multi-case workflows.
            "ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_filename_key",
            "DROP INDEX IF EXISTS ix_documents_filename",
            "CREATE INDEX IF NOT EXISTS ix_documents_filename ON documents(filename)",
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM documents
                    WHERE owner_id IS NOT NULL AND case_id IS NOT NULL
                    GROUP BY owner_id, case_id, filename
                    HAVING COUNT(*) > 1
                ) THEN
                    RAISE NOTICE 'Skipping ux_documents_owner_case_filename (duplicates exist)';
                ELSE
                    EXECUTE 'CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_owner_case_filename ON documents(owner_id, case_id, filename)';
                END IF;
            END$$;
            """,
        ],
    ),
    (
        "2026-02-17_research_runs",
        [
            """
            CREATE TABLE IF NOT EXISTS research_runs (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID,
                user_query TEXT,
                generated_query BOOLEAN DEFAULT FALSE,
                effective_query TEXT NOT NULL,
                search_engine VARCHAR(50) NOT NULL,
                search_mode VARCHAR(20) DEFAULT 'balanced',
                max_sources INTEGER DEFAULT 12,
                domain_policy VARCHAR(20) DEFAULT 'legal_balanced',
                jurisdiction_focus VARCHAR(20) DEFAULT 'de_eu',
                recency_years INTEGER DEFAULT 6,
                selected_document_ids JSONB,
                request_payload JSONB,
                response_payload JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_research_runs_owner_id ON research_runs(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_research_runs_case_id ON research_runs(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_research_runs_created_at ON research_runs(created_at)",
        ],
    ),
    (
        "2026-03-24_generation_jobs",
        [
            """
            CREATE TABLE IF NOT EXISTS generation_jobs (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID,
                status VARCHAR(20) NOT NULL DEFAULT 'queued',
                request_payload JSONB,
                result_payload JSONB,
                error_message TEXT,
                draft_id UUID,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_generation_jobs_owner_id ON generation_jobs(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_generation_jobs_case_id ON generation_jobs(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_generation_jobs_status ON generation_jobs(status)",
            "CREATE INDEX IF NOT EXISTS ix_generation_jobs_created_at ON generation_jobs(created_at)",
        ],
    ),
    (
        "2026-03-24_query_jobs",
        [
            """
            CREATE TABLE IF NOT EXISTS query_jobs (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID,
                status VARCHAR(20) NOT NULL DEFAULT 'queued',
                request_payload JSONB,
                result_payload JSONB,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_query_jobs_owner_id ON query_jobs(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_query_jobs_case_id ON query_jobs(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_query_jobs_status ON query_jobs(status)",
            "CREATE INDEX IF NOT EXISTS ix_query_jobs_created_at ON query_jobs(created_at)",
        ],
    ),
    (
        "2026-03-26_job_worker_state",
        [
            """
            ALTER TABLE generation_jobs
            ADD COLUMN IF NOT EXISTS claimed_by VARCHAR(128),
            ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMP NULL,
            ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMP NULL,
            ADD COLUMN IF NOT EXISTS available_at TIMESTAMP DEFAULT NOW(),
            ADD COLUMN IF NOT EXISTS attempt_count INTEGER DEFAULT 0
            """,
            """
            ALTER TABLE query_jobs
            ADD COLUMN IF NOT EXISTS claimed_by VARCHAR(128),
            ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMP NULL,
            ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMP NULL,
            ADD COLUMN IF NOT EXISTS available_at TIMESTAMP DEFAULT NOW(),
            ADD COLUMN IF NOT EXISTS attempt_count INTEGER DEFAULT 0
            """,
            """
            ALTER TABLE research_jobs
            ADD COLUMN IF NOT EXISTS claimed_by VARCHAR(128),
            ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMP NULL,
            ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMP NULL,
            ADD COLUMN IF NOT EXISTS available_at TIMESTAMP DEFAULT NOW(),
            ADD COLUMN IF NOT EXISTS attempt_count INTEGER DEFAULT 0
            """,
            "UPDATE generation_jobs SET available_at = COALESCE(available_at, created_at, NOW())",
            "UPDATE query_jobs SET available_at = COALESCE(available_at, created_at, NOW())",
            "UPDATE research_jobs SET available_at = COALESCE(available_at, created_at, NOW())",
            "CREATE INDEX IF NOT EXISTS ix_generation_jobs_available_at ON generation_jobs(available_at)",
            "CREATE INDEX IF NOT EXISTS ix_generation_jobs_claimed_by ON generation_jobs(claimed_by)",
            "CREATE INDEX IF NOT EXISTS ix_generation_jobs_heartbeat_at ON generation_jobs(heartbeat_at)",
            "CREATE INDEX IF NOT EXISTS ix_query_jobs_available_at ON query_jobs(available_at)",
            "CREATE INDEX IF NOT EXISTS ix_query_jobs_claimed_by ON query_jobs(claimed_by)",
            "CREATE INDEX IF NOT EXISTS ix_query_jobs_heartbeat_at ON query_jobs(heartbeat_at)",
            "CREATE INDEX IF NOT EXISTS ix_research_jobs_available_at ON research_jobs(available_at)",
            "CREATE INDEX IF NOT EXISTS ix_research_jobs_claimed_by ON research_jobs(claimed_by)",
            "CREATE INDEX IF NOT EXISTS ix_research_jobs_heartbeat_at ON research_jobs(heartbeat_at)",
        ],
    ),
    (
        "2026-03-26_research_jobs",
        [
            """
            CREATE TABLE IF NOT EXISTS research_jobs (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'queued',
                request_payload JSONB DEFAULT '{}'::jsonb,
                result_payload JSONB DEFAULT '{}'::jsonb,
                error_message TEXT NULL,
                research_run_id UUID NULL REFERENCES research_runs(id) ON DELETE SET NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                started_at TIMESTAMP NULL,
                completed_at TIMESTAMP NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_research_jobs_owner_id ON research_jobs(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_research_jobs_case_id ON research_jobs(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_research_jobs_status ON research_jobs(status)",
            "CREATE INDEX IF NOT EXISTS ix_research_jobs_created_at ON research_jobs(created_at)",
            "CREATE INDEX IF NOT EXISTS ix_research_jobs_research_run_id ON research_jobs(research_run_id)",
        ],
    ),
    (
        "2026-03-24_api_tokens",
        [
            """
            CREATE TABLE IF NOT EXISTS api_tokens (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                name VARCHAR(255) NOT NULL,
                token_prefix VARCHAR(32) NOT NULL,
                token_hash VARCHAR(128) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                last_used_at TIMESTAMP,
                expires_at TIMESTAMP,
                revoked_at TIMESTAMP
            )
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_api_tokens_token_hash ON api_tokens(token_hash)",
            "CREATE INDEX IF NOT EXISTS ix_api_tokens_owner_id ON api_tokens(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_api_tokens_token_prefix ON api_tokens(token_prefix)",
            "CREATE INDEX IF NOT EXISTS ix_api_tokens_created_at ON api_tokens(created_at)",
        ],
    ),
    (
        "2026-03-31_document_outline_metadata",
        [
            """
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS outline_title VARCHAR(512),
            ADD COLUMN IF NOT EXISTS hearing_subtype VARCHAR(50)
            """,
            "CREATE INDEX IF NOT EXISTS ix_documents_hearing_subtype ON documents(hearing_subtype)",
        ],
    ),
    (
        "2026-04-28_case_memory_mvp",
        [
            """
            CREATE TABLE IF NOT EXISTS case_briefs (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID NOT NULL,
                content_json JSONB,
                search_text TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                last_reflected_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_case_briefs_owner_case ON case_briefs(owner_id, case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_briefs_owner_id ON case_briefs(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_briefs_case_id ON case_briefs(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_briefs_updated_at ON case_briefs(updated_at)",
            """
            CREATE TABLE IF NOT EXISTS case_strategies (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID NOT NULL,
                content_json JSONB,
                search_text TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                last_reflected_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_case_strategies_owner_case ON case_strategies(owner_id, case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_strategies_owner_id ON case_strategies(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_strategies_case_id ON case_strategies(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_strategies_updated_at ON case_strategies(updated_at)",
            """
            CREATE TABLE IF NOT EXISTS case_brief_sources (
                id UUID PRIMARY KEY,
                case_brief_id UUID NOT NULL REFERENCES case_briefs(id) ON DELETE CASCADE,
                owner_id UUID NOT NULL,
                case_id UUID NOT NULL,
                source_type VARCHAR(32) NOT NULL,
                source_id VARCHAR(128),
                label TEXT,
                excerpt TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_case_brief_sources_case_brief_id ON case_brief_sources(case_brief_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_brief_sources_owner_id ON case_brief_sources(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_brief_sources_case_id ON case_brief_sources(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_brief_sources_source_type ON case_brief_sources(source_type)",
            "CREATE INDEX IF NOT EXISTS ix_case_brief_sources_source_id ON case_brief_sources(source_id)",
            """
            CREATE TABLE IF NOT EXISTS case_strategy_sources (
                id UUID PRIMARY KEY,
                case_strategy_id UUID NOT NULL REFERENCES case_strategies(id) ON DELETE CASCADE,
                owner_id UUID NOT NULL,
                case_id UUID NOT NULL,
                source_type VARCHAR(32) NOT NULL,
                source_id VARCHAR(128),
                label TEXT,
                excerpt TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_case_strategy_sources_case_strategy_id ON case_strategy_sources(case_strategy_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_strategy_sources_owner_id ON case_strategy_sources(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_strategy_sources_case_id ON case_strategy_sources(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_strategy_sources_source_type ON case_strategy_sources(source_type)",
            "CREATE INDEX IF NOT EXISTS ix_case_strategy_sources_source_id ON case_strategy_sources(source_id)",
            """
            CREATE TABLE IF NOT EXISTS case_memory_revisions (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID NOT NULL,
                target_type VARCHAR(50) NOT NULL,
                target_id UUID NOT NULL,
                previous_content_json JSONB,
                new_content_json JSONB,
                source_refs JSONB,
                actor VARCHAR(128),
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_case_memory_revisions_owner_id ON case_memory_revisions(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_memory_revisions_case_id ON case_memory_revisions(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_memory_revisions_target_type ON case_memory_revisions(target_type)",
            "CREATE INDEX IF NOT EXISTS ix_case_memory_revisions_target_id ON case_memory_revisions(target_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_memory_revisions_created_at ON case_memory_revisions(created_at)",
            """
            CREATE TABLE IF NOT EXISTS case_document_extractions (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID NOT NULL,
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                extraction_json JSONB,
                source_refs JSONB,
                model VARCHAR(50),
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_case_document_extractions_owner_id ON case_document_extractions(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_document_extractions_case_id ON case_document_extractions(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_document_extractions_document_id ON case_document_extractions(document_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_document_extractions_created_at ON case_document_extractions(created_at)",
            """
            CREATE TABLE IF NOT EXISTS memory_update_proposals (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID NOT NULL,
                target_type VARCHAR(50) NOT NULL,
                target_id UUID NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                expected_version INTEGER,
                ops JSONB,
                source_refs JSONB,
                model VARCHAR(50),
                confidence FLOAT,
                reviewed_by VARCHAR(128),
                reviewed_at TIMESTAMP,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_memory_update_proposals_owner_id ON memory_update_proposals(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_memory_update_proposals_case_id ON memory_update_proposals(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_memory_update_proposals_target_type ON memory_update_proposals(target_type)",
            "CREATE INDEX IF NOT EXISTS ix_memory_update_proposals_target_id ON memory_update_proposals(target_id)",
            "CREATE INDEX IF NOT EXISTS ix_memory_update_proposals_status ON memory_update_proposals(status)",
            "CREATE INDEX IF NOT EXISTS ix_memory_update_proposals_created_at ON memory_update_proposals(created_at)",
        ],
    ),
]


_SEGMENT_EXPLANATION_PATTERN = re.compile(
    r"Segment \((?P<doc_type>[^)]+)\) aus Akte (?P<source_filename>.+?), Seiten (?P<start>\d+)-(?P<end>\d+)"
)


def backfill_segment_outline_metadata() -> None:
    """Populate outline metadata for existing Akte segments where it is still missing."""

    updated = 0
    with SessionLocal() as db:
        segments = (
            db.query(Document)
            .filter(
                Document.outline_title.is_(None),
                Document.explanation.is_not(None),
            )
            .filter(
                Document.explanation.like("Segment (% aus Akte %, Seiten %-%")
            )
            .all()
        )

        outline_cache: Dict[tuple[str, Optional[uuid.UUID], Optional[uuid.UUID]], List[Dict[str, Any]]] = {}

        for segment in segments:
            explanation = (segment.explanation or "").strip()
            match = _SEGMENT_EXPLANATION_PATTERN.search(explanation)
            if not match:
                continue

            source_filename = match.group("source_filename").strip()
            start_page = int(match.group("start"))
            end_page = int(match.group("end"))
            cache_key = (source_filename, segment.owner_id, segment.case_id)

            items = outline_cache.get(cache_key)
            if items is None:
                source_doc = (
                    db.query(Document)
                    .filter(
                        Document.filename == source_filename,
                        Document.owner_id == segment.owner_id,
                        Document.case_id == segment.case_id,
                    )
                    .first()
                )
                if not source_doc or not source_doc.file_path:
                    outline_cache[cache_key] = []
                    continue
                source_path = Path(source_doc.file_path)
                if not source_path.exists():
                    outline_cache[cache_key] = []
                    continue
                try:
                    with pikepdf.Pdf.open(str(source_path)) as pdf_doc:
                        items = collect_outline_items(pdf_doc)
                except Exception as exc:
                    print(f"[BACKFILL] Outline read failed for {source_filename}: {exc}")
                    items = []
                outline_cache[cache_key] = items

            if not items:
                continue

            matching_item = next(
                (
                    item
                    for item in items
                    if item.get("start", -1) + 1 == start_page and item.get("end", -1) + 1 == end_page
                ),
                None,
            )
            if not matching_item:
                continue

            outline_title = matching_item.get("title") or None
            if not outline_title:
                continue

            inferred_type = _infer_document_type_from_title(outline_title)
            if inferred_type != segment.category:
                continue

            segment.outline_title = outline_title
            if segment.category == "Anhörung":
                segment.hearing_subtype = _infer_hearing_subtype_from_title(outline_title)
            updated += 1

        if updated:
            db.commit()
            print(f"[BACKFILL] Updated outline metadata for {updated} segmented documents")
        else:
            db.rollback()


def apply_schema_migrations() -> None:
    """Apply idempotent SQL migrations at startup."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    executed_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
        )
        applied = {
            row[0]
            for row in conn.execute(text("SELECT name FROM schema_migrations"))
        }

    for name, statements in MIGRATIONS:
        if name in applied:
            continue
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
            conn.execute(
                text("INSERT INTO schema_migrations (name) VALUES (:name)"),
                {"name": name},
            )
        print(f"[MIGRATION] Applied {name}")


# j-lawyer configuration
# Initialize database tables on startup
@app.on_event("startup")
async def startup_event():
    """Create database tables on startup"""
    Base.metadata.create_all(bind=engine)
    apply_schema_migrations()
    backfill_segment_outline_metadata()
    loop = asyncio.get_running_loop()

    # Create unified broadcast hub for all updates
    hub = BroadcastHub(loop)
    app.state.document_hub = hub

    # Start PostgreSQL listeners for both channels
    docs_listener = PostgresListener(DATABASE_URL, hub, DOCUMENTS_CHANNEL)
    docs_listener.start()
    app.state.document_listener = docs_listener

    sources_listener = PostgresListener(DATABASE_URL, hub, SOURCES_CHANNEL)
    sources_listener.start()
    app.state.sources_listener = sources_listener

    print("Database tables created successfully")
    print(f"Listening on PostgreSQL channels: {DOCUMENTS_CHANNEL}, {SOURCES_CHANNEL}")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop all PostgreSQL listeners on shutdown"""
    docs_listener: Optional[PostgresListener] = getattr(app.state, "document_listener", None)
    if docs_listener:
        docs_listener.stop()

    sources_listener: Optional[PostgresListener] = getattr(app.state, "sources_listener", None)
    if sources_listener:
        sources_listener.stop()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
        reload_dirs=["/app"],
        reload_includes=["*.py"],
        reload_excludes=[
            "*.json",
            "*.pdf",
            "*.log",
            "*/__pycache__/*",
            "*/.pytest_cache/*",
            "*/uploads/*",
            "*/downloaded_sources/*",
            "*/tmp/*",
            "*/anonymized_text/*",
            "*/ocr_text/*",
            "*/anon/venv/*",
        ],
    )

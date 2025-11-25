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
from database import get_db, engine, Base, DATABASE_URL
from models import Document, ResearchSource
from events import BroadcastHub, PostgresListener, DOCUMENTS_CHANNEL, SOURCES_CHANNEL
from endpoints import (
    classification as classification_endpoints,
    documents as documents_endpoints,
    ocr as ocr_endpoints,
    research_sources as research_endpoints,
    generation as generation_endpoints,
    root as root_endpoints,
    system as system_endpoints,
    anonymization as anonymization_endpoints,
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

app = FastAPI(title="Rechtmaschine Document Classifier")
register_fastapi_app(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.source_subscribers = set()
app.state.document_hub = None
app.state.document_listener = None

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(classification_endpoints.router)
app.include_router(documents_endpoints.router)
app.include_router(ocr_endpoints.router)
app.include_router(anonymization_endpoints.router)
app.include_router(research_endpoints.router)
app.include_router(generation_endpoints.router)
app.include_router(root_endpoints.router)
app.include_router(system_endpoints.router)

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
]


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
        reload_includes=["*.py"],
        reload_excludes=["*.json", "*.pdf", "*.log"],
    )

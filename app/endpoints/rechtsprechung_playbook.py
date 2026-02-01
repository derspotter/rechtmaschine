"""Endpoints for the Aktuelle Rechtsprechung playbook."""

import re
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import get_db
from models import Document, RechtsprechungEntry, User
from shared import (
    RechtsprechungEntryCreate,
    RechtsprechungEntryResponse,
    RechtsprechungEntryUpdate,
    get_document_for_upload,
    get_gemini_client,
    limiter,
)

from google.genai import types
from pydantic import BaseModel

router = APIRouter(tags=["rechtsprechung-playbook"])


class ArgumentPattern(BaseModel):
    use_when: Optional[str] = None
    rebuttal: Optional[str] = None
    notes: Optional[str] = None


class CitationRef(BaseModel):
    court: Optional[str] = None
    date: Optional[str] = None
    az: Optional[str] = None


class RechtsprechungExtraction(BaseModel):
    country: str
    tags: List[str] = []
    court: Optional[str] = None
    court_level: Optional[str] = None
    decision_date: Optional[str] = None
    aktenzeichen: Optional[str] = None
    outcome: Optional[str] = None
    key_facts: List[str] = []
    key_holdings: List[str] = []
    argument_patterns: List[ArgumentPattern] = []
    citations: List[CitationRef] = []
    summary: Optional[str] = None
    confidence: Optional[float] = None
    warnings: List[str] = []


def _normalize_tags(tags: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for tag in tags or []:
        value = re.sub(r"\s+", " ", str(tag).strip().lower())
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _extract_playbook_entry(document: Document) -> RechtsprechungExtraction:
    client = get_gemini_client()

    prompt = (
        "Analysiere dieses deutsche Gerichtsurteil oder diesen Beschluss (Rechtsprechung) "
        "und extrahiere die folgenden Informationen. Antworte ausschließlich im JSON-Format "
        "gemäß Schema.\n\n"
        "Felder:\n"
        "- country: Herkunftsland des Antragstellers (z. B. Iran, Afghanistan)\n"
        "- tags: Stichwörter wie frau/mann, politische Verfolgung, Religion/Konversion, "
        "Wehrdienst, Nachfluchtgründe, Hazara, LGBTQ, etc.\n"
        "- court: Gericht\n"
        "- court_level: VG/OVG/BVerwG/BVerfG/EGMR\n"
        "- decision_date: Datum (YYYY-MM-DD wenn möglich)\n"
        "- aktenzeichen\n"
        "- outcome: grant|partial|deny|remand (falls unklar: unknown)\n"
        "- key_facts: 3-7 knappe Stichpunkte\n"
        "- key_holdings: 3-7 knappe Stichpunkte\n"
        "- argument_patterns: Liste von Objekten mit use_when/rebuttal/notes (kurz)\n"
        "- citations: Liste von zitierten Entscheidungen (court/date/az)\n"
        "- summary: kurze Zusammenfassung (2-4 Sätze)\n"
        "- confidence: 0-1\n"
        "- warnings: Liste optionaler Warnungen\n"
    )

    entry = {
        "filename": document.filename,
        "file_path": document.file_path,
        "extracted_text_path": document.extracted_text_path,
        "anonymization_metadata": document.anonymization_metadata,
        "is_anonymized": document.is_anonymized,
    }
    try:
        file_path, mime_type, _ = get_document_for_upload(entry)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[WARN] Playbook extraction file lookup failed for {document.filename}: {exc}")
        raise HTTPException(status_code=400, detail="Dokumentdatei nicht verfügbar")

    try:
        if mime_type == "text/plain":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            snippet = content[:30000]
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[prompt, snippet],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=RechtsprechungExtraction,
                ),
            )
        else:
            with open(file_path, "rb") as f:
                uploaded = client.files.upload(
                    file=f,
                    config={
                        "mime_type": "application/pdf",
                        "display_name": document.filename,
                    },
                )
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[prompt, uploaded],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=RechtsprechungExtraction,
                ),
            )
    except Exception as exc:
        print(f"[WARN] Playbook extraction failed for {document.filename}: {exc}")
        raise HTTPException(status_code=502, detail="Extraktion fehlgeschlagen")

    return response.parsed


@router.get("/rechtsprechung/playbook")
@limiter.limit("60/hour")
async def list_playbook_entries(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    country: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    active: Optional[bool] = Query(True),
):
    """List playbook entries with optional country/tag filters."""
    query = db.query(RechtsprechungEntry)
    if active is not None:
        query = query.filter(RechtsprechungEntry.is_active == active)
    if country:
        query = query.filter(func.lower(RechtsprechungEntry.country) == country.strip().lower())
    if tag:
        query = query.filter(RechtsprechungEntry.tags.contains([tag.strip().lower()]))

    entries = [entry.to_dict() for entry in query.order_by(RechtsprechungEntry.decision_date.desc().nullslast()).all()]
    countries = [
        row[0] for row in db.query(RechtsprechungEntry.country)
        .filter(RechtsprechungEntry.is_active == True)  # noqa: E712
        .distinct()
        .order_by(RechtsprechungEntry.country.asc())
        .all()
    ]

    return {"entries": entries, "countries": countries}


@router.post("/rechtsprechung/playbook/from-document", response_model=RechtsprechungEntryResponse)
@limiter.limit("20/hour")
async def create_playbook_entry(
    request: Request,
    body: RechtsprechungEntryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create or update a playbook entry from a Rechtsprechung document."""
    try:
        doc_uuid = uuid.UUID(body.document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.category != "Rechtsprechung":
        raise HTTPException(status_code=400, detail="Dokument ist keine Rechtsprechung")

    extracted = _extract_playbook_entry(document)
    decision_date = _parse_date(extracted.decision_date)
    tags = _normalize_tags(extracted.tags)
    country = (extracted.country or "").strip()
    if not country:
        country = "Unbekannt"
    # Always include country as a tag
    country_tag = country.strip().lower()
    if country_tag and country_tag not in tags:
        tags.append(country_tag)

    existing = None
    if extracted.court and decision_date and extracted.aktenzeichen:
        existing = (
            db.query(RechtsprechungEntry)
            .filter(func.lower(RechtsprechungEntry.court) == extracted.court.strip().lower())
            .filter(RechtsprechungEntry.decision_date == decision_date)
            .filter(func.lower(RechtsprechungEntry.aktenzeichen) == extracted.aktenzeichen.strip().lower())
            .first()
        )

    entry = existing or RechtsprechungEntry(id=uuid.uuid4())
    entry.document_id = document.id
    entry.country = country
    entry.tags = tags
    entry.court = extracted.court
    entry.court_level = extracted.court_level
    entry.decision_date = decision_date
    entry.aktenzeichen = extracted.aktenzeichen
    entry.outcome = extracted.outcome or "unknown"
    entry.key_facts = extracted.key_facts or []
    entry.key_holdings = extracted.key_holdings or []
    entry.argument_patterns = [
        item.model_dump() for item in (extracted.argument_patterns or [])
    ]
    entry.citations = [
        item.model_dump() for item in (extracted.citations or [])
    ]
    entry.summary = extracted.summary
    entry.extracted_at = datetime.utcnow()
    entry.model = "gemini-3-flash-preview"
    entry.confidence = extracted.confidence
    entry.warnings = extracted.warnings or []
    entry.is_active = True
    entry.updated_at = datetime.utcnow()
    if not existing:
        entry.created_at = datetime.utcnow()
        db.add(entry)

    db.commit()
    db.refresh(entry)
    return entry.to_dict()


@router.patch("/rechtsprechung/playbook/{entry_id}", response_model=RechtsprechungEntryResponse)
@limiter.limit("60/hour")
async def update_playbook_entry(
    request: Request,
    entry_id: str,
    body: RechtsprechungEntryUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update tags or metadata for a playbook entry."""
    try:
        entry_uuid = uuid.UUID(entry_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid entry ID")

    entry = db.query(RechtsprechungEntry).filter(RechtsprechungEntry.id == entry_uuid).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Eintrag nicht gefunden")

    if body.country is not None:
        entry.country = body.country.strip() or entry.country
    if body.tags is not None:
        entry.tags = _normalize_tags(body.tags)
    if body.court is not None:
        entry.court = body.court
    if body.court_level is not None:
        entry.court_level = body.court_level
    if body.decision_date is not None:
        entry.decision_date = _parse_date(body.decision_date)
    if body.aktenzeichen is not None:
        entry.aktenzeichen = body.aktenzeichen
    if body.outcome is not None:
        entry.outcome = body.outcome
    if body.key_facts is not None:
        entry.key_facts = body.key_facts
    if body.key_holdings is not None:
        entry.key_holdings = body.key_holdings
    if body.argument_patterns is not None:
        entry.argument_patterns = body.argument_patterns
    if body.citations is not None:
        entry.citations = body.citations
    if body.summary is not None:
        entry.summary = body.summary
    if body.confidence is not None:
        entry.confidence = body.confidence
    if body.warnings is not None:
        entry.warnings = body.warnings
    if body.is_active is not None:
        entry.is_active = body.is_active

    entry.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(entry)
    return entry.to_dict()

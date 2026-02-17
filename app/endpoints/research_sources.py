"""
Research sources endpoint - refactored and modular.

Routes research requests to appropriate providers and manages saved sources.
"""

import asyncio
import hashlib
import re
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from playwright.async_api import async_playwright
from sqlalchemy import desc
from sqlalchemy.orm import Session

from shared import (
    AddSourceRequest,
    ResearchRequest,
    ResearchResult,
    SavedSource,
    DOWNLOADS_DIR,
    load_document_text,
    DocumentCategory,
    broadcast_sources_snapshot,
    get_document_for_upload,
    limiter,
)
from auth import get_current_active_user
from database import SessionLocal, get_db
from models import Document, RechtsprechungEntry, ResearchSource, User

# Import from new modular research modules
from .research.gemini import research_with_gemini
from .research.grok import research_with_grok
from .research.openai_search import research_with_openai_search
from .research.asylnet import search_asylnet_with_provisions, ASYL_NET_ALL_SUGGESTIONS
from .research.utils import _looks_like_pdf, download_source_as_pdf

router = APIRouter()


@router.get("/research/suggestions")
async def get_research_suggestions(
    q: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get autocomplete suggestions for asyl.net keywords."""
    if not q:
        return ASYL_NET_ALL_SUGGESTIONS[:20]
    
    q_lower = q.lower()
    matches = [s for s in ASYL_NET_ALL_SUGGESTIONS if s.lower().startswith(q_lower)]
    if len(matches) < 20:
        matches.extend([s for s in ASYL_NET_ALL_SUGGESTIONS if q_lower in s.lower() and s not in matches])
    
    return matches[:20]


# Constants
ASYL_NET_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_COURT_HINT_KEYWORDS = (
    "OVG",
    "VG",
    "BVERW G",
    "BVERWGS",
    "BVERW G",
    "BVERWG",
    "BVERFG",
    "BVERF",
    "BVERFSG",
    "BVerfG",
    "EGMR",
    "EUGH",
    "BGH",
    "VGH",
)

_DECISION_NUMBER_RE = re.compile(r"\b\d{1,4}\s*[A-Za-zÄÖÜäöüß./ -]{0,12}/\d{2,4}\b", re.IGNORECASE)


def _extract_context_hints_from_text(text: str) -> List[str]:
    """Extract court and case-number style hints from free text."""
    if not text:
        return []

    normalized = " ".join(str(text).replace("\n", " ").split())
    if not normalized:
        return []

    lowered = normalized.lower()
    hints: List[str] = []
    seen: Set[str] = set()

    court_hits = [
        kw for kw in _COURT_HINT_KEYWORDS
        if re.search(rf"\\b{re.escape(kw.lower())}\\b", lowered)
    ]
    number_hits = [
        hit.strip().replace("  ", " ")
        for hit in _DECISION_NUMBER_RE.findall(normalized)
    ]
    number_hits = [h for h in number_hits if len(h.replace(" ", "")) <= 24]

    for hit in court_hits:
        hit_norm = hit.strip().upper()
        if hit_norm not in seen:
            seen.add(hit_norm)
            hints.append(hit_norm)

        if number_hits:
            for number in number_hits[:2]:
                combined = f"{hit_norm} {number.strip()}"
                if combined.lower() not in seen:
                    seen.add(combined.lower())
                    hints.append(combined)

    for number in number_hits:
        combined = f"Aktenzeichen {number}"
        if combined.lower() not in seen:
            seen.add(combined.lower())
            hints.append(combined)

    return hints[:8]


def _build_context_query(base_query: Optional[str], context_hints: List[str]) -> str:
    query = (base_query or "").strip()
    if not context_hints:
        return query

    anchor_block = "; ".join(context_hints[:5])
    if not query:
        return (
            "Vergleichsrecherche zu bekannten Entscheidungen: "
            f"{anchor_block}."
        )

    return f"{query} | Fokus auf bekannte Referenzentscheidungen: {anchor_block}"


def _append_unique_identifier(target: List[str], seen: Set[str], value: Optional[str]) -> None:
    if not value:
        return
    normalized = str(value).strip()
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    target.append(normalized)


def _collect_selected_document_identifiers(selection: Any) -> List[str]:
    if not selection:
        return []

    ordered: List[str] = []
    seen: Set[str] = set()

    bescheid = getattr(selection, "bescheid", None)
    if bescheid:
        _append_unique_identifier(ordered, seen, getattr(bescheid, "primary", None))
        for value in getattr(bescheid, "others", []) or []:
            _append_unique_identifier(ordered, seen, value)

    vorinstanz = getattr(selection, "vorinstanz", None)
    if vorinstanz:
        _append_unique_identifier(ordered, seen, getattr(vorinstanz, "primary", None))
        for value in getattr(vorinstanz, "others", []) or []:
            _append_unique_identifier(ordered, seen, value)

    for field_name in ("anhoerung", "rechtsprechung", "akte", "sonstiges"):
        for value in getattr(selection, field_name, []) or []:
            _append_unique_identifier(ordered, seen, value)

    return ordered


def _collect_rechtsprechung_identifiers(selection: Any) -> List[str]:
    if not selection:
        return []

    ordered: List[str] = []
    seen: Set[str] = set()
    for value in getattr(selection, "rechtsprechung", []) or []:
        _append_unique_identifier(ordered, seen, value)
    return ordered


def _normalize_seed_hint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    normalized = " ".join(str(value).split())
    if not normalized:
        return None
    return normalized[:360]


def _append_seed_hint(target: List[str], seen: Set[str], value: Optional[str]) -> None:
    normalized = _normalize_seed_hint(value)
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    target.append(normalized)


def _build_research_context_hints(
    documents: List[Document],
    db: Session,
) -> List[str]:
    hints: List[str] = []
    seen: Set[str] = set()

    for doc in documents:
        _append_seed_hint(hints, seen, doc.filename)
        for parsed_hint in _extract_context_hints_from_text(doc.filename):
            _append_seed_hint(hints, seen, parsed_hint)

        if doc.explanation:
            _append_seed_hint(hints, seen, doc.explanation.strip())
            for parsed_hint in _extract_context_hints_from_text(doc.explanation):
                _append_seed_hint(hints, seen, parsed_hint)

        try:
            entry = (
                db.query(RechtsprechungEntry)
                .filter(RechtsprechungEntry.document_id == doc.id)
                .first()
            )
        except Exception as exc:
            print(f"[WARN] Failed to load Rechtsprechung entry for {doc.filename}: {exc}")
            entry = None

        if not entry:
            continue

        parts: List[str] = []
        if entry.court:
            parts.append(entry.court.strip())
        if entry.aktenzeichen:
            parts.append(entry.aktenzeichen.strip())
        if entry.decision_date:
            parts.append(entry.decision_date.isoformat())
        if parts:
            _append_seed_hint(hints, seen, " ".join(parts))
        if entry.summary:
            _append_seed_hint(hints, seen, entry.summary.strip())

        try:
            extracted_text = load_document_text(doc)
        except Exception as exc:
            print(f"[WARN] Failed to load extracted text for context hints {doc.filename}: {exc}")
            extracted_text = None

        if extracted_text:
            for parsed_hint in _extract_context_hints_from_text(extracted_text[:9000]):
                _append_seed_hint(hints, seen, parsed_hint)

    return hints


def _resolve_document_identifier(
    db: Session,
    current_user: User,
    identifier: str,
) -> Optional[Document]:
    try:
        identifier_uuid = uuid.UUID(identifier)
        return (
            db.query(Document)
            .filter(
                Document.id == identifier_uuid,
                Document.owner_id == current_user.id,
                Document.case_id == current_user.active_case_id,
            )
            .first()
        )
    except ValueError:
        return (
            db.query(Document)
            .filter(
                Document.filename == identifier,
                Document.owner_id == current_user.id,
                Document.case_id == current_user.active_case_id,
            )
            .first()
        )


# ============================================================================
# BACKGROUND DOWNLOAD HELPERS
# ============================================================================

# Helper removed - imported from .research.utils


async def download_and_update_source(source_id: str, url: str, title: str):
    """Background task to download a source and update its status"""
    db = SessionLocal()
    try:
        # Update status to downloading
        source_uuid = uuid.UUID(source_id)
        source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
        if source:
            source.download_status = 'downloading'
            db.commit()
            broadcast_sources_snapshot(db, 'download_started', {'source_id': source_id})

        # Download the PDF
        download_path = await download_source_as_pdf(url, title)

        # Update status
        source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
        if source:
            if download_path:
                source.download_status = 'completed'
                source.download_path = download_path
            else:
                source.download_status = 'failed'
            db.commit()
            broadcast_sources_snapshot(db, 'download_completed' if download_path else 'download_failed', {'source_id': source_id})

    except Exception as e:
        print(f"Error in background download for {url}: {e}")
        # Mark as failed
        try:
            source_uuid = uuid.UUID(source_id)
            source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
            if source:
                source.download_status = 'failed'
                db.commit()
                broadcast_sources_snapshot(db, 'download_failed', {'source_id': source_id})
        except Exception:
            pass
    finally:
        db.close()


# ============================================================================
# RESEARCH ENDPOINTS
# ============================================================================

@router.post("/research", response_model=ResearchResult)
@limiter.limit("10/hour")
async def research(
    request: Request,
    body: ResearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Perform web research using Gemini, Grok, or Meta-Search (Combined)."""
    try:
        requested_engine = (body.search_engine or "meta").strip().lower()
        print(f"[RESEARCH] Search engine: {requested_engine}")
        raw_query = (body.query or "").strip()
        attachment_path: Optional[str] = None
        attachment_label: Optional[str] = None
        attachment_text_path: Optional[str] = None
        attachment_ocr_text: Optional[str] = None
        classification_hint: Optional[str] = None
        chatgpt_attachment_documents: List[Dict[str, Optional[str]]] = []

        # Handle selected/reference documents (new payload + legacy fields)
        reference_doc = None

        resolved_selected_docs: List[Document] = []
        resolved_selected_doc_ids: Set[uuid.UUID] = set()

        selected_identifiers = _collect_selected_document_identifiers(body.selected_documents)
        for identifier in selected_identifiers:
            try:
                doc = _resolve_document_identifier(db, current_user, identifier)
                if doc and doc.id not in resolved_selected_doc_ids:
                    resolved_selected_docs.append(doc)
                    resolved_selected_doc_ids.add(doc.id)
            except Exception as e:
                print(f"[WARN] Error resolving selected document {identifier}: {e}")

        research_context_hints: List[str] = []
        rechtsprechung_identifiers = _collect_rechtsprechung_identifiers(body.selected_documents)
        resolved_rechtsprechung_docs: List[Document] = []
        resolved_rechtsprechung_ids: Set[uuid.UUID] = set()
        for identifier in rechtsprechung_identifiers:
            try:
                doc = _resolve_document_identifier(db, current_user, identifier)
                if not doc or doc.id in resolved_rechtsprechung_ids:
                    continue

                resolved_rechtsprechung_docs.append(doc)
                resolved_rechtsprechung_ids.add(doc.id)
                if doc.id not in resolved_selected_doc_ids:
                    resolved_selected_docs.append(doc)
                    resolved_selected_doc_ids.add(doc.id)
            except Exception as e:
                print(f"[WARN] Error resolving rechtsprechung document {identifier}: {e}")

        research_context_hints = _build_research_context_hints(
            resolved_rechtsprechung_docs,
            db,
        )
        if not research_context_hints and resolved_selected_docs:
            research_context_hints = _build_research_context_hints(
                resolved_selected_docs,
                db,
            )

        if resolved_selected_docs:
            reference_doc = next(
                (d for d in resolved_selected_docs if d.file_path and os.path.exists(d.file_path)),
                resolved_selected_docs[0],
            )

        # 2. Try explicit reference_document_id (from old dropdown if it still existed, or direct API usage)
        if not reference_doc and body.reference_document_id:
            try:
                ref_uuid = uuid.UUID(body.reference_document_id)
                reference_doc = db.query(Document).filter(
                    Document.id == ref_uuid,
                    Document.owner_id == current_user.id,
                    Document.case_id == current_user.active_case_id,
                ).first()
            except ValueError:
                pass
            if not reference_doc:
                # If specifically requested but not found -> 404
                raise HTTPException(status_code=404, detail=f"Referenzdokument '{body.reference_document_id}' nicht gefunden.")
            if reference_doc.id not in resolved_selected_doc_ids:
                resolved_selected_docs.append(reference_doc)
                resolved_selected_doc_ids.add(reference_doc.id)

        # 3. Try legacy primary_bescheid (Filename)
        if not reference_doc and body.primary_bescheid:
            bescheid = db.query(Document).filter(
                Document.filename == body.primary_bescheid,
                Document.owner_id == current_user.id,
                Document.case_id == current_user.active_case_id,
            ).first()
            if not bescheid:
                raise HTTPException(status_code=404, detail=f"Bescheid '{body.primary_bescheid}' wurde nicht gefunden.")
            reference_doc = bescheid
            if reference_doc.id not in resolved_selected_doc_ids:
                resolved_selected_docs.append(reference_doc)
                resolved_selected_doc_ids.add(reference_doc.id)

        if not raw_query and not reference_doc and not resolved_selected_docs:
            raise HTTPException(
                status_code=400,
                detail="Bitte wählen Sie mindestens ein Dokument aus (z.B. Bescheid, Urteil) oder geben Sie eine Suchanfrage ein."
            )

        if reference_doc:
            if not reference_doc.file_path or not os.path.exists(reference_doc.file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Datei für '{reference_doc.filename}' wurde nicht auf dem Server gefunden."
                )
            
            attachment_label = reference_doc.filename
            classification_hint = (reference_doc.explanation or "").strip() or None

            upload_entry = {
                "filename": reference_doc.filename,
                "file_path": reference_doc.file_path,
                "extracted_text_path": reference_doc.extracted_text_path,
                "anonymization_metadata": reference_doc.anonymization_metadata,
                "is_anonymized": reference_doc.is_anonymized,
            }

            try:
                selected_path, mime_type, _ = get_document_for_upload(upload_entry)
            except Exception as exc:
                print(f"[WARN] Failed to resolve attachment for research: {exc}")
                raise HTTPException(
                    status_code=404,
                    detail="Dokumentdatei für die Recherche nicht verfügbar."
                )

            if mime_type == "text/plain":
                attachment_text_path = selected_path
                attachment_path = None
                attachment_ocr_text = None
                print(f"[INFO] Using text file for research: {attachment_text_path}")
            else:
                attachment_path = selected_path
                attachment_text_path = None
                attachment_ocr_text = None
                print(f"[INFO] Using PDF for research: {attachment_path}")
            
            if not raw_query:
                # Generate a default query based on the document
                derived_parts = [
                    f"Automatische Recherche basierend auf dem Dokument: {reference_doc.filename}",
                    f"Kategorie: {reference_doc.category}"
                ]
                if classification_hint:
                    derived_parts.append(f"Inhalt/Kontext: {classification_hint}")
                raw_query = "\n".join(derived_parts)

        effective_query = _build_context_query(raw_query, research_context_hints)

        if resolved_selected_docs:
            for doc in resolved_selected_docs:
                if not doc.file_path or not os.path.exists(doc.file_path):
                    print(f"[WARN] ChatGPT context doc skipped (missing file): {doc.filename}")
                    continue

                upload_entry = {
                    "filename": doc.filename,
                    "file_path": doc.file_path,
                    "extracted_text_path": doc.extracted_text_path,
                    "anonymization_metadata": doc.anonymization_metadata,
                    "is_anonymized": doc.is_anonymized,
                }
                try:
                    selected_path, mime_type, _ = get_document_for_upload(upload_entry)
                except Exception as exc:
                    print(f"[WARN] ChatGPT context doc skipped ({doc.filename}): {exc}")
                    continue

                payload = {"attachment_display_name": doc.filename}
                if mime_type == "text/plain":
                    payload["attachment_text_path"] = selected_path
                    payload["attachment_path"] = None
                    payload["attachment_ocr_text"] = None
                    payload["anonymization_metadata"] = doc.anonymization_metadata
                    payload["is_anonymized"] = doc.is_anonymized
                else:
                    payload["attachment_path"] = selected_path
                    payload["attachment_text_path"] = None
                    payload["attachment_ocr_text"] = None
                    payload["anonymization_metadata"] = doc.anonymization_metadata
                    payload["is_anonymized"] = doc.is_anonymized

                chatgpt_attachment_documents.append(payload)

        if chatgpt_attachment_documents:
            print(f"[RESEARCH] ChatGPT context documents: {len(chatgpt_attachment_documents)}")

        print(f"Starting research pipeline for query: {effective_query}")

        # META SEARCH implementation
        if requested_engine == "meta":
            from .research.meta import aggregate_search_results
            print("[RESEARCH] Starting META SEARCH (Grok + Gemini + ChatGPT + Asyl.net)")
            
            # Prepare tasks
            tasks = []
            
            # 1. Grok
            tasks.append(research_with_grok(
                effective_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path,
                attachment_anonymization_metadata=reference_doc.anonymization_metadata if reference_doc else None,
                attachment_is_anonymized=reference_doc.is_anonymized if reference_doc else False,
                research_context_hints=research_context_hints,
            ))
            
            # 2. Gemini
            tasks.append(research_with_gemini(
                effective_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path,
                attachment_anonymization_metadata=reference_doc.anonymization_metadata if reference_doc else None,
                attachment_is_anonymized=reference_doc.is_anonymized if reference_doc else False,
                document_id=str(reference_doc.id) if reference_doc else None,
                owner_id=str(current_user.id),
                case_id=str(current_user.active_case_id) if current_user.active_case_id else None,
                research_context_hints=research_context_hints,
            ))

            # 3. OpenAI ChatGPT Search
            tasks.append(research_with_openai_search(
                effective_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path,
                attachment_anonymization_metadata=reference_doc.anonymization_metadata if reference_doc else None,
                attachment_is_anonymized=reference_doc.is_anonymized if reference_doc else False,
                attachment_documents=chatgpt_attachment_documents or None,
                research_context_hints=research_context_hints,
            ))
            
            # 4. Asyl.net (with provided keywords if available)
            asyl_query = (body.query or "").strip()
            if not asyl_query:
                asyl_query = classification_hint or attachment_label or effective_query

            # Build document entry for provision extraction/asylnet
            doc_entry = None
            if attachment_path or attachment_text_path:
                doc_entry = {
                    "filename": attachment_label or "Bescheid",
                    "file_path": attachment_path,
                    "extracted_text_path": attachment_text_path,
                    "anonymization_metadata": reference_doc.anonymization_metadata if reference_doc else None,
                    "is_anonymized": reference_doc.is_anonymized if reference_doc else False,
                    "extracted_text": attachment_ocr_text,
                    "ocr_applied": bool(attachment_ocr_text),
                }

            tasks.append(search_asylnet_with_provisions(
                asyl_query,
                attachment_label=attachment_label,
                attachment_doc=doc_entry,
                manual_keywords=body.asylnet_keywords
            ))

            # Run all
            results = await asyncio.gather(*tasks, return_exceptions=True)

            if all(isinstance(res, Exception) for res in results):
                raise HTTPException(
                    status_code=502,
                    detail="Recherche fehlgeschlagen: Alle Meta-Suchanbieter sind fehlgeschlagen.",
                )

            valid_results = []
            provider_names = ["grok-4-1-fast", "gemini", "chatgpt-search", "asyl.net"]
            for idx, res in enumerate(results):
                if isinstance(res, ResearchResult):
                    valid_results.append(res)
                elif isinstance(res, dict) and "asylnet_sources" in res:
                    # Convert asylnet dict result to ResearchResult-like object or source list
                    # Asylnet module returns a dict, not ResearchResult object.
                    # We need to adapt it.
                    asyl_res = ResearchResult(
                        query=asyl_query,
                        summary="",
                        sources=res.get("asylnet_sources", []) + res.get("legal_sources", []),
                        suggestions=res.get("keywords", [])
                    )
                    valid_results.append(asyl_res)
                elif isinstance(res, Exception):
                    provider_name = provider_names[idx] if idx < len(provider_names) else f"provider-{idx}"
                    print(f"[RESEARCH] Meta provider failed ({provider_name}): {res}")

            if not valid_results:
                raise HTTPException(
                    status_code=502,
                    detail="Recherche fehlgeschlagen: Keine verwertbaren Ergebnisse aus Meta-Suche.",
                )
            
            # Aggregate and Rank
            final_result = await aggregate_search_results(effective_query, valid_results)
            return final_result

        # FALLBACK / STANDARD LOGIC (unchanged for specific engines)
        # Prepare asyl.net query and document entry
        asyl_query = (body.query or "").strip()
        if not asyl_query:
            asyl_query = classification_hint or attachment_label or raw_query

        # Build document entry for provision extraction
        doc_entry = None
        if attachment_path or attachment_text_path:
            doc_entry = {
                "filename": attachment_label or "Bescheid",
                "file_path": attachment_path,
                "extracted_text_path": attachment_text_path,
                "anonymization_metadata": reference_doc.anonymization_metadata if reference_doc else None,
                "is_anonymized": reference_doc.is_anonymized if reference_doc else False,
                "extracted_text": attachment_ocr_text,
                "ocr_applied": bool(attachment_ocr_text),
            }

        # Run web search and asyl.net search CONCURRENTLY
        print("[RESEARCH] Starting concurrent API calls (web search + asyl.net + legal provisions)")

        if requested_engine == "grok-4-1-fast":
            print("[RESEARCH] Using Grok-4.1-Fast (web_search tool)")
            web_task = research_with_grok(
                effective_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path,
                attachment_anonymization_metadata=reference_doc.anonymization_metadata if reference_doc else None,
                attachment_is_anonymized=reference_doc.is_anonymized if reference_doc else False,
                research_context_hints=research_context_hints,
            )
        elif requested_engine == "chatgpt-search":
            print("[RESEARCH] Using ChatGPT Search (OpenAI Responses API web_search)")
            web_task = research_with_openai_search(
                effective_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path,
                attachment_anonymization_metadata=reference_doc.anonymization_metadata if reference_doc else None,
                attachment_is_anonymized=reference_doc.is_anonymized if reference_doc else False,
                attachment_documents=chatgpt_attachment_documents or None,
                research_context_hints=research_context_hints,
            )
        else:
            print("[RESEARCH] Using Gemini with Google Search")
            web_task = research_with_gemini(
                effective_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path,
                attachment_anonymization_metadata=reference_doc.anonymization_metadata if reference_doc else None,
                attachment_is_anonymized=reference_doc.is_anonymized if reference_doc else False,
                document_id=str(reference_doc.id) if reference_doc else None,
                owner_id=str(current_user.id),
                case_id=str(current_user.active_case_id) if current_user.active_case_id else None,
                research_context_hints=research_context_hints,
            )

        asylnet_task = search_asylnet_with_provisions(
            asyl_query,
            attachment_label=attachment_label,
            attachment_doc=doc_entry,
            manual_keywords=body.asylnet_keywords
        )

        # Execute both concurrently; degrade gracefully on partial failures.
        web_result_raw, asylnet_result_raw = await asyncio.gather(
            web_task,
            asylnet_task,
            return_exceptions=True,
        )
        print("[RESEARCH] Concurrent API calls completed")

        web_result: Optional[ResearchResult] = None
        if isinstance(web_result_raw, Exception):
            print(f"[RESEARCH] Web search provider failed ({requested_engine}): {web_result_raw}")
        elif isinstance(web_result_raw, ResearchResult):
            web_result = web_result_raw
        else:
            print(f"[RESEARCH] Unexpected web_result type: {type(web_result_raw)}")

        asylnet_result = {
            "keywords": [],
            "asylnet_sources": [],
            "legal_sources": [],
        }
        if isinstance(asylnet_result_raw, Exception):
            print(f"[RESEARCH] asyl.net/provisions pipeline failed: {asylnet_result_raw}")
        elif isinstance(asylnet_result_raw, dict):
            asylnet_result["keywords"] = asylnet_result_raw.get("keywords", []) or []
            asylnet_result["asylnet_sources"] = asylnet_result_raw.get("asylnet_sources", []) or []
            asylnet_result["legal_sources"] = asylnet_result_raw.get("legal_sources", []) or []
        else:
            print(f"[RESEARCH] Unexpected asylnet_result type: {type(asylnet_result_raw)}")

        if web_result is None and not asylnet_result["asylnet_sources"] and not asylnet_result["legal_sources"]:
            raise HTTPException(
                status_code=502,
                detail="Recherche fehlgeschlagen: Websuche und asyl.net konnten nicht verarbeitet werden.",
            )

        web_source_count = len(web_result.sources) if web_result else 0
        asylnet_sources = asylnet_result["asylnet_sources"] + asylnet_result["legal_sources"]

        if not web_result and asylnet_sources:
            print(f"Returning asyl.net-only result with {len(asylnet_sources)} sources")
            return ResearchResult(
                query=effective_query,
                summary="Asyl.net-Resultate und relevante Gesetzestexte wurden gefunden.",
                sources=asylnet_sources,
                suggestions=asylnet_result["keywords"],
            )

        merged_results = []
        if web_result:
            merged_results.append(web_result)

        if asylnet_sources:
            merged_results.append(
                ResearchResult(
                    query=effective_query,
                    summary="Asyl.net-Resultate und relevante Gesetzestexte wurden gefunden.",
                    sources=asylnet_sources,
                    suggestions=asylnet_result["keywords"],
                )
            )

        if len(merged_results) == 1:
            return merged_results[0]

        print(f"Combined research returned {web_source_count} web + {len(asylnet_sources)} asyl sources")

        from .research.meta import aggregate_search_results
        final_result = await aggregate_search_results(effective_query, merged_results)
        return final_result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Research failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler bei der Recherche. Bitte später erneut versuchen.",
        )


# ============================================================================
# SOURCE MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/sources", response_model=SavedSource)
@limiter.limit("100/hour")
async def add_source_endpoint(
    request: Request,
    body: AddSourceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Manually add a research source and optionally download its PDF."""
    source_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    new_source = ResearchSource(
        id=uuid.UUID(source_id),
        url=body.url,
        title=body.title,
        description=body.description,
        document_type=body.document_type,
        pdf_url=body.pdf_url,
        download_status="pending" if body.auto_download else "skipped",
        research_query=body.research_query or "Manuell hinzugefügt",
        owner_id=current_user.id,
        case_id=current_user.active_case_id,
    )
    db.add(new_source)
    db.commit()
    db.refresh(new_source)

    broadcast_sources_snapshot(db, "add", {"source_id": source_id})

    saved_source = SavedSource(
        id=source_id,
        url=body.url,
        title=body.title,
        description=body.description,
        document_type=body.document_type,
        pdf_url=body.pdf_url,
        download_status="pending" if body.auto_download else "skipped",
        research_query=body.research_query or "Manuell hinzugefügt",
        timestamp=timestamp,
    )

    if body.auto_download:
        download_target = body.pdf_url or body.url
        asyncio.create_task(download_and_update_source(source_id, download_target, body.title))

    return saved_source


@router.get("/sources")
@limiter.limit("1000/hour")
async def get_sources(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all saved research sources."""
    sources = db.query(ResearchSource).filter(
        ResearchSource.owner_id == current_user.id,
        ResearchSource.case_id == current_user.active_case_id,
    ).order_by(desc(ResearchSource.created_at)).all()
    sources_dict = [s.to_dict() for s in sources]
    return JSONResponse(
        content={
            "count": len(sources_dict),
            "sources": sources_dict,
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/sources/download/{source_id}")
@limiter.limit("50/hour")
async def download_source_file(
    request: Request,
    source_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Download a saved source PDF."""
    try:
        source_uuid = uuid.UUID(source_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source ID format")

    source = db.query(ResearchSource).filter(
        ResearchSource.id == source_uuid,
        ResearchSource.owner_id == current_user.id,
        ResearchSource.case_id == current_user.active_case_id,
    ).first()
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    if not source.download_path:
        raise HTTPException(status_code=404, detail="Source not downloaded yet")

    download_path = Path(source.download_path)
    if not download_path.exists():
        raise HTTPException(status_code=404, detail="Downloaded file not found")

    return FileResponse(
        path=download_path,
        media_type="application/pdf",
        filename=f"{source.title}.pdf",
    )


@router.delete("/sources/{source_id}")
@limiter.limit("100/hour")
async def delete_source_endpoint(
    request: Request,
    source_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a saved source."""
    try:
        source_uuid = uuid.UUID(source_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source ID format")

    source = db.query(ResearchSource).filter(
        ResearchSource.id == source_uuid,
        ResearchSource.owner_id == current_user.id,
        ResearchSource.case_id == current_user.active_case_id,
    ).first()
    
    if not source:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

    if source.download_path:
        download_path = Path(source.download_path)
        if download_path.exists():
            try:
                download_path.unlink()
            except Exception as exc:
                print(f"Error deleting file {download_path}: {exc}")

    db.delete(source)
    db.commit()
    broadcast_sources_snapshot(db, "delete", {"source_id": source_id})
    return {"message": f"Source {source_id} deleted successfully"}


@router.delete("/sources")
@limiter.limit("50/hour")
async def delete_all_sources_endpoint(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete all saved sources."""
    sources = db.query(ResearchSource).filter(
        ResearchSource.owner_id == current_user.id,
        ResearchSource.case_id == current_user.active_case_id,
    ).all()

    if not sources:
        return {"message": "No sources to delete", "count": 0}

    deleted_count = 0
    for source in sources:
        if source.download_path:
            download_path = Path(source.download_path)
            if download_path.exists():
                try:
                    download_path.unlink()
                    deleted_count += 1
                except Exception as exc:
                    print(f"Error deleting file {download_path}: {exc}")

    sources_count = len(sources)
    db.query(ResearchSource).filter(
        ResearchSource.owner_id == current_user.id,
        ResearchSource.case_id == current_user.active_case_id,
    ).delete(synchronize_session=False)
    db.commit()

    broadcast_sources_snapshot(db, "delete_all", {"count": sources_count})
    return {
        "message": "All sources deleted successfully",
        "count": sources_count,
        "files_deleted": deleted_count,
    }

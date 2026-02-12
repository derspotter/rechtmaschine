"""
Research sources endpoint - refactored and modular.

Routes research requests to appropriate providers and manages saved sources.
"""

import asyncio
import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    DocumentCategory,
    broadcast_sources_snapshot,
    get_document_for_upload,
    limiter,
)
from auth import get_current_active_user
from database import SessionLocal, get_db
from models import Document, ResearchSource, User

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

        # Handle reference document (generic) OR primary_bescheid (legacy/specific)
        # Handle reference document from selected_documents (New Logic) OR legacy fields
        reference_doc = None
        
        # 1. Try new generic selection payload
        if body.selected_documents:
            candidate_id = None
            # Prioritize Bescheid (Primary)
            if body.selected_documents.bescheid and body.selected_documents.bescheid.primary:
                candidate_id = body.selected_documents.bescheid.primary
            # Then Vorinstanz (Primary)
            elif body.selected_documents.vorinstanz and body.selected_documents.vorinstanz.primary:
                 candidate_id = body.selected_documents.vorinstanz.primary
            # Then any Rechtsprechung
            elif body.selected_documents.rechtsprechung and len(body.selected_documents.rechtsprechung) > 0:
                candidate_id = body.selected_documents.rechtsprechung[0]
            # Then first of others
            elif body.selected_documents.akte and len(body.selected_documents.akte) > 0:
                candidate_id = body.selected_documents.akte[0]
            
            if candidate_id:
                try:
                    # In case the frontend sends a filename (legacy fallback), check if it's UUID
                    try:
                        ref_uuid = uuid.UUID(candidate_id)
                        reference_doc = db.query(Document).filter(
                            Document.id == ref_uuid,
                            Document.owner_id == current_user.id,
                            Document.case_id == current_user.active_case_id,
                        ).first()
                    except ValueError:
                        # Maybe it is a filename?
                        reference_doc = db.query(Document).filter(
                            Document.filename == candidate_id,
                            Document.owner_id == current_user.id,
                            Document.case_id == current_user.active_case_id,
                        ).first()
                except Exception as e:
                    print(f"[WARN] Error resolving selected document {candidate_id}: {e}")

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

        if not raw_query and not reference_doc:
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

        print(f"Starting research pipeline for query: {raw_query}")

        # META SEARCH implementation
        if requested_engine == "meta":
            from .research.meta import aggregate_search_results
            print("[RESEARCH] Starting META SEARCH (Grok + Gemini + ChatGPT + Asyl.net)")
            
            # Prepare tasks
            tasks = []
            
            # 1. Grok
            tasks.append(research_with_grok(
                raw_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path
            ))
            
            # 2. Gemini
            tasks.append(research_with_gemini(
                raw_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path,
                document_id=str(reference_doc.id) if reference_doc else None,
                owner_id=str(current_user.id),
                case_id=str(current_user.active_case_id) if current_user.active_case_id else None,
            ))

            # 3. OpenAI ChatGPT Search
            tasks.append(research_with_openai_search(
                raw_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path,
            ))
            
            # 4. Asyl.net (with provided keywords if available)
            asyl_query = (body.query or "").strip()
            if not asyl_query:
                asyl_query = classification_hint or attachment_label or raw_query

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
            final_result = await aggregate_search_results(raw_query, valid_results)
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
                raw_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path
            )
        elif requested_engine == "chatgpt-search":
            print("[RESEARCH] Using ChatGPT Search (OpenAI Responses API web_search)")
            web_task = research_with_openai_search(
                raw_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path
            )
        else:
            print("[RESEARCH] Using Gemini with Google Search")
            web_task = research_with_gemini(
                raw_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path,
                document_id=str(reference_doc.id) if reference_doc else None,
                owner_id=str(current_user.id),
                case_id=str(current_user.active_case_id) if current_user.active_case_id else None,
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

        all_sources = list(web_result.sources) if web_result else []
        summaries = [web_result.summary] if web_result and web_result.summary else []

        # Add asyl.net sources
        all_sources.extend(asylnet_result["asylnet_sources"])

        # Add legal provision sources
        all_sources.extend(asylnet_result["legal_sources"])

        combined_summary = "<hr/>".join(summaries) if summaries else ""

        print(f"Combined research returned {len(all_sources)} total sources")
        print(f"  - Web sources: {len(web_result.sources) if web_result else 0}")
        print(f"  - asyl.net sources: {len(asylnet_result['asylnet_sources'])}")
        print(f"  - Legal provisions: {len(asylnet_result['legal_sources'])}")

        return ResearchResult(
            query=raw_query,
            summary=combined_summary,
            sources=all_sources,
            suggestions=asylnet_result["keywords"]  # asyl.net keywords for UI
        )

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

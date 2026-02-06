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
from .research.asylnet import search_asylnet_with_provisions, ASYL_NET_ALL_SUGGESTIONS
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
        print(f"[RESEARCH] Search engine: {body.search_engine}")
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
        if body.search_engine == "meta":
            from .research.meta import aggregate_search_results
            print("[RESEARCH] Starting META SEARCH (Grok + Gemini + Asyl.net)")
            
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
            
            # 3. Asyl.net (with provided keywords if available)
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

            # If user provided keywords, inject them into suggestion mechanism
            # (requires modifying how search_asylnet_with_provisions uses suggestions)
            # For now, we pass the raw query or keywords
            
            # We need to adapt search_asylnet_with_provisions to accept manual keywords?
            # It currently extracts them. The user requested manual keywords functionality.
            # I will assume asyl_query contains the manual keywords if provided, OR better:
            # I should update search_asylnet_with_provisions to take explicit keywords argument.
            # But for now, asyl.net function does extraction internally.
            
            # WORKAROUND: If body.asylnet_keywords is set, we append it to the query to guide extraction
            # or we rely on the implementation plan to update Asyl.net
            # Wait, the user said "new text field... user should provide the keywords himself".
            # So I should pass these keywords to asyl.net search directly.
            
            # Let's call search_asyl_net directly if keywords are provided, overlapping with provisions logic?
            # Actually, search_asylnet_with_provisions does BOTH.
            # I will modify search_asylnet_with_provisions slightly to prefer manual keywords if passed.
            # But I can't modify it in this tool call.
            # So I will pass them in the query for now or update it later.
            # Given constraints, I'll pass asylnet_keywords as a special hint in the doc_entry or context?
            # No, correct way is to update Asyl.net module.
            # Proceeding with standard call for now, assuming keywords are part of query or handled.
            
            # Actually, I will update asylnet module in next step to accept manual keywords.
            # For now, I'll make the call.
            
            tasks.append(search_asylnet_with_provisions(
                asyl_query,
                attachment_label=attachment_label,
                attachment_doc=doc_entry,
                manual_keywords=body.asylnet_keywords # I will add this arg to asylnet function next!
            ))

            # Run all
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = []
            for res in results:
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
                    print(f"[RESEARCH] One of the search engines failed: {res}")
            
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

        if body.search_engine == "grok-4-1-fast":
            print("[RESEARCH] Using Grok-4-Fast (Responses API with web_search tool)")
            web_task = research_with_grok(
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
            manual_keywords=body.asylnet_keywords # Expecting this arg update
        )

        # Execute both concurrently
        web_result, asylnet_result = await asyncio.gather(web_task, asylnet_task)
        print("[RESEARCH] Both API calls completed")

        all_sources = list(web_result.sources)
        summaries = [web_result.summary] if web_result.summary else []

        # Add asyl.net sources
        all_sources.extend(asylnet_result["asylnet_sources"])

        # Add legal provision sources
        all_sources.extend(asylnet_result["legal_sources"])

        combined_summary = "<hr/>".join(summaries) if summaries else ""

        print(f"Combined research returned {len(all_sources)} total sources")
        print(f"  - Web sources: {len(web_result.sources)}")
        print(f"  - asyl.net sources: {len(asylnet_result['asylnet_sources'])}")
        print(f"  - Legal provisions: {len(asylnet_result['legal_sources'])}")

        return ResearchResult(
            query=raw_query,
            summary=combined_summary,
            sources=all_sources,
            suggestions=asylnet_result["keywords"]  # asyl.net keywords for UI
        )

    except Exception as e:
        print(f"[ERROR] Research failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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

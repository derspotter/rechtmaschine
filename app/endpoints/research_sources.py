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
    limiter,
)
from database import SessionLocal, get_db
from models import Document, ResearchSource

# Import from new modular research modules
from .research.gemini import research_with_gemini
from .research.grok import research_with_grok
from .research.asylnet import search_asylnet_with_provisions
from .research.utils import _looks_like_pdf

router = APIRouter()


# Constants
ASYL_NET_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# ============================================================================
# BACKGROUND DOWNLOAD HELPERS
# ============================================================================

async def download_source_as_pdf(url: str, filename: str) -> Optional[str]:
    """
    Download a source as PDF. Handles both direct PDFs and HTML pages.
    Returns the path to the downloaded file, or None if failed.
    """

    # Ensure downloads directory exists
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()
    file_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    output_path = DOWNLOADS_DIR / f"{file_hash}_{safe_filename}.pdf"

    try:
        # Check if URL is already a PDF
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            headers={"User-Agent": ASYL_NET_USER_AGENT}
        ) as http_client:
            direct_pdf_url: Optional[str] = None

            try:
                head_response = await http_client.head(url)
                if _looks_like_pdf(head_response.headers):
                    direct_pdf_url = str(head_response.url)
            except httpx.HTTPError as err:
                status = getattr(getattr(err, "response", None), "status_code", None)
                if status not in {401, 403, 404, 405, 406, 409, 410}:
                    print(f"HEAD download probe failed for {url}: {err}")

            if not direct_pdf_url:
                try:
                    probe_response = await http_client.get(
                        url,
                        headers={"Range": "bytes=0-0"}
                    )
                    if _looks_like_pdf(probe_response.headers):
                        direct_pdf_url = str(probe_response.url)
                except httpx.HTTPError as err:
                    status = getattr(getattr(err, "response", None), "status_code", None)
                    if status not in {401, 403, 404, 405, 406, 409, 410, 416}:
                        print(f"GET download probe failed for {url}: {err}")

            if direct_pdf_url:
                print(f"Downloading PDF directly: {direct_pdf_url}")
                response = await http_client.get(direct_pdf_url, timeout=60.0)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"PDF downloaded to: {output_path}")
                return str(output_path)

        # HTML page - convert to PDF using Playwright
        print(f"Converting HTML to PDF: {url}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await page.pdf(path=str(output_path), format='A4')
            await browser.close()

        print(f"HTML converted to PDF: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


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
async def research(request: Request, body: ResearchRequest, db: Session = Depends(get_db)):
    """Perform web research using Gemini or Grok with asyl.net + legal texts"""
    try:
        print(f"[RESEARCH] Search engine: {body.search_engine}")
        raw_query = (body.query or "").strip()
        attachment_path: Optional[str] = None
        attachment_label: Optional[str] = None
        attachment_text_path: Optional[str] = None
        attachment_ocr_text: Optional[str] = None
        classification_hint: Optional[str] = None

        # Handle primary_bescheid attachment
        if not raw_query:
            if not body.primary_bescheid:
                raise HTTPException(
                    status_code=400,
                    detail="Bitte geben Sie eine Rechercheanfrage ein oder wählen Sie einen Hauptbescheid aus."
                )

            bescheid = db.query(Document).filter(Document.filename == body.primary_bescheid).first()
            if not bescheid:
                raise HTTPException(status_code=404, detail=f"Bescheid '{body.primary_bescheid}' wurde nicht gefunden.")
            if bescheid.category != DocumentCategory.BESCHEID.value:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dokument '{bescheid.filename}' ist kein Bescheid und kann nicht für die automatische Recherche verwendet werden."
                )
            if not bescheid.file_path or not os.path.exists(bescheid.file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF-Datei für '{bescheid.filename}' wurde nicht auf dem Server gefunden."
                )

            attachment_label = bescheid.filename
            classification_hint = (bescheid.explanation or "").strip() or None

            attachment_text_path = bescheid.extracted_text_path if bescheid.extracted_text_path and os.path.exists(bescheid.extracted_text_path) else None

            if attachment_text_path:
                print(f"[INFO] Using OCR text file for research: {attachment_text_path}")
                attachment_path = None
                attachment_ocr_text = None
            else:
                print(f"[INFO] No OCR text file available, will use PDF")
                attachment_path = bescheid.file_path
                attachment_ocr_text = None

            derived_parts = [
                "Automatische Recherche basierend auf dem beigefügten BAMF-Bescheid.",
                f"Dateiname: {bescheid.filename}"
            ]
            if classification_hint:
                derived_parts.append(f"Kurze Einordnung laut Klassifikation: {classification_hint}")
            raw_query = "\n".join(derived_parts)

        print(f"Starting research pipeline for query: {raw_query}")

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
                "extracted_text": attachment_ocr_text,
                "ocr_applied": bool(attachment_ocr_text),
            }

        # Run web search and asyl.net search CONCURRENTLY
        print("[RESEARCH] Starting concurrent API calls (web search + asyl.net + legal provisions)")

        if body.search_engine == "grok-4-fast":
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
                attachment_text_path=attachment_text_path
            )

        asylnet_task = search_asylnet_with_provisions(
            asyl_query,
            attachment_label=attachment_label,
            attachment_doc=doc_entry
        )

        # Execute both concurrently
        web_result, asylnet_result = await asyncio.gather(web_task, asylnet_task)
        print("[RESEARCH] Both API calls completed")

        all_sources = list(web_result.sources)
        summaries = [web_result.summary] if web_result.summary else []

        # Add asyl.net sources
        all_sources.extend(asylnet_result["asylnet_sources"])

        # Add legal provision sources (NEW!)
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
async def add_source_endpoint(request: Request, body: AddSourceRequest, db: Session = Depends(get_db)):
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
async def get_sources(request: Request, db: Session = Depends(get_db)):
    """Get all saved research sources."""
    sources = db.query(ResearchSource).order_by(desc(ResearchSource.created_at)).all()
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
async def download_source_file(request: Request, source_id: str, db: Session = Depends(get_db)):
    """Download a saved source PDF."""
    try:
        source_uuid = uuid.UUID(source_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source ID format")

    source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
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
async def delete_source_endpoint(request: Request, source_id: str, db: Session = Depends(get_db)):
    """Delete a saved source."""
    try:
        source_uuid = uuid.UUID(source_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source ID format")

    source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
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
async def delete_all_sources_endpoint(request: Request, db: Session = Depends(get_db)):
    """Delete all saved sources."""
    sources = db.query(ResearchSource).all()

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
    db.query(ResearchSource).delete()
    db.commit()

    broadcast_sources_snapshot(db, "delete_all", {"count": sources_count})
    return {
        "message": "All sources deleted successfully",
        "count": sources_count,
        "files_deleted": deleted_count,
    }

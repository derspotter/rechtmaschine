

import asyncio
import mimetypes
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from datetime import datetime
import shutil

from shared import (
    DOWNLOADS_DIR,
    UPLOADS_DIR,
    OCR_TEXT_DIR,
    limiter,
    build_documents_snapshot,
    build_sources_snapshot,
    broadcast_documents_snapshot,
    broadcast_sources_snapshot,
    clear_directory_contents,
    delete_document_text,
    AddDocumentFromUrlRequest,
)
from auth import get_current_active_user
from database import get_db
from models import Document, ResearchSource, User, GeneratedDraft
from .research.utils import download_source_as_pdf

router = APIRouter()


@router.get("/documents/stream")
async def documents_stream(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Unified SSE stream for all updates (documents, sources, etc.)."""
    hub = getattr(request.app.state, "document_hub", None)
    if not hub:
        raise HTTPException(status_code=503, detail="Event stream unavailable")

    queue = await hub.subscribe()

    async def event_generator():
        try:
            docs_payload = {
                "type": "documents_snapshot",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "initial",
                "documents": build_documents_snapshot(db, current_user.id),
            }
            yield f"data: {json.dumps(docs_payload, ensure_ascii=False)}\n\n"

            sources_payload = {
                "type": "sources_snapshot",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "initial",
                "sources": build_sources_snapshot(db, current_user.id),
            }
            yield f"data: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            await hub.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/documents")
@limiter.limit("200/hour")
async def get_documents(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all classified documents grouped by category."""
    grouped = build_documents_snapshot(db, current_user.id)
    return JSONResponse(
        content=grouped,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/documents/{filename}")
@limiter.limit("100/hour")
async def get_document_file(
    request: Request,
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Serve a specific document file."""
    doc = db.query(Document).filter(
        Document.filename == filename,
        Document.owner_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if not doc.file_path or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    mime_type, _ = mimetypes.guess_type(doc.file_path)
    media_type = mime_type or "application/octet-stream"

    return FileResponse(
        doc.file_path,
        media_type=media_type,
        filename=filename
    )


@router.delete("/documents/{filename}")
@limiter.limit("100/hour")
async def delete_document(
    request: Request,
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a classified document."""
    doc = db.query(Document).filter(
        Document.filename == filename,
        Document.owner_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.file_path:
        try:
            fp = Path(doc.file_path)
            if fp.exists():
                fp.unlink()
        except Exception as exc:
            print(f"Error deleting uploaded file for {filename}: {exc}")

    delete_document_text(doc)

    db.delete(doc)
    db.commit()
    broadcast_documents_snapshot(db, "delete", {"filename": filename})
    return {"message": f"Document {filename} deleted successfully"}


@router.post("/documents/from-url")
@limiter.limit("20/hour")
async def add_document_from_url(
    request: Request,
    body: AddDocumentFromUrlRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Add a document directly from a URL (e.g., from research results)."""
    
    # 1. Download the PDF
    temp_path_str = await download_source_as_pdf(body.url, body.title, target_dir=DOWNLOADS_DIR)
    if not temp_path_str:
        raise HTTPException(status_code=400, detail="Could not download document from URL")
        
    temp_path = Path(temp_path_str)
    
    # 2. Prepare destination in UPLOADS_DIR
    safe_filename = "".join(c for c in body.title if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_filename.lower().endswith('.pdf'):
        safe_filename += ".pdf"
        
    # Ensure unique filename
    base_name = safe_filename
    counter = 1
    while (UPLOADS_DIR / safe_filename).exists() or db.query(Document).filter(Document.filename == safe_filename, Document.owner_id == current_user.id).first():
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        safe_filename = f"{stem}_{counter}{suffix}"
        counter += 1
        
    final_path = UPLOADS_DIR / safe_filename
    
    # 3. Move file
    try:
        shutil.move(str(temp_path), str(final_path))
    except Exception as e:
        print(f"Error moving file: {e}")
        # Try copy if move fails
        shutil.copy(str(temp_path), str(final_path))
        temp_path.unlink()
        
    # 4. Create Document record
    new_doc = Document(
        filename=safe_filename,
        category=body.category,
        confidence=1.0, # Manually added, high confidence
        explanation=f"Importiert von URL: {body.url}",
        file_path=str(final_path),
        processing_status="pending",
        owner_id=current_user.id,
        needs_ocr=True # Assume needs OCR usually
    )
    
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    
    broadcast_documents_snapshot(db, "add_source", {"filename": safe_filename})
    
    return new_doc.to_dict()


@router.delete("/reset")
@limiter.limit("10/hour")
async def reset_application(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete all stored data (documents, sources, files) for the current user."""
    documents = db.query(Document).filter(Document.owner_id == current_user.id).all()
    sources = db.query(ResearchSource).filter(ResearchSource.owner_id == current_user.id).all()

    document_paths = [doc.file_path for doc in documents if doc.file_path]
    segment_dirs: set[Path] = set()
    text_paths = []
    for doc in documents:
        file_path = doc.file_path
        if file_path:
            path_obj = Path(file_path)
            if path_obj.exists():
                document_paths.append(str(path_obj))
            if path_obj.parent.name.endswith("_segments"):
                segment_dirs.add(path_obj.parent)
        if doc.extracted_text_path:
            text_paths.append(doc.extracted_text_path)

    document_count = len(documents)
    source_count = len(sources)

    try:
        # Delete generated drafts first to satisfy foreign key constraints
        db.query(GeneratedDraft).filter(GeneratedDraft.user_id == current_user.id).delete(synchronize_session=False)
        
        # Also delete any drafts referencing the documents we are about to delete
        # (This catches cases where user_id might be missing but the link exists)
        doc_ids = [d.id for d in documents]
        if doc_ids:
            db.query(GeneratedDraft).filter(GeneratedDraft.primary_document_id.in_(doc_ids)).delete(synchronize_session=False)

        db.query(Document).filter(Document.owner_id == current_user.id).delete(synchronize_session=False)
        db.query(ResearchSource).filter(ResearchSource.owner_id == current_user.id).delete(synchronize_session=False)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear stored data: {exc}")

    additional_removed = 0
    for raw_path in document_paths:
        try:
            file_path = Path(raw_path)
            if file_path.exists():
                file_path.unlink()
                additional_removed += 1
        except Exception as exc:
            print(f"Error deleting document file {raw_path}: {exc}")

    for directory in segment_dirs:
        try:
            shutil.rmtree(directory)
            additional_removed += 1
        except Exception as exc:
            print(f"Error deleting segment directory {directory}: {exc}")

    for raw_path in text_paths:
        try:
            path_obj = Path(raw_path)
            if path_obj.exists():
                path_obj.unlink()
                additional_removed += 1
        except Exception as exc:
            print(f"Error deleting OCR text file {raw_path}: {exc}")

    # Note: We do NOT clear the global directories (UPLOADS_DIR, etc.) because they might contain other users' files.
    # We only delete the specific files we tracked.
    
    broadcast_documents_snapshot(
        db,
        "reset",
        {
            "documents_deleted": document_count,
            "sources_deleted": source_count,
        },
    )
    broadcast_sources_snapshot(
        db,
        "reset",
        {
            "documents_deleted": document_count,
            "sources_deleted": source_count,
        },
    )

    return {
        "message": "All stored data cleared successfully",
        "documents_deleted": document_count,
        "sources_deleted": source_count,
        "additional_entries_removed": additional_removed,
    }


__all__ = ["router"]

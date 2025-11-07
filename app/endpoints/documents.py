from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from datetime import datetime

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
)
from database import get_db
from models import Document, ResearchSource

router = APIRouter()


@router.get("/documents/stream")
async def documents_stream(request: Request, db: Session = Depends(get_db)):
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
                "documents": build_documents_snapshot(db),
            }
            yield f"data: {json.dumps(docs_payload, ensure_ascii=False)}\n\n"

            sources_payload = {
                "type": "sources_snapshot",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "initial",
                "sources": build_sources_snapshot(db),
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
async def get_documents(request: Request, db: Session = Depends(get_db)):
    """Get all classified documents grouped by category."""
    grouped = build_documents_snapshot(db)
    return JSONResponse(
        content=grouped,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.delete("/documents/{filename}")
@limiter.limit("100/hour")
async def delete_document(request: Request, filename: str, db: Session = Depends(get_db)):
    """Delete a classified document."""
    doc = db.query(Document).filter(Document.filename == filename).first()
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


@router.delete("/reset")
@limiter.limit("10/hour")
async def reset_application(request: Request, db: Session = Depends(get_db)):
    """Delete all stored data (documents, sources, files)."""
    documents = db.query(Document).all()
    sources = db.query(ResearchSource).all()

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
        db.query(Document).delete(synchronize_session=False)
        db.query(ResearchSource).delete(synchronize_session=False)
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

    uploads_cleared = clear_directory_contents(UPLOADS_DIR)
    downloads_cleared = clear_directory_contents(DOWNLOADS_DIR)
    ocr_text_cleared = clear_directory_contents(OCR_TEXT_DIR)

    broadcast_documents_snapshot(
        db,
        "reset",
        {
            "documents_deleted": document_count,
            "sources_deleted": source_count,
            "ocr_text_cleared": ocr_text_cleared,
        },
    )
    broadcast_sources_snapshot(
        db,
        "reset",
        {
            "documents_deleted": document_count,
            "sources_deleted": source_count,
            "uploads_entries_removed": uploads_cleared,
            "downloads_entries_removed": downloads_cleared,
        },
    )

    return {
        "message": "All stored data cleared successfully",
        "documents_deleted": document_count,
        "sources_deleted": source_count,
        "uploads_entries_removed": uploads_cleared,
        "downloads_entries_removed": downloads_cleared,
        "additional_entries_removed": additional_removed,
        "ocr_text_entries_removed": ocr_text_cleared,
    }


__all__ = ["router"]

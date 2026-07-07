import asyncio
import json
import mimetypes
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from sqlalchemy.orm import Session

from shared import (
    ANONYMIZED_TEXT_DIR,
    TRANSLATED_TEXT_DIR,
    DOWNLOADS_DIR,
    UPLOADS_DIR,
    OCR_TEXT_DIR,
    limiter,
    build_documents_snapshot,
    build_sources_snapshot,
    broadcast_documents_snapshot,
    broadcast_sources_snapshot,
    clear_directory_contents,
    AddDocumentFromUrlRequest,
    resolve_case_uuid_for_request,
    get_owned_document,
)
from auth import get_current_active_user, get_current_user
from database import get_db, SessionLocal
from events import SSETicketStore
from document_segmentation import (
    ensure_physical_document_segments,
    schedule_segment_child_post_processing,
    segment_document_with_qwen,
)
from models import (
    Document,
    DocumentSegment,
    DocumentTranslation,
    ResearchRun,
    ResearchSource,
    User,
    GeneratedDraft,
    RechtsprechungEntry,
)
from .research.utils import download_source_as_pdf

router = APIRouter()

# One-time tickets for authenticating SSE connections (EventSource cannot send an
# Authorization header). PROCESS-LOCAL store: there is a single app container, which
# is sufficient. A multi-process deployment would need a shared store (Redis/DB).
_sse_ticket_store = SSETicketStore(ttl_seconds=60)


async def _resolve_stream_user(request: Request, db: Session) -> User:
    """Authenticate an SSE connection via a one-time ticket (?ticket=...) or, as a
    fallback for curl/API clients, an Authorization: Bearer header.

    The old ?token=<JWT> query support has been removed so credentials no longer
    land in reverse-proxy access logs.
    """
    unauthorized = HTTPException(status_code=401, detail="Nicht authentifiziert")

    ticket = request.query_params.get("ticket")
    if ticket:
        user_id = _sse_ticket_store.consume(ticket)
        if not user_id:
            raise unauthorized
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise unauthorized
        return user

    # Header fallback for terminal/API usage (curl with a Bearer token).
    auth_header = request.headers.get("Authorization") or ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        user = await get_current_user(request=request, token=token, db=db)
        if not user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        return user

    raise unauthorized


def _download_text_filename(document: Document, suffix: str) -> str:
    stem = Path(document.filename or "document").stem or "document"
    safe_stem = re.sub(r"[\\/\x00-\x1f]+", "_", stem).strip(" ._") or "document"
    return f"{safe_stem}_{suffix}.txt"


def _safe_text_path(path: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_file():
        return None
    try:
        resolved = path_obj.resolve()
        resolved.relative_to(base_dir.resolve())
    except Exception:
        return None
    return resolved


def _gather_document_file_paths(document: Document, db: Session) -> list[Path]:
    """Collect all on-disk text files belonging to a document: extracted OCR text,
    anonymized text (from anonymization_metadata) and translations.

    Must be called BEFORE the document row is deleted from the DB: the DB cascade
    removes document_translations rows automatically, but not the files on disk,
    so their paths have to be read here while the rows still exist.

    Only paths under the known base directories are returned (safety guard via
    `_safe_text_path`). Missing files are simply not included (missing_ok semantics
    are then applied by the caller when unlinking).
    """
    paths: list[Path] = []

    extracted = _safe_text_path(document.extracted_text_path, OCR_TEXT_DIR)
    if extracted:
        paths.append(extracted)

    metadata = document.anonymization_metadata or {}
    anonymized = _safe_text_path(metadata.get("anonymized_text_path"), ANONYMIZED_TEXT_DIR)
    if anonymized:
        paths.append(anonymized)

    translations = (
        db.query(DocumentTranslation)
        .filter(DocumentTranslation.document_id == document.id)
        .all()
    )
    for translation in translations:
        translation_path = _safe_text_path(translation.text_path, TRANSLATED_TEXT_DIR)
        if translation_path:
            paths.append(translation_path)

    return paths


def _delete_document_files(document: Document, db: Session) -> None:
    """Delete all on-disk text files associated with a document (extracted,
    anonymized, translations). Missing files are skipped silently.

    Must be called BEFORE the document row is deleted (see `_gather_document_file_paths`).
    """
    for path_obj in _gather_document_file_paths(document, db):
        try:
            path_obj.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[WARN] Failed to delete file {path_obj} for {document.filename}: {exc}")


def _get_active_document_by_id(
    document_id: str,
    db: Session,
    current_user: User,
) -> Document:
    return get_owned_document(db, current_user, document_id)


def _segment_to_row(
    document: Document,
    current_user: User,
    segment: Dict[str, Any],
) -> DocumentSegment:
    return DocumentSegment(
        document_id=document.id,
        owner_id=current_user.id,
        case_id=document.case_id,
        start_page=int(segment.get("start_page") or 0),
        end_page=int(segment.get("end_page") or 0),
        document_type=segment.get("document_type"),
        title=segment.get("title"),
        date=segment.get("date"),
        sender_or_authority=segment.get("sender_or_authority"),
        addressee=segment.get("addressee"),
        topic=segment.get("topic"),
        confidence=segment.get("confidence"),
        boundary_reason=segment.get("boundary_reason"),
        model=segment.get("model") or "qwen3.6",
        metadata_={
            key: value
            for key, value in segment.items()
            if key
            not in {
                "start_page",
                "end_page",
                "document_type",
                "title",
                "date",
                "sender_or_authority",
                "addressee",
                "topic",
                "confidence",
                "boundary_reason",
                "model",
            }
        },
    )


@router.post("/documents/stream-ticket")
async def documents_stream_ticket(
    request: Request,
    current_user: User = Depends(get_current_active_user),
):
    """Issue a short-lived, one-time ticket for connecting to the SSE stream.

    Authenticated via the normal Bearer header. The frontend fetches a ticket and
    then opens EventSource('/documents/stream?ticket=...').
    """
    ticket = _sse_ticket_store.issue(str(current_user.id))
    return {"ticket": ticket, "expires_in": _sse_ticket_store.ttl_seconds}


@router.get("/documents/stream")
async def documents_stream(request: Request):
    """Unified SSE stream for all updates (documents, sources, etc.)."""
    hub = getattr(request.app.state, "document_hub", None)
    if not hub:
        raise HTTPException(status_code=503, detail="Event stream unavailable")

    # Resolve the user and build the initial snapshots with a short-lived session,
    # then CLOSE it before entering the streaming loop. Holding a session for the
    # whole SSE lifetime pins a pool connection (idle in transaction) for as long
    # as the client stays connected, which exhausts the pool (10+20 -> ~30 conns).
    session = SessionLocal()
    try:
        current_user = await _resolve_stream_user(request, session)
        user_id = current_user.id
        docs_snapshot = build_documents_snapshot(
            session, current_user.id, current_user.active_case_id
        )
        sources_snapshot = build_sources_snapshot(
            session, current_user.id, current_user.active_case_id
        )
    finally:
        session.close()

    # Subscribe scoped to this user so we only receive this owner's events plus
    # ownerless system events (e.g. resync).
    queue = await hub.subscribe(str(user_id))

    async def event_generator():
        try:
            docs_payload = {
                "type": "documents_snapshot",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "initial",
                "documents": docs_snapshot,
            }
            yield f"data: {json.dumps(docs_payload, ensure_ascii=False)}\n\n"

            sources_payload = {
                "type": "sources_snapshot",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "initial",
                "sources": sources_snapshot,
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
@limiter.limit("1000/hour")
async def get_documents(
    request: Request,
    case_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all classified documents grouped by category."""
    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    grouped = build_documents_snapshot(db, current_user.id, target_case_id)
    return JSONResponse(
        content=grouped,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/documents/{document_id}/ocr-text/download")
@limiter.limit("100/hour")
async def download_document_ocr_text(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Download the cached OCR text for a document in the active case."""
    document = _get_active_document_by_id(document_id, db, current_user)
    path_obj = _safe_text_path(document.extracted_text_path, OCR_TEXT_DIR)
    if not path_obj:
        fallback_path = OCR_TEXT_DIR / f"{document.id}.txt"
        path_obj = _safe_text_path(str(fallback_path), OCR_TEXT_DIR)
    if not path_obj:
        raise HTTPException(status_code=404, detail="OCR text not found")

    return FileResponse(
        path_obj,
        media_type="text/plain",
        filename=_download_text_filename(document, "ocr"),
    )


@router.get("/documents/{document_id}/anonymized-text/download")
@limiter.limit("100/hour")
async def download_document_anonymized_text(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Download the cached anonymized text for a document in the active case."""
    document = _get_active_document_by_id(document_id, db, current_user)
    metadata = document.anonymization_metadata or {}
    path_obj = None
    if document.is_anonymized:
        path_obj = _safe_text_path(metadata.get("anonymized_text_path"), ANONYMIZED_TEXT_DIR)
        if not path_obj:
            fallback_path = ANONYMIZED_TEXT_DIR / f"{document.id}.txt"
            path_obj = _safe_text_path(str(fallback_path), ANONYMIZED_TEXT_DIR)
    if not path_obj:
        raise HTTPException(status_code=404, detail="Anonymized text not found")

    return FileResponse(
        path_obj,
        media_type="text/plain",
        filename=_download_text_filename(document, "anonymisiert"),
    )


@router.get("/documents/{document_id}/segments")
@limiter.limit("1000/hour")
async def get_document_segments(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return stored logical segments for a document in the active case."""
    document = _get_active_document_by_id(document_id, db, current_user)
    segments = (
        db.query(DocumentSegment)
        .filter(
            DocumentSegment.document_id == document.id,
            DocumentSegment.owner_id == current_user.id,
            DocumentSegment.case_id == document.case_id,
        )
        .order_by(DocumentSegment.start_page.asc(), DocumentSegment.end_page.asc())
        .all()
    )
    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "segments": [segment.to_dict() for segment in segments],
    }


@router.post("/documents/{document_id}/segment")
@limiter.limit("30/hour")
async def segment_document(
    request: Request,
    document_id: str,
    force: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Segment a page-preserved OCR document into logical source units using Qwen."""
    document = _get_active_document_by_id(document_id, db, current_user)
    existing = (
        db.query(DocumentSegment)
        .filter(
            DocumentSegment.document_id == document.id,
            DocumentSegment.owner_id == current_user.id,
            DocumentSegment.case_id == document.case_id,
        )
        .order_by(DocumentSegment.start_page.asc(), DocumentSegment.end_page.asc())
        .all()
    )
    if existing and not force:
        try:
            created_documents = ensure_physical_document_segments(
                document,
                existing,
                db,
                current_user,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        for segment in existing:
            db.refresh(segment)
        broadcast_documents_snapshot(
            db,
            "document_segmented",
            {"filename": document.filename, "created_documents": len(created_documents)},
            owner_id=current_user.id,
        )
        schedule_segment_child_post_processing(created_documents, current_user)
        return {
            "status": "cached",
            "document_id": str(document.id),
            "filename": document.filename,
            "segments": [segment.to_dict() for segment in existing],
            "created_documents": created_documents,
        }

    try:
        result = await segment_document_with_qwen(document)
    except Exception as exc:
        print(f"[SEGMENTATION ERROR] {document.filename}: {exc}")
        raise HTTPException(status_code=503, detail=f"Segmentation failed: {exc}")

    raw_segments = result.get("segments") or []
    if force and existing and not raw_segments and not result.get("skipped"):
        try:
            created_documents = ensure_physical_document_segments(
                document,
                existing,
                db,
                current_user,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        for segment in existing:
            db.refresh(segment)
        schedule_segment_child_post_processing(created_documents, current_user)
        return {
            "status": "existing_retained",
            "document_id": str(document.id),
            "filename": document.filename,
            "page_count": result.get("page_count"),
            "model": result.get("model"),
            "reason": "Segmentation returned no usable segments; existing rows were kept.",
            "segments": [segment.to_dict() for segment in existing],
            "created_documents": created_documents,
        }

    if force and existing:
        for row in existing:
            db.delete(row)
        db.commit()

    rows = []
    for segment in raw_segments:
        segment = dict(segment)
        segment.setdefault("model", result.get("model"))
        row = _segment_to_row(document, current_user, segment)
        db.add(row)
        rows.append(row)
    db.commit()
    for row in rows:
        db.refresh(row)

    try:
        created_documents = ensure_physical_document_segments(
            document,
            rows,
            db,
            current_user,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    for row in rows:
        db.refresh(row)

    broadcast_documents_snapshot(
        db,
        "document_segmented",
        {"filename": document.filename, "created_documents": len(created_documents)},
        owner_id=current_user.id,
    )
    schedule_segment_child_post_processing(created_documents, current_user)
    return {
        "status": "success" if rows else ("skipped" if result.get("skipped") else "empty"),
        "document_id": str(document.id),
        "filename": document.filename,
        "page_count": result.get("page_count"),
        "image_count": result.get("image_count"),
        "model": result.get("model"),
        "segmentation_source": result.get("segmentation_source"),
        "skipped": bool(result.get("skipped")),
        "reason": result.get("reason"),
        "segments": [row.to_dict() for row in rows],
        "created_documents": created_documents,
    }


@router.get("/documents/{filename}")
@limiter.limit("100/hour")
async def get_document_file(
    request: Request,
    filename: str,
    case_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Serve a specific document file."""
    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    doc = (
        db.query(Document)
        .filter(
            Document.filename == filename,
            Document.owner_id == current_user.id,
            Document.case_id == target_case_id,
        )
        .first()
    )
    
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
    case_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a classified document."""
    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    doc = (
        db.query(Document)
        .filter(
            Document.filename == filename,
            Document.owner_id == current_user.id,
            Document.case_id == target_case_id,
        )
        .first()
    )
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.file_path:
        try:
            fp = Path(doc.file_path)
            if fp.exists():
                fp.unlink()
        except Exception as exc:
            print(f"Error deleting uploaded file for {filename}: {exc}")

    _delete_document_files(doc, db)

    db.query(GeneratedDraft).filter(
        GeneratedDraft.primary_document_id == doc.id
    ).delete(synchronize_session=False)

    db.delete(doc)
    db.commit()
    broadcast_documents_snapshot(db, "delete", {"filename": filename}, owner_id=current_user.id)
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
    target_case_id = resolve_case_uuid_for_request(db, current_user, body.case_id)
    
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
    while (UPLOADS_DIR / safe_filename).exists() or db.query(Document).filter(
        Document.filename == safe_filename,
        Document.owner_id == current_user.id,
        Document.case_id == target_case_id,
    ).first():
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        safe_filename = f"{stem}_{counter}{suffix}"
        counter += 1
        
    final_path = UPLOADS_DIR / safe_filename
    
    # 3. Move file
    try:
        try:
            shutil.move(str(temp_path), str(final_path))
        except Exception as e:
            print(f"Error moving file: {e}")
            # Try copy if move fails
            shutil.copy(str(temp_path), str(final_path))
    finally:
        # Clean up the temp download regardless of whether move/copy succeeded,
        # so a failed copy never leaves an orphaned file in DOWNLOADS_DIR.
        temp_path.unlink(missing_ok=True)

    # 4. Create Document record
    new_doc = Document(
        filename=safe_filename,
        category=body.category,
        confidence=1.0, # Manually added, high confidence
        explanation=f"Importiert von URL: {body.url}",
        file_path=str(final_path),
        processing_status="pending",
        owner_id=current_user.id,
        case_id=target_case_id,
        needs_ocr=True # Assume needs OCR usually
    )
    
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    
    broadcast_documents_snapshot(db, "add_source", {"filename": safe_filename}, owner_id=current_user.id)
    
    return new_doc.to_dict()


@router.delete("/reset")
@limiter.limit("10/hour")
async def reset_application(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete all stored data (documents, sources, files) for the current user, preserving playbook entries."""
    playbook_doc_ids = {
        row[0]
        for row in db.query(RechtsprechungEntry.document_id)
        .filter(RechtsprechungEntry.document_id.isnot(None))
        .all()
    }

    documents_query = db.query(Document).filter(
        Document.owner_id == current_user.id,
        Document.case_id == current_user.active_case_id,
    )
    if playbook_doc_ids:
        documents_query = documents_query.filter(~Document.id.in_(playbook_doc_ids))
    documents = documents_query.all()
    sources = db.query(ResearchSource).filter(
        ResearchSource.owner_id == current_user.id,
        ResearchSource.case_id == current_user.active_case_id,
    ).all()

    document_paths = [doc.file_path for doc in documents if doc.file_path]
    segment_dirs: set[Path] = set()
    # Must be gathered BEFORE the documents are deleted below: the DB cascade removes
    # document_translations rows, but not the files on disk, so their paths have to be
    # read from the still-existing rows here.
    cleanup_paths: list[Path] = []
    for doc in documents:
        file_path = doc.file_path
        if file_path:
            path_obj = Path(file_path)
            if path_obj.exists():
                document_paths.append(str(path_obj))
            if path_obj.parent.name.endswith("_segments"):
                segment_dirs.add(path_obj.parent)
        cleanup_paths.extend(_gather_document_file_paths(doc, db))

    document_count = len(documents)
    source_count = len(sources)

    try:
        # Delete generated drafts first to satisfy foreign key constraints
        db.query(GeneratedDraft).filter(
            GeneratedDraft.user_id == current_user.id,
            GeneratedDraft.case_id == current_user.active_case_id,
        ).delete(synchronize_session=False)
        
        # Also delete any drafts referencing the documents we are about to delete
        # (This catches cases where user_id might be missing but the link exists)
        doc_ids = [d.id for d in documents]
        if doc_ids:
            db.query(GeneratedDraft).filter(GeneratedDraft.primary_document_id.in_(doc_ids)).delete(synchronize_session=False)

        if doc_ids:
            db.query(Document).filter(Document.id.in_(doc_ids)).delete(synchronize_session=False)
        db.query(ResearchRun).filter(
            ResearchRun.owner_id == current_user.id,
            ResearchRun.case_id == current_user.active_case_id,
        ).delete(synchronize_session=False)

        db.query(ResearchSource).filter(
            ResearchSource.owner_id == current_user.id,
            ResearchSource.case_id == current_user.active_case_id,
        ).delete(synchronize_session=False)
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

    for path_obj in cleanup_paths:
        try:
            path_obj.unlink(missing_ok=True)
            additional_removed += 1
        except Exception as exc:
            print(f"Error deleting document text file {path_obj}: {exc}")

    # Note: We do NOT clear the global directories (UPLOADS_DIR, etc.) because they might contain other users' files.
    # We only delete the specific files we tracked.
    
    broadcast_documents_snapshot(
        db,
        "reset",
        {
            "documents_deleted": document_count,
            "sources_deleted": source_count,
        },
        owner_id=current_user.id,
    )
    broadcast_sources_snapshot(
        db,
        "reset",
        {
            "documents_deleted": document_count,
            "sources_deleted": source_count,
        },
        owner_id=current_user.id,
    )

    return {
        "message": "All stored data cleared successfully",
        "documents_deleted": document_count,
        "sources_deleted": source_count,
        "additional_entries_removed": additional_removed,
    }


__all__ = ["router"]

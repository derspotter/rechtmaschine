import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import get_db
from models import Case, Document, GeneratedDraft, ResearchSource, User
from shared import limiter
from shared import (
    ANONYMIZED_TEXT_DIR,
    broadcast_documents_snapshot,
    broadcast_sources_snapshot,
    delete_document_text,
)

router = APIRouter(tags=["cases"])


class CaseCreateRequest(BaseModel):
    name: Optional[str] = Field(default=None)
    state: Optional[Dict[str, Any]] = Field(default=None)


class CaseRenameRequest(BaseModel):
    name: str


class CaseStateRequest(BaseModel):
    state: Dict[str, Any]


def _case_to_dict(case: Case) -> Dict[str, Any]:
    return {
        "id": str(case.id),
        "name": case.name or "",
        "archived": bool(case.archived),
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "updated_at": case.updated_at.isoformat() if case.updated_at else None,
    }


@router.get("/cases")
@limiter.limit("200/hour")
async def list_cases(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    cases = (
        db.query(Case)
        .filter(Case.owner_id == current_user.id)
        .order_by(desc(Case.updated_at), desc(Case.created_at))
        .all()
    )
    return JSONResponse(
        content={
            "active_case_id": str(current_user.active_case_id) if current_user.active_case_id else None,
            "cases": [_case_to_dict(c) for c in cases],
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.post("/cases")
@limiter.limit("50/hour")
async def create_case(
    request: Request,
    body: CaseCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    case = Case(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        name=(body.name or "").strip() or None,
        state=body.state or None,
        archived=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return {"case": _case_to_dict(case)}


@router.post("/cases/{case_id}/activate")
@limiter.limit("200/hour")
async def activate_case(
    request: Request,
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")

    case = (
        db.query(Case)
        .filter(Case.id == case_uuid, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    current_user.active_case_id = case.id
    case.updated_at = datetime.utcnow()
    db.commit()
    return {"active_case_id": str(case.id)}


@router.patch("/cases/{case_id}")
@limiter.limit("100/hour")
async def rename_case(
    request: Request,
    case_id: str,
    body: CaseRenameRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")

    case = (
        db.query(Case)
        .filter(Case.id == case_uuid, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    new_name = (body.name or "").strip()
    if not new_name:
        raise HTTPException(status_code=422, detail="Case name cannot be empty")

    case.name = new_name
    case.updated_at = datetime.utcnow()
    db.commit()
    return {"case": _case_to_dict(case)}


@router.get("/cases/{case_id}/state")
@limiter.limit("500/hour")
async def get_case_state(
    request: Request,
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")

    case = (
        db.query(Case)
        .filter(Case.id == case_uuid, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return {"state": case.state or {}}


@router.put("/cases/{case_id}/state")
@limiter.limit("500/hour")
async def put_case_state(
    request: Request,
    case_id: str,
    body: CaseStateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")

    case = (
        db.query(Case)
        .filter(Case.id == case_uuid, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    case.state = body.state or {}
    case.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


@router.delete("/cases/{case_id}")
@limiter.limit("50/hour")
async def delete_case(
    request: Request,
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete a case and all its scoped data (documents, sources, drafts).
    Playbook entries are preserved; FK to documents is ON DELETE SET NULL.
    """
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")

    case = (
        db.query(Case)
        .filter(Case.id == case_uuid, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Load scoped rows
    documents = (
        db.query(Document)
        .filter(Document.owner_id == current_user.id, Document.case_id == case_uuid)
        .all()
    )
    sources = (
        db.query(ResearchSource)
        .filter(ResearchSource.owner_id == current_user.id, ResearchSource.case_id == case_uuid)
        .all()
    )

    document_paths: list[str] = []
    segment_dirs: set[Path] = set()
    anonymized_paths: list[str] = []

    for doc in documents:
        if doc.file_path:
            document_paths.append(doc.file_path)
            try:
                path_obj = Path(doc.file_path)
                if path_obj.parent.name.endswith("_segments"):
                    segment_dirs.add(path_obj.parent)
            except Exception:
                pass

        if doc.extracted_text_path:
            document_paths.append(doc.extracted_text_path)

        # OCR text is stored separately by document id
        try:
            delete_document_text(doc)
        except Exception:
            pass

        # Anonymized text file
        anon_path = None
        try:
            meta = doc.anonymization_metadata or {}
            anon_path = meta.get("anonymized_text_path")
        except Exception:
            anon_path = None
        if not anon_path:
            anon_path = str(ANONYMIZED_TEXT_DIR / f"{doc.id}.txt")
        anonymized_paths.append(anon_path)

    # Remove downloaded source files
    source_paths = [s.download_path for s in sources if s.download_path]

    # Delete DB rows first
    try:
        db.query(GeneratedDraft).filter(
            GeneratedDraft.user_id == current_user.id,
            GeneratedDraft.case_id == case_uuid,
        ).delete(synchronize_session=False)

        db.query(ResearchSource).filter(
            ResearchSource.owner_id == current_user.id,
            ResearchSource.case_id == case_uuid,
        ).delete(synchronize_session=False)

        db.query(Document).filter(
            Document.owner_id == current_user.id,
            Document.case_id == case_uuid,
        ).delete(synchronize_session=False)

        db.delete(case)

        # If this was the active case, choose another (or create a new default).
        was_active = bool(getattr(current_user, "active_case_id", None) == case_uuid)
        if was_active:
            replacement = (
                db.query(Case)
                .filter(Case.owner_id == current_user.id)
                .order_by(desc(Case.updated_at), desc(Case.created_at))
                .first()
            )
            if not replacement:
                replacement = Case(
                    id=uuid.uuid4(),
                    owner_id=current_user.id,
                    name="Fall 1",
                    state={},
                    archived=False,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(replacement)
            current_user.active_case_id = replacement.id

        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete case: {exc}")

    # Delete files on disk
    removed_files = 0
    removed_dirs = 0

    for raw_path in list(document_paths) + list(source_paths) + list(anonymized_paths):
        if not raw_path:
            continue
        try:
            p = Path(raw_path)
            if str(p).startswith("/app/") and p.exists() and p.is_file():
                p.unlink()
                removed_files += 1
        except Exception as exc:
            print(f"[WARN] Failed to remove file {raw_path}: {exc}")

    for d in segment_dirs:
        try:
            if str(d).startswith("/app/") and d.exists() and d.is_dir():
                shutil.rmtree(d)
                removed_dirs += 1
        except Exception as exc:
            print(f"[WARN] Failed to remove segment dir {d}: {exc}")

    broadcast_documents_snapshot(db, "case_deleted", {"case_id": case_id})
    broadcast_sources_snapshot(db, "case_deleted", {"case_id": case_id})

    return {
        "ok": True,
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
    }


__all__ = ["router"]

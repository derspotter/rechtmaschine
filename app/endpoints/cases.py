import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import get_db
from models import Case, User
from shared import limiter

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


__all__ = ["router"]


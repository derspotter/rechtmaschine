from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Dict, Any, Optional
import uuid

from database import get_db
from models import GeneratedDraft, User
from auth import get_current_active_user

router = APIRouter(prefix="/drafts", tags=["drafts"])

@router.get("/", response_model=Dict[str, Any])
async def list_drafts(
    limit: int = Query(50, ge=1, le=100),
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List recent drafts for the current user."""
    
    # Query drafts for current user
    # Note: If we haven't strictly enforced user_id on all old data, this might need adjustment,
    # but for new drafts we will enforce it.
    
    query = (
        db.query(GeneratedDraft)
        .filter(
            (GeneratedDraft.user_id == current_user.id) | (GeneratedDraft.user_id == None),  # noqa: E711
            GeneratedDraft.case_id == current_user.active_case_id,
        )
        .order_by(desc(GeneratedDraft.created_at))
    )
    
    drafts = query.offset(offset).limit(limit).all()
    
    return JSONResponse(
        content={
            "total": len(drafts),
            "items": [
                {
                    "id": str(d.id),
                    "document_type": d.document_type,
                    "created_at": d.created_at.isoformat(),
                    "model_used": d.model_used,
                    "preview": d.generated_text[:200] + "..." if d.generated_text else "",
                    "user_prompt": d.user_prompt[:100] + "..." if d.user_prompt else ""
                }
                for d in drafts
            ]
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@router.get("/{draft_id}", response_model=Dict[str, Any])
async def get_draft(
    draft_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get full details of a specific draft."""
    draft = db.query(GeneratedDraft).filter(GeneratedDraft.id == draft_id).first()
    
    if not draft:
        raise HTTPException(status_code=404, detail="Entwurf nicht gefunden")
        
    # Optional: Check ownership
    if draft.user_id and draft.user_id != current_user.id:
        # Allow if admin or specific rules, but strictly:
        # For now, allow simplified access or verify ownership
        if str(current_user.email) != "admin@example.com": # Simple admin check example
             raise HTTPException(status_code=403, detail="Keine Berechtigung")

    if draft.case_id and draft.case_id != current_user.active_case_id:
        raise HTTPException(status_code=403, detail="Entwurf gehört zu einem anderen Fall")

    return draft.to_dict()

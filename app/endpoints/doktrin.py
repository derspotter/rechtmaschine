"""Read-only observability for the Doktrin layer (wiki.aufentha.lt mirror).

Sync itself is not HTTP-triggered — it runs via
`docker exec rechtmaschine-app python /app/doktrin_sync.py` (nightly
doktrin-sync.timer). These endpoints only expose sync state and page
bookkeeping for the CLI and debugging.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import get_db
from models import DoktrinPage, User
from shared import limiter

router = APIRouter(prefix="/doktrin", tags=["doktrin"])


@router.get("/status")
@limiter.limit("300/hour")
async def doktrin_status(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    by_status = dict(
        db.query(DoktrinPage.status, func.count()).group_by(DoktrinPage.status).all()
    )
    chunks_total = (
        db.query(func.coalesce(func.sum(DoktrinPage.chunk_count), 0))
        .filter(DoktrinPage.status == "active")
        .scalar()
    )
    last_synced = db.query(func.max(DoktrinPage.last_synced_at)).scalar()
    last_changed = db.query(func.max(DoktrinPage.last_changed_at)).scalar()
    return {
        "pages_by_status": by_status,
        "chunks_total": int(chunks_total or 0),
        "last_synced_at": last_synced.isoformat() if last_synced else None,
        "last_changed_at": last_changed.isoformat() if last_changed else None,
    }


@router.get("/pages")
@limiter.limit("300/hour")
async def doktrin_pages(
    request: Request,
    status: Optional[str] = Query(default=None),
    country: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    query = db.query(DoktrinPage)
    if status:
        query = query.filter(DoktrinPage.status == status)
    if country:
        query = query.filter(func.lower(DoktrinPage.country) == country.lower())
    if q:
        needle = f"%{q.lower()}%"
        query = query.filter(
            or_(
                func.lower(DoktrinPage.page_id).like(needle),
                func.lower(DoktrinPage.title).like(needle),
            )
        )
    rows = query.order_by(DoktrinPage.page_id).limit(limit).all()
    return {"pages": [row.to_dict() for row in rows]}


@router.get("/pages/{page_id:path}")
@limiter.limit("300/hour")
async def doktrin_page_detail(
    request: Request,
    page_id: str,
    include_text: bool = Query(default=False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = db.query(DoktrinPage).filter(DoktrinPage.page_id == page_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Doktrin page not found")
    return {"page": row.to_dict(include_text=include_text)}

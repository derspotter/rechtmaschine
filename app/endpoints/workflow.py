import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import get_db
from draft_context import assemble_draft_context, verify_facts_with_sources
from models import Case, GeneratedDraft, User
from shared import (
    JLawyerResponse,
    JLawyerSendDraftRequest,
    WorkflowInventoryResponse,
    WorkflowJLawyerCaseResolveResponse,
    build_documents_snapshot,
    build_sources_snapshot,
    limiter,
    resolve_case_uuid_for_request,
)
from .generation import (
    _is_jlawyer_configured,
    _jlawyer_api_base_url,
    _markdown_to_plain_text,
    _normalize_jlawyer_output_file_name,
    _rag_block_for_generation,
    _resolve_jlawyer_case_id,
    JLAWYER_PASSWORD,
    JLAWYER_PLACEHOLDER_KEY,
    JLAWYER_TEMPLATE_FOLDER_DEFAULT,
    JLAWYER_USERNAME,
    NEUTRAL_LEGAL_TONE_RULES,
)

try:
    from legal_context import build_statute_block
except ImportError:
    build_statute_block = None

try:
    from agent_memory_service import get_case_memory_prompt_context
except ImportError:
    get_case_memory_prompt_context = None

import httpx
from urllib.parse import quote

router = APIRouter(prefix="/workflow", tags=["workflow"])


class DraftContextRequest(BaseModel):
    query: str
    case_id: Optional[str] = None
    rag_limit: Optional[int] = None


class VerifyFactsRequest(BaseModel):
    text: str
    case_id: Optional[str] = None
    sources: List[str] = []


@router.get("/inventory", response_model=WorkflowInventoryResponse)
@limiter.limit("240/hour")
async def get_case_inventory(
    request: Request,
    case_id: Optional[str] = Query(default=None),
    draft_limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return a case-scoped inventory for CLI clients.

    This consolidates the common multi-call workflow of listing documents,
    sources, and recent drafts for a case into one endpoint.
    """
    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)

    drafts = (
        db.query(GeneratedDraft)
        .filter(
            GeneratedDraft.user_id == current_user.id,
            GeneratedDraft.case_id == target_case_id,
        )
        .order_by(desc(GeneratedDraft.created_at))
        .limit(draft_limit)
        .all()
    )

    return WorkflowInventoryResponse(
        case_id=str(target_case_id) if target_case_id else None,
        documents=build_documents_snapshot(db, current_user.id, target_case_id),
        sources=build_sources_snapshot(db, current_user.id, target_case_id),
        drafts=[
            {
                "id": str(draft.id),
                "document_type": draft.document_type,
                "model_used": draft.model_used,
                "resolved_legal_area": (draft.metadata_ or {}).get("resolved_legal_area"),
                "created_at": draft.created_at.isoformat() if draft.created_at else None,
                "primary_document_id": str(draft.primary_document_id) if draft.primary_document_id else None,
                "preview": draft.generated_text[:200] + "..." if draft.generated_text else "",
                "user_prompt": draft.user_prompt or "",
            }
            for draft in drafts
        ],
    )


@router.get("/jlawyer/resolve-case", response_model=WorkflowJLawyerCaseResolveResponse)
@limiter.limit("60/hour")
async def resolve_jlawyer_case_reference(
    request: Request,
    reference: str = Query(..., min_length=1),
    current_user: User = Depends(get_current_active_user),
):
    """Resolve a j-lawyer case reference or file number to the internal case id."""
    if not _is_jlawyer_configured():
        raise HTTPException(status_code=503, detail="j-lawyer Integration ist nicht konfiguriert")

    auth = (JLAWYER_USERNAME, JLAWYER_PASSWORD)
    resolved_case_id = await _resolve_jlawyer_case_id(reference, auth)
    return WorkflowJLawyerCaseResolveResponse(
        requested_reference=reference,
        resolved_case_id=resolved_case_id,
    )


@router.post("/jlawyer/send-draft", response_model=JLawyerResponse)
@limiter.limit("20/hour")
async def send_saved_draft_to_jlawyer(
    request: Request,
    body: JLawyerSendDraftRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Send an already saved draft directly to j-lawyer by draft id."""
    if not _is_jlawyer_configured():
        raise HTTPException(status_code=503, detail="j-lawyer Integration ist nicht konfiguriert")

    try:
        draft_uuid = uuid.UUID(body.draft_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Ungültige draft_id")

    draft = (
        db.query(GeneratedDraft)
        .filter(GeneratedDraft.id == draft_uuid)
        .first()
    )
    if not draft:
        raise HTTPException(status_code=404, detail="Entwurf nicht gefunden")

    if draft.user_id:
        if draft.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Keine Berechtigung")
    else:
        if not draft.case_id:
            raise HTTPException(status_code=403, detail="Keine Berechtigung")
        owning_case = (
            db.query(Case)
            .filter(
                Case.id == draft.case_id,
                Case.owner_id == current_user.id,
            )
            .first()
        )
        if not owning_case:
            raise HTTPException(status_code=403, detail="Keine Berechtigung")

    case_reference = body.case_reference.strip()
    template_name = body.template_name.strip()
    file_name = _normalize_jlawyer_output_file_name(body.file_name)
    template_folder = (body.template_folder or JLAWYER_TEMPLATE_FOLDER_DEFAULT or "").strip()

    if not case_reference or not template_name or not file_name:
        raise HTTPException(status_code=400, detail="draft_id, case_reference, template_name und file_name sind Pflichtfelder")
    if not template_folder:
        raise HTTPException(status_code=400, detail="Kein Template-Ordner konfiguriert")

    auth = (JLAWYER_USERNAME, JLAWYER_PASSWORD)
    resolved_case_id = await _resolve_jlawyer_case_id(case_reference, auth)
    payload = [
        {
            "placeHolderKey": JLAWYER_PLACEHOLDER_KEY,
            "placeHolderValue": _markdown_to_plain_text(draft.generated_text or ""),
        }
    ]

    url = (
        f"{_jlawyer_api_base_url()}/v6/templates/documents/"
        f"{quote(template_folder, safe='')}/"
        f"{quote(template_name, safe='')}/"
        f"{quote(resolved_case_id, safe='')}/"
        f"{quote(file_name, safe='')}"
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(url, auth=auth, json=payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"j-lawyer Anfrage fehlgeschlagen: {exc}")

    if response.status_code >= 400:
        detail = response.text or response.reason_phrase or "Unbekannter Fehler"
        raise HTTPException(status_code=502, detail=f"j-lawyer Fehler ({response.status_code}): {detail}")

    response_payload = None
    created_document_id = None
    try:
        parsed = response.json()
        if isinstance(parsed, dict):
            response_payload = parsed
            created_document_id = (
                str(parsed.get("id") or parsed.get("documentId") or parsed.get("docId") or "").strip()
                or None
            )
    except ValueError:
        response_payload = None

    # Sending to j-lawyer is the de-facto acceptance of a draft (same rationale as
    # /send-to-jlawyer in generation.py, which this endpoint previously lacked):
    # 1. reflect the accepted draft into the case memory, and
    # 2. distill the case into anonymized Muster-Wiki pattern entries (review-gated,
    #    they land "pending") - a finished work product is the natural distillation
    #    moment, and enqueue_memory_reflection dedupes concurrent runs per case.
    try:
        from agent_memory_service import enqueue_memory_reflection

        if draft.case_id:
            reflect_owner = draft.user_id or current_user.id
            enqueue_memory_reflection(
                db,
                reflect_owner,
                draft.case_id,
                trigger="draft",
                draft_id=str(draft.id),
            )
            enqueue_memory_reflection(
                db,
                reflect_owner,
                draft.case_id,
                trigger="pattern_wiki",
            )
    except Exception as exc:
        print(f"[MEMORY WARN] Reflection enqueue after j-lawyer send-draft failed: {exc}")

    return JLawyerResponse(
        success=True,
        message="Gespeicherter Entwurf erfolgreich an j-lawyer gesendet",
        requested_case_reference=case_reference,
        resolved_case_id=resolved_case_id,
        template_folder=template_folder,
        template_name=template_name,
        file_name=file_name,
        created_document_id=created_document_id,
        jlawyer_response=response_payload,
    )


@router.post("/draft-context")
@limiter.limit("240/hour")
async def workflow_draft_context(
    request: Request,
    body: DraftContextRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """The /generate context blocks (RAG precedent, statutes, case memory,
    style rules) as JSON, for terminal-side drafting. Blocks degrade to ''."""
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="query must be non-empty")
    grounding: dict = {}

    def _memory_block() -> str:
        if not (body.case_id and get_case_memory_prompt_context):
            return ""
        text = get_case_memory_prompt_context(
            db, current_user, body.case_id, collect=grounding
        )
        return f"KOMPAKTES FALLGEDÄCHTNIS:\n{text}\n\n" if text else ""

    payload = assemble_draft_context(
        body.query,
        rag_block_fn=lambda: _rag_block_for_generation(
            db, current_user, body.case_id, body.query,
            collect=grounding, limit=body.rag_limit,
        ),
        statute_block_fn=lambda: (
            build_statute_block(body.query) if build_statute_block else ""
        ),
        memory_block_fn=_memory_block,
        style_rules=NEUTRAL_LEGAL_TONE_RULES,
    )
    payload["grounding"] = grounding
    return payload


@router.post("/verify-facts")
@limiter.limit("240/hour")
async def workflow_verify_facts(
    request: Request,
    body: VerifyFactsRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Deterministic date/Aktenzeichen/amount check of a draft against case
    memory + provided source texts (citation_verifier.verify_facts)."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    memory_text = ""
    if body.case_id and get_case_memory_prompt_context:
        try:
            memory_text = get_case_memory_prompt_context(db, current_user, body.case_id)
        except Exception as exc:  # noqa: BLE001 — memory absence must not block the check
            print(f"[WARN] verify-facts memory load failed: {exc}")
    result = verify_facts_with_sources(body.text, memory_text, body.sources)
    result["corpus_empty"] = not (
        memory_text.strip() or any(s.strip() for s in body.sources)
    )
    return result

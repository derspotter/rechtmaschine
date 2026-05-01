import asyncio
import json
import os
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from google.genai import types
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agent_memory_service import (
    BRIEF_TARGET,
    STRATEGY_TARGET,
    accept_memory_update_proposal,
    create_memory_update_proposal,
    default_case_brief_json,
    default_case_strategy_json,
    get_or_create_case_brief,
    get_or_create_case_strategy,
    list_memory_update_proposals,
    memory_row_to_dict,
    proposal_to_dict,
    reject_memory_update_proposal,
    render_case_brief_compact,
    render_case_strategy_compact,
    update_case_brief_manual,
    update_case_strategy_manual,
)
from auth import get_current_active_user
from database import get_db
from models import Case, Document, ResearchSource, User
from shared import (
    collect_selected_document_identifiers,
    get_gemini_client,
    load_document_text,
    MemoryPatchOperation,
    MemorySourceRef,
    SelectedDocuments,
    limiter,
    resolve_document_identifier,
    resolve_case_uuid_for_request,
)

router = APIRouter(prefix="/memory", tags=["memory"])
MEMORY_EXTRACTION_MODEL = (
    os.getenv("MEMORY_EXTRACTION_MODEL", "gemini-3-flash-preview").strip()
    or "gemini-3-flash-preview"
)
MAX_MEMORY_SOURCE_CHARS = int(
    (os.getenv("MAX_MEMORY_SOURCE_CHARS", "60000") or "60000").strip()
)


class CaseMemoryCombinedRequest(BaseModel):
    overview: str = ""
    strategy: str = ""


class ProposalFromSelectionRequest(BaseModel):
    selected_documents: SelectedDocuments


class MemoryProposalCreateRequest(BaseModel):
    target_type: str
    case_id: Optional[str] = None
    target_id: Optional[str] = None
    expected_version: int
    ops: list[MemoryPatchOperation] = Field(default_factory=list)
    source_refs: list[MemorySourceRef] = Field(default_factory=list)
    confidence: Optional[float] = None
    model: Optional[str] = None


class CaseMemoryExtractionResult(BaseModel):
    beteiligte: list[str] = Field(default_factory=list)
    verfahrensstand: list[str] = Field(default_factory=list)
    sachverhalt: list[str] = Field(default_factory=list)
    antraege_ziele: list[str] = Field(default_factory=list)
    streitige_punkte: list[str] = Field(default_factory=list)
    beweismittel: list[str] = Field(default_factory=list)
    risiken: list[str] = Field(default_factory=list)
    offene_fragen_fall: list[str] = Field(default_factory=list)
    fall_notizen: str = ""
    kernstrategie: str = ""
    argumentationslinien: list[str] = Field(default_factory=list)
    rechtliche_ansatzpunkte: list[str] = Field(default_factory=list)
    beweisstrategie: list[str] = Field(default_factory=list)
    prozessuale_schritte: list[str] = Field(default_factory=list)
    vergleich_oder_taktik: list[str] = Field(default_factory=list)
    risiken_und_gegenargumente: list[str] = Field(default_factory=list)
    offene_fragen_strategie: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    warnings: list[str] = Field(default_factory=list)


def _assert_owned_case(db: Session, current_user: User, case_id: str):
    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    case = (
        db.query(Case)
        .filter(Case.id == target_case_id, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return target_case_id


def _squash_text(value: str, limit: int = 1200) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    return cleaned[:limit]


def _selected_source_material(
    db: Session,
    current_user: User,
    case_id: Any,
    selection: SelectedDocuments,
) -> tuple[str, list[MemorySourceRef]]:
    chunks: list[str] = []
    source_refs: list[MemorySourceRef] = []
    remaining_chars = MAX_MEMORY_SOURCE_CHARS
    seen_documents: set[str] = set()

    for identifier in collect_selected_document_identifiers(selection):
        document = resolve_document_identifier(db, current_user, case_id, identifier)
        if not document or str(document.id) in seen_documents:
            continue
        seen_documents.add(str(document.id))
        text = load_document_text(document) or ""
        if not text.strip():
            continue
        excerpt = _squash_text(text)
        source_refs.append(
            MemorySourceRef(
                source_type="document",
                source_id=str(document.id),
                label=document.outline_title or document.filename,
                excerpt=excerpt,
                metadata={"category": document.category, "filename": document.filename},
            )
        )
        snippet = text[: min(len(text), max(0, remaining_chars))]
        if snippet:
            chunks.append(
                f"### Dokument: {document.outline_title or document.filename}\n"
                f"Kategorie: {document.category}\n"
                f"{snippet}"
            )
            remaining_chars -= len(snippet)
        if remaining_chars <= 0:
            break

    for source_id in selection.saved_sources or []:
        if remaining_chars <= 0:
            break
        source = None
        try:
            source = (
                db.query(ResearchSource)
                .filter(
                    ResearchSource.id == source_id,
                    ResearchSource.owner_id == current_user.id,
                    ResearchSource.case_id == case_id,
                )
                .first()
            )
        except Exception:
            source = None
        if not source:
            continue
        source_text = "\n".join(
            part
            for part in [
                source.title,
                source.description,
                source.research_query,
                source.url,
            ]
            if part
        )
        if not source_text.strip():
            continue
        excerpt = _squash_text(source_text)
        source_refs.append(
            MemorySourceRef(
                source_type="research_run",
                source_id=str(source.id),
                label=source.title,
                excerpt=excerpt,
                metadata={"url": source.url, "document_type": source.document_type},
            )
        )
        snippet = source_text[: min(len(source_text), max(0, remaining_chars))]
        chunks.append(f"### Quelle: {source.title}\n{snippet}")
        remaining_chars -= len(snippet)

    return "\n\n".join(chunks).strip(), source_refs


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    try:
        parts = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)
        return "\n".join(parts)
    except Exception:
        return ""


async def _extract_case_memory_from_material(material: str) -> CaseMemoryExtractionResult:
    if not MEMORY_EXTRACTION_MODEL.startswith("gemini"):
        raise HTTPException(
            status_code=501,
            detail=(
                "MEMORY_EXTRACTION_MODEL is not supported yet. "
                "Use a Gemini model for now or add the OpenAI-compatible Qwen provider."
            ),
        )

    prompt = f"""
Du extrahierst gepflegten Fall-Speicher für eine deutsche Kanzlei im Asyl-/Migrationsrecht.

Arbeite streng quellengebunden:
- Keine Tatsachen erfinden.
- Unsicherheiten in warnings oder offene_fragen aufnehmen.
- Kurz, konkret und wiederverwendbar formulieren.
- "beteiligte" als knappe Strings, z.B. "Mandant: ...", "Ehefrau: ...".
- "fall_notizen" ist ein kurzer Fallüberblick: Mandant, Familie, Historie, Ziel, guter nächster Schritt.
- "kernstrategie" ist die destillierte anwaltliche Strategie.

Antworte ausschließlich im JSON-Schema.

QUELLEN:
{material}
"""
    client = get_gemini_client()
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MEMORY_EXTRACTION_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=CaseMemoryExtractionResult,
            ),
        )
    except Exception as exc:
        print(f"[WARN] Case memory extraction failed: {exc}")
        raise HTTPException(status_code=502, detail="Fall-Speicher-Extraktion fehlgeschlagen")

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, CaseMemoryExtractionResult):
        return parsed
    if isinstance(parsed, dict):
        return CaseMemoryExtractionResult(**parsed)

    raw_text = _extract_response_text(response)
    try:
        return CaseMemoryExtractionResult(**json.loads(raw_text))
    except Exception as exc:
        print(f"[WARN] Case memory extraction returned invalid JSON: {exc}; raw={raw_text[:500]}")
        raise HTTPException(status_code=502, detail="Fall-Speicher-Extraktion lieferte kein gültiges JSON")


def _append_ops(field: str, values: list[Any]) -> list[MemoryPatchOperation]:
    return [
        MemoryPatchOperation(op="append", path=f"/{field}/-", value=value)
        for value in values
        if str(value or "").strip()
    ]


def _brief_ops_from_extraction(extraction: CaseMemoryExtractionResult) -> list[MemoryPatchOperation]:
    ops: list[MemoryPatchOperation] = []
    if extraction.fall_notizen.strip():
        ops.append(MemoryPatchOperation(op="set", path="/notizen", value=extraction.fall_notizen.strip()))
    ops.extend(
        _append_ops(
            "beteiligte",
            [{"name": value.strip()} for value in extraction.beteiligte if value.strip()],
        )
    )
    ops.extend(_append_ops("verfahrensstand", extraction.verfahrensstand))
    ops.extend(_append_ops("sachverhalt", extraction.sachverhalt))
    ops.extend(_append_ops("antraege_ziele", extraction.antraege_ziele))
    ops.extend(_append_ops("streitige_punkte", extraction.streitige_punkte))
    ops.extend(_append_ops("beweismittel", extraction.beweismittel))
    ops.extend(_append_ops("risiken", extraction.risiken))
    ops.extend(_append_ops("offene_fragen", extraction.offene_fragen_fall))
    return ops


def _strategy_ops_from_extraction(extraction: CaseMemoryExtractionResult) -> list[MemoryPatchOperation]:
    ops: list[MemoryPatchOperation] = []
    if extraction.kernstrategie.strip():
        ops.append(MemoryPatchOperation(op="set", path="/kernstrategie", value=extraction.kernstrategie.strip()))
    ops.extend(_append_ops("argumentationslinien", extraction.argumentationslinien))
    ops.extend(_append_ops("rechtliche_ansatzpunkte", extraction.rechtliche_ansatzpunkte))
    ops.extend(_append_ops("beweisstrategie", extraction.beweisstrategie))
    ops.extend(_append_ops("prozessuale_schritte", extraction.prozessuale_schritte))
    ops.extend(_append_ops("vergleich_oder_taktik", extraction.vergleich_oder_taktik))
    ops.extend(_append_ops("risiken_und_gegenargumente", extraction.risiken_und_gegenargumente))
    ops.extend(_append_ops("offene_fragen", extraction.offene_fragen_strategie))
    return ops


def _combined_payload(brief: Any, strategy: Any) -> Dict[str, Any]:
    brief_content = brief.content_json or {}
    strategy_content = strategy.content_json or {}
    return {
        "overview": brief_content.get("notizen", ""),
        "strategy": strategy_content.get("kernstrategie", ""),
        "memory": {
            "overview": brief_content.get("notizen", ""),
            "strategy": strategy_content.get("kernstrategie", ""),
        },
        "brief": memory_row_to_dict(brief, render_case_brief_compact(brief_content)),
        "case_brief": memory_row_to_dict(brief, render_case_brief_compact(brief_content)),
        "case_strategy": memory_row_to_dict(strategy, render_case_strategy_compact(strategy_content)),
    }


def _proposal_frontend_payload(proposal: Any) -> Dict[str, Any]:
    payload = proposal_to_dict(proposal)
    target_type = payload.get("target_type")
    ops = payload.get("ops") or []
    payload["section"] = "strategy" if target_type == STRATEGY_TARGET else "overview"
    payload["title"] = "Strategie-Vorschlag" if target_type == STRATEGY_TARGET else "Fall-Überblick-Vorschlag"
    payload["content"] = "\n".join(
        str(op.get("value", "")).strip()
        for op in ops
        if isinstance(op, dict) and str(op.get("value", "")).strip()
    )
    return payload


@router.get("/cases/{case_id}")
@limiter.limit("200/hour")
async def get_case_memory(
    request: Request,
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    target_case_id = _assert_owned_case(db, current_user, case_id)
    brief = get_or_create_case_brief(db, current_user.id, target_case_id)
    strategy = get_or_create_case_strategy(db, current_user.id, target_case_id)
    return JSONResponse(
        content=_combined_payload(brief, strategy),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.put("/cases/{case_id}")
@limiter.limit("100/hour")
async def update_case_memory(
    request: Request,
    case_id: str,
    body: CaseMemoryCombinedRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    target_case_id = _assert_owned_case(db, current_user, case_id)
    current_brief = get_or_create_case_brief(db, current_user.id, target_case_id)
    current_strategy = get_or_create_case_strategy(db, current_user.id, target_case_id)
    brief_content = default_case_brief_json()
    brief_content.update(current_brief.content_json or {})
    brief_content["notizen"] = body.overview.strip()
    strategy_content = default_case_strategy_json()
    strategy_content.update(current_strategy.content_json or {})
    strategy_content["kernstrategie"] = body.strategy.strip()
    brief = update_case_brief_manual(db, current_user.id, target_case_id, brief_content, actor="user")
    strategy = update_case_strategy_manual(db, current_user.id, target_case_id, strategy_content, actor="user")
    return _combined_payload(brief, strategy)


@router.get("/cases/{case_id}/proposals")
@limiter.limit("200/hour")
async def get_case_memory_proposals(
    request: Request,
    case_id: str,
    status: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    target_case_id = _assert_owned_case(db, current_user, case_id)
    proposals = list_memory_update_proposals(db, current_user.id, case_id=target_case_id, status=status)
    return {"proposals": [_proposal_frontend_payload(proposal) for proposal in proposals]}


@router.post("/cases/{case_id}/proposals")
@limiter.limit("80/hour")
async def create_case_memory_proposal(
    request: Request,
    case_id: str,
    body: MemoryProposalCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    target_case_id = _assert_owned_case(db, current_user, case_id)
    try:
        proposal = create_memory_update_proposal(
            db,
            current_user.id,
            body.target_type,
            body.expected_version,
            body.ops,
            body.source_refs,
            case_id=target_case_id,
            target_id=body.target_id,
            confidence=body.confidence,
            model=body.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _proposal_frontend_payload(proposal)


@router.post("/cases/{case_id}/proposals/from-selection")
@limiter.limit("40/hour")
async def propose_case_memory_from_selection(
    request: Request,
    case_id: str,
    body: ProposalFromSelectionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    target_case_id = _assert_owned_case(db, current_user, case_id)
    brief = get_or_create_case_brief(db, current_user.id, target_case_id)
    strategy = get_or_create_case_strategy(db, current_user.id, target_case_id)
    material, source_refs = _selected_source_material(db, current_user, target_case_id, body.selected_documents)
    if not material or not source_refs:
        raise HTTPException(
            status_code=400,
            detail="Keine auswertbaren OCR-Texte oder Quellen in der Auswahl gefunden.",
        )

    extraction = await _extract_case_memory_from_material(material)
    proposals = []

    brief_ops = _brief_ops_from_extraction(extraction)
    if brief_ops:
        proposals.append(
            create_memory_update_proposal(
                db,
                current_user.id,
                BRIEF_TARGET,
                int(brief.version or 1),
                brief_ops,
                source_refs,
                case_id=target_case_id,
                confidence=extraction.confidence,
                model=MEMORY_EXTRACTION_MODEL,
            )
        )

    strategy_ops = _strategy_ops_from_extraction(extraction)
    if strategy_ops:
        proposals.append(
            create_memory_update_proposal(
                db,
                current_user.id,
                STRATEGY_TARGET,
                int(strategy.version or 1),
                strategy_ops,
                source_refs,
                case_id=target_case_id,
                confidence=extraction.confidence,
                model=MEMORY_EXTRACTION_MODEL,
            )
        )

    if not proposals:
        raise HTTPException(status_code=422, detail="Extraktion enthielt keine verwertbaren Speicher-Vorschläge.")

    return {"proposals": [_proposal_frontend_payload(proposal) for proposal in proposals]}


@router.post("/proposals/{proposal_id}/accept")
@limiter.limit("100/hour")
async def accept_case_memory_proposal(
    request: Request,
    proposal_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        proposal = accept_memory_update_proposal(db, current_user.id, proposal_id, actor="user")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _proposal_frontend_payload(proposal)


@router.post("/proposals/{proposal_id}/reject")
@limiter.limit("100/hour")
async def reject_case_memory_proposal(
    request: Request,
    proposal_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        proposal = reject_memory_update_proposal(db, current_user.id, proposal_id, actor="user")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _proposal_frontend_payload(proposal)

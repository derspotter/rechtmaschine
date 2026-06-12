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
# Default to the local Qwen worker so case material never leaves our
# infrastructure for memory extraction. The desktop service manager runs
# llama-server with this model (and force-overrides request models anyway).
# Set MEMORY_EXTRACTION_MODEL to a gemini-* model to use the cloud path instead.
MEMORY_EXTRACTION_MODEL = (
    os.getenv(
        "MEMORY_EXTRACTION_MODEL",
        os.getenv("LLAMA_SERVER_MODEL", "qwen3.6-27b-udq5xl-vision"),
    ).strip()
    or "qwen3.6-27b-udq5xl-vision"
)
MAX_MEMORY_SOURCE_CHARS = int(
    (os.getenv("MAX_MEMORY_SOURCE_CHARS", "60000") or "60000").strip()
)
MEMORY_EXTRACTION_NUM_CTX = int(
    (os.getenv("MEMORY_EXTRACTION_NUM_CTX", "32768") or "32768").strip()
)
MEMORY_AUTO_APPLY = (
    os.getenv("MEMORY_AUTO_APPLY", "false").strip().lower() in {"1", "true", "yes", "on"}
)
# Hard caps on proposal size: the memory is a distilled brief, not a protocol.
MEMORY_MAX_ITEMS_PER_FIELD = int(
    (os.getenv("MEMORY_MAX_ITEMS_PER_FIELD", "3") or "3").strip()
)
MEMORY_MAX_OPS_PER_PROPOSAL = int(
    (os.getenv("MEMORY_MAX_OPS_PER_PROPOSAL", "8") or "8").strip()
)
# Case context handed to the extractor alongside the new documents, so a new
# Schriftsatz is read against the whole Akte instead of in isolation.
MEMORY_CONTEXT_TOTAL_CHARS = int(
    (os.getenv("MEMORY_CONTEXT_TOTAL_CHARS", "20000") or "20000").strip()
)
MEMORY_CONTEXT_DOC_CHARS = int(
    (os.getenv("MEMORY_CONTEXT_DOC_CHARS", "4000") or "4000").strip()
)


def _notify_memory_changed(case_id: Any, reason: str, pending: Optional[int] = None) -> None:
    """Push a memory_snapshot SSE event so open UIs refresh the memory panel.

    Uses raw pg_notify so it works from both the app and the job worker; the
    payload carries only the case id, clients re-fetch with their own auth.
    """
    try:
        from database import DATABASE_URL
        from events import DOCUMENTS_CHANNEL, notify_postgres

        payload: Dict[str, Any] = {
            "type": "memory_snapshot",
            "case_id": str(case_id),
            "reason": reason,
        }
        if pending is not None:
            payload["pending"] = pending
        notify_postgres(DATABASE_URL, json.dumps(payload, ensure_ascii=False), DOCUMENTS_CHANNEL)
    except Exception as exc:
        print(f"[MEMORY WARN] memory_snapshot notify failed: {exc}")


class CaseMemoryCombinedRequest(BaseModel):
    overview: str = ""
    strategy: str = ""
    # Full structured content (list fields). When present, replaces the
    # respective content wholesale (notizen/kernstrategie still come from
    # overview/strategy above).
    brief_content: Optional[Dict[str, Any]] = None
    strategy_content: Optional[Dict[str, Any]] = None


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


_MEMORY_EXTRACTION_RULES = """
Du extrahierst dauerhaften Fall-Speicher für eine deutsche Kanzlei im Asyl-/Migrationsrecht.

Das Fallgedächtnis ist KEIN Protokoll. Gespeichert wird nur, was ein Anwalt in sechs
Monaten noch wissen muss, wenn er die Akte wieder öffnet.

Extrahiere äußerst selektiv:
- Relevanztest für jeden Eintrag: Ändert er, WAS DER MANDANT HAT (Status, Fakten,
  Nachweise), WAS ER WILL (Ziel, begehrter Status) oder WAS NOCH FEHLT (offene
  Voraussetzungen, fehlende Nachweise)? Wenn nein, nicht speichern.
- Nur dauerhafte, fallprägende Fakten: Identität und Status des Mandanten, Kernkonflikt,
  Verfahrensstand mit Gericht und Aktenzeichen, harte Fristen, erklärte Ziele.
- Höchstens 3 Einträge pro Liste, weniger ist besser. Leere Felder sind ausdrücklich erwünscht.
- Jeder Eintrag muss für sich allein verständlich und in sechs Monaten noch nützlich sein.
- Derselbe Sachverhalt gehört in genau EIN Feld, niemals in mehrere.

Extrahiere NICHT:
- die eigene Kanzlei oder den eigenen Anwalt als Beteiligte
- Korrespondenz-Logistik (Posteingang, "Schriftsatz eingegangen", "Stellungnahme angefordert")
- kurzfristige Erledigungen und To-dos — die gehören in keine Gedächtnisliste
- Spekulationen oder Verdachtsmomente, die nicht ausdrücklich in der Quelle stehen
- Selbstverständlichkeiten, die sich aus anderen Einträgen bereits ergeben

Zurechnung ist entscheidend:
- Bedingte oder hypothetische Äußerungen ("sollte ...", "falls ...") sind KEINE Tatsachen
  und KEINE streitigen Punkte. Aus "Sollte der Mandant nicht interessiert sein, bitte um
  Mitteilung" folgt NICHT, dass sein Interesse zweifelhaft ist.
- Äußerungen von Gericht oder Gegenseite immer als zugeschriebene Äußerung kennzeichnen
  ("VG fragt an, ob ..."), niemals als Fakt über den Mandanten speichern.

Arbeite streng quellengebunden, erfinde keine Tatsachen. Unsicherheiten gehören in
warnings, nicht in die Faktenfelder.

Aufbau des Materials:
- "BISHERIGER FALL-SPEICHER", "AKTENÜBERSICHT" und "BISHERIGER AKTENINHALT" sind nur
  Kontext zum Verständnis. Extrahiere daraus NICHTS erneut.
- Extrahiere ausschließlich aus den "NEUE QUELLEN" — und nur Fakten, die im bisherigen
  Fall-Speicher noch fehlen.
- Nutze den Kontext, um die neuen Quellen richtig einzuordnen (wer ist wer, worum geht
  der Streit, was ist schon bekannt).

Formales:
- "beteiligte" nur Mandant, Gegenseite, Gericht: knappe Strings wie "Mandant: ...".
- "fall_notizen": höchstens 2 Sätze Fallüberblick, ohne Wiederholung der Listeneinträge.
- "kernstrategie" und alle Strategie-Felder NUR füllen, wenn die Quelle echte rechtliche
  Argumentation enthält (Schriftsatz, Bescheid, Urteil). Bei Anschreiben, Weiterleitungen
  und Fristsachen: Strategie-Felder leer lassen.
- Enthält die Quelle kaum dauerhafte Substanz (z.B. ein bloßes Begleitschreiben),
  gib fast nichts oder nichts zurück. Das ist das richtige Verhalten.
"""

_MEMORY_EXTRACTION_JSON_SPEC = """
Antworte ausschließlich mit einem JSON-Objekt mit exakt diesen Feldern:
{
  "beteiligte": ["string"],
  "verfahrensstand": ["string"],
  "sachverhalt": ["string"],
  "antraege_ziele": ["string"],
  "streitige_punkte": ["string"],
  "beweismittel": ["string"],
  "risiken": ["string"],
  "offene_fragen_fall": ["string"],
  "fall_notizen": "string",
  "kernstrategie": "string",
  "argumentationslinien": ["string"],
  "rechtliche_ansatzpunkte": ["string"],
  "beweisstrategie": ["string"],
  "prozessuale_schritte": ["string"],
  "vergleich_oder_taktik": ["string"],
  "risiken_und_gegenargumente": ["string"],
  "offene_fragen_strategie": ["string"],
  "confidence": 0.0,
  "warnings": ["string"]
}
Kein Text außerhalb des JSON-Objekts.
"""


async def _extract_case_memory_with_qwen(material: str) -> CaseMemoryExtractionResult:
    """Run memory extraction on the local Qwen worker via the service manager."""
    from citation_qwen import call_qwen_json
    from shared import ensure_anonymization_service_ready

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        raise HTTPException(
            status_code=503,
            detail="ANONYMIZATION_SERVICE_URL ist nicht konfiguriert (lokaler Qwen-Worker).",
        )
    await ensure_anonymization_service_ready()

    prompt = (
        f"{_MEMORY_EXTRACTION_RULES}\n{_MEMORY_EXTRACTION_JSON_SPEC}\n\nQUELLEN:\n{material}"
    )
    parsed = await call_qwen_json(
        service_url,
        prompt,
        model=MEMORY_EXTRACTION_MODEL,
        num_predict=2600,
        temperature=0.0,
        num_ctx=MEMORY_EXTRACTION_NUM_CTX,
    )
    if not parsed:
        raise HTTPException(
            status_code=502,
            detail="Fall-Speicher-Extraktion (Qwen) lieferte kein gültiges JSON",
        )
    try:
        return CaseMemoryExtractionResult(**parsed)
    except Exception as exc:
        print(f"[WARN] Qwen memory extraction returned invalid schema: {exc}; raw={str(parsed)[:500]}")
        raise HTTPException(
            status_code=502,
            detail="Fall-Speicher-Extraktion (Qwen) lieferte ein ungültiges Schema",
        )


async def _extract_case_memory_from_material(material: str) -> CaseMemoryExtractionResult:
    model_lower = MEMORY_EXTRACTION_MODEL.lower()
    if model_lower.startswith("qwen") or ":" in model_lower:
        return await _extract_case_memory_with_qwen(material)
    if not MEMORY_EXTRACTION_MODEL.startswith("gemini"):
        raise HTTPException(
            status_code=501,
            detail="MEMORY_EXTRACTION_MODEL is not supported. Use a local qwen* or a gemini-* model.",
        )

    prompt = f"{_MEMORY_EXTRACTION_RULES}\nAntworte ausschließlich im JSON-Schema.\n\nQUELLEN:\n{material}"
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


def _normalize_memory_value(value: Any) -> str:
    if isinstance(value, dict):
        value = (
            value.get("name")
            or value.get("label")
            or json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _pending_proposal_values(db: Session, owner_id: Any, case_id: Any, target_type: str) -> set:
    """Collect normalized values already proposed (pending) for a target."""
    values: set = set()
    try:
        for proposal in list_memory_update_proposals(
            db, owner_id, case_id=case_id, status="pending", target_type=target_type
        ):
            payload = proposal_to_dict(proposal)
            for op in payload.get("ops") or []:
                if not isinstance(op, dict):
                    continue
                field = str(op.get("path") or "").strip("/").split("/")[0]
                norm = _normalize_memory_value(op.get("value"))
                if field and norm:
                    values.add((field, norm))
    except Exception as exc:
        print(f"[MEMORY WARN] Failed to collect pending proposal values: {exc}")
    return values


def _dedupe_ops(
    ops: list[MemoryPatchOperation],
    current_content: Dict[str, Any],
    pending_values: set,
    protect_scalars: bool = False,
) -> list[MemoryPatchOperation]:
    """Drop ops that would duplicate existing content or pending proposals.

    With protect_scalars (background reflection), `set` ops never overwrite a
    non-empty scalar — the lawyer's curated text wins over automation.
    """
    kept: list[MemoryPatchOperation] = []
    for op in ops:
        field = op.path.strip("/").split("/")[0]
        norm = _normalize_memory_value(op.value)
        if op.op == "append":
            if not norm:
                continue
            existing = {
                _normalize_memory_value(item)
                for item in (current_content.get(field) or [])
            }
            if norm in existing or (field, norm) in pending_values:
                continue
            pending_values.add((field, norm))
            kept.append(op)
            continue
        if op.op == "set":
            current = current_content.get(field)
            if norm == _normalize_memory_value(current):
                continue
            if protect_scalars and str(current or "").strip():
                continue
            if (field, norm) in pending_values:
                continue
            pending_values.add((field, norm))
            kept.append(op)
            continue
        kept.append(op)
    return kept


def _cap_ops(ops: list[MemoryPatchOperation]) -> list[MemoryPatchOperation]:
    """Enforce hard size limits on a proposal, keeping the model's own ordering."""
    per_field: Dict[str, int] = {}
    kept: list[MemoryPatchOperation] = []
    for op in ops:
        if len(kept) >= MEMORY_MAX_OPS_PER_PROPOSAL:
            break
        if op.op == "append":
            field = op.path.strip("/").split("/")[0]
            if per_field.get(field, 0) >= MEMORY_MAX_ITEMS_PER_FIELD:
                continue
            per_field[field] = per_field.get(field, 0) + 1
        kept.append(op)
    return kept


class MemoryReflectionRequest(BaseModel):
    case_id: str
    trigger: str = "documents"  # documents | draft | query
    document_ids: list[str] = Field(default_factory=list)
    draft_id: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None


def _document_material(
    db: Session,
    current_user: User,
    case_id: Any,
    document_ids: list[str],
) -> tuple[str, list[MemorySourceRef]]:
    chunks: list[str] = []
    source_refs: list[MemorySourceRef] = []
    remaining = MAX_MEMORY_SOURCE_CHARS
    for raw_id in document_ids:
        if remaining <= 0:
            break
        document = (
            db.query(Document)
            .filter(
                Document.id == raw_id,
                Document.owner_id == current_user.id,
                Document.case_id == case_id,
            )
            .first()
        )
        if not document:
            continue
        text = load_document_text(document) or ""
        if not text.strip():
            continue
        source_refs.append(
            MemorySourceRef(
                source_type="document",
                source_id=str(document.id),
                label=document.outline_title or document.filename,
                excerpt=_squash_text(text),
                metadata={"category": document.category, "filename": document.filename},
            )
        )
        snippet = text[:remaining]
        chunks.append(
            f"### Dokument: {document.outline_title or document.filename}\n"
            f"Kategorie: {document.category}\n{snippet}"
        )
        remaining -= len(snippet)
    return "\n\n".join(chunks).strip(), source_refs


def _case_context_material(
    db: Session,
    current_user: User,
    case_id: Any,
    brief: Any,
    strategy: Any,
    exclude_ids: Optional[set[str]] = None,
    include_doc_excerpts: bool = True,
) -> str:
    """Build the context block: current memory, chronological Akte overview,
    and excerpts of earlier documents from the beginning of the case."""
    exclude_ids = exclude_ids or set()
    sections: list[str] = []

    try:
        memory_block = (
            f"{render_case_brief_compact(brief.content_json or {})}\n\n"
            f"{render_case_strategy_compact(strategy.content_json or {})}"
        )
        sections.append(
            "BISHERIGER FALL-SPEICHER (nur Kontext, NICHT erneut extrahieren):\n" + memory_block
        )
    except Exception as exc:
        print(f"[MEMORY WARN] Failed to render memory context: {exc}")

    documents = (
        db.query(Document)
        .filter(Document.owner_id == current_user.id, Document.case_id == case_id)
        .order_by(Document.created_at.asc())
        .all()
    )
    if documents:
        inventory_lines = []
        for doc in documents:
            created = doc.created_at.strftime("%d.%m.%Y") if doc.created_at else "?"
            marker = " [NEU]" if str(doc.id) in exclude_ids else ""
            inventory_lines.append(
                f"- {created}: {doc.outline_title or doc.filename} ({doc.category}){marker}"
            )
        sections.append("AKTENÜBERSICHT (chronologisch):\n" + "\n".join(inventory_lines))

    if include_doc_excerpts:
        remaining = MEMORY_CONTEXT_TOTAL_CHARS
        excerpt_chunks: list[str] = []
        for doc in documents:
            if remaining <= 0:
                break
            if str(doc.id) in exclude_ids:
                continue
            text = load_document_text(doc) or ""
            if not text.strip():
                continue
            excerpt = text[: min(MEMORY_CONTEXT_DOC_CHARS, remaining)]
            excerpt_chunks.append(
                f"### {doc.outline_title or doc.filename} ({doc.category})\n{excerpt}"
            )
            remaining -= len(excerpt)
        if excerpt_chunks:
            sections.append(
                "BISHERIGER AKTENINHALT (Auszüge, chronologisch, nur Kontext):\n"
                + "\n\n".join(excerpt_chunks)
            )

    return "\n\n".join(sections)


def _reflection_material(
    db: Session,
    current_user: User,
    case_id: Any,
    body: "MemoryReflectionRequest",
) -> tuple[str, list[MemorySourceRef]]:
    if body.trigger == "draft" and body.draft_id:
        from models import GeneratedDraft

        draft = (
            db.query(GeneratedDraft)
            .filter(
                GeneratedDraft.id == body.draft_id,
                GeneratedDraft.user_id == current_user.id,
                GeneratedDraft.case_id == case_id,
            )
            .first()
        )
        if not draft or not (draft.generated_text or "").strip():
            return "", []
        text = draft.generated_text[:MAX_MEMORY_SOURCE_CHARS]
        refs = [
            MemorySourceRef(
                source_type="draft",
                source_id=str(draft.id),
                label=f"Entwurf: {draft.document_type}",
                excerpt=_squash_text(text),
                metadata={"document_type": draft.document_type, "model": draft.model_used},
            )
        ]
        material = (
            f"### Generierter Entwurf ({draft.document_type})\n"
            f"Anweisung des Anwalts: {_squash_text(draft.user_prompt, 600)}\n\n{text}"
        )
        return material, refs

    if body.trigger == "query" and (body.question or body.answer):
        question = _squash_text(body.question or "", 2000)
        answer = (body.answer or "")[:MAX_MEMORY_SOURCE_CHARS]
        refs = [
            MemorySourceRef(
                source_type="chat",
                source_id=str(case_id),
                label="Dokument-Befragung",
                excerpt=_squash_text(f"F: {question} A: {answer}"),
                metadata={"trigger": "query"},
            )
        ]
        material = f"### Dokument-Befragung\nFRAGE: {question}\n\nANTWORT:\n{answer}"
        return material, refs

    return _document_material(db, current_user, case_id, body.document_ids)


async def _execute_memory_reflection_request(
    body: MemoryReflectionRequest,
    db: Session,
    current_user: User,
) -> Dict[str, Any]:
    """Job-worker executor: extract memory proposals after case material changed."""
    target_case_id = resolve_case_uuid_for_request(db, current_user, body.case_id)
    case = (
        db.query(Case)
        .filter(Case.id == target_case_id, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        return {"created": 0, "skipped": "case not found"}

    material, source_refs = _reflection_material(db, current_user, target_case_id, body)
    if not material or not source_refs:
        return {"created": 0, "skipped": "no usable material"}

    brief = get_or_create_case_brief(db, current_user.id, target_case_id)
    strategy = get_or_create_case_strategy(db, current_user.id, target_case_id)

    # Read the new material against the whole Akte: current memory, chronological
    # overview, and excerpts of earlier documents from the beginning of the case.
    context = _case_context_material(
        db,
        current_user,
        target_case_id,
        brief,
        strategy,
        exclude_ids={str(d) for d in (body.document_ids or [])},
        include_doc_excerpts=(body.trigger == "documents"),
    )
    if context:
        material = f"{context}\n\nNEUE QUELLEN (Extraktionsquelle):\n{material}"

    extraction = await _extract_case_memory_from_material(material)

    created: list[Dict[str, Any]] = []
    auto_applied = 0
    for target_type, target_row, ops in (
        (BRIEF_TARGET, brief, _brief_ops_from_extraction(extraction)),
        (STRATEGY_TARGET, strategy, _strategy_ops_from_extraction(extraction)),
    ):
        pending_values = _pending_proposal_values(db, current_user.id, target_case_id, target_type)
        ops = _cap_ops(_dedupe_ops(ops, target_row.content_json or {}, pending_values, protect_scalars=True))
        if not ops:
            continue
        proposal = create_memory_update_proposal(
            db,
            current_user.id,
            target_type,
            int(target_row.version or 1),
            ops,
            source_refs,
            case_id=target_case_id,
            confidence=extraction.confidence,
            model=MEMORY_EXTRACTION_MODEL,
        )
        created.append({"id": str(proposal.id), "target_type": target_type, "ops": len(ops)})
        if MEMORY_AUTO_APPLY:
            try:
                accept_memory_update_proposal(
                    db,
                    current_user.id,
                    proposal.id,
                    actor=f"auto:{MEMORY_EXTRACTION_MODEL}",
                )
                auto_applied += 1
            except Exception as exc:
                print(f"[MEMORY WARN] Auto-apply failed for proposal {proposal.id}: {exc}")

    if created:
        _notify_memory_changed(
            target_case_id,
            "reflection",
            pending=max(0, len(created) - auto_applied),
        )

    return {
        "created": len(created),
        "auto_applied": auto_applied,
        "proposals": created,
        "trigger": body.trigger,
        "warnings": extraction.warnings,
    }


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
    content_lines = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        value = op.get("value", "")
        if isinstance(value, dict):
            value = value.get("name") or value.get("label") or json.dumps(value, ensure_ascii=False)
        text = str(value or "").strip()
        if text:
            content_lines.append(text)
    payload["content"] = "\n".join(content_lines)
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
    if body.brief_content is not None:
        brief_content.update(body.brief_content)
    brief_content["notizen"] = body.overview.strip()
    strategy_content = default_case_strategy_json()
    strategy_content.update(current_strategy.content_json or {})
    if body.strategy_content is not None:
        strategy_content.update(body.strategy_content)
    strategy_content["kernstrategie"] = body.strategy.strip()
    try:
        brief = update_case_brief_manual(db, current_user.id, target_case_id, brief_content, actor="user")
        strategy = update_case_strategy_manual(db, current_user.id, target_case_id, strategy_content, actor="user")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Ungültiger Fall-Speicher: {exc}")
    _notify_memory_changed(target_case_id, "manual_update")
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

    selected_doc_ids = {
        ref.source_id for ref in source_refs if ref.source_type == "document" and ref.source_id
    }
    context = _case_context_material(
        db,
        current_user,
        target_case_id,
        brief,
        strategy,
        exclude_ids=selected_doc_ids,
    )
    if context:
        material = f"{context}\n\nNEUE QUELLEN (Extraktionsquelle):\n{material}"

    extraction = await _extract_case_memory_from_material(material)
    proposals = []

    brief_ops = _cap_ops(_dedupe_ops(
        _brief_ops_from_extraction(extraction),
        brief.content_json or {},
        _pending_proposal_values(db, current_user.id, target_case_id, BRIEF_TARGET),
    ))
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

    strategy_ops = _cap_ops(_dedupe_ops(
        _strategy_ops_from_extraction(extraction),
        strategy.content_json or {},
        _pending_proposal_values(db, current_user.id, target_case_id, STRATEGY_TARGET),
    ))
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

    _notify_memory_changed(target_case_id, "from_selection", pending=len(proposals))
    return {"proposals": [_proposal_frontend_payload(proposal) for proposal in proposals]}


@router.post("/cases/{case_id}/reflect")
@limiter.limit("40/hour")
async def enqueue_case_memory_reflection(
    request: Request,
    case_id: str,
    body: MemoryReflectionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Queue a background memory-reflection job for this case."""
    from agent_memory_service import enqueue_memory_reflection

    target_case_id = _assert_owned_case(db, current_user, case_id)
    job = enqueue_memory_reflection(
        db,
        current_user.id,
        target_case_id,
        trigger=body.trigger,
        document_ids=body.document_ids or None,
        draft_id=body.draft_id,
        question=body.question,
        answer=body.answer,
    )
    if not job:
        raise HTTPException(status_code=503, detail="Reflection konnte nicht eingeplant werden (deaktiviert oder Fehler).")
    return {"job_id": str(job.id), "status": job.status}


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
    if getattr(proposal, "case_id", None):
        _notify_memory_changed(proposal.case_id, "proposal_accepted")
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
    if getattr(proposal, "case_id", None):
        _notify_memory_changed(proposal.case_id, "proposal_rejected")
    return _proposal_frontend_payload(proposal)

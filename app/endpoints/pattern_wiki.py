"""Anonymized cross-case pattern wiki (Phase 8 of the agent-memory plan).

Reusable argument/risk/evidence patterns distilled from mature case memory,
keyed by fingerprint and tags instead of case_id. Distillation runs as a
memory_reflection job (trigger "pattern_wiki"); entries land as 'pending'
and only become retrievable after lawyer review. A deterministic
anonymization gate drops any entry that still carries identifying tokens
from the source case.
"""

import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from auth import get_current_active_user
from citation_identity import NORM_RE, canonical_az, find_citations
from database import get_db
from models import Case, PatternWikiEntry, PatternWikiSource, User
from shared import limiter, resolve_case_uuid_for_request

router = APIRouter(prefix="/wiki", tags=["pattern-wiki"])

PATTERN_WIKI_INJECT_ENABLED = (
    os.getenv("PATTERN_WIKI_INJECT_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
PATTERN_WIKI_MAX_INJECTED = int(
    (os.getenv("PATTERN_WIKI_MAX_INJECTED", "3") or "3").strip()
)
# ~2k tokens. Sized so PATTERN_WIKI_MAX_INJECTED typical entries (1.7-3.6k chars
# observed) fit WITHOUT truncation. Do not grow this much further: past ~3 full
# patterns the block starts diluting attention from documents and task, so the
# lever for bigger knowledge is entry discipline, not budget.
PATTERN_WIKI_INJECT_MAX_CHARS = int(
    (os.getenv("PATTERN_WIKI_INJECT_MAX_CHARS", "8000") or "8000").strip()
)
PATTERN_WIKI_MIN_TAG_MATCHES = int(
    (os.getenv("PATTERN_WIKI_MIN_TAG_MATCHES", "2") or "2").strip()
)


class PatternWikiExtractionEntry(BaseModel):
    title: str = ""
    summary: str = ""
    fingerprint: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    argument_patterns: List[str] = Field(default_factory=list)
    risk_patterns: List[str] = Field(default_factory=list)
    evidence_patterns: List[str] = Field(default_factory=list)
    recommended_next_steps: List[str] = Field(default_factory=list)
    confidence: float = 0.5


class PatternWikiExtractionResult(BaseModel):
    entries: List[PatternWikiExtractionEntry] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


_DISTILL_RULES = """Du destillierst aus dem Fall-Speicher einer deutschen Kanzlei im Asyl-/Migrationsrecht
wiederverwendbare MUSTER für künftige ähnliche Fälle (anonymisiertes Pattern-Wiki).

Ein Muster ist nur wertvoll, wenn es einem Anwalt in einem ANDEREN, ähnlichen Fall hilft:
- Argumentationsmuster: welche rechtliche Argumentation wurde vorgebracht oder bietet sich an
- Risikomuster: welche Einwände von Behörde/Gericht sind zu erwarten
- Beweismuster: welche Nachweise eingesetzt wurden und wie sie beschafft/dokumentiert werden
- Nächste Schritte: Vorgehensweisen, die sich aus der Konstellation ergeben

ERKENNTNISSTAND — niemals Ungeprüftes als bewährt darstellen:
- Kennzeichne jede Aussage zu Argumenten und Vorgehensweisen mit ihrem Stand am Satzanfang:
  [bestätigt] = Gericht/Behörde hat dem tatsächlich stattgegeben,
  [signalisiert] = Gericht/Behörde hat eine Tendenz geäußert, aber noch nicht entschieden,
  [vorgetragen] = von der Kanzlei argumentiert, Ausgang offen,
  [Praxis] = beobachtete Verwaltungs-/Gerichtspraxis.
- Schreibe niemals "überzeugt", "hat funktioniert" oder "entkräftet", wenn der Fall noch
  nicht entschieden ist. Ein laufendes Verfahren liefert höchstens [signalisiert] oder
  [vorgetragen].

TERMINOLOGIE — keine erfundenen Begriffe:
- Verwende ausschließlich Gesetzesbegriffe (z.B. "Identitätsklärung", "Ausbildungsduldung",
  "Passbeschaffungsbemühungen"), etablierte Rechtsbegriffe oder schlichte Beschreibung.
- ERFINDE KEINE Komposita oder Etiketten, die wie Fachbegriffe klingen, aber keine sind.
  Wenn es für ein Konzept keinen etablierten Begriff gibt, beschreibe es in normalem Deutsch.
- Der Titel: höchstens 100 Zeichen, Gesetzesbegriff plus Normzitat oder schlichte Beschreibung.

GENERALISIERUNG — Muster, nicht Fallverlauf:
- Ein Muster muss über den konkreten Fallverlauf hinaus tragen. Einmalige prozessuale
  Zufälle dieses Falls (z.B. ein versäumter Termin, eine konkrete Verfahrenslage wie
  "Eilantrag zurücknehmen, weil vorläufige Erlaubnis erteilt") sind KEINE Muster.
- Generalisiere quellengetreu: keine Rechtsbehauptungen erfinden, die der Fall nicht trägt.
- Nur Muster mit Substanz; wenn der Fall nichts Wiederverwendbares hergibt, gib leere entries zurück.

STRENGE ANONYMISIERUNG — das Wiki ist fallübergreifend:
- KEINE Namen (Mandant, Richter, Anwälte, Behördenmitarbeiter), stattdessen Rollen
  ("der Antragsteller", "das Gericht", "die ABH")
- KEINE Aktenzeichen, Geschäftszeichen, ID-Nummern, Geburtsdaten, Adressen DES FALLS
- KEINE konkreten Daten (Datumsangaben) aus dem Fallverlauf; Fristen nur abstrakt
  ("innerhalb der 6-Monats-Frist des § 60c AufenthG")
- AUSNAHME: Fundstellen veröffentlichter Gerichtsentscheidungen sind erwünscht und
  MÜSSEN vollständig zitiert werden, in genau dieser Form:
  "VG Bremen, Urteil vom 16.02.2024 – 3 K 320/22" (Gericht, Urteil/Beschluss vom
  <Datum> – <Az>). Nur das Aktenzeichen des eigenen Verfahrens bleibt verboten.
- Gerichte/Behörden nur als Institution, wenn das Muster davon abhängt ("VG Düsseldorf"),
  nie einzelne Personen
- Herkunftsland und Rechtsgebiet DÜRFEN genannt werden, sie sind Teil des Fingerprints

Formales:
- 1 bis 3 Einträge, jeder für sich verständlich; weniger ist besser
- "fingerprint": Objekt mit "herkunftsland", "rechtsbehelf" (z.B. "Eilantrag § 123 VwGO",
  "Klage"), "verfahrensgegenstand" (z.B. "Ausbildungsduldung § 60c AufenthG"),
  "themen" (Liste von Schlagworten)
- "tags": 4-8 kleingeschriebene Schlagworte in normaler deutscher Rechtschreibung (mit
  Umlauten), Normzitate in der Form "§ 60c aufenthg"
  (z.B. "ausbildungsduldung", "§ 60c aufenthg", "identitätsklärung", "passbeschaffung",
  "tadschikistan")

Fundstellen: Nenne die tragende Entscheidung im Format "Gericht, Urteil oder
Beschluss vom TT.MM.JJJJ - Az" direkt im Argumentationsmuster. Verwende
ausschließlich Fundstellen, die im FALL-SPEICHER stehen, niemals eigene.
"""

_DISTILL_JSON_SPEC = """
Antworte ausschließlich mit einem JSON-Objekt:
{
  "entries": [
    {
      "title": "string",
      "summary": "string",
      "fingerprint": {"herkunftsland": "string", "rechtsbehelf": "string", "verfahrensgegenstand": "string", "themen": ["string"]},
      "tags": ["string"],
      "argument_patterns": ["string"],
      "risk_patterns": ["string"],
      "evidence_patterns": ["string"],
      "recommended_next_steps": ["string"],
      "confidence": 0.0
    }
  ],
  "warnings": ["string"]
}
Kein Text außerhalb des JSON-Objekts.
"""


def _occurs_outside(token: str, text: str, spans: List[tuple]) -> bool:
    """True if ``token`` appears in ``text`` at least once outside ``spans``
    (case-insensitively). Absent tokens count as not occurring."""
    return any(
        not any(s <= m.start() and m.end() <= e for s, e in spans)
        for m in re.finditer(re.escape(token), text, flags=re.IGNORECASE)
    )


def _forbidden_tokens(
    case: Case,
    brief_content: Dict[str, Any],
    strategy_content: Dict[str, Any],
    allowed_az: Optional[set] = None,
    assessment_content: Optional[Dict[str, Any]] = None,
) -> set:
    """Identifying tokens from the source case that must not survive into the wiki:
    party names, file numbers, Aktenzeichen, ID numbers, concrete dates.

    ``allowed_az`` excludes the whitelisted Aktenzeichen of verified Gutachten
    Fundstellen: those are legitimate wiki vocabulary, not identifying data.
    ``assessment_content`` is the third memory target the distillation reads
    from -- without it a birth date or a foreign Az that only the Gutachten
    knows would pass the gate."""
    from endpoints.agent_memory import _critical_tokens

    # Gutachten-Datumsfelder (``stand``, ``fundstellen[].datum``) sind ISO und
    # treffen das dd.mm.yyyy-Muster der Token-Suche nie -- der Inhalt geht
    # unverändert in den Blob.
    blob = json.dumps(
        [brief_content, strategy_content, assessment_content or {}], ensure_ascii=False
    )

    # EU-Normzitate ("Art. 14 Abs. 2 RL 2008/115/EG") sehen fuer den Az-Sammler
    # wie Aktenzeichen aus, sind aber Wiki-Vokabular (Live-Abnahme 07.09.2026).
    # Ein Blob-Treffer fällt nur, wenn er ausschließlich in einer Normspanne
    # steht -- dieselbe Sperrlogik, die find_citations anwendet. Gilt nur hier:
    # die Namenswörter unten stammen aus case.name und stehen nicht im Blob.
    norm_spans = [m.span() for m in NORM_RE.finditer(blob)]
    tokens: set = {
        token for token in _critical_tokens(blob)
        if not norm_spans or _occurs_outside(token, blob, norm_spans)
    }

    # Client name words only: institutions among the Beteiligte (Gericht, ABH,
    # Stadt ...) are legitimate wiki vocabulary, person names are not.
    name_sources: List[str] = [re.sub(r"^\s*\d{1,4}/\d{2}\s*", "", case.name or "")]
    for item in brief_content.get("beteiligte") or []:
        value = item.get("name", "") if isinstance(item, dict) else str(item)
        if value.strip().casefold().startswith(("mandant", "antragsteller", "kläger")):
            name_sources.append(value)
    institution_words = {
        "Mandant", "Mandantin", "Antragsteller", "Antragstellerin", "Kläger", "Klägerin",
        "Stadt", "Bundesrepublik", "Deutschland", "Bundesamt", "Migration", "Flüchtlinge",
        "Verwaltungsgericht", "Oberverwaltungsgericht", "Landeshauptstadt", "Gericht",
        "Ausländerbehörde", "Behörde", "Rechtsanwalt", "Rechtsanwälte", "Kanzlei",
    }
    for source in name_sources:
        for word in re.findall(r"[A-ZÄÖÜ][a-zäöüß]{3,}", source):
            if word not in institution_words:
                tokens.add(word)

    if allowed_az:
        tokens = {t for t in tokens if canonical_az(t) not in allowed_az}
    return tokens


def _entry_violations(entry: PatternWikiExtractionEntry, forbidden: set) -> List[str]:
    """Identifying tokens still present in the entry.

    Exempt are only the two parts of a published decision's citation that
    legitimately look like identifying data: its Aktenzeichen and its decision
    date. Court and prose around them stay checkable, so a client name that
    the model wrote *into* a citation phrase is still caught."""
    text = json.dumps(entry.model_dump(), ensure_ascii=False)
    exempt: List[tuple] = []
    for hit in find_citations(text):
        if hit.kind != "decision":
            continue
        exempt.append((hit.az_start, hit.az_end))
        if hit.date:
            date_at = text.find(hit.date, hit.start, hit.end)
            if date_at >= 0:
                exempt.append((date_at, date_at + len(hit.date)))

    return sorted(
        token for token in forbidden if token and _occurs_outside(token, text, exempt)
    )


def strip_foreign_citations(text: str, whitelist: set) -> tuple:
    """Remove citations whose Az is not in the whitelist. Returns (text, stripped).

    ``whitelist`` holds ``canonical_az`` values. A full citation loses its
    whole phrase, a bare Aktenzeichen only its own token -- so a hallucinated
    naked Az cannot slip through either. EU-Normzitate sind für die Az-Suche
    gesperrt (citation_identity.NORM_RE) und bleiben stehen."""
    stripped: List[Dict[str, str]] = []
    cuts: List[tuple] = []
    for hit in find_citations(text):
        if hit.canonical in whitelist:
            continue
        cuts.append((hit.start, hit.end))
        stripped.append({"az": hit.az, "citation": hit.raw})

    for start, end in reversed(cuts):
        text = text[:start] + text[end:]

    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip(), stripped


def _clean_fingerprint(fingerprint: Dict[str, Any], clean) -> Dict[str, Any]:
    """Fingerprint values are strings or string lists (herkunftsland,
    rechtsbehelf, verfahrensgegenstand, themen) -- run those through ``clean``
    and leave anything else the model invented untouched."""
    out: Dict[str, Any] = {}
    for key, value in (fingerprint or {}).items():
        if isinstance(value, str):
            out[key] = clean(value)
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            out[key] = [item for item in (clean(item) for item in value) if item]
        else:
            out[key] = value
    return out


def _strip_entry_citations(
    entry: PatternWikiExtractionEntry, whitelist: set
) -> List[Dict[str, str]]:
    """Strip foreign citations from every text field of one entry -- title,
    summary, tags, fingerprint and the four pattern lists -- and return what
    was removed. A foreign Az in the title or a tag reached the wiki
    unchecked while only summary and lists were cleaned."""
    removed_all: List[Dict[str, str]] = []

    def clean(value: str) -> str:
        cleaned, removed = strip_foreign_citations(value or "", whitelist)
        removed_all.extend(removed)
        return cleaned

    entry.title = clean(entry.title)
    entry.summary = clean(entry.summary)
    entry.tags = [tag for tag in (clean(tag) for tag in entry.tags or []) if tag]
    entry.fingerprint = _clean_fingerprint(entry.fingerprint, clean)
    for field in ("argument_patterns", "risk_patterns", "evidence_patterns",
                  "recommended_next_steps"):
        setattr(entry, field, [
            value for value in (clean(v) for v in getattr(entry, field) or []) if value
        ])
    return removed_all


async def _execute_pattern_wiki_distillation(
    db: Session,
    current_user: User,
    target_case_id: Any,
    case: Case,
) -> Dict[str, Any]:
    """Job executor: distill anonymized patterns from this case's memory."""
    from agent_memory_service import (
        get_or_create_case_assessment,
        get_or_create_case_brief,
        get_or_create_case_strategy,
        render_case_brief_compact,
        render_case_strategy_compact,
    )
    from assessment_memory import render_assessment_for_wiki, verified_az_whitelist
    from citation_qwen import call_qwen_json
    from endpoints.agent_memory import (
        MEMORY_EXTRACTION_MODEL,
        MEMORY_EXTRACTION_NUM_CTX,
        _notify_memory_changed,
    )
    from shared import ensure_anonymization_service_ready

    brief = get_or_create_case_brief(db, current_user.id, target_case_id)
    strategy = get_or_create_case_strategy(db, current_user.id, target_case_id)
    assessment = get_or_create_case_assessment(db, current_user.id, target_case_id)
    brief_content = brief.content_json or {}
    strategy_content = strategy.content_json or {}
    assessment_content = assessment.content_json or {}
    assessment_block, rendered_ids = render_assessment_for_wiki(assessment_content)
    whitelist = verified_az_whitelist(assessment_content, only_ids=rendered_ids)

    memory_block = "\n\n".join(
        part
        for part in (
            assessment_block,
            render_case_brief_compact(brief_content),
            render_case_strategy_compact(strategy_content),
        )
        if part
    )
    if len(memory_block.strip()) < 200:
        return {"created": 0, "trigger": "pattern_wiki", "skipped": "Fall-Speicher zu dünn für Muster-Ableitung"}

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        raise RuntimeError("ANONYMIZATION_SERVICE_URL ist nicht konfiguriert")
    await ensure_anonymization_service_ready()

    parsed = await call_qwen_json(
        service_url,
        f"{_DISTILL_RULES}\n\nFALL-SPEICHER:\n{memory_block}\n{_DISTILL_JSON_SPEC}",
        model=MEMORY_EXTRACTION_MODEL,
        # 3 Muster mit Fundstellen sind ~3.3k Tokens (157/26, 07.09.2026); bei
        # 3000 brach Qwen mitten im JSON ab (done_reason=length) -> Job failed.
        num_predict=6000,
        temperature=0.0,
        num_ctx=MEMORY_EXTRACTION_NUM_CTX,
    )
    if not parsed:
        raise RuntimeError("Muster-Destillation (Qwen) lieferte kein gültiges JSON")
    extraction = PatternWikiExtractionResult(**parsed)

    forbidden = _forbidden_tokens(
        case, brief_content, strategy_content,
        allowed_az=whitelist, assessment_content=assessment_content,
    )
    created: List[Dict[str, Any]] = []
    warnings = list(extraction.warnings or [])

    stripped_citations: List[Dict[str, str]] = []
    if whitelist or assessment_block:
        for entry in extraction.entries:
            removed = _strip_entry_citations(entry, whitelist)
            for item in removed:
                item["entry_title"] = entry.title
            stripped_citations.extend(removed)

    # Avoid re-creating patterns this case already produced (same title).
    existing_rows = (
        db.query(PatternWikiEntry)
        .join(PatternWikiSource, PatternWikiSource.pattern_wiki_entry_id == PatternWikiEntry.id)
        .filter(PatternWikiSource.source_id == str(target_case_id))
        .all()
    )
    existing_titles = {(row.title or "").strip().casefold() for row in existing_rows}

    for entry in extraction.entries:
        if not entry.title.strip():
            continue
        if entry.title.strip().casefold() in existing_titles:
            warnings.append(f"Übersprungen (bereits vorhanden): {entry.title}")
            continue
        violations = _entry_violations(entry, forbidden)
        if violations:
            warnings.append(
                f"Eintrag '{entry.title}' verworfen: enthält identifizierende Angaben ({', '.join(violations[:5])})"
            )
            continue
        row = PatternWikiEntry(
            owner_id=current_user.id,
            scope="firm",
            status="pending",
            fingerprint=entry.fingerprint,
            tags=[t.strip().casefold() for t in entry.tags if t.strip()],
            title=entry.title.strip()[:300],
            summary=entry.summary.strip(),
            argument_patterns=entry.argument_patterns,
            risk_patterns=entry.risk_patterns,
            evidence_patterns=entry.evidence_patterns,
            recommended_next_steps=entry.recommended_next_steps,
            confidence=entry.confidence,
            model=MEMORY_EXTRACTION_MODEL,
        )
        db.add(row)
        db.flush()
        db.add(
            PatternWikiSource(
                pattern_wiki_entry_id=row.id,
                source_type="case_assessment" if assessment_block else "case_brief",
                source_id=str(target_case_id),
                anonymized_note=(
                    (
                        "Destilliert aus Gutachten + Brief + Strategie, Namen, Aktenzeichen, "
                        "Nummern und konkrete Daten entfernt bzw. per Token-Gate geprüft, "
                        "Fundstellen außerhalb des Gutachtens per Az-Whitelist entfernt."
                    )
                    if assessment_block
                    else (
                        "Destilliert aus Fall-Speicher (Brief+Strategie); Namen, Aktenzeichen, "
                        "Nummern und konkrete Daten entfernt bzw. per Token-Gate geprüft."
                    )
                ),
            )
        )
        created.append({"id": str(row.id), "title": row.title})

    db.commit()
    if created:
        _notify_memory_changed(target_case_id, "pattern_wiki", pending=len(created))
    result = {
        "created": len(created),
        "trigger": "pattern_wiki",
        "entries": created,
        "warnings": warnings,
        "stripped_citations": stripped_citations,
    }
    if stripped_citations:
        result["warnings"].append(
            f"{len(stripped_citations)} Fundstelle(n) außerhalb des Gutachtens entfernt"
        )
    return result


def _scope_filter(query, current_user: Any):
    """Firm entries are kanzlei-shared by design; private entries are
    visible/mutable only to their owner."""
    owner_id = getattr(current_user, "id", current_user)
    return query.filter(
        or_(PatternWikiEntry.scope == "firm", PatternWikiEntry.owner_id == owner_id)
    )


def render_pattern_wiki_context(
    db: Session,
    current_user: Any,
    case_memory_text: str,
    collect: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Deterministic retrieval: active entries whose tags appear in the rendered
    case memory, scored by match count. Returns a capped prompt block. If a
    ``collect`` list is passed, each injected entry's {id, title} is appended."""
    if not PATTERN_WIKI_INJECT_ENABLED or not (case_memory_text or "").strip():
        return ""

    def _norm(value: str) -> str:
        value = value.casefold()
        for umlaut, ascii_form in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
            value = value.replace(umlaut, ascii_form)
        return value

    haystack = _norm(case_memory_text)
    scored = []
    rows = _scope_filter(
        db.query(PatternWikiEntry).filter(PatternWikiEntry.status == "active"), current_user
    ).all()
    for row in rows:
        tags = [_norm(str(t)) for t in (row.tags or [])]
        matches = sum(1 for t in tags if t and t in haystack)
        if matches >= PATTERN_WIKI_MIN_TAG_MATCHES:
            scored.append((matches, row))
    if not scored:
        return ""
    scored.sort(key=lambda pair: (-pair[0], -(pair[1].confidence or 0)))

    chunks: List[str] = []
    used: List[PatternWikiEntry] = []
    remaining = PATTERN_WIKI_INJECT_MAX_CHARS
    for _, row in scored[:PATTERN_WIKI_MAX_INJECTED]:
        lines = [f"### {row.title}"]
        if row.summary:
            lines.append(row.summary)
        for label, values in (
            ("Argumentationsmuster", row.argument_patterns),
            ("Risiken", row.risk_patterns),
            ("Beweismuster", row.evidence_patterns),
            ("Bewährte nächste Schritte", row.recommended_next_steps),
        ):
            if values:
                lines.append(f"{label}:")
                lines.extend(f"- {v}" for v in values)
        block = "\n".join(lines)
        if len(block) > remaining:
            # An oversized entry must not silently vanish (a detailed entry would
            # otherwise never inject) nor block smaller entries behind it (break
            # skipped ALL remaining entries). Truncate at a line boundary while a
            # useful budget remains; the section order (summary, arguments, risks,
            # evidence, steps) keeps the most important parts.
            if remaining < 800:
                continue
            block = block[:remaining].rsplit("\n", 1)[0] + "\n[Eintrag gekürzt]"
        chunks.append(block)
        used.append(row)
        remaining -= len(block)
    if not chunks:
        return ""
    if collect is not None:
        collect.extend({"id": str(row.id), "title": row.title} for row in used)
    now = datetime.utcnow()
    for row in used:
        row.last_used_at = now
        db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


class PatternWikiUpdateRequest(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    fingerprint: Optional[Dict[str, Any]] = None
    argument_patterns: Optional[List[str]] = None
    risk_patterns: Optional[List[str]] = None
    evidence_patterns: Optional[List[str]] = None
    recommended_next_steps: Optional[List[str]] = None


class PatternWikiCreateRequest(BaseModel):
    title: str = Field(..., max_length=300)
    summary: str = ""
    tags: List[str] = Field(default_factory=list)
    fingerprint: Dict[str, Any] = Field(default_factory=dict)
    argument_patterns: List[str] = Field(default_factory=list)
    risk_patterns: List[str] = Field(default_factory=list)
    evidence_patterns: List[str] = Field(default_factory=list)
    recommended_next_steps: List[str] = Field(default_factory=list)
    scope: Literal["private", "firm"] = "firm"
    confidence: Optional[float] = None
    model: Optional[str] = None


@router.get("/entries")
@limiter.limit("300/hour")
async def list_wiki_entries(
    request: Request,
    status: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    query = _scope_filter(db.query(PatternWikiEntry), current_user)
    if status:
        query = query.filter(PatternWikiEntry.status == status)
    rows = query.order_by(PatternWikiEntry.created_at.desc()).limit(200).all()
    return {"entries": [row.to_dict() for row in rows]}


@router.post("/cases/{case_id}/distill")
@limiter.limit("20/hour")
async def distill_case_patterns(
    request: Request,
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Queue pattern distillation for this case (memory_reflection job)."""
    from agent_memory_service import enqueue_memory_reflection

    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    case = (
        db.query(Case)
        .filter(Case.id == target_case_id, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    job = enqueue_memory_reflection(db, current_user.id, target_case_id, trigger="pattern_wiki")
    if not job:
        raise HTTPException(status_code=503, detail="Muster-Ableitung konnte nicht eingeplant werden.")
    return {"job_id": str(job.id), "status": job.status}


def _get_entry(db: Session, entry_id: str, current_user: User) -> PatternWikiEntry:
    try:
        entry_uuid = uuid.UUID(entry_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=404, detail="Wiki-Eintrag nicht gefunden")
    row = (
        _scope_filter(db.query(PatternWikiEntry), current_user)
        .filter(PatternWikiEntry.id == entry_uuid)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Wiki-Eintrag nicht gefunden")
    return row


@router.post("/entries/{entry_id}/accept")
@limiter.limit("100/hour")
async def accept_wiki_entry(
    request: Request,
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = _get_entry(db, entry_id, current_user)
    row.status = "active"
    row.reviewed_by = current_user.email
    row.updated_at = datetime.utcnow()
    db.commit()
    return row.to_dict()


@router.post("/entries/{entry_id}/reject")
@limiter.limit("100/hour")
async def reject_wiki_entry(
    request: Request,
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = _get_entry(db, entry_id, current_user)
    row.status = "rejected"
    row.reviewed_by = current_user.email
    row.updated_at = datetime.utcnow()
    db.commit()
    return row.to_dict()


@router.put("/entries/{entry_id}")
@limiter.limit("100/hour")
async def update_wiki_entry(
    request: Request,
    entry_id: str,
    body: PatternWikiUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = _get_entry(db, entry_id, current_user)
    for field in (
        "title", "summary", "fingerprint", "argument_patterns",
        "risk_patterns", "evidence_patterns", "recommended_next_steps",
    ):
        value = getattr(body, field)
        if value is not None:
            setattr(row, field, value)
    if body.tags is not None:
        row.tags = [t.strip().casefold() for t in body.tags if t.strip()]
    row.updated_at = datetime.utcnow()
    db.commit()
    return row.to_dict()


@router.delete("/entries/{entry_id}")
@limiter.limit("100/hour")
async def delete_wiki_entry(
    request: Request,
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = _get_entry(db, entry_id, current_user)
    db.delete(row)
    db.commit()
    return {"deleted": entry_id}


@router.post("/entries", status_code=201)
@limiter.limit("100/hour")
async def create_wiki_entry(
    request: Request,
    body: PatternWikiCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a curated wiki entry. Lands as 'pending' for review, like a distillation."""
    title = (body.title or "").strip()
    if not title:
        raise HTTPException(status_code=422, detail="title darf nicht leer sein")
    now = datetime.utcnow()
    row = PatternWikiEntry(
        owner_id=current_user.id,
        scope=body.scope,
        status="pending",
        fingerprint=body.fingerprint or {},
        tags=[t.strip().casefold() for t in body.tags if t.strip()],
        title=title,
        summary=body.summary or "",
        argument_patterns=body.argument_patterns or [],
        risk_patterns=body.risk_patterns or [],
        evidence_patterns=body.evidence_patterns or [],
        recommended_next_steps=body.recommended_next_steps or [],
        confidence=body.confidence,
        model=body.model,
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.to_dict()


@router.get("/entries/{entry_id}")
@limiter.limit("300/hour")
async def get_wiki_entry(
    request: Request,
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return _get_entry(db, entry_id, current_user).to_dict()

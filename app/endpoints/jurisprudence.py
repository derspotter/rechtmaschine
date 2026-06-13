"""Freshness-gated jurisprudence packs (Phase 9 of the agent-memory plan).

A jurisprudence pack is a compact, per-fingerprint bundle of current
case-law — recent decisions, holdings, argument patterns — assembled from
what we already store (RechtsprechungEntry rows and prior ResearchRuns).
It is the prompt payload the agent injects, not a searchable corpus
(that is Phase 10).

Flow at generation/query time (via get_case_memory_prompt_context):
  derive fingerprint from case memory → load/build pack → check freshness
  → inject the compact block. If the pack is stale/thin/missing, a research
  job is enqueued in the background (non-blocking) so the next draft is
  fresher; the current draft is never blocked on live research.
"""

import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import get_db
from models import (
    Case,
    JurisprudencePack,
    RechtsprechungEntry,
    ResearchRun,
    User,
)
from shared import limiter, resolve_case_uuid_for_request

router = APIRouter(prefix="/jurisprudence", tags=["jurisprudence"])

PACK_INJECT_ENABLED = (
    os.getenv("JURIS_PACK_INJECT_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
PACK_INJECT_MAX_CHARS = int((os.getenv("JURIS_PACK_INJECT_MAX_CHARS", "3500") or "3500").strip())
PACK_MAX_DECISIONS = int((os.getenv("JURIS_PACK_MAX_DECISIONS", "5") or "5").strip())
# Freshness thresholds (days) by urgency; the pack stores the resolved value.
PACK_MAX_AGE_URGENT = int((os.getenv("JURIS_MAX_AGE_URGENT_DAYS", "7") or "7").strip())
PACK_MAX_AGE_NORMAL = int((os.getenv("JURIS_MAX_AGE_NORMAL_DAYS", "30") or "30").strip())
JURIS_COLLECTION = os.getenv("JURIS_COLLECTION", "jurisprudence")
PACK_MAX_AGE_EVERGREEN = int((os.getenv("JURIS_MAX_AGE_EVERGREEN_DAYS", "180") or "180").strip())
# Don't re-enqueue background research more often than this.
PACK_RESEARCH_COOLDOWN_HOURS = int((os.getenv("JURIS_RESEARCH_COOLDOWN_HOURS", "24") or "24").strip())
# A pack with fewer than this many decisions counts as thin.
PACK_MIN_DECISIONS = int((os.getenv("JURIS_PACK_MIN_DECISIONS", "2") or "2").strip())
PACK_AUTO_RESEARCH = (
    os.getenv("JURIS_PACK_AUTO_RESEARCH", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Country names that appear in asylum memory; used to derive the fingerprint.
_COUNTRY_WORDS = [
    "afghanistan", "syrien", "irak", "iran", "türkei", "tuerkei", "tadschikistan",
    "russland", "somalia", "eritrea", "äthiopien", "aethiopien", "nigeria", "guinea",
    "pakistan", "bangladesch", "georgien", "armenien", "aserbaidschan", "ukraine",
    "libanon", "ägypten", "aegypten", "tunesien", "marokko", "algerien", "gambia",
    "kamerun", "kongo", "sudan", "südsudan", "sri lanka", "indien", "china",
]

# Legal-area signals → (area label, urgency for freshness policy).
_AREA_SIGNALS = [
    ("ausbildungsduldung", "Ausbildungsduldung § 60c AufenthG", "urgent"),
    ("§ 60c", "Ausbildungsduldung § 60c AufenthG", "urgent"),
    ("beschäftigungsduldung", "Beschäftigungsduldung § 60d AufenthG", "urgent"),
    ("widerruf", "Widerrufsverfahren", "normal"),
    ("§ 25b", "Aufenthalt § 25b AufenthG", "normal"),
    ("§ 25a", "Aufenthalt § 25a AufenthG", "normal"),
    ("eilantrag", "Eilrechtsschutz § 123 VwGO", "urgent"),
    ("§ 123", "Eilrechtsschutz § 123 VwGO", "urgent"),
    ("dublin", "Dublin-Verfahren", "urgent"),
    ("familiennachzug", "Familiennachzug", "normal"),
    ("einbürgerung", "Einbürgerung", "evergreen"),
    ("asyl", "Asylverfahren", "normal"),
]

_SECTION_RE = re.compile(r"§\s?\d+[a-z]?(?:\s?abs\.?\s?\d+)?", re.IGNORECASE)


def _norm(value: str) -> str:
    value = (value or "").casefold()
    for u, a in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        value = value.replace(u, a)
    return value


def derive_fingerprint(case_memory_text: str) -> Dict[str, Any]:
    """Heuristic fingerprint from the rendered case memory: countries, legal
    area + urgency, and issue tags (statute sections + area keywords)."""
    hay = case_memory_text or ""
    hay_n = _norm(hay)

    countries = sorted({c.replace("ae", "ä") for c in _COUNTRY_WORDS if _norm(c) in hay_n})
    # Re-derive display names without the lossy ascii fold for matched ones.
    countries = sorted({c for c in _COUNTRY_WORDS if _norm(c) in hay_n})

    legal_area = None
    urgency = "normal"
    for needle, area, urg in _AREA_SIGNALS:
        if _norm(needle) in hay_n:
            legal_area, urgency = area, urg
            break

    sections = sorted({re.sub(r"\s+", " ", m.group(0)).strip().lower() for m in _SECTION_RE.finditer(hay)})
    area_tags = sorted({area_kw for area_kw, _, _ in _AREA_SIGNALS if _norm(area_kw) in hay_n})
    issue_tags = sorted(set(sections) | set(area_tags))

    return {
        "countries": countries,
        "legal_area": legal_area,
        "urgency": urgency,
        "issue_tags": issue_tags,
    }


def fingerprint_key(fp: Dict[str, Any]) -> str:
    parts = [
        ",".join(sorted(fp.get("countries") or [])),
        (fp.get("legal_area") or "").strip().lower(),
        ",".join(sorted(fp.get("issue_tags") or [])),
    ]
    return _norm("|".join(parts))[:255]


def _refresh_after_days(urgency: str) -> int:
    return {
        "urgent": PACK_MAX_AGE_URGENT,
        "normal": PACK_MAX_AGE_NORMAL,
        "evergreen": PACK_MAX_AGE_EVERGREEN,
    }.get(urgency, PACK_MAX_AGE_NORMAL)


def _fingerprint_query(fp: Dict[str, Any]) -> str:
    parts = list(fp.get("countries") or []) + list(fp.get("issue_tags") or [])
    if fp.get("legal_area"):
        parts.append(str(fp["legal_area"]))
    return " ".join(str(p) for p in parts).strip()


def _hybrid_entries(db: Session, fp: Dict[str, Any]) -> List[RechtsprechungEntry]:
    """Relevance-ranked decisions from the hybrid jurisprudence store, blended
    with instance weight + freshness. Returns [] if the store/helper is
    unavailable or yields nothing (caller then falls back to SQL recency)."""
    query = _fingerprint_query(fp)
    if not query:
        return []
    try:
        from rag_context import retrieve_chunks
    except Exception:
        return []
    chunks = retrieve_chunks(query, limit=12, use_reranker=True, collection=JURIS_COLLECTION)
    relevance: Dict[str, float] = {}
    for c in chunks:
        eid = (c.get("metadata") or {}).get("rechtsprechung_entry_id")
        if not eid:
            continue
        score = float(c.get("score") or 0.0)
        if eid not in relevance or score > relevance[eid]:
            relevance[eid] = score
    if not relevance:
        return []

    import uuid as _uuid
    ids = []
    for eid in relevance:
        try:
            ids.append(_uuid.UUID(eid))
        except (ValueError, AttributeError):
            continue
    rows = (
        db.query(RechtsprechungEntry)
        .filter(RechtsprechungEntry.id.in_(ids), RechtsprechungEntry.is_active == True)  # noqa: E712
        .all()
    )

    def _blended(e: RechtsprechungEntry) -> float:
        rel = relevance.get(str(e.id), 0.0)
        inst = (e.instance_weight or 0) * 0.05  # +0.05/level (BVerfG/EuGH = +0.15)
        fresh = 0.0
        if e.decision_date:
            age_days = (date.today() - e.decision_date).days
            fresh = max(0.0, 0.2 * (1 - age_days / 3650))  # newer → up to +0.2
        return rel + inst + fresh

    return sorted(rows, key=_blended, reverse=True)


def _assemble_contents(
    db: Session, owner_id: Any, fp: Dict[str, Any]
) -> tuple[Dict[str, Any], List[str], Optional[date], float]:
    """Gather candidate case-law from RechtsprechungEntry + prior ResearchRuns
    matching the fingerprint. Returns (contents, source_ids, newest_date, coverage)."""
    countries_n = {_norm(c) for c in (fp.get("countries") or [])}
    tags_n = {_norm(t) for t in (fp.get("issue_tags") or [])}

    decisions: List[Dict[str, Any]] = []
    source_ids: List[str] = []
    newest: Optional[date] = None

    def _emit(e: RechtsprechungEntry) -> None:
        nonlocal newest
        decisions.append({
            "court": e.court,
            "decision_date": e.decision_date.isoformat() if e.decision_date else None,
            "aktenzeichen": e.aktenzeichen,
            "outcome": e.outcome,
            "holdings": (e.key_holdings or [])[:3],
            "argument_patterns": (e.argument_patterns or [])[:2],
            "country": e.country,
            "source": "rechtsprechung_entry",
        })
        source_ids.append(f"rechtsprechung_entry:{e.id}")
        if e.decision_date and (newest is None or e.decision_date > newest):
            newest = e.decision_date

    # Preferred: semantic relevance from the hybrid jurisprudence store, blended
    # with instance weight + freshness. Falls back to SQL recency + tag-match
    # when the store is empty/unreachable (e.g. before the corpus is built).
    selected = _hybrid_entries(db, fp)
    if selected:
        for e in selected[:PACK_MAX_DECISIONS]:
            _emit(e)
    else:
        entries = (
            db.query(RechtsprechungEntry)
            .filter(RechtsprechungEntry.is_active == True)  # noqa: E712
            .order_by(RechtsprechungEntry.decision_date.desc().nullslast())
            .limit(200)
            .all()
        )
        for e in entries:
            country_ok = (not countries_n) or (_norm(e.country or "") in countries_n)
            etags = {_norm(str(t)) for t in (e.tags or [])}
            tag_ok = bool(tags_n & etags) if tags_n else True
            if not (country_ok and tag_ok):
                continue
            _emit(e)
            if len(decisions) >= PACK_MAX_DECISIONS:
                break

    # Prior research runs matching the fingerprint give recency signal.
    research_count = 0
    runs = (
        db.query(ResearchRun)
        .filter(ResearchRun.owner_id == owner_id)
        .order_by(ResearchRun.created_at.desc())
        .limit(40)
        .all()
    )
    for r in runs:
        blob = _norm(f"{r.effective_query or ''} {r.user_query or ''}")
        if countries_n and not any(c in blob for c in countries_n):
            continue
        if tags_n and not any(t in blob for t in tags_n):
            continue
        source_ids.append(f"research_run:{r.id}")
        research_count += 1
        if research_count >= 5:
            break

    # Coverage: blends decision count and presence of recent research.
    coverage = min(1.0, len(decisions) / max(1, PACK_MAX_DECISIONS))
    if research_count:
        coverage = min(1.0, coverage + 0.2)

    contents = {
        "decisions": decisions,
        "research_run_count": research_count,
    }
    return contents, source_ids, newest, round(coverage, 3)


def build_or_refresh_pack(db: Session, owner_id: Any, fp: Dict[str, Any]) -> Optional[JurisprudencePack]:
    """Create or update the pack for this fingerprint from current material."""
    key = fingerprint_key(fp)
    if not key.strip("|"):
        return None
    contents, source_ids, newest, coverage = _assemble_contents(db, owner_id, fp)

    pack = (
        db.query(JurisprudencePack)
        .filter(JurisprudencePack.owner_id == owner_id, JurisprudencePack.fingerprint_key == key)
        .first()
    )
    now = datetime.utcnow()
    if not pack:
        pack = JurisprudencePack(owner_id=owner_id, fingerprint_key=key, created_at=now)
        db.add(pack)
    pack.fingerprint = fp
    pack.legal_area = fp.get("legal_area")
    pack.countries = fp.get("countries") or []
    pack.issue_tags = fp.get("issue_tags") or []
    pack.contents = contents
    pack.source_ids = source_ids
    pack.newest_decision_date = newest
    pack.coverage_confidence = coverage
    pack.refresh_after_days = _refresh_after_days(fp.get("urgency", "normal"))
    pack.last_refreshed_at = now
    pack.updated_at = now
    db.commit()
    db.refresh(pack)
    return pack


def evaluate_freshness(pack: JurisprudencePack) -> Dict[str, Any]:
    """Classify the pack: fresh | stale | thin | missing, with a reason."""
    if pack is None:
        return {"status": "missing", "reason": "Kein Rechtsprechungs-Pack für diesen Fall-Fingerprint."}
    decisions = (pack.contents or {}).get("decisions") or []
    if len(decisions) < PACK_MIN_DECISIONS:
        return {"status": "thin", "reason": f"Nur {len(decisions)} Entscheidung(en) im Pack."}
    if pack.last_refreshed_at:
        age_days = (datetime.utcnow() - pack.last_refreshed_at).days
        if age_days > (pack.refresh_after_days or PACK_MAX_AGE_NORMAL):
            return {"status": "stale", "reason": f"Pack ist {age_days} Tage alt (Schwelle {pack.refresh_after_days})."}
    return {"status": "fresh", "reason": "Pack ist aktuell und ausreichend abgedeckt."}


def _maybe_enqueue_research(db: Session, owner_id: Any, case_id: Any, pack: JurisprudencePack, fp: Dict[str, Any]) -> bool:
    """Enqueue a background research job to refresh a stale/thin/missing pack,
    respecting the per-pack cooldown. Never blocks; never raises."""
    if not PACK_AUTO_RESEARCH or pack is None:
        return False
    try:
        if pack.last_research_enqueued_at and (
            datetime.utcnow() - pack.last_research_enqueued_at < timedelta(hours=PACK_RESEARCH_COOLDOWN_HOURS)
        ):
            return False
        from models import ResearchJob

        query_bits = []
        if fp.get("legal_area"):
            query_bits.append(fp["legal_area"])
        if fp.get("countries"):
            query_bits.append(", ".join(fp["countries"]))
        query_bits.append("aktuelle Rechtsprechung")
        query = " ".join(query_bits)

        payload = {
            "query": query,
            "case_id": str(case_id) if case_id else None,
            "search_engine": "meta",
            "search_mode": "balanced",
            "max_sources": 12,
            "domain_policy": "legal_balanced",
            "jurisdiction_focus": "de_eu",
            "recency_years": 3,
        }
        job = ResearchJob(
            owner_id=owner_id,
            case_id=case_id,
            status="queued",
            request_payload=payload,
            result_payload={},
            updated_at=datetime.utcnow(),
        )
        db.add(job)
        pack.last_research_enqueued_at = datetime.utcnow()
        db.add(pack)
        db.commit()
        print(f"[JURIS] Enqueued background research for pack {pack.fingerprint_key}: {query}")
        return True
    except Exception as exc:
        db.rollback()
        print(f"[JURIS WARN] research enqueue failed: {exc}")
        return False


def render_pack_block(pack: JurisprudencePack) -> str:
    """Compact prompt block for an active pack, capped."""
    if pack is None:
        return ""
    decisions = (pack.contents or {}).get("decisions") or []
    if not decisions:
        return ""
    lines: List[str] = []
    remaining = PACK_INJECT_MAX_CHARS
    for d in decisions[:PACK_MAX_DECISIONS]:
        head = " · ".join(p for p in [d.get("court"), d.get("decision_date"), d.get("aktenzeichen"), d.get("outcome")] if p)
        chunk_lines = [f"### {head}" if head else "### Entscheidung"]
        for h in (d.get("holdings") or [])[:3]:
            chunk_lines.append(f"- {h}")
        for a in (d.get("argument_patterns") or [])[:2]:
            chunk_lines.append(f"- Argumentationsmuster: {a}")
        chunk = "\n".join(chunk_lines)
        if len(chunk) > remaining:
            break
        lines.append(chunk)
        remaining -= len(chunk)
    return "\n\n".join(lines)


def maybe_render_jurisprudence_context(
    db: Session, current_user: Any, case_id: Any, case_memory_text: str
) -> str:
    """Entry point used by get_case_memory_prompt_context. Derives the
    fingerprint, builds/refreshes the pack, evaluates freshness, enqueues
    background research if needed, and returns the compact pack block."""
    if not PACK_INJECT_ENABLED or not (case_memory_text or "").strip() or not case_id:
        return ""
    owner_id = getattr(current_user, "id", current_user)
    fp = derive_fingerprint(case_memory_text)
    if not (fp.get("countries") or fp.get("legal_area") or fp.get("issue_tags")):
        return ""
    try:
        key = fingerprint_key(fp)
        pack = (
            db.query(JurisprudencePack)
            .filter(JurisprudencePack.owner_id == owner_id, JurisprudencePack.fingerprint_key == key)
            .first()
        )
        # Keep the hot path read-only: only (re)build when missing or when the
        # existing pack has aged past a short rebuild interval. The pack draws
        # from local rows, so a frequent rebuild just churns writes.
        stale_secs = max(3600, (pack.refresh_after_days * 86400) if pack else 0)
        needs_build = pack is None or (
            pack.last_refreshed_at is None
            or (datetime.utcnow() - pack.last_refreshed_at).total_seconds() > stale_secs
        )
        if needs_build:
            pack = build_or_refresh_pack(db, owner_id, fp)
    except Exception as exc:
        db.rollback()
        print(f"[JURIS WARN] pack build failed: {exc}")
        return ""
    status = evaluate_freshness(pack)
    if status["status"] in {"stale", "thin", "missing"}:
        _maybe_enqueue_research(db, owner_id, case_id, pack, fp)
    return render_pack_block(pack)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@router.get("/cases/{case_id}/pack")
@limiter.limit("120/hour")
async def get_case_pack(
    request: Request,
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Show the jurisprudence pack + freshness status for a case."""
    from agent_memory_service import (
        get_or_create_case_brief,
        get_or_create_case_strategy,
        render_case_brief_compact,
        render_case_strategy_compact,
    )

    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    case = db.query(Case).filter(Case.id == target_case_id, Case.owner_id == current_user.id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    brief = get_or_create_case_brief(db, current_user.id, target_case_id)
    strategy = get_or_create_case_strategy(db, current_user.id, target_case_id)
    memory_text = (
        f"{render_case_brief_compact(brief.content_json or {})}\n\n"
        f"{render_case_strategy_compact(strategy.content_json or {})}"
    )
    fp = derive_fingerprint(memory_text)
    pack = (
        db.query(JurisprudencePack)
        .filter(
            JurisprudencePack.owner_id == current_user.id,
            JurisprudencePack.fingerprint_key == fingerprint_key(fp),
        )
        .first()
    )
    return {
        "fingerprint": fp,
        "pack": pack.to_dict() if pack else None,
        "freshness": evaluate_freshness(pack) if pack else {"status": "missing", "reason": "Noch kein Pack."},
    }


@router.post("/cases/{case_id}/refresh")
@limiter.limit("40/hour")
async def refresh_case_pack(
    request: Request,
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Rebuild the pack from current material and (if stale/thin) enqueue research."""
    from agent_memory_service import (
        get_or_create_case_brief,
        get_or_create_case_strategy,
        render_case_brief_compact,
        render_case_strategy_compact,
    )

    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    case = db.query(Case).filter(Case.id == target_case_id, Case.owner_id == current_user.id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    brief = get_or_create_case_brief(db, current_user.id, target_case_id)
    strategy = get_or_create_case_strategy(db, current_user.id, target_case_id)
    memory_text = (
        f"{render_case_brief_compact(brief.content_json or {})}\n\n"
        f"{render_case_strategy_compact(strategy.content_json or {})}"
    )
    fp = derive_fingerprint(memory_text)
    if not (fp.get("countries") or fp.get("legal_area") or fp.get("issue_tags")):
        raise HTTPException(status_code=422, detail="Kein verwertbarer Fall-Fingerprint (Fall-Speicher zu dünn).")
    pack = build_or_refresh_pack(db, current_user.id, fp)
    freshness = evaluate_freshness(pack)
    enqueued = False
    if freshness["status"] in {"stale", "thin", "missing"}:
        enqueued = _maybe_enqueue_research(db, current_user.id, target_case_id, pack, fp)
    return {
        "fingerprint": fp,
        "pack": pack.to_dict() if pack else None,
        "freshness": freshness,
        "research_enqueued": enqueued,
    }

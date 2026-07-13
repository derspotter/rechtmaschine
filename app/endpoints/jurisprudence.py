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
from rechtsgebiete import uses_asyl_layers

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
# Pillar 4: facet-primary fingerprint, field-to-field matching, scored pack.
FACETS_ENABLED = (
    os.getenv("JURIS_FACETS_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Fingerprint + matching + scoring semantics live in the pure module
# juris_facets (facets primary, prose scrape as fallback); re-exported here
# so existing importers keep working.
from juris_facets import (  # noqa: E402
    _norm,
    derive_fingerprint,
    entry_matches,
    fingerprint_key,
    freeform_text,
    render_scored_block,
    score_entry,
)


def _load_case_pack_inputs(db: Session, owner_id: Any, case_id: Any) -> tuple:
    """(Facetten, Rechtsgebiet) des Falls, owner-gescoped (Facetten tragen
    Profil-/Gesundheitsdaten). (None, None) bei fehlendem Fall/Fehler."""
    if not case_id:
        return None, None
    try:
        row = (
            db.query(Case.facets_json, Case.rechtsgebiet, Case.rechtsgebiete)
            .filter(Case.id == case_id, Case.owner_id == owner_id)
            .first()
        )
        if not row:
            return None, None
        facets = dict(row[0]) if (FACETS_ENABLED and row[0]) else None
        return facets, (row[2] or row[1])
    except Exception as exc:
        db.rollback()
        print(f"[JURIS WARN] facets load failed: {exc}")
        return None, None


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
    # Firm-wide jurisprudence corpus, not user-owned case content — no owner filter.
    chunks = retrieve_chunks(
        query, owner_id=None, limit=12, use_reranker=True, collection=JURIS_COLLECTION
    )
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

    def _emit(e: RechtsprechungEntry, score: Optional[Dict[str, Any]] = None) -> None:
        nonlocal newest
        decision = {
            "court": e.court,
            "decision_date": e.decision_date.isoformat() if e.decision_date else None,
            "aktenzeichen": e.aktenzeichen,
            "outcome": e.outcome,
            "holdings": (e.key_holdings or [])[:3],
            "argument_patterns": (e.argument_patterns or [])[:2],
            "country": e.country,
            "leitsatz": getattr(e, "leitsatz", None),
            # Matching/scoring inputs, so the pack can be RE-scored against the
            # CURRENT case facets at render time (packs are shared per
            # fingerprint; profil-dependent risk must never be served stale).
            "normen": list(getattr(e, "normen", None) or []),
            "schlagworte": list(getattr(e, "schlagworte", None) or []),
            "instance_weight": getattr(e, "instance_weight", 0) or 0,
            "profil": getattr(e, "profil", None),
            "reliance": getattr(e, "reliance", None),
            "source": "rechtsprechung_entry",
        }
        if score:
            decision.update(score)
        decisions.append(decision)
        source_ids.append(f"rechtsprechung_entry:{e.id}")
        if e.decision_date and (newest is None or e.decision_date > newest):
            newest = e.decision_date

    if fp.get("facets"):
        # Facet path (Pillar 4): candidates from the hybrid store AND recent
        # rows, filtered field-to-field against the curated country/normen/
        # schlagworte columns, then deterministically scored and ranked by fit.
        candidates: Dict[str, RechtsprechungEntry] = {}
        hybrid = _hybrid_entries(db, fp)
        for e in hybrid:
            candidates.setdefault(str(e.id), e)
        for e in (
            db.query(RechtsprechungEntry)
            .filter(RechtsprechungEntry.is_active == True)  # noqa: E712
            .order_by(RechtsprechungEntry.decision_date.desc().nullslast())
            .limit(200)
            .all()
        ):
            candidates.setdefault(str(e.id), e)
        scored = [
            (score_entry(fp, e), e)
            for e in candidates.values()
            if entry_matches(fp, e)
        ]
        scored.sort(key=lambda pair: pair[0]["fit"], reverse=True)
        for score, e in scored[:PACK_MAX_DECISIONS]:
            _emit(e, score)
        if not decisions and hybrid:
            # Never worse than the pre-facet behavior: if strict field
            # matching leaves nothing, fall back to the top semantic hits
            # rather than silently dropping the whole jurisprudence block.
            for e in hybrid[:PACK_MAX_DECISIONS]:
                _emit(e, score_entry(fp, e))
    else:
        # Prose-fallback path: semantic relevance from the hybrid store,
        # falling back to SQL recency + free-form tag match.
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
                if not entry_matches(fp, e):
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


def build_or_refresh_pack(
    db: Session,
    owner_id: Any,
    fp: Dict[str, Any],
    legacy_key: Optional[str] = None,
) -> Optional[JurisprudencePack]:
    """Create or update the pack for this fingerprint from current material.

    ``legacy_key``: the prose-era fingerprint key of the same case. When the
    facet dialect creates a NEW pack, the research cooldown is inherited from
    the old pack so the key change does not trigger a fleet-wide burst of
    duplicate background research jobs."""
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
        if legacy_key and legacy_key != key:
            old = (
                db.query(JurisprudencePack)
                .filter(
                    JurisprudencePack.owner_id == owner_id,
                    JurisprudencePack.fingerprint_key == legacy_key,
                )
                .first()
            )
            if old is not None:
                pack.last_research_enqueued_at = old.last_research_enqueued_at
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
    # Facet-scored packs render sectioned: STÜTZEND / MIT VORSICHT / GEGEN UNS.
    if any("lager" in d for d in decisions):
        return render_scored_block(decisions[:PACK_MAX_DECISIONS], max_chars=PACK_INJECT_MAX_CHARS)
    lines: List[str] = []
    remaining = PACK_INJECT_MAX_CHARS
    for d in decisions[:PACK_MAX_DECISIONS]:
        head = " · ".join(p for p in [d.get("court"), d.get("decision_date"), d.get("aktenzeichen"), d.get("outcome")] if p)
        chunk_lines = [f"### {head}" if head else "### Entscheidung"]
        for h in (d.get("holdings") or [])[:3]:
            chunk_lines.append(f"- {freeform_text(h)}")
        for a in (d.get("argument_patterns") or [])[:2]:
            chunk_lines.append(f"- Argumentationsmuster: {freeform_text(a)}")
        chunk = "\n".join(chunk_lines)
        if len(chunk) > remaining:
            break
        lines.append(chunk)
        remaining -= len(chunk)
    return "\n\n".join(lines)


def maybe_render_jurisprudence_context(
    db: Session,
    current_user: Any,
    case_id: Any,
    case_memory_text: str,
    collect: Optional[Dict[str, Any]] = None,
) -> str:
    """Entry point used by get_case_memory_prompt_context. Derives the
    fingerprint, builds/refreshes the pack, evaluates freshness, enqueues
    background research if needed, and returns the compact pack block. If a
    ``collect`` dict is passed, it records the fingerprint and the decisions
    that grounded the draft."""
    if not PACK_INJECT_ENABLED or not case_id:
        return ""
    owner_id = getattr(current_user, "id", current_user)
    # Facets first: a fresh case with a processed Bescheid fingerprints from
    # day one, before any case memory exists (kills the empty-memory gate).
    facets, rechtsgebiet = _load_case_pack_inputs(db, owner_id, case_id)
    if not uses_asyl_layers(rechtsgebiet):
        return ""
    if not facets and not (case_memory_text or "").strip():
        return ""
    fp = derive_fingerprint(case_memory_text, facets=facets)
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
            # Facets change the fingerprint dialect; hand the prose-era key
            # over so a fresh pack inherits the research cooldown.
            legacy_key = (
                fingerprint_key(derive_fingerprint(case_memory_text))
                if fp.get("facets") and (case_memory_text or "").strip()
                else None
            )
            pack = build_or_refresh_pack(db, owner_id, fp, legacy_key=legacy_key)
    except Exception as exc:
        db.rollback()
        print(f"[JURIS WARN] pack build failed: {exc}")
        return ""
    status = evaluate_freshness(pack)
    if status["status"] in {"stale", "thin", "missing"}:
        _maybe_enqueue_research(db, owner_id, case_id, pack, fp)

    decisions = (pack.contents or {}).get("decisions") or [] if pack else []
    if fp.get("facets") and decisions:
        # Packs are shared per fingerprint (which excludes profil), and the
        # nightly enrichment moves under them — so lager/distinguish_risk are
        # RE-scored here against THIS case's facets, never served from the
        # cached snapshot.
        decisions = [
            {**d, **score_entry(fp, d)} for d in decisions[:PACK_MAX_DECISIONS]
        ]
        block = render_scored_block(decisions, max_chars=PACK_INJECT_MAX_CHARS)
    else:
        block = render_pack_block(pack)
    if collect is not None and block:
        collect["fingerprint_key"] = pack.fingerprint_key
        collect["legal_area"] = pack.legal_area
        collect["decisions"] = [
            {
                "court": d.get("court"),
                "aktenzeichen": d.get("aktenzeichen"),
                "decision_date": d.get("decision_date"),
                **({"lager": d["lager"], "distinguish_risk": d.get("distinguish_risk")} if "lager" in d else {}),
            }
            for d in decisions[:PACK_MAX_DECISIONS]
        ]
    return block


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
    facets, rechtsgebiet = _load_case_pack_inputs(db, current_user.id, target_case_id)
    if not uses_asyl_layers(rechtsgebiet):
        raise HTTPException(status_code=409, detail="Kein Migrationsrechtsfall — Jurisprudenz-Pack ist für dieses Rechtsgebiet nicht verfügbar")
    fp = derive_fingerprint(memory_text, facets=facets)
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
    facets, rechtsgebiet = _load_case_pack_inputs(db, current_user.id, target_case_id)
    if not uses_asyl_layers(rechtsgebiet):
        raise HTTPException(status_code=409, detail="Kein Migrationsrechtsfall — Jurisprudenz-Pack ist für dieses Rechtsgebiet nicht verfügbar")
    fp = derive_fingerprint(memory_text, facets=facets)
    if not (fp.get("countries") or fp.get("legal_area") or fp.get("issue_tags")):
        raise HTTPException(status_code=422, detail="Kein verwertbarer Fall-Fingerprint (Fall-Speicher zu dünn).")
    legacy_key = (
        fingerprint_key(derive_fingerprint(memory_text))
        if fp.get("facets") and (memory_text or "").strip()
        else None
    )
    pack = build_or_refresh_pack(db, current_user.id, fp, legacy_key=legacy_key)
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

"""Doktrin injection: wiki.aufentha.lt as the firm's authoritative layer.

Selects the most relevant DoktrinPage content for a case and renders the
prompt block that get_case_memory_prompt_context appends ABOVE Muster-Wiki
and Rechtsprechungs-Pack. Selection is hybrid:

1. deterministic — case facets (herkunftsland/schutzgruende/verfahrensart)
   scored against the bookkeeping rows in the app DB, no network,
2. semantic — retrieve_chunks against the "doktrin" RAG collection with the
   rendered base case memory as query.

Never raises: any failure degrades to an empty string so generation and
query keep working without the block.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from dokuwiki_markup import split_sections
from models import Case, DoktrinPage
from rag_context import retrieve_chunks

DOKTRIN_INJECT_ENABLED = (
    os.getenv("DOKTRIN_INJECT_ENABLED", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
DOKTRIN_COLLECTION = os.getenv("DOKTRIN_COLLECTION", "doktrin")
# Larger than the pattern wiki's 8k: this layer is the firm's normative line
# and deliberately the most prominent knowledge block. If draft quality drops
# under prompt pressure, shrink THIS first — never grow the other budgets at
# the same time.
DOKTRIN_INJECT_MAX_CHARS = int(
    (os.getenv("DOKTRIN_INJECT_MAX_CHARS", "11000") or "11000").strip()
)
DOKTRIN_MAX_ENTRIES = int((os.getenv("DOKTRIN_MAX_ENTRIES", "3") or "3").strip())
DOKTRIN_DETERMINISTIC_MAX = int(
    (os.getenv("DOKTRIN_DETERMINISTIC_MAX", "2") or "2").strip()
)
DOKTRIN_SEMANTIC_LIMIT = int(
    (os.getenv("DOKTRIN_SEMANTIC_LIMIT", "6") or "6").strip()
)
DOKTRIN_ENTRY_MAX_CHARS = int(
    (os.getenv("DOKTRIN_ENTRY_MAX_CHARS", "4000") or "4000").strip()
)

# verfahrensart facet enum -> keyword aliases matched in page_id/title.
_VERFAHRENSART_ALIASES: Dict[str, tuple] = {
    "dublin": ("dublin",),
    "asyl_folgeantrag": ("folgeantrag",),
    "asyl_eilverfahren": ("eilantrag", "80 abs. 5", "80_vwgo"),
    "abschiebungshaft": ("abschiebungshaft", "abschiebehaft"),
    "widerruf": ("widerruf",),
    "einbuergerung": ("einbuergerung", "einbürgerung", "stag"),
    "aufenthaltsrecht": ("aufenthaltserlaubnis", "aufenthg", "niederlassung"),
}


def _norm(value: str) -> str:
    value = (value or "").casefold()
    for umlaut, ascii_form in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        value = value.replace(umlaut, ascii_form)
    return value


def _load_case_facets(db: Session, current_user: Any, case_id: Any) -> Dict[str, Any]:
    """Owner-scoped facet block, mirroring endpoints/jurisprudence.py."""
    owner_id = getattr(current_user, "id", current_user)
    if not case_id:
        return {}
    try:
        row = (
            db.query(Case.facets_json)
            .filter(Case.id == case_id, Case.owner_id == owner_id)
            .first()
        )
        return dict(row[0]) if row and row[0] else {}
    except Exception as exc:
        db.rollback()
        print(f"[DOKTRIN WARN] facets load failed: {exc}")
        return {}


def _score_page(row: DoktrinPage, facets: Dict[str, Any]) -> int:
    score = 0
    herkunftsland = _norm(str(facets.get("herkunftsland") or ""))
    if herkunftsland and _norm(row.country or "") == herkunftsland:
        score += 3
    schutzgruende = {str(n) for n in (facets.get("schutzgruende") or [])}
    score += 2 * len(schutzgruende & {str(n) for n in (row.normen or [])})
    verfahrensart = str(facets.get("verfahrensart") or "")
    haystack = _norm(f"{row.page_id} {row.title or ''}")
    if any(alias in haystack for alias in _VERFAHRENSART_ALIASES.get(verfahrensart, ())):
        score += 2
    themen = {_norm(str(t)) for t in (facets.get("themen") or [])}
    score += len(themen & {_norm(str(t)) for t in (row.themen or [])})
    return score


def _slice_at_section_boundary(row: DoktrinPage, budget: int) -> str:
    """clean_text cut at the end of the section that crosses the budget, so a
    deterministic pick never ends mid-sentence."""
    text = row.clean_text or ""
    if len(text) <= budget:
        return text
    parts: List[str] = []
    used = 0
    for section in split_sections(text, row.title or row.page_id):
        block = f"## {section.heading_path}\n{section.text}"
        parts.append(block)
        used += len(block)
        if used >= budget:
            break
    return "\n\n".join(parts)[: budget + 2000]


def _format_entry(title: str, subtitle: str, text: str) -> str:
    label = f"### {title}" + (f" — {subtitle}" if subtitle else "")
    return f"{label} (Kanzlei-Wiki, nur intern)\n{text}"


def render_doktrin_context(
    db: Session,
    current_user: Any,
    case_id: Any,
    base_memory: str,
    collect: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if not DOKTRIN_INJECT_ENABLED:
        return ""
    try:
        return _render(db, current_user, case_id, base_memory, collect)
    except Exception as exc:
        print(f"[DOKTRIN WARN] context render failed: {exc}")
        return ""


def _render(
    db: Session,
    current_user: Any,
    case_id: Any,
    base_memory: str,
    collect: Optional[List[Dict[str, Any]]],
) -> str:
    facets = _load_case_facets(db, current_user, case_id)
    if not facets and not (base_memory or "").strip():
        return ""

    entries: List[Dict[str, Any]] = []
    picked_page_ids: set = set()

    # 1) Deterministic: facet-scored bookkeeping rows, no network.
    if facets:
        rows = db.query(DoktrinPage).filter(DoktrinPage.status == "active").all()
        scored = [(s, row) for row in rows if (s := _score_page(row, facets)) >= 2]
        scored.sort(key=lambda pair: (-pair[0], -(pair[1].clean_chars or 0)))
        for _, row in scored[:DOKTRIN_DETERMINISTIC_MAX]:
            entries.append(
                {
                    "page_id": row.page_id,
                    "title": row.title or row.page_id,
                    "subtitle": "",
                    "url": row.url or "",
                    "mode": "deterministic",
                    "text": _slice_at_section_boundary(row, DOKTRIN_ENTRY_MAX_CHARS),
                }
            )
            picked_page_ids.add(row.page_id)

    # 2) Semantic: fill remaining slots from the doktrin RAG collection.
    if len(entries) < DOKTRIN_MAX_ENTRIES and (base_memory or "").strip():
        # Firm-wide doctrine corpus, not user-owned case content — no owner filter.
        chunks = retrieve_chunks(
            base_memory.strip()[:1600],
            owner_id=None,
            limit=DOKTRIN_SEMANTIC_LIMIT,
            use_reranker=True,
            collection=DOKTRIN_COLLECTION,
        )
        by_page: Dict[str, List[dict]] = {}
        page_order: List[str] = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            page_id = str(metadata.get("page_id") or "")
            if not page_id or page_id in picked_page_ids:
                continue
            if page_id not in by_page:
                by_page[page_id] = []
                page_order.append(page_id)
            by_page[page_id].append(chunk)
        for page_id in page_order:
            if len(entries) >= DOKTRIN_MAX_ENTRIES:
                break
            group = sorted(
                by_page[page_id],
                key=lambda c: (c.get("metadata") or {}).get("chunk_index") or 0,
            )
            # Title/path/url can be missing on individual chunks; take the
            # first non-empty value across the group.
            def _meta(key: str) -> str:
                for c in group:
                    value = (c.get("metadata") or {}).get(key)
                    if value:
                        return str(value)
                return ""

            entries.append(
                {
                    "page_id": page_id,
                    "title": _meta("page_title") or page_id,
                    "subtitle": _meta("heading_path"),
                    "url": _meta("url"),
                    "mode": "semantic",
                    "text": "\n\n".join(
                        (c.get("text") or "").strip() for c in group
                    ).strip(),
                }
            )
            picked_page_ids.add(page_id)

    if not entries:
        return ""

    # 3) Budgeted rendering, truncation semantics identical to pattern_wiki.
    blocks: List[str] = []
    used: List[Dict[str, Any]] = []
    remaining = DOKTRIN_INJECT_MAX_CHARS
    for entry in entries:
        block = _format_entry(entry["title"], entry["subtitle"], entry["text"])
        if len(block) > remaining:
            if remaining < 800:
                continue
            block = block[:remaining].rsplit("\n", 1)[0] + "\n[Eintrag gekürzt]"
        blocks.append(block)
        entry["chars"] = len(block)
        used.append(entry)
        remaining -= len(block)
    if not blocks:
        return ""

    if collect is not None:
        collect.extend(
            {
                "page_id": e["page_id"],
                "title": e["title"],
                "url": e["url"],
                "mode": e["mode"],
                "chars": e["chars"],
            }
            for e in used
        )

    now = datetime.utcnow()
    try:
        db.query(DoktrinPage).filter(
            DoktrinPage.page_id.in_([e["page_id"] for e in used])
        ).update({DoktrinPage.last_used_at: now}, synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()

    return "\n\n".join(blocks)

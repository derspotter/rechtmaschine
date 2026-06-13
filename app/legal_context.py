"""Statutory ground-truth for generation/query prompts.

Scans the prompt inputs for references to AsylG/AufenthG/AsylbLG/GG provisions
and injects the *verbatim current statute text* as an authoritative block —
the deterministic ground-truth layer beside case memory (this case), kanzlei
precedent (RAG), and jurisprudence (case law). LLMs misquote German provision
wording and numbers; exact text prevents that at near-zero cost (local lookup,
no embedding, no external call).

Gated behind LEGAL_GROUNDING_ENABLED (default on — the statutes are local and
correct). Any failure degrades to an empty block.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

from legal_texts.downloader import get_law_path
from legal_texts.extractor import extract_provision, parse_provision_reference

# Only laws we hold locally; other citations (e.g. VwGO) are ignored.
_KNOWN_LAWS = {"asylg": "AsylG", "aufenthg": "AufenthG", "asylblg": "AsylbLG", "gg": "GG"}

_REF_RE = re.compile(
    r"(?:§|Art\.?)\s*\d+[a-z]?"
    r"(?:\s*Abs\.?\s*\d+)?"
    r"(?:\s*S(?:atz)?\.?\s*\d+)?"
    r"\s*(AsylG|AufenthG|AsylbLG|GG)",
    re.IGNORECASE,
)


def grounding_enabled() -> bool:
    return os.getenv("LEGAL_GROUNDING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _snapshot_date() -> str:
    try:
        return datetime.fromtimestamp(get_law_path("AsylG").stat().st_mtime).strftime("%Y-%m-%d")
    except Exception:
        return "unbekannt"


def build_statute_block(text: str, max_provisions: int = 6, per_cap: int = 1600) -> str:
    """Return a verbatim-statute block for the provisions referenced in `text`,
    or '' (no refs / disabled / failure). Safe to concatenate unconditionally."""
    if not grounding_enabled() or not text:
        return ""

    keys: list[tuple[str, str]] = []  # ordered-unique (canonical_law, paragraph)
    for match in _REF_RE.finditer(text):
        parsed = parse_provision_reference(match.group(0))
        if not parsed:
            continue
        law = _KNOWN_LAWS.get((parsed.get("law") or "").lower())
        if not law:
            continue
        key = (law, parsed["paragraph"])
        if key not in keys:
            keys.append(key)
        if len(keys) >= max_provisions:
            break

    parts: list[str] = []
    for law, paragraph in keys:
        provision = extract_provision(law, paragraph)
        if not provision or provision.startswith("[FEHLER]"):
            continue
        if len(provision) > per_cap:
            provision = provision[:per_cap].rstrip() + " […]"
        parts.append(provision)

    if not parts:
        return ""

    header = (
        f"GESETZESTEXT (geltende Fassung, Stand des lokalen Auszugs: {_snapshot_date()}) — "
        "maßgeblicher Wortlaut der zitierten Vorschriften. Nutze ausschließlich diesen "
        "Wortlaut als verbindliche Grundlage und zitiere Vorschriften nicht aus dem Gedächtnis:"
    )
    return header + "\n\n" + "\n\n".join(parts) + "\n\n"

"""Controlled vocabulary for RAG tagging.

A single canonical taxonomy (themen/Herkunftsländer/Normen) shared by every
collection so the same concept is the same token across jurisprudence and the
firm's own filings. Normalization = lowercase/strip/collapse-whitespace, apply
an alias map, then keep only terms in the canonical set. Pure functions that
take a Vocabulary, so they are testable without the generated JSON.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

_WS = re.compile(r"\s+")

DEFAULT_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "rag_vocabulary.json")


@dataclass
class Vocabulary:
    themen: list[str] = field(default_factory=list)
    themen_aliases: dict[str, str] = field(default_factory=dict)
    laender: list[str] = field(default_factory=list)
    laender_aliases: dict[str, str] = field(default_factory=dict)
    normen: list[str] = field(default_factory=list)
    normen_aliases: dict[str, str] = field(default_factory=dict)


def _norm_key(value: str) -> str:
    return _WS.sub(" ", (value or "").strip().lower())


def load_vocabulary(path: str = DEFAULT_VOCAB_PATH) -> Vocabulary:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return Vocabulary(
        themen=data.get("themen", []),
        themen_aliases=data.get("themen_aliases", {}),
        laender=data.get("laender", []),
        laender_aliases=data.get("laender_aliases", {}),
        normen=data.get("normen", []),
        normen_aliases=data.get("normen_aliases", {}),
    )


def _normalize_list(canonical: list[str], aliases: dict[str, str], raw: list[str]) -> list[str]:
    """Map each raw term through aliases, keep only canonical members, dedup
    preserving first-seen order. Canonical membership is checked case-folded;
    the returned token is the canonical entry's own casing."""
    canon_by_key = {_norm_key(c): c for c in canonical}
    alias_by_key = {_norm_key(k): v for k, v in aliases.items()}
    out: list[str] = []
    seen: set[str] = set()
    for term in raw or []:
        key = _norm_key(term)
        if key in alias_by_key:
            key = _norm_key(alias_by_key[key])
        canon = canon_by_key.get(key)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def normalize_themen(vocab: Vocabulary, raw: list[str]) -> list[str]:
    return _normalize_list(vocab.themen, vocab.themen_aliases, raw)


def normalize_normen(vocab: Vocabulary, raw: list[str]) -> list[str]:
    return _normalize_list(vocab.normen, vocab.normen_aliases, raw)


def normalize_country(vocab: Vocabulary, raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    result = _normalize_list(vocab.laender, vocab.laender_aliases, [raw])
    return result[0] if result else None


def tag_line(themen: list[str], country: Optional[str], normen: list[str]) -> str:
    """Compact suffix appended to context_header so both hybrid channels index
    the tags. Empty string when there is nothing to add."""
    parts: list[str] = []
    if themen:
        parts.append("Schlagwörter: " + ", ".join(themen))
    if country:
        parts.append("Herkunftsland: " + country)
    if normen:
        parts.append("Normen: " + ", ".join(normen))
    return " | ".join(parts)


def facet_metadata(themen: list[str], country: Optional[str], normen: list[str]) -> dict:
    """Metadata facets aligned to the RAG API's existing RagFilters keys
    (applicant_origin, citations) plus schlagworte. Omits empty fields."""
    md: dict = {}
    if themen:
        md["schlagworte"] = themen
    if country:
        md["applicant_origin"] = country
    if normen:
        md["citations"] = normen
    return md

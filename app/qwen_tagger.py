"""Tag a single (already-anonymized) document against the controlled vocabulary
using the desktop Qwen service. Returns normalized facets only — the model is
asked to choose from the vocab, and whatever it returns is re-normalized so
out-of-vocab hallucinations are dropped.

Anonymization invariant: callers pass anonymized text only. Controlled-vocab
tags are categories and cannot reintroduce PII.
"""
from __future__ import annotations

import os
from typing import Optional

from citation_qwen import call_qwen_json
from rag_vocabulary import (
    Vocabulary, normalize_themen, normalize_country, normalize_normen,
)

# Keep the prompt vocab bounded so it fits the context window comfortably.
_MAX_THEMEN_IN_PROMPT = 300
_MAX_DOC_CHARS = 12000


def _service_url() -> str:
    url = os.environ.get("ANONYMIZATION_SERVICE_URL", "").strip()
    if not url:
        raise RuntimeError("ANONYMIZATION_SERVICE_URL not set; cannot reach desktop Qwen")
    return url


def _build_prompt(vocab: Vocabulary, text: str) -> str:
    themen = vocab.themen[:_MAX_THEMEN_IN_PROMPT]
    laender = vocab.laender
    return (
        "Du bist ein juristischer Klassifikator für deutsches Asyl- und "
        "Aufenthaltsrecht. Wähle ausschließlich aus den vorgegebenen Listen. "
        "Erfinde keine neuen Begriffe.\n\n"
        f"ERLAUBTE SCHLAGWÖRTER:\n{', '.join(themen)}\n\n"
        f"ERLAUBTE HERKUNFTSLÄNDER:\n{', '.join(laender)}\n\n"
        "Gib NUR JSON zurück mit den Feldern: "
        '{"schlagworte": [..], "herkunftsland": "<eines oder null>", "normen": ["§ .. Gesetz", ..]}. '
        "schlagworte: die 3-8 treffendsten aus der Liste. herkunftsland: das "
        "betroffene Herkunftsland oder null. normen: die zentral einschlägigen "
        "Normen (z.B. \"§ 3 AsylG\", \"§ 60 Abs. 7 AufenthG\", \"Art. 3 EMRK\").\n\n"
        f"DOKUMENT (anonymisiert):\n{text[:_MAX_DOC_CHARS]}"
    )


async def tag_document(text: str, vocab: Vocabulary) -> dict:
    """Return {"schlagworte": [...], "herkunftsland": str|None, "normen": [...]},
    all normalized through the vocabulary. On any failure returns empty facets so
    a single bad document never aborts a retag run."""
    if not (text or "").strip():
        return {"schlagworte": [], "herkunftsland": None, "normen": []}
    try:
        parsed = await call_qwen_json(
            _service_url(), _build_prompt(vocab, text),
            num_predict=400, temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — degrade, don't abort the batch
        print(f"[tagger] qwen call failed: {exc}")
        return {"schlagworte": [], "herkunftsland": None, "normen": []}

    raw_themen = parsed.get("schlagworte") or []
    raw_country = parsed.get("herkunftsland")
    raw_normen = parsed.get("normen") or []
    if isinstance(raw_themen, str):
        raw_themen = [raw_themen]
    if isinstance(raw_normen, str):
        raw_normen = [raw_normen]
    return {
        "schlagworte": normalize_themen(vocab, raw_themen),
        "herkunftsland": normalize_country(vocab, raw_country),
        "normen": normalize_normen(vocab, raw_normen),
    }

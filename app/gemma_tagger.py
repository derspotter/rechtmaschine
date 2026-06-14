"""Tag a single (already-anonymized) document against the controlled vocabulary
using the debian-hosted Gemma 4 12B llama-server (OpenAI-compatible endpoint).

Sibling of qwen_tagger.py — same `tag_document(text, vocab)` interface, returning
normalized facets so callers (retag_rag, ingest) are backend-agnostic. Adds a
`thinking` switch: Gemma 4 has a reasoning channel that is overkill for
classification and costs a lot of throughput, so it defaults OFF.

thinking=False: response_format json_object forces clean JSON (no reasoning).
thinking=True : no grammar (it would block the thought channel); reasoning goes
                to reasoning_content, the final JSON is parsed out of content.

Out-of-vocab terms the model invents are dropped by the normalizer, exactly like
the Qwen path. Anonymization invariant: callers pass anonymized text only.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import httpx

from rag_vocabulary import (
    Vocabulary, normalize_themen, normalize_country, normalize_normen,
)

_MAX_THEMEN_IN_PROMPT = 300
# Per-slot context is 4096 tokens (-c 8192 / -np 2). The vocab system prompt is
# ~1300 tokens, so cap the document so the whole prompt fits; the head of a legal
# filing (Rubrum, parties, Sachverhalt, Anträge) is what classification needs.
_MAX_DOC_CHARS = 5000
_TIMEOUT = float(os.getenv("GEMMA_TAGGER_TIMEOUT_SEC", "120"))
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _service_url() -> str:
    return os.getenv("GEMMA_TAGGER_URL", "http://debian:8011").strip().rstrip("/")


def _headers() -> dict[str, str]:
    # The debian gemma endpoint is API-key protected (see rag/docker-compose.debian.yml).
    # Sent only when configured, so this also works against an unauthenticated local server.
    key = os.getenv("GEMMA_TAGGER_API_KEY", "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _messages(vocab: Vocabulary, text: str) -> list[dict]:
    themen = ", ".join(vocab.themen[:_MAX_THEMEN_IN_PROMPT])
    laender = ", ".join(vocab.laender)
    system = (
        "Du bist ein juristischer Klassifikator für deutsches Asyl- und "
        "Aufenthaltsrecht. Wähle ausschließlich aus den vorgegebenen Listen. "
        "Erfinde keine neuen Begriffe. Antworte ausschließlich mit JSON.\n\n"
        f"ERLAUBTE SCHLAGWÖRTER:\n{themen}\n\n"
        f"ERLAUBTE HERKUNFTSLÄNDER:\n{laender}"
    )
    user = (
        'Gib NUR JSON zurück: {"schlagworte": [..], "herkunftsland": '
        '"<eines oder null>", "normen": ["§ .. Gesetz", ..]}. schlagworte: die '
        "3-8 treffendsten aus der Liste. herkunftsland: das betroffene "
        "Herkunftsland oder null. normen: die zentral einschlägigen Normen "
        '(z.B. "§ 3 AsylG", "§ 60 Abs. 7 AufenthG", "Art. 3 EMRK").\n\n'
        f"DOKUMENT (anonymisiert):\n{text[:_MAX_DOC_CHARS]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json(content: str) -> dict:
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:].lstrip() if raw.lower().startswith("json") else raw
    try:
        return json.loads(raw)
    except Exception:
        match = _JSON_RE.search(content or "")
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
        return {}


async def tag_document(text: str, vocab: Vocabulary, *, thinking: bool = False) -> dict:
    """Return {"schlagworte": [...], "herkunftsland": str|None, "normen": [...]},
    all normalized. Degrades to empty facets on any failure so a single bad
    document never aborts a batch."""
    if not (text or "").strip():
        return {"schlagworte": [], "herkunftsland": None, "normen": []}
    payload: dict = {
        "messages": _messages(vocab, text),
        "temperature": 0.0,
        "max_tokens": 800 if thinking else 256,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    # A JSON grammar would block the thought channel, so only constrain when
    # thinking is off.
    if not thinking:
        payload["response_format"] = {"type": "json_object"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                f"{_service_url()}/v1/chat/completions", json=payload, headers=_headers()
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content") or ""
    except Exception as exc:  # noqa: BLE001 — degrade, don't abort the batch
        print(f"[gemma-tagger] call failed: {exc}")
        return {"schlagworte": [], "herkunftsland": None, "normen": []}

    parsed = _extract_json(content)
    raw_themen = parsed.get("schlagworte") or []
    raw_normen = parsed.get("normen") or []
    if isinstance(raw_themen, str):
        raw_themen = [raw_themen]
    if isinstance(raw_normen, str):
        raw_normen = [raw_normen]
    return {
        "schlagworte": normalize_themen(vocab, raw_themen),
        "herkunftsland": normalize_country(vocab, parsed.get("herkunftsland")),
        "normen": normalize_normen(vocab, raw_normen),
    }

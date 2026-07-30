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

from tagger_windowing import merge_window_facets, split_windows
from rag_vocabulary import (
    Vocabulary, normalize_themen, normalize_country, normalize_normen,
)

# Volles Vokabular (373 Begriffe, Stand 07/2026) — die alte 300er-Kappung machte
# die alphabetisch hinteren Begriffe (traumatisierung..wohnsitzauflage) unvergebbar.
_MAX_THEMEN_IN_PROMPT = 400
# Mac-mini llama-server (com.rechtmaschine.gemma.plist): -c 131072 / -np 2
# = 65536 Tokens pro Slot (seit 30.07.2026, vorher 32k). Konservativ mit
# ~2 Zeichen/Token kalkuliert passt damit ein ganzes Dokument bis ~120k Zeichen
# in EINEN Call — nur noch längere werden gefenstert und die Facetten vereinigt
# (tagger_windowing). Token-dense docs that still overflow are caught and
# retried at half length (see _tag_window).
_MAX_DOC_CHARS = 120000
# Prompt-Processing auf dem M4 schafft nur ~100 tok/s — ein 30k-Token-Dokument
# braucht ~5 min, plus Queue hinter Mail-Intake-Calls. 120s war viel zu knapp.
_TIMEOUT = float(os.getenv("GEMMA_TAGGER_TIMEOUT_SEC", "900"))
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
        "3-8 treffendsten aus der Liste. Bevorzuge spezifische Begriffe — "
        "allgemeine Oberbegriffe (z.B. asylverfahren, aufenthaltsrecht) nur, "
        "wenn kein spezifischerer Begriff passt. herkunftsland: das betroffene "
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


async def _tag_window(text: str, vocab: Vocabulary, *, thinking: bool = False) -> dict:
    """Tag ein einzelnes Dokument-Fenster. Returns normalized facets;
    degrades to empty facets on any failure."""

    def _payload(doc: str) -> dict:
        p: dict = {
            "messages": _messages(vocab, doc),
            "temperature": 0.0,
            "max_tokens": 800 if thinking else 256,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        # A JSON grammar would block the thought channel, so only constrain
        # when thinking is off.
        if not thinking:
            p["response_format"] = {"type": "json_object"}
        return p

    content = ""
    # Try full length, then halve once on a 400 (token-dense doc overflowing the
    # per-slot context) so a long filing still gets tagged.
    doc = text[:_MAX_DOC_CHARS]
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for attempt in range(2):
                try:
                    response = await client.post(
                        f"{_service_url()}/v1/chat/completions",
                        json=_payload(doc), headers=_headers(),
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"].get("content") or ""
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 400 and attempt == 0 and len(doc) > 800:
                        doc = doc[: len(doc) // 2]
                        continue
                    raise
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


async def tag_document(text: str, vocab: Vocabulary, *, thinking: bool = False) -> dict:
    """Return {"schlagworte": [...], "herkunftsland": str|None, "normen": [...]},
    all normalized. Lange Dokumente werden fensterweise getaggt und die
    Facetten vereinigt, damit auch spät im Dokument liegende Themen die Tags
    erreichen. Degrades to empty facets on failure so a single bad document
    never aborts a batch."""
    if not (text or "").strip():
        return {"schlagworte": [], "herkunftsland": None, "normen": []}
    windows = split_windows(text, _MAX_DOC_CHARS)
    if len(windows) == 1:
        return await _tag_window(windows[0], vocab, thinking=thinking)
    results = []
    for window in windows:
        results.append(await _tag_window(window, vocab, thinking=thinking))
    return merge_window_facets(results)

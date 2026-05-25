"""Document translation helpers for anonymized Rechtmaschine documents."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai import types

import httpx

from citation_qwen import CITATION_QWEN_MODEL, CITATION_QWEN_NUM_CTX, CITATION_QWEN_TIMEOUT_SEC
from shared import (
    TRANSLATED_TEXT_DIR,
    ensure_anonymization_service_ready,
    get_gemini_client,
)


DEFAULT_GEMINI_TRANSLATION_MODEL = os.getenv(
    "GEMINI_TRANSLATION_MODEL", "gemini-3.1-pro-preview"
)
DEFAULT_QWEN_TRANSLATION_MODEL = os.getenv(
    "QWEN_TRANSLATION_MODEL",
    os.getenv(
        "OLLAMA_MODEL_QWEN",
        os.getenv("ANON_MODEL", os.getenv("ANON_MODEL_NAME", CITATION_QWEN_MODEL)),
    ),
)
TRANSLATION_CHUNK_CHARS = int(
    (os.getenv("DOCUMENT_TRANSLATION_CHUNK_CHARS", "22000") or "22000").strip()
)
TRANSLATION_QWEN_CHUNK_CHARS = int(
    (os.getenv("DOCUMENT_TRANSLATION_QWEN_CHUNK_CHARS", "5000") or "5000").strip()
)
TRANSLATION_QWEN_NUM_PREDICT = int(
    (os.getenv("DOCUMENT_TRANSLATION_QWEN_NUM_PREDICT", "5000") or "5000").strip()
)
TRANSLATION_COMPARE_CHAR_LIMIT = int(
    (os.getenv("DOCUMENT_TRANSLATION_COMPARE_CHAR_LIMIT", "65000") or "65000").strip()
)


def provider_for_model(model: str) -> str:
    model_lower = (model or "").strip().lower()
    if model_lower.startswith("gemini"):
        return "gemini"
    if "qwen" in model_lower:
        return "qwen"
    raise ValueError(f"Unsupported translation model: {model}")


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def chunk_text(text: str, max_chars: int = TRANSLATION_CHUNK_CHARS) -> List[str]:
    text = text or ""
    if len(text) <= max_chars:
        return [text]

    page_marker_chunks = []
    current = []
    current_len = 0
    for block in re.split(r"(?=--- Seite \d+ ---)", text):
        block = block.strip()
        if not block:
            continue
        if current and current_len + len(block) + 2 > max_chars:
            page_marker_chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0
        if len(block) > max_chars:
            if current:
                page_marker_chunks.append("\n\n".join(current).strip())
                current = []
                current_len = 0
            for start in range(0, len(block), max_chars):
                part = block[start : start + max_chars].strip()
                if part:
                    page_marker_chunks.append(part)
            continue
        current.append(block)
        current_len += len(block) + 2
    if current:
        page_marker_chunks.append("\n\n".join(current).strip())
    if len(page_marker_chunks) > 1:
        return page_marker_chunks

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for block in text.split("\n\n"):
        block_len = len(block) + 2
        if current and current_len + block_len > max_chars:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0
        if block_len > max_chars:
            for start in range(0, len(block), max_chars):
                part = block[start : start + max_chars].strip()
                if part:
                    chunks.append(part)
            continue
        current.append(block)
        current_len += block_len
    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def translation_prompt(
    text: str,
    *,
    target_language: str,
    source_language: Optional[str],
    chunk_index: int,
    chunk_count: int,
) -> str:
    source = source_language or "Belarussisch, Russisch oder gemischter OCR-Text"
    chunk_note = (
        f"Teil {chunk_index} von {chunk_count}. "
        if chunk_count > 1
        else ""
    )
    return f"""Übersetze den folgenden anonymisierten OCR-Text vollständig ins {target_language}.
Ausgangssprache: {source}.
{chunk_note}Bewahre Seitenmarker wie "--- Seite 1 ---", Absätze, Listen, Tabellenstruktur und Platzhalter exakt bei.
Übersetze keine Anonymisierungsplatzhalter wie [NAME_1], [ADRESSE_1], [DATUM_1].
Erfinde nichts. Markiere unleserliche oder unsichere OCR-Stellen knapp mit [unleserlich].
Gib nur die Übersetzung zurück, keine Zusammenfassung und keine Kommentare.

TEXT:
{text}
"""


def _clean_qwen_translation_text(payload: Dict[str, Any]) -> str:
    raw = ""
    for key in ("response", "content", "text", "output", "message", "thinking"):
        value = payload.get(key)
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str) and value.strip():
            raw = value.strip()
            break

    if not raw:
        raw = str(payload).strip()

    cleaned = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"(?is)^```(?:json|markdown|text)?\s*", "", cleaned)
        cleaned = re.sub(r"(?is)\s*```$", "", cleaned).strip()

    if cleaned.startswith("{"):
        try:
            import json

            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                for key in ("translation", "translated_text", "text", "content"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except Exception:
            pass

    return cleaned


async def translate_with_gemini(
    text: str,
    *,
    model: str,
    target_language: str,
    source_language: Optional[str],
) -> str:
    client = get_gemini_client()
    chunks = chunk_text(text)
    translated_chunks: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        response = await client.aio.models.generate_content(
            model=model or DEFAULT_GEMINI_TRANSLATION_MODEL,
            contents=translation_prompt(
                chunk,
                target_language=target_language,
                source_language=source_language,
                chunk_index=index,
                chunk_count=len(chunks),
            ),
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=16000,
            ),
        )
        translated = (getattr(response, "text", None) or "").strip()
        if not translated:
            raise RuntimeError(f"Gemini returned an empty translation for chunk {index}/{len(chunks)}")
        translated_chunks.append(translated)
    return "\n\n".join(translated_chunks).strip()


async def translate_with_qwen(
    text: str,
    *,
    model: str,
    target_language: str,
    source_language: Optional[str],
) -> str:
    service_url = os.getenv("ANONYMIZATION_SERVICE_URL", "").strip()
    if not service_url:
        raise RuntimeError("ANONYMIZATION_SERVICE_URL is not configured")
    await ensure_anonymization_service_ready()

    chunks = chunk_text(text, TRANSLATION_QWEN_CHUNK_CHARS)
    translated_chunks: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        prompt = translation_prompt(
            chunk,
            target_language=target_language,
            source_language=source_language,
            chunk_index=index,
            chunk_count=len(chunks),
        )
        payload = {
            "model": model or DEFAULT_QWEN_TRANSLATION_MODEL,
            "prompt": f"/no_think\n{prompt}",
            "stream": False,
            "think": False,
            "format": "",
            "options": {
                "temperature": 0.0,
                "num_ctx": CITATION_QWEN_NUM_CTX,
                "num_predict": TRANSLATION_QWEN_NUM_PREDICT,
            },
        }
        async with httpx.AsyncClient(timeout=CITATION_QWEN_TIMEOUT_SEC) as client:
            response = await client.post(f"{service_url.rstrip('/')}/ollama-json", json=payload)
            response.raise_for_status()
            translated = _clean_qwen_translation_text(response.json())
        if not translated:
            raise RuntimeError(f"Qwen returned no usable text for chunk {index}/{len(chunks)}")
        translated_chunks.append(translated)
    return "\n\n".join(translated_chunks).strip()


async def translate_text(
    text: str,
    *,
    model: str,
    target_language: str,
    source_language: Optional[str],
) -> tuple[str, str]:
    provider = provider_for_model(model)
    if provider == "gemini":
        return (
            await translate_with_gemini(
                text,
                model=model,
                target_language=target_language,
                source_language=source_language,
            ),
            provider,
        )
    return (
        await translate_with_qwen(
            text,
            model=model,
            target_language=target_language,
            source_language=source_language,
        ),
        provider,
    )


def translation_text_path(translation_id: Any) -> Path:
    TRANSLATED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSLATED_TEXT_DIR / f"{translation_id}.txt"


def comparison_prompt(
    *,
    source_text: str,
    translations: List[Dict[str, str]],
    target_language: str,
) -> str:
    source_excerpt = source_text[:TRANSLATION_COMPARE_CHAR_LIMIT]
    translation_blocks = []
    remaining = TRANSLATION_COMPARE_CHAR_LIMIT
    for item in translations:
        text = item.get("text", "")
        take = max(2000, min(len(text), remaining // max(1, len(translations))))
        remaining -= take
        translation_blocks.append(
            f"## {item.get('model')}\n{text[:take]}"
        )

    return f"""Vergleiche die folgenden Übersetzungen eines anonymisierten OCR-Dokuments ins {target_language}.
Bewerte ausschließlich Übersetzungsqualität, Vollständigkeit, OCR-Robustheit, erhaltene Struktur und mögliche Bedeutungsabweichungen.
Gib ein kurzes, praktisches Ergebnis auf Deutsch:
1. Welche Übersetzung ist für anwaltliche Arbeit verlässlicher?
2. Wichtige Unterschiede oder Risiken.
3. Stellen, die man manuell prüfen sollte.

ANONYMISIERTER AUSGANGSTEXT:
{source_excerpt}

ÜBERSETZUNGEN:
{chr(10).join(translation_blocks)}
"""


async def compare_translations_with_gemini(
    *,
    source_text: str,
    translations: List[Dict[str, str]],
    target_language: str,
) -> str:
    client = get_gemini_client()
    response = await client.aio.models.generate_content(
        model=DEFAULT_GEMINI_TRANSLATION_MODEL,
        contents=comparison_prompt(
            source_text=source_text,
            translations=translations,
            target_language=target_language,
        ),
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
        ),
    )
    return (getattr(response, "text", None) or "").strip()

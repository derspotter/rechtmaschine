import asyncio
import json
import tempfile
import time
import re
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, create_model
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
import pikepdf
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
import anthropic
import traceback
import os
from openai import OpenAI
from google import genai
from google.genai import types

from shared import (
    collect_selected_document_identifiers,
    DocumentCategory,
    GenerationRequest,
    GenerationResponse,
    get_document_for_upload,
    GenerationMetadata,
    GenerationJobResponse,
    JLawyerSendRequest,
    JLawyerResponse,
    JLawyerTemplatesResponse,
    SavedSource,
    TokenUsage,
    broadcast_documents_snapshot,
    get_anthropic_client,
    get_openai_client,
    openai_file_uploads_enabled,
    resolve_openai_model,
    limiter,
    load_document_text,
    resolve_case_uuid_for_request,
    resolve_document_identifier,
    store_document_text,
    get_gemini_client,
    ensure_document_on_gemini,
    ensure_anonymization_service_ready,
)
from auth import get_current_active_user
from citation_qwen import run_citation_checks
from database import SessionLocal, get_db
from models import Document, ResearchSource, User, GeneratedDraft, GenerationJob

try:
    from agent_memory_service import get_case_memory_prompt_context
except Exception:
    get_case_memory_prompt_context = None

router = APIRouter()


GEMINI_GENERATION_MODEL = (
    os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.1-pro-preview").strip()
    or "gemini-3.1-pro-preview"
)
OPENAI_GPT5_MAX_OUTPUT_TOKENS = int(
    (os.getenv("OPENAI_GPT5_MAX_OUTPUT_TOKENS", "50000") or "50000").strip()
)

OPENAI_GPT5_REASONING_EFFORT = (
    os.getenv("OPENAI_GPT5_REASONING_EFFORT", "xhigh").strip() or "xhigh"
)
GEMINI_MAX_OUTPUT_TOKENS = int(
    (os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "12288") or "12288").strip()
)
OPENAI_GPT5_INPUT_PRICE_PER_1M = float(
    (os.getenv("OPENAI_GPT5_INPUT_PRICE_PER_1M", "2.5") or "2.5").strip()
)
OPENAI_GPT5_OUTPUT_PRICE_PER_1M = float(
    (os.getenv("OPENAI_GPT5_OUTPUT_PRICE_PER_1M", "15") or "15").strip()
)
GEMINI_PRO_INPUT_PRICE_PER_1M = float(
    (os.getenv("GEMINI_PRO_INPUT_PRICE_PER_1M", "2.5") or "2.5").strip()
)
GEMINI_PRO_OUTPUT_PRICE_PER_1M = float(
    (os.getenv("GEMINI_PRO_OUTPUT_PRICE_PER_1M", "15") or "15").strip()
)


NEUTRAL_LEGAL_TONE_RULES = (
    "TONALITÄT:\n"
    "- Schreibe nüchtern, sachlich und professionell.\n"
    "- Vermeide unnötig zugespitzte oder polemische Formulierungen.\n"
    "- Formuliere rechtliche Kritik zurückhaltend und belegt.\n"
    "- Vermeide apodiktische Gesamtwertungen zu Beginn; entwickle die Kritik aus Tatsachen und Normen.\n"
    "- Stelle Tatsachen und rechtliche Bewertung sauber getrennt dar.\n"
    "- Verwende im Endtext keine Semikolons; nutze stattdessen Punkt oder Komma.\n"
    "- Schreibe aus anwaltlicher Parteiperspektive. Mandantenseitige Tatsachen sind im Schriftsatz grundsätzlich als eigene Tatsachendarstellung im Indikativ zu formulieren, nicht mit distanzierenden Formeln wie 'laut Mandant', 'nach Angaben des Mandanten' oder 'der Kläger behauptet'.\n"
    "- Verwende attributionale Formeln wie 'Der Kläger trägt vor' nur gezielt, etwa bei neuem Tatsachenvortrag, Beweisangeboten, Glaubhaftigkeitsfragen oder wenn ausdrücklich Unsicherheit markiert werden soll.\n"
    "- Stelle keine Tatsache als bewiesen dar, wenn sie nur Parteivortrag ist. Die richtige Balance ist anwaltlich behauptend, aber nicht gerichtsfeststellend: 'Der Kläger wurde verletzt' statt 'angeblich wurde der Kläger verletzt'; bei Bedarf anschließend 'Beweis: ...'.\n\n"
)


def _get_gemini_generation_model() -> str:
    """Return the Gemini model used for generation paths.

    Kept in an env-driven helper so model rollouts can be changed without code edits.
    """
    configured = (
        os.getenv("GEMINI_GENERATION_MODEL", GEMINI_GENERATION_MODEL).strip()
        if os.getenv("GEMINI_GENERATION_MODEL") is not None
        else GEMINI_GENERATION_MODEL
    )
    return configured or "gemini-3.1-pro-preview"


def _format_stream_exception(exc: Exception) -> str:
    """Return a user-facing error message for streamed provider failures."""
    message = str(exc).strip() or exc.__class__.__name__
    request_id = getattr(exc, "request_id", None)
    status_code = getattr(exc, "status_code", None)

    if request_id and request_id not in message:
        message = f"{message} (Request-ID: {request_id})"
    if status_code and f"HTTP {status_code}" not in message:
        message = f"HTTP {status_code}: {message}"
    return message


def _extract_openai_incomplete_reason(response: Any) -> str:
    """Extract a short reason from OpenAI incomplete_details if present."""
    incomplete_details = getattr(response, "incomplete_details", None)
    if not incomplete_details:
        return ""

    reason = getattr(incomplete_details, "reason", None)
    if reason:
        return str(reason)

    if isinstance(incomplete_details, dict):
        return str(incomplete_details.get("reason") or "")

    return ""


def _estimate_openai_gpt5_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens or 0) * (OPENAI_GPT5_INPUT_PRICE_PER_1M / 1_000_000)
        + (output_tokens or 0) * (OPENAI_GPT5_OUTPUT_PRICE_PER_1M / 1_000_000)
    )


def _estimate_gemini_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens or 0) * (GEMINI_PRO_INPUT_PRICE_PER_1M / 1_000_000)
        + (output_tokens or 0) * (GEMINI_PRO_OUTPUT_PRICE_PER_1M / 1_000_000)
    )


def _merge_token_usages(*usages: Optional[TokenUsage], model: Optional[str] = None) -> Optional[TokenUsage]:
    valid = [usage for usage in usages if usage is not None]
    if not valid:
        return None

    return TokenUsage(
        input_tokens=sum(usage.input_tokens or 0 for usage in valid),
        output_tokens=sum(usage.output_tokens or 0 for usage in valid),
        thinking_tokens=sum(usage.thinking_tokens or 0 for usage in valid),
        cache_read_tokens=sum(usage.cache_read_tokens or 0 for usage in valid),
        cache_write_tokens=sum(usage.cache_write_tokens or 0 for usage in valid),
        total_tokens=sum(usage.total_tokens or 0 for usage in valid),
        cost_usd=round(sum(usage.cost_usd or 0.0 for usage in valid), 4),
        model=model or valid[-1].model,
    )


def _run_page_citation_checks(
    generated_text: str,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> tuple[Dict[str, Any], List[str]]:
    """Run deterministic page checks and return metadata plus user-facing warnings."""
    try:
        citation_checks = verify_page_citations(generated_text, collected)
    except Exception as exc:
        message = f"Deterministische Seitenprüfung der Fundstellen fehlgeschlagen: {exc}"
        print(f"[CITATION CHECK ERROR] {exc}")
        return {
            "checks": [],
            "warnings": [message],
            "summary": {},
        }, [message]

    warnings = list(citation_checks.get("warnings") or [])
    summary = citation_checks.get("summary") or {}
    problem_count = sum(
        int(summary.get(status, 0) or 0)
        for status in (
            "found_on_different_page",
            "not_found",
            "ambiguous",
            "no_page_text_available",
        )
    )
    if problem_count:
        warnings.append(
            f"Deterministische Seitenprüfung: {problem_count} Fundstelle(n) konnten nicht eindeutig auf der zitierten Seite bestätigt werden."
        )
    return citation_checks, warnings


def _summarize_page_citation_checks(checks: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "verified_on_cited_page": 0,
        "found_on_different_page": 0,
        "not_found": 0,
        "ambiguous": 0,
        "no_page_text_available": 0,
    }
    for check in checks:
        status = str(check.get("status") or "")
        if status in summary:
            summary[status] += 1
    return summary


def _parse_ollama_json_response(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = data.get("response") or data.get("thinking")
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


async def _call_qwen_json(
    service_url: str,
    prompt: str,
    *,
    num_predict: int = 700,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    payload = {
        "model": CITATION_QWEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_ctx": CITATION_QWEN_NUM_CTX,
            "num_predict": num_predict,
        },
    }
    async with httpx.AsyncClient(timeout=CITATION_QWEN_TIMEOUT_SEC) as client:
        response = await client.post(f"{service_url.rstrip('/')}/extract-entities", json=payload)
        response.raise_for_status()
        return _parse_ollama_json_response(response.json())


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _citation_document_entries(
    check: Dict[str, Any],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    document = check.get("document")
    if isinstance(document, dict):
        entries.append(document)
    for candidate in check.get("candidate_documents") or []:
        if isinstance(candidate, dict):
            entries.append(candidate)
    return entries


def _build_document_page_lookup(
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> Dict[str, Dict[int, str]]:
    return {
        document.label: document.pages
        for document in _load_document_texts(collected)
        if document.label and document.pages
    }


def _citation_source_inventory(
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> List[Dict[str, Any]]:
    inventory: List[Dict[str, Any]] = []
    for idx, document in enumerate(_load_document_texts(collected), start=1):
        if document.is_internal_note:
            continue
        pages = sorted(document.pages.keys())
        inventory.append(
            {
                "key": f"D{idx}",
                "id": document.id,
                "label": document.label,
                "category": document.category,
                "role": document.role,
                "pages": pages[:80],
                "page_start": document.page_start,
                "page_end": document.page_end,
            }
        )
    return inventory


def _split_draft_for_citation_extraction(text: str) -> List[Dict[str, Any]]:
    normalized = (text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= CITATION_QWEN_DRAFT_CHUNK_CHARS:
        return [{"index": 1, "text": normalized}]

    chunks: List[Dict[str, Any]] = []
    remaining = normalized
    idx = 1
    while remaining:
        if len(remaining) <= CITATION_QWEN_DRAFT_CHUNK_CHARS:
            chunk = remaining.strip()
            remaining = ""
        else:
            cut = remaining.rfind("\n\n", 0, CITATION_QWEN_DRAFT_CHUNK_CHARS)
            if cut < int(CITATION_QWEN_DRAFT_CHUNK_CHARS * 0.55):
                cut = remaining.rfind(". ", 0, CITATION_QWEN_DRAFT_CHUNK_CHARS)
            if cut < int(CITATION_QWEN_DRAFT_CHUNK_CHARS * 0.55):
                cut = CITATION_QWEN_DRAFT_CHUNK_CHARS
            chunk = remaining[:cut].strip()
            remaining = remaining[cut:].strip()
        if chunk:
            chunks.append({"index": idx, "text": chunk})
            idx += 1
    return chunks


def _safe_int_list(value: Any) -> List[int]:
    if isinstance(value, int):
        return [value] if value > 0 else []
    if isinstance(value, str):
        numbers = re.findall(r"\d+", value)
        return [int(number) for number in numbers if int(number) > 0]
    if isinstance(value, list):
        pages: List[int] = []
        for item in value:
            try:
                page = int(item)
            except Exception:
                continue
            if page > 0 and page not in pages:
                pages.append(page)
        return pages
    return []


def _normalize_source_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _extract_page_numbers_from_citation_text(citation_text: str) -> List[int]:
    text = citation_text or ""
    match = re.search(r"\b(?:S\.|Seite|page|p\.)\s*(\d+)\s*(f\.|ff\.)?", text, re.IGNORECASE)
    if not match:
        match = re.search(r"\bBl\.?\s*(\d+)\s*(f\.|ff\.)?\s*d\.?\s*A\.?", text, re.IGNORECASE)
    if not match:
        return []
    page = int(match.group(1))
    suffix = (match.group(2) or "").strip().lower()
    if suffix == "f.":
        return [page, page + 1]
    if suffix == "ff.":
        return [page, page + 1, page + 2]
    return [page]


def _resolve_qwen_citation_document(
    citation: Dict[str, Any],
    inventory: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    by_key = {item["key"]: item for item in inventory}
    source_key = _normalize_source_key(citation.get("source_key"))
    if source_key in by_key:
        return by_key[source_key]

    citation_text = str(citation.get("citation_text") or "")
    hint = _normalize_for_prompt_match(
        " ".join(
            str(citation.get(key) or "")
            for key in ("source_hint", "document_hint", "citation_text", "sentence")
        )
    )
    if re.search(r"anlage\s*k\s*2", citation_text, re.IGNORECASE):
        primary = next(
            (
                item for item in inventory
                if item.get("category") == "bescheid" and item.get("role") == "primary"
            ),
            None,
        )
        if primary:
            return primary

    if re.search(r"\bbl\.?\s*\d+", citation_text, re.IGNORECASE):
        pages = _safe_int_list(citation.get("page_numbers")) or _extract_page_numbers_from_citation_text(citation_text)
        page_covering = [
            item for item in inventory
            if item.get("page_start")
            and item.get("page_end")
            and any(int(item["page_start"]) <= page <= int(item["page_end"]) for page in pages)
        ]
        if len(page_covering) == 1:
            return page_covering[0]
        akte = [item for item in inventory if item.get("category") == "akte"]
        if len(akte) == 1:
            return akte[0]

    label_matches = [
        item for item in inventory
        if _normalize_for_prompt_match(item.get("label")) in hint
    ]
    if len(label_matches) == 1:
        return label_matches[0]

    category_keywords = {
        "bescheid": ("bescheid", "ablehnungsbescheid", "widerspruchsbescheid"),
        "anhoerung": ("anhoerung", "anhörung", "anhörung", "interview"),
        "vorinstanz": ("urteil", "beschluss", "vorinstanz"),
        "rechtsprechung": ("entscheidung", "urteil", "beschluss"),
        "sonstiges": ("schreiben", "nachweis", "unterlage", "attest"),
    }
    for category, keywords in category_keywords.items():
        if any(_normalize_for_prompt_match(keyword) in hint for keyword in keywords):
            matches = [item for item in inventory if item.get("category") == category]
            if len(matches) == 1:
                return matches[0]

    if len(inventory) == 1:
        return inventory[0]
    return None


def _normalize_for_prompt_match(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


async def _extract_citations_from_chunk_with_qwen(
    service_url: str,
    chunk: Dict[str, Any],
    inventory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    inventory_for_prompt = [
        {
            "key": item["key"],
            "label": item["label"],
            "category": item["category"],
            "role": item["role"],
            "pages": item["pages"],
        }
        for item in inventory
    ]
    prompt = f"""/no_think
Du extrahierst Fundstellen aus einem deutschen juristischen Entwurf.

Aufgabe:
- Lies nur den ENTWURF-AUSSCHNITT.
- Finde jede Tatsachenbehauptung oder Beweisbehauptung, die mit einer Seitenfundstelle belegt wird.
- Extrahiere nicht bloß "S. 1" isoliert, sondern den tragenden Behauptungssatz dazu.
- Ordne die Fundstelle möglichst einem Dokument aus der DOKUMENTLISTE zu.
- Wenn die Quelle nicht sicher bestimmbar ist, setze source_key auf null und erkläre den Hinweis in source_hint.
- Ignoriere Gesetzesverweise wie "§ 77 Abs. 2 AsylG, S. 2".
- Gib nur JSON zurück.

JSON-Schema:
{{
  "citations": [
    {{
      "id": "c1",
      "claim": "kurze zu prüfende Tatsachenbehauptung",
      "sentence": "voller Satz aus dem Entwurf",
      "citation_text": "exakte Fundstelle, z.B. Anlage K2, S. 3",
      "source_key": "D1 oder null",
      "source_hint": "Wortlaut im Entwurf, der auf das Dokument hinweist",
      "page_numbers": [3]
    }}
  ]
}}

DOKUMENTLISTE:
{json.dumps(inventory_for_prompt, ensure_ascii=False)}

ENTWURF-AUSSCHNITT #{chunk.get("index")}:
{chunk.get("text") or ""}
"""
    parsed = await _call_qwen_json(service_url, prompt, num_predict=2200, temperature=0.0)
    raw_citations = parsed.get("citations") or parsed.get("items") or []
    if not isinstance(raw_citations, list):
        return []

    extracted: List[Dict[str, Any]] = []
    for item_idx, raw in enumerate(raw_citations, start=1):
        if not isinstance(raw, dict):
            continue
        citation_text = str(raw.get("citation_text") or raw.get("citation") or "").strip()
        claim = str(raw.get("claim") or "").strip()
        sentence = str(raw.get("sentence") or claim).strip()
        if not citation_text or not claim:
            continue
        pages = _safe_int_list(raw.get("page_numbers")) or _extract_page_numbers_from_citation_text(citation_text)
        extracted.append(
            {
                "id": f"chunk{chunk.get('index')}_c{item_idx}",
                "claim": claim,
                "sentence": sentence,
                "citation_text": citation_text,
                "source_key": raw.get("source_key"),
                "source_hint": raw.get("source_hint") or "",
                "page_numbers": pages,
                "chunk_index": chunk.get("index"),
            }
        )
    return extracted


async def _check_citation_group_with_qwen(
    service_url: str,
    group: Dict[str, Any],
) -> List[Dict[str, Any]]:
    citations = [
        {
            "id": item["id"],
            "claim": item.get("claim") or "",
            "sentence": item.get("sentence") or "",
            "citation_text": item.get("citation_text") or "",
        }
        for item in group.get("citations") or []
    ]
    prompt = f"""/no_think
Du prüfst juristische Fundstellen seitenbezogen.

Aufgabe:
- Nutze ausschließlich den TEXT DER ZITIERTEN SEITE(N).
- Prüfe jede CITATION einzeln.
- verdict = "yes", wenn derselbe Tatsachengehalt auf den zitierten Seiten steht, auch bei anderer Formulierung.
- verdict = "no", wenn der Tatsachengehalt fehlt oder widersprochen wird.
- verdict = "unclear", wenn OCR/Text unklar ist oder die Behauptung nicht sicher entscheidbar ist.
- Prüfe nicht, ob die Behauptung irgendwo anders im Dokument steht.
- Gib nur JSON zurück.

JSON-Schema:
{{
  "results": [
    {{"id":"chunk1_c1","verdict":"yes|no|unclear","confidence":0.0,"reason":"kurz"}}
  ]
}}

DOKUMENT:
{group.get("document_label")}

ZITIERTE SEITEN:
{group.get("page_numbers")}

CITATIONS:
{json.dumps(citations, ensure_ascii=False)}

TEXT DER ZITIERTEN SEITE(N):
{str(group.get("page_text") or "")[:CITATION_QWEN_PAGE_CHAR_LIMIT]}
"""
    parsed = await _call_qwen_json(service_url, prompt, num_predict=1400, temperature=0.0)
    raw_results = parsed.get("results") or parsed.get("citations") or []
    if isinstance(raw_results, dict):
        raw_results = [raw_results]
    if not isinstance(raw_results, list):
        return []
    results: List[Dict[str, Any]] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        verdict = str(raw.get("verdict") or raw.get("answer") or "unclear").strip().lower()
        if verdict not in {"yes", "no", "unclear"}:
            verdict = "unclear"
        confidence = _safe_float(raw.get("confidence"))
        if "confidence" not in raw and verdict in {"yes", "no"}:
            confidence = 0.5
        results.append(
            {
                "id": str(raw.get("id") or "").strip(),
                "verdict": verdict,
                "confidence": confidence,
                "reason": str(raw.get("reason") or "")[:500],
                "model": CITATION_QWEN_MODEL,
            }
        )
    return results


def _status_from_qwen_verdict(verdict: str) -> str:
    if verdict == "yes":
        return "verified_on_cited_page"
    if verdict == "no":
        return "not_found"
    return "ambiguous"


async def _run_qwen_extracted_citation_checks(
    generated_text: str,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> tuple[Optional[Dict[str, Any]], List[str]]:
    if not CITATION_QWEN_VERIFICATION_ENABLED:
        return None, []
    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        return None, ["Qwen-Fundstellenprüfung übersprungen: ANONYMIZATION_SERVICE_URL ist nicht konfiguriert."]

    inventory = _citation_source_inventory(collected)
    if not inventory:
        return None, []

    try:
        await ensure_anonymization_service_ready()
    except Exception as exc:
        message = f"Qwen-Fundstellenprüfung übersprungen: service_manager nicht erreichbar ({exc})."
        print(f"[CITATION QWEN WARN] {message}")
        return None, [message]

    chunks = _split_draft_for_citation_extraction(generated_text)
    extracted: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for chunk in chunks:
        try:
            extracted.extend(await _extract_citations_from_chunk_with_qwen(service_url, chunk, inventory))
        except Exception as exc:
            message = f"Qwen-Fundstellenextraktion für Entwurfsabschnitt {chunk.get('index')} fehlgeschlagen: {exc}"
            print(f"[CITATION QWEN WARN] {message}")
            warnings.append(message)

    if not extracted:
        return {
            "checks": [],
            "warnings": warnings,
            "summary": _summarize_page_citation_checks([]),
            "qwen_extraction": {
                "enabled": True,
                "model": CITATION_QWEN_MODEL,
                "chunks": len(chunks),
                "extracted": 0,
            },
        }, warnings

    page_lookup = _build_document_page_lookup(collected)
    checks: List[Dict[str, Any]] = []
    groups: Dict[str, Dict[str, Any]] = {}
    by_id: Dict[str, Dict[str, Any]] = {}
    for citation in extracted:
        pages = _safe_int_list(citation.get("page_numbers")) or _extract_page_numbers_from_citation_text(citation.get("citation_text") or "")
        document = _resolve_qwen_citation_document(citation, inventory)
        base_check = {
            "id": citation["id"],
            "citation": citation.get("citation_text") or "",
            "claim": citation.get("claim") or "",
            "sentence": citation.get("sentence") or "",
            "cited_pages": pages,
            "kind": "qwen_extracted",
            "source_hint": citation.get("source_hint") or "",
            "chunk_index": citation.get("chunk_index"),
        }
        if not document:
            check = {
                **base_check,
                "status": "ambiguous",
                "reason": "Qwen konnte die Fundstelle keinem ausgewählten Dokument eindeutig zuordnen.",
            }
            checks.append(check)
            by_id[citation["id"]] = check
            continue
        check = {
            **base_check,
            "document": {
                "id": document.get("id"),
                "label": document.get("label"),
                "category": document.get("category"),
                "role": document.get("role"),
            },
        }
        if not pages:
            check["status"] = "ambiguous"
            check["reason"] = "Qwen hat keine zitierte Seitenzahl extrahiert."
            checks.append(check)
            by_id[citation["id"]] = check
            continue
        document_pages = page_lookup.get(str(document.get("label") or ""), {})
        page_text = "\n\n".join(document_pages.get(page, "") for page in pages).strip()
        if not page_text:
            check["status"] = "no_page_text_available"
            check["reason"] = "Für die von Qwen extrahierte zitierte Seite liegt kein Text vor."
            checks.append(check)
            by_id[citation["id"]] = check
            continue
        check["status"] = "ambiguous"
        check["reason"] = "Qwen-Seitenprüfung ausstehend."
        checks.append(check)
        by_id[citation["id"]] = check
        group_key = f"{document.get('id')}:{','.join(str(page) for page in pages)}"
        group = groups.setdefault(
            group_key,
            {
                "document_id": document.get("id"),
                "document_label": document.get("label"),
                "page_numbers": pages,
                "page_text": page_text,
                "citations": [],
            },
        )
        group["citations"].append(citation)

    checked_groups = 0
    checked_citations = 0
    for group in groups.values():
        try:
            results = await _check_citation_group_with_qwen(service_url, group)
        except Exception as exc:
            message = f"Qwen-Seitenprüfung für {group.get('document_label')} S. {group.get('page_numbers')} fehlgeschlagen: {exc}"
            print(f"[CITATION QWEN WARN] {message}")
            warnings.append(message)
            continue
        checked_groups += 1
        result_by_id = {result["id"]: result for result in results if result.get("id")}
        for citation in group.get("citations") or []:
            check = by_id.get(citation["id"])
            if not check:
                continue
            judgment = result_by_id.get(citation["id"])
            if not judgment:
                check["status"] = "ambiguous"
                check["reason"] = "Qwen-Seitenprüfung lieferte kein Ergebnis für diese Fundstelle."
                continue
            checked_citations += 1
            check["qwen_page_judgment"] = judgment
            check["status"] = _status_from_qwen_verdict(judgment.get("verdict") or "unclear")
            if check["status"] == "verified_on_cited_page":
                check["reason"] = "Qwen bestätigt den Tatsachengehalt auf der zitierten Seite."
            elif check["status"] == "not_found":
                check["reason"] = "Qwen findet den Tatsachengehalt nicht auf der zitierten Seite."
            else:
                check["reason"] = "Qwen konnte die Fundstelle nicht sicher entscheiden."

    result = {
        "checks": checks,
        "warnings": warnings,
        "summary": _summarize_page_citation_checks(checks),
        "qwen_extraction": {
            "enabled": True,
            "model": CITATION_QWEN_MODEL,
            "chunks": len(chunks),
            "extracted": len(extracted),
            "checked_groups": checked_groups,
            "checked_citations": checked_citations,
            "draft_chunk_chars": CITATION_QWEN_DRAFT_CHUNK_CHARS,
            "page_char_limit": CITATION_QWEN_PAGE_CHAR_LIMIT,
        },
    }
    return result, warnings


def _qwen_citation_samples(
    citation_checks: Dict[str, Any],
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> List[Dict[str, Any]]:
    page_lookup = _build_document_page_lookup(collected)
    samples: List[Dict[str, Any]] = []
    for idx, check in enumerate(citation_checks.get("checks") or []):
        status = str(check.get("status") or "")
        if status not in {"ambiguous", "not_found", "found_on_different_page"}:
            continue
        pages = [int(page) for page in check.get("cited_pages") or [] if str(page).isdigit()]
        if not pages:
            continue
        for document in _citation_document_entries(check):
            label = str(document.get("label") or "").strip()
            if not label or label not in page_lookup:
                continue
            page_text = "\n\n".join(page_lookup[label].get(page, "") for page in pages).strip()
            if not page_text:
                continue
            samples.append(
                {
                    "check_index": idx,
                    "document_label": label,
                    "page_numbers": pages,
                    "page_text": page_text[:CITATION_QWEN_PAGE_CHAR_LIMIT],
                    "deterministic_status": status,
                    "claim": check.get("claim") or "",
                    "sentence": check.get("sentence") or "",
                    "citation": check.get("citation") or "",
                }
            )
            break
        if len(samples) >= CITATION_QWEN_MAX_CHECKS:
            break
    return samples


async def _judge_citation_page_with_qwen(
    service_url: str,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = f"""/no_think
Du bist ein strenger Prüfer für juristische Fundstellen.

Aufgabe: Entscheide, ob die AUSSAGE durch den TEXT DER ZITIERTEN SEITE gestützt wird.

Regeln:
- Nutze ausschließlich den TEXT DER ZITIERTEN SEITE.
- Antworte "yes", wenn derselbe Tatsachengehalt auf der Seite steht, auch bei anderer Formulierung.
- Antworte "no", wenn die Aussage fehlt oder widersprochen wird.
- Antworte "unclear", wenn OCR/Text zu unklar ist oder die Seite nicht sicher reicht.
- Prüfe nicht, ob die Aussage irgendwo anders im Dokument steht.
- Gib nur JSON mit genau diesen Schlüsseln zurück:
  {{"verdict":"yes|no|unclear","confidence":0.0,"reason":"kurze Begründung"}}
- Nutze nicht den Schlüssel "answer".

AUSSAGE:
{sample.get("claim") or ""}

GANZER SATZ:
{sample.get("sentence") or ""}

FUNDSTELLE:
{sample.get("citation") or ""}

TEXT DER ZITIERTEN SEITE:
{sample.get("page_text") or ""}
"""
    payload = {
        "model": CITATION_QWEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_ctx": CITATION_QWEN_NUM_CTX,
            "num_predict": 300,
        },
    }
    async with httpx.AsyncClient(timeout=CITATION_QWEN_TIMEOUT_SEC) as client:
        response = await client.post(f"{service_url.rstrip('/')}/extract-entities", json=payload)
        response.raise_for_status()
        parsed = _parse_ollama_json_response(response.json())

    verdict = str(parsed.get("verdict") or parsed.get("answer") or "unclear").strip().lower()
    if verdict not in {"yes", "no", "unclear"}:
        verdict = "unclear"
    confidence = _safe_float(parsed.get("confidence"))
    if "confidence" not in parsed and verdict in {"yes", "no"}:
        confidence = 0.5
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "")[:500],
        "model": CITATION_QWEN_MODEL,
        "document_label": sample.get("document_label"),
        "page_numbers": sample.get("page_numbers") or [],
    }


def _apply_qwen_citation_judgments(
    citation_checks: Dict[str, Any],
    judgments: Dict[int, Dict[str, Any]],
) -> None:
    checks = citation_checks.get("checks") or []
    for idx, judgment in judgments.items():
        if idx < 0 or idx >= len(checks):
            continue
        check = checks[idx]
        deterministic_status = str(check.get("status") or "")
        verdict = str(judgment.get("verdict") or "unclear")
        check["deterministic_status"] = deterministic_status
        check["qwen_page_judgment"] = judgment
        if verdict == "yes":
            check["status"] = "verified_on_cited_page"
            check["reason"] = "Qwen-Seitenprüfung bestätigt den Tatsachengehalt auf der zitierten Seite."
        elif verdict == "no":
            if deterministic_status == "found_on_different_page":
                check["status"] = "found_on_different_page"
                check["reason"] = "Qwen-Seitenprüfung bestätigt keinen Support auf der zitierten Seite; deterministischer Treffer liegt auf anderer Seite."
            else:
                check["status"] = "not_found"
                check["reason"] = "Qwen-Seitenprüfung findet den Tatsachengehalt nicht auf der zitierten Seite."
        else:
            check["status"] = "ambiguous"
            check["reason"] = "Qwen-Seitenprüfung konnte die Fundstelle nicht sicher entscheiden."
    citation_checks["summary"] = _summarize_page_citation_checks(checks)


async def _run_qwen_page_citation_judgments(
    citation_checks: Dict[str, Any],
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> List[str]:
    if not CITATION_QWEN_VERIFICATION_ENABLED:
        return []
    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        return ["Qwen-Seitenprüfung übersprungen: ANONYMIZATION_SERVICE_URL ist nicht konfiguriert."]

    samples = _qwen_citation_samples(citation_checks, collected)
    if not samples:
        return []

    try:
        await ensure_anonymization_service_ready()
    except Exception as exc:
        message = f"Qwen-Seitenprüfung übersprungen: service_manager nicht erreichbar ({exc})."
        print(f"[CITATION QWEN WARN] {message}")
        return [message]

    judgments: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []
    for sample in samples:
        try:
            judgment = await _judge_citation_page_with_qwen(service_url, sample)
            judgments[int(sample["check_index"])] = judgment
        except Exception as exc:
            message = f"Qwen-Seitenprüfung für Fundstelle '{sample.get('citation')}' fehlgeschlagen: {exc}"
            print(f"[CITATION QWEN WARN] {message}")
            warnings.append(message)

    if judgments:
        _apply_qwen_citation_judgments(citation_checks, judgments)
        citation_checks["qwen_page_judge"] = {
            "enabled": True,
            "model": CITATION_QWEN_MODEL,
            "checked": len(judgments),
            "max_checks": CITATION_QWEN_MAX_CHECKS,
            "page_char_limit": CITATION_QWEN_PAGE_CHAR_LIMIT,
        }
    return warnings


async def _run_page_citation_checks_with_qwen(
    generated_text: str,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> tuple[Dict[str, Any], List[str]]:
    citation_checks, deterministic_warnings = _run_page_citation_checks(generated_text, collected)
    qwen_warnings = await _run_qwen_page_citation_judgments(citation_checks, collected)
    warnings = list(citation_checks.get("warnings") or [])
    summary = citation_checks.get("summary") or {}
    problem_count = sum(
        int(summary.get(status, 0) or 0)
        for status in (
            "found_on_different_page",
            "not_found",
            "ambiguous",
            "no_page_text_available",
        )
    )
    deterministic_problem_warning = next(
        (
            warning
            for warning in deterministic_warnings
            if warning.startswith("Deterministische Seitenprüfung:")
        ),
        "",
    )
    if problem_count and not deterministic_problem_warning:
        warnings.append(
            f"Seitenprüfung: {problem_count} Fundstelle(n) konnten nicht eindeutig auf der zitierten Seite bestätigt werden."
        )
    warnings.extend(qwen_warnings)
    return citation_checks, warnings


def _build_senior_partner_critique_prompt() -> str:
    return (
        "Du bist ein Senior Partner einer Top-Kanzlei, bekannt für extrem strenge Qualitätskontrolle.\n"
        "Analysiere den folgenden Entwurf gnadenlos auf:\n"
        "1. HALLUZINATIONEN: Prüfe jedes zitierte Urteil. Sieht das Aktenzeichen echt aus? Gibt es das Gericht?\n"
        "2. LOGIK: Ist die juristische Argumentation schlüssig? Gibt es Sprünge?\n"
        "3. TONALITÄT: Ist der Schriftsatz professionell und überzeugend?\n"
        "4. INTERNE KANZLEINOTIZEN: Beanstande jede Formulierung, die interne Kanzleinotizen, Gesprächsnotizen, Transkripte oder Besprechungsvermerke als Quelle, Anlage oder Fundstelle zitiert. Solche Notizen dürfen nur in Parteivortrag und Beweisangebote transformiert werden.\n"
        "Liste konkrete Mängel auf. Sei pedantisch."
    )


def _build_gemini_finalize_system_prompt() -> str:
    return (
        "Du bist ein erfahrener Fachanwalt. Deine Aufgabe ist die Einarbeitung der Kritik des Senior Partners.\n"
        "VORGEHENSWEISE:\n"
        "1. Lies die KRITIK sorgfältig.\n"
        "2. Überarbeite den ENTWURF: Korrigiere jeden kritisierten Punkt.\n"
        "3. Halluzinationen entfernen: Wenn der Senior Partner ein Urteil anzweifelt, LÖSCHE es oder ersetze es durch eine allgemeine Formulierung.\n"
        "4. Behalte die XML-Tags (<strategy> usw.) NICHT bei - nur den reinen juristischen Text.\n\n"
        "INTERNE KANZLEINOTIZEN:\n"
        "Falls der Entwurf interne Kanzleinotizen, Gesprächsnotizen, Transkripte oder Besprechungsvermerke als Quelle zitiert, entferne diese Zitate. "
        "Wandle ihren Inhalt stattdessen in klägerischen Vortrag, Beweisangebote oder anwaltliche Subsumtion um. "
        "Nenne sie nicht als Anlage, Quelle, Aktennotiz oder Fundstelle.\n\n"
        "WICHTIG: Verwende KEINE Markdown-Formatierung für Überschriften (wie **Fett** oder ##). "
        "Nutze stattdessen normale Absätze und Leerzeilen zur Gliederung.\n\n"
        f"{NEUTRAL_LEGAL_TONE_RULES}"
        "FASSUNG & LÄNGE (VERBOSITY: HIGH):\n"
        "- Die Argumentation muss ausführlich und tiefgehend sein.\n"
        "- Nutze die volle verfügbare Länge (bis zu 12.000 Token), um den Sachverhalt umfassend zu würdigen."
    )


def _document_to_context_dict(doc) -> Dict[str, Optional[str]]:
    """Convert a Document ORM instance into a context dictionary used for prompting."""
    from pathlib import Path
    stored_path = Path(doc.file_path) if doc.file_path else None
    if stored_path and not stored_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Dokument {doc.filename} wurde nicht auf dem Server gefunden")

    return {
        "id": str(doc.id),
        "filename": doc.filename,
        "category": doc.category,
        "file_path": str(stored_path) if stored_path else None,
        "confidence": doc.confidence,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "explanation": doc.explanation,
        "extracted_text_path": doc.extracted_text_path,
        "ocr_applied": doc.ocr_applied,
        "needs_ocr": doc.needs_ocr,
        "anonymization_metadata": doc.anonymization_metadata,
        "is_anonymized": doc.is_anonymized,
    }


_INTERNAL_NOTE_FILENAME_RE = re.compile(
    r"(^|[_\-\s])(notiz|notizen|aktennotiz|besprechungsnotiz|gespraechsnotiz|gesprächsnotiz|"
    r"vermerk|memo|transkript|transcript)([_\-\s.]|$)",
    re.IGNORECASE,
)


def _is_internal_note_entry(entry: Dict[str, Optional[str]]) -> bool:
    """Return True for Kanzlei-internal notes that must not be cited as sources.

    These files are useful factual input for drafting party submissions, but they are
    not official evidence or court-file documents. The model may transform their
    contents into client-side Vortrag and Beweisangebote, but must not cite them.
    """
    name = entry.get("filename") or entry.get("title") or ""
    explanation = entry.get("explanation") or ""
    haystack = f"{name}\n{explanation}"
    return bool(_INTERNAL_NOTE_FILENAME_RE.search(haystack))


def _validate_category(doc, expected_category: str) -> None:
    """Ensure a document matches the expected category."""
    if doc.category != expected_category:
        raise HTTPException(
            status_code=400,
            detail=f"Dokument {doc.filename} gehört zur Kategorie '{doc.category}', erwartet war '{expected_category}'",
        )


def _collect_selected_documents(
    selection,
    db: Session,
    current_user: User,
    target_case_id: Optional[uuid.UUID],
    require_bescheid: bool = True,
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    """Validate and collect selected documents for generation.

    Supports both legacy filename selectors and new UUID-based selectors.
    """
    bescheid_selection = selection.bescheid
    total_selected = len(collect_selected_document_identifiers(selection)) + len(
        selection.saved_sources or []
    )
    if total_selected == 0:
        raise HTTPException(status_code=400, detail="Bitte wählen Sie mindestens ein Dokument aus")

    collected: Dict[str, List[Dict[str, Optional[str]]]] = {
        "anhoerung": [],
        "bescheid": [],
        "vorinstanz": [],
        "rechtsprechung": [],
        "saved_sources": [],
        "sonstiges": [],
        "internal_notes": [],
        "akte": [],
    }

    def resolve_required_document(identifier: Optional[str], expected_category: str) -> Document:
        doc = resolve_document_identifier(db, current_user, target_case_id, identifier or "")
        if not doc:
            raise HTTPException(status_code=404, detail=f"Dokument nicht gefunden: {identifier}")
        _validate_category(doc, expected_category)
        return doc

    def append_unique_entry(
        bucket: str,
        doc: Document,
        role: Optional[str] = None,
        seen_ids: Optional[set] = None,
    ) -> None:
        if seen_ids is not None and doc.id in seen_ids:
            return
        entry = _document_to_context_dict(doc)
        if role:
            entry["role"] = role
        collected[bucket].append(entry)
        if seen_ids is not None:
            seen_ids.add(doc.id)

    if require_bescheid and not (bescheid_selection.primary_id or bescheid_selection.primary):
        raise HTTPException(
            status_code=400,
            detail="Bitte markieren Sie einen Bescheid als Hauptbescheid (Anlage K2)",
        )

    anhoerung_seen = set()
    for identifier in list(selection.anhoerung or []) + list(getattr(selection, "anhoerung_ids", []) or []):
        append_unique_entry(
            "anhoerung",
            resolve_required_document(identifier, DocumentCategory.ANHOERUNG.value),
            seen_ids=anhoerung_seen,
        )

    bescheid_seen = set()
    primary_bescheid_identifier = bescheid_selection.primary_id or bescheid_selection.primary
    if primary_bescheid_identifier:
        append_unique_entry(
            "bescheid",
            resolve_required_document(primary_bescheid_identifier, DocumentCategory.BESCHEID.value),
            role="primary",
            seen_ids=bescheid_seen,
        )
    for identifier in list(getattr(bescheid_selection, "other_ids", []) or []) + list(bescheid_selection.others or []):
        append_unique_entry(
            "bescheid",
            resolve_required_document(identifier, DocumentCategory.BESCHEID.value),
            role="secondary",
            seen_ids=bescheid_seen,
        )

    vorinstanz_seen = set()
    primary_vorinstanz_identifier = selection.vorinstanz.primary_id or selection.vorinstanz.primary
    if primary_vorinstanz_identifier:
        append_unique_entry(
            "vorinstanz",
            resolve_required_document(primary_vorinstanz_identifier, DocumentCategory.VORINSTANZ.value),
            role="primary",
            seen_ids=vorinstanz_seen,
        )
    for identifier in list(getattr(selection.vorinstanz, "other_ids", []) or []) + list(selection.vorinstanz.others or []):
        append_unique_entry(
            "vorinstanz",
            resolve_required_document(identifier, DocumentCategory.VORINSTANZ.value),
            role="secondary",
            seen_ids=vorinstanz_seen,
        )

    rechtsprechung_seen = set()
    for identifier in list(selection.rechtsprechung or []) + list(getattr(selection, "rechtsprechung_ids", []) or []):
        append_unique_entry(
            "rechtsprechung",
            resolve_required_document(identifier, DocumentCategory.RECHTSPRECHUNG.value),
            seen_ids=rechtsprechung_seen,
        )

    akte_seen = set()
    for identifier in list(selection.akte or []) + list(getattr(selection, "akte_ids", []) or []):
        append_unique_entry(
            "akte",
            resolve_required_document(identifier, DocumentCategory.AKTE.value),
            seen_ids=akte_seen,
        )

    sonstiges_seen = set()
    for identifier in list(selection.sonstiges or []) + list(getattr(selection, "sonstiges_ids", []) or []):
        doc = resolve_required_document(identifier, DocumentCategory.SONSTIGES.value)
        entry = _document_to_context_dict(doc)
        bucket = "internal_notes" if _is_internal_note_entry(entry) else "sonstiges"
        if doc.id in sonstiges_seen:
            continue
        collected[bucket].append(entry)
        sonstiges_seen.add(doc.id)

    collected_sources = []
    for source_id in selection.saved_sources or []:
        try:
            source_uuid = uuid.UUID(source_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Ungültige Quellen-ID (erwartet UUID, erhalten '{source_id}')",
            )
        source = (
            db.query(ResearchSource)
            .filter(
                ResearchSource.id == source_uuid,
                ResearchSource.owner_id == current_user.id,
                ResearchSource.case_id == target_case_id,
            )
            .first()
        )
        if not source:
            raise HTTPException(status_code=404, detail=f"Quelle {source_id} wurde nicht gefunden")
        collected_sources.append(
            {
                "id": str(source.id),
                "title": source.title,
                "url": source.url,
                "description": source.description,
                "document_type": source.document_type,
                "category": "saved_source",
                "download_path": source.download_path,
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "gemini_file_uri": source.gemini_file_uri,
            }
        )
    collected["saved_sources"] = collected_sources

    return collected


def _prepare_generation_inputs(
    body: GenerationRequest,
    db: Session,
    current_user: User,
) -> Dict[str, Any]:
    """Resolve case/document context and prompts for generation flows."""
    target_case_id = resolve_case_uuid_for_request(db, current_user, body.case_id)
    collected = _collect_selected_documents(
        body.selected_documents,
        db,
        current_user,
        target_case_id,
        require_bescheid=False,
    )

    document_entries: List[Dict[str, Any]] = []
    for _, items in collected.items():
        document_entries.extend(items)

    resolved_legal_area = (getattr(body, "legal_area", None) or "migrationsrecht").lower()
    explicit_field_set = getattr(body, "model_fields_set", None) or getattr(body, "__fields_set__", set()) or set()
    legal_area_explicit = "legal_area" in explicit_field_set

    context_summary = _summarize_selection_for_prompt(collected)
    case_memory_context = ""
    if target_case_id and get_case_memory_prompt_context:
        try:
            case_memory_context = (
                get_case_memory_prompt_context(
                    db,
                    current_user,
                    target_case_id,
                    include_strategy=True,
                )
                or ""
            ).strip()
        except Exception as exc:
            print(f"[WARN] Failed to load case memory context for generation: {exc}")
            case_memory_context = ""
    case_memory_block = (
        f"KOMPAKTES FALLGEDÄCHTNIS:\n{case_memory_context}\n\n"
        if case_memory_context
        else ""
    )
    if case_memory_block:
        context_summary = f"{case_memory_block}{context_summary}"
    primary_bescheid_entry = next(
        (entry for entry in collected.get("bescheid", []) if entry.get("role") == "primary"),
        None,
    )
    primary_bescheid_label = (
        (primary_bescheid_entry.get("filename") or "—") if primary_bescheid_entry else "—"
    )
    primary_bescheid_description = ""
    if primary_bescheid_entry:
        explanation = primary_bescheid_entry.get("explanation")
        if explanation:
            primary_bescheid_description = explanation.strip()

    if body.chat_history:
        sys_p, _ = _build_generation_prompts(
            body,
            collected,
            primary_bescheid_label,
            primary_bescheid_description,
            context_summary,
        )
        system_prompt = sys_p
        prompt_for_generation = f"{case_memory_block}{body.user_prompt}" if case_memory_block else body.user_prompt
    else:
        system_prompt, prompt_for_generation = _build_generation_prompts(
            body,
            collected,
            primary_bescheid_label,
            primary_bescheid_description,
            context_summary,
        )

    return {
        "target_case_id": target_case_id,
        "collected": collected,
        "document_entries": document_entries,
        "system_prompt": system_prompt,
        "prompt_for_generation": prompt_for_generation,
        "resolved_legal_area": resolved_legal_area,
        "legal_area_explicit": legal_area_explicit,
        "primary_bescheid_entry": primary_bescheid_entry,
    }


def _persist_generated_draft(
    db: Session,
    current_user: User,
    body: GenerationRequest,
    target_case_id: Optional[uuid.UUID],
    collected: Dict[str, List[Dict[str, Optional[str]]]],
    primary_bescheid_entry: Optional[Dict[str, Optional[str]]],
    resolved_legal_area: str,
    generated_text: str,
    thinking_text: str,
    token_usage_acc: Optional[TokenUsage],
    citation_checks: Optional[Dict[str, Any]] = None,
) -> Optional[GeneratedDraft]:
    structured_used_documents = []
    for cat, entries in collected.items():
        for entry in entries:
            fname = entry.get("filename") or entry.get("title")
            if fname:
                payload = {"filename": fname, "category": cat}
                role = entry.get("role")
                if role:
                    payload["role"] = role
                structured_used_documents.append(payload)

    draft = GeneratedDraft(
        user_id=current_user.id,
        primary_document_id=uuid.UUID(primary_bescheid_entry["id"]) if primary_bescheid_entry else None,
        case_id=target_case_id,
        document_type=body.document_type,
        user_prompt=body.user_prompt,
        generated_text=generated_text,
        model_used=body.model,
        metadata_={
            "tokens": token_usage_acc.total_tokens if token_usage_acc else 0,
            "estimated_cost_usd": token_usage_acc.cost_usd if token_usage_acc else None,
            "token_usage": token_usage_acc.model_dump() if token_usage_acc else None,
            "used_documents": structured_used_documents,
            "resolved_legal_area": resolved_legal_area,
            "thinking_text": thinking_text,
            "citation_checks": citation_checks or {},
        },
    )
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return draft


async def _execute_generation_request(
    body: GenerationRequest,
    db: Session,
    current_user: User,
) -> Dict[str, Any]:
    """Execute generation without browser streaming and return structured result."""
    prepared = _prepare_generation_inputs(body, db, current_user)
    target_case_id = prepared["target_case_id"]
    collected = prepared["collected"]
    document_entries = prepared["document_entries"]
    system_prompt = prepared["system_prompt"]
    prompt_for_generation = prepared["prompt_for_generation"]
    resolved_legal_area = prepared["resolved_legal_area"]
    legal_area_explicit = prepared["legal_area_explicit"]
    primary_bescheid_entry = prepared["primary_bescheid_entry"]

    generated_text = ""
    thinking_text = ""
    token_usage_acc: Optional[TokenUsage] = None
    gemini_files = None

    if body.model in {"two-step-expert", "multi-step-expert"}:
        if body.chat_history:
            client = get_gemini_client()
            files = _upload_documents_to_gemini(client, document_entries)
            default_model = _get_gemini_generation_model()
            generated_text, token_usage_acc = _generate_with_gemini(
                client,
                system_prompt,
                prompt_for_generation,
                files,
                chat_history=body.chat_history,
                model=default_model,
            )
            gemini_files = files
        else:
            gemini_client = get_gemini_client()
            gemini_files = _upload_documents_to_gemini(gemini_client, document_entries)
            draft_text, draft_usage = _generate_with_gemini(
                gemini_client,
                system_prompt,
                prompt_for_generation,
                gemini_files,
                chat_history=[],
                model=_get_gemini_generation_model(),
            )

            openai_client = get_openai_client()
            openai_file_blocks = _upload_documents_to_openai(openai_client, document_entries)
            if body.model == "two-step-expert":
                final_system_prompt = (
                    "Du bist ein sehr strenger Senior Partner einer migrationsrechtlichen Kanzlei.\n"
                    "Deine Aufgabe ist es, den vorliegenden ENTWURF auf Basis der beigefügten Dokumente vollständig zu überarbeiten und zu finalisieren.\n"
                    "Arbeite dabei wie ein Red-Team-Reviewer und Endredakteur zugleich.\n"
                    "Prüfe den Entwurf auf Halluzinationen, unzulässige Tatsachenzusätze, logische Sprünge, überdehnte Rechtsprechung, unpräzise Verweise und überzogene Tonalität.\n"
                    "Korrigiere alle diese Punkte unmittelbar im Endtext.\n"
                    "Erstelle am Ende nur die bereinigte finale Fassung des Schriftsatzes, ohne Vorbemerkung und ohne Aufzählung der Mängel.\n\n"
                    f"{NEUTRAL_LEGAL_TONE_RULES}"
                )
                final_user_prompt = (
                    f"Hier ist der zu überarbeitende Entwurf:\n\n{draft_text}\n\n"
                    "Erstelle nun auf Basis der beigefügten Dokumente die prozessfest bereinigte Endfassung."
                )
                generated_text, final_usage = _generate_with_gpt5(
                    openai_client,
                    final_system_prompt,
                    final_user_prompt,
                    openai_file_blocks,
                    chat_history=[],
                    reasoning_effort=OPENAI_GPT5_REASONING_EFFORT,
                    verbosity="medium",
                    model="gpt-5.5",
                )
                token_usage_acc = _merge_token_usages(draft_usage, final_usage, model="two-step-expert")
            else:
                critique_system_prompt = _build_senior_partner_critique_prompt()
                critique_user_prompt = f"Hier ist der zu prüfende Entwurf:\n\n{draft_text}"
                critique_text, critique_usage = _generate_with_gpt5(
                    openai_client,
                    critique_system_prompt,
                    critique_user_prompt,
                    openai_file_blocks,
                    chat_history=[],
                    reasoning_effort=OPENAI_GPT5_REASONING_EFFORT,
                    verbosity="low",
                    model="gpt-5.5",
                )

                final_system_prompt = _build_gemini_finalize_system_prompt()
                final_user_prompt = (
                    f"ENTWURF (mit Vorüberlegungen):\n{draft_text}\n\n"
                    f"KRITIK DES SENIOR PARTNERS:\n{critique_text}\n\n"
                    "Erstelle nun die finale, bereinigte Version (ohne <document_analysis> etc.)."
                )
                generated_text, final_usage = _generate_with_gemini(
                    gemini_client,
                    final_system_prompt,
                    final_user_prompt,
                    gemini_files,
                    chat_history=[],
                    model=_get_gemini_generation_model(),
                )
                token_usage_acc = _merge_token_usages(
                    draft_usage,
                    critique_usage,
                    final_usage,
                    model="multi-step-expert",
                )
    elif body.model.startswith("gpt"):
        client = get_openai_client()
        file_blocks = _upload_documents_to_openai(client, document_entries)
        generated_text, token_usage_acc = _generate_with_gpt5(
            client,
            system_prompt,
            prompt_for_generation,
            file_blocks,
            chat_history=body.chat_history,
            reasoning_effort=OPENAI_GPT5_REASONING_EFFORT,
            verbosity=body.verbosity,
            model=body.model,
        )
    elif body.model.startswith("gemini"):
        client = get_gemini_client()
        gemini_files = _upload_documents_to_gemini(client, document_entries)
        generated_text, token_usage_acc = _generate_with_gemini(
            client,
            system_prompt,
            prompt_for_generation,
            gemini_files,
            chat_history=body.chat_history,
            model=body.model,
        )
    else:
        client = get_anthropic_client()
        document_blocks = _upload_documents_to_claude(client, document_entries)
        text_chunks: List[str] = []
        thinking_chunks: List[str] = []
        for chunk_str in _generate_with_claude_stream(
            client,
            system_prompt,
            prompt_for_generation,
            document_blocks,
            chat_history=body.chat_history,
        ):
            chunk = json.loads(chunk_str)
            if chunk["type"] == "text":
                text_chunks.append(chunk["text"])
            elif chunk["type"] == "thinking":
                thinking_chunks.append(chunk["text"])
            elif chunk["type"] == "usage":
                token_usage_acc = TokenUsage(**chunk["data"])
        generated_text = "".join(text_chunks)
        thinking_text = "".join(thinking_chunks)

    citation_checks, citation_check_warnings = await run_citation_checks(generated_text, collected)
    citation_summary = citation_checks.get("summary") or {}

    metadata = GenerationMetadata(
        documents_used={
            "anhoerung": len(collected.get("anhoerung", [])),
            "bescheid": len(collected.get("bescheid", [])),
            "rechtsprechung": len(collected.get("rechtsprechung", [])),
            "saved_sources": len(collected.get("saved_sources", [])),
            "akte": len(collected.get("akte", [])),
            "sonstiges": len(collected.get("sonstiges", [])),
            "internal_notes": len(collected.get("internal_notes", [])),
        },
        resolved_legal_area=resolved_legal_area,
        citations_found=int(citation_summary.get("verified_on_cited_page", 0) or 0),
        missing_citations=[],
        pinpoint_missing=[],
        citation_checks=citation_checks,
        warnings=citation_check_warnings,
        word_count=len(generated_text.split()),
        token_count=token_usage_acc.total_tokens if token_usage_acc else 0,
        token_usage=token_usage_acc,
    )
    if not legal_area_explicit:
        metadata.warnings.append(
            f"legal_area nicht explizit gesetzt, Fallback auf '{resolved_legal_area}' verwendet."
        )

    draft = _persist_generated_draft(
        db,
        current_user,
        body,
        target_case_id,
        collected,
        primary_bescheid_entry,
        resolved_legal_area,
        generated_text,
        thinking_text,
        token_usage_acc,
        citation_checks,
    )

    result = GenerationResponse(
        document_type=body.document_type,
        user_prompt=body.user_prompt,
        generated_text=generated_text,
        thinking_text=thinking_text or None,
        used_documents=[
            {
                "filename": entry.get("filename") or entry.get("title") or "",
                "category": category,
            }
            for category, entries in collected.items()
            for entry in entries
            if entry.get("filename") or entry.get("title")
        ],
        metadata=metadata,
    )

    return {
        "draft_id": str(draft.id) if draft else None,
        "case_id": str(target_case_id) if target_case_id else None,
        "result": result.model_dump(),
    }


def _build_sozialrecht_prompts(
    body,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
    primary_bescheid_label: str,
    primary_bescheid_description: str,
    context_summary: str,
) -> tuple[str, str]:
    """Build system and user prompts for Sozialrecht generation."""

    sozialrecht_citation_rules = (
        "ZITIERWEISE:\n"
        "- Jede Tatsachenbehauptung aus Dokumenten muss mit einer konkreten Fundstelle belegt werden.\n"
        "- Verwende bei PDF-Dokumenten grundsätzlich Seitenangaben im Format 'S. X' oder 'Seite X'.\n"
        "- Beispiele: 'Bescheid vom 06.03.2026, S. 2', 'Forderungsaufstellung vom 16.03.2026, S. 1', "
        "'Räumungsklage vom 18.02.2026, S. 3', 'gerichtliches Schreiben vom 09.03.2026, S. 1'.\n"
        "- Verwende 'Bl. X d.A.' nur dann, wenn eine echte Blattzahl bekannt ist.\n"
        "- Verwende niemals Platzhalter wie 'Bl. ... der Akte', 'vgl. Akte' oder 'vgl. Unterlagen'.\n"
        "- Wenn keine konkrete Fundstelle angegeben werden kann, streiche die Behauptung oder formuliere sie ohne Dokumentenbezug um.\n\n"
    )

    doc_type_lower = body.document_type.lower()
    primary_bescheid_entry = next(
        (entry for entry in collected.get("bescheid", []) if entry.get("role") == "primary"),
        None,
    )
    is_klage_or_bescheid = "klage" in doc_type_lower or primary_bescheid_entry is not None

    if is_klage_or_bescheid:
        system_prompt = (
            "DENKWEISE:\n"
            "Denke gründlich und ausführlich über diesen Fall nach, bevor du schreibst. "
            "Analysiere ALLE vorliegenden Dokumente sorgfältig. "
            "Betrachte verschiedene Argumentationsansätze und wähle die überzeugendsten. "
            "Prüfe deine Argumentation auf Lücken und Schwächen, bevor du sie finalisierst.\n\n"

            "Du bist ein erfahrener Fachanwalt für Sozialrecht. Du schreibst eine überzeugende, "
            f"strategisch durchdachte Klagebegründung gegen den Hauptbescheid ({primary_bescheid_label}).\n\n"

            "STRATEGISCHER ANSATZ:\n"
            "Konzentriere dich auf die aussichtsreichsten Argumente. Nicht jeder Punkt des Bescheids muss widerlegt werden - "
            "wähle die stärksten rechtlichen und tatsächlichen Ansatzpunkte aus den bereitgestellten Dokumenten.\n\n"

            "RECHTSGRUNDLAGEN:\n"
            "- Prüfe die Anspruchsvoraussetzungen und Ausschlusstatbestände (SGB II / SGB XII, je nach Leistung).\n"
            "- Berücksichtige Verfahrensrecht nach SGB X (u.a. Anhörung § 24, Begründung § 35, Rücknahme/Widerruf §§ 45/48) "
            "und das SGG (Klage, ggf. § 86b Eilrechtsschutz).\n"
            "- Achte auf Fehler bei Bedarfsermittlung, Einkommen/Vermögen und Kosten der Unterkunft.\n\n"

            "BEWEISFÜHRUNG:\n"
            "- Bescheid/Widerspruchsbescheid: Zeige konkret, wo die Würdigung fehlerhaft ist (mit Seitenzahlen)\n"
            "- Anträge/Nachweise: Belege mit direkten Zitaten oder Fundstellen, was eingereicht wurde\n"
            "- Rechtsprechung: Zeige vergleichbare Fälle und übertragbare Rechtssätze\n"
            "- Gesetzestexte: Lege die Tatbestandsmerkmale zutreffend aus\n\n"

            "INTERNE KANZLEINOTIZEN:\n"
            "- Interne Kanzleinotizen, Gesprächsnotizen, Transkripte und Besprechungsvermerke sind KEINE zitierfähigen Quellen.\n"
            "- Nutze sie nur als internes Tatsachen- und Strategieinput für Parteivortrag, Beweisangebote und Subsumtion.\n"
            "- Zitiere sie niemals als 'Aktennotiz', 'Notiz', 'Anlage', 'Quelle' oder Fundstelle.\n\n"

            f"{sozialrecht_citation_rules}"
            "GESETZESZITATE:\n"
            "- Rechtsprechung: Volles Aktenzeichen, Gericht, Datum\n"
            "- Gesetzestexte: '§ X SGB II/SGB XII' bzw. '§ X Abs. Y SGB X'\n\n"

            "STIL & FORMAT:\n"
            "- Durchgehender Fließtext ohne Aufzählungen oder Zwischenüberschriften\n"
            "- Keine Nummerierung, keine Gliederungspunkte\n"
            "- Klare Absatzstruktur: Einleitung, mehrere Argumentationsblöcke, Schluss\n"
            "- Jede Behauptung mit konkretem Beleg (Zitat, Fundstelle)\n"
            "- Präzise juristische Sprache, keine Floskeln\n"
            "- Einstieg sachlich und zurückhaltend; keine pauschale Totalwertung des Bescheids im ersten Satz\n"
            "- Beginne ohne Vorbemerkungen direkt mit dem juristischen Fließtext, keine Adresszeilen oder Anreden\n"
            "- KEINE Antragsformulierung - nur die rechtliche Würdigung\n\n"
            f"{NEUTRAL_LEGAL_TONE_RULES}"
            "Drei starke, gut belegte Argumente sind besser als zehn oberflächliche Punkte. Aber diese drei Argumente müssen erschöpfend behandelt werden."
        )

        verbosity = body.verbosity
        if verbosity == "low":
            system_prompt += (
                "\n\n"
                "FASSUNG & LÄNGE (VERBOSITY: LOW):\n"
                "- Fasse dich kurz und prägnant.\n"
                "- Konzentriere dich ausschließlich auf die absolut wesentlichen Punkte.\n"
                "- Vermeide ausschweifende Erklärungen oder Wiederholungen.\n"
                "- Ziel ist eine kompakte, schnell erfassbare Argumentation."
            )
        elif verbosity == "medium":
            system_prompt += (
                "\n\n"
                "FASSUNG & LÄNGE (VERBOSITY: MEDIUM):\n"
                "- Wähle einen ausgewogenen Ansatz zwischen Detailtiefe und Lesbarkeit.\n"
                "- Erkläre die wichtigen Punkte gründlich, aber komme schnell zum Punkt.\n"
                "- Vermeide unnötige Füllwörter."
            )
        else:  # high
            system_prompt += (
                "\n\n"
                "FASSUNG & LÄNGE (VERBOSITY: HIGH):\n"
                "- Die Argumentation muss ausführlich und tiefgehend sein.\n"
                "- Nutze die volle verfügbare Länge, um den Sachverhalt umfassend zu würdigen.\n"
                "- Gehe detailliert auf jeden Widerspruch und jedes Beweismittel ein."
            )

        primary_bescheid_section = f"Hauptbescheid: {primary_bescheid_label}"
        if primary_bescheid_description:
            primary_bescheid_section += f"\nBeschreibung: {primary_bescheid_description}"

        user_prompt = (
            f"Dokumententyp: {body.document_type}\n"
            f"{primary_bescheid_section}\n\n"

            f"Auftrag:\n{body.user_prompt.strip()}\n\n"

            "Verfügbare Dokumente:\n"
            f"{context_summary or '- (Keine Dokumente)'}\n\n"

            "AUFGABE:\n"
            "Analysiere die Dokumente sorgfältig und verfasse die detaillierte rechtliche Würdigung als Fließtext.\n"
            "- Identifiziere die tragenden Ablehnungsgründe und widerlege sie mit Fakten aus den Unterlagen.\n"
            "- Jede dokumentengestützte Tatsachenbehauptung braucht eine konkrete Seitenangabe.\n"
            "- Verwende keine Platzhalterzitate wie 'Bl. ... der Akte'.\n"
            "- Interne Kanzleinotizen nicht zitieren; ihren Inhalt nur in Parteivortrag und Beweisangebote übersetzen.\n"
            "- Beginne direkt mit der juristischen Argumentation ohne Adressblock oder Anrede.\n"
            "- Keine Nummerierung, keine Aufzählungen, keine Gliederungspunkte."
        )
    else:
        system_prompt = (
            "Du bist ein erfahrener Fachanwalt für Sozialrecht. "
            "Du erstellst einen rechtlichen Schriftsatz oder Antrag (z.B. Widerspruch, Überprüfungsantrag nach § 44 SGB X, "
            "Eilantrag nach § 86b SGG) für Mandanten.\n\n"

            "ZIEL & FOKUS:\n"
            "Fokussiere dich auf die Anspruchsvoraussetzungen für die begehrte Leistung und die Fehler des Bescheids.\n"
            "- Identifiziere die einschlägige Rechtsgrundlage (SGB II/SGB XII/SGB X/SGG).\n"
            "- Subsumiere die Fakten aus den Dokumenten unter die Tatbestandsmerkmale.\n"
            "- Argumentiere präzise und lösungsorientiert.\n\n"

            "BEWEISFÜHRUNG:\n"
            "Nutze alle verfügbaren Dokumente (Bescheide, Anträge, Nachweise, Kontoauszüge, Mietunterlagen), "
            "um die Voraussetzungen zu belegen.\n"
            "- Jede dokumentengestützte Tatsachenbehauptung braucht eine konkrete Seitenangabe.\n"
            f"{sozialrecht_citation_rules}"

            "INTERNE KANZLEINOTIZEN:\n"
            "- Interne Kanzleinotizen, Gesprächsnotizen, Transkripte und Besprechungsvermerke sind KEINE zitierfähigen Quellen.\n"
            "- Nutze sie nur als internes Tatsachen- und Strategieinput.\n"
            "- Zitiere sie niemals als 'Aktennotiz', 'Notiz', 'Anlage', 'Quelle' oder Fundstelle.\n\n"

            "STIL & FORMAT:\n"
            "- Juristischer Profi-Stil (Sachlich, Überzeugend).\n"
            "- Klar strukturiert (Sachverhalt -> Rechtliche Würdigung -> Ergebnis).\n"
            "- Wenn der Auftrag einen Antrag verlangt, formuliere die konkreten Anträge am Anfang. Nummerierte Anträge sind zulässig.\n"
            "- Keine Floskeln.\n"
            "- Beginne direkt mit dem juristischen Text, keine Adresszeilen oder Anreden.\n"
            "- Die Begründung soll als Fließtext mit klaren Absätzen formuliert sein. Keine unnötigen Aufzählungen in der Begründung.\n\n"
            f"{NEUTRAL_LEGAL_TONE_RULES}"
        )

        verbosity = body.verbosity
        if verbosity == "low":
            system_prompt += "\n\nFASSUNG (LOW): Kurz und bündig. Nur Key-Facts."
        elif verbosity == "medium":
            system_prompt += "\n\nFASSUNG (MEDIUM): Standard-Schriftsatzlänge. Ausgewogen."
        else:  # high
            system_prompt += "\n\nFASSUNG (HIGH): Ausführliche Darlegung aller Voraussetzungen und detaillierte Würdigung aller Belege."

        user_prompt = (
            f"Dokumententyp: {body.document_type}\n"
            f"Auftrag: {body.user_prompt.strip()}\n\n"

            "VERFÜGBARE DOKUMENTE:\n"
            f"{context_summary or '- (Keine Dokumente)'}\n\n"

            "VORGEHENSWEISE:\n"
            "Bevor du den Text schreibst, analysiere den Fall in den folgenden XML-Tags:\n\n"

            "<document_analysis>\n"
            "- **Auftrag:** Was genau ist das Ziel? (z.B. Leistungen nach SGB II)\n"
            "- **Voraussetzungen:** Welche gesetzlichen Merkmale (§§) müssen erfüllt sein?\n"
            "- **Belege:** Welche Dokumente beweisen diese Merkmale?\n"
            "- **Fundstellen:** Notiere für jeden verwendeten Beleg die konkrete Seite des Dokuments.\n"
            "</document_analysis>\n\n"

            "<strategy>\n"
            "1. **Ziel:** Anspruchsdurchsetzung.\n"
            "2. **Fakten/Belege:** Zuordnung der Dokumente zu den Tatbestandsmerkmalen.\n"
            "3. **Argumentation:** Subsumtion der Fakten unter die Rechtsnorm.\n"
            "</strategy>\n\n"

            "Verfasse nun basierend auf dieser Strategie den Schriftsatz als Fließtext (OHNE die XML-Tags im Output zu wiederholen). "
            "Verwende keine Platzhalterzitate wie 'Bl. ... der Akte'. "
            "Keine Nummerierung, keine Aufzählungen, keine Gliederungspunkte."
        )

    return system_prompt, user_prompt


def _build_zivilrecht_prompts(
    body,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
    primary_bescheid_label: str,
    primary_bescheid_description: str,
    context_summary: str,
) -> tuple[str, str]:
    """Build system and user prompts for Zivilrecht generation."""

    zivilrecht_citation_rules = (
        "ZITIERWEISE:\n"
        "- Jede dokumentengestützte Tatsachenbehauptung braucht eine konkrete Fundstelle.\n"
        "- Verwende bei PDF-Dokumenten Seitenangaben im Format 'S. X' oder 'Seite X'.\n"
        "- Beispiele: 'Vertrag vom 12.01.2026, S. 2', 'E-Mail vom 04.02.2026, S. 1', 'Mahnschreiben vom 18.02.2026, S. 3'.\n"
        "- Verwende niemals Platzhalter wie 'vgl. Akte' oder 'Bl. ...'.\n"
        "- Wenn keine konkrete Fundstelle angegeben werden kann, streiche die Behauptung oder formuliere sie ohne Dokumentenbezug um.\n\n"
    )

    system_prompt = (
        "DENKWEISE:\n"
        "Denke gründlich und präzise über den Fall nach, bevor du schreibst. "
        "Analysiere alle Dokumente sorgfältig, ordne die Tatsachen sauber ein und prüfe die Anspruchsgrundlagen.\n\n"

        "Du bist ein erfahrener Fachanwalt für Zivilrecht. "
        "Du verfasst einen prozessfesten zivilrechtlichen Schriftsatz auf Basis der beigefügten Unterlagen.\n\n"

        "ARBEITSWEISE:\n"
        "- Ermittle die tragenden Tatsachen aus Vertrag, Korrespondenz, Mahnungen, Rechnungen, Gutachten und sonstigen Unterlagen.\n"
        "- Ordne die Fakten den einschlägigen Anspruchsgrundlagen und Einwendungen zu.\n"
        "- Berücksichtige materielle Anspruchsvoraussetzungen ebenso wie prozessuale Gesichtspunkte nach der ZPO.\n"
        "- Zeige Widersprüche, Beweisprobleme und Angriffs- bzw. Verteidigungspunkte klar auf.\n\n"

        "RECHTLICHER FOKUS:\n"
        "- Prüfe insbesondere Vertrag, Pflichtverletzung, Verzug, Schaden, Kausalität, Fälligkeit, Einwendungen und Beweislast.\n"
        "- Ziehe je nach Fall insbesondere BGB, ZPO, HGB oder Nebengesetze heran.\n"
        "- Arbeite mit sauberer Subsumtion statt bloßer Behauptung.\n\n"

        f"{zivilrecht_citation_rules}"
        "INTERNE KANZLEINOTIZEN:\n"
        "- Interne Kanzleinotizen, Gesprächsnotizen, Transkripte und Besprechungsvermerke sind KEINE zitierfähigen Quellen.\n"
        "- Nutze sie nur als internes Tatsachen- und Strategieinput für Parteivortrag, Beweisangebote und Subsumtion.\n"
        "- Zitiere sie niemals als 'Aktennotiz', 'Notiz', 'Anlage', 'Quelle' oder Fundstelle.\n\n"

        "GESETZES- UND RECHTSPRECHUNGSZITATE:\n"
        "- Rechtsprechung: Gericht, Datum und Aktenzeichen vollständig nennen.\n"
        "- Gesetzestexte: präzise mit Paragraph, Absatz und Gesetz abkürzen.\n\n"

        "STIL & FORMAT:\n"
        "- Juristisch präzise, sachlich und professionell.\n"
        "- Keine Nummerierung, keine Aufzählungen, keine Gliederungspunkte.\n"
        "- Durchgehender Fließtext mit klarer Absatzstruktur.\n"
        "- Beginne direkt mit der rechtlichen Würdigung ohne Adressblock oder Anrede.\n"
        "- Trenne Tatsachenvortrag, Beleg und rechtliche Bewertung sauber.\n\n"
        f"{NEUTRAL_LEGAL_TONE_RULES}"
    )

    verbosity = body.verbosity
    if verbosity == "low":
        system_prompt += "\n\nFASSUNG (LOW): Kurz, fokussiert und nur die tragenden Punkte."
    elif verbosity == "medium":
        system_prompt += "\n\nFASSUNG (MEDIUM): Ausgewogen, mit solider Begründungstiefe."
    else:
        system_prompt += "\n\nFASSUNG (HIGH): Ausführlich, mit vertiefter Subsumtion und vollständiger Würdigung der Unterlagen."

    user_prompt = (
        f"Dokumententyp: {body.document_type}\n"
        f"Auftrag: {body.user_prompt.strip()}\n\n"
        "VERFÜGBARE DOKUMENTE:\n"
        f"{context_summary or '- (Keine Dokumente)'}\n\n"
        "AUFGABE:\n"
        "Analysiere die Dokumente sorgfältig und verfasse den zivilrechtlichen Schriftsatz als Fließtext.\n"
        "- Jede dokumentengestützte Tatsachenbehauptung braucht eine konkrete Fundstelle.\n"
        "- Verwende keine Platzhalterzitate wie 'vgl. Akte' oder 'Bl. ...'.\n"
        "- Interne Kanzleinotizen nicht zitieren; ihren Inhalt nur in Parteivortrag und Beweisangebote übersetzen.\n"
        "- Arbeite Anspruchsgrundlagen, Einwendungen und Beweisfragen präzise heraus.\n"
        "- Keine Nummerierung, keine Aufzählungen, keine Gliederungspunkte."
    )

    return system_prompt, user_prompt


def _build_generation_prompts(
    body,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
    primary_bescheid_label: str,
    primary_bescheid_description: str,
    context_summary: str
) -> tuple[str, str]:
    """Build system and user prompts for document generation with strategic, flexible approach."""

    legal_area = (getattr(body, "legal_area", None) or "migrationsrecht").lower()
    if legal_area == "sozialrecht":
        return _build_sozialrecht_prompts(
            body,
            collected,
            primary_bescheid_label,
            primary_bescheid_description,
            context_summary,
        )
    if legal_area == "zivilrecht":
        return _build_zivilrecht_prompts(
            body,
            collected,
            primary_bescheid_label,
            primary_bescheid_description,
            context_summary,
        )

    # Determine Mode based on Document Type and Selections
    doc_type_lower = body.document_type.lower()
    
    # 1. AZB (Appeal) Mode
    # Triggered explicitly by classification "AZB" or "Berufung"
    is_azb = "azb" in doc_type_lower or "zulassung der berufung" in doc_type_lower

    # 2. Klage/Bescheid Mode (Legacy)
    # Triggered by "Klage" type OR presence of a selected Primary Bescheid
    primary_bescheid_entry = next(
        (entry for entry in collected.get("bescheid", []) if entry.get("role") == "primary"),
        None,
    )
    is_klage_or_bescheid = "klage" in doc_type_lower or primary_bescheid_entry is not None
    
    if is_azb:
        # --- AZB PROMPT LOGIC ---
        print("[DEBUG] AZB Mode activated (Vorinstanz documents present)")
        
        # Identify primary Vorinstanz document (Judgment)
        primary_vorinstanz_doc = next((d for d in collected.get("vorinstanz", []) if d.get("role") == "primary"), None)
        primary_vorinstanz_label = primary_vorinstanz_doc["filename"] if primary_vorinstanz_doc else "das Urteil"

        system_prompt = (
            # Anthropic extended thinking tip: Use high-level instructions for thorough reasoning
            "DENKWEISE:\n"
            "Denke gründlich und ausführlich über diesen Fall nach, bevor du schreibst. "
            "Analysiere ALLE vorliegenden Dokumente sorgfältig. "
            "Betrachte verschiedene Argumentationsansätze und wähle die überzeugendsten. "
            "Prüfe deine Argumentation auf Lücken und Schwächen, bevor du sie finalisierst.\n\n"
            
            "Du bist ein erfahrener Fachanwalt für Migrationsrecht, spezialisiert auf das Berufungszulassungsrecht. "
            f"Du schreibst eine Begründung für einen Antrag auf Zulassung der Berufung (AZB) gegen ein Urteil des Verwaltungsgerichts ({primary_vorinstanz_label}) in einer Asylstreitigkeit.\n\n"

            "WICHTIGE RECHTSLAGE (§ 78 Abs. 3 AsylG):\n"
            "In Asylverfahren gibt es den Zulassungsgrund der 'ernstlichen Zweifel' (§ 124 Abs. 2 Nr. 1 VwGO) NICHT. "
            "Die Berufung ist nur zuzulassen bei:\n"
            "1. Grundsätzlicher Bedeutung (§ 78 Abs. 3 Nr. 1 AsylG)\n"
            "2. Divergenz (§ 78 Abs. 3 Nr. 2 AsylG)\n"
            "3. Verfahrensmangel (§ 78 Abs. 3 Nr. 3 AsylG)\n\n"

            "ZIEL & FOKUS:\n"
            "Der Fokus liegt auf der Rüge von VERFAHRENSMÄNGELN, insbesondere der Verletzung des rechtlichen Gehörs (§ 78 Abs. 3 Nr. 3 AsylG i.V.m. § 138 Nr. 3 VwGO / Art. 103 Abs. 1 GG).\n\n"

            "STRATEGIE (GEHÖRSRÜGE):\n"
            f"1. Darlegung des übergangenen Vortrags: Welches konkrete Vorbringen oder Beweisangebot hat das Gericht in {primary_vorinstanz_label} ignoriert?\n"
            "2. Darlegung der Entscheidungserheblichkeit (Kausalität): Warum hätte das Gericht anders entschieden, wenn es den Vortrag berücksichtigt hätte?\n"
            "   - Argumentiere NICHT mit 'falscher Rechtsanwendung' (das wäre ein materieller Fehler), sondern mit 'Nichtzurkenntnisnahme von Vortrag' (Verfahrensfehler).\n\n"

            "WEITERE ZULASSUNGSGRÜNDE (NUR WENN EINSCHLÄGIG):\n"
            "- Grundsätzliche Bedeutung: Wenn eine klärungsbedürftige Rechts- oder Tatsachenfrage vorliegt, die über den Einzelfall hinausgeht.\n"
            "- Divergenz: Wenn das Urteil von einer Entscheidung des OVG, BVerwG oder BVerfG abweicht (genaue Bezeichnung der Abweichung erforderlich).\n\n"

            "FORMAT:\n"
            "- Juristischer Schriftsatzstil.\n"
            "- Keine Floskeln.\n"
            "- Keine Nummerierung, keine Aufzählungen, keine Gliederungspunkte.\n"
            f"- Konkrete Bezugnahme auf das VG-Urteil ({primary_vorinstanz_label}).\n"
            "- Beginne direkt mit der Begründung, keine Adresszeilen.\n"
            "- Durchgehender Fließtext mit klaren Absätzen.\n\n"
            f"{NEUTRAL_LEGAL_TONE_RULES}"
        )
        
        # Verbosity for AZB
        verbosity = body.verbosity
        if verbosity == "low":
            system_prompt += "\n\nFASSUNG (LOW): Fasse dich kurz. Nur die stärkste Gehörsrüge ausführen."
        elif verbosity == "medium":
            system_prompt += "\n\nFASSUNG (MEDIUM): Ausgewogene Begründung. Fokus auf die Gehörsverletzung."
        else: # high
            system_prompt += (
                "\n\nFASSUNG (HIGH):\n"
                "- Ausführliche und tiefe Auseinandersetzung mit dem Verfahrensfehler.\n"
                "- Nutze die volle Token-Kapazität für eine erschöpfende Begründung.\n"
                "- Arbeite die Kausalität des Fehlers für das Urteil detailliert heraus."
            )

        user_prompt = (
            f"Dokumententyp: Antrag auf Zulassung der Berufung (AZB) nach § 78 AsylG\n\n"
            
            f"AUFTRAG:\n{body.user_prompt.strip()}\n\n"

            "VERFÜGBARE DOKUMENTE:\n"
            f"{context_summary or '- (Keine Dokumente)'}\n\n"

            "VORGEHENSWEISE:\n"
            "Bevor du den Text schreibst, analysiere den Fall tiefgehend in den folgenden XML-Tags:\n\n"
            
            "<document_analysis>\n"
            "- **Vorinstanz:** Wo genau hat das Gericht Vortrag ignoriert? (Seite/Absatz zitieren)\n"
            "- **Protokolle:** Was steht im Protokoll, das im Urteil fehlt? Zitiere den Wortlaut.\n"
            "- **Schriftsätze:** Welcher schriftliche Vortrag wurde übergangen?\n"
            "</document_analysis>\n\n"

            "<strategy>\n"
            "1. **Ziel:** Zulassung der Berufung wegen Verfahrensmangel (§ 78 Abs. 3 Nr. 3 AsylG).\n"
            "2. **Fakten:** Die stärksten ignorierten Punkte aus der Analyse.\n"
            "3. **Rechtsgrundlage:** Art. 103 Abs. 1 GG (Rechtliches Gehör).\n"
            "4. **Argumentation:**\n"
            "   - Das Gericht hat X ignoriert.\n"
            "   - Das war entscheidungserheblich (Kausalität), weil...\n"
            "</strategy>\n\n"

            "Verfasse nun basierend auf dieser Strategie die Begründung des Zulassungsantrags als Fließtext (OHNE die XML-Tags im Output zu wiederholen). "
            "Keine Nummerierung, keine Aufzählungen, keine Gliederungspunkte."
        )
        
    elif is_klage_or_bescheid:
        # --- LEGACY / BESCHEID PROMPT LOGIC ---
        system_prompt = (
            # Anthropic extended thinking tip: Use high-level instructions for thorough reasoning
            "DENKWEISE:\n"
            "Denke gründlich und ausführlich über diesen Fall nach, bevor du schreibst. "
            "Analysiere ALLE vorliegenden Dokumente sorgfältig. "
            "Betrachte verschiedene Argumentationsansätze und wähle die überzeugendsten. "
            "Prüfe deine Argumentation auf Lücken und Schwächen, bevor du sie finalisierst.\n\n"
            
            f"Du bist ein erfahrener Fachanwalt für Migrationsrecht. Du schreibst eine überzeugende, "
            f"strategisch durchdachte juristische Argumentation gegen den Hauptbescheid (Anlage K2: {primary_bescheid_label}).\n\n"
    
            "STRATEGISCHER ANSATZ:\n"
            "Konzentriere dich auf die aussichtsreichsten Argumente. Nicht jeder Punkt des BAMF-Bescheids muss widerlegt werden - "
            "wähle die stärksten rechtlichen und tatsächlichen Ansatzpunkte aus den bereitgestellten Dokumenten.\n\n"
    
            "RECHTSGRUNDLAGEN:\n"
            "Stütze deine Argumentation auf die relevanten Vorschriften (§ 3 AsylG, § 4 AsylG, § 60 AufenthG etc.) "
            "und arbeite heraus, wo das BAMF diese fehlerhaft angewendet hat.\n\n"
    
            "BEWEISFÜHRUNG:\n"
            "- Hauptbescheid (Anlage K2): Zeige konkret, wo die Würdigung fehlerhaft ist (mit Seitenzahlen)\n"
            "- Anhörungen: Belege mit direkten Zitaten, was der Mandant tatsächlich ausgesagt hat (Bl. X d.A.)\n"
            "- Vorinstanz: Gehe auf Urteile oder Protokolle der Vorinstanz ein, falls vorhanden\n"
            "- Rechtsprechung: Zeige vergleichbare Fälle und übertragbare Rechtssätze\n"
            "- Gesetzestexte: Lege die Tatbestandsmerkmale zutreffend aus\n\n"

            "INTERNE KANZLEINOTIZEN:\n"
            "- Interne Kanzleinotizen, Gesprächsnotizen, Transkripte und Besprechungsvermerke sind KEINE zitierfähigen Quellen.\n"
            "- Nutze sie nur als internes Tatsachen- und Strategieinput, um klägerischen Vortrag, Beweisangebote und Subsumtion zu entwickeln.\n"
            "- Zitiere sie niemals als 'Aktennotiz', 'Notiz', 'Anlage', 'Quelle', 'Bl. ... d.A.' oder mit Datum im Schriftsatz.\n"
            "- Überführe ihren Inhalt in anwaltliche Tatsachendarstellung im Indikativ und in Beweisangebote. Formuliere nicht unnötig distanziert mit 'laut Mandant', 'nach Angaben' oder 'angeblich'.\n\n"
    
            "ZITIERWEISE:\n"
            "- Hauptbescheid: 'Anlage K2, S. X'\n"
            "- Anhörungen/Aktenbestandteile: 'Bl. X d.A.' oder 'Bl. X ff. d.A.'\n"
            "- Rechtsprechung: Volles Aktenzeichen, Gericht, Datum\n"
            "- Gesetzestexte: '§ X AsylG' bzw. '§ X Abs. Y AufenthG'\n\n"
    
            "STIL & FORMAT:\n"
            "- Durchgehender Fließtext ohne Aufzählungen oder Zwischenüberschriften\n"
            "- Keine Nummerierung, keine Gliederungspunkte\n"
            "- Klare Absatzstruktur: Einleitung, mehrere Argumentationsblöcke, Schluss\n"
            "- Jede Behauptung mit konkretem Beleg (Zitat, Fundstelle)\n"
            "- Präzise juristische Sprache, keine Floskeln\n"
            "- Einstieg sachlich und zurückhaltend; keine pauschale Totalwertung des Bescheids im ersten Satz\n"
            "- Beginne ohne Vorbemerkungen direkt mit dem juristischen Fließtext, keine Adresszeilen oder Anreden\n"
            "- KEINE Antragsformulierung - nur die rechtliche Würdigung\n\n"
            f"{NEUTRAL_LEGAL_TONE_RULES}"
    
            "Drei starke, gut belegte Argumente sind besser als zehn oberflächliche Punkte. Aber diese drei Argumente müssen erschöpfend behandelt werden."
        )
    
        # Adjust instructions based on verbosity
        verbosity = body.verbosity
        if verbosity == "low":
            system_prompt += (
                "\n\n"
                "FASSUNG & LÄNGE (VERBOSITY: LOW):\n"
                "- Fasse dich kurz und prägnante.\n"
                "- Konzentriere dich ausschließlich auf die absolut wesentlichen Punkte.\n"
                "- Vermeide ausschweifende Erklärungen oder Wiederholungen.\n"
                "- Ziel ist eine kompakte, schnell erfassbare Argumentation."
            )
        elif verbosity == "medium":
            system_prompt += (
                "\n\n"
                "FASSUNG & LÄNGE (VERBOSITY: MEDIUM):\n"
                "- Wähle einen ausgewogenen Ansatz zwischen Detailtiefe und Lesbarkeit.\n"
                "- Erkläre die wichtigen Punkte gründlich, aber komme schnell zum Punkt.\n"
                "- Vermeide unnötige Füllwörter."
            )
        else:  # high (default)
            system_prompt += (
                "\n\n"
                "FASSUNG & LÄNGE (VERBOSITY: HIGH):\n"
                "- Die Argumentation muss ausführlich und tiefgehend sein.\n"
                "- Nutze die volle verfügbare Länge (bis zu 12.000 Token), um den Sachverhalt umfassend zu würdigen.\n"
                "- Gehe detailliert auf jeden Widerspruch und jedes Beweismittel ein."
            )
    
    
        primary_bescheid_section = f"Hauptbescheid (Anlage K2): {primary_bescheid_label}"
        if primary_bescheid_description:
            primary_bescheid_section += f"\nBeschreibung: {primary_bescheid_description}"
    
        user_prompt = (
            f"Dokumententyp: {body.document_type}\n"
            f"{primary_bescheid_section}\n\n"
    
            f"Auftrag:\n{body.user_prompt.strip()}\n\n"
    
            "Verfügbare Dokumente:\n"
            f"{context_summary or '- (Keine Dokumente)'}\n\n"
    
            "AUFGABE:\n"
            "Analysiere die Dokumente sorgfältig und verfasse die detaillierte rechtliche Würdigung als Fließtext.\n"
            "- Identifiziere die Ablehnungsgründe des BAMF und widerlege sie mit Fakten aus der Anhörung.\n"
            "- Zitiere konkret aus den beigefügten Urteilen und Quellen.\n"
            "- Interne Kanzleinotizen nicht zitieren; ihren Inhalt nur in Parteivortrag und Beweisangebote übersetzen.\n"
            "- Beginne direkt mit der juristischen Argumentation ohne Adressblock oder Anrede.\n"
            "- Keine Nummerierung, keine Aufzählungen, keine Gliederungspunkte."
        )

    else:
        # --- GENERAL APPLICATION / SCHRIFTSATZ PROMPT (No Bescheid) ---
        print("[DEBUG] General Application Mode activated (No Bescheid selected)")
        
        system_prompt = (
            "Du bist ein erfahrener Fachanwalt für Migrations- und Ausländerrecht. "
            f"Du erstellst einen rechtlichen Schriftsatz oder Antrag (z.B. Niederlassungserlaubnis, Einbürgerung, Stellungnahme) für Mandanten.\n\n"

            "ZIEL & FOKUS:\n"
            "Fokussiere dich auf die positive Darlegung der Anspruchsvoraussetzungen für das begehrte Ziel (z.B. Erteilung einer Erlaubnis).\n"
            "- Identifiziere die einschlägige Rechtsgrundlage (z.B. AufenthG, StAG, FreizügG/EU).\n"
            "- Subsumiere die Fakten aus den Dokumenten unter die Tatbestandsmerkmale.\n"
            "- Argumentiere präzise und lösungsorientiert.\n\n"

            "BEWEISFÜHRUNG:\n"
            "Nutze alle verfügbaren Dokumente (Aktenauszüge, Zertifikate, Protokolle, 'Sonstiges'), um die Voraussetzungen (z.B. Lebensunterhalt, Identität, Aufenthaltszeiten, Straffreiheit) zu belegen.\n"
            "- Zitiere konkret aus den Unterlagen, wo immer möglich.\n\n"

            "INTERNE KANZLEINOTIZEN:\n"
            "- Interne Kanzleinotizen, Gesprächsnotizen, Transkripte und Besprechungsvermerke sind KEINE zitierfähigen Quellen.\n"
            "- Nutze sie nur als internes Tatsachen- und Strategieinput.\n"
            "- Zitiere sie niemals als 'Aktennotiz', 'Notiz', 'Anlage', 'Quelle' oder Fundstelle.\n\n"

            "STIL & FORMAT:\n"
            "- Juristischer Profi-Stil (Sachlich, Überzeugend).\n"
            "- Klar strukturiert (Sachverhalt -> Rechtliche Würdigung -> Ergebnis).\n"
            "- Wenn der Auftrag einen Antrag verlangt, formuliere die konkreten Anträge am Anfang. Nummerierte Anträge sind zulässig.\n"
            "- Keine Floskeln.\n"
            "- Beginne direkt mit dem juristischen Text, keine Adresszeilen oder Anreden.\n"
            "- Die Begründung soll als Fließtext mit klaren Absätzen formuliert sein. Keine unnötigen Aufzählungen in der Begründung.\n\n"
            f"{NEUTRAL_LEGAL_TONE_RULES}"
        )

        # Verbosity
        verbosity = body.verbosity
        if verbosity == "low":
            system_prompt += "\n\nFASSUNG (LOW): Kurz und bündig. Nur Key-Facts."
        elif verbosity == "medium":
            system_prompt += "\n\nFASSUNG (MEDIUM): Standard-Schriftsatzlänge. Ausgewogen."
        else: # high
            system_prompt += "\n\nFASSUNG (HIGH): Ausführliche Darlegung aller Voraussetzungen und detaillierte Würdigung aller Belege."

        user_prompt = (
            f"Dokumententyp: {body.document_type}\n"
            f"Auftrag: {body.user_prompt.strip()}\n\n"

            "VERFÜGBARE DOKUMENTE:\n"
            f"{context_summary or '- (Keine Dokumente)'}\n\n"

            "VORGEHENSWEISE:\n"
            "Bevor du den Text schreibst, analysiere den Fall in den folgenden XML-Tags:\n\n"

            "<document_analysis>\n"
            "- **Auftrag:** Was genau ist das Ziel? (z.B. Niederlassungserlaubnis)\n"
            "- **Voraussetzungen:** Welche gesetzlichen Merkmale (§§) müssen erfüllt sein?\n"
            "- **Belege:** Welche Dokumente beweisen diese Merkmale?\n"
            "</document_analysis>\n\n"

            "<strategy>\n"
            "1. **Ziel:** Anspruchsdurchsetzung.\n"
            "2. **Fakten/Belege:** Zuordnung der Dokumente zu den Tatbestandsmerkmalen.\n"
            "3. **Argumentation:** Subsumtion der Fakten unter die Rechtsnorm.\n"
            "</strategy>\n\n"

            "Verfasse nun basierend auf dieser Strategie den Schriftsatz als Fließtext (OHNE die XML-Tags im Output zu wiederholen). "
            "Wenn der Auftrag einen Antrag verlangt, beginne mit konkreten Anträgen. "
            "Nummerierte Anträge sind zulässig. Die anschließende Begründung soll als Fließtext mit klaren Absätzen formuliert sein."
        )

    return system_prompt, user_prompt


JLAWYER_BASE_URL = os.environ.get("JLAWYER_BASE_URL")
if JLAWYER_BASE_URL:
    JLAWYER_BASE_URL = JLAWYER_BASE_URL.rstrip("/")
JLAWYER_USERNAME = os.environ.get("JLAWYER_USERNAME")
JLAWYER_PASSWORD = os.environ.get("JLAWYER_PASSWORD")
JLAWYER_TEMPLATE_FOLDER_DEFAULT = os.environ.get("JLAWYER_TEMPLATE_FOLDER")
JLAWYER_PLACEHOLDER_KEY = os.environ.get("JLAWYER_PLACEHOLDER_KEY", "HAUPTTEXT")

def _sanitize_filename_for_claude(filename: str) -> str:
    """Sanitize filename to only contain ASCII characters for Claude Files API."""
    # Replace German umlauts and special characters
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'
    }
    for char, replacement in replacements.items():
        filename = filename.replace(char, replacement)

    # Remove any remaining non-ASCII characters and keep only safe chars
    sanitized = ''.join(
        c if c.isascii() and (c.isalnum() or c in '.-_ ') else '_'
        for c in filename
    )

    # Clean up multiple underscores/spaces and trim
    sanitized = re.sub(r'[_\s]+', '_', sanitized).strip('_')

    # Ensure we keep the .pdf extension
    if not sanitized.lower().endswith('.pdf'):
        sanitized += '.pdf'

    return sanitized


def _normalize_jlawyer_file_number(value: str) -> str:
    return re.sub(r"\s+", "", value or "").casefold()


def _looks_like_jlawyer_case_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", (value or "").strip(), re.IGNORECASE))


async def _resolve_jlawyer_case_id(case_reference: str, auth: tuple[str, str]) -> str:
    """Resolve a user-supplied j-lawyer case reference to the internal case id.

    Accepts either the raw j-lawyer case id or a file number such as ``008/26``.
    """
    reference = (case_reference or "").strip()
    if not reference:
        return ""

    if _looks_like_jlawyer_case_id(reference):
        return reference

    wanted = _normalize_jlawyer_file_number(reference)
    url = f"{_jlawyer_api_base_url()}/v1/cases/list/active"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, auth=auth)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"j-lawyer Aktenauflösung fehlgeschlagen: {exc}")

    if response.status_code >= 400:
        detail = response.text or response.reason_phrase or "Unbekannter Fehler"
        raise HTTPException(status_code=502, detail=f"j-lawyer Aktenauflösung fehlgeschlagen ({response.status_code}): {detail}")

    try:
        cases = response.json()
    except Exception:
        raise HTTPException(status_code=502, detail="j-lawyer Aktenauflösung lieferte kein gültiges JSON")

    matches = [
        case for case in (cases or [])
        if _normalize_jlawyer_file_number(str(case.get("fileNumber") or "")) == wanted
    ]

    if not matches:
        raise HTTPException(status_code=400, detail=f"Keine aktive j-lawyer-Akte zum Aktenzeichen '{reference}' gefunden")

    if len(matches) > 1:
        raise HTTPException(status_code=409, detail=f"Mehrere aktive j-lawyer-Akten zum Aktenzeichen '{reference}' gefunden")

    resolved_case_id = str(matches[0].get("id") or "").strip()
    if not resolved_case_id:
        raise HTTPException(status_code=502, detail=f"j-lawyer Aktenauflösung für '{reference}' lieferte keine caseId")

    return resolved_case_id


def _build_inline_text_block(entry: Dict[str, Optional[str]]) -> Optional[str]:
    """Build an inline text fallback for sources without uploadable files."""
    title = _model_display_title(entry)
    description = (entry.get("description") or "").strip()
    content = (entry.get("content") or "").strip()
    url = (entry.get("url") or "").strip()

    text_parts = []
    if content:
        text_parts.append(content)
    elif description:
        text_parts.append(description)

    if url:
        text_parts.append(f"URL: {url}")

    if not text_parts:
        return None

    return f"DOKUMENT: {title}\n\n" + "\n\n".join(text_parts)


def _model_display_title(entry: Dict[str, Optional[str]], fallback: str = "document") -> str:
    title = entry.get("filename") or entry.get("title") or fallback
    if entry.get("category") == "internal_notes":
        return f"INTERNE KANZLEINOTIZ - NICHT ZITIEREN - {title}"
    return title


def _upload_documents_to_claude(client: anthropic.Anthropic, documents: List[Dict[str, Optional[str]]]) -> List[Dict[str, str]]:
    """Upload local documents using Claude Files API and return document content blocks.

    Prefers OCR'd text when available for better accuracy and significantly lower token usage.
    
    OPTIMIZATION:
    If we have extracted text (text/plain), we embed it DIRECTLY as a text block
    instead of using the Files API. This matches our benchmark findings where
    text embedding is ~2.3x more efficient than PDF upload and cleaner than text file upload.
    """
    content_blocks: List[Dict[str, str]] = []
    MAX_PAGES = 100  # Claude Files API limit

    for entry in documents:
        original_filename = _model_display_title(entry)

        try:
            # Get the appropriate file for upload (OCR text or original PDF)
            file_path, mime_type, needs_cleanup = get_document_for_upload(entry)

            if mime_type == "text/plain":
                text_type = "Anonymized" if entry.get("is_anonymized") else "OCR"
                print(f"[INFO] Embedding {text_type} Text for {original_filename} (skipping upload)")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create a text block with the document content, similar to OpenAI pipeline
                    text_block = f"DOKUMENT: {original_filename}\n\n{content}"
                    
                    content_blocks.append({
                        "type": "text",
                        "text": text_block
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to read text file {file_path}: {e}")

            else:
                # It's a PDF (or other supported binary)
                
                # OPTIMIZATION: Try to extract text from native PDFs on-the-fly
                # This catches PDFs that didn't need OCR (native text) and avoids expensive vision processing
                extracted_text = None
                if mime_type == "application/pdf":
                    try:
                        import fitz  # PyMuPDF
                        with fitz.open(file_path) as pdf_doc:
                            # 1. Quick check: Is there enough text?
                            full_text = ""
                            page_count = len(pdf_doc)
                            
                            # Limit extraction for huge documents to avoid context overflow, 
                            # but 100 pages is usually fine for text-only
                            pages_to_extract = min(page_count, 100) 
                            
                            for i in range(pages_to_extract):
                                full_text += pdf_doc[i].get_text() + "\n\n"
                                
                            # If we have substantial text (avg > 100 chars per page), assume it's good
                            if len(full_text) > (50 * pages_to_extract):
                                print(f"[INFO] On-the-fly text extraction successful for {original_filename} ({len(full_text)} chars)")
                                extracted_text = f"DOKUMENT: {original_filename} (Text-Extrakt)\n\n{full_text}"
                    except Exception as e:
                        print(f"[WARN] On-the-fly text extraction failed for {original_filename}: {e}")

                if extracted_text:
                    # Use the on-the-fly extracted text
                    content_blocks.append({
                        "type": "text",
                        "text": extracted_text
                    })
                else: 
                    # FALLBACK: Upload as File (Vision)
                    # Check PDF page count before uploading (only for PDFs)
                    if mime_type == "application/pdf":
                        try:
                            pdf = pikepdf.open(file_path)
                            page_count = len(pdf.pages)
                            pdf.close()

                            if page_count > MAX_PAGES:
                                print(f"[WARN] Datei {original_filename} hat {page_count} Seiten (max {MAX_PAGES}), wird übersprungen")
                                continue
                        except Exception as exc:
                            print(f"[WARN] Seitenzahl für {original_filename} konnte nicht ermittelt werden: {exc}")

                    sanitized_filename = _sanitize_filename_for_claude(original_filename)

                    # Upload file
                    try:
                        with open(file_path, "rb") as file_handle:
                            uploaded_file = client.beta.files.upload(
                                file=(sanitized_filename, file_handle, mime_type),
                                betas=["files-api-2025-04-14"],
                            )

                        content_blocks.append({
                            "type": "document",
                            "source": {
                                "type": "file",
                                "file_id": uploaded_file.id,
                            },
                            "title": original_filename,
                        })
                        print(f"[DEBUG] Uploaded {original_filename} (PDF) -> file_id: {uploaded_file.id}")

                    except Exception as exc:
                        print(f"[ERROR] Upload für {original_filename} fehlgeschlagen: {exc}")


        except (ValueError, FileNotFoundError) as exc:
            inline_text = _build_inline_text_block(entry)
            if inline_text:
                print(f"[INFO] Embedding inline source text for {original_filename} (no file upload available)")
                content_blocks.append({
                    "type": "text",
                    "text": inline_text,
                })
                continue
            print(f"[WARN] Überspringe {original_filename}: {exc}")
            continue
        finally:
            # Clean up temporary file if needed
            if 'needs_cleanup' in locals() and needs_cleanup:
                try:
                    os.unlink(file_path)
                except:
                    pass

    return content_blocks


def _upload_documents_to_openai(client: OpenAI, documents: List[Dict[str, Optional[str]]]) -> List[Dict[str, str]]:
    """Upload local documents using OpenAI Files API and return input_file blocks.

    Prefers anonymized text, then OCR text, then the original PDF.
    For text/plain content we embed it directly as input_text blocks since
    the Responses API does not accept .txt uploads.
    """
    file_blocks: List[Dict[str, str]] = []

    for entry in documents:
        original_filename = _model_display_title(entry)

        try:
            file_path, mime_type, needs_cleanup = get_document_for_upload(entry)

            if mime_type == "text/plain":
                text_type = "Anonymized" if entry.get("is_anonymized") else "OCR"
                print(f"[INFO] Embedding {text_type} Text for {original_filename} (skipping upload)")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    text_block = f"DOKUMENT: {original_filename}\n\n{content}"

                    file_blocks.append({
                        "type": "input_text",
                        "text": text_block,
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to read text file {file_path}: {e}")

            else:
                if not openai_file_uploads_enabled():
                    print(f"[INFO] Embedding extracted PDF text for {original_filename} (Azure file uploads disabled)")
                    try:
                        from .ocr import extract_pdf_text

                        content = extract_pdf_text(
                            file_path,
                            max_pages=50,
                            include_page_headers=True,
                        ).strip()
                        if content:
                            file_blocks.append({
                                "type": "input_text",
                                "text": f"DOKUMENT: {original_filename}\n\n{content}",
                            })
                        else:
                            print(f"[WARN] No extractable PDF text for {original_filename}; run OCR first.")
                    except Exception as exc:
                        print(f"[ERROR] Failed to extract PDF text for {original_filename}: {exc}")
                    continue

                print(f"[INFO] Uploading PDF for {original_filename}")
                try:
                    with open(file_path, "rb") as file_handle:
                        uploaded_file = client.files.create(
                            file=file_handle,
                            purpose="user_data"
                        )

                    file_blocks.append({
                        "type": "input_file",
                        "file_id": uploaded_file.id,
                    })
                    print(f"[DEBUG] Uploaded {original_filename} ({mime_type}) -> file_id: {uploaded_file.id}")
                except Exception as exc:
                    print(f"[ERROR] OpenAI upload failed for {original_filename}: {exc}")

        except (ValueError, FileNotFoundError) as exc:
            inline_text = _build_inline_text_block(entry)
            if inline_text:
                print(f"[INFO] Embedding inline source text for {original_filename} (no file upload available)")
                file_blocks.append({
                    "type": "input_text",
                    "text": inline_text,
                })
                continue
            print(f"[WARN] Skipping {original_filename}: {exc}")
            continue
        finally:
            if 'needs_cleanup' in locals() and needs_cleanup:
                try:
                    os.unlink(file_path)
                except:
                    pass

    return file_blocks


@router.post("/generate")
@limiter.limit("10/hour")
async def generate(
    request: Request,
    body: GenerationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Generate drafts using Claude Opus 4.7, GPT-5, Gemini, or multi-step flow (Streaming)."""
    # TODO: Add a Rechtsprechung playbook file (CLAUDE.md/AGENTS.md-style) that the generate endpoint
    # learns from to keep the latest jurisprudence and typical argumentation for common case types.
    
    target_case_id = resolve_case_uuid_for_request(db, current_user, body.case_id)

    print(f"[DEBUG] Incoming selection: bescheid.primary={body.selected_documents.bescheid.primary}, bescheid.others={body.selected_documents.bescheid.others}")
    print(f"[DEBUG] Incoming selection: anhoerung={body.selected_documents.anhoerung}, rechtsprechung={body.selected_documents.rechtsprechung}")
    history_len = len(body.chat_history or [])
    history_roles = [msg.get("role", "unknown") for msg in (body.chat_history or [])[:6]]
    print(f"[DEBUG] Generate chat_history: {history_len} messages, roles={history_roles}")

    start_time = time.time()
    resolved_legal_area = (getattr(body, "legal_area", None) or "migrationsrecht").lower()
    explicit_field_set = getattr(body, "model_fields_set", None) or getattr(body, "__fields_set__", set()) or set()
    legal_area_explicit = "legal_area" in explicit_field_set
    print(f"[DEBUG] Generate legal_area resolved={resolved_legal_area} explicit={legal_area_explicit}")
    collected = _collect_selected_documents(
        body.selected_documents,
        db,
        current_user,
        target_case_id,
        require_bescheid=False,
    )

    # Flatten for context window
    document_entries = []
    for cat, items in collected.items():
        document_entries.extend(items)
        
    print(f"[DEBUG] Collected {len(document_entries)} document entries for upload")
    
    # 3. Build prompts
    context_summary = _summarize_selection_for_prompt(collected)
    case_memory_context = ""
    if target_case_id and get_case_memory_prompt_context:
        try:
            case_memory_context = (
                get_case_memory_prompt_context(
                    db,
                    current_user,
                    target_case_id,
                    include_strategy=True,
                )
                or ""
            ).strip()
        except Exception as exc:
            print(f"[WARN] Failed to load case memory context for generation: {exc}")
            case_memory_context = ""
    case_memory_block = (
        f"KOMPAKTES FALLGEDÄCHTNIS:\n{case_memory_context}\n\n"
        if case_memory_context
        else ""
    )
    if case_memory_block:
        context_summary = f"{case_memory_block}{context_summary}"
    
    primary_bescheid_entry = next(
        (entry for entry in collected.get("bescheid", []) if entry.get("role") == "primary"),
        None,
    )
    primary_bescheid_label = (
        (primary_bescheid_entry.get("filename") or "—") if primary_bescheid_entry else "—"
    )
    primary_bescheid_description = ""
    if primary_bescheid_entry:
        explanation = primary_bescheid_entry.get("explanation")
        if explanation:
            primary_bescheid_description = explanation.strip()

    if body.chat_history:
        print("[DEBUG] Amelioration detected: Using raw user_prompt")
        sys_p, _ = _build_generation_prompts(
            body, collected, primary_bescheid_label,
            primary_bescheid_description, context_summary
        )
        system_prompt = sys_p
        user_prompt = f"{case_memory_block}{body.user_prompt}" if case_memory_block else body.user_prompt
    else:
        system_prompt, user_prompt = _build_generation_prompts(
            body, collected, primary_bescheid_label,
            primary_bescheid_description, context_summary
        )

    prompt_for_generation = user_prompt

    # STREAM GENERATOR
    async def stream_generator():
        generated_text_acc = []
        thinking_text_acc = []
        token_usage_acc = None
        
        try:
            # 4. Route to provider
            if body.model in {"two-step-expert", "multi-step-expert"}:
                # Multi-step is tricky to stream step-by-step unless we rewrite it to yield events.
                # For now, we will execute it synchronously (blocking threadpool) and yield one big chunks.
                # Or yield progress events?
                expert_label = "Two-Step Expert" if body.model == "two-step-expert" else "Multi-Step Expert"
                yield json.dumps({"type": "thinking", "text": f"Starting {expert_label} Workflow...\n"}) + "\n"
                
                # ... (Existing Multi-step logic) ...
                # To avoid duplicating massive logic inside the generator, we should ideally refactor multi-step to be a generator too.
                # But for time constraints, we will call the synchronous logic and yield the result.
                # NOTE: This will NOT solve timeout if multi-step takes > 10m and is opaque.
                # But Multi-Step uses multiple calls, each < 10m likely.
                # However, the user asked to fix Claude timeout.
                
                # We can replicate the logic:
                if body.chat_history:
                     # Gemini Amelioration logic
                     client = get_gemini_client()
                     files = _upload_documents_to_gemini(client, document_entries)
                     default_model = _get_gemini_generation_model()
                     text, usage = _generate_with_gemini(
                        client, system_prompt, prompt_for_generation, files,
                        chat_history=body.chat_history,
                        model=default_model,
                     )
                     generated_text_acc.append(text)
                     yield json.dumps({"type": "text", "text": text}) + "\n"
                     token_usage_acc = usage
                else:
                    gemini_client = get_gemini_client()
                    gemini_files = _upload_documents_to_gemini(gemini_client, document_entries)

                    if body.model == "two-step-expert":
                        yield json.dumps({"type": "thinking", "text": "[Step 1/2] Drafting with Gemini 3...\n"}) + "\n"
                    else:
                        yield json.dumps({"type": "thinking", "text": "[Step 1/3] Drafting with Gemini 3...\n"}) + "\n"

                    draft_text, draft_usage = _generate_with_gemini(
                        gemini_client, system_prompt, prompt_for_generation, gemini_files,
                        chat_history=[],
                        model=_get_gemini_generation_model()
                    )

                    if body.model == "two-step-expert":
                        yield json.dumps({"type": "thinking", "text": "\n[Step 2/2] Final Review and Rewrite with GPT-5.5...\n"}) + "\n"
                    else:
                        yield json.dumps({"type": "thinking", "text": "\n[Step 2/3] Critiquing with GPT-5.5...\n"}) + "\n"

                    openai_client = get_openai_client()
                    openai_file_blocks = _upload_documents_to_openai(openai_client, document_entries)
                    if body.model == "two-step-expert":
                        final_system_prompt = (
                            "Du bist ein sehr strenger Senior Partner einer migrationsrechtlichen Kanzlei.\n"
                            "Deine Aufgabe ist es, den vorliegenden ENTWURF auf Basis der beigefügten Dokumente vollständig zu überarbeiten und zu finalisieren.\n"
                            "Arbeite dabei wie ein Red-Team-Reviewer und Endredakteur zugleich.\n"
                            "Prüfe den Entwurf auf Halluzinationen, unzulässige Tatsachenzusätze, logische Sprünge, überdehnte Rechtsprechung, unpräzise Verweise und überzogene Tonalität.\n"
                            "Korrigiere alle diese Punkte unmittelbar im Endtext.\n"
                            "Erstelle am Ende nur die bereinigte finale Fassung des Schriftsatzes, ohne Vorbemerkung und ohne Aufzählung der Mängel.\n\n"
                            f"{NEUTRAL_LEGAL_TONE_RULES}"
                        )
                        final_user_prompt = (
                            f"Hier ist der zu überarbeitende Entwurf:\n\n{draft_text}\n\n"
                            "Erstelle nun auf Basis der beigefügten Dokumente die prozessfest bereinigte Endfassung."
                        )
                        final_text, final_usage = _generate_with_gpt5(
                            openai_client, final_system_prompt, final_user_prompt, openai_file_blocks,
                            chat_history=[],
                            reasoning_effort=OPENAI_GPT5_REASONING_EFFORT,
                            verbosity="medium",
                            model="gpt-5.5"
                        )
                        generated_text_acc.append(final_text)
                        yield json.dumps({"type": "text", "text": final_text}) + "\n"
                        token_usage_acc = _merge_token_usages(draft_usage, final_usage, model="two-step-expert")
                    else:
                        critique_system_prompt = _build_senior_partner_critique_prompt()
                        critique_user_prompt = f"Hier ist der zu prüfende Entwurf:\n\n{draft_text}"
                        critique_text, critique_usage = _generate_with_gpt5(
                            openai_client, critique_system_prompt, critique_user_prompt, openai_file_blocks,
                            chat_history=[],
                            reasoning_effort=OPENAI_GPT5_REASONING_EFFORT,
                            verbosity="low",
                            model="gpt-5.5"
                        )
                        yield json.dumps({"type": "thinking", "text": f"Critique: {critique_text[:200]}...\n"}) + "\n"

                        yield json.dumps({"type": "thinking", "text": "\n[Step 3/3] Finalizing with Gemini 3...\n"}) + "\n"
                        final_system_prompt = _build_gemini_finalize_system_prompt()
                        final_user_prompt = (
                            f"ENTWURF (mit Vorüberlegungen):\n{draft_text}\n\n"
                            f"KRITIK DES SENIOR PARTNERS:\n{critique_text}\n\n"
                            "Erstelle nun die finale, bereinigte Version (ohne <document_analysis> etc.)."
                        )

                        final_text, final_usage = _generate_with_gemini(
                            gemini_client, final_system_prompt, final_user_prompt, gemini_files,
                            chat_history=[],
                            model=_get_gemini_generation_model()
                        )

                        generated_text_acc.append(final_text)
                        yield json.dumps({"type": "text", "text": final_text}) + "\n"
                        token_usage_acc = _merge_token_usages(
                            draft_usage, critique_usage, final_usage, model="multi-step-expert"
                        )

            elif body.model.startswith("gpt"):
                client = get_openai_client()
                file_blocks = _upload_documents_to_openai(client, document_entries)
                input_messages = _build_openai_input_messages(
                    system_prompt, prompt_for_generation, file_blocks, body.chat_history
                )
                openai_stream_completed = False
                print(f"[DEBUG] Streaming GPT Responses API:")
                print(f"  Model: {body.model}")
                resolved_openai_model = resolve_openai_model(body.model)
                print(f"  Files: {len(file_blocks)}")
                print(f"  Reasoning effort: {OPENAI_GPT5_REASONING_EFFORT}")
                print(f"  Output verbosity: {body.verbosity}")
                print(f"  Max output tokens: {OPENAI_GPT5_MAX_OUTPUT_TOKENS}")

                stream = client.responses.create(
                    model=resolved_openai_model,
                    input=input_messages,
                    reasoning={"effort": OPENAI_GPT5_REASONING_EFFORT},
                    text={"verbosity": body.verbosity},
                    max_output_tokens=OPENAI_GPT5_MAX_OUTPUT_TOKENS,
                    stream=True,
                )

                for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            generated_text_acc.append(delta)
                            yield json.dumps({"type": "text", "text": delta}) + "\n"
                    elif event_type == "response.completed":
                        openai_stream_completed = True
                        response = getattr(event, "response", None)
                        usage = getattr(response, "usage", None) if response else None
                        if usage:
                            output_details = getattr(usage, "output_tokens_details", None)
                            reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) if output_details else 0
                            token_usage_acc = TokenUsage(
                                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                                thinking_tokens=reasoning_tokens or 0,
                                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                                cost_usd=round(
                                    _estimate_openai_gpt5_cost_usd(
                                        getattr(usage, "input_tokens", 0) or 0,
                                        getattr(usage, "output_tokens", 0) or 0,
                                    ),
                                    4,
                                ),
                                model=body.model,
                            )
                    elif event_type in {"response.failed", "response.incomplete"}:
                        response = getattr(event, "response", None)
                        status = getattr(response, "status", None) if response else None
                        incomplete_reason = _extract_openai_incomplete_reason(response) if response else ""
                        status_text = f" ({status})" if status else ""
                        if incomplete_reason:
                            print(f"[WARN] OpenAI incomplete_details.reason: {incomplete_reason}")
                            raise RuntimeError(
                                f"OpenAI stream ended without completion{status_text}: {incomplete_reason}"
                            )
                        raise RuntimeError(f"OpenAI stream ended without completion{status_text}")
                    elif event_type == "error":
                        message = getattr(event, "message", None) or str(event)
                        raise RuntimeError(message)
                    else:
                        continue

                if not openai_stream_completed:
                    raise RuntimeError("OpenAI stream ended before response.completed")
                
            elif body.model.startswith("gemini"):
                client = get_gemini_client()
                files = _upload_documents_to_gemini(client, document_entries)
                text, usage = _generate_with_gemini(
                    client, system_prompt, prompt_for_generation, files,
                    chat_history=body.chat_history,
                    model=body.model
                )
                generated_text_acc.append(text)
                yield json.dumps({"type": "text", "text": text}) + "\n"
                token_usage_acc = usage
                
            else:
                # Claude (Streaming)
                print(f"[DEBUG] Using Anthropic Claude: {body.model} (STREAMING)")
                client = get_anthropic_client()
                document_blocks = _upload_documents_to_claude(client, document_entries)
                
                for chunk_str in _generate_with_claude_stream(
                    client, system_prompt, prompt_for_generation, document_blocks,
                    chat_history=body.chat_history
                ):
                    chunk = json.loads(chunk_str)
                    if chunk["type"] == "text":
                        generated_text_acc.append(chunk["text"])
                    elif chunk["type"] == "thinking":
                        thinking_text_acc.append(chunk["text"])
                    elif chunk["type"] == "usage":
                        token_usage_acc = TokenUsage(**chunk["data"])
                    yield chunk_str
                    
        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"type": "error", "message": _format_stream_exception(e)}) + "\n"
            return

        # FINALIZE AND SAVE
        generated_text = "".join(generated_text_acc)
        thinking_text = "".join(thinking_text_acc)
        
        citation_checks, citation_check_warnings = await run_citation_checks(generated_text, collected)
        citation_summary = citation_checks.get("summary") or {}
        
        # Build metadata
        metadata = GenerationMetadata(
            documents_used={
                "anhoerung": len(collected.get("anhoerung", [])),
                "bescheid": len(collected.get("bescheid", [])),
                "rechtsprechung": len(collected.get("rechtsprechung", [])),
                "saved_sources": len(collected.get("saved_sources", [])),
                "akte": len(collected.get("akte", [])),
                "sonstiges": len(collected.get("sonstiges", [])),
                "internal_notes": len(collected.get("internal_notes", [])),
            },
            resolved_legal_area=resolved_legal_area,
            citations_found=int(citation_summary.get("verified_on_cited_page", 0) or 0),
            missing_citations=[],
            pinpoint_missing=[],
            citation_checks=citation_checks,
            warnings=citation_check_warnings,
            word_count=len(generated_text.split()),
            token_count=token_usage_acc.total_tokens if token_usage_acc else 0,
            token_usage=token_usage_acc,
        )
        if not legal_area_explicit:
            metadata.warnings.append(
                f"legal_area nicht explizit gesetzt, Fallback auf '{resolved_legal_area}' verwendet."
            )
        
        # Send metadata event
        yield json.dumps({"type": "metadata", "data": metadata.model_dump()}) + "\n"
        
        # Save to DB
        draft_id = None
        try:
            # Re-construct used documents list
            structured_used_documents = []
            for cat, entries in collected.items():
                for entry in entries:
                     fname = entry.get("filename") or entry.get("title")
                     if fname:
                        payload = {"filename": fname, "category": cat}
                        role = entry.get("role")
                        if role:
                            payload["role"] = role
                        structured_used_documents.append(payload)

            draft = GeneratedDraft(
                user_id=current_user.id,
                primary_document_id=uuid.UUID(primary_bescheid_entry["id"]) if primary_bescheid_entry else None,
                case_id=target_case_id,
                document_type=body.document_type,
                user_prompt=body.user_prompt,
                generated_text=generated_text,
                model_used=body.model,
                metadata_={
                    "tokens": token_usage_acc.total_tokens if token_usage_acc else 0,
                    "estimated_cost_usd": token_usage_acc.cost_usd if token_usage_acc else None,
                    "token_usage": token_usage_acc.model_dump() if token_usage_acc else None,
                    "used_documents": structured_used_documents,
                    "resolved_legal_area": resolved_legal_area,
                    "thinking_text": thinking_text, # Save thinking too if available
                    "citation_checks": citation_checks,
                }
            )
            db.add(draft)
            db.commit()
            db.refresh(draft)
            draft_id = str(draft.id)
            print(f"[INFO] Saved generated draft with ID: {draft_id}")
        except Exception as e:
            print(f"[ERROR] Failed to save draft: {e}")
            
        yield json.dumps(
            {
                "type": "done",
                "draft_id": draft_id,
                "case_id": str(target_case_id) if target_case_id else None,
                "document_type": body.document_type,
                "model_used": body.model,
                "resolved_legal_area": resolved_legal_area,
                "token_usage": token_usage_acc.model_dump() if token_usage_acc else None,
                "estimated_cost_usd": token_usage_acc.cost_usd if token_usage_acc else None,
                "word_count": metadata.word_count,
            }
        ) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


def _generation_job_to_response(job: GenerationJob) -> GenerationJobResponse:
    return GenerationJobResponse(
        id=str(job.id),
        status=job.status,
        case_id=str(job.case_id) if job.case_id else None,
        draft_id=str(job.draft_id) if job.draft_id else None,
        error_message=job.error_message,
        claimed_by=job.claimed_by,
        claimed_at=job.claimed_at.isoformat() if job.claimed_at else None,
        heartbeat_at=job.heartbeat_at.isoformat() if job.heartbeat_at else None,
        available_at=job.available_at.isoformat() if job.available_at else None,
        attempt_count=int(job.attempt_count or 0),
        created_at=job.created_at.isoformat() if job.created_at else None,
        updated_at=job.updated_at.isoformat() if job.updated_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        result_payload=job.result_payload or None,
    )


async def _run_generation_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job_uuid = uuid.UUID(job_id)
        job = db.query(GenerationJob).filter(GenerationJob.id == job_uuid).first()
        if not job:
            return

        job.status = "running"
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()

        user = db.query(User).filter(User.id == job.owner_id).first()
        if not user:
            raise RuntimeError("Owner user not found for generation job")

        request_payload = dict(job.request_payload or {})
        body = GenerationRequest.model_validate(request_payload)
        result = await _execute_generation_request(body, db, user)

        draft_id = result.get("draft_id")
        job.status = "completed"
        job.result_payload = result
        job.draft_id = uuid.UUID(draft_id) if draft_id else None
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()
    except Exception as exc:
        print(f"[ERROR] Generation job failed {job_id}: {exc}")
        db.rollback()
        try:
            failed_job = db.query(GenerationJob).filter(GenerationJob.id == uuid.UUID(job_id)).first()
            if failed_job:
                failed_job.status = "failed"
                failed_job.error_message = _format_stream_exception(exc)
                failed_job.completed_at = datetime.utcnow()
                failed_job.updated_at = datetime.utcnow()
                db.commit()
        except Exception as nested_exc:
            print(f"[ERROR] Failed to persist generation job failure state {job_id}: {nested_exc}")
            db.rollback()
    finally:
        db.close()


@router.post("/generate/jobs", response_model=GenerationJobResponse, status_code=202)
@limiter.limit("20/hour")
async def create_generation_job(
    request: Request,
    response: Response,
    body: GenerationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a background generation job for CLI/API usage."""
    target_case_id = resolve_case_uuid_for_request(db, current_user, body.case_id)
    request_payload = body.model_dump()
    request_payload["case_id"] = str(target_case_id) if target_case_id else None

    job = GenerationJob(
        owner_id=current_user.id,
        case_id=target_case_id,
        status="queued",
        request_payload=request_payload,
        result_payload={},
        updated_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    response.headers["Location"] = f"/generate/jobs/{job.id}"
    return _generation_job_to_response(job)


@router.get("/generate/jobs/{job_id}", response_model=GenerationJobResponse)
@limiter.limit("120/hour")
async def get_generation_job(
    request: Request,
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    job = (
        db.query(GenerationJob)
        .filter(
            GenerationJob.id == job_id,
            GenerationJob.owner_id == current_user.id,
        )
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")
    return _generation_job_to_response(job)


@router.get("/generate/jobs/{job_id}/result")
@limiter.limit("120/hour")
async def get_generation_job_result(
    request: Request,
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    job = (
        db.query(GenerationJob)
        .filter(
            GenerationJob.id == job_id,
            GenerationJob.owner_id == current_user.id,
        )
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")
    if job.status == "failed":
        raise HTTPException(status_code=409, detail=job.error_message or "Generation job failed")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Generation job not completed ({job.status})")
    return JSONResponse(content=job.result_payload or {})


@router.get("/generate/jobs/{job_id}/events")
async def stream_generation_job_events(
    request: Request,
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Simple SSE stream for generation job status changes."""

    def _load_job_payload(session: Session) -> Optional[Dict[str, Any]]:
        job = (
            session.query(GenerationJob)
            .filter(
                GenerationJob.id == job_id,
                GenerationJob.owner_id == current_user.id,
            )
            .first()
        )
        return job.to_dict() if job else None

    async def event_generator():
        last_updated = None
        while True:
            if await request.is_disconnected():
                break
            session = SessionLocal()
            try:
                payload = _load_job_payload(session)
            finally:
                session.close()

            if payload is None:
                yield "event: error\ndata: {\"detail\":\"Generation job not found\"}\n\n"
                break

            updated_at = payload.get("updated_at")
            if updated_at != last_updated:
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_updated = updated_at

            if payload.get("status") in {"completed", "failed"}:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


_CATEGORY_LABELS = {
    "anhoerung": "Anhörung",
    "bescheid": "Bescheid",
    "vorinstanz": "Vorinstanz",
    "rechtsprechung": "Rechtsprechung",
    "saved_sources": "Gespeicherte Quelle",
}


def _normalize_for_match(value: Optional[str]) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^\w\s/.:,-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


_DATE_REGEX = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})")
_AZ_REGEX = re.compile(r"(?:Az\.?|Aktenzeichen)\s*[:]?\s*([A-Za-z0-9./\-\s]+)")


def _build_reference_candidates(category: str, entry: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
    specific: List[str] = []
    generic: List[str] = []

    filename = entry.get("filename")
    if filename:
        stem = Path(filename).stem
        specific.extend(
            {
                filename,
                stem,
                stem.replace("_", " "),
                stem.replace("-", " "),
                stem.replace("_", "-"),
            }
        )

    title = entry.get("title")
    if title:
        specific.append(title)

    explanation = entry.get("explanation")
    if explanation:
        specific.append(explanation)
        specific.extend(_DATE_REGEX.findall(explanation))
        specific.extend([f"vom {m}" for m in _DATE_REGEX.findall(explanation)])
        for az in _AZ_REGEX.findall(explanation):
            cleaned = az.strip()
            if cleaned:
                specific.append(cleaned)
                specific.append(f"az {cleaned.lower()}")
                specific.append(f"az. {cleaned}")

    url = entry.get("url")
    if url:
        specific.append(url)
        try:
            parsed = urlparse(url)
            if parsed.netloc:
                specific.append(parsed.netloc)
            if parsed.path:
                specific.append(parsed.path)
        except Exception:
            pass

    description = entry.get("description")
    if description:
        specific.append(description)
        specific.extend(_DATE_REGEX.findall(description))
        specific.extend([f"vom {m}" for m in _DATE_REGEX.findall(description)])
        for az in _AZ_REGEX.findall(description):
            cleaned = az.strip()
            if cleaned:
                specific.append(cleaned)
                specific.append(f"az {cleaned.lower()}")
                specific.append(f"az. {cleaned}")

    for key in ("date", "aktenzeichen"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            if key == "date":
                specific.append(value.strip())
                specific.append(f"vom {value.strip()}")
            else:
                specific.append(value.strip())
                specific.append(f"az {value.strip().lower()}")
                specific.append(f"az. {value.strip()}")

    category_generic = {
        "anhoerung": ["Anhörung"],
        "bescheid": ["Bescheid"],
        "vorinstanz": ["Vorinstanz", "Urteil", "Protokoll"],
        "rechtsprechung": ["Rechtsprechung", "Urteil"],
        "saved_sources": ["Quelle", "Research"],
    }
    generic.extend(category_generic.get(category, []))

    def _dedupe(items):
        seen = set()
        result = []
        for item in items:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    return {"specific": _dedupe(specific), "generic": _dedupe(generic)}


def _generate_with_claude_stream(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_prompt: Optional[str],
    document_blocks: List[Dict],
    chat_history: List[Dict[str, str]] = []
):
    """
    Generator that calls Claude API with extended thinking and yields NDJSON events.
    Yields:
        - {"type": "thinking", "text": "..."}
        - {"type": "text", "text": "..."}
        - {"type": "usage", "data": {...}}
        - {"type": "done"}
    """
    
    messages = []

    if chat_history:
        # 1. First message: Files + Initial Prompt
        first_msg = chat_history[0]
        first_content = []
        # Add documents to the first message
        first_content.extend(document_blocks)
        first_content.append({"type": "text", "text": first_msg.get("content", "")})
        messages.append({"role": "user", "content": first_content})

        # 2. Subsequent messages
        for msg in chat_history[1:]:
            role = msg.get("role")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})
        
        # 3. Current prompt
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
            
    else:
        # Standard single-turn
        content = []
        if user_prompt:
            content.append({"type": "text", "text": user_prompt})
        content.extend(document_blocks)
        
        if not content:
             return
             
        messages.append({"role": "user", "content": content})

    # Debug log
    print(f"[DEBUG] Claude Messages: {len(messages)} turns")

    # Enable extended thinking with budget
    # Note: Using beta API with streaming for long-running requests
    print("[DEBUG] Starting Claude streaming request with adaptive thinking...")
    
    # Use streaming to avoid timeout on long thinking operations
    with client.beta.messages.stream(
        model="claude-opus-4-7",
        system=system_prompt,
        max_tokens=20000, 
        messages=messages,
        thinking={
            "type": "adaptive"
        },
        betas=["files-api-2025-04-14", "interleaved-thinking-2025-05-14"],
    ) as stream:
        for event in stream:
            # We can iterate over specific event types if we want granular control
            # But the stream object simplifies this if we use specialized iterators:
            # - stream.text_stream
            # - stream.events
            # But here 'event' is likely a MessageStreamEvent if we iterate directly?
            # Actually, `client.beta.messages.stream` context manager returns a `MessageStreamManager` -> `MessageStream`.
            # We can iterate `stream` directly to get events.
            
            # Map event type to our NDJSON format
            # Event types: content_block_start, content_block_delta, content_block_stop, etc.
            
            if event.type == "content_block_start":
                block = event.content_block
                if block.type == "thinking":
                    # Thinking block started
                    pass
            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "thinking_delta":
                    yield json.dumps({"type": "thinking", "text": delta.thinking}) + "\n"
                elif delta.type == "text_delta":
                    yield json.dumps({"type": "text", "text": delta.text}) + "\n"
            elif event.type == "message_stop":
                pass
    
        # Get final message to extract usage
        response = stream.get_final_message()
    
    # Log API response metadata
    stop_reason = getattr(response, "stop_reason", None)
    usage = getattr(response, "usage", None)
    
    if stop_reason == "max_tokens":
        print("[WARN] Generation stopped due to max_tokens limit - output may be incomplete!")

    # Extract detailed token usage
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    
    total_tokens = input_tokens + output_tokens
    
    # Calculate cost
    OPUS_INPUT_PRICE = 5.0 / 1_000_000
    OPUS_OUTPUT_PRICE = 25.0 / 1_000_000
    OPUS_CACHE_READ_PRICE = 0.5 / 1_000_000
    OPUS_CACHE_WRITE_PRICE = 6.25 / 1_000_000
    
    cost_usd = (
        (input_tokens - cache_read_tokens) * OPUS_INPUT_PRICE +
        output_tokens * OPUS_OUTPUT_PRICE +
        cache_read_tokens * OPUS_CACHE_READ_PRICE +
        cache_create_tokens * OPUS_CACHE_WRITE_PRICE
    )
    
    # Construct usage payload
    usage_data = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": 0, # Included in output
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_create_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost_usd, 4),
        "model": "claude-opus-4-7"
    }
    
    yield json.dumps({"type": "usage", "data": usage_data}) + "\n"



def _build_openai_input_messages(
    system_prompt: str,
    user_prompt: Optional[str],
    file_blocks: List[Dict],
    chat_history: List[Dict[str, str]],
) -> List[Dict]:
    """Build OpenAI Responses API input messages with files and optional history."""
    input_messages = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": system_prompt}
            ]
        }
    ]

    # If we have history, we need to reconstruct the conversation.
    # The first user message MUST contain the files.
    if chat_history:
        first_user_msg = chat_history[0]
        input_messages.append({
            "role": "user",
            "content": [
                *file_blocks,
                {"type": "input_text", "text": first_user_msg.get("content", "")}
            ]
        })

        # Subsequent messages (Assistant replies, User follow-ups)
        remaining_history = chat_history[1:]
        if len(remaining_history) > 4:
            print(f"[DEBUG] Truncating history from {len(remaining_history)} to last 4 messages to save tokens.")
            remaining_history = remaining_history[-4:]

        for msg in remaining_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                input_messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}]
                })
            elif role == "assistant":
                input_messages.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}]
                })

        # Current user prompt (if not already in history)
        if user_prompt and (not chat_history or chat_history[-1].get("content") != user_prompt):
            input_messages.append({
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}]
            })

    else:
        # No history - Standard single-turn request
        input_messages.append({
            "role": "user",
            "content": [
                *file_blocks,
                {"type": "input_text", "text": user_prompt}
            ]
        })

    return input_messages


def _generate_with_gpt5(
    client: OpenAI,
    system_prompt: str,
    user_prompt: Optional[str],
    file_blocks: List[Dict],
    chat_history: List[Dict[str, str]] = [],
    reasoning_effort: str = OPENAI_GPT5_REASONING_EFFORT,
    verbosity: str = "high",
    model: str = "gpt-5.5"
) -> tuple[str, TokenUsage]:
    """Call GPT-5 Responses API and return generated text.

    Uses OpenAI Responses API with:
    - Reasoning effort: configurable (minimal/low/medium/high)
    - Output verbosity: configurable (low/medium/high)
    - Max output tokens: configurable via OPENAI_GPT5_MAX_OUTPUT_TOKENS
    """

    input_messages = _build_openai_input_messages(
        system_prompt, user_prompt, file_blocks, chat_history
    )

    print(f"[DEBUG] Calling GPT-5.x Responses API:")
    print(f"  Model: {model}")
    print(f"  Files: {len(file_blocks)}")
    print(f"  Reasoning effort: {reasoning_effort}")
    print(f"  Output verbosity: {verbosity}")
    print(f"  Max output tokens: {OPENAI_GPT5_MAX_OUTPUT_TOKENS}")

    # Responses API call
    # Note: temperature, top_p, logprobs NOT supported for GPT-5!
    response = client.responses.create(
        model=resolve_openai_model(model),
        input=input_messages,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity},
        max_output_tokens=OPENAI_GPT5_MAX_OUTPUT_TOKENS,
    )

    # Extract text from response
    # Method 1: Direct property (if available)
    if hasattr(response, 'output_text') and response.output_text:
        generated_text = response.output_text
    # Method 2: Navigate output structure
    elif response.output and len(response.output) > 0:
        message = response.output[0]
        if hasattr(message, 'content') and message.content and len(message.content) > 0:
            content_block = message.content[0]
            generated_text = content_block.text if hasattr(content_block, 'text') else ""
        else:
            generated_text = ""
    else:
        generated_text = ""

    # Log response metadata
    if hasattr(response, 'status'):
        print(f"[DEBUG] GPT-5 Response - status: {response.status}")

    if hasattr(response, 'usage'):
        usage = response.usage
        print(f"[DEBUG] GPT-5 Response - tokens: input={usage.input_tokens}, output={usage.output_tokens}, total={usage.total_tokens}")

        # Log reasoning tokens (unique to reasoning models)
        if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
            reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0)
            if reasoning_tokens > 0:
                print(f"[DEBUG] GPT-5 Reasoning tokens: {reasoning_tokens} ({reasoning_tokens/usage.output_tokens*100:.1f}% of output)")

    # Check if response was completed
    if hasattr(response, 'status') and response.status != 'completed':
        print(f"[WARN] GPT-5 response status: {response.status}")
        if hasattr(response, 'incomplete_details') and response.incomplete_details:
            print(f"[WARN] Incomplete details: {response.incomplete_details}")
            reason = _extract_openai_incomplete_reason(response)
            if reason:
                print(f"[WARN] Incomplete reason: {reason}")
        reason = _extract_openai_incomplete_reason(response)
        if reason:
            raise RuntimeError(f"OpenAI response incomplete: {reason}")
        raise RuntimeError(f"OpenAI response incomplete: {response.status}")

    usage_payload = TokenUsage(model=model)
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        output_details = getattr(usage, 'output_tokens_details', None)
        reasoning_tokens = getattr(output_details, 'reasoning_tokens', 0) if output_details else 0
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        usage_payload = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=reasoning_tokens or 0,
            total_tokens=getattr(usage, 'total_tokens', 0) or 0,
            cost_usd=round(_estimate_openai_gpt5_cost_usd(input_tokens, output_tokens), 4),
            model=model,
        )

    return generated_text.strip(), usage_payload


def _upload_documents_to_gemini(client: genai.Client, documents: List[Dict[str, Optional[str]]]) -> List[types.File]:
    """Upload local documents using Gemini Files API.

    Prefers anonymized text, then OCR text, then original PDFs.
    """
    uploaded_files: List[types.File] = []

    # We need a DB session to find the ORM objects for specific documents
    from database import SessionLocal
    db = SessionLocal()

    try:
        for entry in documents:
            original_filename = _model_display_title(entry)
            needs_cleanup = False

            try:
                file_path, mime_type, needs_cleanup = get_document_for_upload(entry)

                if mime_type == "text/plain":
                    print(f"[INFO] Uploading text for {original_filename} to Gemini")
                    with open(file_path, "rb") as f:
                        uploaded_file = client.files.upload(
                            file=f,
                            config={
                                "mime_type": "text/plain",
                                "display_name": f"{original_filename}.txt",
                            },
                        )
                    uploaded_files.append(uploaded_file)
                    continue

                doc_id = entry.get("id")
                if doc_id:
                    try:
                        doc_uuid = uuid.UUID(doc_id)
                        db_obj = None

                        if entry.get("category") == "saved_source" or entry.get("document_type") in ["Rechtsprechung", "Quelle"]:
                            if entry.get("category") == "saved_source":
                                db_obj = db.query(ResearchSource).filter(ResearchSource.id == doc_uuid).first()
                            else:
                                db_obj = db.query(Document).filter(Document.id == doc_uuid).first()
                        else:
                            db_obj = db.query(Document).filter(Document.id == doc_uuid).first()

                        if db_obj:
                            gemini_file = ensure_document_on_gemini(db_obj, db)
                            if gemini_file:
                                uploaded_files.append(gemini_file)
                                continue
                        else:
                            print(f"[WARN] Document {doc_id} not found in DB for Gemini upload.")
                    except Exception as e:
                        print(f"[WARN] Failed to reuse Gemini file for {original_filename}: {e}")

                print(f"[INFO] Uploading PDF for {original_filename} to Gemini")
                with open(file_path, "rb") as f:
                    uploaded_file = client.files.upload(
                        file=f,
                        config={
                            "mime_type": mime_type,
                            "display_name": original_filename,
                        },
                    )
                uploaded_files.append(uploaded_file)

            except Exception as e:
                inline_text = _build_inline_text_block(entry)
                if inline_text:
                    print(f"[INFO] Uploading inline source text for {original_filename} to Gemini")
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            encoding="utf-8",
                            suffix=".txt",
                            delete=False,
                        ) as tmp_file:
                            tmp_file.write(inline_text)
                            temp_path = tmp_file.name

                        with open(temp_path, "rb") as file_handle:
                            uploaded_file = client.files.upload(
                                file=file_handle,
                                config={
                                    "mime_type": "text/plain",
                                    "display_name": f"{original_filename}.txt",
                                },
                            )
                        uploaded_files.append(uploaded_file)
                        continue
                    except Exception as inline_exc:
                        print(f"[WARN] Failed to upload inline source text for {original_filename}: {inline_exc}")
                    finally:
                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.unlink(temp_path)
                            except Exception:
                                pass
                print(f"[WARN] Failed to process document {entry.get('filename')} for Gemini: {e}")
                continue
            finally:
                if needs_cleanup:
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass

    finally:
        db.close()

    return uploaded_files


def _generate_with_gemini(
    client: genai.Client,
    system_prompt: str,
    user_prompt: Optional[str],
    files: List[types.File],
    chat_history: List[Dict[str, str]] = [],
    model: str = GEMINI_GENERATION_MODEL
) -> tuple[str, TokenUsage]:
    """Call Gemini API and return generated text."""
    
    print(f"[DEBUG] Calling Gemini API:")
    print(f"  Model: {model}")
    print(f"  Files: {len(files)}")
    print(f"  History: {len(chat_history)} messages")
    print(f"  User Prompt: '{user_prompt}'")

    # 1. Prepare effective history and current message
    effective_history = list(chat_history)
    current_msg_text = user_prompt
    
    print(f"[DEBUG] Effective history length: {len(effective_history)}")
    print(f"[DEBUG] Current message text: '{current_msg_text}'")

    # 2. Build Gemini History
    gemini_history = []
    if effective_history:
        # Handle first message with files
        first_entry = effective_history[0]
        first_parts = []
        if files:
            # Convert types.File to types.Part
            for f in files:
                first_parts.append(
                    types.Part(
                        file_data=types.FileData(
                            file_uri=f.uri,
                            mime_type=f.mime_type
                        )
                    )
                )
        
        text = first_entry.get("content", "")
        if text:
            first_parts.append(types.Part.from_text(text=text))
            
        gemini_history.append(types.Content(role="user", parts=first_parts))
        
        # Handle rest
        for entry in effective_history[1:]:
             role = "model" if entry.get("role") == "assistant" else "user"
             text = entry.get("content", "")
             if text:
                 gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    # 3. Create the chat session
    chat = client.chats.create(
        model=model,
        history=gemini_history,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True
            )
        )
    )

    # 4. Prepare the message to send
    message_parts = []
    
    # If history was empty, files must go here (in the new message)
    # If history was empty, files must go here (in the new message)
    if not effective_history and files:
        for f in files:
            message_parts.append(
                types.Part(
                    file_data=types.FileData(
                        file_uri=f.uri,
                        mime_type=f.mime_type
                    )
                )
            )
        
    if current_msg_text:
        message_parts.append(current_msg_text)

    try:
        response = chat.send_message(message_parts)
        
        usage_payload = TokenUsage(model=model)
        # Log usage if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            print(f"[DEBUG] Gemini Response - tokens: input={usage.prompt_token_count}, output={usage.candidates_token_count}")
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            usage_payload = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=round(_estimate_gemini_cost_usd(input_tokens, output_tokens), 4),
                model=model,
            )

        # Check for Thought Signature (Gemini 3.0)
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'thought_signature') and part.thought_signature:
                    print(f"[DEBUG] Found Thought Signature: {part.thought_signature[:50]}...")
                if hasattr(part, 'thought') and part.thought:
                    print(f"[DEBUG] Found Thought Process: {str(part.thought)[:50]}...")

        return response.text or "", usage_payload

    except Exception as e:
        print(f"[ERROR] Gemini generation failed: {e}")
        # Print more details if available
        if hasattr(e, 'details'):
            print(f"Details: {e.details}")
        raise








def verify_citations_with_llm(
    generated_text: str,
    selected_documents: Dict[str, List[Dict[str, Optional[str]]]],
    gemini_files: Optional[List[types.File]] = None,
) -> Dict[str, List[str]]:
    """
    Deprecated: generation now uses app.citation_qwen.run_citation_checks().

    Verify citations using Gemini 2.5 Flash with a strict, dynamic Pydantic model.
    Checks if cited documents are actually used and if page numbers are correct.
    """
    if not generated_text.strip():
        return {
            "cited": [],
            "missing": [],
            "pinpoint_missing": [],
            "warnings": ["Generierter Text ist leer."],
        }

    client = get_gemini_client()
    
    # 1. Prepare expected documents and citation hints
    expected_docs: Dict[str, str] = {} # filename -> citation_hint

    def _internal_note_reference_warnings() -> List[str]:
        warnings: List[str] = []
        if not generated_text.strip():
            return warnings
        note_entries = selected_documents.get("internal_notes", []) or []
        if note_entries and re.search(r"\b(Aktennotiz|Kanzleinotiz|Notiz|Gesprächsnotiz|Besprechungsnotiz|Transkript)\b", generated_text, re.IGNORECASE):
            warnings.append(
                "Der Entwurf erwähnt interne Kanzleinotizen/Notizen/Transkripte. "
                "Diese dürfen nicht als Quelle oder Fundstelle zitiert werden; Inhalte nur als Parteivortrag/Beweisangebot verwenden."
            )
        for entry in note_entries:
            filename = entry.get("filename") or entry.get("title") or ""
            if filename and filename in generated_text:
                warnings.append(
                    f"Interne Kanzleinotiz '{filename}' wird im Entwurf namentlich erwähnt. "
                    "Nicht zitieren; in Vortrag oder Beweisangebot umformulieren."
                )
        return warnings
    
    # Helper to sanitize filenames for Pydantic field names
    def _sanitize_field_name(name: str) -> str:
        # Replace non-alphanumeric chars with underscore, ensure start with letter
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
        if not sanitized[0].isalpha():
            sanitized = 'f_' + sanitized
        return sanitized

    # Collect documents with hints
    # Bescheid (Primary)
    for entry in selected_documents.get("bescheid", []):
        filename = entry.get("filename")
        if not filename: continue
        if entry.get("role") == "primary":
            expected_docs[filename] = "Anlage K2"
        else:
            expected_docs[filename] = "Dokument, S. X"
            
    # Anhörung
    for entry in selected_documents.get("anhoerung", []):
        filename = entry.get("filename")
        if filename:
            expected_docs[filename] = "Dokument, S. X"

    # Vorinstanz
    for entry in selected_documents.get("vorinstanz", []):
        filename = entry.get("filename")
        if filename:
            expected_docs[filename] = "Dokument, S. X"
            
    # Rechtsprechung
    for entry in selected_documents.get("rechtsprechung", []):
        filename = entry.get("filename")
        if filename:
            expected_docs[filename] = "Urteil / Beschluss"

    # Akte and Sonstiges
    for entry in selected_documents.get("akte", []):
        filename = entry.get("filename")
        if filename:
            expected_docs[filename] = "Dokument, S. X"

    for entry in selected_documents.get("sonstiges", []):
        filename = entry.get("filename")
        if filename:
            expected_docs[filename] = "Dokument, S. X"
            
    # Saved Sources
    for entry in selected_documents.get("saved_sources", []):
        title = entry.get("title") or entry.get("id")
        if title:
            expected_docs[title] = "Quelle / Titel"

    if not expected_docs:
        return {
            "cited": [],
            "missing": [],
            "pinpoint_missing": [],
            "warnings": ["Keine Dokumente zur Verifizierung ausgewählt."] + _internal_note_reference_warnings(),
        }

    # 2. Create Dynamic Pydantic Model
    field_definitions = {}
    filename_map = {} # field_name -> original_filename
    
    for filename, hint in expected_docs.items():
        field_name = _sanitize_field_name(filename)
        # Handle potential collisions
        counter = 1
        base_field_name = field_name
        while field_name in filename_map:
            field_name = f"{base_field_name}_{counter}"
            counter += 1
            
        filename_map[field_name] = filename
        
        used_description = f"Wurde das Dokument '{filename}' (z.B. zitiert als '{hint}') im Text verwendet/zitiert?"
        pinpoint_description = (
            f"Falls '{filename}' verwendet wird: enthält der Text dafür eine konkrete Fundstelle "
            f"(z.B. 'S. 1', 'Seite 1', 'Bescheid vom ..., S. 2') statt eines Platzhalters?"
        )
        field_definitions[f"{field_name}__used"] = (bool, Field(..., description=used_description))
        field_definitions[f"{field_name}__pinpoint"] = (bool, Field(..., description=pinpoint_description))

    # Add warnings field
    field_definitions["warnings"] = (List[str], Field(default_factory=list, description="Liste von Warnungen (z.B. falsche Seitenzahlen, halluzinierte Dokumente)"))
    
    VerificationModel = create_model("VerificationModel", **field_definitions)

    # 3. Construct Prompt
    prompt = f"""Du bist ein strenger juristischer Prüfer. Überprüfe die Zitate im folgenden Text.
    
    TEXT ZUR PRÜFUNG:
    {generated_text}
    
    AUFGABE:
    Prüfe für JEDES der folgenden Dokumente:
    1. ob es im Text zitiert oder inhaltlich verwendet wurde.
    2. ob dabei eine konkrete Fundstelle vorhanden ist.

    Als konkrete Fundstelle gelten insbesondere:
    - 'S. X'
    - 'Seite X'
    - 'Bescheid vom ..., S. X'
    - 'Forderungsaufstellung vom ..., S. X'

    NICHT als konkrete Fundstelle gelten:
    - 'Bl. ... der Akte'
    - 'vgl. Akte'
    - 'vgl. Unterlagen'
    - jede sonstige Platzhalterformulierung ohne echte Seitenangabe

    Sei streng:
    - Wenn ein Dokument nicht erwähnt wird, setze used=False und pinpoint=False.
    - Wenn ein Dokument erwähnt wird, aber ohne konkrete Fundstelle, setze used=True und pinpoint=False.
    - Wenn Platzhalterzitate wie 'Bl. ... der Akte' vorkommen, erwähne das zusätzlich ausdrücklich bei warnings.
    - Ignoriere Dokumente, die im Text erwähnt werden, aber NICHT in der Liste der erwarteten Dokumente stehen.
    """

    try:
        # If we have files, use them. If not, we can only check text-based references.
        parts = []
        if gemini_files:
            for f in gemini_files:
                # Check if f is already a Part or File object
                if hasattr(f, 'uri'):
                    parts.append(
                        types.Part(
                            file_data=types.FileData(
                                file_uri=f.uri,
                                mime_type=f.mime_type
                            )
                        )
                    )
                else:
                    # Fallback or error logging
                    print(f"[WARN] Unexpected file object type in verification: {type(f)}")

        parts.append(types.Part.from_text(text=prompt))

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=VerificationModel,
            ),
        )
        
        # 4. Parse Response
        import json
        if not response.text:
            print(f"[VERIFICATION ERROR] Empty response from Gemini. Candidates: {response.candidates}")
            return {
                "cited": [],
                "missing": [],
                "pinpoint_missing": [],
                "warnings": ["Verifizierung fehlgeschlagen: Keine Antwort vom Modell."],
            }
            
        result_dict = json.loads(response.text)
        
        cited = []
        missing = []
        pinpoint_missing = []
        warnings = result_dict.get("warnings", [])
        warnings.extend(_internal_note_reference_warnings())
        
        for field_name, original_filename in filename_map.items():
            is_cited = result_dict.get(f"{field_name}__used", False)
            has_pinpoint = result_dict.get(f"{field_name}__pinpoint", False)
            if is_cited:
                cited.append(original_filename)
                if not has_pinpoint:
                    pinpoint_missing.append(original_filename)
            else:
                missing.append(original_filename)
                
        return {
            "cited": cited,
            "missing": missing,
            "pinpoint_missing": pinpoint_missing,
            "warnings": warnings,
        }

    except Exception as e:
        print(f"[VERIFICATION ERROR] {e}")
        return {
            "cited": [],
            "missing": [],
            "pinpoint_missing": [],
            "warnings": [f"Verifizierung fehlgeschlagen: {e}"],
        }



def _is_jlawyer_configured() -> bool:
    return all([
        JLAWYER_BASE_URL,
        JLAWYER_USERNAME,
        JLAWYER_PASSWORD,
        JLAWYER_PLACEHOLDER_KEY,
    ])


def _jlawyer_api_base_url() -> str:
    base_url = (JLAWYER_BASE_URL or "").strip().rstrip("/")
    if not base_url:
        return ""
    if base_url.endswith("/rest"):
        return base_url
    return f"{base_url}/rest"


def _normalize_jlawyer_output_file_name(file_name: str) -> str:
    normalized = (file_name or "").strip()
    if normalized.lower().endswith(".odt"):
        normalized = normalized[:-4]
    return normalized


def _markdown_to_plain_text(text: str) -> str:
    """Conservatively remove Markdown markup for j-lawyer templates.

    Keep paragraph structure, bullets, numbering, and headings as visible text.
    Only strip formatting markers that regularly leak from model output.
    """
    if not text:
        return ""

    plain = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fenced code blocks: keep content, drop fences/language hints.
    plain = re.sub(r"```[^\n]*\n", "", plain)
    plain = plain.replace("```", "")

    # Links and images: keep human-readable label.
    plain = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", plain)
    plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)

    # Inline code and emphasis.
    plain = plain.replace("`", "")
    plain = re.sub(r"(\*\*|__)(.*?)\1", r"\2", plain)
    plain = re.sub(r"(\*|_)(.*?)\1", r"\2", plain)
    plain = re.sub(r"^>\s?", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain.strip()


@router.get("/jlawyer/templates", response_model=JLawyerTemplatesResponse)
@limiter.limit("20/hour")
async def get_jlawyer_templates(request: Request, folder: Optional[str] = None):
    if not _is_jlawyer_configured():
        raise HTTPException(status_code=503, detail="j-lawyer Integration ist nicht konfiguriert")

    folder_name = (folder or JLAWYER_TEMPLATE_FOLDER_DEFAULT or "").strip()
    if not folder_name:
        raise HTTPException(status_code=400, detail="Kein Template-Ordner konfiguriert")

    url = f"{_jlawyer_api_base_url()}/v6/templates/documents/{quote(folder_name, safe='')}"
    auth = (JLAWYER_USERNAME, JLAWYER_PASSWORD)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, auth=auth)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"j-lawyer Anfrage fehlgeschlagen: {exc}")

    if response.status_code >= 400:
        detail = response.text or response.reason_phrase or "Unbekannter Fehler"
        raise HTTPException(status_code=502, detail=f"j-lawyer Fehler ({response.status_code}): {detail}")

    templates: List[str] = []
    try:
        payload = response.json()
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    templates.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("template") or item.get("fileName")
                    if isinstance(name, str):
                        templates.append(name)
    except ValueError:
        text = response.text or ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                templates.append(line)

    return JLawyerTemplatesResponse(templates=templates, folder=folder_name)





def _summarize_selection_for_prompt(collected: Dict[str, List[Dict[str, Optional[str]]]]) -> str:
    """Create a short textual summary of the selected sources for the Claude prompt."""
    lines: List[str] = []
    primary_entry = None
    for entry in collected.get("bescheid", []):
        if entry.get("role") == "primary":
            primary_entry = entry
            break

    if primary_entry:
        lines.append(f"- Hauptbescheid (Anlage K2): {primary_entry.get('filename')}")
        primary_explanation = (primary_entry.get("explanation") or "").strip()
        if primary_explanation:
            lines.append(f"  Beschreibung: {primary_explanation}")

    def _append_section(header: str, entries: List[Dict[str, Optional[str]]], note: Optional[str] = None) -> None:
        if not entries:
            return
        lines.append(f"\n{header}")
        if note:
            lines.append(note)
        for entry in entries:
            label = entry.get("filename") or entry.get("title") or entry.get("id") or "Unbekanntes Dokument"
            explanation = (entry.get("explanation") or "").strip()
            if explanation:
                lines.append(f"- {label} — {explanation}")
            else:
                lines.append(f"- {label}")

    _append_section(
        "📋 Anhörungen:",
        collected.get("anhoerung", []),
        (
            "Mit konkreter Seite im jeweiligen Dokument zitieren. "
            "Nur dann als 'Bl. X d.A.' zitieren, wenn eine echte Blattzahl bekannt ist. "
            "Keine Platzhalterzitate wie 'Bl. ... der Akte' verwenden."
        ),
    )

    other_bescheide = [e for e in collected.get("bescheid", []) if e.get("role") != "primary"]
    _append_section(
        "📄 Weitere Bescheide / Aktenauszüge:",
        other_bescheide,
        (
            "Mit Dokumentname und konkreter Seite zitieren. "
            "Nur dann als 'Bl. X d.A.' zitieren, wenn eine echte Blattzahl bekannt ist. "
            "Keine Platzhalterzitate wie 'Bl. ... der Akte' verwenden."
        ),
    )

    _append_section(
        "⚖️ Rechtsprechung:",
        collected.get("rechtsprechung", []),
    )

    _append_section(
        "🏛️ Vorinstanz:",
        collected.get("vorinstanz", []),
    )

    _append_section(
        "📁 Akte / Beiakten:",
        collected.get("akte", []),
        (
            "Nur dann als 'Bl. X d.A.' zitieren, wenn eine echte Blattzahl bekannt ist. "
            "Sonst mit Dokumentname und konkreter Seite zitieren. "
            "Keine Platzhalterzitate wie 'Bl. ... der Akte' verwenden."
        ),
    )

    _append_section(
        "📂 Sonstiges (zitierfähige sonstige Dokumente):",
        collected.get("sonstiges", []),
        (
            "Nicht als Akte zitieren, sofern das Dokument nicht tatsächlich aus der Akte stammt. "
            "Bitte mit Dokumentname und konkreter Seite zitieren, z.B. 'Dokument ..., S. 1'. "
            "Keine Platzhalterzitate wie 'Bl. ... der Akte' verwenden."
        ),
    )

    _append_section(
        "📝 Interne Kanzleinotizen (NICHT zitieren):",
        collected.get("internal_notes", []),
        (
            "Nur als internes Tatsachen- und Strategieinput verwenden. "
            "Nicht als Quelle, Anlage, Aktennotiz, Notiz oder Fundstelle zitieren. "
            "Inhalte stattdessen in Parteivortrag, Beweisangebote oder anwaltliche Subsumtion übertragen."
        ),
    )

    saved_sources = collected.get("saved_sources", [])
    if saved_sources:
        lines.append("\n🔗 Gespeicherte Quellen:")
        for entry in saved_sources:
            title = entry.get("title") or entry.get("id") or "Unbekannte Quelle"
            url = entry.get("url") or "keine URL"
            description = (entry.get("description") or "").strip()
            base_line = f"- {title} ({url})"
            if description:
                lines.append(f"{base_line} — {description}")
            else:
                lines.append(base_line)

    return "\n".join(lines)


@router.post("/send-to-jlawyer", response_model=JLawyerResponse)
@limiter.limit("10/hour")
async def send_to_jlawyer(request: Request, body: JLawyerSendRequest):
    if not _is_jlawyer_configured():
        raise HTTPException(status_code=503, detail="j-lawyer Integration ist nicht konfiguriert")

    case_reference = body.case_id.strip()
    template_name = body.template_name.strip()
    file_name = body.file_name.strip()
    template_folder = (body.template_folder or JLAWYER_TEMPLATE_FOLDER_DEFAULT or "").strip()

    if not case_reference or not template_name or not file_name:
        raise HTTPException(status_code=400, detail="case_id, template_name und file_name sind Pflichtfelder")

    if not template_folder:
        raise HTTPException(status_code=400, detail="Kein Template-Ordner konfiguriert")

    file_name = _normalize_jlawyer_output_file_name(file_name)

    if not file_name:
        raise HTTPException(status_code=400, detail="Dateiname darf nicht leer sein")

    auth = (JLAWYER_USERNAME, JLAWYER_PASSWORD)
    case_id = await _resolve_jlawyer_case_id(case_reference, auth)
    placeholder_value = _markdown_to_plain_text(body.generated_text or "")

    url = (
        f"{_jlawyer_api_base_url()}/v6/templates/documents/"
        f"{quote(template_folder, safe='')}/"
        f"{quote(template_name, safe='')}/"
        f"{quote(case_id, safe='')}/"
        f"{quote(file_name, safe='')}"
    )

    payload = [
        {
            "placeHolderKey": JLAWYER_PLACEHOLDER_KEY,
            "placeHolderValue": placeholder_value,
        }
    ]

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(url, auth=auth, json=payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"j-lawyer Anfrage fehlgeschlagen: {exc}")

    if response.status_code >= 400:
        detail = response.text or response.reason_phrase or "Unbekannter Fehler"
        raise HTTPException(status_code=502, detail=f"j-lawyer Fehler ({response.status_code}): {detail}")
    response_payload: Optional[Dict[str, Any]] = None
    created_document_id: Optional[str] = None
    try:
        parsed_payload = response.json()
        if isinstance(parsed_payload, dict):
            response_payload = parsed_payload
            created_document_id = (
                str(
                    parsed_payload.get("id")
                    or parsed_payload.get("documentId")
                    or parsed_payload.get("docId")
                    or ""
                ).strip()
                or None
            )
    except ValueError:
        response_payload = None

    return JLawyerResponse(
        success=True,
        message="Vorlage erfolgreich an j-lawyer gesendet",
        requested_case_reference=case_reference,
        resolved_case_id=case_id,
        template_folder=template_folder,
        template_name=template_name,
        file_name=file_name,
        created_document_id=created_document_id,
        jlawyer_response=response_payload,
    )

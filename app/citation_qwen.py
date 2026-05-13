"""Qwen-based citation extraction and page-level verification."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from citation_verifier import _load_document_texts, verify_page_citations
from shared import ensure_anonymization_service_ready


CITATION_QWEN_VERIFICATION_ENABLED = (
    os.getenv("CITATION_QWEN_VERIFICATION_ENABLED", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)
CITATION_QWEN_MODEL = (
    os.getenv(
        "CITATION_QWEN_MODEL",
        os.getenv("OLLAMA_MODEL_QWEN", os.getenv("OLLAMA_MODEL", "qwen3.6:27b-q4_K_M")),
    ).strip()
    or "qwen3.6:27b-q4_K_M"
)
CITATION_QWEN_PAGE_CHAR_LIMIT = int(
    (os.getenv("CITATION_QWEN_PAGE_CHAR_LIMIT", "12000") or "12000").strip()
)
CITATION_QWEN_DRAFT_CHUNK_CHARS = int(
    (os.getenv("CITATION_QWEN_DRAFT_CHUNK_CHARS", "4500") or "4500").strip()
)
CITATION_QWEN_NUM_CTX = int(
    (os.getenv("CITATION_QWEN_NUM_CTX", os.getenv("OLLAMA_NUM_CTX_DEFAULT", "32768")) or "32768").strip()
)
CITATION_QWEN_TIMEOUT_SEC = float(
    (os.getenv("CITATION_QWEN_TIMEOUT_SEC", "300") or "300").strip()
)


def run_deterministic_citation_checks(
    generated_text: str,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> tuple[Dict[str, Any], List[str]]:
    """Run deterministic page checks as fallback/comparison metadata."""
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


def summarize_citation_checks(checks: List[Dict[str, Any]]) -> Dict[str, int]:
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


def parse_ollama_json_response(data: Dict[str, Any]) -> Dict[str, Any]:
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


async def call_qwen_json(
    service_url: str,
    prompt: str,
    *,
    num_predict: int = 700,
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "model": model or CITATION_QWEN_MODEL,
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
        return parse_ollama_json_response(response.json())


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def build_document_page_lookup(
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> Dict[str, Dict[int, str]]:
    return {
        document.label: document.pages
        for document in _load_document_texts(collected)
        if document.label and document.pages
    }


def citation_source_inventory(
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


def split_draft_for_citation_extraction(text: str) -> List[Dict[str, Any]]:
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


def safe_int_list(value: Any) -> List[int]:
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


def merge_page_numbers_with_citation_text(raw_pages: Any, citation_text: str) -> List[int]:
    explicit_pages = safe_int_list(raw_pages)
    text_pages = extract_page_numbers_from_citation_text(citation_text)
    text_lower = (citation_text or "").lower()
    if re.search(r"\b(?:f\.|ff\.)", text_lower) and text_pages:
        return text_pages
    return explicit_pages or text_pages


def extract_page_numbers_from_citation_text(citation_text: str) -> List[int]:
    text = citation_text or ""
    match = re.search(r"\b(?:S\.|Seite|page|p\.)\s*(\d+)\s*(f\.|ff\.)?", text, re.IGNORECASE)
    if not match:
        match = re.search(
            r"\bBl\.?\s*(\d+)\s*(ff\.|f\.)?\s*(?:d\.?\s*A\.?|der\s+Akte)?",
            text,
            re.IGNORECASE,
        )
    if not match:
        return []
    page = int(match.group(1))
    suffix = (match.group(2) or "").strip().lower()
    if suffix == "f.":
        return [page, page + 1]
    if suffix == "ff.":
        return [page, page + 1, page + 2]
    return [page]


def expand_pages_for_document(pages: List[int], citation_text: str, document: Dict[str, Any]) -> List[int]:
    if not pages:
        return []
    text_lower = (citation_text or "").lower()
    available_pages = [int(page) for page in (document.get("pages") or []) if int(page) > 0]
    if not available_pages:
        return pages
    start_page = min(pages)
    if re.search(r"\bff\.", text_lower):
        return [page for page in available_pages if page >= start_page]
    if re.search(r"\bf\.", text_lower):
        return [page for page in (start_page, start_page + 1) if page in available_pages]
    return [page for page in pages if page in available_pages] or pages


def _normalize_source_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_for_prompt_match(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def resolve_qwen_citation_document(
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
        pages = merge_page_numbers_with_citation_text(citation.get("page_numbers"), citation_text)
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
        page_available = [
            item for item in inventory
            if pages and all(page in (item.get("pages") or []) for page in pages)
        ]
        if len(page_available) == 1:
            return page_available[0]
        if len(page_available) > 1 and len(inventory) == 1:
            return sorted(
                page_available,
                key=lambda item: len(item.get("pages") or []),
                reverse=True,
            )[0]

    label_matches = [
        item for item in inventory
        if _normalize_for_prompt_match(item.get("label")) in hint
    ]
    if len(label_matches) == 1:
        return label_matches[0]

    category_keywords = {
        "bescheid": ("bescheid", "ablehnungsbescheid", "widerspruchsbescheid"),
        "anhoerung": ("anhoerung", "anhörung", "interview"),
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


async def extract_citations_from_chunk_with_qwen(
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
    parsed = await call_qwen_json(service_url, prompt, num_predict=2200, temperature=0.0)
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
        pages = merge_page_numbers_with_citation_text(raw.get("page_numbers"), citation_text)
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


async def check_citation_group_with_qwen(
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
    parsed = await call_qwen_json(service_url, prompt, num_predict=1400, temperature=0.0)
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


async def judge_citation_page_with_qwen(
    service_url: str,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """Single-page judge kept for benchmarks and debugging."""
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
    parsed = await call_qwen_json(service_url, prompt, num_predict=300, temperature=0.1)
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


def _status_from_qwen_verdict(verdict: str) -> str:
    if verdict == "yes":
        return "verified_on_cited_page"
    if verdict == "no":
        return "not_found"
    return "ambiguous"


async def run_qwen_extracted_citation_checks(
    generated_text: str,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> tuple[Optional[Dict[str, Any]], List[str]]:
    if not CITATION_QWEN_VERIFICATION_ENABLED:
        return None, []
    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        return None, ["Qwen-Fundstellenprüfung übersprungen: ANONYMIZATION_SERVICE_URL ist nicht konfiguriert."]

    inventory = citation_source_inventory(collected)
    if not inventory:
        return None, []

    try:
        await ensure_anonymization_service_ready()
    except Exception as exc:
        message = f"Qwen-Fundstellenprüfung übersprungen: service_manager nicht erreichbar ({exc})."
        print(f"[CITATION QWEN WARN] {message}")
        return None, [message]

    chunks = split_draft_for_citation_extraction(generated_text)
    extracted: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for chunk in chunks:
        try:
            extracted.extend(await extract_citations_from_chunk_with_qwen(service_url, chunk, inventory))
        except Exception as exc:
            message = f"Qwen-Fundstellenextraktion für Entwurfsabschnitt {chunk.get('index')} fehlgeschlagen: {exc}"
            print(f"[CITATION QWEN WARN] {message}")
            warnings.append(message)

    if not extracted:
        return {
            "checks": [],
            "warnings": warnings,
            "summary": summarize_citation_checks([]),
            "qwen_extraction": {
                "enabled": True,
                "model": CITATION_QWEN_MODEL,
                "chunks": len(chunks),
                "extracted": 0,
            },
        }, warnings

    page_lookup = build_document_page_lookup(collected)
    checks: List[Dict[str, Any]] = []
    groups: Dict[str, Dict[str, Any]] = {}
    by_id: Dict[str, Dict[str, Any]] = {}
    for citation in extracted:
        pages = merge_page_numbers_with_citation_text(
            citation.get("page_numbers"),
            citation.get("citation_text") or "",
        )
        document = resolve_qwen_citation_document(citation, inventory)
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
        pages = expand_pages_for_document(pages, citation.get("citation_text") or "", document)
        check["cited_pages"] = pages
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
            results = await check_citation_group_with_qwen(service_url, group)
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
        "summary": summarize_citation_checks(checks),
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


async def run_citation_checks(
    generated_text: str,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
) -> tuple[Dict[str, Any], List[str]]:
    deterministic, deterministic_warnings = run_deterministic_citation_checks(generated_text, collected)
    qwen_checks, qwen_warnings = await run_qwen_extracted_citation_checks(generated_text, collected)

    if qwen_checks is None:
        warnings = list(deterministic.get("warnings") or []) + qwen_warnings
        deterministic["provider"] = "deterministic"
        return deterministic, warnings

    qwen_checks["provider"] = "qwen3.6"
    qwen_checks["deterministic"] = {
        "summary": deterministic.get("summary") or {},
        "checks": deterministic.get("checks") or [],
        "warnings": deterministic.get("warnings") or [],
    }
    warnings = list(qwen_checks.get("warnings") or []) + qwen_warnings
    summary = qwen_checks.get("summary") or {}
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
            f"Qwen-Fundstellenprüfung: {problem_count} Fundstelle(n) konnten nicht eindeutig auf der zitierten Seite bestätigt werden."
        )
    if deterministic_warnings:
        qwen_checks.setdefault("deterministic_warnings", deterministic_warnings)
    return qwen_checks, warnings

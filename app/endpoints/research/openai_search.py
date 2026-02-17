"""
OpenAI ChatGPT web search research provider.
"""

import asyncio
import os
import re
from datetime import datetime
import traceback
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import markdown

from shared import ResearchResult, get_document_for_upload, get_openai_client
from .prompting import build_research_priority_prompt


OFFICIAL_SOURCE_HINTS = (
    "nrwe.justiz.nrw.de",
    "justiz.nrw.de",
    "justiz.de",
    "berlin.de",
    "justiz-berlin",
    "verwaltungsgericht",
    "bverwg",
    "bverfg",
    "eur-lex",
    "curia.europa",
    "juris.de",
    "dejure.org",
    "hudoc.echr",
    "ec.europa",
    "ec.eu",
)

OFF_TOPIC_SOURCE_HINTS = (
    "blog",
    "nachrichten",
    "news",
    "press",
    "presse",
    "kommentar",
    "comment",
    "anwaltskanzlei",
    "kanzlei",
    "forum",
    "youtube",
    "instagram",
    "facebook",
    "twitter",
    "x.com",
    "facebook.com",
    "linkedin",
)

DECISION_KEYWORDS = (
    "entscheid",
    "beschluss",
    "urteil",
    "aktenzeichen",
    "aktenzeichen:",
    "ecli",
    "rechtsgrundlage",
    "leitentscheidung",
    "revision",
)

HIGHEST_COURTS = (
    "bverwg",
    "bverfg",
    "bgh",
    "egmr",
    "eugh",
    "verfassungsgericht",
    "verwaltungsgerichtshof",
)

DATE_REGEX = re.compile(
    r"\b(?:(?:(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2})|(?:\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}))\b"
)

_RESEARCH_CONTEXT_TOKENS = (
    "ovg",
    "vg",
    "bverwg",
    "bverfg",
    "bgh",
    "egmr",
    "eugh",
    "bverf",
)


def _normalize_query_text(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip())


def _build_context_hints_prompt(context_hints: Optional[List[str]]) -> str:
    if not context_hints:
        return ""

    normalized = []
    for idx, hint in enumerate(context_hints[:12], start=1):
        normalized_hint = _normalize_query_text(hint)
        if not normalized_hint:
            continue
        normalized.append(f"{idx}. {normalized_hint}")

    if not normalized:
        return ""

    return (
        "Bekannte Referenzentscheidungen aus dem Fallkontext (Pflichtanker, zuerst prüfen):\n"
        + "\n".join(normalized)
        + "\nFormuliere mindestens zwei konkrete Suchanfragen, die diese Referenzen direkt berücksichtigen.\n"
    )


def _build_query_focus_prompt(
    query: str,
    research_context_hints: Optional[List[str]] = None,
) -> str:
    normalized_query = _normalize_query_text(query)
    if not normalized_query or normalized_query == "Keine zusätzliche Notiz.":
        base = (
            "Suchanfrage ist nicht explizit vorgegeben. Leite die Suchanfragen aus den "
            "angehängten Dokumenten ab."
        )
        context_block = _build_context_hints_prompt(research_context_hints)
        return f"{base}\n\n{context_block}" if context_block else base

    context_block = _build_context_hints_prompt(research_context_hints)
    return (
        "Suchanfrage (verbindlich):\n"
        f"- {normalized_query}\n"
        "Formuliere mindestens drei konkrete Web-Suchanfragen, die die eigentliche Fallkonstellation präzise abbilden "
        "(z. B. Herkunftsregion, Sicherheitslage, Verfahrensstatus, Wehr- oder Kriegsdienstkontext).\n"
        "Leite die Suchanfragen aus Dokumentlage und bekannten Referenzentscheidungen ab (klassische BAMF-Felder wie "
        "Dublin/Überstellung, § 60 Abs. 5/7, medizinische Abschiebungshindernisse, "
        "Art. 3 EMRK-Risikolagen, Vulnerabilität/Familienkonstellationen, Herkunftsstaatenlage).\n"
        "\n"
        f"{context_block}"
    )


def _context_hint_tokens(context_hints: Optional[List[str]]) -> List[str]:
    if not context_hints:
        return []

    terms: List[str] = []
    seen = set()
    for hint in context_hints:
        normalized = _normalize_query_text(hint).lower()
        if not normalized:
            continue
        if normalized not in seen:
            terms.append(normalized)
            seen.add(normalized)
        for token in re.findall(r"\b[0-9a-zäöüß./-]+\b", normalized):
            if token in seen:
                continue
            if len(token) >= 4 and (token in _RESEARCH_CONTEXT_TOKENS or token.count("/") >= 1):
                terms.append(token)
                seen.add(token)
            elif token.isdigit() and len(token) >= 4:
                seen.add(token)

        for token in _RESEARCH_CONTEXT_TOKENS:
            if token in normalized and token not in seen:
                terms.append(token)
                seen.add(token)

    return terms


def _obj_get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize_source_entry(entry: Any) -> Optional[Dict[str, str]]:
    url = _obj_get(entry, "url")
    if not url:
        return None

    title = _obj_get(entry, "title") or url
    description = (
        _obj_get(entry, "snippet")
        or _obj_get(entry, "description")
        or _obj_get(entry, "summary")
        or ""
    )

    source = {
        "url": str(url),
        "title": str(title),
        "description": str(description),
        "source": "ChatGPT Search",
    }

    lowered = source["url"].lower()
    if lowered.endswith(".pdf") or ".pdf?" in lowered:
        source["pdf_url"] = source["url"]

    return source


def _dedupe_sources(sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen_urls = set()
    for src in sources:
        url = src.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(src)
    return deduped


def _is_official_source(url: str, title: str = "", description: str = "") -> bool:
    blob = " ".join([url, title or "", description or ""]).lower()
    return any(token in blob for token in OFFICIAL_SOURCE_HINTS)


def _contains_offtopic_signal(url: str, title: str, description: str) -> bool:
    blob = " ".join([url, title, description]).lower()
    return any(token in blob for token in OFF_TOPIC_SOURCE_HINTS)


def _extract_decision_year(source: Dict[str, str]) -> int:
    candidate_text = " ".join(
        [
            source.get("url", ""),
            source.get("title", ""),
            source.get("description", ""),
        ]
    ).lower()
    date_years = []
    for match in DATE_REGEX.finditer(candidate_text):
        year_candidates = re.findall(r"(?:19|20)\d{2}", match.group(0))
        date_years.extend(int(year) for year in year_candidates)
    if date_years:
        return max(date_years)

    year_matches = [int(group) for group in re.findall(r"\b(?:19|20)\d{2}\b", candidate_text)]
    if year_matches:
        current_year = datetime.utcnow().year
        filtered = [year for year in year_matches if year <= current_year + 1]
        if filtered:
            return max(filtered)
        return max(year_matches)
    return 0


def _contains_decision_signal(url: str, title: str, description: str) -> int:
    blob = f"{url} {title} {description}".lower()
    score = 0
    for token in DECISION_KEYWORDS:
        if token in blob:
            score += 10

    if "/entscheid" in url or "entscheidung" in blob:
        score += 15

    return score


def _contains_highest_court_signal(blob: str) -> int:
    lowered = blob.lower()
    score = 0
    for token in HIGHEST_COURTS:
        if token in lowered:
            score += 12
    return min(score, 24)


def _sort_sources_by_quality(
    sources: List[Dict[str, str]],
    context_hints: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    context_terms = _context_hint_tokens(context_hints)

    def quality(src: Dict[str, str]) -> Dict[str, float]:
        url = (src.get("url") or "").lower()
        title = (src.get("title") or "").lower()
        description = (src.get("description") or "").lower()
        raw_score = 0
        blob = f"{url} {title} {description}"

        for token in context_terms:
            if token in blob:
                raw_score += 30

        if _contains_offtopic_signal(url, title, description):
            raw_score -= 40

        if _is_official_source(url, title, description):
            raw_score += 70

        raw_score += _contains_highest_court_signal(blob)
        raw_score += _contains_decision_signal(url, title, description)

        if ".pdf" in url:
            raw_score += 4

        if "aktenzeichen" in blob:
            raw_score += 8

        year = _extract_decision_year(
            {
                "url": url,
                "title": title,
                "description": description,
            }
        )
        current_year = datetime.utcnow().year
        recency_score = max(0, year - 2000) if year else 0
        if year and year >= current_year:
            recency_score += 20

        # Keep both components so we can rebalance court concentration later.
        return {
            "source": src,
            "raw_score": float(raw_score),
            "recency_score": float(recency_score),
            "url": url,
        }

    scored_sources = [quality(src) for src in _dedupe_sources(sources)]
    if not scored_sources:
        return []

    # Start from strongest raw relevance order before applying soft court diversification.
    scored_sources.sort(
        key=lambda item: (item["raw_score"], item["recency_score"], item["url"]),
        reverse=True,
    )

    has_alternative_courts = any(
        _court_bucket(
            source_data.get("url", ""),
            (source_data.get("source", {}).get("title", "") or ""),
            (source_data.get("source", {}).get("description", "") or ""),
        )
        not in ("other", "bverwg")
        for source_data in scored_sources
    )
    preferred_repeats = 1 if has_alternative_courts else 2

    court_counts: Dict[str, int] = {}

    diversified: List[Dict[str, float]] = []
    for entry in scored_sources:
        source = entry["source"]
        bucket = _court_bucket(
            (source.get("url") or "").lower(),
            (source.get("title") or "").lower(),
            (source.get("description") or "").lower(),
        )
        count = court_counts.get(bucket, 0)
        if bucket in ("bverwg", "bverfg") and count >= preferred_repeats:
            entry["raw_score"] -= 44 * (count - preferred_repeats + 1)
        elif bucket != "other" and count >= 2:
            entry["raw_score"] -= 28 * (count - 1)
        court_counts[bucket] = count + 1
        diversified.append(entry)

    diversified.sort(
        key=lambda item: (item["raw_score"], item["recency_score"], item["url"]),
        reverse=True,
    )
    return [entry["source"] for entry in diversified]


def _court_bucket(url: str, title: str, description: str) -> str:
    text = f"{url} {title} {description}".lower()
    if "nrwe.justiz.nrw.de" in text or "justiz.nrw.de" in text:
        return "nrw"
    for token in ("bverwg", "bverfg", "egmr", "eugh", "bgh", "ovg"):
        if token in text:
            return token
    if "vg berlin" in text or "verwg" in text or "verwaltungsgericht berlin" in text:
        return "vg_berlin"
    if "vg " in text or "verwaltungsgericht" in text:
        return "vg"
    return "other"


def _extract_text_from_response(response: Any) -> str:
    direct = _obj_get(response, "output_text")
    if direct:
        return str(direct)

    output = _obj_get(response, "output") or []
    texts: List[str] = []

    for item in output:
        if _obj_get(item, "type") != "message":
            continue
        content = _obj_get(item, "content") or []
        for block in content:
            if _obj_get(block, "type") == "output_text":
                text = _obj_get(block, "text")
                if text:
                    texts.append(str(text))

    return "\n".join(texts).strip()


def _extract_sources_from_response(response: Any, summary_text: str) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    output = _obj_get(response, "output") or []

    # Preferred source path from OpenAI docs: include=["web_search_call.action.sources"]
    for item in output:
        if _obj_get(item, "type") != "web_search_call":
            continue
        action = _obj_get(item, "action")
        action_sources = (
            _obj_get(action, "sources")
            or _obj_get(action, "search_results")
            or _obj_get(action, "results")
            or []
        )
        for src in action_sources:
            normalized = _normalize_source_entry(src)
            if normalized:
                sources.append(normalized)

    # Fallback: annotations in output text blocks
    for item in output:
        if _obj_get(item, "type") != "message":
            continue
        content = _obj_get(item, "content") or []
        for block in content:
            if _obj_get(block, "type") != "output_text":
                continue
            annotations = _obj_get(block, "annotations") or []
            for ann in annotations:
                normalized = _normalize_source_entry(ann)
                if normalized:
                    sources.append(normalized)

    # Last fallback: URL regex from generated text
    if not sources and summary_text:
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\])]+'
        seen = set()
        for url in re.findall(url_pattern, summary_text):
            if url in seen:
                continue
            seen.add(url)
            sources.append(
                {
                    "url": url,
                    "title": url,
                    "description": "",
                    "source": "ChatGPT Search",
                }
            )

    return _dedupe_sources(sources)


def _build_openai_focus_prompt(context_hints: Optional[List[str]] = None) -> str:
    context_block = _build_context_hints_prompt(context_hints)
    focus = (
        "Suchfokus (obligat):\n"
        "- Formuliere mindestens drei konkrete Suchanfragen auf Primärquellenebene (Gerichtsentscheidungen, Entscheidungen von Behörden).\n"
        "- Leite die Suchanfragen aus der konkreten Falllage (Dokumentkontext) ab und suche gezielt nach vergleichbaren Entscheidungen.\n"
        "- Bevorzuge offizielle Datenbanken und gerichtliche Veröffentlichungen mit Entscheidungstext, Datum und Aktenzeichen.\n"
        "- Nutze bei der Fragestellung kontextnahe Begriffe wie Staatsangehörigkeit/Region, "
        "Asyl-/Einordnung der Ablehnungsgründe, sowie klassische BAMF-Felder wie "
        "Dublin-Überstellung, § 60 Abs. 5/7, medizinische Gefahrenlagen, "
        "Art. 3 EMRK-Risikolagen, Vulnerabilität/Familien- und Kindesschutz.\n"
        "- Sortiere Ergebnisse intern nach Aktualität (neueste zuerst) und Relevanz; nenne vor allem Entscheidungen der letzten Jahre.\n"
        "- Vermeide Nachrichtenquellen, Kommentare, Blogartikel, Pressetexte oder Pressesiegel."
    )
    if context_block:
        focus = f"{focus}\n{context_block}"
    return focus


def _build_user_prompt(
    query: str,
    attachment_label: Optional[str],
    attachment_text: Optional[str],
    research_context_hints: Optional[List[str]] = None,
) -> str:
    trimmed_query = (query or "").strip() or "Keine zusätzliche Notiz."
    context_block = _build_context_hints_prompt(research_context_hints)
    context_anchor = f"Bekannte Kontextanker:\n{context_block}" if context_block else ""

    if attachment_text:
        snippet = attachment_text[:16000]
        truncated_note = ""
        if len(attachment_text) > len(snippet):
            truncated_note = "\n\n(Hinweis: Dokumenttext wurde für den Prompt gekürzt.)"

        return (
            f"""Rechercheauftrag:
{trimmed_query}

    Beigefügtes Dokument: {attachment_label or "Bescheid"}
Analysiere zuerst die rechtlichen Kernpunkte aus dem Dokument und recherchiere dann im Web gezielt dazu.
{context_anchor}

Dokumenttext:
---
{snippet}
---{truncated_note}
            """
            + "\n\n"
            + build_research_priority_prompt(
                "Nutze das beigefügte Dokument als Ausgangspunkt für entscheidungsbezogene Suchanfragen."
            )
            + "\n\n"
            + _build_openai_focus_prompt(research_context_hints)
        )

    return (
        f"""Rechercheauftrag:
{trimmed_query}

{context_anchor}
Führe eine tiefgehende Web-Recherche zu deutschem Migrationsrecht durch."""
        + "\n\n"
        + build_research_priority_prompt(
            "Konzentriere dich auf aktuelle und nachprüfbare Rechtsgrundlagen für die Fragestellung."
        )
        + "\n\n"
        + _build_openai_focus_prompt(research_context_hints)
    )


def _load_attachment_text(
    attachment_path: Optional[str],
    attachment_display_name: Optional[str],
    attachment_ocr_text: Optional[str],
    attachment_text_path: Optional[str],
    attachment_anonymization_metadata: Optional[dict] = None,
    attachment_is_anonymized: bool = False,
) -> Optional[str]:
    if attachment_ocr_text:
        return attachment_ocr_text

    if not attachment_path and not attachment_text_path:
        return None

    upload_entry = {
        "filename": attachment_display_name or "Bescheid",
        "file_path": attachment_path,
        "extracted_text_path": attachment_text_path,
        "extracted_text": attachment_ocr_text,
        "ocr_applied": bool(attachment_ocr_text),
        "anonymization_metadata": attachment_anonymization_metadata,
        "is_anonymized": attachment_is_anonymized,
    }
    selected_path, mime_type, _ = get_document_for_upload(upload_entry)

    if mime_type == "text/plain":
        with open(selected_path, "r", encoding="utf-8") as f:
            return f.read()

    pdf_doc = fitz.open(selected_path)
    try:
        parts = [page.get_text() for page in pdf_doc]
    finally:
        pdf_doc.close()
    return "\n".join(parts)


def _build_multi_document_prompt(
    query: str,
    attachment_sections: List[Dict[str, str]],
    research_context_hints: Optional[List[str]] = None,
) -> str:
    trimmed_query = (query or "").strip() or "Keine zusätzliche Notiz."
    context_block = _build_context_hints_prompt(research_context_hints)
    context_anchor = f"Bekannte Kontextanker:\n{context_block}" if context_block else ""

    if not attachment_sections:
        return _build_user_prompt(
            trimmed_query,
            None,
            None,
            research_context_hints=research_context_hints,
        )

    max_total_chars = 42000
    max_per_doc = 10000
    used_chars = 0
    section_blocks: List[str] = []

    for idx, sec in enumerate(attachment_sections, start=1):
        if used_chars >= max_total_chars:
            break

        label = (sec.get("label") or f"Dokument {idx}").strip()
        text = sec.get("text") or ""
        if not text:
            continue

        remaining = max_total_chars - used_chars
        take = min(max_per_doc, remaining)
        snippet = text[:take]
        used_chars += len(snippet)

        truncated_note = ""
        if len(text) > len(snippet):
            truncated_note = "\n(Hinweis: Dokumentauszug gekürzt.)"

        section_blocks.append(
            f"Dokument {idx}: {label}\n---\n{snippet}\n---{truncated_note}"
        )

    docs_blob = "\n\n".join(section_blocks) if section_blocks else "(Keine Dokumenttexte verfügbar)"
    included_docs = len(section_blocks)

    return (
        f"""Rechercheauftrag:
{trimmed_query}

Du erhältst mehrere Dokumente aus dem Fallkontext. Analysiere sie gemeinsam und leite daraus eine konsistente Recherche ab.

Dokumente (eingebettet): {included_docs}
{docs_blob}
{context_anchor}

"""
        + "\n\n"
        + build_research_priority_prompt(
            "Leite aus allen Dokumenten entscheidungsbezogene Suchanfragen ab und mappe die Resultate auf die jeweilige Dokumentlage."
        )
        + "\n\n"
        + _build_openai_focus_prompt(research_context_hints)
    )


async def _call_responses_with_search(
    client: Any,
    model: str,
    input_messages: List[Dict[str, Any]],
    *,
    fast_mode: bool = False,
) -> Any:
    include_paths = ["web_search_call.action.sources"]
    full_web_search_tool = {
        "type": "web_search",
        "user_location": {"type": "approximate", "country": "DE"},
        "search_context_size": "low",
    }
    attempts = [{"tools": [full_web_search_tool], "include": include_paths}]

    if not fast_mode:
        simple_web_search_tool = {"type": "web_search"}
        legacy_web_search_tool = {"type": "web_search_preview"}
        attempts.extend(
            [
                {"tools": [simple_web_search_tool], "include": include_paths},
                {"tools": [simple_web_search_tool], "include": None},
                {"tools": [legacy_web_search_tool], "include": include_paths},
                {"tools": [legacy_web_search_tool], "include": None},
            ]
        )

    last_exc: Optional[Exception] = None
    for idx, attempt in enumerate(attempts):
        try:
            payload: Dict[str, Any] = {
                "model": model,
                "input": input_messages,
                "tools": attempt["tools"],
                "reasoning": {"effort": "low"},
                "text": {"verbosity": "low"},
                "max_output_tokens": 2200,
            }
            include = attempt.get("include")
            if include:
                payload["include"] = include

            return await asyncio.to_thread(client.responses.create, **payload)
        except Exception as exc:
            last_exc = exc
            print(f"[CHATGPT-SEARCH] web_search attempt {idx + 1}/{len(attempts)} failed: {exc}")
            continue

    raise last_exc if last_exc else RuntimeError("Unknown web_search failure")


async def research_with_openai_search(
    query: str,
    attachment_path: Optional[str] = None,
    attachment_display_name: Optional[str] = None,
    attachment_ocr_text: Optional[str] = None,
    attachment_text_path: Optional[str] = None,
    attachment_anonymization_metadata: Optional[dict] = None,
    attachment_is_anonymized: bool = False,
    attachment_documents: Optional[List[Dict[str, Optional[str]]]] = None,
    research_context_hints: Optional[List[str]] = None,
) -> ResearchResult:
    """
    Perform web research using OpenAI Responses API + web search tool.
    """
    try:
        client = get_openai_client()
        model = os.getenv("OPENAI_RESEARCH_MODEL", "gpt-5.2").strip() or "gpt-5.2"
        trimmed_query = _normalize_query_text(query) or "Keine zusätzliche Notiz."
        fast_mode = os.getenv("OPENAI_RESEARCH_FAST_MODE", "1").strip().lower() in ("1", "true", "yes", "on")
        query_focus_prompt = _build_query_focus_prompt(
            trimmed_query,
            research_context_hints=research_context_hints,
        )

        attachment_sections: List[Dict[str, str]] = []

        if attachment_documents:
            print(f"[CHATGPT-SEARCH] Loading multi-document context ({len(attachment_documents)} docs)")
            try:
                max_doc_count = int(os.getenv("OPENAI_RESEARCH_DOC_LIMIT", "6"))
            except ValueError:
                max_doc_count = 6
            for doc in attachment_documents[:max_doc_count]:
                doc_label = (
                    doc.get("attachment_display_name")
                    or doc.get("filename")
                    or "Dokument"
                )
                try:
                    doc_text = _load_attachment_text(
                        attachment_path=doc.get("attachment_path"),
                        attachment_display_name=doc_label,
                        attachment_ocr_text=doc.get("attachment_ocr_text"),
                        attachment_text_path=doc.get("attachment_text_path"),
                        attachment_anonymization_metadata=doc.get("anonymization_metadata"),
                        attachment_is_anonymized=bool(doc.get("is_anonymized")),
                    )
                    if doc_text:
                        attachment_sections.append({"label": str(doc_label), "text": doc_text})
                except Exception as exc:
                    print(f"[CHATGPT-SEARCH] Failed to load document '{doc_label}': {exc}")

        if not attachment_sections:
            try:
                attachment_text = _load_attachment_text(
                    attachment_path=attachment_path,
                    attachment_display_name=attachment_display_name,
                    attachment_ocr_text=attachment_ocr_text,
                    attachment_text_path=attachment_text_path,
                    attachment_anonymization_metadata=attachment_anonymization_metadata,
                    attachment_is_anonymized=attachment_is_anonymized,
                )
                if attachment_text:
                    attachment_sections.append(
                        {
                            "label": attachment_display_name or "Bescheid",
                            "text": attachment_text,
                        }
                    )
            except Exception as exc:
                print(f"[CHATGPT-SEARCH] Attachment text extraction failed: {exc}")

        if attachment_sections:
            total_chars = sum(len(x.get("text", "")) for x in attachment_sections)
            print(
                f"[CHATGPT-SEARCH] Loaded attachment context from {len(attachment_sections)} docs "
                f"({total_chars} chars)"
            )

        system_prompt = (
            build_research_priority_prompt(
                "Du bist ein Rechercheassistent für deutsches Migrationsrecht und nutzt Websuche aktiv."
            )
        )
        user_prompt = (
            _build_multi_document_prompt(
                trimmed_query,
                attachment_sections,
                research_context_hints=research_context_hints,
            )
            + "\n\n"
            + query_focus_prompt
        )

        input_messages = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ]

        print(f"[CHATGPT-SEARCH] Calling OpenAI Responses API (model={model})")
        if fast_mode:
            print("[CHATGPT-SEARCH] Fast mode enabled: single-pass strategy.")
        response = await _call_responses_with_search(
            client,
            model,
            input_messages,
            fast_mode=fast_mode,
        )
        print("[CHATGPT-SEARCH] Responses API call completed (primary)")

        summary_markdown = (_extract_text_from_response(response) or "").strip()
        if summary_markdown:
            summary_markdown = "\n".join(
                line.rstrip() for line in summary_markdown.replace("\r\n", "\n").split("\n")
            )
        else:
            summary_markdown = (
                "**Web-Recherche**\n\n"
                "Die Ergebniszusammenfassung konnte nicht direkt übernommen werden. "
                "Bitte prüfen Sie die Quellenliste."
            )
        if not summary_markdown.lower().startswith("**web-recherche**"):
            summary_markdown = f"**Web-Recherche**\n\n{summary_markdown}"

        sources = _extract_sources_from_response(response, summary_markdown)

        sources = _sort_sources_by_quality(
            sources,
            context_hints=research_context_hints,
        )
        sources = sources[:40]

        if sources:
            official_sources = [
                source
                for source in sources
                if _is_official_source(
                    source.get("url", ""),
                    source.get("title", ""),
                    source.get("description", ""),
                )
            ]
            if official_sources:
                if len(official_sources) < len(sources):
                    fallback_sources = [
                        source
                        for source in sources
                        if source not in official_sources
                    ]
                    sources = official_sources + fallback_sources
            # PDF detection adds noticeable latency; disable for speed in the search path.
            # If needed, enable again per source with a stricter allowlist later.

        summary_html = markdown.markdown(
            summary_markdown,
            extensions=["extra", "sane_lists"],
            output_format="html",
        )

        return ResearchResult(
            query=query,
            summary=summary_html,
            sources=sources,
            suggestions=[],
        )
    except Exception as e:
        print(f"[ERROR] in research_with_openai_search: {e}")
        print(traceback.format_exc())
        raise Exception(f"ChatGPT search failed: {e}")

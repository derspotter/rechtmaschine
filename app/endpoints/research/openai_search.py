"""
OpenAI ChatGPT web search research provider.
"""

import asyncio
import os
import re
import traceback
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import markdown

from shared import ResearchResult, get_document_for_upload, get_openai_client
from .prompting import build_research_priority_prompt
from .utils import enrich_web_sources_with_pdf


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
        "source": "ChatGPT",
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
                    "source": "ChatGPT",
                }
            )

    return _dedupe_sources(sources)


def _load_attachment_text(
    attachment_path: Optional[str],
    attachment_display_name: Optional[str],
    attachment_ocr_text: Optional[str],
    attachment_text_path: Optional[str],
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


def _build_user_prompt(query: str, attachment_label: Optional[str], attachment_text: Optional[str]) -> str:
    trimmed_query = (query or "").strip() or "Keine zusätzliche Notiz."

    if attachment_text:
        snippet = attachment_text[:25000]
        truncated_note = ""
        if len(attachment_text) > len(snippet):
            truncated_note = "\n\n(Hinweis: Dokumenttext wurde für den Prompt gekürzt.)"

        return (
            f"""Rechercheauftrag:
{trimmed_query}

Beigefügtes Dokument: {attachment_label or "Bescheid"}
Analysiere zuerst die rechtlichen Kernpunkte aus dem Dokument und recherchiere dann im Web gezielt dazu.

Dokumenttext:
---
{snippet}
---{truncated_note}
"""
            + "\n\n"
            + build_research_priority_prompt(
                "Nutze das beigefügte Dokument als Ausgangspunkt für gerichtsfokussierte Suchanfragen."
            )
        )

    return (
        f"""Rechercheauftrag:
{trimmed_query}

Führe eine tiefgehende Web-Recherche zu deutschem Migrationsrecht durch."""
        + "\n\n"
        + build_research_priority_prompt(
            "Konzentriere dich auf aktuelle und nachprüfbare Rechtsgrundlagen für die Fragestellung."
        )
    )


def _build_multi_document_prompt(
    query: str,
    attachment_sections: List[Dict[str, str]],
) -> str:
    trimmed_query = (query or "").strip() or "Keine zusätzliche Notiz."

    if not attachment_sections:
        return _build_user_prompt(trimmed_query, None, None)

    max_total_chars = 70000
    max_per_doc = 14000
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

"""
        + "\n\n"
        + build_research_priority_prompt(
            "Leite aus allen Dokumenten gerichtsnahe Suchanfragen ab und mappe die Resultate auf die jeweilige Dokumentlage."
        )
    )


async def _call_responses_with_search(client: Any, model: str, input_messages: List[Dict[str, Any]]) -> Any:
    include_paths = ["web_search_call.action.sources"]
    full_web_search_tool = {
        "type": "web_search",
        "user_location": {"type": "approximate", "country": "DE"},
        "search_context_size": "high",
    }
    simple_web_search_tool = {"type": "web_search"}
    legacy_web_search_tool = {"type": "web_search_preview"}

    attempts = [
        {"tools": [full_web_search_tool], "include": include_paths},
        {"tools": [simple_web_search_tool], "include": include_paths},
        {"tools": [simple_web_search_tool], "include": None},
        {"tools": [legacy_web_search_tool], "include": include_paths},
        {"tools": [legacy_web_search_tool], "include": None},
    ]

    last_exc: Optional[Exception] = None
    for idx, attempt in enumerate(attempts):
        try:
            payload: Dict[str, Any] = {
                "model": model,
                "input": input_messages,
                "tools": attempt["tools"],
                "reasoning": {"effort": "medium"},
                "text": {"verbosity": "high"},
                "max_output_tokens": 6000,
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
    attachment_documents: Optional[List[Dict[str, Optional[str]]]] = None,
) -> ResearchResult:
    """
    Perform web research using OpenAI Responses API + web search tool.
    """
    try:
        client = get_openai_client()
        model = os.getenv("OPENAI_RESEARCH_MODEL", "gpt-5.2").strip() or "gpt-5.2"

        attachment_sections: List[Dict[str, str]] = []

        if attachment_documents:
            print(f"[CHATGPT-SEARCH] Loading multi-document context ({len(attachment_documents)} docs)")
            for doc in attachment_documents:
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
        user_prompt = _build_multi_document_prompt(query, attachment_sections)

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
        response = await _call_responses_with_search(client, model, input_messages)
        print("[CHATGPT-SEARCH] Responses API call completed")

        summary_markdown = (_extract_text_from_response(response) or "").strip()
        if summary_markdown:
            summary_markdown = "\n".join(
                line.rstrip() for line in summary_markdown.replace("\r\n", "\n").split("\n")
            )
        else:
            summary_markdown = "**Web-Recherche**\n\nKeine Rechercheergebnisse gefunden."
        if not summary_markdown.lower().startswith("**web-recherche**"):
            summary_markdown = f"**Web-Recherche**\n\n{summary_markdown}"

        sources = _extract_sources_from_response(response, summary_markdown)
        if sources:
            await enrich_web_sources_with_pdf(sources, max_checks=10, concurrency=3)

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

"""
Grok-4.1-Fast research with web_search tool and structured outputs.
"""

import json
import os
import re
import traceback
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import markdown
from pydantic import BaseModel, Field
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user
from xai_sdk.tools import web_search

from shared import ResearchResult, get_document_for_upload
from .prompting import build_research_priority_prompt
from .source_quality import normalize_and_rank_sources
from .utils import _enrich_sources_with_pdf_detection


class StructuredSource(BaseModel):
    """Single source with title, URL, and description"""
    title: str = Field(description="Title of the source (e.g., court name and decision)")
    url: str = Field(description="Full URL to the source")
    description: str = Field(description="Brief description or summary of the source's relevance")


class GrokResearchOutput(BaseModel):
    """Structured output for Grok research results"""
    summary: str = Field(description="Detailed summary of research findings in German")
    sources: List[StructuredSource] = Field(description="List of relevant sources found during research")


GROK_DECISION_HINTS = (
    "entscheid",
    "beschluss",
    "urteil",
    "aktenzeichen",
    "ecli",
    "bverwg",
    "bverfg",
    "egmr",
    "eugh",
    "ovg",
)

GROK_OFFTOPIC_HINTS = (
    "blog",
    "nachrichten",
    "news",
    "press",
    "presse",
    "kommentar",
    "comment",
    "kanzlei",
    "anwaltskanzlei",
    "forum",
    "youtube",
    "instagram",
    "facebook",
    "twitter",
    "x.com",
    "linkedin",
)

SEARCH_MODE_CONFIG = {
    "fast": {"max_rounds": 2, "max_duration_sec": 20},
    "balanced": {"max_rounds": 3, "max_duration_sec": 35},
    "deep": {"max_rounds": 4, "max_duration_sec": 55},
}


def _get_mode_config(search_mode: str) -> dict:
    return SEARCH_MODE_CONFIG.get(search_mode, SEARCH_MODE_CONFIG["balanced"])


def _score_grok_source(source: dict) -> int:
    url = (source.get("url") or "").lower()
    title = (source.get("title") or "").lower()
    description = (source.get("description") or "").lower()
    text = f"{url} {title} {description}"

    score = 0
    for token in GROK_DECISION_HINTS:
        if token in text:
            score += 15

    if ".pdf" in url:
        score += 6

    if any(token in text for token in GROK_OFFTOPIC_HINTS):
        score -= 50

    return score


def _court_bucket(url: str, title: str, description: str) -> str:
    text = f"{url} {title} {description}".lower()
    if "nrwe.justiz.nrw.de" in text or "justiz.nrw.de" in text:
        return "nrw"
    if "bverwg" in text:
        return "bverwg"
    if "bverfg" in text:
        return "bverfg"
    if "egmr" in text:
        return "egmr"
    if "eugh" in text:
        return "eugh"
    if "bgh" in text:
        return "bgh"
    if "ovg" in text:
        return "ovg"
    if "vg berlin" in text or "verwaltungsgericht berlin" in text:
        return "vg_berlin"
    if "vg" in text or "verwaltungsgericht" in text:
        return "vg"
    return "other"


def _apply_court_diversity_penalty(
    sources_with_scores: List[tuple[dict, int]],
) -> List[tuple[dict, int]]:
    if not sources_with_scores:
        return []

    has_alternative_courts = any(
        _court_bucket(
            (source.get("url") or "").lower(),
            (source.get("title") or "").lower(),
            (source.get("description") or "").lower(),
        )
        not in ("other", "bverwg")
        for source, _ in sources_with_scores
    )
    preferred_repeats = 1 if has_alternative_courts else 2

    adjusted: List[tuple[dict, int]] = []
    court_counts: dict[str, int] = {}

    for source, score in sources_with_scores:
        bucket = _court_bucket(
            (source.get("url") or "").lower(),
            (source.get("title") or "").lower(),
            (source.get("description") or "").lower(),
        )
        count = court_counts.get(bucket, 0)
        adjusted_score = score
        if bucket in ("bverwg", "bverfg") and count >= preferred_repeats:
            adjusted_score -= 44 * (count - preferred_repeats + 1)
        elif bucket != "other" and count >= 2:
            adjusted_score -= 28 * (count - 1)

        court_counts[bucket] = count + 1
        adjusted.append((source, adjusted_score))

    return adjusted


def _load_attachment_text(
    path: Optional[str],
    display_name: Optional[str],
    text_path: Optional[str],
    ocr_text: Optional[str],
    anonymization_metadata: Optional[dict],
    is_anonymized: bool,
) -> Optional[str]:
    if ocr_text:
        return ocr_text

    if not path and not text_path:
        return None

    upload_entry = {
        "filename": display_name or "Bescheid",
        "file_path": path,
        "extracted_text_path": text_path,
        "anonymization_metadata": anonymization_metadata,
        "is_anonymized": is_anonymized,
    }
    selected_path, mime_type, _ = get_document_for_upload(upload_entry)

    if mime_type == "text/plain":
        with open(selected_path, "r", encoding="utf-8") as f:
            return f.read()

    pdf_doc = fitz.open(selected_path)
    try:
        return "\n".join(page.get_text() for page in pdf_doc)
    finally:
        pdf_doc.close()


def _build_grok_attachment_sections(
    attachment_documents: Optional[List[Dict[str, Optional[str]]]] = None,
    attachment_path: Optional[str] = None,
    attachment_display_name: Optional[str] = None,
    attachment_text_path: Optional[str] = None,
    attachment_ocr_text: Optional[str] = None,
    attachment_anonymization_metadata: Optional[dict] = None,
    attachment_is_anonymized: bool = False,
    max_docs: int = 6,
) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []

    if attachment_documents:
        for idx, doc in enumerate(attachment_documents[:max_docs], start=1):
            label = (
                doc.get("attachment_display_name")
                or doc.get("filename")
                or f"Dokument {idx}"
            )
            try:
                content = _load_attachment_text(
                    path=doc.get("attachment_path"),
                    display_name=label,
                    text_path=doc.get("attachment_text_path"),
                    ocr_text=doc.get("attachment_ocr_text"),
                    anonymization_metadata=doc.get("anonymization_metadata"),
                    is_anonymized=bool(
                        doc.get("attachment_is_anonymized", doc.get("is_anonymized", False))
                    ),
                )
            except Exception as exc:
                print(f"[GROK] Failed to load attached document '{label}': {exc}")
                continue

            if content:
                sections.append({"label": str(label), "text": content})

        if sections:
            return sections

    try:
        content = _load_attachment_text(
            path=attachment_path,
            display_name=attachment_display_name,
            text_path=attachment_text_path,
            ocr_text=attachment_ocr_text,
            anonymization_metadata=attachment_anonymization_metadata,
            is_anonymized=attachment_is_anonymized,
        )
    except Exception as exc:
        print(f"[GROK] Failed to load fallback attachment context: {exc}")
        return sections

    if content:
        sections.append(
            {
                "label": attachment_display_name or "Dokument",
                "text": content,
            }
        )

    return sections


async def research_with_grok(
    query: str,
    attachment_path: Optional[str] = None,
    attachment_display_name: Optional[str] = None,
    attachment_ocr_text: Optional[str] = None,
    attachment_text_path: Optional[str] = None,
    attachment_anonymization_metadata: Optional[dict] = None,
    attachment_is_anonymized: bool = False,
    attachment_documents: Optional[List[Dict[str, Optional[str]]]] = None,
    research_context_hints: Optional[List[str]] = None,
    search_mode: str = "balanced",
    max_sources: int = 12,
    domain_policy: str = "legal_balanced",
    jurisdiction_focus: str = "de_eu",
    recency_years: int = 6,
) -> ResearchResult:
    """
    Perform web research using Grok-4-Fast with agentic tool calling.
    Uses the new Responses API with web_search tool and structured outputs.
    Returns relevant links and sources for the user's query.

    Prefers OCR text when available for better accuracy.
    """

    try:

        # Build system prompt for legal research
        system_prompt = build_research_priority_prompt(
            "Du bist ein Rechercheassistent für deutsches Migrationsrecht und nutzt das web_search Tool aktiv."
        )

        # Build user message - include document text if available
        context_anchor = ""
        if research_context_hints:
            ordered_hints = []
            seen = set()
            for idx, hint in enumerate(research_context_hints[:8], start=1):
                normalized = " ".join(str(hint).split())
                if not normalized or normalized in seen:
                    continue
                ordered_hints.append(f"{idx}. {normalized}")
                seen.add(normalized)

            if ordered_hints:
                context_anchor = (
                    "Pflichtanker aus dem Fallkontext:\n"
                    + "\n".join(ordered_hints)
                    + "\nNutze diese Referenzen zur Plausibilisierung, aber formuliere Suchanfragen auf Basis der Fallstruktur."
                )

        attachment_sections = _build_grok_attachment_sections(
            attachment_documents=attachment_documents,
            attachment_path=attachment_path,
            attachment_display_name=attachment_display_name,
            attachment_text_path=attachment_text_path,
            attachment_ocr_text=attachment_ocr_text,
            attachment_anonymization_metadata=attachment_anonymization_metadata,
            attachment_is_anonymized=attachment_is_anonymized,
        )

        docs_anchor = ""
        if attachment_sections:
            blocks: List[str] = []
            max_total_chars = 36000
            used_chars = 0
            for idx, section in enumerate(attachment_sections, start=1):
                if used_chars >= max_total_chars:
                    break

                label = section.get("label") or f"Dokument {idx}"
                text = section.get("text") or ""
                if not text:
                    continue

                remaining = max_total_chars - used_chars
                snippet = text[: min(10000, remaining)]
                used_chars += len(snippet)
                blocks.append(
                    f"Dokument {idx}: {label}\n---\n{snippet}\n---"
                    + ("\n(Hinweis: Dokumentauszug gekürzt.)" if len(text) > len(snippet) else "")
                )

            if blocks:
                docs_anchor = "Dokumentenbezug (zu analysierende Falllage):\n" + "\n\n".join(blocks)

        if query is None:
            query = ""

        if not query and not docs_anchor:
            query = "Relevante Rechtsprechungsentscheidungen für den vorliegenden Fall ermitteln."

        if docs_anchor:
            user_message = f"""Rechercheauftrag: {query}

{docs_anchor}

AUFGABE:
1. ANALYSIERE die bereitgestellten Dokumente und identifiziere zentrale Rechtsfragen sowie Ablehnungsgründe.
2. NUTZE DAS WEB_SEARCH TOOL für konkrete Recherchen nach vergleichbaren Entscheidungen.
{context_anchor if context_anchor else ''}

Zusätzliche Hinweise:
- Betrachte vorrangig Entscheidungen deutscher Gerichte und behalte ggf. höhere Rechtsprechung im Blick.
- Relevante Entwicklungen aus NRW separat besonders hervorheben.
- Leite Suchbegriffe ausschließlich aus den im Dokumentenbestand festgestellten Tatsachen und Streitpunkten ab.
""" + "\n\n" + build_research_priority_prompt(
                "Vergleiche die Treffer strukturiert mit der konkreten Fallkonstellation."
            )
        else:
            user_message = (
                f"""Rechercheauftrag: {query}

WICHTIG: Nutze das web_search Tool, um aktuelle und prüfbare Quellen zu recherchieren."""
                + (f"\n\n{context_anchor}" if context_anchor else "")
                + "\n\n"
                + build_research_priority_prompt(
                    "Starte mit einer Fallanalyse und leite daraus konkrete Recherchestrategien für vergleichbare Primärentscheidungen ab."
                )
            )

        # Use xAI SDK with web_search tool and structured output
        print("[GROK] Calling xAI SDK with web_search tool and structured output")
        xai_api_key = os.getenv("XAI_API_KEY")
        if not xai_api_key:
            raise ValueError("XAI_API_KEY environment variable is not set")

        xai_client = XAIClient(api_key=xai_api_key)
        model_name = "grok-4-1-fast"
        mode_config = _get_mode_config(search_mode)
        started_at = perf_counter()

        full_user_message = (
            f"{system_prompt}\n\n{user_message}\n\n"
            "Rechercheparameter:\n"
            f"- Suchmodus: {search_mode}\n"
            f"- Jurisdiktionsfokus: {jurisdiction_focus}\n"
            f"- Ziel-Aktualität: letzte {recency_years} Jahre\n"
            f"- Domain-Policy: {domain_policy}\n"
        )

        json_instruction = (
            "\n\nWICHTIG: Gib deine Antwort als JSON im folgenden Format zurück:\n"
            "{\n"
            '  "summary": "Ausführliche Zusammenfassung in Markdown",\n'
            '  "sources": [\n'
            '    {"title": "Titel", "url": "https://...", "description": "Beschreibung"},\n'
            '    ...\n'
            '  ]\n'
            "}\n"
            "Nutze ZUERST das web_search Tool, DANN antworte mit dem JSON!"
        )

        summary_text = ""
        all_sources: List[dict] = []
        seen_urls = set()
        query_count = 0

        for round_index in range(mode_config["max_rounds"]):
            round_instruction = ""
            if round_index > 0 and seen_urls:
                known_urls = "\n".join(f"- {url}" for url in list(seen_urls)[:12])
                round_instruction = (
                    "\n\nZusatzrunde:\n"
                    "Finde weitere relevante, aktuelle Entscheidungen, die nicht in dieser Liste enthalten sind:\n"
                    f"{known_urls}\n"
                    "Liefere neue Primärquellen mit klarem Entscheidungskern."
                )

            chat = xai_client.chat.create(
                model=model_name,
                tools=[web_search()],
            )
            chat.append(xai_user(full_user_message + round_instruction + json_instruction))
            print(
                f"[GROK] Sampling response (round {round_index + 1}/{mode_config['max_rounds']})..."
            )
            response = chat.sample()
            query_count += 1

            if hasattr(response, "usage") and response.usage:
                print(f"[GROK] Usage: {response.usage}")

            if hasattr(response, "tool_calls") and response.tool_calls:
                print(f"[GROK] Tool calls made: {len(response.tool_calls)}")

            round_sources: List[dict] = []
            if hasattr(response, "content") and response.content:
                response_text = str(response.content)
                print(f"[GROK] Raw response length: {len(response_text)} characters")
                try:
                    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    elif response_text.strip().startswith("{"):
                        json_str = response_text.strip()
                    else:
                        json_match = re.search(r'\{.*"summary".*"sources".*\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            raise ValueError("No JSON found in response")

                    parsed_output = GrokResearchOutput.model_validate_json(json_str)
                    if parsed_output.summary and len(parsed_output.summary) > len(summary_text):
                        summary_text = parsed_output.summary

                    for src in parsed_output.sources:
                        round_sources.append(
                            {
                                "url": src.url,
                                "title": src.title,
                                "description": src.description,
                                "source": "Grok",
                            }
                        )
                    print(f"[GROK] Extracted {len(round_sources)} structured sources in round")
                except Exception as parse_exc:
                    print(f"[ERROR] Failed to parse JSON from response: {parse_exc}")
                    print(f"[DEBUG] Response content (first 500 chars): {response_text[:500]}")
                    if response_text and len(response_text) > len(summary_text):
                        summary_text = response_text
            else:
                print("[WARN] No content in response")

            new_count = 0
            for source in round_sources:
                url = (source.get("url") or "").strip()
                if not url:
                    continue
                all_sources.append(source)
                if url not in seen_urls:
                    seen_urls.add(url)
                    new_count += 1

            elapsed = perf_counter() - started_at
            if elapsed >= mode_config["max_duration_sec"]:
                print(f"[GROK] Stopping due to mode timeout window ({elapsed:.1f}s)")
                break
            if round_index >= 1 and new_count < 2:
                print("[GROK] Early stop: low marginal gain in latest round")
                break

        print("[GROK] Research completed")

        quality_stats = {}
        structured_sources = normalize_and_rank_sources(
            all_sources,
            provider="Grok",
            context_hints=research_context_hints,
            limit=max_sources,
            domain_policy=domain_policy,
            recency_years=recency_years,
            stats=quality_stats,
        )

        # Check URLs and detect PDFs for the first 10 sources
        if structured_sources:
            max_checks = min(10, len(structured_sources))
            print(f"[GROK] Checking first {max_checks} sources for PDF detection...")
            await _enrich_sources_with_pdf_detection(structured_sources[:max_checks])

        supplied_sources = structured_sources

        # NOTE: asyl.net keyword suggestions are now generated in asylnet.py module
        # Grok module doesn't handle this anymore

        # Fix LaTeX formatting in text (replace $\S$ with §)
        clean_text = summary_text.strip()
        clean_text = clean_text.replace(r"$\S$", "§")
        clean_text = clean_text.replace(r"\S", "§")

        duration_ms = int((perf_counter() - started_at) * 1000)
        metadata = {
            "provider": "grok-4-1-fast",
            "model": model_name,
            "search_mode": search_mode,
            "max_sources": max_sources,
            "domain_policy": domain_policy,
            "jurisdiction_focus": jurisdiction_focus,
            "recency_years": recency_years,
            "query_count": query_count,
            "filtered_count": quality_stats.get("filtered_count", 0),
            "reranked_count": quality_stats.get("reranked_count", len(supplied_sources)),
            "source_count": len(supplied_sources),
            "duration_ms": duration_ms,
        }

        return ResearchResult(
            query=query,
            summary=markdown.markdown(
                clean_text,
                extensions=["extra", "sane_lists"],
                output_format="html"
            ),
            sources=supplied_sources,
            suggestions=[],  # Keywords generated by asylnet module
            metadata=metadata,
        )

    except Exception as e:
        print(f"[ERROR] in research_with_grok: {e}")
        print(traceback.format_exc())
        raise Exception(f"Grok research failed: {e}")

"""
Grok-4.1-Fast research with web_search tool and structured outputs.
"""

import json
import os
import re
import traceback
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import markdown
from pydantic import BaseModel, Field
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user
from xai_sdk.tools import web_search

from shared import ResearchResult, get_document_for_upload
from .prompting import build_research_priority_prompt
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


async def research_with_grok(
    query: str,
    attachment_path: Optional[str] = None,
    attachment_display_name: Optional[str] = None,
    attachment_ocr_text: Optional[str] = None,
    attachment_text_path: Optional[str] = None,
    attachment_anonymization_metadata: Optional[dict] = None,
    attachment_is_anonymized: bool = False,
    research_context_hints: Optional[List[str]] = None,
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

        document_text = None

        if attachment_path or attachment_text_path:
            try:
                upload_entry = {
                    "filename": attachment_display_name or "Bescheid",
                    "file_path": attachment_path,
                    "extracted_text_path": attachment_text_path,
                    "anonymization_metadata": attachment_anonymization_metadata,
                    "is_anonymized": attachment_is_anonymized,
                }
                selected_path, mime_type, _ = get_document_for_upload(upload_entry)

                if mime_type == "text/plain":
                    try:
                        with open(selected_path, "r", encoding="utf-8") as f:
                            document_text = f.read()
                        print(f"[GROK] Loaded {len(document_text)} characters from text file")
                    except Exception as exc:
                        print(f"[WARN] Failed to read text from disk: {exc}")
                        document_text = attachment_ocr_text if attachment_ocr_text else None
                else:
                    print(f"[GROK] No text file available, extracting text from PDF with PyMuPDF")
                    try:
                        pdf_doc = fitz.open(selected_path)
                        text_parts = []
                        for page in pdf_doc:
                            text_parts.append(page.get_text())
                        pdf_doc.close()
                        document_text = "\n".join(text_parts)
                        print(f"[GROK] Extracted {len(document_text)} characters from PDF")
                    except Exception as exc:
                        print(f"[WARN] Failed to extract text from PDF: {exc}")
            except Exception as exc:
                print(f"[WARN] Failed to resolve attachment for Grok: {exc}")

        if document_text:
            user_message = f"""Rechercheauftrag: {query}

Beigefügter BAMF-Bescheid (vollständig):
---
{document_text}
---

AUFGABE:
1. ANALYSIERE den Bescheid und identifiziere zentrale Rechtsfragen sowie Ablehnungsgründe.
2. NUTZE DAS WEB_SEARCH TOOL für konkrete Recherchen nach ähnlich gelagerten Entscheidungen und vergleichbaren Rechtsfragen in der gerichtlichen Praxis.
{context_anchor if context_anchor else ''}

Zusätzliche Hinweise:
- Betrachte vorrangig Entscheidungen deutscher Gerichte und behalte ggf. höhere Rechtsprechung explizit im Blick.
- Relevante Entwicklungen aus NRW separat besonders hervorheben.
- Beziehe, wenn sinnvoll, klassische BAMF-Felder mit ein: Dublin-/Überstellungsfragen, § 60 Abs. 5/7-AufenthG, medizinische oder humanitäre Abschiebungshindernisse, Art. 3 EMRK-Risikolagen, Vulnerabilität (Familie, Kinder, Traumatisierung) und Konstellationen wie Russland, Wehrdienst oder Kriegsdienstverweigerung.
""" + "\n\n" + build_research_priority_prompt(
                "Vergleiche die Ergebnisse strukturiert mit den im Bescheid aufgestellten Fragen."
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

        # Build input for xAI SDK (system prompt combined with user message)
        full_user_message = f"{system_prompt}\n\n{user_message}"

        # Use xAI SDK with web_search tool and structured output
        print(f"[GROK] Calling xAI SDK with web_search tool and structured output")

        # Initialize xAI SDK client
        xai_api_key = os.getenv("XAI_API_KEY")
        if not xai_api_key:
            raise ValueError("XAI_API_KEY environment variable is not set")

        xai_client = XAIClient(api_key=xai_api_key)

        # Create chat with web_search tool
        chat = xai_client.chat.create(
            model="grok-4-1-fast",
            tools=[web_search()],
        )

        # Add instruction to return JSON format in the prompt
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

        # Append user message with JSON instruction
        chat.append(xai_user(full_user_message + json_instruction))

        # Use sample() to allow web_search tool use (agentic behavior)
        print(f"[GROK] Sampling response (allowing web_search tool use)...")
        response = chat.sample()

        # Log usage statistics if available
        if hasattr(response, 'usage') and response.usage:
            print(f"[GROK] Usage: {response.usage}")

        # Check if tool calls were made
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"[GROK] Tool calls made: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"[GROK] Tool call {i+1}: {tool_call}")

        # Extract and parse JSON response
        summary_text = ""
        structured_sources = []

        if hasattr(response, 'content') and response.content:
            response_text = str(response.content)
            print(f"[GROK] Raw response length: {len(response_text)} characters")

            # Try to parse as JSON
            try:
                # Look for JSON in the response (might be wrapped in markdown code blocks)
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                elif response_text.strip().startswith('{'):
                    json_str = response_text.strip()
                else:
                    # Try to find JSON object in the text
                    json_match = re.search(r'\{.*"summary".*"sources".*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        raise ValueError("No JSON found in response")

                parsed_output = GrokResearchOutput.model_validate_json(json_str)
                summary_text = parsed_output.summary
                print(f"[GROK] Extracted {len(summary_text)} characters from structured output")

                # Convert StructuredSource objects to dict format
                for src in parsed_output.sources:
                    structured_sources.append({
                        "url": src.url,
                        "title": src.title,
                        "description": src.description,
                        "source": "Grok"
                    })
                print(f"[GROK] Extracted {len(structured_sources)} structured sources")
            except Exception as parse_exc:
                print(f"[ERROR] Failed to parse JSON from response: {parse_exc}")
                print(f"[DEBUG] Response content (first 500 chars): {response_text[:500]}")
                # Fallback to text response
                summary_text = response_text
        else:
            print(f"[WARN] No content in response")

        print(f"[GROK] Research completed")

        # Check URLs and detect PDFs for the first 10 sources
        if structured_sources:
            scored = [
                (source, _score_grok_source(source))
                for source in structured_sources
                if source.get("url")
            ]
            if scored:
                diversified = _apply_court_diversity_penalty(scored)
                diversified.sort(key=lambda item: item[1], reverse=True)
                structured_sources = [source for source, _ in diversified][:40]
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

        return ResearchResult(
            query=query,
            summary=markdown.markdown(
                clean_text,
                extensions=["extra", "sane_lists"],
                output_format="html"
            ),
            sources=supplied_sources,
            suggestions=[]  # Keywords generated by asylnet module
        )

    except Exception as e:
        print(f"[ERROR] in research_with_grok: {e}")
        print(traceback.format_exc())
        raise Exception(f"Grok research failed: {e}")

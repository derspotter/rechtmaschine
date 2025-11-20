"""
Grok-4-Fast research with web_search tool and structured outputs.
"""

import json
import os
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import markdown
from pydantic import BaseModel, Field
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user
from xai_sdk.tools import web_search

from shared import ResearchResult
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


async def research_with_grok(
    query: str,
    attachment_path: Optional[str] = None,
    attachment_display_name: Optional[str] = None,
    attachment_ocr_text: Optional[str] = None,
    attachment_text_path: Optional[str] = None
) -> ResearchResult:
    """
    Perform web research using Grok-4-Fast with agentic tool calling.
    Uses the new Responses API with web_search tool and structured outputs.
    Returns relevant links and sources for the user's query.

    Prefers OCR text when available for better accuracy.
    """

    try:

        # Build system prompt for legal research
        system_prompt = (
            "Du bist ein Rechercheassistent für deutsches Ausländerrecht. "
            "WICHTIG: Nutze das web_search Tool, um aktuelle Informationen im Internet zu suchen. "
            "Suche nach aktuellen Urteilen, Gesetzen, Verwaltungsvorschriften und Fachliteratur. "
            "Fokussiere dich auf seriöse Quellen wie Gerichte, Ministerien, asyl.net, und Fachzeitschriften. "
            "\n\n"
            "WICHTIGE AUSSCHLUSSKRITERIEN:\n"
            "- KEINE Pressemitteilungen von Gerichten (verlinke direkt auf die Urteile)\n"
            "- KEINE Nachrichtenartikel oder Blogposts als Hauptquellen\n"
            "- Suche nach den ORIGINALEN Gerichtsentscheidungen (Urteile, Beschlüsse), nicht nach Berichten darüber\n"
            "\n"
            "Durchsuche systematisch das Web und gib eine strukturierte Zusammenfassung mit konkreten Quellenangaben und Links."
        )

        # Build user message - include document text if available
        document_text = None
        suggestion_doc_entry: Optional[Dict[str, Optional[str]]] = None

        if attachment_ocr_text or attachment_path or attachment_text_path:
            suggestion_doc_entry = {
                "filename": attachment_display_name or "Bescheid",
                "extracted_text": attachment_ocr_text,
                "ocr_applied": bool(attachment_ocr_text),
                "file_path": attachment_path,
                "extracted_text_path": attachment_text_path,
            }

        if attachment_text_path and os.path.exists(attachment_text_path):
            try:
                with open(attachment_text_path, 'r', encoding='utf-8') as f:
                    document_text = f.read()
                print(f"[GROK] Loaded {len(document_text)} characters from OCR text file")
            except Exception as exc:
                print(f"[WARN] Failed to read OCR text from disk: {exc}")
                document_text = attachment_ocr_text if attachment_ocr_text else None
        elif attachment_ocr_text:
            print(f"[GROK] Using OCR text from fallback cache")
            document_text = attachment_ocr_text
        elif attachment_path:
            # No OCR available, extract text from PDF using PyMuPDF
            print(f"[GROK] No OCR text available, extracting text from PDF with PyMuPDF")
            try:
                pdf_doc = fitz.open(attachment_path)
                text_parts = []
                for page in pdf_doc:
                    text_parts.append(page.get_text())
                pdf_doc.close()
                document_text = "\n".join(text_parts)
                print(f"[GROK] Extracted {len(document_text)} characters from PDF")
            except Exception as exc:
                print(f"[WARN] Failed to extract text from PDF: {exc}")

        if document_text and suggestion_doc_entry and not suggestion_doc_entry.get("extracted_text"):
            suggestion_doc_entry["extracted_text"] = document_text
            suggestion_doc_entry["ocr_applied"] = True

        if document_text:
            user_message = f"""Rechercheauftrag: {query}

Beigefügter BAMF-Bescheid (vollständig):
---
{document_text}
---

AUFGABE IN 2 SCHRITTEN:

1. ANALYSIERE den beigefügten Bescheid und identifiziere:
   - Die Hauptablehnungsgründe
   - Die relevanten rechtlichen Fragen (z.B. Dublin, systemische Mängel, etc.)
   - Die betroffenen Länder und Verfahren
   - Medizinische oder humanitäre Aspekte

2. NUTZE DAS WEB_SEARCH TOOL um gezielt nach Rechtsprechung zu suchen:

**ABSOLUTE PRIORITÄT - POSITIVE ENTSCHEIDUNGEN DEUTSCHER GERICHTE:**
   - VG/OVG-Urteile, die BAMF-Ablehnungen in vergleichbaren Fällen AUFGEHOBEN haben
   - Urteile, die ZUGUNSTEN der Kläger/Antragsteller entschieden haben
   - Entscheidungen, die ähnliche Ablehnungsgründe WIDERLEGT oder differenziert haben
   - Gerichtsbeschlüsse, die einstweiligen Rechtsschutz GEWÄHRT haben

Diese positiven Entscheidungen sind am wichtigsten für die Argumentation!

Suche zusätzlich nach:
   - BVerwG-Rechtsprechung zu den identifizierten Rechtsfragen
   - EGMR/EuGH-Urteilen zu systemischen Mängeln
   - Aktuellen Lageberichten (AIDA, UNHCR, Pro Asyl) oder Rechtsgutachten
   - Aktuelle Rechtsentwicklungen (2023-2025)

Suchstrategie:
   - Nutze Suchbegriffe wie: "VG aufgehoben", "OVG stattgegeben", "Klage erfolgreich", "BAMF unterlag"
   - Priorisiere asyl.net, openjur.de, Gerichtsdatenbanken
   - Fokus auf aktuelle Urteile (letzten 2 Jahre)
   - VERMEIDE Pressemitteilungen - suche direkt nach den Originalentscheidungen

WICHTIG: Die Analyse des Bescheids ist NUR die Vorbereitung. Du MUSST anschließend mit web_search nach aktueller Rechtsprechung suchen!

Liefere strukturierte Ergebnisse mit konkreten Links zu den ORIGINALURTEILEN (nicht zu Pressemitteilungen)."""
        else:
            user_message = f"""Rechercheauftrag: {query}

WICHTIG: Nutze das web_search Tool, um aktuelle Quellen zu recherchieren.

**ABSOLUTE PRIORITÄT - POSITIVE ENTSCHEIDUNGEN DEUTSCHER GERICHTE:**
Suche nach VG/OVG-Urteilen, die in vergleichbaren Fällen ZUGUNSTEN der Kläger/Antragsteller entschieden haben.
Diese positiven Entscheidungen sind am wichtigsten für die Argumentation!

Suchstrategie:
- Nutze Suchbegriffe wie: "VG aufgehoben", "OVG stattgegeben", "Klage erfolgreich", "BAMF unterlag"
- Priorisiere asyl.net, openjur.de, Gerichtsdatenbanken
- Fokus auf aktuelle Urteile (2023-2025)
- VERMEIDE Pressemitteilungen - suche direkt nach den Originalentscheidungen

Durchsuche systematisch das Web und liefere relevante Ergebnisse mit konkreten Links zu den ORIGINALURTEILEN (nicht zu Pressemitteilungen)."""

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
            model="grok-4-fast",
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

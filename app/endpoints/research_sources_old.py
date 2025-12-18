import asyncio
import hashlib
import json
import os
import re
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin, urlparse

import fitz  # PyMuPDF
import httpx
import markdown
import pikepdf
import traceback
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from google.genai import types
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user
from xai_sdk.tools import web_search

from shared import (
    AddSourceRequest,
    DocumentCategory,
    ResearchRequest,
    ResearchResult,
    SavedSource,
    DOWNLOADS_DIR,
    broadcast_sources_snapshot,
    get_document_for_upload,
    get_gemini_client,
    get_openai_client,
    get_xai_client,
    limiter,
    load_document_text,
    store_document_text,
)
from database import SessionLocal, get_db
from models import Document, ResearchSource

router = APIRouter()

ASYL_NET_BASE_URL = "https://www.asyl.net"
ASYL_NET_SEARCH_PATH = "/recht/entscheidungsdatenbank"
ASYL_NET_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
ASYL_NET_SUGGESTIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "asyl_net_suggestions.json"
try:
    with open(ASYL_NET_SUGGESTIONS_FILE, "r", encoding="utf-8") as f:
        _asyl_suggestions_payload = json.load(f)
    ASYL_NET_ALL_SUGGESTIONS: List[str] = _asyl_suggestions_payload.get("suggestions", [])
except FileNotFoundError:
    print(f"Warning: asyl.net suggestions file not found at {ASYL_NET_SUGGESTIONS_FILE}")
    ASYL_NET_ALL_SUGGESTIONS = []


# Database helper functions (no longer needed - using ORM directly in endpoints)


def _looks_like_pdf(headers) -> bool:
    """Check HTTP headers for indicators that the response is a PDF."""
    if not headers:
        return False
    try:
        content_type = str(headers.get('content-type', '')).lower()
    except Exception:
        content_type = ''
    try:
        content_disposition = str(headers.get('content-disposition', '')).lower()
    except Exception:
        content_disposition = ''
    if 'application/pdf' in content_type:
        return True
    if '.pdf' in content_disposition:
        return True
    return False

async def generate_asylnet_keyword_suggestions(
    query: str,
    attachment_label: Optional[str] = None,
    attachment_doc: Optional[Dict[str, Optional[str]]] = None,
    existing_upload: Optional[Any] = None,
    client: Optional[Any] = None,
) -> List[str]:
    """
    Generate asyl.net keyword suggestions using Gemini.
    Returns a list of keyword strings from the ASYL_NET_ALL_SUGGESTIONS list.
    """
    gemini_client = client
    uploaded_file = existing_upload
    uploaded_name: Optional[str] = None
    temp_path: Optional[str] = None

    try:
        print("[KEYWORD GEN] Calling Gemini for asyl.net keyword suggestions...")
        if gemini_client is None:
            gemini_client = get_gemini_client()

        suggestion_text = "\n".join(f"- {s}" for s in ASYL_NET_ALL_SUGGESTIONS) if ASYL_NET_ALL_SUGGESTIONS else "- (keine Schlagwörter geladen)"
        trimmed_query = (query or "").strip()

        if attachment_label:
            suggestions_block = f"""Analysiere den beigefügten BAMF-Bescheid "{attachment_label}" und leite daraus die geeignetsten Schlagwörter ab.
Nutze insbesondere Tatbestand, rechtliche Würdigung, Länder- oder Herkunftsbezüge sowie angewandte Rechtsnormen.
Zusätzliche Aufgabenstellung / Notiz:
{trimmed_query or "- (keine zusätzliche Notiz)"}"""
        else:
            suggestions_block = f"""Die folgende Anfrage lautet:
{trimmed_query or "(Keine Anfrage angegeben)"}"""

        prompt_suggestions = f"""Du bist ein Rechercheassistent für deutsches Asylrecht.

{suggestions_block}

Hier ist die Schlagwort-Liste für asyl.net (verwende ausschließlich Begriffe aus dieser Liste):
{suggestion_text}

WICHTIGE HINWEISE:
- Bevorzuge IMMER aktuelle Rechtsbegriffe gegenüber veralteten Begriffen
- Bei Dublin-Fällen: Nutze "Dublin-III VO" (aktuelle Verordnung), NICHT "Dublin-Übereinkommen" (veraltet)
- Bei Verordnungen: Nutze die aktuellste Version (z.B. "Dublin-III VO", nicht "Dublin-II VO")
- Fokussiere auf die im Bescheid tatsächlich angewendeten Rechtsgrundlagen

Gib mir genau 1 bis 3 Schlagwörter aus der Liste zurück, die am besten zur Anfrage passen.
Antwortformat: {{"suggestions": ["...", "..."]}} (keine zusätzlichen Erklärungen, kein Markdown)."""

        contents: List[Any] = [prompt_suggestions]

        if uploaded_file is None and attachment_doc:
            try:
                upload_entry = dict(attachment_doc)
                file_path, mime_type, needs_cleanup = get_document_for_upload(upload_entry)
                if needs_cleanup:
                    temp_path = file_path
                with open(file_path, "rb") as file_handle:
                    display_name = attachment_label or upload_entry.get("filename") or "Bescheid"
                    uploaded_file = gemini_client.files.upload(
                        file=file_handle,
                        config={
                            "mime_type": mime_type,
                            "display_name": f"{display_name}{'.txt' if mime_type == 'text/plain' else ''}"
                        }
                    )
                uploaded_name = uploaded_file.name
            except Exception as prep_exc:
                print(f"[KEYWORD GEN] Failed to prepare attachment for keyword suggestions: {prep_exc}")
                uploaded_file = None

        if uploaded_file:
            contents.append(uploaded_file)

        response_suggestions = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-3-flash-preview",
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0)
        )

        raw_text_suggestions = response_suggestions.text if response_suggestions.text else "{}"
        try:
            suggestions_data = json.loads(raw_text_suggestions)
            asyl_suggestions = suggestions_data.get("suggestions", [])
            if isinstance(asyl_suggestions, str):
                asyl_suggestions = [asyl_suggestions]
            asyl_suggestions = [s.strip() for s in asyl_suggestions if isinstance(s, str) and s.strip()]
            # Deduplicate
            seen = set()
            unique_suggestions = []
            for s in asyl_suggestions:
                low = s.lower()
                if low not in seen:
                    seen.add(low)
                    unique_suggestions.append(s)
            asyl_suggestions = unique_suggestions[:5]
            print(f"[KEYWORD GEN] Generated {len(asyl_suggestions)} keyword suggestions: {asyl_suggestions}")
            return asyl_suggestions
        except json.JSONDecodeError:
            print("[KEYWORD GEN] Failed to parse JSON response")
            return []
    except Exception as e:
        print(f"[KEYWORD GEN] Failed to generate keywords: {e}")
        return []
    finally:
        if uploaded_name and existing_upload is None and gemini_client:
            try:
                gemini_client.files.delete(name=uploaded_name)
            except Exception as cleanup_exc:
                print(f"[KEYWORD GEN] Failed to delete temporary Gemini file {uploaded_name}: {cleanup_exc}")
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception as cleanup_exc:
                print(f"[KEYWORD GEN] Failed to delete temporary text file {temp_path}: {cleanup_exc}")


async def research_with_gemini(
    query: str,
    attachment_path: Optional[str] = None,
    attachment_display_name: Optional[str] = None,
    attachment_ocr_text: Optional[str] = None,
    attachment_text_path: Optional[str] = None
) -> ResearchResult:
    """
    Perform web research using Gemini with Google Search grounding.
    Returns relevant links and sources for the user's query.

    Prefers OCR text when available for better accuracy.
    """
    uploaded_file = None
    uploaded_name: Optional[str] = None
    attachment_label = None
    temp_text_path = None
    doc_entry: Optional[Dict[str, Optional[str]]] = None

    try:
        client = get_gemini_client()
        print("[GEMINI] Gemini client initialized")

        # Upload attachment if provided (OCR text or PDF)
        if attachment_ocr_text or attachment_path or attachment_text_path:
            attachment_label = attachment_display_name or "Bescheid"

            # Create a document entry dict for the helper function
            doc_entry = {
                "filename": attachment_label,
                "extracted_text": attachment_ocr_text,
                "ocr_applied": bool(attachment_ocr_text),
                "file_path": attachment_path,
                "extracted_text_path": attachment_text_path,
            }

            try:
                file_path, mime_type, needs_cleanup = get_document_for_upload(doc_entry)
                temp_text_path = file_path if needs_cleanup else None

                if mime_type == "text/plain":
                    print(f"[INFO] Using OCR text for research: {attachment_label}")

                with open(file_path, "rb") as file_handle:
                    uploaded_file = client.files.upload(
                        file=file_handle,
                        config={
                            "mime_type": mime_type,
                            "display_name": f"{attachment_label}{'.txt' if mime_type == 'text/plain' else ''}"
                        }
                    )
                uploaded_name = uploaded_file.name
                print(f"Attachment uploaded successfully for research: {attachment_label} ({mime_type})")

            except Exception as exc:
                print(f"[ERROR] Attachment upload failed: {exc}")
                if temp_text_path:
                    try:
                        os.unlink(temp_text_path)
                        temp_text_path = None
                    except:
                        pass
                raise

        # Configure Google Search grounding tool
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        suggestion_text = "\n".join(f"- {s}" for s in ASYL_NET_ALL_SUGGESTIONS) if ASYL_NET_ALL_SUGGESTIONS else "- (keine Schlagwörter geladen)"
        trimmed_query = (query or "").strip()

        if attachment_label:
            query_block = f"""Analysiere den beigefügten BAMF-Bescheid "{attachment_label}" (PDF im Anhang).
Nutze den vollständigen Inhalt, um die tragenden Erwägungen, Rechtsgrundlagen, Länderbezüge sowie strittigen Punkte herauszuarbeiten.
Leite daraus die wichtigsten Recherchefragen ab, mit denen aktuelle Rechtsprechung, Verwaltungsvorschriften oder Lageberichte gefunden werden können.
Zusätzliche Aufgabenstellung / Notiz:
{trimmed_query or "- (keine zusätzliche Notiz)"}"""
        else:
            query_block = f"""Recherchiere und liste relevante Quellen zur folgenden Anfrage auf:
{trimmed_query or "(Keine Anfrage angegeben)"}"""

        prompt_summary = f"""Du bist ein Rechercheassistent für deutsches Asylrecht.

{query_block}

WICHTIG: Nutze Google Search Grounding ausschließlich für Quellen von offiziellen Stellen wie Gerichten oder Verwaltungsbehörden (z. B. BAMF, BMI, EU-Behörden) sowie wissenschaftliche Fachveröffentlichungen. Suche gezielt nach faktenbasierten Berichten, gerichtlichen Entscheidungen, administrativen Veröffentlichungen und peer-reviewten Studien. Ignoriere Treffer, die nicht von solchen Institutionen stammen.
- Gerichtsentscheidungen (VG, OVG, BVerwG, EuGH, EGMR)
- Veröffentlichungen von BAMF, Verwaltungsgerichten und anderen Behörden
- Gesetzestexte und Verordnungen (AsylG, AufenthG, GG)
- Faktenbasierte Lageberichte, COI-Analysen und andere behördliche Sachstandsberichte
- Wissenschaftliche Publikationen und peer-reviewte Studien (Universitäten, NIH, WHO, akademische Journale)
- Rechtswissenschaftliche Veröffentlichungen mit amtlichem bzw. gerichtlichem Ursprung
- Offizielle Behörden- und Forschungs-Websites (.gov, .bund.de, .europa.eu, .int, .edu, .ac)

VERMEIDE:
- Blogs und persönliche Meinungen
- Journalistische Artikel oder Presseportale
- Kommerzielle Beratungsseiten
- Nicht-verifizierte Quellen
- asyl.net (wird separat recherchiert)

Gib eine kurze Übersicht (2-3 Sätze) der wichtigsten Erkenntnisse. Erwähne die Quellen nur kurz im Text (z.B. "laut Bundesverwaltungsgericht" oder "BAMF-Bericht vom ..."), aber füge keine URLs oder vollständige Quellenangaben hinzu - diese werden separat angezeigt."""

        if attachment_label:
            suggestions_block = f"""Analysiere den beigefügten BAMF-Bescheid "{attachment_label}" (PDF im Anhang) und leite daraus die geeignetsten Schlagwörter ab.
Nutze insbesondere Tatbestand, rechtliche Würdigung, Länder- oder Herkunftsbezüge sowie angewandte Rechtsnormen.
Zusätzliche Aufgabenstellung / Notiz:
{trimmed_query or "- (keine zusätzliche Notiz)"}"""
        else:
            suggestions_block = f"""Die folgende Anfrage lautet:
{trimmed_query or "(Keine Anfrage angegeben)"}"""

        prompt_suggestions = f"""Du bist ein Rechercheassistent für deutsches Asylrecht.

{suggestions_block}

Hier ist die Schlagwort-Liste für asyl.net (verwende ausschließlich Begriffe aus dieser Liste):
{suggestion_text}

Gib mir genau 1 bis 3 Schlagwörter aus der Liste zurück, die am besten zur Anfrage passen.
Antwortformat: {{\"suggestions\": [\"...\", \"...\"]}} (keine zusätzlichen Erklärungen, kein Markdown)."""

        async def call_summary():
            contents = [prompt_summary, uploaded_file] if uploaded_file else [prompt_summary]
            return await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3-flash-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=0.0
                )
            )

        print("Calling Gemini API for summary and keyword suggestions in parallel...")
        response_summary, asyl_suggestions = await asyncio.gather(
            call_summary(),
            generate_asylnet_keyword_suggestions(
                query,
                attachment_label,
                attachment_doc=doc_entry,
                existing_upload=uploaded_file,
                client=client
            )
        )
        print("Gemini calls successful")

        summary_markdown = (response_summary.text or "").strip()
        if summary_markdown:
            summary_markdown = "\n".join(line.rstrip() for line in summary_markdown.replace("\r\n", "\n").split("\n"))
        else:
            summary_markdown = "**Web-Recherche**\n\nKeine Rechercheergebnisse gefunden."
        if not summary_markdown.lower().startswith("**web-recherche**"):
            summary_markdown = f"**Web-Recherche**\n\n{summary_markdown}"

        summary_html = markdown.markdown(
            summary_markdown,
            extensions=["extra", "sane_lists"],
            output_format="html"
        )
        print(f"Summary extracted: {summary_html[:100]}...")

        # Extract sources from grounding metadata
        sources = []
        if hasattr(response_summary, 'candidates') and response_summary.candidates:
            candidate = response_summary.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                grounding_meta = candidate.grounding_metadata

                # Extract grounding chunks (search results with titles, URLs, and snippets)
                if hasattr(grounding_meta, 'grounding_chunks') and grounding_meta.grounding_chunks:
                    for chunk in grounding_meta.grounding_chunks:
                        if hasattr(chunk, 'web') and chunk.web:
                            source = {}
                            if hasattr(chunk.web, 'uri'):
                                source['url'] = chunk.web.uri
                            if hasattr(chunk.web, 'title'):
                                source['title'] = chunk.web.title
                            else:
                                source['title'] = source.get('url', 'Quelle')

                            if source.get('url'):
                                lowered = source['url'].lower()
                                if lowered.endswith('.pdf') or '.pdf?' in lowered:
                                    source['pdf_url'] = source['url']

                            # Try to extract snippet/description from grounding chunk
                            description = None

                            # Check if there's a snippet in the web object
                            if hasattr(chunk.web, 'snippet'):
                                description = chunk.web.snippet

                            # Check if there's content in the grounding support
                            if not description and hasattr(chunk, 'grounding_support'):
                                gs = chunk.grounding_support
                                if hasattr(gs, 'segment'):
                                    seg = gs.segment
                                    if hasattr(seg, 'text'):
                                        description = seg.text

                            # Fallback: check if chunk itself has text
                            if not description and hasattr(chunk, 'retrieved_context'):
                                rc = chunk.retrieved_context
                                if hasattr(rc, 'text'):
                                    description = rc.text[:200]  # Limit to 200 chars

                            source['description'] = description if description else "Relevante Quelle aus Web-Recherche"

                            if source.get('url'):
                                print(f"DEBUG: Extracted source with description: {source.get('description', 'N/A')[:100]}")
                                sources.append(source)

        await enrich_web_sources_with_pdf(sources)

        print(f"Extracted {len(sources)} sources from grounding metadata with descriptions from Gemini")

        return ResearchResult(
            query=query,
            summary=summary_html,
            sources=sources,
            suggestions=asyl_suggestions
        )

    except Exception as e:
        print(f"ERROR in research_with_gemini: {e}")
        print(traceback.format_exc())
        raise Exception(f"Research failed: {e}")
    finally:
        if uploaded_name:
            try:
                cleanup_client = get_gemini_client()
                cleanup_client.files.delete(name=uploaded_name)
                print(f"Deleted uploaded attachment from Gemini: {uploaded_name}")
            except Exception as cleanup_exc:
                print(f"Failed to delete uploaded attachment {uploaded_name}: {cleanup_exc}")

        # Clean up temporary text file if it was created
        if temp_text_path:
            try:
                os.unlink(temp_text_path)
                print(f"Deleted temporary OCR text file: {temp_text_path}")
            except Exception as cleanup_exc:
                print(f"Failed to delete temporary file {temp_text_path}: {cleanup_exc}")


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

        # Generate asyl.net keyword suggestions using Gemini (Grok doesn't do this)
        asyl_suggestions = []
        try:
            suggestion_label = attachment_display_name or (Path(attachment_path).name if attachment_path else None)
            asyl_suggestions = await generate_asylnet_keyword_suggestions(
                query,
                suggestion_label,
                attachment_doc=suggestion_doc_entry
            )
        except Exception as e:
            print(f"[GROK] Gemini keyword suggestion call failed: {e}")
            asyl_suggestions = []

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
            suggestions=asyl_suggestions  # Generated by Gemini for asyl.net search
        )

    except Exception as e:
        print(f"[ERROR] in research_with_grok: {e}")
        print(traceback.format_exc())
        raise Exception(f"Grok research failed: {e}")


def _extract_grok_structured_sources(text: str) -> List[Dict[str, str]]:
    """
    Extract structured sources from Grok's formatted response text.
    Grok formats sources like:
    "VG München, Urteil vom 20.12.2023: Description...
     Quelle: https://example.com
     Relevanz: ..."
    """
    if not text:
        return []

    sources: List[Dict[str, str]] = []
    seen_urls = set()

    # Pattern: Look for "Quelle:" or "Link:" followed by URLs in parentheses
    # Format: "... Quelle: source_name (URL); source_name (URL)."
    lines = text.split('\n')

    for i, line in enumerate(lines):
        # Look for lines with "Quelle:" or "Link:"
        quelle_match = re.search(r'(?:Link|Quelle):', line, re.IGNORECASE)
        if not quelle_match:
            continue

        # Extract all URLs in parentheses after "Quelle:"
        text_after_quelle = line[quelle_match.end():]
        url_matches = re.findall(r'\((https?://[^\)]+)\)', text_after_quelle)

        if not url_matches:
            continue

        # Extract title and description from the text before "Quelle:"
        text_before_quelle = line[:quelle_match.start()].strip()

        # If text is too short, look at previous lines
        if len(text_before_quelle) < 50 and i > 0:
            # Look back up to 3 lines
            for j in range(1, min(4, i + 1)):
                prev_line = lines[i - j].strip()
                text_before_quelle = prev_line + " " + text_before_quelle
                if len(text_before_quelle) > 100:
                    break

        # Extract title (look for court name, decision type, date, case number)
        title_pattern = r'^([^:]+?(?:Urteil|Beschluss|Bericht|Report|Update|vom).*?(?:\d{4}|[A-Z]\s+\d+)[^:]*?)(?:\s*[\(:)]|:|$)'
        title_match = re.match(title_pattern, text_before_quelle, re.IGNORECASE)

        if title_match:
            title = title_match.group(1).strip()
        else:
            # Fallback: take text up to first colon or parenthesis or 120 chars
            for delimiter in [':', '(']:
                pos = text_before_quelle.find(delimiter)
                if pos > 20:
                    title = text_before_quelle[:pos].strip()
                    break
            else:
                title = text_before_quelle[:120].strip()

        # Description is everything after the title and before "Quelle:"
        if ':' in text_before_quelle and title:
            remaining = text_before_quelle[len(title):].strip()
            if remaining.startswith(':'):
                description = remaining[1:].strip()
            else:
                description = remaining
        else:
            description = text_before_quelle if not title else ""

        # Limit description length
        if len(description) > 300:
            description = description[:297] + "..."

        # Make sure we have at least some title
        if not title or len(title) < 10:
            title = text_before_quelle[:100].strip() if text_before_quelle else "Quelle"

        # Add each URL found as a separate source with the same title/description
        for url in url_matches:
            url = url.strip().rstrip('.,;:')
            if url in seen_urls:
                continue
            seen_urls.add(url)

            sources.append({
                "url": url,
                "title": title,
                "description": description,
                "source": "Grok"
            })

    print(f"[GROK] Parsed {len(sources)} structured sources from text")
    return sources


async def _enrich_sources_with_pdf_detection(sources: List[Dict[str, str]]) -> None:
    """
    Enrich sources with PDF detection using the existing enrich_web_sources_with_pdf function.
    """
    await enrich_web_sources_with_pdf(sources, max_checks=len(sources), concurrency=3)


def _extract_sources_from_grok_response(text: Any) -> List[Dict[str, str]]:
    """
    Extract source URLs and titles from Grok's response text.
    Grok embeds citations in the text - we need to parse them.
    """
    source_text = _get_text_from_chat_message(text)
    if not source_text:
        return []

    sources: List[Dict[str, str]] = []

    # Look for URL patterns
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\])]+'
    urls = re.findall(url_pattern, source_text)

    # Deduplicate and create source objects
    seen_urls = set()
    for url in urls:
        if url not in seen_urls:
            seen_urls.add(url)
            # Try to extract title from surrounding text
            title = _extract_title_near_url(source_text, url)
            sources.append({
                "url": url,
                "title": title or url,
                "description": "",  # Grok doesn't provide separate snippets
                "source": "Grok"
            })

    print(f"[GROK] Extracted {len(sources)} unique URLs from response")
    return sources


def _convert_grok_citations(citations: Optional[Any]) -> List[Dict[str, str]]:
    """
    Convert Grok's structured citations (AnnotationURLCitation objects) into ResearchResult format.
    """
    if not citations:
        return []

    normalized_sources: List[Dict[str, str]] = []

    def _normalize_entry(entry: Any) -> Optional[Dict[str, str]]:
        # Handle string URLs
        if isinstance(entry, str):
            return {"url": entry, "title": entry, "description": "", "source": "Grok"}

        # Handle dict citations
        if isinstance(entry, dict):
            url = entry.get("url") or entry.get("link") or entry.get("href")
            if not url:
                return None
            title = entry.get("title") or entry.get("name") or url
            description = entry.get("description") or entry.get("snippet") or ""
            return {"url": url, "title": title, "description": description, "source": "Grok"}

        # Handle AnnotationURLCitation objects from xAI SDK
        if hasattr(entry, 'url'):
            url = entry.url
            if not url:
                return None
            title = getattr(entry, 'title', None) or url
            description = getattr(entry, 'description', '') or ""
            return {"url": url, "title": title, "description": description, "source": "Grok"}

        return None

    for item in citations:
        normalized = _normalize_entry(item)
        if normalized:
            normalized_sources.append(normalized)

    if normalized_sources:
        print(f"[GROK] Converted {len(normalized_sources)} citations from structured response")
    return normalized_sources


def _get_text_from_chat_message(content: Any) -> str:
    """Normalize OpenAI-style message content (str or list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    parts.append(str(text_value))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _extract_title_near_url(text: str, url: str) -> Optional[str]:
    """Try to extract a title from text near the URL."""

    # Look for text in brackets before the URL: [Title](url)
    markdown_pattern = rf'\[([^\]]+)\]\({re.escape(url)}\)'
    match = re.search(markdown_pattern, text)
    if match:
        return match.group(1)

    # Look for text immediately before the URL (within 100 chars)
    url_index = text.find(url)
    if url_index > 0:
        before = text[max(0, url_index-100):url_index].strip()
        # Take the last sentence or phrase
        sentences = before.split('.')
        if sentences and len(sentences[-1].strip()) > 5:
            return sentences[-1].strip()

    return None


# ===== LEGAL DATABASE SEARCH TOOLS =====

async def search_open_legal_data(query: str, court: Optional[str] = None, date_range: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search Open Legal Data API for German case law.
    Args:
        query: Search query string
        court: Optional court filter (e.g., "BGH", "BVerwG")
        date_range: Optional date range (e.g., "2020-2025")
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching Open Legal Data: query='{query}', court={court}, date_range={date_range}")

        # Open Legal Data API appears to be having issues, skip for now
        # The API endpoint may have changed or requires different authentication
        print("Open Legal Data API currently unavailable, skipping")
        return []

        # Keeping old code commented out for reference:
        # async with httpx.AsyncClient(timeout=30.0) as client:
        #     params = {"q": query}
        #     if court:
        #         params["court"] = court
        #     if date_range:
        #         params["date"] = date_range
        #     response = await client.get("https://de.openlegaldata.io/api/cases/", params=params)
        #     response.raise_for_status()
        #     data = response.json()
        #     sources = []
        #     results = data.get("results", [])[:5]
        #     for case in results:
        #         sources.append({
        #             "title": case.get("title", "Open Legal Data Case"),
        #             "url": case.get("url", f"https://de.openlegaldata.io/case/{case.get('slug', '')}")
        #         })
        #     print(f"Open Legal Data returned {len(sources)} sources")
        #     return sources

    except Exception as e:
        print(f"Error searching Open Legal Data: {e}")
        return []

async def search_refworld(query: str, country: Optional[str] = None, doc_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search Refworld (UNHCR) using direct HTTP requests.
    Args:
        query: Search query
        country: Optional country filter
        doc_type: Optional document type filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching Refworld: query='{query}', country={country}, doc_type={doc_type}")

        # Build search URL with query parameters
        search_query = query
        if country:
            search_query += f" {country}"

        search_url = f"https://www.refworld.org/search?query={search_query}&type=caselaw"

        print(f"Fetching Refworld: {search_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, follow_redirects=True)
            response.raise_for_status()
            html = response.text

            # Parse HTML to extract results
            sources = []

            # Extract links from search results
            # Refworld typically uses /cases/ or /docid/ URLs
            pattern = r'<a[^>]+href="(/cases/[^"]+|/docid/[^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, html)

            for url_path, title in matches[:5]:  # Limit to 5 results
                if not title.strip() or len(title.strip()) < 5:
                    continue

                full_url = f"https://www.refworld.org{url_path}"

                sources.append({
                    "title": title.strip(),
                    "url": full_url,
                    "description": "UNHCR Refworld - Rechtsprechung und Länderdokumentation"
                })

            print(f"Refworld returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching Refworld: {e}")
        traceback.print_exc()
        return []

async def search_euaa_coi(query: str, country: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search EUAA COI Portal using direct HTTP requests.
    Args:
        query: Search query
        country: Optional country filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching EUAA COI: query='{query}', country={country}")

        # Build search URL with query parameters
        search_query = query
        if country:
            search_query += f" {country}"

        search_url = f"https://coi.euaa.europa.eu/search?q={search_query}"

        print(f"Fetching EUAA COI: {search_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, follow_redirects=True)
            response.raise_for_status()
            html = response.text

            # Parse HTML to extract results
            sources = []

            # Extract links from search results
            pattern = r'<a[^>]+href="(/[^"]*?document[^"]+|/admin/[^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, html)

            for url_path, title in matches[:5]:  # Limit to 5 results
                if not title.strip() or len(title.strip()) < 5:
                    continue

                full_url = f"https://coi.euaa.europa.eu{url_path}" if not url_path.startswith('http') else url_path

                sources.append({
                    "title": title.strip(),
                    "url": full_url,
                    "description": "EUAA COI Portal - Country of Origin Information"
                })

            print(f"EUAA COI returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching EUAA COI: {e}")
        traceback.print_exc()
        return []

async def get_asyl_net_keyword_suggestions(partial_query: str) -> List[str]:
    """
    Get keyword suggestions from asyl.net using Playwright to handle autocomplete.
    Args:
        partial_query: Partial keyword to get suggestions for
        Returns:
        List of suggested keywords
    """
    try:
        print(f"Getting asyl.net keyword suggestions for: '{partial_query}'")
        if not ASYL_NET_ALL_SUGGESTIONS:
            print("No cached asyl.net suggestions loaded")
            return []

        prefix = (partial_query or "").strip().lower()
        if not prefix:
            return ASYL_NET_ALL_SUGGESTIONS[:10]

        matches = [s for s in ASYL_NET_ALL_SUGGESTIONS if s.lower().startswith(prefix)]

        if len(matches) < 5:
            matches.extend(
                s for s in ASYL_NET_ALL_SUGGESTIONS
                if prefix in s.lower() and s not in matches
            )

        result = matches[:10]
        print(f"Found {len(result)} keyword suggestions")
        return result

    except Exception as e:
        print(f"Error getting keyword suggestions from cache: {e}")
        traceback.print_exc()
        return []


async def enrich_web_sources_with_pdf(
    sources: List[Dict[str, str]],
    max_checks: int = 10,
    concurrency: int = 3
) -> None:
    """Use Playwright to detect direct PDF links for web research sources."""
    if not sources:
        return

    targets = [src for src in sources if src.get('url')][:max_checks]
    if not targets:
        return


    async def detect_pdf_via_http(url: str) -> Optional[str]:
        """Probe a URL via HTTP to determine if it serves a PDF directly."""
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=10.0,
                headers={"User-Agent": ASYL_NET_USER_AGENT}
            ) as client:
                try:
                    head_response = await client.head(url)
                    if _looks_like_pdf(head_response.headers):
                        return str(head_response.url)
                except httpx.HTTPError as err:
                    status = getattr(getattr(err, "response", None), "status_code", None)
                    if status not in {401, 403, 404, 405, 406, 409, 410}:
                        print(f"HEAD probe failed for {url}: {err}")

                try:
                    get_response = await client.get(
                        url,
                        headers={"Range": "bytes=0-0"}
                    )
                    if _looks_like_pdf(get_response.headers):
                        return str(get_response.url)
                except httpx.HTTPError as err:
                    status = getattr(getattr(err, "response", None), "status_code", None)
                    if status not in {401, 403, 404, 405, 406, 409, 410}:
                        print(f"GET probe failed for {url}: {err}")
        except Exception as exc:
            print(f"HTTP probing failed for {url}: {exc}")
        return None

    async def process_source(browser, source):
        url = source.get('url')
        if not url:
            return

        lowered = url.lower()
        if lowered.endswith('.pdf') or '.pdf?' in lowered:
            source['pdf_url'] = url
            return

        direct_pdf = await detect_pdf_via_http(url)
        if direct_pdf:
            source['pdf_url'] = direct_pdf
            return

        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        try:
            response = await page.goto(url, wait_until='load', timeout=15000)

            if response and _looks_like_pdf(response.headers):
                source['pdf_url'] = response.url or url
                return

            current_url = page.url or url

            # Update the source URL with the resolved URL (after following redirects)
            if current_url != url:
                source['url'] = current_url
                print(f"Resolved URL: {url} -> {current_url}")

            lowered_current = current_url.lower()
            if lowered_current.endswith('.pdf') or '.pdf?' in lowered_current:
                source['pdf_url'] = current_url
                return

            base_url = current_url

            # Look for PDF links
            print(f"Searching for PDF links on: {current_url}")

            # Strategy 1: Standard PDF links (ending with .pdf)
            pdf_link = await page.query_selector("a[href$='.pdf'], a[href*='.pdf']")
            if pdf_link:
                href = await pdf_link.get_attribute('href')
                print(f"Found PDF link: {href}")
                if href:
                    href = href.strip()
                    if href.lower().startswith('http'):
                        source['pdf_url'] = href
                    else:
                        source['pdf_url'] = urljoin(base_url, href)
                    print(f"Set PDF URL: {source['pdf_url']}")
                    return

            # Strategy 2: Links with "PDF" in text (e.g., "Download PDF", "Entscheidung als PDF")
            pdf_text_links = await page.query_selector_all("a:has-text('PDF'), a:has-text('pdf')")
            for link in pdf_text_links:
                href = await link.get_attribute('href')
                if href:
                    href = href.strip()
                    # Resolve relative URLs
                    full_href = href if href.lower().startswith('http') else urljoin(base_url, href)
                    print(f"Found link with 'PDF' in text: {full_href}")
                    source['pdf_url'] = full_href
                    return

            # Strategy 3: Special patterns for known sites
            # nrwe.justiz.nrw.de uses downloadEntscheidung.php
            if 'nrwe.justiz.nrw.de' in current_url or 'justiz.nrw' in current_url:
                download_link = await page.query_selector("a[href*='downloadEntscheidung.php']")
                if download_link:
                    href = await download_link.get_attribute('href')
                    if href:
                        full_href = href if href.lower().startswith('http') else urljoin(base_url, href)
                        print(f"Found NRW Justiz PDF download link: {full_href}")
                        source['pdf_url'] = full_href
                        return

            # Strategy 4: openjur.de - look for NRW Justiz download links
            if 'openjur.de' in current_url:
                # openjur.de pages often link to official court PDFs (e.g., NRW Justiz)
                justiz_link = await page.query_selector("a[href*='justiz.nrw.de'], a[href*='downloadEntscheidung.php']")
                if justiz_link:
                    href = await justiz_link.get_attribute('href')
                    if href:
                        full_href = href if href.lower().startswith('http') else urljoin(base_url, href)
                        print(f"Found NRW Justiz link from openjur.de: {full_href}")
                        source['pdf_url'] = full_href
                        return

                # Fallback: try replacing .html with .pdf
                if current_url.endswith('.html'):
                    potential_pdf_url = current_url.replace('.html', '.pdf')
                    pdf_exists = await detect_pdf_via_http(potential_pdf_url)
                    if pdf_exists:
                        print(f"Found openjur PDF by URL pattern: {potential_pdf_url}")
                        source['pdf_url'] = potential_pdf_url
                        return

            # Strategy 5: Links with download attributes
            download_links = await page.query_selector_all("a[download]")
            for link in download_links:
                href = await link.get_attribute('href')
                if href:
                    full_href = href if href.lower().startswith('http') else urljoin(base_url, href)
                    # Check if it's a PDF
                    if '.pdf' in full_href.lower() or await detect_pdf_via_http(full_href):
                        print(f"Found download link: {full_href}")
                        source['pdf_url'] = full_href
                        return

            print(f"No PDF link found with any strategy on {current_url}")

            # Strategy 6: Check for PDF iframes
            iframe_pdf = await page.query_selector("iframe[src$='.pdf'], iframe[src*='.pdf']")
            if iframe_pdf:
                src = await iframe_pdf.get_attribute('src')
                print(f"Found PDF iframe: {src}")
                if src:
                    src = src.strip()
                    if src.lower().startswith('http'):
                        source['pdf_url'] = src
                    else:
                        source['pdf_url'] = urljoin(base_url, src)
                    print(f"Set PDF URL from iframe: {source['pdf_url']}")
            else:
                print(f"No PDF iframe found on {current_url}")
        except Exception as exc:
            print(f"Error detecting PDF link for {url}: {exc}")
        finally:
            try:
                await page.close()
            except Exception:
                pass
            try:
                await context.close()
            except Exception:
                pass

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                sem = asyncio.Semaphore(max(1, concurrency))

                async def run(source):
                    async with sem:
                        await process_source(browser, source)

                await asyncio.gather(*(run(src) for src in targets))
            finally:
                await browser.close()
    except Exception as exc:
        print(f"Failed to enrich web sources with PDFs: {exc}")


async def search_asyl_net(
    query: str,
    category: Optional[str] = None,
    suggestions: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Search asyl.net Rechtsprechungs-Datenbank using cached Schlagwörter.
    Args:
        query: Search query
        category: Optional category filter (e.g., "Dublin", "EGMR")
        suggestions: Schlagwörter to combine in the search
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching asyl.net: query='{query}', category={category}, suggestions={suggestions}")

        # Prepare candidate keywords using suggestions endpoint
        clean_query = query.replace('"', '').replace("'", "").strip()
        first_term = clean_query.split()[0] if clean_query else ""
        candidate_keywords: List[str] = []

        normalized_set = set()

        if suggestions:
            prepared = [kw.strip() for kw in suggestions if isinstance(kw, str) and kw.strip()]
            unique_prepared = []
            for kw in prepared:
                low = kw.lower()
                if low not in normalized_set:
                    normalized_set.add(low)
                    unique_prepared.append(kw)

            if unique_prepared:
                combined_kw = ",".join(unique_prepared)
                candidate_keywords.append(combined_kw)
                print(f"Combined asyl.net keyword string: '{combined_kw}'")
            else:
                candidate_keywords.append(clean_query or "")

        elif clean_query:
            candidate_keywords.append(clean_query)

        if not candidate_keywords and len(first_term) >= 2:
            fallback_suggestions = (await get_asyl_net_keyword_suggestions(first_term))[:3]
            if fallback_suggestions:
                combined_kw = ",".join(fallback_suggestions)
                candidate_keywords.append(combined_kw)
                print(f"Fallback combined asyl.net keyword string: '{combined_kw}'")

        # Ensure at most one keyword is used per request
        candidate_keywords = [kw for kw in candidate_keywords if kw]
        if candidate_keywords:
            candidate_keywords = [candidate_keywords[0]]

        if not candidate_keywords:
            print("asyl.net: no candidate keywords generated")
            return []


        results: List[Dict[str, str]] = []
        seen_urls = set()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                for idx, keyword in enumerate(candidate_keywords[:3]):
                    encoded_keywords = quote_plus(keyword)
                    search_url = (
                        f"{ASYL_NET_BASE_URL}{ASYL_NET_SEARCH_PATH}"
                        f"?newsearch=1&keywords={encoded_keywords}&keywordConjunction=1&limit=25"
                    )

                    print(f"Fetching asyl.net results for keyword '{keyword}' (rank {idx})")
                    await page.goto(search_url, wait_until="networkidle", timeout=30000)

                    # Dismiss cookie banner if present
                    try:
                        cookie_button = await page.query_selector(
                            "button#CybotCookiebotDialogBodyLevelButtonAccept, button:has-text('Akzeptieren')"
                        )
                        if cookie_button:
                            await cookie_button.click()
                            await page.wait_for_timeout(500)
                    except Exception:
                        pass

                    items = await page.query_selector_all("div.rsdb_listitem")

                    for item in items[:5]:
                        link_elem = await item.query_selector("div.rsdb_listitem_court a")
                        if not link_elem:
                            continue

                        url = await link_elem.get_attribute("href")
                        if not url:
                            continue

                        if not url.startswith("http"):
                            url = f"{ASYL_NET_BASE_URL}{url}"

                        if url in seen_urls:
                            continue

                        seen_urls.add(url)

                        pdf_url = None
                        detail_page = None
                        try:
                            detail_page = await browser.new_page()
                            await detail_page.goto(url, wait_until="domcontentloaded", timeout=30000)
                            pdf_link_elem = await detail_page.query_selector("a[href$='.pdf'], a[href*='.pdf']")
                            if pdf_link_elem:
                                href = await pdf_link_elem.get_attribute("href")
                                if href:
                                    pdf_url = urljoin(url, href)
                                    print(f"Found PDF link for asyl.net result using '{keyword}': {pdf_url}")
                        except Exception as detail_error:
                            print(f"Could not inspect detail page for {url}: {detail_error}")
                        finally:
                            if detail_page:
                                await detail_page.close()

                        court_elem = await item.query_selector(".rsdb_listitem_court .courttitle")
                        headnote_elem = await item.query_selector(".rsdb_listitem_court .headnote")
                        footer_elem = await item.query_selector(".rsdb_listitem_footer")

                        def clean_text(text: Optional[str]) -> str:
                            if not text:
                                return ""
                            return re.sub(r"\s+", " ", text).strip()

                        court_text = clean_text(await court_elem.text_content() if court_elem else "")
                        headnote_text = clean_text(await headnote_elem.text_content() if headnote_elem else "")
                        footer_text = clean_text(await footer_elem.text_content() if footer_elem else "")

                        title_parts = [part for part in [court_text, headnote_text] if part]
                        title = " – ".join(title_parts) if title_parts else f"asyl.net Ergebnis zu {keyword}"

                        description_parts = [headnote_text, footer_text]
                        description = " ".join(part for part in description_parts if part)
                        if not description:
                            description = "Rechtsprechungsfundstelle aus der asyl.net Entscheidungsdatenbank."

                        results.append({
                            "title": title,
                            "url": url,
                            "description": description,
                            "pdf_url": pdf_url,
                            "search_keyword": keyword,
                            "source": "asyl.net",
                            "suggestions": ",".join(suggestions) if suggestions else keyword,
                        })

                if results:
                    print(f"asyl.net returned {len(results)} direct results across {len(candidate_keywords[:3])} keyword variants")
                else:
                    print("asyl.net returned no direct results")

                return results

            finally:
                await browser.close()

    except Exception as e:
        print(f"Error searching asyl.net: {e}")
        traceback.print_exc()
        return []

async def search_bamf(query: str, topic: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search BAMF website using Playwright scraping.
    Args:
        query: Search query
        topic: Optional topic filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching BAMF: query='{query}', topic={topic}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to BAMF advanced search
            search_url = f"https://www.bamf.de/EN/Service/Suche/suche_node.html?query={query}"
            if topic:
                search_url += f"&topic={topic}"

            await page.goto(search_url, wait_until="networkidle", timeout=30000)

            # Extract search results
            sources = []
            all_results = await page.query_selector_all(".search-result, .result-item")
            results = all_results[:5]

            for result in results:
                title_elem = await result.query_selector("h3, .title, a")
                link_elem = await result.query_selector("a")

                if title_elem and link_elem:
                    title = await title_elem.text_content()
                    url = await link_elem.get_attribute("href")

                    if url and not url.startswith("http"):
                        url = f"https://www.bamf.de{url}"

                    sources.append({"title": title.strip(), "url": url})

            await browser.close()
            print(f"BAMF returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching BAMF: {e}")
        return []

async def search_edal(query: str, country: Optional[str] = None, court: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search EDAL (European Database of Asylum Law) using Playwright scraping.
    Args:
        query: Search query
        country: Optional country filter
        court: Optional court filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching EDAL: query='{query}', country={country}, court={court}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to EDAL case law search
            search_url = f"https://www.asylumlawdatabase.eu/en/case-law-search?search={query}"
            if country:
                search_url += f"&country={country}"
            if court:
                search_url += f"&court={court}"

            await page.goto(search_url, wait_until="networkidle", timeout=30000)

            # Extract search results
            sources = []
            all_results = await page.query_selector_all(".case-item, .result")
            results = all_results[:5]

            for result in results:
                title_elem = await result.query_selector("h3, .title, a")
                link_elem = await result.query_selector("a")

                if title_elem and link_elem:
                    title = await title_elem.text_content()
                    url = await link_elem.get_attribute("href")

                    if url and not url.startswith("http"):
                        url = f"https://www.asylumlawdatabase.eu{url}"

                    sources.append({"title": title.strip(), "url": url})

            await browser.close()
            print(f"EDAL returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching EDAL: {e}")
        return []

# ===== END LEGAL DATABASE SEARCH TOOLS =====

async def research_with_legal_databases(query: str) -> ResearchResult:
    """
    Perform targeted legal database research using Gemini with function calling.
    Gemini intelligently selects which databases to query and crafts appropriate parameters.
    """
    try:
        print(f"Starting legal database research for query: {query}")
        client = get_gemini_client()

        # Define function declarations for Gemini (asyl.net only)
        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_asyl_net",
                        description="Durchsuche die asyl.net Rechtsprechungsdatenbank. Übergib deine Schlagwörter im Feld 'suggestions' (Liste von Strings).",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(type=types.Type.STRING, description="Suchanfrage oder Kontextbeschreibung"),
                                "category": types.Schema(type=types.Type.STRING, description="(Optional) Kategorie, z. B. 'Dublin' oder 'EGMR'"),
                                "suggestions": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(type=types.Type.STRING),
                                    description="Liste von 1-3 Schlagwörtern aus der Liste, die kombiniert durchsucht werden sollen"
                                )
                            },
                            required=["query"]
                        )
                    )
                ]
            )
        ]

        suggestion_text = "\n".join(f"- {s}" for s in ASYL_NET_ALL_SUGGESTIONS) if ASYL_NET_ALL_SUGGESTIONS else "- (keine Schlagwörter geladen)"

        # Build the research prompt
        prompt = f"""Du bist ein Rechercheassistent für deutsches Asylrecht mit Zugriff auf spezialisierte Rechtsdatenbanken.

Analysiere folgende Anfrage und fokussiere dich ausschließlich auf die asyl.net Rechtsprechungsdatenbank:

{query}

WICHTIG:
- Formuliere präzise Suchbegriffe für asyl.net.
- Extrahiere relevante Parameter wie Länder, Gerichte, Kategorien aus der Anfrage.
- Nutze die Schlagwort-Liste, um 1-3 passende Begriffe auszuwählen und übergib sie im Feld 'suggestions' (Liste von Strings).
- Führe genau EINEN Funktionsaufruf zu search_asyl_net aus. Keine weiteren Funktionsaufrufe tätigen.

Schlagwort-Liste (bitte exakt übernehmen):
{suggestion_text}

Antwortformat:
Gib ausschließlich gültige JSON im Format {"summary_markdown": "...", "asyl_suggestions": ["..."]} zurück (keine zusätzlichen Erklärungen).

Führe die Suche aus und liefere anschließend ausschließlich die Quellen; keine Zusammenfassung generieren."""

        # Make the API call with function calling
        print("Calling Gemini API with function calling...")
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=tools,
                temperature=0.0
            )
        )

        # Handle function calls
        all_sources = []
        function_calls_made = []

        asyl_net_processed = False

        if hasattr(response.candidates[0].content, 'parts'):
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    function_name = fc.name
                    function_args = dict(fc.args)

                    print(f"Gemini called function: {function_name} with args: {function_args}")
                    function_calls_made.append(f"{function_name}({function_args})")

                    # Execute the function call
                    if function_name == "search_asyl_net":
                        if asyl_net_processed:
                            print("search_asyl_net was already executed once; ignoring additional call.")
                            continue
                        sources = await search_asyl_net(**function_args)
                        all_sources.extend(sources)
                        asyl_net_processed = True
                    else:
                        print(f"Unknown function call ignored: {function_name}")

        print(f"Legal databases returned {len(all_sources)} total sources from {len(function_calls_made)} database(s)")
        summary = ""

        return ResearchResult(
            query=query,
            summary=summary,
            sources=all_sources
        )

    except Exception as e:
        print(f"ERROR in research_with_legal_databases: {e}")
        print(traceback.format_exc())
        raise Exception(f"Legal database research failed: {e}")

async def download_and_update_source(source_id: str, url: str, title: str):
    """Background task to download a source and update its status"""
    db = SessionLocal()
    try:
        # Update status to downloading
        source_uuid = uuid.UUID(source_id)
        source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
        if source:
            source.download_status = 'downloading'
            db.commit()
            broadcast_sources_snapshot(db, 'download_started', {'source_id': source_id})

        # Download the PDF
        download_path = await download_source_as_pdf(url, title)

        # Update status
        source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
        if source:
            if download_path:
                source.download_status = 'completed'
                source.download_path = download_path
            else:
                source.download_status = 'failed'
            db.commit()
            broadcast_sources_snapshot(db, 'download_completed' if download_path else 'download_failed', {'source_id': source_id})

    except Exception as e:
        print(f"Error in background download for {url}: {e}")
        # Mark as failed
        try:
            source_uuid = uuid.UUID(source_id)
            source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
            if source:
                source.download_status = 'failed'
                db.commit()
                broadcast_sources_snapshot(db, 'download_failed', {'source_id': source_id})
        except Exception:
            pass
    finally:
        db.close()


@router.post("/sources", response_model=SavedSource)
@limiter.limit("100/hour")
async def add_source_endpoint(request: Request, body: AddSourceRequest, db: Session = Depends(get_db)):
    """Manually add a research source and optionally download its PDF."""
    source_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    new_source = ResearchSource(
        id=uuid.UUID(source_id),
        url=body.url,
        title=body.title,
        description=body.description,
        document_type=body.document_type,
        pdf_url=body.pdf_url,
        download_status="pending" if body.auto_download else "skipped",
        research_query=body.research_query or "Manuell hinzugefügt",
    )
    db.add(new_source)
    db.commit()
    db.refresh(new_source)

    broadcast_sources_snapshot(db, "add", {"source_id": source_id})

    saved_source = SavedSource(
        id=source_id,
        url=body.url,
        title=body.title,
        description=body.description,
        document_type=body.document_type,
        pdf_url=body.pdf_url,
        download_status="pending" if body.auto_download else "skipped",
        research_query=body.research_query or "Manuell hinzugefügt",
        timestamp=timestamp,
    )

    if body.auto_download:
        download_target = body.pdf_url or body.url
        asyncio.create_task(download_and_update_source(source_id, download_target, body.title))

    return saved_source


@router.get("/sources")
@limiter.limit("1000/hour")
async def get_sources(request: Request, db: Session = Depends(get_db)):
    """Get all saved research sources."""
    sources = db.query(ResearchSource).order_by(desc(ResearchSource.created_at)).all()
    sources_dict = [s.to_dict() for s in sources]
    return JSONResponse(
        content={
            "count": len(sources_dict),
            "sources": sources_dict,
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/sources/download/{source_id}")
@limiter.limit("50/hour")
async def download_source_file(request: Request, source_id: str, db: Session = Depends(get_db)):
    """Download a saved source PDF."""
    try:
        source_uuid = uuid.UUID(source_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source ID format")

    source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    if not source.download_path:
        raise HTTPException(status_code=404, detail="Source not downloaded yet")

    download_path = Path(source.download_path)
    if not download_path.exists():
        raise HTTPException(status_code=404, detail="Downloaded file not found")

    return FileResponse(
        path=download_path,
        media_type="application/pdf",
        filename=f"{source.title}.pdf",
    )


@router.delete("/sources/{source_id}")
@limiter.limit("100/hour")
async def delete_source_endpoint(request: Request, source_id: str, db: Session = Depends(get_db)):
    """Delete a saved source."""
    try:
        source_uuid = uuid.UUID(source_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source ID format")

    source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
    if not source:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

    if source.download_path:
        download_path = Path(source.download_path)
        if download_path.exists():
            try:
                download_path.unlink()
            except Exception as exc:
                print(f"Error deleting file {download_path}: {exc}")

    db.delete(source)
    db.commit()
    broadcast_sources_snapshot(db, "delete", {"source_id": source_id})
    return {"message": f"Source {source_id} deleted successfully"}


@router.delete("/sources")
@limiter.limit("50/hour")
async def delete_all_sources_endpoint(request: Request, db: Session = Depends(get_db)):
    """Delete all saved sources."""
    sources = db.query(ResearchSource).all()

    if not sources:
        return {"message": "No sources to delete", "count": 0}

    deleted_count = 0
    for source in sources:
        if source.download_path:
            download_path = Path(source.download_path)
            if download_path.exists():
                try:
                    download_path.unlink()
                    deleted_count += 1
                except Exception as exc:
                    print(f"Error deleting file {download_path}: {exc}")

    sources_count = len(sources)
    db.query(ResearchSource).delete()
    db.commit()

    broadcast_sources_snapshot(db, "delete_all", {"count": sources_count})
    return {
        "message": "All sources deleted successfully",
        "count": sources_count,
        "files_deleted": deleted_count,
    }

@router.post("/debug-research")
async def debug_research(body: ResearchRequest):
    """Debug endpoint to inspect Gemini grounding metadata structure."""
    try:
        client = get_gemini_client()
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        debug_query = (body.query or "Automatische Bescheid-Recherche")

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"Recherchiere Quellen für: {debug_query}",
            config=types.GenerateContentConfig(tools=[grounding_tool], temperature=0.0)
        )

        debug_info = {
            "text": response.text,
            "candidates": []
        }

        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                cand_info = {"content_parts": []}

                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        cand_info["content_parts"].append(str(part))

                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    gm = candidate.grounding_metadata
                    cand_info["grounding_metadata"] = {
                        "chunks_count": len(gm.grounding_chunks) if hasattr(gm, 'grounding_chunks') else 0,
                        "chunks": []
                    }

                    if hasattr(gm, 'grounding_chunks'):
                        for chunk in gm.grounding_chunks[:3]:
                            chunk_info = {"type": str(type(chunk))}

                            if hasattr(chunk, 'web'):
                                web_attrs = dir(chunk.web)
                                chunk_info["web_attrs"] = [a for a in web_attrs if not a.startswith('_')]
                                chunk_info["web_data"] = {}

                                for attr in ['uri', 'title', 'snippet', 'text']:
                                    if hasattr(chunk.web, attr):
                                        val = getattr(chunk.web, attr)
                                        chunk_info["web_data"][attr] = str(val)[:200] if val else None

                            chunk_attrs = dir(chunk)
                            chunk_info["chunk_attrs"] = [a for a in chunk_attrs if not a.startswith('_')]

                            cand_info["grounding_metadata"]["chunks"].append(chunk_info)

                debug_info["candidates"].append(cand_info)

        return debug_info

    except Exception as exc:  # pragma: no cover - diagnostic endpoint
        return {"error": str(exc), "traceback": traceback.format_exc()}


@router.post("/debug-research")
async def debug_research(body: ResearchRequest):
    """Debug endpoint to inspect Gemini grounding metadata structure."""
    try:
        client = get_gemini_client()
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        debug_query = (body.query or "Automatische Bescheid-Recherche")

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"Recherchiere Quellen für: {debug_query}",
            config=types.GenerateContentConfig(tools=[grounding_tool], temperature=0.0)
        )

        debug_info = {
            "text": response.text,
            "candidates": []
        }

        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                cand_info = {"content_parts": []}

                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        cand_info["content_parts"].append(str(part))

                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    gm = candidate.grounding_metadata
                    cand_info["grounding_metadata"] = {
                        "chunks_count": len(gm.grounding_chunks) if hasattr(gm, 'grounding_chunks') else 0,
                        "chunks": []
                    }

                    if hasattr(gm, 'grounding_chunks'):
                        for chunk in gm.grounding_chunks[:3]:
                            chunk_info = {"type": str(type(chunk))}

                            if hasattr(chunk, 'web'):
                                web_attrs = dir(chunk.web)
                                chunk_info["web_attrs"] = [a for a in web_attrs if not a.startswith('_')]
                                chunk_info["web_data"] = {}

                                for attr in ['uri', 'title', 'snippet', 'text']:
                                    if hasattr(chunk.web, attr):
                                        val = getattr(chunk.web, attr)
                                        chunk_info["web_data"][attr] = str(val)[:200] if val else None

                            chunk_attrs = dir(chunk)
                            chunk_info["chunk_attrs"] = [a for a in chunk_attrs if not a.startswith('_')]

                            cand_info["grounding_metadata"]["chunks"].append(chunk_info)

                debug_info["candidates"].append(cand_info)

        return debug_info

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


async def summarize_source(url: str, title: str, client) -> str:
    """
    Generate a brief summary/description of a source using Gemini.
    Returns a 1-2 sentence description of what the source contains.
    """
    prompt = f"""Erstelle eine sehr kurze Zusammenfassung (1-2 Sätze) dieser Quelle für deutsches Asylrecht:

URL: {url}
Titel: {title}

Beschreibe kurz und prägnant, welche Informationen diese Quelle enthält und warum sie relevant sein könnte. Verwende Deutsch."""

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        return response.text.strip() if response.text else "Quelle gefunden"
    except Exception as e:
        print(f"Error summarizing source {url}: {e}")
        return "Relevante Quelle gefunden"

async def download_source_as_pdf(url: str, filename: str) -> Optional[str]:
    """
    Download a source as PDF. Handles both direct PDFs and HTML pages.
    Returns the path to the downloaded file, or None if failed.
    """

    # Ensure downloads directory exists
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()
    file_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    output_path = DOWNLOADS_DIR / f"{file_hash}_{safe_filename}.pdf"

    try:
        # Check if URL is already a PDF
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            headers={"User-Agent": ASYL_NET_USER_AGENT}
        ) as http_client:
            direct_pdf_url: Optional[str] = None

            try:
                head_response = await http_client.head(url)
                if _looks_like_pdf(head_response.headers):
                    direct_pdf_url = str(head_response.url)
            except httpx.HTTPError as err:
                status = getattr(getattr(err, "response", None), "status_code", None)
                if status not in {401, 403, 404, 405, 406, 409, 410}:
                    print(f"HEAD download probe failed for {url}: {err}")

            if not direct_pdf_url:
                try:
                    probe_response = await http_client.get(
                        url,
                        headers={"Range": "bytes=0-0"}
                    )
                    if _looks_like_pdf(probe_response.headers):
                        direct_pdf_url = str(probe_response.url)
                except httpx.HTTPError as err:
                    status = getattr(getattr(err, "response", None), "status_code", None)
                    if status not in {401, 403, 404, 405, 406, 409, 410, 416}:
                        print(f"GET download probe failed for {url}: {err}")

            if direct_pdf_url:
                print(f"Downloading PDF directly: {direct_pdf_url}")
                response = await http_client.get(direct_pdf_url, timeout=60.0)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"PDF downloaded to: {output_path}")
                return str(output_path)

        # HTML page - convert to PDF using Playwright
        print(f"Converting HTML to PDF: {url}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await page.pdf(path=str(output_path), format='A4')
            await browser.close()

        print(f"HTML converted to PDF: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


@router.post("/research", response_model=ResearchResult)
@limiter.limit("10/hour")
async def research(request: Request, body: ResearchRequest, db: Session = Depends(get_db)):
    """Perform web research using Gemini or Grok-4-Fast with specialized legal databases"""
    try:
        print(f"[RESEARCH] Search engine: {body.search_engine}")
        raw_query = (body.query or "").strip()
        attachment_path: Optional[str] = None
        attachment_label: Optional[str] = None
        attachment_text_path: Optional[str] = None
        attachment_ocr_text: Optional[str] = None
        classification_hint: Optional[str] = None

        if not raw_query:
            if not body.primary_bescheid:
                raise HTTPException(
                    status_code=400,
                    detail="Bitte geben Sie eine Rechercheanfrage ein oder wählen Sie einen Hauptbescheid aus."
                )

            bescheid = db.query(Document).filter(Document.filename == body.primary_bescheid).first()
            if not bescheid:
                raise HTTPException(status_code=404, detail=f"Bescheid '{body.primary_bescheid}' wurde nicht gefunden.")
            if bescheid.category != DocumentCategory.BESCHEID.value:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dokument '{bescheid.filename}' ist kein Bescheid und kann nicht für die automatische Recherche verwendet werden."
                )
            if not bescheid.file_path or not os.path.exists(bescheid.file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF-Datei für '{bescheid.filename}' wurde nicht auf dem Server gefunden."
                )

            attachment_label = bescheid.filename
            classification_hint = (bescheid.explanation or "").strip() or None

            attachment_text_path = bescheid.extracted_text_path if bescheid.extracted_text_path and os.path.exists(bescheid.extracted_text_path) else None

            if attachment_text_path:
                print(f"[INFO] Using OCR text file for research: {attachment_text_path}")
                attachment_path = None
                attachment_ocr_text = None  # Let research function read from disk
            else:
                print(f"[INFO] No OCR text file available, will use PDF")
                attachment_path = bescheid.file_path
                attachment_ocr_text = None

            derived_parts = [
                "Automatische Recherche basierend auf dem beigefügten BAMF-Bescheid.",
                f"Dateiname: {bescheid.filename}"
            ]
            if classification_hint:
                derived_parts.append(f"Kurze Einordnung laut Klassifikation: {classification_hint}")
            raw_query = "\n".join(derived_parts)

        print(f"Starting research pipeline for query: {raw_query}")

        # Route to appropriate search engine
        if body.search_engine == "grok-4-fast":
            print("[RESEARCH] Using Grok-4-Fast (Responses API with web_search tool)")
            web_result = await research_with_grok(
                raw_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path
            )
        else:
            print("[RESEARCH] Using Gemini with Google Search")
            web_result = await research_with_gemini(
                raw_query,
                attachment_path=attachment_path,
                attachment_display_name=attachment_label,
                attachment_ocr_text=attachment_ocr_text,
                attachment_text_path=attachment_text_path
            )
        all_sources: List[Dict[str, str]] = list(web_result.sources)
        summaries = [web_result.summary] if web_result.summary else []

        asyl_sources: List[Dict[str, str]] = []
        try:
            suggestions = web_result.suggestions or []
            print(f"Using asyl.net suggestions from web search: {suggestions}")

            asyl_query = (body.query or "").strip()
            if not asyl_query:
                asyl_query = classification_hint or attachment_label or raw_query

            asyl_sources = await search_asyl_net(asyl_query, suggestions=suggestions or None)
            all_sources.extend(asyl_sources)
        except Exception as e:
            print(f"asyl.net search failed: {e}")

        combined_summary = "<hr/>".join(summaries) if summaries else ""

        print(f"Combined research returned {len(all_sources)} total sources")

        display_query = (body.query or "").strip()
        if not display_query:
            display_query = f"Automatische Recherche zu Bescheid: {attachment_label}"

        return ResearchResult(
            query=display_query,
            summary=combined_summary,
            sources=all_sources,
            suggestions=web_result.suggestions
        )
    except Exception as e:
        print(f"ERROR in research endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Research failed: {e}")

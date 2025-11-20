"""
Gemini research with Google Search grounding.
"""

import asyncio
import json
import os
import re
import traceback
from typing import Dict, List, Optional

import markdown
from google.genai import errors as genai_errors, types

from shared import ResearchResult, get_gemini_client, get_document_for_upload
from .utils import enrich_web_sources_with_pdf


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

        # Configure Google Search grounding tool (retrieval not supported on this endpoint)
        grounding_tools = [types.Tool(google_search=types.GoogleSearch())]

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

Führe zwingend eine Websuche durch und binde mindestens 5 passende Quellen (mit URL) aus den oben genannten Kategorien ein. Antworte nicht ohne Suchergebnisse.

VERMEIDE:
- Blogs und persönliche Meinungen
- Journalistische Artikel oder Presseportale
- Kommerzielle Beratungsseiten
- Nicht-verifizierte Quellen
- asyl.net (wird separat recherchiert)

Gib eine kurze Übersicht (2-3 Sätze) der wichtigsten Erkenntnisse. Erwähne die Quellen nur kurz im Text (z.B. "laut Bundesverwaltungsgericht" oder "BAMF-Bericht vom ..."), aber füge keine URLs oder vollständige Quellenangaben hinzu - diese werden separat angezeigt."""

        async def call_summary(tools):
            contents = [prompt_summary, uploaded_file] if uploaded_file else [prompt_summary]
            return await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash-preview-09-2025",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools if tools else None,
                    temperature=0.0
                )
            )

        print("Calling Gemini API for summary...")
        try:
            response_summary = await call_summary(grounding_tools)
            print("Gemini call successful (with google_search)")
        except genai_errors.ClientError as primary_err:
            # Some deployments reject google_search_retrieval – fall back gracefully
            print(f"Gemini call failed with tools ({primary_err}), retrying without tools...")
            response_summary = await call_summary([])
            print("Gemini call successful (fallback without search tool)")

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

        # Extract sources (grounding metadata + citation metadata fallback)
        sources = []
        if hasattr(response_summary, 'candidates') and response_summary.candidates:
            candidate = response_summary.candidates[0]
            grounding_meta = getattr(candidate, 'grounding_metadata', None)
            chunk_descriptions = {}
            grounding_chunks = []
            supports = []
            web_queries = []

            if grounding_meta:
                grounding_chunks = list(getattr(grounding_meta, 'grounding_chunks', []) or [])
                supports = list(getattr(grounding_meta, 'grounding_supports', []) or [])
                web_queries = list(getattr(grounding_meta, 'web_search_queries', []) or [])

                try:
                    # Temporary verbose logging to inspect grounding responses
                    import json
                    meta_dict = grounding_meta.to_dict() if hasattr(grounding_meta, 'to_dict') else grounding_meta
                    print("DEBUG: Grounding metadata (raw):", json.dumps(meta_dict, ensure_ascii=False)[:4000])
                except Exception as log_exc:
                    print(f"DEBUG: Failed to serialize grounding metadata: {log_exc}")

                # Map grounding chunk index -> support text
                for support in supports:
                    seg = getattr(support, 'segment', None)
                    text = getattr(seg, 'text', None) if seg else None
                    idxs = getattr(support, 'grounding_chunk_indices', None) or []
                    if text:
                        for idx in idxs:
                            chunk_descriptions[idx] = text

            print(f"DEBUG: Grounding chunks: {len(grounding_chunks)}, supports: {len(supports)}, web_queries: {len(web_queries)}")

            for idx, chunk in enumerate(grounding_chunks):
                web = getattr(chunk, 'web', None)
                url = getattr(web, 'uri', None) if web else None
                if not url:
                    continue

                title = getattr(web, 'title', None) or url
                description = chunk_descriptions.get(idx)

                # fallback: pull text from retrieved_context when available
                if not description:
                    rc = getattr(chunk, 'retrieved_context', None)
                    if rc and getattr(rc, 'text', None):
                        description = rc.text[:500]

                lowered = url.lower()
                pdf_url = url if lowered.endswith('.pdf') or '.pdf?' in lowered else None

                sources.append({
                    "title": title,
                    "url": url,
                    "description": description or "Relevante Quelle aus Web-Recherche",
                    "pdf_url": pdf_url,
                    "source": "Gemini"
                })

            # Fallback: use citation metadata when no grounding chunks are present
            if not sources:
                citation_meta = getattr(candidate, 'citation_metadata', None)
                citations = list(getattr(citation_meta, 'citations', []) or [])
                print(f"DEBUG: Using citation metadata fallback, citations: {len(citations)}")
                for cit in citations:
                    url = getattr(cit, 'uri', None) or ""
                    title = getattr(cit, 'title', None) or (url if url else "Quelle")
                    if not url and not title:
                        continue
                    sources.append({
                        "title": title,
                        "url": url,
                        "description": "Quelle aus Gemini-Citations",
                        "pdf_url": url if url.lower().endswith(".pdf") else None,
                        "source": "Gemini"
                    })

        # Deduplicate by URL
        deduped = []
        seen_urls = set()
        for src in sources:
            url = (src.get("url") or "").strip()
            key = url or (src.get("title") or "")
            if not key:
                continue
            if key in seen_urls:
                continue
            seen_urls.add(key)
            deduped.append(src)
        # Normalize and ensure pdf_url is always a string for validation
        sources = []
        for src in deduped:
            pdf_url = src.get("pdf_url") or ""
            src["pdf_url"] = pdf_url
            sources.append(src)

        await enrich_web_sources_with_pdf(sources)

        print(f"Extracted {len(sources)} sources from Gemini grounding/citations")

        # NOTE: asyl.net keywords are generated in the asylnet module now
        return ResearchResult(
            query=query,
            summary=summary_html,
            sources=sources,
            suggestions=[]  # Keywords generated by asylnet module
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

"""
Gemini research with Google Search grounding.
"""

import asyncio
import os
import traceback
from typing import Dict, List, Optional

import markdown
from google.genai import types

from shared import ResearchResult, get_gemini_client, get_document_for_upload
from ..segmentation import chunk_pdf_for_upload
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
        uploaded_files = []
        temp_files_to_cleanup = []

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
                if needs_cleanup:
                    temp_files_to_cleanup.append(file_path)

                if mime_type == "text/plain":
                    print(f"[INFO] Using OCR text for research: {attachment_label}")
                    # Text files are small, just upload
                    with open(file_path, "rb") as file_handle:
                        uploaded_file = client.files.upload(
                            file=file_handle,
                            config={
                                "mime_type": mime_type,
                                "display_name": f"{attachment_label}.txt"
                            }
                        )
                    uploaded_files.append(uploaded_file)
                    uploaded_name = uploaded_file.name # Keep track for cleanup (last one)
                    print(f"Attachment uploaded successfully for research: {attachment_label} ({mime_type})")

                elif mime_type == "application/pdf":
                    # PDF - check if chunking is needed
                    print(f"[INFO] Checking PDF size for chunking: {file_path}")
                    chunks = chunk_pdf_for_upload(file_path)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_path = chunk.path
                        if chunk_path != file_path:
                            temp_files_to_cleanup.append(chunk_path)
                        
                        chunk_label = f"{attachment_label}_part{i+1}"
                        print(f"[INFO] Uploading chunk {i+1}/{len(chunks)}: {chunk_label}")
                        
                        with open(chunk_path, "rb") as file_handle:
                            uploaded_file = client.files.upload(
                                file=file_handle,
                                config={
                                    "mime_type": "application/pdf",
                                    "display_name": chunk_label
                                }
                            )
                        uploaded_files.append(uploaded_file)
                        print(f"Chunk {i+1} uploaded: {uploaded_file.name}")

            except Exception as exc:
                print(f"[ERROR] Attachment upload failed: {exc}")
                # Cleanup handled in finally
                raise

        # Configure Google Search grounding tool
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

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

        async def call_summary():
            contents = [prompt_summary]
            if uploaded_files:
                contents.extend(uploaded_files)
            
            return await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash-preview-09-2025",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=0.0
                )
            )

        print("Calling Gemini API for summary...")
        response_summary = await call_summary()
        print("Gemini call successful")

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
        # Cleanup uploaded files from Gemini
        if uploaded_files:
            cleanup_client = get_gemini_client()
            for up_file in uploaded_files:
                try:
                    cleanup_client.files.delete(name=up_file.name)
                    print(f"Deleted uploaded attachment from Gemini: {up_file.name}")
                except Exception as cleanup_exc:
                    print(f"Failed to delete uploaded attachment {up_file.name}: {cleanup_exc}")

        # Clean up temporary files (chunks and OCR text)
        for temp_path in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"Deleted temporary file: {temp_path}")
            except Exception as cleanup_exc:
                print(f"Failed to delete temporary file {temp_path}: {cleanup_exc}")

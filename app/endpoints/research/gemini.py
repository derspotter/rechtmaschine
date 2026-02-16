import asyncio
import os
import re
import traceback
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

import markdown
from google.genai import types

from shared import ResearchResult, get_gemini_client, get_document_for_upload, ensure_document_on_gemini
from ..segmentation import chunk_pdf_for_upload
from .utils import enrich_web_sources_with_pdf
from models import Document
from database import SessionLocal
from .prompting import build_research_priority_prompt

DECISION_KEYWORDS = (
    "entscheidung",
    "entscheidungen",
    "beschluss",
    "urteilstext",
    "urteil",
    "aktenzeichen",
    "aktenzeichen:",
    "revision",
    "ecli",
    "dokumentationsblatt",
    "gerichtliche",
)

OFFICIAL_SOURCE_HINTS = (
    "bverwg",
    "bverfg",
    "eugh",
    "egmr",
    "bverf",
    "juris",
    "dejure",
    "nrwe.justiz.nrw.de",
    "justiz.nrw.de",
    "berlin.de",
    "nrw",
    "verwaltungsvr",
    "verfassungsg",
    "ecrl",
    ".eu",
    ".gov",
)

OFFTOPIC_SOURCE_HINTS = (
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
    "linkedin",
)

COURT_LEVEL_HINTS = (
    "bverwg",
    "bverfg",
    "bgh",
    "eugh",
    "egmr",
    "ovg",
    "vg",
    "verwaltungsgericht",
    "vg",
)

_RESEARCH_CONTEXT_TOKENS = (
    "ovg",
    "vg",
    "verwaltungsgericht",
    "bverwg",
    "bverfg",
    "bgh",
    "egmr",
    "eugh",
)

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

def _normalize_url_candidate(url: str) -> str:
    return url.strip().rstrip(").,;")


def _normalize_source_entry(
    url: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    normalized_url = (url or "").strip()
    if not normalized_url:
        return None

    source = {
        "url": normalized_url,
        "title": (title or "").strip() or normalized_url,
        "description": (description or "Relevante Quelle aus Web-Recherche").strip(),
        "source": "Gemini",
    }

    lowered = normalized_url.lower()
    if lowered.endswith(".pdf") or ".pdf?" in lowered:
        source["pdf_url"] = normalized_url

    return source


def _is_official_source(url: str, title: str, description: str) -> bool:
    blob = " ".join([url, title, description]).lower()
    return any(token in blob for token in OFFICIAL_SOURCE_HINTS)


def _contains_offtopic_signal(url: str, title: str, description: str) -> bool:
    blob = " ".join([url, title, description]).lower()
    return any(token in blob for token in OFFTOPIC_SOURCE_HINTS)


def _contains_court_signal(url: str, title: str, description: str) -> int:
    blob = f"{url} {title} {description}".lower()
    score = 0
    for token in COURT_LEVEL_HINTS:
        if token in blob:
            score += 12
    return score


def _contains_decision_signal(url: str, title: str, description: str) -> int:
    blob = f"{url} {title} {description}".lower()
    score = 0
    for token in DECISION_KEYWORDS:
        if token in blob:
            score += 9
    if "/entscheid" in blob or "/urteil" in blob or "entscheid" in blob:
        score += 8
    return score


def _extract_context_terms(context_hints: Optional[List[str]]) -> List[str]:
    if not context_hints:
        return []

    terms: List[str] = []
    seen = set()
    for hint in context_hints:
        normalized = (str(hint).strip() or "").lower()
        if not normalized:
            continue
        if normalized not in seen:
            terms.append(normalized)
            seen.add(normalized)

        # Include important court tokens separately for stronger partial matching.
        for token in re.findall(r"\b[0-9a-zäöüß./-]+\b", normalized):
            if token in seen:
                continue
            if len(token) >= 4 and (token in _RESEARCH_CONTEXT_TOKENS or token.count("/") >= 1):
                terms.append(token)
                seen.add(token)

        for token in _RESEARCH_CONTEXT_TOKENS:
            if token in normalized and token not in seen:
                terms.append(token)
                seen.add(token)

    return terms


def _contains_context_signal(blob: str, context_hints: Optional[List[str]]) -> int:
    if not context_hints:
        return 0
    lowered = blob.lower()
    score = 0
    seen = set()
    for term in _extract_context_terms(context_hints):
        if term in seen:
            continue
        if term in lowered:
            score += 30
            seen.add(term)
    return score


def _score_for_rechtsprechung_priority(
    source: Dict[str, str],
    context_hints: Optional[List[str]] = None,
) -> int:
    url = (source.get("url") or "").lower()
    title = (source.get("title") or "").lower()
    description = (source.get("description") or "").lower()

    score = 0

    # Prioritize decisions, then official court sources, then recent years.
    if _is_official_source(url, title, description):
        score += 70

    decision_score = _contains_decision_signal(url, title, description)
    if decision_score >= 1:
        score += 140 + (decision_score * 6)
    else:
        # Deprioritize generic pages that don't carry clear decision signals.
        score -= 100

    score += _contains_court_signal(url, title, description)
    score += decision_score * 2
    score += _contains_context_signal(
        f"{url} {title} {description}",
        context_hints,
    )

    if _contains_offtopic_signal(url, title, description):
        score -= 90

    year_matches = [int(v) for v in _YEAR_RE.findall(f"{url} {title} {description}")]
    if year_matches:
        current_year = datetime.utcnow().year
        valid_years = [y for y in year_matches if y <= current_year + 1]
        if not valid_years:
            valid_years = year_matches
        recent = max(valid_years)
        if recent:
            score += max(0, 3 * (recent - 2014))

    if ".pdf" in url:
        score += 6

    return score


def _prioritize_rechtsprechung(
    sources: List[Dict[str, str]],
    context_hints: Optional[List[str]] = None,
    limit: int = 40,
) -> List[Dict[str, str]]:
    if not sources:
        return []

    scored = [
        (source, _score_for_rechtsprechung_priority(source, context_hints=context_hints))
        for source in sources
    ]
    scored.sort(key=lambda item: (item[1], item[0].get("url", "")), reverse=True)

    ranked_sources = [source for source, _ in scored if source.get("url")]
    if not ranked_sources:
        return []

    decision_sources = [
        source
        for source in ranked_sources
        if _contains_decision_signal(
            (source.get("url") or "").lower(),
            (source.get("title") or "").lower(),
            (source.get("description") or "").lower(),
        ) >= 9
    ]
    ranked_sources = (decision_sources + [source for source in ranked_sources if source not in decision_sources])[:limit]

    deduped: List[Dict[str, str]] = []
    seen_urls = set()
    for source in ranked_sources:
        url = source.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(source)

    return deduped


def _extract_gemini_sources(response_summary: Any) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []

    if not response_summary:
        return sources

    # 1) Primary path: grounding metadata chunks
    candidates = getattr(response_summary, "candidates", None)
    if candidates:
        first_candidate = candidates[0]
        grounding_meta = getattr(first_candidate, "grounding_metadata", None)
        grounding_chunks = getattr(grounding_meta, "grounding_chunks", None)
        if grounding_chunks:
            for chunk in grounding_chunks:
                web = getattr(chunk, "web", None)
                description = ""
                if web and getattr(web, "snippet", None):
                    description = str(web.snippet)

                if not description:
                    support = getattr(chunk, "grounding_support", None)
                    segment = getattr(support, "segment", None) if support else None
                    if segment and getattr(segment, "text", None):
                        description = str(segment.text)

                if not description:
                    context = getattr(chunk, "retrieved_context", None)
                    if context and getattr(context, "text", None):
                        description = str(context.text)[:200]

                url = None
                if web and getattr(web, "uri", None):
                    url = str(web.uri)
                elif context := getattr(chunk, "retrieved_context", None):
                    if getattr(context, "uri", None):
                        url = str(context.uri)
                    elif getattr(context, "document_name", None):
                        url = str(context.document_name)

                if not url:
                    continue

                normalized_url = _normalize_url_candidate(url)
                source = _normalize_source_entry(
                    url=normalized_url,
                    title=str(getattr(web, "title", "") or ""),
                    description=description,
                )
                if source:
                    sources.append(source)

    # 2) Fallback path: parse response text for URLs (for SDK/model changes)
    if not sources:
        response_text = str(getattr(response_summary, "text", "") or "")
        if response_text:
            for match in re.finditer(r"https?://[^\s'\")>\],;]+", response_text):
                entry = _normalize_source_entry(url=match.group(0), description="Relevante Quelle aus Web-Recherche")
                if entry:
                    sources.append(entry)

    deduped: List[Dict[str, str]] = []
    seen_urls = set()
    for source in sources:
        url = source.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(source)

    return deduped


def _format_research_context_hints(context_hints: Optional[List[str]]) -> str:
    if not context_hints:
        return ""

    normalized = []
    for idx, hint in enumerate(context_hints[:12], start=1):
        normalized_hint = " ".join(str(hint).split())
        if normalized_hint:
            normalized.append(f"{idx}. {normalized_hint}")

    if not normalized:
        return ""

    return (
        "Bekannte Referenzentscheidungen aus dem Fallkontext:\n"
        + "\n".join(normalized)
        + "\nBitte nutze diese zuerst als harte Vergleichsanker."
    )


async def research_with_gemini(
    query: str,
    attachment_path: Optional[str] = None,
    attachment_display_name: Optional[str] = None,
    attachment_ocr_text: Optional[str] = None,
    attachment_text_path: Optional[str] = None,
    attachment_anonymization_metadata: Optional[dict] = None,
    attachment_is_anonymized: bool = False,
    document_id: Optional[str] = None,
    owner_id: Optional[str] = None,
    case_id: Optional[str] = None,
    research_context_hints: Optional[List[str]] = None,
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
    
    # Track files we explicitly want to clean up (temp files)
    temp_files_to_cleanup = []
    # Track files created here that should be deleted (e.g. ad-hoc uploads)
    files_to_delete_from_gemini = []

    try:
        client = get_gemini_client()
        print("[GEMINI] Gemini client initialized")

        # Upload attachment if provided (OCR text or PDF)
        uploaded_files = []
        
        # 1. Try Shared Logic with Document ID first (Persistence/Reuse)
        if document_id:
            try:
                print(f"[GEMINI] Attempting to reuse/upload Document {document_id}")
                db = SessionLocal()
                try:
                    doc_uuid = uuid.UUID(document_id)
                    q = db.query(Document).filter(Document.id == doc_uuid)
                    if owner_id:
                        try:
                            q = q.filter(Document.owner_id == uuid.UUID(owner_id))
                        except Exception:
                            pass
                    if case_id:
                        try:
                            q = q.filter(Document.case_id == uuid.UUID(case_id))
                        except Exception:
                            pass
                    db_doc = q.first()
                    if db_doc:
                        shared_file = ensure_document_on_gemini(db_doc, db)
                        if shared_file:
                            uploaded_files.append(shared_file)
                            attachment_label = attachment_display_name or db_doc.filename
                            print(f"[GEMINI] Using shared file: {shared_file.name}")
                            # Do NOT add to files_to_delete_from_gemini
                finally:
                    db.close()
            except Exception as e:
                print(f"[WARN] Failed to use shared document logic: {e}")

        # 2. Fallback to ad-hoc upload if no shared file used
        if not uploaded_files and (attachment_ocr_text or attachment_path or attachment_text_path):
            attachment_label = attachment_display_name or "Bescheid"

            # Create a document entry dict for the helper function
            doc_entry = {
                "filename": attachment_label,
                "extracted_text": attachment_ocr_text,
                "ocr_applied": bool(attachment_ocr_text),
                "file_path": attachment_path,
                "extracted_text_path": attachment_text_path,
                "anonymization_metadata": attachment_anonymization_metadata,
                "is_anonymized": attachment_is_anonymized,
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
                    files_to_delete_from_gemini.append(uploaded_file)
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
                        files_to_delete_from_gemini.append(uploaded_file)
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

        # Common search strategy instruction (used for both attachment and generic queries)
        context_block = _format_research_context_hints(research_context_hints)
        context_anchor = f"Zusätzliche harte Suchanker:\n{context_block}" if context_block else ""
        search_strategy = build_research_priority_prompt(
            "Führe eine präzise Web-Recherche zu den Kernthemen der Anfrage durch. "
            "Starte mit der gezielten Suche nach Gerichtsentscheidungen (Urteile/Beschlüsse) "
            "zu konkreten Aktenzeichen und Datumsangaben. "
            "Der Hauptteil der Trefferliste muss aus Entscheidungen bestehen; "
            "ergänzende Sekundärquellen nur, wenn sie für den Kontext zwingend nötig sind.\n"
            + (f"{context_anchor}\n" if context_anchor else "")
        )

        if attachment_label:
            query_block = f"""Analysiere das beigefügte Dokument "{attachment_label}".
Schritt 1: ANALYSE. Identifiziere die zentralen rechtlichen Fragen, Ablehnungsgründe oder Themen (z.B. Dublin, Inlandfluchtalternative, medizinische Abschiebungshindernisse).
Schritt 2: RECHERCHE. Nutze die Ergebnisse aus Schritt 1, um **gemäß der untenstehenden Suchstrategie** nach externen Quellen zu suchen.
{context_anchor}

Zusätzliche Aufgabenstellung / Notiz:
{trimmed_query or "- (keine zusätzliche Notiz)"}"""
        else:
            query_block = f"""Rechercheauftrag:
{trimmed_query or "(Keine Anfrage angegeben)"}

Führe eine umfassende Recherche durch, um die rechtliche Einschätzung zu stützen. Nutze dabei zwingend die **untenstehende Suchstrategie**.
{context_anchor}"""

        prompt_summary = f"""{build_research_priority_prompt("Du bist ein spezialisierter Rechercheassistent für deutsches Asylrecht.")}

{query_block}

{search_strategy}

WICHTIG: Nutze Google Search Grounding, um echte, zitierfähige Primärquellen zu finden.
Nur primäre Urteile/Beschlüsse mit klarem Entscheidungskern sind zulässig. Wenn eine Quelle keine klare Entscheidung enthält, lasse sie weg.

Ergänze die Antwort mit:
- Gericht, Datum und nach Möglichkeit Aktenzeichen.
- Konkrete Verknüpfung zwischen den gefundenen Entscheidungen und der Fragestellung.
- Explizite Kennzeichnung von Abweichungen in der Rechtsprechung.
"""

        async def call_summary():
            contents = [prompt_summary]
            if uploaded_files:
                contents.extend(uploaded_files)
            
            return await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3-flash-preview",
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

        # Extract sources from response metadata with fallback parsing.
        sources = _extract_gemini_sources(response_summary)
        sources = _prioritize_rechtsprechung(
            sources,
            context_hints=research_context_hints,
        )
        for source in sources:
            if source.get("description"):
                print(
                    "DEBUG: Extracted source with description: "
                    f"{source.get('description', 'N/A')[:100]}"
                )

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
        # Cleanup uploaded files from Gemini (ONLY those we explicitly marked for deletion)
        if files_to_delete_from_gemini:
            cleanup_client = get_gemini_client()
            for up_file in files_to_delete_from_gemini:
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

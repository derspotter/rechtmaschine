"""
Rechtmaschine - Document Classifier
Simplified document classification system for German asylum law documents
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
import tempfile
import pikepdf
import markdown
import re
from openai import OpenAI
from enum import Enum
from google import genai
from google.genai import types
import httpx
import anthropic
import uuid
import unicodedata
from urllib.parse import quote_plus, urljoin, urlparse, quote
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session
from sqlalchemy import desc
from fastapi import Depends

# Database imports
from database import get_db, engine, Base
from models import Document, ResearchSource, ProcessedDocument
from kanzlei_gemini import (
    GeminiConfig as SegmentationGeminiConfig,
    segment_pdf_with_gemini,
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])

app = FastAPI(title="Rechtmaschine Document Classifier")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.source_subscribers = set()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Storage directories
DOWNLOADS_DIR = Path("/app/downloaded_sources")
UPLOADS_DIR = Path("/app/uploads")

# j-lawyer configuration
JLAWYER_BASE_URL = os.environ.get("JLAWYER_BASE_URL")
if JLAWYER_BASE_URL:
    JLAWYER_BASE_URL = JLAWYER_BASE_URL.rstrip("/")
JLAWYER_USERNAME = os.environ.get("JLAWYER_USERNAME")
JLAWYER_PASSWORD = os.environ.get("JLAWYER_PASSWORD")
JLAWYER_TEMPLATE_FOLDER_DEFAULT = os.environ.get("JLAWYER_TEMPLATE_FOLDER")
JLAWYER_PLACEHOLDER_KEY = os.environ.get("JLAWYER_PLACEHOLDER_KEY", "HAUPTTEXT")

# Initialize database tables on startup
@app.on_event("startup")
async def startup_event():
    """Create database tables on startup"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")

ASYL_NET_BASE_URL = "https://www.asyl.net"
ASYL_NET_SEARCH_PATH = "/recht/entscheidungsdatenbank"
ASYL_NET_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
ASYL_NET_SUGGESTIONS_FILE = Path(__file__).resolve().parent / "data" / "asyl_net_suggestions.json"
try:
    with open(ASYL_NET_SUGGESTIONS_FILE, "r", encoding="utf-8") as f:
        _asyl_suggestions_payload = json.load(f)
    ASYL_NET_ALL_SUGGESTIONS: List[str] = _asyl_suggestions_payload.get("suggestions", [])
except FileNotFoundError:
    print(f"Warning: asyl.net suggestions file not found at {ASYL_NET_SUGGESTIONS_FILE}")
    ASYL_NET_ALL_SUGGESTIONS = []

# Document categories
class DocumentCategory(str, Enum):
    ANHOERUNG = "Anhörung"  # Hearing protocols
    BESCHEID = "Bescheid"  # Administrative decisions/rulings
    RECHTSPRECHUNG = "Rechtsprechung"  # Case law
    SONSTIGES = "Sonstiges"  # Other
    AKTE = "Akte"  # Complete BAMF case file (Beakte)

class ClassificationResult(BaseModel):
    category: DocumentCategory
    confidence: float
    explanation: str
    filename: str


class GeminiClassification(BaseModel):
    category: DocumentCategory
    confidence: float
    explanation: str

class ResearchRequest(BaseModel):
    query: str

class ResearchResult(BaseModel):
    query: str
    summary: str
    sources: List[Dict[str, str]] = []  # List of {"title": "...", "url": "...", "description": "..."}
    suggestions: List[str] = []

class SavedSource(BaseModel):
    id: str
    url: str
    title: str
    description: Optional[str] = None
    document_type: str
    pdf_url: Optional[str] = None
    download_path: Optional[str] = None
    download_status: str = "pending"  # "pending", "downloading", "completed", "failed"
    research_query: str
    timestamp: str

class AddSourceRequest(BaseModel):
    title: str
    url: str
    description: Optional[str] = None
    pdf_url: Optional[str] = None
    document_type: str = "Rechtsprechung"
    research_query: Optional[str] = None
    auto_download: bool = True

# Anonymization models
class AnonymizationRequest(BaseModel):
    text: str
    document_type: str

class AnonymizationResult(BaseModel):
    anonymized_text: str
    plaintiff_names: List[str]
    confidence: float
    original_text: str
    processed_characters: int


def _notify_sources_updated(event_type: str = "sources_updated", payload: Optional[Dict[str, str]] = None) -> None:
    """Notify connected SSE subscribers that sources changed."""
    subscribers = getattr(app.state, 'source_subscribers', set())
    if not subscribers:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    message = {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat()
    }
    if payload:
        message.update(payload)
    data = json.dumps(message, ensure_ascii=False)

    for queue in list(subscribers):
        try:
            if loop:
                loop.call_soon(queue.put_nowait, data)
            else:
                queue.put_nowait(data)
        except asyncio.QueueFull:
            # Drop oldest event by retrieving once, then retry
            try:
                queue.get_nowait()
                if loop:
                    loop.call_soon(queue.put_nowait, data)
                else:
                    queue.put_nowait(data)
            except Exception:
                subscribers.discard(queue)
        except Exception:
            subscribers.discard(queue)

class SelectedDocuments(BaseModel):
    anhoerung: List[str] = []
    bescheid: "BescheidSelection"
    rechtsprechung: List[str] = []
    saved_sources: List[str] = []


class BescheidSelection(BaseModel):
    primary: str
    others: List[str] = []


class GenerationRequest(BaseModel):
    document_type: Literal["Klagebegründung", "Schriftsatz"]
    user_prompt: str
    selected_documents: SelectedDocuments


class GenerationMetadata(BaseModel):
    documents_used: Dict[str, int]
    citations_found: int = 0
    missing_citations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    word_count: int = 0


class GenerationResponse(BaseModel):
    success: bool = True
    document_type: str
    user_prompt: str
    generated_text: str
    used_documents: List[Dict[str, str]] = []
    metadata: GenerationMetadata


class JLawyerSendRequest(BaseModel):
    case_id: str
    template_name: str
    file_name: str
    generated_text: str
    template_folder: Optional[str] = None


class JLawyerResponse(BaseModel):
    success: bool
    message: str


class JLawyerTemplatesResponse(BaseModel):
    templates: List[str]
    folder: str


try:
    GenerationRequest.model_rebuild()
    GenerationResponse.model_rebuild()
except AttributeError:
    GenerationRequest.update_forward_refs(SelectedDocuments=SelectedDocuments, BescheidSelection=BescheidSelection)
    GenerationResponse.update_forward_refs(GenerationMetadata=GenerationMetadata)

# Database helper functions (no longer needed - using ORM directly in endpoints)

# Initialize OpenAI client
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)

# Initialize Gemini client
def get_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)

# Initialize Anthropic (Claude) client
def get_anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)

# Anonymization service configuration
# Anonymization service configuration (resolved dynamically to pick up env changes)
def get_anonymization_service_settings():
    return (
        os.environ.get("ANONYMIZATION_SERVICE_URL"),
        os.environ.get("ANONYMIZATION_API_KEY"),
    )

# OCR service configuration (read lazily since service URL may change without restart)
def get_ocr_service_settings():
    return (
        os.environ.get("OCR_SERVICE_URL"),
        os.environ.get("OCR_API_KEY"),
    )

def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (' ', '-', '_', '.')).strip()

def extract_pdf_text(pdf_path: str, max_pages: int = 5) -> str:
    """Extract text from first few pages of PDF"""
    try:
        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = min(total_pages, max_pages)

            text_parts = []
            for i in range(pages_to_read):
                # Extract text from page
                page = pdf.pages[i]
                if hasattr(page, 'extract_text'):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"--- Page {i+1} ---\n{text}")

            return "\n\n".join(text_parts) if text_parts else ""
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")


async def perform_ocr_on_pdf(pdf_path: str) -> Optional[str]:
    """
    Perform OCR on a PDF file using the home PC OCR service.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text if successful, None if service unavailable
    """
    ocr_service_url, ocr_api_key = get_ocr_service_settings()

    if not ocr_service_url:
        print("[WARNING] OCR_SERVICE_URL not configured")
        return None

    try:
        headers = {}
        if ocr_api_key:
            headers["X-API-Key"] = ocr_api_key

        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()

        print(f"[INFO] Sending PDF to OCR service (size: {len(pdf_content)} bytes)")

        async with httpx.AsyncClient(timeout=180.0) as client:
            files = {
                'file': (os.path.basename(pdf_path), pdf_content, 'application/pdf')
            }

            response = await client.post(
                f"{ocr_service_url}/ocr",
                files=files,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            text = data.get("full_text", "")
            confidence = data.get("avg_confidence", 0.0)
            page_count = data.get("page_count", 0)

            print(f"[SUCCESS] OCR completed: {len(text)} characters, "
                  f"{page_count} pages, confidence: {confidence:.2f}")

            return text

    except httpx.TimeoutException:
        print("[ERROR] OCR service timeout (>180s)")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] OCR service HTTP error: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[ERROR] OCR service error: {e}")
        return None


async def anonymize_document_text(text: str, document_type: str) -> Optional[AnonymizationResult]:
    """
    Call anonymization service on home PC via Tailscale.

    Args:
        text: Extracted text from PDF document
        document_type: Either "Anhörung" or "Bescheid"

    Returns:
        AnonymizationResult if successful, None if service unavailable
    """
    anonymization_service_url, anonymization_api_key = get_anonymization_service_settings()

    if not anonymization_service_url:
        print("[WARNING] ANONYMIZATION_SERVICE_URL not configured")
        return None

    try:
        headers = {}
        if anonymization_api_key:
            headers["X-API-Key"] = anonymization_api_key

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{anonymization_service_url}/anonymize",
                json={
                    "text": text,
                    "document_type": document_type
                },
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            return AnonymizationResult(
                anonymized_text=data["anonymized_text"],
                plaintiff_names=data["plaintiff_names"],
                confidence=data["confidence"],
                original_text=text,
                processed_characters=len(text)
            )

    except httpx.TimeoutException:
        print("[ERROR] Anonymization service timeout (>120s)")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] Anonymization service HTTP error: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[ERROR] Anonymization service error: {e}")
        return None


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

async def classify_document(file_content: bytes, filename: str) -> ClassificationResult:
    """Classify a document using Gemini 2.5 Flash."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name

    client = get_gemini_client()

    prompt = """Analysiere dieses deutsche Rechtsdokument und ordne es einer Kategorie zu:

1. **Anhörung** – BAMF-Anhörungsprotokoll / Niederschrift
   - Titel „Niederschrift …“, BAMF-Briefkopf, „Es erscheint …“, Frage-Antwort-Struktur, Unterschriften

2. **Bescheid** – BAMF-Entscheidungsbescheid
   - Überschrift „BESCHEID“, Gesch.-Z., nummerierte Entscheidungen, Rechtsbehelfsbelehrung

3. **Rechtsprechung** – Gerichtliche Entscheidung (Urteil/Beschluss)
   - Gericht als Absender, Aktenzeichen, Tenor, Tatbestand, Entscheidungsgründe

4. **Akte** – Vollständige BAMF-Beakte / Fallakte
   - Enthält mehrere Dokumentarten (z. B. Anhörungen, Bescheide, Vermerke) in einer PDF, oft mit Register- oder Blattnummern. Wähle **Akte**, sobald klar ist, dass das PDF eine komplette Beakte bzw. eine umfangreiche Sammlung enthält – auch dann, wenn darin einzelne Bescheide vorkommen.

5. **Sonstiges** – Alle anderen Dokumente

Erzeuge ausschließlich JSON:
{
  "category": "<Anhörung|Bescheid|Rechtsprechung|Akte|Sonstiges>",
  "confidence": <float 0.0-1.0>,
  "explanation": "kurze deutschsprachige Begründung"
}
"""

    with open(tmp_path, "rb") as pdf_file:
        uploaded = client.files.upload(
            file=pdf_file,
            config={
                "mime_type": "application/pdf",
                "display_name": filename,
            },
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=[prompt, uploaded],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=GeminiClassification,
            ),
        )

        parsed: GeminiClassification = response.parsed

        return ClassificationResult(
            category=parsed.category,
            confidence=parsed.confidence,
            explanation=parsed.explanation,
            filename=filename,
        )

    finally:
        client.files.delete(name=uploaded.name)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

async def research_with_gemini(query: str) -> ResearchResult:
    """
    Perform web research using Gemini with Google Search grounding.
    Returns relevant links and sources for the user's query.
    """
    try:
        print(f"Starting research for query: {query}")
        client = get_gemini_client()
        print("Gemini client initialized")

        # Configure Google Search grounding tool
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        suggestion_text = "\n".join(f"- {s}" for s in ASYL_NET_ALL_SUGGESTIONS) if ASYL_NET_ALL_SUGGESTIONS else "- (keine Schlagwörter geladen)"

        prompt_summary = f"""Du bist ein Rechercheassistent für deutsches Asylrecht.

Recherchiere und liste relevante Quellen zur folgenden Anfrage auf:

{query}

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

        prompt_suggestions = f"""Du bist ein Rechercheassistent für deutsches Asylrecht.

Die folgende Anfrage lautet:
{query}

Hier ist die Schlagwort-Liste für asyl.net (verwende ausschließlich Begriffe aus dieser Liste):
{suggestion_text}

Gib mir genau 1 bis 3 Schlagwörter aus der Liste zurück, die am besten zur Anfrage passen.
Antwortformat: {{\"suggestions\": [\"...\", \"...\"]}} (keine zusätzlichen Erklärungen, kein Markdown)."""

        async def call_summary():
            return await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash-preview-09-2025",
                contents=prompt_summary,
                config=types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=0.0
                )
            )

        async def call_suggestions():
            return await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash-preview-09-2025",
                contents=prompt_suggestions,
                config=types.GenerateContentConfig(temperature=0.0)
            )

        print("Calling Gemini API for summary and suggestions in parallel...")
        response_summary, response_suggestions = await asyncio.gather(call_summary(), call_suggestions())
        print("Gemini calls successful")

        summary_markdown = (response_summary.text or "").strip()
        if summary_markdown:
            summary_markdown = "\n".join(line.rstrip() for line in summary_markdown.replace("\r\n", "\n").split("\n"))
        else:
            summary_markdown = "**Web-Recherche**\n\nKeine Rechercheergebnisse gefunden."
        if not summary_markdown.lower().startswith("**web-recherche**"):
            summary_markdown = f"**Web-Recherche**\n\n{summary_markdown}"

        raw_text_suggestions = response_suggestions.text if response_suggestions.text else "{}"
        try:
            suggestions_data = json.loads(raw_text_suggestions)
            asyl_suggestions = suggestions_data.get("suggestions", [])
            if isinstance(asyl_suggestions, str):
                asyl_suggestions = [asyl_suggestions]
            asyl_suggestions = [s.strip() for s in asyl_suggestions if isinstance(s, str) and s.strip()]
            seen = set()
            unique_suggestions = []
            for s in asyl_suggestions:
                low = s.lower()
                if low not in seen:
                    seen.add(low)
                    unique_suggestions.append(s)
            asyl_suggestions = unique_suggestions[:5]
        except json.JSONDecodeError:
            asyl_suggestions = []

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
        import traceback
        print(f"ERROR in research_with_gemini: {e}")
        print(traceback.format_exc())
        raise Exception(f"Research failed: {e}")

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
            import re

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
        import traceback
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
            import re

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
        import traceback
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
        import traceback
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

    from playwright.async_api import async_playwright

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
            else:
                print(f"No PDF link found with standard selectors on {current_url}")

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

        from playwright.async_api import async_playwright

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
        import traceback
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
        from playwright.async_api import async_playwright

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
        from playwright.async_api import async_playwright

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
            model="gemini-2.5-flash-preview-09-2025",
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
        import traceback
        print(f"ERROR in research_with_legal_databases: {e}")
        print(traceback.format_exc())
        raise Exception(f"Legal database research failed: {e}")

async def download_and_update_source(source_id: str, url: str, title: str):
    """Background task to download a source and update its status"""
    from database import SessionLocal
    db = SessionLocal()
    try:
        # Update status to downloading
        source_uuid = uuid.UUID(source_id)
        source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
        if source:
            source.download_status = 'downloading'
            db.commit()
            _notify_sources_updated('download_started', {'source_id': source_id})

        # Download the PDF
        download_path = await download_source_as_pdf(url, title)

        # Update status
        source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
        if source:
            if download_path:
                source.download_status = 'completed'
                source.download_path = download_path
                _notify_sources_updated('download_completed', {'source_id': source_id})
            else:
                source.download_status = 'failed'
                _notify_sources_updated('download_failed', {'source_id': source_id})
            db.commit()

    except Exception as e:
        print(f"Error in background download for {url}: {e}")
        # Mark as failed
        try:
            source_uuid = uuid.UUID(source_id)
            source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
            if source:
                source.download_status = 'failed'
                db.commit()
                _notify_sources_updated('download_failed', {'source_id': source_id})
        except Exception:
            pass
    finally:
        db.close()

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
            model="gemini-2.5-flash-preview-09-2025",
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
    import hashlib
    from playwright.async_api import async_playwright

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

@app.get("/")
async def root():
    """Serve document classification interface"""
    return FileResponse("templates/index.html")

@app.post("/classify", response_model=ClassificationResult)
@limiter.limit("20/hour")
async def classify(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Classify uploaded PDF document"""

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # Persist upload to disk for later context use
    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(file.filename) or "upload.pdf"
        unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        stored_path = UPLOADS_DIR / unique_name
        with open(stored_path, 'wb') as out:
            out.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded file: {e}")

    # Classify document
    try:
        result = await classify_document(content, file.filename)

        existing_doc = db.query(Document).filter(Document.filename == result.filename).first()
        if existing_doc:
            existing_doc.category = result.category.value
            existing_doc.confidence = result.confidence
            existing_doc.explanation = result.explanation
            existing_doc.file_path = str(stored_path)
        else:
            new_doc = Document(
                filename=result.filename,
                category=result.category.value,
                confidence=result.confidence,
                explanation=result.explanation,
                file_path=str(stored_path)
            )
            db.add(new_doc)

        if result.category == DocumentCategory.AKTE:
            try:
                segment_client = get_gemini_client()
                segment_dir = stored_path.parent / f"{stored_path.stem}_segments"
                sections, extracted_pairs = segment_pdf_with_gemini(
                    str(stored_path),
                    segment_dir,
                    client=segment_client,
                    config=SegmentationGeminiConfig(),
                    verbose=False,
                )

                for section, path in extracted_pairs:
                    try:
                        category_enum = DocumentCategory(section.document_type)
                    except ValueError:
                        category_enum = DocumentCategory.SONSTIGES

                    segment_filename = Path(path).name

                    existing_segment = (
                        db.query(Document)
                        .filter(Document.filename == segment_filename)
                        .first()
                    )
                    segment_explanation = (
                        f"Segment ({section.document_type}) aus Akte {result.filename}, "
                        f"Seiten {section.start_page}-{section.end_page}"
                    )

                    if existing_segment:
                        existing_segment.category = category_enum.value
                        existing_segment.confidence = section.confidence
                        existing_segment.explanation = segment_explanation
                        existing_segment.file_path = path
                    else:
                        db.add(
                            Document(
                                filename=segment_filename,
                                category=category_enum.value,
                                confidence=section.confidence,
                                explanation=segment_explanation,
                                file_path=path,
                            )
                        )
            except Exception as segmentation_error:
                print(f"Segmentation failed for {stored_path}: {segmentation_error}")

        db.commit()
        return result
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

@app.get("/documents")
@limiter.limit("200/hour")
async def get_documents(request: Request, db: Session = Depends(get_db)):
    """Get all classified documents grouped by category"""
    documents = db.query(Document).order_by(desc(Document.created_at)).all()

    # Group by category
    grouped = {
        "Anhörung": [],
        "Bescheid": [],
        "Rechtsprechung": [],
        "Akte": [],
        "Sonstiges": []
    }

    for doc in documents:
        category = doc.category if doc.category in grouped else 'Sonstiges'
        grouped[category].append(doc.to_dict())

    return JSONResponse(
        content=grouped,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.delete("/documents/{filename}")
@limiter.limit("100/hour")
async def delete_document(request: Request, filename: str, db: Session = Depends(get_db)):
    """Delete a classified document"""
    doc = db.query(Document).filter(Document.filename == filename).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete associated uploaded file
    if doc.file_path:
        try:
            fp = Path(doc.file_path)
            if fp.exists():
                fp.unlink()
        except Exception as e:
            print(f"Error deleting uploaded file for {filename}: {e}")

    db.delete(doc)
    db.commit()
    success = True
    if success:
        return {"message": f"Document {filename} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Document {filename} not found")


@app.post("/documents/{document_id}/ocr")
@limiter.limit("10/hour")
async def run_document_ocr(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db)
):
    """Run OCR for a document and cache the extracted text"""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_path = document.file_path
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on server")

    print(f"[INFO] Manual OCR triggered for {document.filename}")
    text = await perform_ocr_on_pdf(pdf_path)
    if not text:
        raise HTTPException(
            status_code=503,
            detail="OCR service unavailable. Please ensure home PC OCR service is running."
        )

    normalized_text = text.strip()
    if len(normalized_text) < 50:
        raise HTTPException(
            status_code=422,
            detail="OCR completed but returned insufficient text. The document may have very low quality."
        )

    metadata = {
        "ocr_text_length": len(normalized_text),
        "processed_at": datetime.utcnow().isoformat(),
        "preview": normalized_text[:300]
    }

    existing_ocr = db.query(ProcessedDocument).filter(
        ProcessedDocument.document_id == doc_uuid,
        ProcessedDocument.is_anonymized == False,
        ProcessedDocument.ocr_applied == True
    ).order_by(desc(ProcessedDocument.created_at)).first()

    if existing_ocr:
        existing_ocr.extracted_text = normalized_text
        existing_ocr.anonymization_metadata = metadata
        existing_ocr.processing_status = "ocr_ready"
        processed_record = existing_ocr
        print(f"[INFO] Updated existing OCR cache for {document.filename}")
    else:
        processed_record = ProcessedDocument(
            document_id=doc_uuid,
            extracted_text=normalized_text,
            is_anonymized=False,
            ocr_applied=True,
            anonymization_metadata=metadata,
            processing_status="ocr_ready"
        )
        db.add(processed_record)
        print(f"[INFO] Created new OCR cache entry for {document.filename}")

    db.commit()

    return {
        "status": "success",
        "document_id": document_id,
        "text_length": len(normalized_text),
        "preview": normalized_text[:200],
        "processing_id": str(processed_record.id),
        "message": "OCR completed successfully and cached for anonymization."
    }


@app.post("/documents/{document_id}/anonymize")
@limiter.limit("5/hour")
async def anonymize_document_endpoint(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Anonymize a classified document (Anhörung or Bescheid only).

    Process:
    1. Verify document exists and is anonymizable
    2. Extract text from PDF
    3. Call anonymization service on home PC
    4. Store result in processed_documents table
    5. Return anonymized text

    Rate limit: 5 requests per hour (processing takes 30-60s each)
    """

    # Validate and fetch document
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Only anonymize Anhörung and Bescheid
    if document.category not in ["Anhörung", "Bescheid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Document type '{document.category}' does not support anonymization. Only 'Anhörung' and 'Bescheid' can be anonymized."
        )

    # Check if already processed
    existing_processed = db.query(ProcessedDocument).filter(
        ProcessedDocument.document_id == doc_uuid,
        ProcessedDocument.is_anonymized == True
    ).first()

    if existing_processed and existing_processed.anonymization_metadata:
        # Return cached result
        return {
            "status": "success",
            "anonymized_text": existing_processed.anonymization_metadata.get("anonymized_text", ""),
            "plaintiff_names": existing_processed.anonymization_metadata.get("plaintiff_names", []),
            "confidence": existing_processed.anonymization_metadata.get("confidence", 0.0),
            "cached": True
        }

    # Extract text from PDF (try direct extraction, then OCR if needed)
    pdf_path = document.file_path
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on server")

    extracted_text = None
    ocr_used = False

    # Check for cached OCR text (from manual OCR run)
    cached_ocr = db.query(ProcessedDocument).filter(
        ProcessedDocument.document_id == doc_uuid,
        ProcessedDocument.is_anonymized == False,
        ProcessedDocument.ocr_applied == True,
        ProcessedDocument.extracted_text.isnot(None)
    ).order_by(desc(ProcessedDocument.created_at)).first()

    if cached_ocr and cached_ocr.extracted_text:
        extracted_text = cached_ocr.extracted_text
        ocr_used = True
        cached_ocr.processing_status = "ocr_cached_used"
        print(f"[INFO] Using cached OCR text for {document.filename}: {len(extracted_text)} characters")

    # Try direct text extraction first
    if not extracted_text:
        try:
            extracted_text = extract_pdf_text(pdf_path, max_pages=50)
            if extracted_text and len(extracted_text.strip()) >= 100:
                print(f"[INFO] Direct text extraction successful: {len(extracted_text)} characters")
            else:
                print("[INFO] Direct extraction insufficient, trying OCR...")
                extracted_text = None
        except Exception as e:
            print(f"[INFO] Direct extraction failed: {e}, trying OCR...")
            extracted_text = None

    # If direct extraction failed, try OCR
    if not extracted_text:
        extracted_text = await perform_ocr_on_pdf(pdf_path)
        if extracted_text:
            ocr_used = True
            print(f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters")
        else:
            raise HTTPException(
                status_code=503,
                detail="Could not extract text from PDF. OCR service unavailable. Please ensure home PC OCR service is running."
            )

    # Verify we have sufficient text
    if not extracted_text or len(extracted_text.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail="Insufficient text extracted from PDF. The document may be empty or corrupted."
        )

    # Call anonymization service (limit to first 10k characters to avoid timeouts)
    # Plaintiff names are typically at the beginning of legal documents
    text_for_anonymization = extracted_text[:10000]
    print(f"[INFO] Sending {len(text_for_anonymization)} characters to anonymization service")

    result = await anonymize_document_text(text_for_anonymization, document.category)

    if not result:
        raise HTTPException(
            status_code=503,
            detail="Anonymization service unavailable. Please ensure home PC is online and connected via Tailscale."
        )

    # Combine anonymized section with untouched remainder
    processed_chars = min(result.processed_characters, len(extracted_text))
    remaining_text = extracted_text[processed_chars:]
    anonymized_full_text = result.anonymized_text or extracted_text[:processed_chars]

    if remaining_text:
        separator = ""
        if anonymized_full_text and not anonymized_full_text.endswith("\n"):
            separator = "\n\n"
        anonymized_full_text = f"{anonymized_full_text}{separator}{remaining_text}"

    # Store in processed_documents table
    processed_doc = ProcessedDocument(
        document_id=doc_uuid,
        extracted_text=extracted_text,
        is_anonymized=True,
        ocr_applied=ocr_used,
        anonymization_metadata={
            "plaintiff_names": result.plaintiff_names,
            "confidence": result.confidence,
            "anonymized_at": datetime.utcnow().isoformat(),
            "anonymized_text": anonymized_full_text,
            "anonymized_excerpt": result.anonymized_text,
            "processed_characters": processed_chars,
            "remaining_characters": len(remaining_text),
            "ocr_used": ocr_used
        },
        processing_status="completed"
    )
    db.add(processed_doc)
    db.commit()

    return {
        "status": "success",
        "anonymized_text": anonymized_full_text,
        "plaintiff_names": result.plaintiff_names,
        "confidence": result.confidence,
        "processed_characters": processed_chars,
        "remaining_characters": len(remaining_text),
        "ocr_used": ocr_used,
        "cached": False
    }

@app.post("/research", response_model=ResearchResult)
@limiter.limit("10/hour")
async def research(request: Request, body: ResearchRequest):
    """Perform web research using both Gemini web search and specialized legal databases"""
    try:
        print(f"Starting research pipeline for query: {body.query}")

        web_result = await research_with_gemini(body.query)
        all_sources: List[Dict[str, str]] = list(web_result.sources)
        summaries = [web_result.summary] if web_result.summary else []

        asyl_sources: List[Dict[str, str]] = []
        try:
            suggestions = web_result.suggestions or []
            print(f"Using asyl.net suggestions from web search: {suggestions}")
            asyl_sources = await search_asyl_net(body.query, suggestions=suggestions or None)
            all_sources.extend(asyl_sources)
        except Exception as e:
            print(f"asyl.net search failed: {e}")

        combined_summary = "<hr/>".join(summaries) if summaries else ""

        print(f"Combined research returned {len(all_sources)} total sources")

        return ResearchResult(
            query=body.query,
            summary=combined_summary,
            sources=all_sources,
            suggestions=web_result.suggestions
        )
    except Exception as e:
        import traceback
        print(f"ERROR in research endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Research failed: {e}")

def _document_to_context_dict(doc: Document) -> Dict[str, Optional[str]]:
    """Convert a Document ORM instance into a context dictionary used for prompting."""
    stored_path = Path(doc.file_path) if doc.file_path else None
    if stored_path and not stored_path.exists():
        raise HTTPException(status_code=404, detail=f"Dokument {doc.filename} wurde nicht auf dem Server gefunden")

    return {
        "id": str(doc.id),
        "filename": doc.filename,
        "category": doc.category,
        "file_path": str(stored_path) if stored_path else None,
        "confidence": doc.confidence,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "explanation": doc.explanation,
    }


def _validate_category(doc: Document, expected_category: str) -> None:
    """Ensure a document matches the expected category."""
    if doc.category != expected_category:
        raise HTTPException(
            status_code=400,
            detail=f"Dokument {doc.filename} gehört zur Kategorie '{doc.category}', erwartet war '{expected_category}'",
        )


def _collect_selected_documents(selection: SelectedDocuments, db: Session) -> Dict[str, List[Dict[str, Optional[str]]]]:
    """Validate and collect document metadata for Klagebegründung generation."""
    bescheid_selection = selection.bescheid

    total_selected = len(selection.anhoerung) + len(selection.rechtsprechung) + len(selection.saved_sources)
    total_selected += 1 if bescheid_selection.primary else 0
    total_selected += len(bescheid_selection.others)

    if total_selected == 0:
        raise HTTPException(status_code=400, detail="Bitte wählen Sie mindestens ein Dokument aus")

    collected: Dict[str, List[Dict[str, Optional[str]]]] = {
        "anhoerung": [],
        "bescheid": [],
        "rechtsprechung": [],
        "saved_sources": [],
    }

    # Load Anhörung documents
    if selection.anhoerung:
        query = (
            db.query(Document)
            .filter(Document.filename.in_(selection.anhoerung))
            .all()
        )
        found_map = {doc.filename: doc for doc in query}
        missing = [fn for fn in selection.anhoerung if fn not in found_map]
        if missing:
            raise HTTPException(status_code=404, detail=f"Anhörung-Dokumente nicht gefunden: {', '.join(missing)}")
        for doc in query:
            _validate_category(doc, DocumentCategory.ANHOERUNG.value)
            collected["anhoerung"].append(_document_to_context_dict(doc))

    # Load Bescheid documents (primary + others)
    if not bescheid_selection.primary:
        raise HTTPException(status_code=400, detail="Bitte markieren Sie einen Bescheid als Hauptbescheid (Anlage K2)")

    bescheid_filenames = [bescheid_selection.primary] + (bescheid_selection.others or [])
    bescheid_query = (
        db.query(Document)
        .filter(Document.filename.in_(bescheid_filenames))
        .all()
    )
    bescheid_map = {doc.filename: doc for doc in bescheid_query}

    missing_bescheide = [fn for fn in bescheid_filenames if fn not in bescheid_map]
    if missing_bescheide:
        raise HTTPException(status_code=404, detail=f"Bescheid-Dokumente nicht gefunden: {', '.join(missing_bescheide)}")

    primary_doc = bescheid_map[bescheid_selection.primary]
    _validate_category(primary_doc, DocumentCategory.BESCHEID.value)
    collected["bescheid"].append({**_document_to_context_dict(primary_doc), "role": "primary"})

    for other_name in bescheid_selection.others or []:
        doc = bescheid_map[other_name]
        _validate_category(doc, DocumentCategory.BESCHEID.value)
        collected["bescheid"].append({**_document_to_context_dict(doc), "role": "secondary"})

    # Load Rechtsprechung documents
    if selection.rechtsprechung:
        query = (
            db.query(Document)
            .filter(Document.filename.in_(selection.rechtsprechung))
            .all()
        )
        found_map = {doc.filename: doc for doc in query}
        missing = [fn for fn in selection.rechtsprechung if fn not in found_map]
        if missing:
            raise HTTPException(status_code=404, detail=f"Rechtsprechung-Dokumente nicht gefunden: {', '.join(missing)}")
        for doc in query:
            _validate_category(doc, DocumentCategory.RECHTSPRECHUNG.value)
            collected["rechtsprechung"].append(_document_to_context_dict(doc))

    # Load saved sources
    if selection.saved_sources:
        collected_sources = []
        for source_id in selection.saved_sources:
            try:
                source_uuid = uuid.UUID(source_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Ungültige Quellen-ID: {source_id}")
            source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
            if not source:
                raise HTTPException(status_code=404, detail=f"Quelle {source_id} wurde nicht gefunden")
            collected_sources.append(
                {
                    "id": str(source.id),
                    "title": source.title,
                    "url": source.url,
                    "description": source.description,
                    "document_type": source.document_type,
                    "download_path": source.download_path,
                    "created_at": source.created_at.isoformat() if source.created_at else None,
                }
            )
        collected["saved_sources"] = collected_sources

    return collected


def _sanitize_filename_for_claude(filename: str) -> str:
    """Sanitize filename to only contain ASCII characters for Claude Files API."""
    # Replace German umlauts and special characters
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'
    }
    for char, replacement in replacements.items():
        filename = filename.replace(char, replacement)

    # Remove any remaining non-ASCII characters and keep only safe chars
    sanitized = ''.join(
        c if c.isascii() and (c.isalnum() or c in '.-_ ') else '_'
        for c in filename
    )

    # Clean up multiple underscores/spaces and trim
    sanitized = re.sub(r'[_\s]+', '_', sanitized).strip('_')

    # Ensure we keep the .pdf extension
    if not sanitized.lower().endswith('.pdf'):
        sanitized += '.pdf'

    return sanitized


def _upload_documents_to_claude(client: anthropic.Anthropic, documents: List[Dict[str, Optional[str]]]) -> List[Dict[str, str]]:
    """Upload local documents using Claude Files API and return document content blocks."""
    content_blocks: List[Dict[str, str]] = []
    MAX_PAGES = 100  # Claude Files API limit

    for entry in documents:
        file_path = entry.get("file_path")
        if not file_path:
            continue
        path_obj = Path(file_path)
        if not path_obj.exists():
            print(f"[WARN] Datei {file_path} wurde nicht gefunden, überspringe Upload")
            continue
        original_filename = entry.get("filename") or path_obj.name

        # Sanitize filename for Claude API (remove umlauts, special chars)
        sanitized_filename = _sanitize_filename_for_claude(original_filename)

        # Check PDF page count before uploading
        try:
            pdf = pikepdf.open(path_obj)
            page_count = len(pdf.pages)
            pdf.close()

            if page_count > MAX_PAGES:
                print(f"[WARN] Datei {original_filename} hat {page_count} Seiten (max {MAX_PAGES}), wird übersprungen")
                continue
        except Exception as exc:
            print(f"[WARN] Seitenzahl für {original_filename} konnte nicht ermittelt werden: {exc}")
            # Continue anyway, let Claude API reject if needed

        try:
            with open(path_obj, "rb") as file_handle:
                uploaded_file = client.beta.files.upload(
                    file=(sanitized_filename, file_handle, "application/pdf"),
                    betas=["files-api-2025-04-14"],
                )
        except Exception as exc:
            print(f"[ERROR] Upload für {file_path} fehlgeschlagen: {exc}")
            continue

        content_blocks.append(
            {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": uploaded_file.id,
                },
                "title": original_filename,  # Keep original filename for display
            }
        )

    return content_blocks


_CATEGORY_LABELS = {
    "anhoerung": "Anhörung",
    "bescheid": "Bescheid",
    "rechtsprechung": "Rechtsprechung",
    "saved_sources": "Gespeicherte Quelle",
}


def _normalize_for_match(value: Optional[str]) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^\w\s/.:,-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


_DATE_REGEX = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})")
_AZ_REGEX = re.compile(r"(?:Az\.?|Aktenzeichen)\s*[:]?\s*([A-Za-z0-9./\-\s]+)")


def _build_reference_candidates(category: str, entry: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
    specific: List[str] = []
    generic: List[str] = []

    filename = entry.get("filename")
    if filename:
        stem = Path(filename).stem
        specific.extend(
            {
                filename,
                stem,
                stem.replace("_", " "),
                stem.replace("-", " "),
                stem.replace("_", "-"),
            }
        )

    title = entry.get("title")
    if title:
        specific.append(title)

    explanation = entry.get("explanation")
    if explanation:
        specific.append(explanation)
        specific.extend(_DATE_REGEX.findall(explanation))
        specific.extend([f"vom {m}" for m in _DATE_REGEX.findall(explanation)])
        for az in _AZ_REGEX.findall(explanation):
            cleaned = az.strip()
            if cleaned:
                specific.append(cleaned)
                specific.append(f"az {cleaned.lower()}")
                specific.append(f"az. {cleaned}")

    url = entry.get("url")
    if url:
        specific.append(url)
        try:
            parsed = urlparse(url)
            if parsed.netloc:
                specific.append(parsed.netloc)
            if parsed.path:
                specific.append(parsed.path)
        except Exception:
            pass

    description = entry.get("description")
    if description:
        specific.append(description)
        specific.extend(_DATE_REGEX.findall(description))
        specific.extend([f"vom {m}" for m in _DATE_REGEX.findall(description)])
        for az in _AZ_REGEX.findall(description):
            cleaned = az.strip()
            if cleaned:
                specific.append(cleaned)
                specific.append(f"az {cleaned.lower()}")
                specific.append(f"az. {cleaned}")

    for key in ("date", "aktenzeichen"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            if key == "date":
                specific.append(value.strip())
                specific.append(f"vom {value.strip()}")
            else:
                specific.append(value.strip())
                specific.append(f"az {value.strip().lower()}")
                specific.append(f"az. {value.strip()}")

    category_generic = {
        "anhoerung": ["Anhörung"],
        "bescheid": ["Bescheid"],
        "rechtsprechung": ["Rechtsprechung", "Urteil"],
        "saved_sources": ["Quelle", "Research"],
    }
    generic.extend(category_generic.get(category, []))

    def _dedupe(items):
        seen = set()
        result = []
        for item in items:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    return {"specific": _dedupe(specific), "generic": _dedupe(generic)}


def verify_citations(
    generated_text: str,
    selected_documents: Dict[str, List[Dict[str, Optional[str]]]],
    sources_metadata: Optional[Dict[str, List[Dict[str, Optional[str]]]]] = None,
) -> Dict[str, List[str]]:
    """
    Verify that selected sources appear in the generated text.
    Returns dict with keys `cited`, `missing`, `warnings`.
    """
    normalized_text = _normalize_for_match(generated_text or "")
    result: Dict[str, List[str]] = {"cited": [], "missing": [], "warnings": []}

    if not normalized_text:
        result["warnings"].append("Generierter Text ist leer; Zitatprüfung nicht möglich.")
        for category, entries in (selected_documents or {}).items():
            category_label = _CATEGORY_LABELS.get(category, category)
            for entry in entries:
                label = entry.get("filename") or entry.get("title") or entry.get("id") or "Unbekanntes Dokument"
                result["missing"].append(f"{category_label}: {label}")
        return result

    seen_labels: set[str] = set()

    for category, entries in (selected_documents or {}).items():
        category_label = _CATEGORY_LABELS.get(category, category)
        for entry in entries:
            base_label = entry.get("filename") or entry.get("title") or entry.get("id") or "Unbekanntes Dokument"
            label = f"{category_label}: {base_label}"

            if label in seen_labels:
                continue
            seen_labels.add(label)

            candidates = _build_reference_candidates(category, entry)
            match_found = False
            generic_hit = False

            for candidate in candidates["specific"]:
                normalized_candidate = _normalize_for_match(candidate)
                if normalized_candidate and normalized_candidate in normalized_text:
                    match_found = True
                    break

            if not match_found:
                for candidate in candidates["generic"]:
                    normalized_candidate = _normalize_for_match(candidate)
                    if normalized_candidate and normalized_candidate in normalized_text:
                        match_found = True
                        generic_hit = True
                        break

            if match_found:
                result["cited"].append(label)
                if generic_hit:
                    result["warnings"].append(
                        f"{label}: nur generischer Hinweis gefunden – bitte Zitierung prüfen."
                    )
            else:
                result["missing"].append(label)

            if entry.get("role") == "primary":
                if "anlage k2" not in normalized_text:
                    result["warnings"].append(
                        f"{label}: Referenz 'Anlage K2' nicht gefunden – bitte kontrollieren."
                    )

    return result


def _is_jlawyer_configured() -> bool:
    return all([
        JLAWYER_BASE_URL,
        JLAWYER_USERNAME,
        JLAWYER_PASSWORD,
        JLAWYER_PLACEHOLDER_KEY,
    ])


@app.get("/jlawyer/templates", response_model=JLawyerTemplatesResponse)
@limiter.limit("20/hour")
async def get_jlawyer_templates(request: Request, folder: Optional[str] = None):
    if not _is_jlawyer_configured():
        raise HTTPException(status_code=503, detail="j-lawyer Integration ist nicht konfiguriert")

    folder_name = (folder or JLAWYER_TEMPLATE_FOLDER_DEFAULT or "").strip()
    if not folder_name:
        raise HTTPException(status_code=400, detail="Kein Template-Ordner konfiguriert")

    url = f"{JLAWYER_BASE_URL}/v6/templates/documents/{quote(folder_name, safe='')}"
    auth = (JLAWYER_USERNAME, JLAWYER_PASSWORD)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, auth=auth)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"j-lawyer Anfrage fehlgeschlagen: {exc}")

    if response.status_code >= 400:
        detail = response.text or response.reason_phrase or "Unbekannter Fehler"
        raise HTTPException(status_code=502, detail=f"j-lawyer Fehler ({response.status_code}): {detail}")

    templates: List[str] = []
    try:
        payload = response.json()
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    templates.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("template") or item.get("fileName")
                    if isinstance(name, str):
                        templates.append(name)
    except ValueError:
        text = response.text or ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                templates.append(line)

    return JLawyerTemplatesResponse(templates=templates, folder=folder_name)


@app.post("/generate", response_model=GenerationResponse)
@limiter.limit("10/hour")
async def generate(request: Request, body: GenerationRequest, db: Session = Depends(get_db)):
    """Generate drafts (generic or structured Klagebegründung) using Claude and the Files API."""
    try:
        client = get_anthropic_client()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    collected = _collect_selected_documents(body.selected_documents, db)

    document_entries: List[Dict[str, Optional[str]]] = []
    for category in ("anhoerung", "bescheid", "rechtsprechung"):
        document_entries.extend(collected.get(category, []))

    for source_entry in collected.get("saved_sources", []):
        download_path = source_entry.get("download_path")
        if not download_path:
            continue
        document_entries.append(
            {
                "filename": source_entry.get("title") or source_entry.get("id"),
                "file_path": download_path,
                "category": source_entry.get("document_type") or "Quelle",
            }
        )

    print(f"[DEBUG] Collected {len(document_entries)} document entries for upload:")
    for entry in document_entries:
        print(f"  - {entry.get('category', 'N/A')}: {entry.get('filename', 'N/A')}")

    document_blocks = _upload_documents_to_claude(client, document_entries)
    context_summary = _summarize_selection_for_prompt(collected)

    print(f"[DEBUG] Uploaded {len(document_blocks)} documents to Claude Files API")
    print(f"[DEBUG] Context summary:\n{context_summary}")

    try:
        system_prompt = (
            "Du bist ein Anwalt für Ausländerrecht und schreibst professionelle juristische Schriftsätze. "
            "Du fokussierst dich auf klare, widerspruchsfreie und konkrete juristische Argumentation. "
            "Du widerlegst die Bescheide der Gegenseite. Dabei konzentrierst du dich auf das Wesentliche. "
            "Eine Nacherzählung des Sachverhalts ist nicht nötig. Fokus liegt auf der rechtlichen Würdigung.\n\n"
            "KRITISCH WICHTIG - Du musst die hochgeladenen PDF-Dokumente TATSÄCHLICH LESEN:\n"
            "1. HAUPTBESCHEID (Anlage K2): LIES den kompletten Bescheid des BAMF. "
            "Identifiziere JEDEN einzelnen Ablehnungsgrund. Widerlege diese Punkt für Punkt mit konkreten "
            "Zitaten (mit Seitenangabe) aus den Anhörungen und der Rechtsprechung.\n"
            "2. ANHÖRUNGEN (Anlage K3+): LIES alle Anhörungsprotokolle vollständig. "
            "Verwende konkrete Aussagen des Mandanten (mit Seitenangabe), um die fehlerhafte Würdigung "
            "des BAMF aufzuzeigen.\n"
            "3. RECHTSPRECHUNG & QUELLEN: LIES die Urteile und Quellen. "
            "Zitiere konkrete Rechtssätze, Aktenzeichen und Passagen zur Untermauerung deiner Argumentation.\n\n"
            "STRUKTUR:\n"
            "- Kurze Einleitung (1-2 Sätze)\n"
            "- Rechtliche Würdigung: Gehe jeden Ablehnungsgrund des BAMF durch und widerlege ihn\n"
            "- Anträge\n\n"
            "STIL: Vermeide generische Formulierungen. Jede Behauptung muss mit konkreten Zitaten "
            "aus den hochgeladenen Dokumenten belegt werden.\n"
            "Zitierweise: Verwende die Anlage-Nomenklatur (z.B. 'vgl. Anlage K2, S. 5')."
        )
        user_prompt = (
            f"Dokumententyp: {body.document_type}\n\n"
            f"Auftrag:\n{body.user_prompt.strip()}\n\n"
            "Hochgeladene Dokumente:\n"
            f"{context_summary or '- (Keine Dokumente)'}\n\n"
            "ARBEITSANWEISUNG:\n"
            "1. LIES ZUERST den Hauptbescheid (Anlage K2) vollständig durch\n"
            "2. LIES dann alle Anhörungen (Anlagen K3+) vollständig durch\n"
            "3. ERSTELLE dann eine Klagebegründung, die:\n"
            "   a) Den Sachverhalt basierend auf konkreten Zitaten aus den Anhörungen darstellt\n"
            "   b) Jeden einzelnen Ablehnungsgrund aus dem Hauptbescheid aufgreift und mit:\n"
            "      - Zitaten aus den Anhörungen widerlegt\n"
            "      - Rechtsprechung untermauert\n"
            "      - Konkrete Seitenzahlen nennt\n"
            "   c) Keine generischen Formulierungen verwendet, sondern nur fallspezifische Argumente\n\n"
            "WICHTIG: Beginne NICHT mit dem Schreiben, bevor du alle Dokumente gelesen hast!"
        )

        content = [{"type": "text", "text": user_prompt}]
        content.extend(document_blocks)

        print(f"[DEBUG] Content blocks being sent to Claude API:")
        for i, block in enumerate(content):
            block_type = block.get("type")
            if block_type == "text":
                print(f"  [{i}] text block (length: {len(block.get('text', ''))})")
            elif block_type == "document":
                print(f"  [{i}] document block: {block.get('title', 'untitled')} (file_id: {block.get('source', {}).get('file_id', 'N/A')})")
            else:
                print(f"  [{i}] {block_type} block")

        response = client.beta.messages.create(
            model="claude-sonnet-4-5",
            system=system_prompt,
            max_tokens=16000,  # Increased from 4000 to allow comprehensive legal briefs (max is 64K)
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            betas=["files-api-2025-04-14"],
        )

        # Log API response metadata
        print(f"[DEBUG] API Response - stop_reason: {response.stop_reason}")
        print(f"[DEBUG] API Response - usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

        text_parts: List[str] = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            else:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)
                if block_type == "text" and block_text:
                    text_parts.append(block_text)
        generated_text = "\n\n".join([part for part in text_parts if part]).strip()

        # Warn if we hit max_tokens limit
        if response.stop_reason == "max_tokens":
            print("[WARN] Generation stopped due to max_tokens limit - output may be incomplete!")
            print(f"[WARN] Consider increasing max_tokens above {response.usage.output_tokens}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generierung fehlgeschlagen: {exc}")

    citations = verify_citations(generated_text, collected)
    if citations.get("warnings"):
        for warning in citations["warnings"]:
            print(f"[CITATION WARNING] {warning}")
    if citations.get("missing"):
        for missing in citations["missing"]:
            print(f"[CITATION MISSING] {missing}")
    metadata = GenerationMetadata(
        documents_used={
            "anhoerung": len(collected.get("anhoerung", [])),
            "bescheid": len(collected.get("bescheid", [])),
            "rechtsprechung": len(collected.get("rechtsprechung", [])),
            "saved_sources": len(collected.get("saved_sources", [])),
        },
        citations_found=len(citations.get("cited", [])),
        missing_citations=citations.get("missing", []),
        warnings=citations.get("warnings", []),
        word_count=len(generated_text.split()) if generated_text else 0,
    )

    structured_used_documents: List[Dict[str, str]] = []
    for category, entries in collected.items():
        for entry in entries:
            filename = entry.get("filename") or entry.get("title")
            if not filename:
                continue
            payload = {"filename": filename, "category": category}
            role = entry.get("role")
            if role:
                payload["role"] = role
            structured_used_documents.append(payload)

    return GenerationResponse(
        document_type=body.document_type,
        user_prompt=body.user_prompt.strip(),
        generated_text=generated_text or "(Kein Text erzeugt)",
        used_documents=structured_used_documents,
        metadata=metadata,
    )


def _summarize_selection_for_prompt(collected: Dict[str, List[Dict[str, Optional[str]]]]) -> str:
    """Create a short textual summary of the selected sources for the Claude prompt."""
    lines: List[str] = []
    anlage_counter = 2  # Start with K2 for Hauptbescheid

    # 1. Hauptbescheid (always K2)
    if collected.get("bescheid"):
        for entry in collected["bescheid"]:
            role = entry.get("role", "secondary")
            if role == "primary":
                lines.append(f"- Anlage K{anlage_counter} (HAUPTBESCHEID): {entry.get('filename')}")
                anlage_counter += 1
                break

    # 2. Anhörungen
    anhoerung_count = len(collected.get("anhoerung", []))
    if anhoerung_count > 0:
        lines.append(f"\n📋 Anhörungen ({anhoerung_count}):")
        for entry in collected["anhoerung"]:
            lines.append(f"- Anlage K{anlage_counter}: {entry.get('filename')}")
            anlage_counter += 1

    # 3. Other Bescheide
    if collected.get("bescheid"):
        other_bescheide = [e for e in collected["bescheid"] if e.get("role") != "primary"]
        if other_bescheide:
            lines.append(f"\n📄 Weitere Bescheide ({len(other_bescheide)}):")
            for entry in other_bescheide:
                lines.append(f"- Anlage K{anlage_counter}: {entry.get('filename')}")
                anlage_counter += 1

    # 4. Rechtsprechung
    rechtsprechung_count = len(collected.get("rechtsprechung", []))
    if rechtsprechung_count > 0:
        lines.append(f"\n⚖️ Rechtsprechung ({rechtsprechung_count}):")
        for entry in collected["rechtsprechung"]:
            lines.append(f"- Anlage K{anlage_counter}: {entry.get('filename')}")
            anlage_counter += 1

    # 5. Saved sources
    sources_count = len(collected.get("saved_sources", []))
    if sources_count > 0:
        lines.append(f"\n🔗 Gespeicherte Quellen ({sources_count}):")
        for entry in collected["saved_sources"]:
            title = entry.get("title") or entry.get("id")
            lines.append(f"- Quelle: {title} ({entry.get('url') or 'keine URL'})")

    return "\n".join(lines)


@app.post("/send-to-jlawyer", response_model=JLawyerResponse)
@limiter.limit("10/hour")
async def send_to_jlawyer(request: Request, body: JLawyerSendRequest):
    if not _is_jlawyer_configured():
        raise HTTPException(status_code=503, detail="j-lawyer Integration ist nicht konfiguriert")

    case_id = body.case_id.strip()
    template_name = body.template_name.strip()
    file_name = body.file_name.strip()
    template_folder = (body.template_folder or JLAWYER_TEMPLATE_FOLDER_DEFAULT or "").strip()

    if not case_id or not template_name or not file_name:
        raise HTTPException(status_code=400, detail="case_id, template_name und file_name sind Pflichtfelder")

    if not template_folder:
        raise HTTPException(status_code=400, detail="Kein Template-Ordner konfiguriert")

    if not file_name.lower().endswith(".odt"):
        file_name = f"{file_name}.odt"

    placeholder_value = body.generated_text or ""

    url = (
        f"{JLAWYER_BASE_URL}/v6/templates/documents/"
        f"{quote(template_folder, safe='')}/"
        f"{quote(template_name, safe='')}/"
        f"{quote(case_id, safe='')}/"
        f"{quote(file_name, safe='')}"
    )

    payload = [
        {
            "placeHolderKey": JLAWYER_PLACEHOLDER_KEY,
            "placeHolderValue": placeholder_value,
        }
    ]

    auth = (JLAWYER_USERNAME, JLAWYER_PASSWORD)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(url, auth=auth, json=payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"j-lawyer Anfrage fehlgeschlagen: {exc}")

    if response.status_code >= 400:
        detail = response.text or response.reason_phrase or "Unbekannter Fehler"
        raise HTTPException(status_code=502, detail=f"j-lawyer Fehler ({response.status_code}): {detail}")

    return JLawyerResponse(success=True, message="Vorlage erfolgreich an j-lawyer gesendet")


@app.post("/sources", response_model=SavedSource)
@limiter.limit("100/hour")
async def add_source_endpoint(request: Request, body: AddSourceRequest, db: Session = Depends(get_db)):
    """Manually add a research source and optionally download its PDF."""
    source_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Create database record
    new_source = ResearchSource(
        id=uuid.UUID(source_id),
        url=body.url,
        title=body.title,
        description=body.description,
        document_type=body.document_type,
        pdf_url=body.pdf_url,
        download_status="pending" if body.auto_download else "skipped",
        research_query=body.research_query or "Manuell hinzugefügt"
    )
    db.add(new_source)
    db.commit()
    db.refresh(new_source)

    # Create response model
    saved_source = SavedSource(
        id=source_id,
        url=body.url,
        title=body.title,
        description=body.description,
        document_type=body.document_type,
        pdf_url=body.pdf_url,
        download_status="pending" if body.auto_download else "skipped",
        research_query=body.research_query or "Manuell hinzugefügt",
        timestamp=timestamp
    )

    if body.auto_download:
        download_target = body.pdf_url or body.url
        asyncio.create_task(download_and_update_source(source_id, download_target, body.title))

    return saved_source

@app.get("/sources")
@limiter.limit("1000/hour")
async def get_sources(request: Request, db: Session = Depends(get_db)):
    """Get all saved research sources"""
    print("="*50)
    print("GET /sources endpoint called!")
    print("="*50)
    sources = db.query(ResearchSource).order_by(desc(ResearchSource.created_at)).all()
    sources_dict = [s.to_dict() for s in sources]
    print(f"Returning {len(sources_dict)} sources to client")
    return JSONResponse(
        content={
            "count": len(sources_dict),
            "sources": sources_dict
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/sources/download/{source_id}")
@limiter.limit("50/hour")
async def download_source_file(request: Request, source_id: str, db: Session = Depends(get_db)):
    """Download a saved source PDF"""
    from fastapi.responses import FileResponse

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
        media_type='application/pdf',
        filename=f"{source.title}.pdf"
    )

@app.delete("/sources/{source_id}")
@limiter.limit("100/hour")
async def delete_source_endpoint(request: Request, source_id: str, db: Session = Depends(get_db)):
    """Delete a saved source"""
    try:
        source_uuid = uuid.UUID(source_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source ID format")

    source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
    if not source:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

    # Delete downloaded file if exists
    if source.download_path:
        download_path = Path(source.download_path)
        if download_path.exists():
            try:
                download_path.unlink()
            except Exception as e:
                print(f"Error deleting file {download_path}: {e}")

    db.delete(source)
    db.commit()
    _notify_sources_updated('source_deleted', {'source_id': source_id})
    return {"message": f"Source {source_id} deleted successfully"}

@app.delete("/sources")
@limiter.limit("50/hour")
async def delete_all_sources_endpoint(request: Request, db: Session = Depends(get_db)):
    """Delete all saved sources"""
    sources = db.query(ResearchSource).all()

    if not sources:
        return {"message": "No sources to delete", "count": 0}

    # Delete all downloaded files
    deleted_count = 0
    for source in sources:
        if source.download_path:
            download_path = Path(source.download_path)
            if download_path.exists():
                try:
                    download_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting file {download_path}: {e}")

    # Delete all records from database
    sources_count = len(sources)
    db.query(ResearchSource).delete()
    db.commit()

    _notify_sources_updated('all_sources_deleted', {'count': sources_count})
    return {"message": f"All sources deleted successfully", "count": sources_count, "files_deleted": deleted_count}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/favicon.ico")
async def favicon():
    """Return 204 No Content for favicon to prevent 404 errors"""
    return Response(status_code=204)

@app.post("/debug-research")
async def debug_research(body: ResearchRequest):
    """Debug endpoint to inspect Gemini grounding metadata structure"""
    try:
        client = get_gemini_client()
        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=f"Recherchiere Quellen für: {body.query}",
            config=types.GenerateContentConfig(tools=[grounding_tool], temperature=0.0)
        )

        # Convert response to dict for inspection
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
                        for chunk in gm.grounding_chunks[:3]:  # First 3 chunks
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
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
        reload_includes=["*.py"],
        reload_excludes=["*.json", "*.pdf", "*.log"],
    )

"""
Rechtmaschine - Document Classifier
Simplified document classification system for German asylum law documents
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import pikepdf
import markdown
import re
from openai import OpenAI
from enum import Enum
from google import genai
from google.genai import types
import httpx
import base64
import anthropic
import uuid
from urllib.parse import quote_plus, urljoin
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session
from sqlalchemy import desc
from fastapi import Depends

# Database imports
from database import get_db, engine, Base
from models import Document, ResearchSource, ProcessedDocument

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])

app = FastAPI(title="Rechtmaschine Document Classifier")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.source_subscribers = set()

# Storage directories
DOWNLOADS_DIR = Path("/app/downloaded_sources")
UPLOADS_DIR = Path("/app/uploads")

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

class ClassificationResult(BaseModel):
    category: DocumentCategory
    confidence: float
    explanation: str
    filename: str

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

class GenerationRequest(BaseModel):
    description: str
    include_categories: Optional[List[DocumentCategory]] = None
    max_context_docs: int = 4

class GenerationResponse(BaseModel):
    description: str
    draft_text: str
    used_documents: List[Dict[str, str]] = []  # {filename, file_path}

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
    """Classify document using OpenAI Responses API with GPT-5-mini"""

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name

    try:
        client = get_openai_client()

        # Upload PDF to OpenAI
        with open(tmp_path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="user_data")

        try:
            # Build prompt for classification
            prompt = """Klassifiziere dieses deutsche Rechtsdokument in eine der folgenden Kategorien:

1. **Anhörung** - Anhörungsprotokolle vom BAMF
   - Merkmale: Frage-Antwort-Format, Dolmetscher, persönliche Geschichte des Antragstellers

2. **Bescheid** - BAMF-Bescheide über Asylanträge
   - Merkmale: Offizieller BAMF-Briefkopf, Verfügungssätze, Rechtsbehelfsbelehrung

3. **Rechtsprechung** - Gerichtsentscheidungen, Urteile
   - Merkmale: Gericht als Absender, Aktenzeichen, Tenor, Tatbestand, Entscheidungsgründe

4. **Sonstiges** - Andere Dokumente

Gib deine Antwort mit category (eine der vier Kategorien), confidence (0.0-1.0) und explanation (kurze Begründung auf Deutsch) zurück."""

            # Build request parameters for Responses API (matching meta_ghpl_gpt5.py)
            request_params = {
                "model": "gpt-5-mini",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": uploaded_file.id},
                            {"type": "input_text", "text": prompt}
                        ]
                    }
                ],
                "text_format": ClassificationResult,
                "service_tier": "flex"
            }

            # Use Responses API with structured output (matching meta_ghpl_gpt5.py pattern)
            response = client.with_options(timeout=900.0).responses.parse(**request_params)

            # Clean up uploaded file
            client.files.delete(uploaded_file.id)

            # Extract the parsed result from ParsedResponse
            parsed_result = response.output_parsed

            # Create ClassificationResult with filename
            return ClassificationResult(
                category=parsed_result.category,
                confidence=parsed_result.confidence,
                explanation=parsed_result.explanation,
                filename=filename
            )

        except Exception as e:
            # Clean up uploaded file in case of error
            try:
                client.files.delete(uploaded_file.id)
            except:
                pass
            raise e

    finally:
        # Clean up temporary file
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

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve document classification interface with category boxes"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rechtmaschine - Document Classifier</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f5f5;
                padding: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .upload-section {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            .btn {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            .btn:hover {
                background-color: #2980b9;
            }
            .loading {
                display: none;
                color: #7f8c8d;
                font-style: italic;
                text-align: center;
                margin: 10px 0;
            }
            .categories-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .category-box {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                min-height: 300px;
            }
            .category-box h3 {
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 3px solid;
            }
            .category-box.anhoerung {
                border-top: 4px solid #3498db;
            }
            .category-box.anhoerung h3 {
                color: #3498db;
                border-color: #3498db;
            }
            .category-box.bescheid {
                border-top: 4px solid #27ae60;
            }
            .category-box.bescheid h3 {
                color: #27ae60;
                border-color: #27ae60;
            }
            .category-box.rechtsprechung {
                border-top: 4px solid #e67e22;
            }
            .category-box.rechtsprechung h3 {
                color: #e67e22;
                border-color: #e67e22;
            }
            .category-box.sonstiges {
                border-top: 4px solid #95a5a6;
            }
            .category-box.sonstiges h3 {
                color: #95a5a6;
                border-color: #95a5a6;
            }
            .document-card {
                background: #f8f9fa;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 3px solid #3498db;
                position: relative;
            }
            .document-card .filename {
                font-weight: bold;
                color: #2c3e50;
                word-break: break-word;
                margin-bottom: 5px;
            }
            .document-card .confidence {
                font-size: 12px;
                color: #7f8c8d;
            }
            .document-card .delete-btn {
                position: absolute;
                top: 5px;
                right: 5px;
                background: #e74c3c;
                color: white;
                border: none;
                border-radius: 3px;
                width: 20px;
                height: 20px;
                cursor: pointer;
                font-size: 12px;
                line-height: 1;
            }
            .document-card .delete-btn:hover {
                background: #c0392b;
            }
            .empty-message {
                color: #95a5a6;
                font-style: italic;
                text-align: center;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏛️ Rechtmaschine</h1>
            <p>Dokumenten-Klassifikation für Asylrecht</p>
        </div>

        <div class="upload-section">
            <h3>Dokument hochladen und klassifizieren</h3>
            <input type="file" id="fileInput" accept=".pdf" />
            <br>
            <button class="btn" onclick="uploadFile()">Dokument klassifizieren</button>
            <div class="loading" id="loading">⏳ Dokument wird analysiert...</div>
        </div>

        <div class="upload-section">
            <h3>Web-Recherche</h3>
            <p style="color: #7f8c8d; font-size: 14px; margin-bottom: 10px;">
                Stellen Sie eine Frage zum deutschen Asylrecht. Die KI durchsucht relevante Quellen und liefert Links zu wichtigen Dokumenten.
            </p>
            <textarea id="outputDescription"
                      placeholder="Beispiel: Welche aktuellen Urteile gibt es zu Abschiebungen nach Afghanistan?"
                      style="width: 100%; min-height: 100px; padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px; font-family: Arial; margin-bottom: 10px;">
            </textarea>
            <button class="btn" onclick="generateDocument()" style="background-color: #27ae60;">
                Recherche starten
            </button>
            <button class="btn" onclick="createDraft()" style="background-color: #8e44ad; margin-left: 10px;">
                Entwurf generieren
            </button>
        </div>

        <div class="categories-grid">
            <div class="category-box anhoerung">
                <h3>📋 Anhörung</h3>
                <div id="anhoerung-docs"></div>
            </div>
            <div class="category-box bescheid">
                <h3>📄 Bescheid</h3>
                <div id="bescheid-docs"></div>
            </div>
            <div class="category-box rechtsprechung">
                <h3>⚖️ Rechtsprechung</h3>
                <div id="rechtsprechung-docs"></div>
            </div>
            <div class="category-box sonstiges">
                <h3>🔗 Gespeicherte Quellen</h3>
                <div style="display: flex; justify-content: flex-end; gap: 8px; margin-bottom: 8px;">
                    <button class="btn" onclick="loadSources()" style="padding: 6px 10px; font-size: 12px; background-color: #7f8c8d;">🔄 Aktualisieren</button>
                    <button class="btn" onclick="deleteAllSources()" style="padding: 6px 10px; font-size: 12px; background-color: #e74c3c;">🗑️ Alle löschen</button>
                </div>
                <div id="sonstiges-docs"></div>
            </div>
        </div>

        <script>
            const debugLog = (...args) => {
                const ts = new Date().toISOString();
                console.log(`[Rechtmaschine] ${ts}`, ...args);
            };

            const debugError = (...args) => {
                const ts = new Date().toISOString();
                console.error(`[Rechtmaschine] ${ts}`, ...args);
            };

            // Load documents and sources on page load
            window.addEventListener('DOMContentLoaded', () => {
                debugLog('DOMContentLoaded: initializing interface');
                loadDocuments();
                loadSources();
            });

            async function loadDocuments() {
                debugLog('loadDocuments: start');
                try {
                    debugLog('loadDocuments: fetching /documents');
                    const response = await fetch('/documents');
                    debugLog('loadDocuments: response status', response.status);
                    const data = await response.json();
                    debugLog('loadDocuments: received data', data);

                    // Clear all boxes
                    document.getElementById('anhoerung-docs').innerHTML = '';
                    document.getElementById('bescheid-docs').innerHTML = '';
                    document.getElementById('rechtsprechung-docs').innerHTML = '';
                    document.getElementById('sonstiges-docs').innerHTML = '';

                    // Populate each category
                    const categoryMap = {
                        'Anhörung': 'anhoerung-docs',
                        'Bescheid': 'bescheid-docs',
                        'Rechtsprechung': 'rechtsprechung-docs',
                        'Sonstiges': 'sonstiges-docs'
                    };

                    for (const [category, documents] of Object.entries(data)) {
                        const docsArray = Array.isArray(documents) ? documents : [];
                        debugLog(`loadDocuments: rendering category ${category}`, { count: docsArray.length });
                        const boxId = categoryMap[category];
                        const box = document.getElementById(boxId);

                        if (box) {
                            if (docsArray.length === 0) {
                                box.innerHTML = '<div class="empty-message">Keine Dokumente</div>';
                            } else {
                                const cards = docsArray
                                    .map(doc => createDocumentCard(doc))
                                    .filter(card => !!card)
                                    .join('');
                                box.innerHTML = cards || '<div class="empty-message">Keine Dokumente</div>';
                            }
                        }
                    }
                } catch (error) {
                    debugError('loadDocuments: failed', error);
                }
            }

            function createDocumentCard(doc) {
                if (!doc || !doc.filename) {
                    debugLog('createDocumentCard: skipping entry without filename', doc);
                    return '';
                }
                const confidenceValue = typeof doc.confidence === 'number'
                    ? `${(doc.confidence * 100).toFixed(0)}% Konfidenz`
                    : 'Konfidenz unbekannt';
                debugLog('createDocumentCard', { filename: doc.filename, confidence: doc.confidence });
                const escapedFilename = doc.filename.replace(/'/g, "\\'").replace(/"/g, '&quot;');
                return `
                    <div class="document-card">
                        <button class="delete-btn" onclick="deleteDocument('${escapedFilename}')" title="Löschen">×</button>
                        <div class="filename">${doc.filename}</div>
                        <div class="confidence">${confidenceValue}</div>
                    </div>
                `;
            }

            async function loadSources() {
                debugLog('loadSources: start');
                try {
                    debugLog('loadSources: fetching /sources');
                    const response = await fetch('/sources');
                    debugLog('loadSources: response status', response.status);
                    const payload = await response.json();
                    const sources = Array.isArray(payload) ? payload : (payload.sources || []);
                    const count = Array.isArray(payload) ? payload.length : (payload.count ?? sources.length);
                    debugLog('loadSources: received payload', { count, sources });

                    const container = document.getElementById('sonstiges-docs');
                    debugLog('loadSources: target container located', container);

                    if (sources.length === 0) {
                        debugLog('loadSources: no sources stored');
                        container.innerHTML = '<div class="empty-message">Keine Quellen gespeichert</div>';
                    } else {
                        debugLog('loadSources: rendering source cards');
                        container.innerHTML = sources.map(source => createSourceCard(source)).join('');
                    }
                } catch (error) {
                    debugError('loadSources: failed', error);
                }
            }

            function createSourceCard(source) {
                debugLog('createSourceCard', source);
                const statusEmoji = {
                    'completed': '✅',
                    'downloading': '⏳',
                    'pending': '📥',
                    'failed': '❌',
                    'skipped': '⏭️'
                }[source.download_status] || '📄';

                const downloadButton = source.download_status === 'completed'
                    ? `<a href="/sources/download/${source.id}" target="_blank" style="color: #2c3e50; text-decoration: none; font-size: 12px;">📥 PDF</a>`
                    : '';
                const pdfLinkButton = source.pdf_url
                    ? `<a href="${source.pdf_url}" target="_blank" style="color: #2c3e50; text-decoration: none; font-size: 12px; margin-left: 6px;">🔗 Original-PDF</a>`
                    : '';
                const descriptionHtml = source.description
                    ? `<div style="margin-top: 6px; color: #555; font-size: 13px; line-height: 1.4;">${source.description}</div>`
                    : '';

                return `
                    <div class="document-card">
                        <button class="delete-btn" onclick="deleteSource('${source.id}')" title="Löschen">×</button>
                        <div class="filename">
                            <a href="${source.url}" target="_blank" style="color: inherit; text-decoration: none;">
                                ${source.title}
                            </a>
                            ${downloadButton}${pdfLinkButton}
                        </div>
                        <div class="confidence">
                            ${statusEmoji} ${source.document_type || 'Quelle'}
                        </div>
                        ${descriptionHtml}
                    </div>
                `;
            }

            async function addSourceFromResults(evt, index) {
                const sources = window.latestResearchSources || [];
                const source = sources[index];
                if (!source) {
                    debugError('addSourceFromResults: no source found for index', { index });
                    alert('❌ Quelle konnte nicht gefunden werden.');
                    return;
                }

                const button = evt?.target;
                if (button) {
                    button.disabled = true;
                    button.textContent = '⏳ Wird hinzugefügt...';
                }

                let addedSuccessfully = false;
                debugLog('addSourceFromResults: start', source);
                try {
                    const response = await fetch('/sources', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            title: source.title,
                            url: source.url,
                            description: source.description,
                            pdf_url: source.pdf_url,
                            document_type: source.document_type || 'Rechtsprechung',
                            research_query: window.latestResearchQuery || 'Recherche',
                            auto_download: !!source.pdf_url
                        })
                    });
                    debugLog('addSourceFromResults: response status', response.status);
                    const data = await response.json();

                    if (response.ok) {
                        addedSuccessfully = true;
                        debugLog('addSourceFromResults: success', data);
                        alert('✅ Quelle gespeichert. Download startet im Hintergrund.');
                        if (button) {
                            button.disabled = true;
                            button.textContent = '✅ Quelle hinzugefügt';
                        }
                        loadSources();
                    } else {
                        debugError('addSourceFromResults: server error', data);
                        alert(`❌ Quelle konnte nicht gespeichert werden: ${data.detail || 'Unbekannter Fehler'}`);
                    }
                } catch (error) {
                    debugError('addSourceFromResults: request error', error);
                    alert(`❌ Fehler: ${error.message}`);
                } finally {
                    if (button && !addedSuccessfully) {
                        button.disabled = false;
                        button.textContent = '➕ Zu gespeicherten Quellen';
                    }
                }
            }

            async function deleteSource(sourceId) {
                debugLog('deleteSource: requested', sourceId);
                if (!confirm('Quelle wirklich löschen?')) {
                    debugLog('deleteSource: user cancelled', sourceId);
                    return;
                }

                try {
                    debugLog('deleteSource: sending DELETE', sourceId);
                    const response = await fetch(`/sources/${sourceId}`, {
                        method: 'DELETE'
                    });
                    debugLog('deleteSource: response status', response.status);

                    if (response.ok) {
                        debugLog('deleteSource: success, refreshing sources');
                        loadSources();
                    } else {
                        const data = await response.json();
                        debugError('deleteSource: server error', data);
                        alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
                    }
                } catch (error) {
                    debugError('deleteSource: failed', error);
                    alert(`❌ Fehler: ${error.message}`);
                }
            }

            async function deleteAllSources() {
                debugLog('deleteAllSources: requested');
                if (!confirm('Wirklich ALLE gespeicherten Quellen löschen? Diese Aktion kann nicht rückgängig gemacht werden.')) {
                    debugLog('deleteAllSources: user cancelled');
                    return;
                }

                try {
                    debugLog('deleteAllSources: sending DELETE /sources');
                    const response = await fetch('/sources', {
                        method: 'DELETE'
                    });
                    debugLog('deleteAllSources: response status', response.status);

                    if (response.ok) {
                        const data = await response.json();
                        debugLog('deleteAllSources: success', data);
                        alert(`✅ ${data.count} Quellen gelöscht`);
                        loadSources();
                    } else {
                        const data = await response.json();
                        debugError('deleteAllSources: server error', data);
                        alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
                    }
                } catch (error) {
                    debugError('deleteAllSources: failed', error);
                    alert(`❌ Fehler: ${error.message}`);
                }
            }

            async function uploadFile() {
                debugLog('uploadFile: start');
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];

                if (!file) {
                    debugLog('uploadFile: no file selected');
                    alert('Bitte wählen Sie eine PDF-Datei aus');
                    return;
                }

                const loading = document.getElementById('loading');
                debugLog('uploadFile: showing loading indicator');
                loading.style.display = 'block';

                const formData = new FormData();
                formData.append('file', file);
                debugLog('uploadFile: prepared form data', { filename: file.name, size: file.size });

                try {
                    debugLog('uploadFile: sending POST /classify');
                    const response = await fetch('/classify', {
                        method: 'POST',
                        body: formData
                    });
                    debugLog('uploadFile: response status', response.status);
                    const data = await response.json();
                    debugLog('uploadFile: response body', data);

                    if (response.ok) {
                        debugLog('uploadFile: classification succeeded');
                        // Reload documents to show the new one
                        await loadDocuments();
                        // Clear file input
                        fileInput.value = '';
                        // Show success message
                        alert(`✅ Dokument klassifiziert als: ${data.category} (${(data.confidence * 100).toFixed(0)}%)`);
                    } else {
                        debugError('uploadFile: classification failed', data);
                        alert(`❌ Fehler: ${data.detail || 'Unbekannter Fehler'}`);
                    }
                } catch (error) {
                    debugError('uploadFile: request error', error);
                    alert(`❌ Fehler: ${error.message}`);
                } finally {
                    debugLog('uploadFile: hiding loading indicator');
                    loading.style.display = 'none';
                }
            }

            async function deleteDocument(filename) {
                debugLog('deleteDocument: requested', filename);
                if (!confirm(`Möchten Sie "${filename}" wirklich löschen?`)) {
                    debugLog('deleteDocument: user cancelled', filename);
                    return;
                }

                try {
                    debugLog('deleteDocument: sending DELETE', filename);
                    const response = await fetch(`/documents/${encodeURIComponent(filename)}`, {
                        method: 'DELETE'
                    });
                    debugLog('deleteDocument: response status', response.status);

                    if (response.ok) {
                        debugLog('deleteDocument: success, refreshing documents');
                        // Reload documents
                        await loadDocuments();
                    } else {
                        const data = await response.json();
                        debugError('deleteDocument: server error', data);
                        alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
                    }
                } catch (error) {
                    debugError('deleteDocument: request error', error);
                    alert(`❌ Fehler: ${error.message}`);
                }
            }

            async function generateDocument() {
                debugLog('generateDocument: start');
                const description = document.getElementById('outputDescription').value.trim();

                if (!description) {
                    debugLog('generateDocument: description missing');
                    alert('Bitte beschreiben Sie das gewünschte Dokument');
                    return;
                }

                // Show loading state
                const button = event.target;
                const originalText = button.textContent;
                button.disabled = true;
                button.textContent = '🔍 Recherchiere...';
                debugLog('generateDocument: sending POST /research', { query: description });

                try {
                    const response = await fetch('/research', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query: description })
                    });

                    debugLog('generateDocument: response status', response.status);
                    const data = await response.json();
                    debugLog('generateDocument: response body', data);

                    if (response.ok) {
                        debugLog('generateDocument: research successful, displaying results');
                        displayResearchResults(data);
                        loadSources();
                    } else {
                        debugError('generateDocument: research failed', data);
                        alert(`❌ Fehler: ${data.detail || 'Recherche fehlgeschlagen'}`);
                    }
                } catch (error) {
                    debugError('generateDocument: request error', error);
                    alert(`❌ Fehler: ${error.message}`);
                } finally {
                    debugLog('generateDocument: restoring button state');
                    button.disabled = false;
                    button.textContent = originalText;
                }
            }

            async function createDraft() {
                debugLog('createDraft: start');
                const description = document.getElementById('outputDescription').value.trim();

                if (!description) {
                    debugLog('createDraft: description missing');
                    alert('Bitte beschreiben Sie das gewünschte Dokument');
                    return;
                }

                const button = event.target;
                const originalText = button.textContent;
                button.disabled = true;
                button.textContent = '✍️ Generiere Entwurf...';
                debugLog('createDraft: sending POST /generate', { description });

                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ description })
                    });
                    debugLog('createDraft: response status', response.status);
                    const data = await response.json();
                    debugLog('createDraft: response body', data);

                    if (response.ok) {
                        debugLog('createDraft: generation successful');
                        displayDraft(data);
                    } else {
                        debugError('createDraft: generation failed', data);
                        alert(`❌ Fehler: ${data.detail || 'Generierung fehlgeschlagen'}`);
                    }
                } catch (error) {
                    debugError('createDraft: request error', error);
                    alert(`❌ Fehler: ${error.message}`);
                } finally {
                    debugLog('createDraft: restoring button state');
                    button.disabled = false;
                    button.textContent = originalText;
                }
            }

            function displayDraft(data) {
                debugLog('displayDraft: rendering draft modal', data);
                const modal = document.createElement('div');
                modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;';

                const content = document.createElement('div');
                content.style.cssText = 'background: white; padding: 30px; border-radius: 10px; max-width: 900px; max-height: 85vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);';

                let usedHtml = '';
                if (data.used_documents && data.used_documents.length > 0) {
                    usedHtml = '<h3 style="margin-top: 20px; color: #2c3e50;">🗂️ Verwendete Dokumente:</h3><ul style="list-style: none; padding: 0;">';
                    data.used_documents.forEach(doc => {
                        usedHtml += `<li style="margin: 6px 0; color: #2c3e50;">${doc.filename}</li>`;
                    });
                    usedHtml += '</ul>';
                }

                content.innerHTML = `
                    <h2 style="color: #2c3e50; margin-bottom: 15px;">✍️ Entwurf</h2>
                    <div style="background: #eaf7ec; padding: 12px; border-radius: 5px; margin-bottom: 12px;">
                        <strong>Beschreibung:</strong> ${data.description}
                    </div>
                    <pre style="white-space: pre-wrap; line-height: 1.45; background: #f8f9fa; padding: 16px; border-radius: 6px; border: 1px solid #e1e4e8;">${data.draft_text}</pre>
                    ${usedHtml}
                    <div style="margin-top: 16px; display: flex; gap: 10px;">
                        <button onclick="navigator.clipboard.writeText(\`${(data.draft_text || '').replace(/`/g,'\\`')}\`)"
                                style="background: #2ecc71; color: white; border: none; padding: 10px 14px; border-radius: 5px; cursor: pointer;">Kopieren</button>
                        <button onclick="this.parentElement.parentElement.parentElement.remove()"
                                style="background: #3498db; color: white; border: none; padding: 10px 14px; border-radius: 5px; cursor: pointer;">Schließen</button>
                    </div>
                `;

                modal.appendChild(content);
                document.body.appendChild(modal);
                modal.onclick = (e) => { if (e.target === modal) modal.remove(); };
            }

            function displayResearchResults(data) {
                debugLog('displayResearchResults: showing results', { query: data.query, sourceCount: (data.sources || []).length });
                window.latestResearchSources = Array.isArray(data.sources) ? data.sources : [];
                window.latestResearchQuery = data.query || '';
                const modal = document.createElement('div');
                modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;';

                const content = document.createElement('div');
                content.style.cssText = 'background: white; padding: 30px; border-radius: 10px; max-width: 900px; max-height: 85vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);';

                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = '<h3 style="margin-top: 20px; color: #2c3e50;">📚 Relevante Quellen:</h3>';
                    sourcesHtml += '<div style="color: #7f8c8d; font-size: 13px; margin-bottom: 12px;">💾 Hinweis: Hochwertige Quellen werden automatisch als PDF gespeichert und erscheinen in "Gespeicherte Quellen"</div>';
                    sourcesHtml += '<div style="display: flex; flex-direction: column; gap: 15px;">';

                    data.sources.forEach((source, index) => {
                        const description = source.description || 'Relevante Quelle für Ihre Recherche';
                        const canAddToSources = !!source.pdf_url;
                        const addButton = `<button onclick="addSourceFromResults(event, ${index})" style="display: inline-block; background: #27ae60; color: white; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; font-size: 13px; font-weight: 500;">➕ Zu gespeicherten Quellen</button>`;
                        const pdfLink = source.pdf_url || (source.url && source.url.toLowerCase().endsWith('.pdf') ? source.url : null);
                        if (pdfLink && !source.pdf_url) {
                            source.pdf_url = pdfLink;
                        }
                        const pdfButton = pdfLink
                            ? `<a href="${pdfLink}" target="_blank" style="display: inline-block; background: #2ecc71; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 13px; font-weight: 500;">📄 PDF öffnen</a>`
                            : '';
                        const pdfBadge = pdfLink ? '<span style="color: #2ecc71; font-size: 12px; font-weight: 600; margin-left: 8px;">📄 PDF erkannt</span>' : '';
                        sourcesHtml += `
                            <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
                                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                                    <a href="${source.url}" target="_blank" style="color: #2c3e50; text-decoration: none; font-weight: 600; font-size: 15px; flex: 1;">
                                        ${index + 1}. ${source.title}
                                    </a>
                                    ${pdfBadge}
                                </div>
                                <p style="color: #555; margin: 8px 0 10px 0; line-height: 1.5; font-size: 14px;">${description}</p>
                                <div style="display: flex; gap: 8px; align-items: center;">
                                    <a href="${source.url}" target="_blank"
                                       style="display: inline-block; background: #3498db; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 13px; font-weight: 500;">
                                        🔗 Zur Quelle
                                    </a>
                                    ${pdfButton}
                                    ${addButton}
                                    <span style="color: #7f8c8d; font-size: 12px;">
                                        ${source.url.length > 50 ? source.url.substring(0, 50) + '...' : source.url}
                                    </span>
                                </div>
                            </div>
                        `;
                    });

                    sourcesHtml += '</div>';
                } else {
                    sourcesHtml = '<p style="color: #7f8c8d; margin-top: 20px;">Keine Quellen gefunden.</p>';
                }

                const summaryHtml = data.summary || '';

                content.innerHTML = `
                    <h2 style="color: #2c3e50; margin-bottom: 15px;">🔍 Rechercheergebnisse</h2>
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                        <strong>Ihre Anfrage:</strong> ${data.query}
                    </div>
                    ${summaryHtml ? `<div style=\"margin-bottom: 20px;\"><strong>Zusammenfassung:</strong><div style=\"margin-top: 10px; line-height: 1.6;\">${summaryHtml}</div></div>` : ''}
                    ${sourcesHtml}
                    <div style="margin-top: 20px; display: flex; gap: 10px;">
                        <button onclick="loadSources(); this.parentElement.parentElement.parentElement.remove();"
                                style="background: #27ae60; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: 500;">
                            📥 Zu gespeicherten Quellen
                        </button>
                        <button onclick="this.parentElement.parentElement.parentElement.remove()"
                                style="background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: 500;">
                            Schließen
                        </button>
                    </div>
                `;

                modal.appendChild(content);
                document.body.appendChild(modal);

                modal.onclick = (e) => {
                    if (e.target === modal) modal.remove();
                };
            }
        </script>
    </body>
    </html>
    """

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

        # Save classification to database
        # Check if document already exists (by filename)
        existing_doc = db.query(Document).filter(Document.filename == result.filename).first()
        if existing_doc:
            # Update existing document
            existing_doc.category = result.category.value
            existing_doc.confidence = result.confidence
            existing_doc.explanation = result.explanation
            existing_doc.file_path = str(stored_path)
        else:
            # Create new document
            new_doc = Document(
                filename=result.filename,
                category=result.category.value,
                confidence=result.confidence,
                explanation=result.explanation,
                file_path=str(stored_path)
            )
            db.add(new_doc)

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
        "Sonstiges": []
    }

    for doc in documents:
        category = doc.category if doc.category in grouped else 'Sonstiges'
        grouped[category].append(doc.to_dict())

    return grouped

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

def _collect_context_attachments(include_categories: Optional[List[DocumentCategory]], max_docs: int, db: Session):
    """Collect uploaded PDFs as Claude message attachments and return (attachments, used_docs)."""
    used_docs: List[Dict[str, str]] = []
    attachments = []

    # Build query
    query = db.query(Document)

    # Optional filter by categories
    if include_categories:
        include_set = set([c.value if isinstance(c, DocumentCategory) else c for c in include_categories])
        query = query.filter(Document.category.in_(include_set))

    # Sort by newest first and limit
    documents = query.order_by(desc(Document.created_at)).limit(max_docs).all()

    for doc in documents:
        if not doc.file_path:
            continue
        p = Path(doc.file_path)
        if not p.exists():
            continue
        try:
            with open(p, 'rb') as f:
                data_b64 = base64.b64encode(f.read()).decode('utf-8')
            attachments.append({
                'name': doc.filename,
                'media_type': 'application/pdf',
                'data': data_b64
            })
            used_docs.append({'filename': doc.filename, 'file_path': doc.file_path})
        except Exception as e:
            print(f"Failed to attach {p}: {e}")
            continue

    return attachments, used_docs

@app.post("/generate", response_model=GenerationResponse)
@limiter.limit("10/hour")
async def generate(request: Request, body: GenerationRequest, db: Session = Depends(get_db)):
    """Generate a legal draft using Claude with uploaded PDFs as context."""
    if not body.description or not body.description.strip():
        raise HTTPException(status_code=400, detail="Description is required")

    try:
        client = get_anthropic_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Collect context
    attachments, used_docs = _collect_context_attachments(body.include_categories, body.max_context_docs, db)

    system_prompt = (
        "Du bist ein juristischer Assistent für deutsches Asylrecht. "
        "Erstelle einen präzisen, wohlstrukturierten Entwurf (Deutsch) auf Basis der bereitgestellten Dokumente. "
        "Beachte: der Entwurf ist ein Vorschlag – keine Rechtsberatung. Verwende klare Überschriften, kurze Absätze "
        "und, falls sinnvoll, nummerierte Argumente. Zitiere Gesetze/Urteile im Text (kurz) und schlage Fußnoten vor."
    )

    user_prompt = (
        "Aufgabe:\n" + body.description.strip() + "\n\n"
        "Kontext: Es sind PDF-Dokumente beigefügt (max. " + str(len(attachments)) + "). "
        "Nutze sie, wenn relevant. Fehlen wichtige Details, mache plausible Annahmen und weise darauf hin."
    )

    try:
        msg = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            max_tokens=2000,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ],
            attachments=attachments if attachments else None,
            temperature=0.2
        )
        # Extract text
        pieces = []
        for block in msg.content:
            try:
                btype = getattr(block, 'type', None)
                btext = getattr(block, 'text', None)
                if btype == 'text' and btext:
                    pieces.append(btext)
                elif isinstance(block, dict) and block.get('type') == 'text' and 'text' in block:
                    pieces.append(block['text'])
            except Exception:
                continue
        draft_text = "\n\n".join(pieces).strip() or "(Kein Text erzeugt)"

        return GenerationResponse(description=body.description, draft_text=draft_text, used_documents=used_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

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
    return {
        "count": len(sources_dict),
        "sources": sources_dict
    }

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

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
from openai import OpenAI
from enum import Enum
from google import genai
from google.genai import types
import httpx
import base64
import anthropic
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])

app = FastAPI(title="Rechtmaschine Document Classifier")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.source_subscribers = set()

# Storage file paths
CLASSIFICATIONS_FILE = Path("/app/classifications.json")
SOURCES_FILE = Path("/app/research_sources.json")
DOWNLOADS_DIR = Path("/app/downloaded_sources")
UPLOADS_DIR = Path("/app/uploads")

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
    sources: List[Dict[str, str]] = []  # List of {"title": "...", "url": "..."}

class SourceQualityScore(BaseModel):
    url: str
    score: int  # 1-5
    document_type: str  # "Gerichtsentscheidung", "Gesetz", "Fachpublikation", "Behördendokument", "Sonstiges"
    reasoning: str

class SavedSource(BaseModel):
    id: str
    url: str
    title: str
    quality_score: int
    document_type: str
    download_path: Optional[str] = None
    download_status: str = "pending"  # "pending", "downloading", "completed", "failed"
    research_query: str
    timestamp: str


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

# Storage functions
def load_classifications() -> List[Dict]:
    """Load classifications from JSON file"""
    if not CLASSIFICATIONS_FILE.exists():
        return []
    try:
        with open(CLASSIFICATIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading classifications: {e}")
        return []

def save_classification(result: ClassificationResult) -> None:
    """Save a classification result to JSON file"""
    classifications = load_classifications()

    # Remove existing entry for same filename if exists
    classifications = [c for c in classifications if c['filename'] != result.filename]

    # Add new classification
    classifications.append({
        'filename': result.filename,
        'category': result.category.value,
        'confidence': result.confidence,
        'explanation': result.explanation,
        'timestamp': datetime.now().isoformat()
    })

    try:
        with open(CLASSIFICATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(classifications, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving classification: {e}")

def delete_classification(filename: str) -> bool:
    """Delete a classification by filename"""
    classifications = load_classifications()
    original_length = len(classifications)

    # If present, delete associated uploaded file
    to_delete = next((c for c in classifications if c.get('filename') == filename), None)
    if to_delete and to_delete.get('file_path'):
        try:
            fp = Path(to_delete['file_path'])
            if fp.exists():
                fp.unlink()
        except Exception as e:
            print(f"Error deleting uploaded file for {filename}: {e}")

    classifications = [c for c in classifications if c['filename'] != filename]

    if len(classifications) == original_length:
        return False  # Nothing was deleted

    try:
        with open(CLASSIFICATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(classifications, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error deleting classification: {e}")
        return False

# Source storage functions
def load_sources() -> List[Dict]:
    """Load saved sources from JSON file"""
    print(f"load_sources called. SOURCES_FILE: {SOURCES_FILE}")
    print(f"File exists: {SOURCES_FILE.exists()}")
    if not SOURCES_FILE.exists():
        print("Sources file does not exist, returning empty list")
        return []
    try:
        with open(SOURCES_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"File content length: {len(content)} bytes")
            sources = json.loads(content)
            print(f"Loaded {len(sources)} sources from file")
            return sources
    except Exception as e:
        print(f"Error loading sources: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_source(source: SavedSource) -> None:
    """Save a research source to JSON file"""
    sources = load_sources()

    # Remove existing entry with same ID if exists
    sources = [s for s in sources if s.get('id') != source.id]

    # Add new source
    sources.append(source.model_dump())

    try:
        with open(SOURCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)
        _notify_sources_updated({
            'source_id': source.id,
            'status': source.download_status
        })
    except Exception as e:
        print(f"Error saving source: {e}")

def delete_source(source_id: str) -> bool:
    """Delete a saved source and its downloaded file"""
    sources = load_sources()
    source_to_delete = next((s for s in sources if s.get('id') == source_id), None)

    if not source_to_delete:
        return False

    # Delete downloaded file if exists
    if source_to_delete.get('download_path'):
        download_path = Path(source_to_delete['download_path'])
        if download_path.exists():
            try:
                download_path.unlink()
            except Exception as e:
                print(f"Error deleting file {download_path}: {e}")

    # Remove from JSON
    sources = [s for s in sources if s.get('id') != source_id]

    try:
        with open(SOURCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)
        _notify_sources_updated({
            'source_id': source_id,
            'status': 'deleted'
        })
        return True
    except Exception as e:
        print(f"Error deleting source: {e}")
        return False

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

        # Build the research prompt for finding relevant sources
        prompt = f"""Du bist ein Rechercheassistent für deutsches Asylrecht.

Recherchiere hochwertige, verlässliche Quellen für folgende Anfrage:

{query}

WICHTIG: Priorisiere folgende Quellentypen:
- Gerichtsentscheidungen (VG, OVG, BVerwG, EuGH, EGMR)
- Offizielle BAMF-Dokumente und Länderfeststellungen
- Gesetzestexte und Verordnungen (AsylG, AufenthG, GG)
- Rechtswissenschaftliche Fachpublikationen
- Asyl.net, beck-online.de, dejure.org, bundesverwaltungsgericht.de
- Offizielle Behörden-Websites (.gov, .europa.eu)

VERMEIDE:
- Blogs und persönliche Meinungen
- Kommerzielle Beratungsseiten
- Nicht-verifizierte Quellen

Gib eine kurze, prägnante Zusammenfassung (2-3 Sätze) der wichtigsten rechtlichen Erkenntnisse auf Deutsch basierend auf den gefundenen Quellen."""

        # Make the API call with search grounding
        print("Calling Gemini API with grounding...")
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=0.3
            )
        )
        print("Gemini API call successful")

        # Extract summary
        summary = response.text if response.text else "Keine Rechercheergebnisse gefunden."
        print(f"Summary extracted: {summary[:100]}...")

        # Extract sources from grounding metadata
        sources = []
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                grounding_meta = candidate.grounding_metadata

                # Extract grounding chunks (search results with titles and URLs)
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
                                sources.append(source)

        print(f"Extracted {len(sources)} sources from grounding metadata")

        # Start background task for scoring and downloading AFTER returning results
        import uuid
        import asyncio
        # Keep a reference to prevent garbage collection
        if len(sources) > 0:
            print(f"Creating background task to process {len(sources)} sources...")
            task = asyncio.create_task(process_sources_in_background(sources, query, client))
            print(f"Background task created: {task}")
        else:
            print("No sources found to process in background")
            task = None
        # Store task reference in app state to prevent GC
        if task is not None:
            if not hasattr(app.state, 'background_tasks'):
                app.state.background_tasks = set()
            app.state.background_tasks.add(task)
            task.add_done_callback(app.state.background_tasks.discard)

        return ResearchResult(
            query=query,
            summary=summary,
            sources=sources
        )

    except Exception as e:
        import traceback
        print(f"ERROR in research_with_gemini: {e}")
        print(traceback.format_exc())
        raise Exception(f"Research failed: {e}")

async def process_sources_in_background(sources: List[Dict], query: str, client):
    """Background task to score and download sources without blocking the response"""
    import uuid
    import asyncio
    import traceback

    try:
        print(f"Background: Scoring {len(sources)} sources...")

        for source in sources:
            try:
                # Score the source
                quality_score = await score_source_quality(
                    source['url'],
                    source['title'],
                    client
                )
                print(f"Background: Source scored: {source['title']} - Score: {quality_score.score}")

                # Save sources with score >= 3
                if quality_score.score >= 3:
                    source_id = str(uuid.uuid4())
                    saved_source = SavedSource(
                        id=source_id,
                        url=source['url'],
                        title=source['title'],
                        quality_score=quality_score.score,
                        document_type=quality_score.document_type,
                        download_status="pending" if quality_score.score >= 4 else "skipped",
                        research_query=query,
                        timestamp=datetime.now().isoformat()
                    )

                    save_source(saved_source)
                    print(f"Background: Saved source: {source['title']}")

                    # Download in background if score >= 4
                    if quality_score.score >= 4:
                        asyncio.create_task(download_and_update_source(source_id, source['url'], source['title']))

            except Exception as e:
                print(f"Background: Error processing source {source.get('title', 'unknown')}: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"Background: Fatal error in process_sources_in_background: {e}")
        traceback.print_exc()

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
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {"q": query}
            if court:
                params["court"] = court
            if date_range:
                params["date"] = date_range

            response = await client.get("https://de.openlegaldata.io/api/cases/", params=params)
            response.raise_for_status()
            data = response.json()

            sources = []
            results = data.get("results", [])[:5]  # Limit to top 5
            for case in results:
                sources.append({
                    "title": case.get("title", "Open Legal Data Case"),
                    "url": case.get("url", f"https://de.openlegaldata.io/case/{case.get('slug', '')}")
                })

            print(f"Open Legal Data returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching Open Legal Data: {e}")
        return []

async def search_refworld(query: str, country: Optional[str] = None, doc_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search Refworld (UNHCR) using Playwright scraping.
    Args:
        query: Search query
        country: Optional country filter
        doc_type: Optional document type filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching Refworld: query='{query}', country={country}, doc_type={doc_type}")
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Build search URL
            search_url = f"https://www.refworld.org/search?query={query}"
            if country:
                search_url += f"&country={country}"

            await page.goto(search_url, wait_until="networkidle", timeout=30000)

            # Extract search results
            sources = []
            all_results = await page.query_selector_all(".search-result-item")
            results = all_results[:5]

            for result in results:
                title_elem = await result.query_selector("h3, .title, a")
                link_elem = await result.query_selector("a")

                if title_elem and link_elem:
                    title = await title_elem.text_content()
                    url = await link_elem.get_attribute("href")

                    if url and not url.startswith("http"):
                        url = f"https://www.refworld.org{url}"

                    sources.append({"title": title.strip(), "url": url})

            await browser.close()
            print(f"Refworld returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching Refworld: {e}")
        return []

async def search_euaa_coi(query: str, country: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search EUAA COI Portal using Playwright scraping.
    Args:
        query: Search query
        country: Optional country filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching EUAA COI: query='{query}', country={country}")
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Build search URL
            search_url = f"https://coi.euaa.europa.eu/search?q={query}"
            if country:
                search_url += f"&country={country}"

            await page.goto(search_url, wait_until="networkidle", timeout=30000)

            # Extract search results
            sources = []
            all_results = await page.query_selector_all(".result-item, .coi-result")
            results = all_results[:5]

            for result in results:
                title_elem = await result.query_selector("h3, .title, a")
                link_elem = await result.query_selector("a")

                if title_elem and link_elem:
                    title = await title_elem.text_content()
                    url = await link_elem.get_attribute("href")

                    if url and not url.startswith("http"):
                        url = f"https://coi.euaa.europa.eu{url}"

                    sources.append({"title": title.strip(), "url": url})

            await browser.close()
            print(f"EUAA COI returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching EUAA COI: {e}")
        return []

async def search_asyl_net(query: str, category: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search asyl.net Rechtsprechungs-Datenbank using Playwright scraping.
    Args:
        query: Search query
        category: Optional category filter (e.g., "Dublin", "EGMR")
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching asyl.net: query='{query}', category={category}")
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to database search
            search_url = f"https://www.asyl.net/rsdb?search={query}"
            if category:
                search_url += f"&category={category}"

            await page.goto(search_url, wait_until="networkidle", timeout=30000)

            # Extract search results
            sources = []
            all_results = await page.query_selector_all(".decision-item, .result")
            results = all_results[:5]

            for result in results:
                title_elem = await result.query_selector("h3, .title, a")
                link_elem = await result.query_selector("a")

                if title_elem and link_elem:
                    title = await title_elem.text_content()
                    url = await link_elem.get_attribute("href")

                    if url and not url.startswith("http"):
                        url = f"https://www.asyl.net{url}"

                    sources.append({"title": title.strip(), "url": url})

            await browser.close()
            print(f"asyl.net returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching asyl.net: {e}")
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

        # Define function declarations for Gemini
        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_open_legal_data",
                        description="Search Open Legal Data API for German case law. Use this for finding court decisions from BGH, BVerwG, OVG, VG, etc.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(type=types.Type.STRING, description="Search query for case law"),
                                "court": types.Schema(type=types.Type.STRING, description="Optional court filter (e.g., 'BGH', 'BVerwG', 'OVG', 'VG')"),
                                "date_range": types.Schema(type=types.Type.STRING, description="Optional date range (e.g., '2020-2025')")
                            },
                            required=["query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_refworld",
                        description="Search Refworld (UNHCR) for refugee case law and country of origin information. Use this for international refugee law and COI.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(type=types.Type.STRING, description="Search query"),
                                "country": types.Schema(type=types.Type.STRING, description="Optional country filter (e.g., 'Somalia', 'Afghanistan', 'Syria')"),
                                "doc_type": types.Schema(type=types.Type.STRING, description="Optional document type")
                            },
                            required=["query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_euaa_coi",
                        description="Search EUAA COI Portal for European asylum country of origin information. Use for EU-specific COI reports.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(type=types.Type.STRING, description="Search query"),
                                "country": types.Schema(type=types.Type.STRING, description="Optional country filter")
                            },
                            required=["query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_asyl_net",
                        description="Search asyl.net Rechtsprechungs-Datenbank for German asylum law decisions. Comprehensive database of German asylum case law.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(type=types.Type.STRING, description="Search query"),
                                "category": types.Schema(type=types.Type.STRING, description="Optional category (e.g., 'Dublin', 'EGMR')")
                            },
                            required=["query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_bamf",
                        description="Search BAMF (Bundesamt für Migration und Flüchtlinge) official documents and guidelines. Use for official German asylum authority information.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(type=types.Type.STRING, description="Search query"),
                                "topic": types.Schema(type=types.Type.STRING, description="Optional topic filter")
                            },
                            required=["query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_edal",
                        description="Search EDAL (European Database of Asylum Law) for European asylum case law up to 2021. Use for EU Member States case law.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(type=types.Type.STRING, description="Search query"),
                                "country": types.Schema(type=types.Type.STRING, description="Optional country filter"),
                                "court": types.Schema(type=types.Type.STRING, description="Optional court filter")
                            },
                            required=["query"]
                        )
                    )
                ]
            )
        ]

        # Build the research prompt
        prompt = f"""Du bist ein Rechercheassistent für deutsches Asylrecht mit Zugriff auf spezialisierte Rechtsdatenbanken.

Analysiere folgende Anfrage und entscheide, welche Datenbanken du durchsuchen solltest:

{query}

WICHTIG:
- Wähle die 2-4 relevantesten Datenbanken für diese Anfrage
- Formuliere präzise Suchbegriffe für jede Datenbank
- Extrahiere relevante Parameter wie Länder, Gerichte, Kategorien aus der Anfrage
- Nutze Open Legal Data und asyl.net für deutsche Rechtsprechung
- Nutze Refworld und EUAA COI für Länderinformationen
- Nutze BAMF für offizielle deutsche Behördeninfos
- Nutze EDAL für europäische Rechtsprechung

Führe die Suchen aus und gib dann eine kurze Zusammenfassung (2-3 Sätze) der wichtigsten Erkenntnisse auf Deutsch."""

        # Make the API call with function calling
        print("Calling Gemini API with function calling...")
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=tools,
                temperature=0.3
            )
        )

        # Handle function calls
        all_sources = []
        function_calls_made = []

        if hasattr(response.candidates[0].content, 'parts'):
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    function_name = fc.name
                    function_args = dict(fc.args)

                    print(f"Gemini called function: {function_name} with args: {function_args}")
                    function_calls_made.append(f"{function_name}({function_args})")

                    # Execute the function call
                    if function_name == "search_open_legal_data":
                        sources = await search_open_legal_data(**function_args)
                        all_sources.extend(sources)
                    elif function_name == "search_refworld":
                        sources = await search_refworld(**function_args)
                        all_sources.extend(sources)
                    elif function_name == "search_euaa_coi":
                        sources = await search_euaa_coi(**function_args)
                        all_sources.extend(sources)
                    elif function_name == "search_asyl_net":
                        sources = await search_asyl_net(**function_args)
                        all_sources.extend(sources)
                    elif function_name == "search_bamf":
                        sources = await search_bamf(**function_args)
                        all_sources.extend(sources)
                    elif function_name == "search_edal":
                        sources = await search_edal(**function_args)
                        all_sources.extend(sources)

        print(f"Legal databases returned {len(all_sources)} total sources from {len(function_calls_made)} database(s)")

        # Generate summary from collected sources
        if len(all_sources) > 0:
            summary_prompt = f"""Basierend auf {len(all_sources)} gefundenen Quellen aus spezialisierten Rechtsdatenbanken zur Anfrage "{query}":

Gib eine kurze, prägnante Zusammenfassung (2-3 Sätze) der wichtigsten rechtlichen Erkenntnisse auf Deutsch."""

            summary_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-09-2025",
                contents=summary_prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
            summary = summary_response.text if summary_response.text else "Quellen gefunden in spezialisierten Rechtsdatenbanken."
        else:
            summary = "Keine relevanten Quellen in den Rechtsdatenbanken gefunden."

        print(f"Generated summary: {summary[:100]}...")

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
    try:
        # Update status to downloading
        sources = load_sources()
        for source in sources:
            if source.get('id') == source_id:
                source['download_status'] = 'downloading'
                break
        with open(SOURCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)

        # Download the PDF
        download_path = await download_source_as_pdf(url, title)

        # Update status
        sources = load_sources()
        for source in sources:
            if source.get('id') == source_id:
                if download_path:
                    source['download_status'] = 'completed'
                    source['download_path'] = download_path
                else:
                    source['download_status'] = 'failed'
                break

        with open(SOURCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error in background download for {url}: {e}")
        # Mark as failed
        sources = load_sources()
        for source in sources:
            if source.get('id') == source_id:
                source['download_status'] = 'failed'
                break
        with open(SOURCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)

async def score_source_quality(url: str, title: str, client) -> SourceQualityScore:
    """
    Score a source's quality using Gemini.
    Returns score 1-5 and document type classification.
    """
    prompt = f"""Bewerte die Qualität dieser Quelle für deutsches Asylrecht:

URL: {url}
Titel: {title}

Bewertungskriterien (Score 1-5):
- Score 5: Gerichtsentscheidungen (VG, OVG, BVerwG, EuGH, EGMR), offizielle Gesetzestexte
- Score 4: BAMF-Dokumente, asyl.net, dejure.org, beck-online.de, offizielle Behörden-Websites
- Score 3: Rechtswissenschaftliche Fachpublikationen, Universitätsquellen
- Score 2: Allgemeine Nachrichtenportale, Wikipedia
- Score 1: Blogs, kommerzielle Beratungsseiten, nicht-verifizierte Quellen

Dokumenttyp:
- Gerichtsentscheidung
- Gesetz
- Fachpublikation
- Behördendokument
- Sonstiges

Gib Score, Dokumenttyp und kurze Begründung zurück."""

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-09-2025",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=SourceQualityScore,
            temperature=0.0
        )
    )

    return SourceQualityScore.model_validate_json(response.text)

async def download_source_as_pdf(url: str, filename: str) -> Optional[str]:
    """
    Download a source as PDF. Handles both direct PDFs and HTML pages.
    Returns the path to the downloaded file, or None if failed.
    """
    import asyncio
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
        async with httpx.AsyncClient() as http_client:
            head_response = await http_client.head(url, follow_redirects=True, timeout=10.0)
            content_type = head_response.headers.get('content-type', '').lower()

            if 'application/pdf' in content_type:
                # Direct PDF download
                print(f"Downloading PDF directly: {url}")
                response = await http_client.get(url, follow_redirects=True, timeout=30.0)
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
            // Load documents and sources on page load
            window.addEventListener('DOMContentLoaded', () => {
                loadDocuments();
                loadSources();
            });

            async function loadDocuments() {
                try {
                    const response = await fetch('/documents');
                    const data = await response.json();

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
                        const boxId = categoryMap[category];
                        const box = document.getElementById(boxId);

                        if (box) {
                            if (documents.length === 0) {
                                box.innerHTML = '<div class="empty-message">Keine Dokumente</div>';
                            } else {
                                box.innerHTML = documents.map(doc => createDocumentCard(doc)).join('');
                            }
                        }
                    }
                } catch (error) {
                    console.error('Error loading documents:', error);
                }
            }

            function createDocumentCard(doc) {
                const confidence = (doc.confidence * 100).toFixed(0);
                const escapedFilename = doc.filename.replace(/'/g, "\\'").replace(/"/g, '&quot;');
                return `
                    <div class="document-card">
                        <button class="delete-btn" onclick="deleteDocument('${escapedFilename}')" title="Löschen">×</button>
                        <div class="filename">${doc.filename}</div>
                        <div class="confidence">${confidence}% Konfidenz</div>
                    </div>
                `;
            }

            async function loadSources() {
                try {
                    console.log('Loading sources...');
                    const response = await fetch('/sources');
                    console.log('Sources response status:', response.status);
                    const sources = await response.json();
                    console.log('Sources loaded:', sources.length, sources);

                    const container = document.getElementById('sonstiges-docs');
                    console.log('Container element:', container);

                    if (sources.length === 0) {
                        container.innerHTML = '<div class="empty-message">Keine Quellen gespeichert</div>';
                    } else {
                        container.innerHTML = sources.map(source => createSourceCard(source)).join('');
                        console.log('Sources rendered to container');
                    }
                } catch (error) {
                    console.error('Error loading sources:', error);
                }
            }

            function createSourceCard(source) {
                const statusEmoji = {
                    'completed': '✅',
                    'downloading': '⏳',
                    'pending': '📥',
                    'failed': '❌',
                    'skipped': '⏭️'
                }[source.download_status] || '📄';

                const downloadButton = source.download_status === 'completed'
                    ? `<a href="/sources/download/${source.id}" target="_blank" style="color: white; text-decoration: none; font-size: 12px;">📥 PDF</a>`
                    : '';

                return `
                    <div class="document-card">
                        <button class="delete-btn" onclick="deleteSource('${source.id}')" title="Löschen">×</button>
                        <div class="filename">
                            <a href="${source.url}" target="_blank" style="color: inherit; text-decoration: none;">
                                ${source.title}
                            </a>
                            ${downloadButton}
                        </div>
                        <div class="confidence">
                            ${statusEmoji} Score: ${source.quality_score}/5 | ${source.document_type}
                        </div>
                    </div>
                `;
            }

            async function deleteSource(sourceId) {
                if (!confirm('Quelle wirklich löschen?')) return;

                try {
                    const response = await fetch(`/sources/${sourceId}`, {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        loadSources();
                    } else {
                        const data = await response.json();
                        alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
                    }
                } catch (error) {
                    alert(`❌ Fehler: ${error.message}`);
                }
            }

            async function deleteAllSources() {
                if (!confirm('Wirklich ALLE gespeicherten Quellen löschen? Diese Aktion kann nicht rückgängig gemacht werden.')) return;

                try {
                    const response = await fetch('/sources', {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        const data = await response.json();
                        alert(`✅ ${data.count} Quellen gelöscht`);
                        loadSources();
                    } else {
                        const data = await response.json();
                        alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
                    }
                } catch (error) {
                    alert(`❌ Fehler: ${error.message}`);
                }
            }

            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];

                if (!file) {
                    alert('Bitte wählen Sie eine PDF-Datei aus');
                    return;
                }

                const loading = document.getElementById('loading');
                loading.style.display = 'block';

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/classify', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        // Reload documents to show the new one
                        await loadDocuments();
                        // Clear file input
                        fileInput.value = '';
                        // Show success message
                        alert(`✅ Dokument klassifiziert als: ${data.category} (${(data.confidence * 100).toFixed(0)}%)`);
                    } else {
                        alert(`❌ Fehler: ${data.detail || 'Unbekannter Fehler'}`);
                    }
                } catch (error) {
                    alert(`❌ Fehler: ${error.message}`);
                } finally {
                    loading.style.display = 'none';
                }
            }

            async function deleteDocument(filename) {
                if (!confirm(`Möchten Sie "${filename}" wirklich löschen?`)) {
                    return;
                }

                try {
                    const response = await fetch(`/documents/${encodeURIComponent(filename)}`, {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        // Reload documents
                        await loadDocuments();
                    } else {
                        const data = await response.json();
                        alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
                    }
                } catch (error) {
                    alert(`❌ Fehler: ${error.message}`);
                }
            }

            async function generateDocument() {
                const description = document.getElementById('outputDescription').value.trim();

                if (!description) {
                    alert('Bitte beschreiben Sie das gewünschte Dokument');
                    return;
                }

                // Show loading state
                const button = event.target;
                const originalText = button.textContent;
                button.disabled = true;
                button.textContent = '🔍 Recherchiere...';

                try {
                    const response = await fetch('/research', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query: description })
                    });

                    const data = await response.json();

                    if (response.ok) {
                        displayResearchResults(data);
                        loadSources();
                    } else {
                        alert(`❌ Fehler: ${data.detail || 'Recherche fehlgeschlagen'}`);
                    }
                } catch (error) {
                    alert(`❌ Fehler: ${error.message}`);
                } finally {
                    button.disabled = false;
                    button.textContent = originalText;
                }
            }

            async function createDraft() {
                const description = document.getElementById('outputDescription').value.trim();

                if (!description) {
                    alert('Bitte beschreiben Sie das gewünschte Dokument');
                    return;
                }

                const button = event.target;
                const originalText = button.textContent;
                button.disabled = true;
                button.textContent = '✍️ Generiere Entwurf...';

                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ description })
                    });
                    const data = await response.json();

                    if (response.ok) {
                        displayDraft(data);
                    } else {
                        alert(`❌ Fehler: ${data.detail || 'Generierung fehlgeschlagen'}`);
                    }
                } catch (error) {
                    alert(`❌ Fehler: ${error.message}`);
                } finally {
                    button.disabled = false;
                    button.textContent = originalText;
                }
            }

            function displayDraft(data) {
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
                const modal = document.createElement('div');
                modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;';

                const content = document.createElement('div');
                content.style.cssText = 'background: white; padding: 30px; border-radius: 10px; max-width: 800px; max-height: 80vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);';

                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = '<h3 style="margin-top: 20px; color: #2c3e50;">📚 Relevante Quellen:</h3><ul style="list-style: none; padding: 0;">';
                    data.sources.forEach(source => {
                        sourcesHtml += `<li style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                            <a href="${source.url}" target="_blank" style="color: #3498db; text-decoration: none; font-weight: bold;">${source.title}</a>
                            <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">${source.url}</div>
                        </li>`;
                    });
                    sourcesHtml += '</ul>';
                } else {
                    sourcesHtml = '<p style="color: #7f8c8d; margin-top: 20px;">Keine Quellen gefunden.</p>';
                }

                content.innerHTML = `
                    <h2 style="color: #2c3e50; margin-bottom: 15px;">🔍 Rechercheergebnisse</h2>
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                        <strong>Ihre Anfrage:</strong> ${data.query}
                    </div>
                    <div style="margin-bottom: 20px;">
                        <strong>Zusammenfassung:</strong>
                        <p style="margin-top: 10px; line-height: 1.6;">${data.summary}</p>
                    </div>
                    ${sourcesHtml}
                    <button onclick="this.parentElement.parentElement.remove()"
                            style="margin-top: 20px; background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                        Schließen
                    </button>
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
async def classify(request: Request, file: UploadFile = File(...)):
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
        # Save classification to storage
        # Augment classification entry with stored file path
        classifications = load_classifications()
        classifications = [c for c in classifications if c['filename'] != result.filename]
        classifications.append({
            'filename': result.filename,
            'category': result.category.value,
            'confidence': result.confidence,
            'explanation': result.explanation,
            'timestamp': datetime.now().isoformat(),
            'file_path': str(stored_path)
        })
        try:
            with open(CLASSIFICATIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(classifications, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving classification: {e}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

@app.get("/documents")
@limiter.limit("200/hour")
async def get_documents(request: Request):
    """Get all classified documents grouped by category"""
    classifications = load_classifications()

    # Group by category
    grouped = {
        "Anhörung": [],
        "Bescheid": [],
        "Rechtsprechung": [],
        "Sonstiges": []
    }

    for classification in classifications:
        category = classification.get('category', 'Sonstiges')
        if category in grouped:
            grouped[category].append(classification)

    return grouped

@app.delete("/documents/{filename}")
@limiter.limit("100/hour")
async def delete_document(request: Request, filename: str):
    """Delete a classified document"""
    success = delete_classification(filename)
    if success:
        return {"message": f"Document {filename} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Document {filename} not found")

@app.post("/research", response_model=ResearchResult)
@limiter.limit("10/hour")
async def research(request: Request, body: ResearchRequest):
    """Perform web research using both Gemini web search and specialized legal databases"""
    try:
        # Run both researchers in parallel
        print(f"Starting parallel research for query: {body.query}")

        web_search_task = asyncio.create_task(research_with_gemini(body.query))
        legal_db_task = asyncio.create_task(research_with_legal_databases(body.query))

        # Wait for both to complete
        web_result, legal_result = await asyncio.gather(web_search_task, legal_db_task, return_exceptions=True)

        # Handle errors gracefully
        all_sources = []
        summaries = []

        if isinstance(web_result, Exception):
            print(f"Web search failed: {web_result}")
        else:
            all_sources.extend(web_result.sources)
            summaries.append(f"Web-Recherche: {web_result.summary}")

        if isinstance(legal_result, Exception):
            print(f"Legal database search failed: {legal_result}")
        else:
            all_sources.extend(legal_result.sources)
            summaries.append(f"Rechtsdatenbanken: {legal_result.summary}")

        # Combine results
        combined_summary = " | ".join(summaries) if summaries else "Keine Rechercheergebnisse gefunden."

        print(f"Combined research returned {len(all_sources)} total sources")

        # Start background processing for all sources
        if len(all_sources) > 0:
            client = get_gemini_client()
            task = asyncio.create_task(process_sources_in_background(all_sources, body.query, client))
            if not hasattr(app.state, 'background_tasks'):
                app.state.background_tasks = set()
            app.state.background_tasks.add(task)
            task.add_done_callback(app.state.background_tasks.discard)

        return ResearchResult(
            query=body.query,
            summary=combined_summary,
            sources=all_sources
        )
    except Exception as e:
        import traceback
        print(f"ERROR in research endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Research failed: {e}")

def _collect_context_attachments(include_categories: Optional[List[DocumentCategory]], max_docs: int):
    """Collect uploaded PDFs as Claude message attachments and return (attachments, used_docs)."""
    used_docs: List[Dict[str, str]] = []
    attachments = []
    classifications = load_classifications()

    # Optional filter by categories
    if include_categories:
        include_set = set([c.value if isinstance(c, DocumentCategory) else c for c in include_categories])
        classifications = [c for c in classifications if c.get('category') in include_set]

    # Sort by newest first
    classifications.sort(key=lambda c: c.get('timestamp', ''), reverse=True)

    for c in classifications[:max_docs]:
        fp = c.get('file_path')
        if not fp:
            continue
        p = Path(fp)
        if not p.exists():
            continue
        try:
            with open(p, 'rb') as f:
                data_b64 = base64.b64encode(f.read()).decode('utf-8')
            attachments.append({
                'name': c.get('filename', p.name),
                'media_type': 'application/pdf',
                'data': data_b64
            })
            used_docs.append({'filename': c.get('filename', p.name), 'file_path': str(p)})
        except Exception as e:
            print(f"Failed to attach {p}: {e}")
            continue

    return attachments, used_docs

@app.post("/generate", response_model=GenerationResponse)
@limiter.limit("10/hour")
async def generate(request: Request, body: GenerationRequest):
    """Generate a legal draft using Claude with uploaded PDFs as context."""
    if not body.description or not body.description.strip():
        raise HTTPException(status_code=400, detail="Description is required")

    try:
        client = get_anthropic_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Collect context
    attachments, used_docs = _collect_context_attachments(body.include_categories, body.max_context_docs)

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

@app.get("/sources")
@limiter.limit("1000/hour")
async def get_sources(request: Request):
    """Get all saved research sources"""
    print("="*50)
    print("GET /sources endpoint called!")
    print("="*50)
    sources = load_sources()
    print(f"Returning {len(sources)} sources to client")
    return sources

@app.get("/sources/download/{source_id}")
@limiter.limit("50/hour")
async def download_source_file(request: Request, source_id: str):
    """Download a saved source PDF"""
    from fastapi.responses import FileResponse

    sources = load_sources()
    source = next((s for s in sources if s.get('id') == source_id), None)

    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    if not source.get('download_path'):
        raise HTTPException(status_code=404, detail="Source not downloaded yet")

    download_path = Path(source['download_path'])
    if not download_path.exists():
        raise HTTPException(status_code=404, detail="Downloaded file not found")

    return FileResponse(
        path=download_path,
        media_type='application/pdf',
        filename=f"{source.get('title', 'document')}.pdf"
    )

@app.delete("/sources/{source_id}")
@limiter.limit("100/hour")
async def delete_source_endpoint(request: Request, source_id: str):
    """Delete a saved source"""
    success = delete_source(source_id)
    if success:
        return {"message": f"Source {source_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

@app.delete("/sources")
@limiter.limit("50/hour")
async def delete_all_sources_endpoint(request: Request):
    """Delete all saved sources"""
    sources = load_sources()

    if not sources:
        return {"message": "No sources to delete", "count": 0}

    # Delete all downloaded files
    deleted_count = 0
    for source in sources:
        if source.get('download_path'):
            download_path = Path(source['download_path'])
            if download_path.exists():
                try:
                    download_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting file {download_path}: {e}")

    # Clear the sources file
    try:
        with open(SOURCES_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        _notify_sources_updated('all_sources_deleted', {'count': len(sources)})
        return {"message": f"All sources deleted successfully", "count": len(sources), "files_deleted": deleted_count}
    except Exception as e:
        print(f"Error clearing sources file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete sources: {e}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

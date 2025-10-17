# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rechtmaschine is an AI-powered legal document classification, research, and generation tool for German asylum lawyers. The project is deployed on a self-hosted server behind a Caddy reverse proxy at `rechtmaschine.de`.

**Current Status:** Fully functional with Gemini-powered document classification, automatic PDF segmentation for Akte files, web research with Google Search grounding, legal database integration (asyl.net), saved sources management, and document generation via Claude.

## Architecture

### Technology Stack
- **Backend:** FastAPI (Python 3.11) running in Docker
- **Database:** PostgreSQL (latest alpine) for persistent storage
- **AI Models:**
  - Google Gemini 2.5 Flash (document classification with structured output)
  - Google Gemini 2.5 Flash (automatic PDF segmentation for Akte files)
  - Google Gemini 2.5 Flash (web research with Google Search grounding)
  - Anthropic Claude 3.5 Sonnet (document generation)
- **Web Scraping:** Playwright (for asyl.net search and PDF detection)
- **Frontend:** Embedded HTML/CSS/JS (no separate framework yet)
- **Reverse Proxy:** Caddy (HTTPS with automatic certificates)
- **Deployment:** Docker Compose on self-hosted server

### Database Schema

**PostgreSQL Tables:**

1. **documents** - Classified legal documents
   - `id` (UUID, PK)
   - `filename` (VARCHAR, unique, indexed)
   - `category` (VARCHAR, indexed)
   - `confidence` (FLOAT)
   - `explanation` (TEXT)
   - `file_path` (VARCHAR)
   - `created_at` (TIMESTAMP, indexed)

2. **research_sources** - Saved legal research sources
   - `id` (UUID, PK)
   - `title` (VARCHAR)
   - `url` (TEXT)
   - `description` (TEXT)
   - `document_type` (VARCHAR, indexed)
   - `pdf_url` (TEXT)
   - `download_path` (VARCHAR)
   - `download_status` (VARCHAR, indexed)
   - `research_query` (TEXT)
   - `created_at` (TIMESTAMP, indexed)

3. **processed_documents** - Future: text extraction & anonymization
   - `id` (UUID, PK)
   - `document_id` (UUID, FK → documents.id, CASCADE)
   - `extracted_text` (TEXT)
   - `is_anonymized` (BOOLEAN)
   - `ocr_applied` (BOOLEAN)
   - `anonymization_metadata` (JSONB)
   - `processing_status` (VARCHAR)
   - `created_at` (TIMESTAMP)

### Document Classification Flow
1. User uploads PDF via web interface
2. PDF sent to `/classify` endpoint
3. File persisted to `/app/uploads` with timestamp
4. File uploaded to Gemini using `client.files.upload()`
5. Classification performed via Gemini `generate_content()` with structured output
   - Model: `gemini-2.5-flash-preview-09-2025`
   - Uses Pydantic schema for JSON response validation
   - Temperature: 0.0 for deterministic results
6. If classified as "Akte", automatic segmentation is triggered:
   - Creates subdirectory `/app/uploads/{filename}_segments/`
   - Invokes `segment_pdf_with_gemini()` from `kanzlei_gemini.py`
   - Extracts individual Anhörung and Bescheid documents
   - Each segment saved as separate PDF and database entry
7. Results saved to PostgreSQL `documents` table
8. Frontend displays document(s) in appropriate category boxes

### Web Research Flow
1. User enters query in research textarea
2. Backend `/research` endpoint runs two searches in parallel:
   - **Gemini with Google Search grounding**: Searches official sources (courts, government agencies, academic)
   - **asyl.net database**: Playwright-based scraping with keyword suggestions
3. Results combined and enriched:
   - First 10 sources processed with Playwright for PDF detection
   - URLs resolved (VertexAI redirects → actual destinations)
   - PDFs automatically detected from links and iframes
4. Summary displayed with structured source cards
5. High-quality sources (with PDFs) can be saved to "Gespeicherte Quellen"

### Saved Sources Management
1. Sources with PDF URLs can be added to saved sources
2. PDFs automatically downloaded in background via Playwright
3. Download status tracked: pending → downloading → completed/failed
4. Saved sources stored in PostgreSQL `research_sources` table
5. SSE (Server-Sent Events) for real-time download status updates
6. PDFs stored in `/app/downloaded_sources/`

### Document Generation Flow
1. User enters description and clicks "Entwurf generieren"
2. Backend `/generate` collects up to 4 recent uploaded PDFs as context
3. PDFs retrieved from database query (newest first)
4. PDFs base64-encoded and attached to Claude message
5. Claude (`claude-3-5-sonnet-20241022`) generates draft in German
6. Draft text returned and shown in modal (not persisted yet)

### Key Implementation Details

**Gemini Classification Pattern:**
```python
with open(pdf_path, "rb") as pdf_file:
    uploaded = client.files.upload(
        file=pdf_file,
        config={
            "mime_type": "application/pdf",
            "display_name": filename,
        },
    )

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
```

**Automatic Akte Segmentation:**
```python
from kanzlei_gemini import segment_pdf_with_gemini, GeminiConfig as SegmentationGeminiConfig

if result.category == DocumentCategory.AKTE:
    segment_dir = stored_path.parent / f"{stored_path.stem}_segments"
    sections, extracted_pairs = segment_pdf_with_gemini(
        str(stored_path),
        segment_dir,
        client=segment_client,
        config=SegmentationGeminiConfig(),
        verbose=False,
    )
    # Each extracted section is saved as a separate document in the database
```

**Gemini Google Search Grounding:**
```python
grounding_tool = types.Tool(google_search=types.GoogleSearch())
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-09-2025",
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.0
    )
)
# Sources extracted from response.candidates[0].grounding_metadata
```

**Playwright PDF Detection:**
- Processes first 10 sources (`max_checks=10`)
- HTTP-based detection first (fast)
- Playwright navigation for complex cases
- Resolves VertexAI redirect URLs to actual destinations
- Searches for: `<a href="*.pdf">`, `<iframe src="*.pdf">`
- Updates source URL and pdf_url in-place

**Document Categories:**
- Anhörung (hearing protocols)
- Bescheid (administrative decisions)
- Akte (complete BAMF case file / Beiakte)
- Rechtsprechung (case law)
- Sonstiges (other)

**Environment Variables:**
- `DATABASE_URL` - PostgreSQL connection string
- `POSTGRES_PASSWORD` - Database password
- `GOOGLE_API_KEY` - Required for classification, segmentation, and web search (Gemini)
- `ANTHROPIC_API_KEY` - Required for document generation (Claude)
- `OPENAI_API_KEY` - Legacy (no longer used for classification)

## Development Commands

### Running the Application

**Start the app:**
```bash
cd /var/opt/docker/rechtmaschine/app
docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f app
docker-compose logs -f postgres
```

**Restart after code changes:**
```bash
docker-compose restart app
```

**Note:** Hot reload is configured but doesn't work reliably with Docker volume mounts. Manual restart is required after changes to `app.py`, `database.py`, or `models.py`.

### Database Management

**Access PostgreSQL:**
```bash
docker exec -it rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db
```

**List tables:**
```bash
docker exec rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db -c "\dt"
```

**View table schema:**
```bash
docker exec rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db -c "\d documents"
```

**Clear all data (fresh session):**
```sql
TRUNCATE TABLE documents, research_sources, processed_documents CASCADE;
```

### Caddy Reverse Proxy

**Update Caddy configuration:**
After editing `/var/opt/docker/caddy/Caddyfile`, restart Caddy:
```bash
docker restart caddy
```

**Note:** `caddy reload` doesn't work reliably for bind-mounted config files. Always use `docker restart caddy`.

### Accessing the Application
- Main app: https://rechtmaschine.de
- n8n instance: https://n8n.rechtmaschine.de (configured but not actively used)

## File Structure

```
/var/opt/docker/rechtmaschine/
├── app/
│   ├── app.py              # Main FastAPI application (all backend + frontend HTML)
│   ├── database.py         # SQLAlchemy engine, session factory, Base
│   ├── models.py           # ORM models (Document, ResearchSource, ProcessedDocument)
│   ├── kanzlei_gemini.py   # Gemini-based PDF segmentation module
│   ├── kanzlei.py          # OpenAI-based segmentation (legacy, with chunking support)
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile          # Container definition
│   ├── docker-compose.yml  # Docker Compose configuration (app + postgres)
│   ├── .env               # API keys and database config (not in git)
│   ├── .env.example       # Example environment variables
│   └── data/              # Static data (asyl.net keyword suggestions)
├── kanzlei-gemini.py       # Standalone CLI script for PDF segmentation
├── CLAUDE.md               # Project documentation for Claude Code
└── README.md               # Project overview
```

**Runtime directories (inside container):**
- `/app/uploads/` - Uploaded classified PDFs
- `/app/uploads/{filename}_segments/` - Extracted documents from Akte files
- `/app/downloaded_sources/` - Downloaded research source PDFs

## Important Constraints

### Docker Network
The app connects to the external `caddy` network (not `caddy_default`). The Caddy reverse proxy is defined elsewhere on the server.

### Dependencies
- `google-genai==1.41.0` - Gemini API client for classification, segmentation, and web search
- `anthropic>=0.36.0` - Claude client for document generation
- `pikepdf==9.4.2` - PDF manipulation for segmentation and extraction
- `playwright==1.49.0` - Browser automation for web scraping and PDF detection
- `httpx>=0.28.1` - HTTP client for web requests
- `sqlalchemy` - ORM for PostgreSQL
- `psycopg2-binary` - PostgreSQL adapter
- `slowapi>=0.1.9` - Rate limiting
- `markdown>=3.6` - Markdown to HTML conversion
- `openai>=1.56.0` - Legacy dependency (retained in requirements but not actively used for classification)

### German Language Support
All UI text and prompts are in German. Category names use German umlauts which must be mapped correctly in JavaScript:
```javascript
const categoryMap = {
    'Anhörung': 'anhoerung-docs',
    'Bescheid': 'bescheid-docs',
    'Akte': 'akte-docs',
    'Rechtsprechung': 'rechtsprechung-docs',
    'Sonstiges': 'sonstiges-docs'
};
```

## API Endpoints

### Document Classification
- `POST /classify` - Upload and classify PDF (rate limit: 20/hour)
- `GET /documents` - Get all documents grouped by category
- `DELETE /documents/{filename}` - Delete document and file

### Research
- `POST /research` - Web research with Gemini + asyl.net (rate limit: 10/hour)

### Saved Sources
- `POST /sources` - Add source to saved collection
- `GET /sources` - Get all saved sources
- `GET /sources/download/{source_id}` - Download saved PDF
- `DELETE /sources/{source_id}` - Delete specific source
- `DELETE /sources` - Delete all sources

### Document Generation
- `POST /generate` - Generate legal draft with Claude (rate limit: 10/hour)

### System
- `GET /` - Main HTML interface
- `GET /health` - Health check
- `GET /api/sse` - Server-Sent Events for real-time updates

## Known Issues & Limitations

1. **PDF Detection:**
   - Only first 10 sources are processed for PDF detection
   - Some sites (asyl.net, ethz.ch) may have PDFs not detected by standard selectors
   - Detailed logging enabled for debugging

2. **asyl.net Search:**
   - Uses Playwright for scraping (slower than API)
   - Keyword-based search via cached suggestions
   - No fallback link if search returns 0 results (by design)

3. **Source Display:**
   - Gemini summary avoids duplicate source listings
   - Sources shown once in structured cards section

4. **Document Generation:**
   - Drafts not persisted (shown in modal only)
   - No ODT/PDF export yet
   - Limited to 4 most recent PDFs as context

5. **PDF Segmentation:**
   - Only processes Anhörung and Bescheid documents from Akte files
   - Rechtsprechung and other document types are not extracted
   - Segmentation failures are logged but do not block the original Akte classification

## Future Development

Planned features (see `plan.md` for full roadmap):
- Text extraction & OCR for uploaded documents
- Anonymization for Anhörung/Bescheid documents
- Export drafts to ODT/PDF format
- Multi-user authentication
- Svelte frontend (currently embedded HTML)
- Full-text search in processed documents
- Alembic migrations for database schema changes

## Reference Implementation

- **Classification**: Originally based on `/home/jay/GHPL/meta_ghpl_gpt5.py` (outside this repo), now migrated to Gemini with structured JSON output
- **Segmentation**: `kanzlei_gemini.py` provides the core PDF segmentation logic used when Akte files are uploaded
- **Standalone Tool**: `kanzlei-gemini.py` can be used independently for batch PDF segmentation outside the web interface
- **External Services**:
  - Anonymization service running on port 8002
  - OCR service running on port 8003
- **Directory**: The project directory is called `rechtmaschine`

## Module: kanzlei_gemini.py

This module provides Gemini-based PDF segmentation functionality:

**Key Functions:**
- `segment_pdf_with_gemini()` - Main entry point for segmentation
  - Uploads PDF to Gemini
  - Identifies Anhörung and Bescheid sections with strict pattern matching
  - Extracts sections to individual PDF files
  - Returns DocumentSections and list of (PageRange, file_path) tuples

- `identify_sections_with_gemini()` - Core AI analysis
  - Uses Gemini 2.5 Flash with German-language prompt
  - Requires strict BAMF document formatting (letterhead, signatures, etc.)
  - Returns confidence scores and page ranges

**Configuration:**
```python
@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash-preview-09-2025"
    temperature: float = 0.0
```

**Usage in app.py:**
```python
from kanzlei_gemini import segment_pdf_with_gemini, GeminiConfig

sections, extracted_pairs = segment_pdf_with_gemini(
    pdf_path=str(stored_path),
    output_dir=segment_dir,
    client=gemini_client,
    config=GeminiConfig(),
    verbose=False
)
```

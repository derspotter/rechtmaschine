# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rechtmaschine is an AI-powered legal document classification, research, and generation tool for German asylum lawyers. The project is deployed on a self-hosted server behind a Caddy reverse proxy at `rechtmaschine.de`.

**Current Status:** Fully functional with document classification, web research with Google Search grounding, legal database integration (asyl.net), saved sources management, and document generation via Claude.

## Architecture

### Technology Stack
- **Backend:** FastAPI (Python 3.11) running in Docker
- **Database:** PostgreSQL (latest alpine) for persistent storage
- **AI Models:**
  - OpenAI GPT-5-mini (document classification via Responses API)
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
4. File uploaded to OpenAI using `client.files.create()`
5. Classification performed via OpenAI Responses API (`client.responses.parse()`)
   - Model: `gpt-5-mini`
   - Uses structured output with Pydantic models
   - Service tier: `flex` for cost optimization
6. Result saved to PostgreSQL `documents` table
7. Frontend displays document in appropriate category box

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

**OpenAI Responses API Pattern:**
```python
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
response = client.with_options(timeout=900.0).responses.parse(**request_params)
parsed_result = response.output_parsed
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
- Rechtsprechung (case law)
- Sonstiges (other)

**Environment Variables:**
- `DATABASE_URL` - PostgreSQL connection string
- `POSTGRES_PASSWORD` - Database password
- `OPENAI_API_KEY` - Required for classification
- `ANTHROPIC_API_KEY` - Required for document generation (Claude)
- `GOOGLE_API_KEY` - Required for web search (Gemini)

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
/var/opt/docker/rechtmaschine/app/
├── app.py              # Main FastAPI application (all backend + frontend HTML)
├── database.py         # SQLAlchemy engine, session factory, Base
├── models.py           # ORM models (Document, ResearchSource, ProcessedDocument)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── docker-compose.yml  # Docker Compose configuration (app + postgres)
├── .env               # API keys and database config (not in git)
├── .env.example       # Example environment variables
├── data/              # Static data (asyl.net keyword suggestions)
└── CLAUDE.md          # This file
```

**Runtime directories (inside container):**
- `/app/uploads/` - Uploaded classified PDFs
- `/app/downloaded_sources/` - Downloaded research source PDFs

## Important Constraints

### Docker Network
The app connects to the external `caddy` network (not `caddy_default`). The Caddy reverse proxy is defined elsewhere on the server.

### Dependencies
- `openai>=1.56.0` - Required for Responses API support
- `httpx>=0.28.1` - HTTP client for web requests
- `pikepdf==9.4.2` - For PDF text extraction (future use)
- `google-genai==1.41.0` - Gemini + Google Search grounding
- `anthropic>=0.36.0` - Claude client for document generation
- `playwright==1.49.0` - Browser automation for web scraping and PDF detection
- `sqlalchemy` - ORM for PostgreSQL
- `psycopg2-binary` - PostgreSQL adapter
- `slowapi>=0.1.9` - Rate limiting
- `markdown>=3.6` - Markdown to HTML conversion

### German Language Support
All UI text and prompts are in German. Category names use German umlauts which must be mapped correctly in JavaScript:
```javascript
const categoryMap = {
    'Anhörung': 'anhoerung-docs',
    'Bescheid': 'bescheid-docs',
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

The classification approach is based on `/home/jay/GHPL/meta_ghpl_gpt5.py` (outside this repo). When modifying classification logic, maintain alignment with that reference pattern.

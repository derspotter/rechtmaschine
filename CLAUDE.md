# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rechtmaschine is an AI-powered legal document classification, research, and generation tool for German asylum lawyers. The project is deployed on a self-hosted server behind a Caddy reverse proxy at `rechtmaschine.de`.

**Current Status:** Fully functional with Gemini-powered document classification, automatic PDF segmentation for Akte files, web research with Google Search grounding, legal database integration (asyl.net), saved sources management, structured document generation via Claude Sonnet 4.5 (Files API), and j-lawyer template integration.

## Real-Time Updates Architecture

**Status:** Fully operational. The UI updates instantly via Server-Sent Events (SSE) with zero polling overhead.

### Current Implementation (October 2025)

**Backend (events.py + app.py):**
- PostgreSQL LISTEN/NOTIFY bridge with dual channels (`documents_updates`, `sources_updates`)
- Unified `BroadcastHub` receives notifications from both channels
- Single SSE endpoint `/documents/stream` broadcasts all entity types (documents and sources)
- All operations trigger broadcasts: classify, delete, reset, anonymize, source changes
- Caddy configured with `flush_interval -1` to prevent SSE buffering

**Frontend (app.js):**
- Single `EventSource` connection to `/documents/stream`
- Unified event handler processes both `documents_snapshot` and `sources_snapshot` events
- Digest comparison prevents unnecessary redraws
- Zero polling - all updates delivered via SSE
- Instant UI updates (<100ms latency)

**Verified Working:**
- Upload PDF → documents appear instantly via SSE
- Delete document → UI updates immediately
- Add/delete sources → instant reflection in UI
- Reset application → both documents and sources clear instantly
- Anonymize → badge appears without manual refresh

### Key Design Decisions

**Why PostgreSQL NOTIFY instead of Redis/RabbitMQ?**
- Already using PostgreSQL for data storage
- Built-in LISTEN/NOTIFY is lightweight and reliable
- No additional infrastructure needed
- Keeps worker instances in sync automatically

**Why single SSE stream for multiple entity types?**
- Simpler frontend code (one EventSource connection)
- Easier to add new entity types in future
- Reduces connection overhead
- Event type field allows frontend to route events appropriately

**Why disable polling entirely?**
- SSE proved 100% reliable in testing
- Polling was creating unnecessary database load
- Simpler codebase without dual update mechanisms
- Trust the established connection over periodic checks

## Architecture

### Technology Stack
- **Backend:** FastAPI (Python 3.11) running in Docker
- **Database:** PostgreSQL (latest alpine) for persistent storage
- **AI Models:**
  - Google Gemini 2.5 Flash (document classification with structured output)
  - Google Gemini 2.5 Flash (automatic PDF segmentation for Akte files)
  - Google Gemini 2.5 Flash (web research with Google Search grounding)
  - Anthropic Claude Sonnet 4.5 (document generation via Files API)
- **Document Processing Services:**
  - OCR Service (`desktop:8004` via Tailscale) - PaddleOCR-based text extraction
  - Anonymization Service (`desktop:8004` via Tailscale) - NER-based plaintiff identification
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

3. **processed_documents** - Text extraction & anonymization results
   - `id` (UUID, PK)
   - `document_id` (UUID, FK → documents.id, CASCADE)
   - `extracted_text` (TEXT) - Full text extracted via OCR or native PDF parsing
   - `is_anonymized` (BOOLEAN) - Whether document has been anonymized
   - `ocr_applied` (BOOLEAN) - Whether OCR was used for text extraction
   - `anonymization_metadata` (JSONB) - Plaintiff names, confidence scores, anonymized text
   - `processing_status` (VARCHAR) - completed, failed, etc.
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
1. User selects relevante Dokumente (Anhörung, Bescheid, Rechtsprechung, gespeicherte Quellen), markiert einen Hauptbescheid (Anlage K2), wählt den Dokumenttyp (Klagebegründung/Schriftsatz) und klickt "Entwurf generieren".
2. Frontend sendet ein strukturiertes Payload an `POST /generate` mit `document_type`, `user_prompt` und expliziten Dateilisten.
3. Backend lädt die angeforderten PDFs von der Festplatte, lädt sie über die Anthropic Files API hoch (`client.beta.files.upload`) und erzeugt eine Prompt-Zusammenfassung inklusive Quellenübersicht.
4. Claude Sonnet 4.5 wird via `client.beta.messages.create` mit den referenzierten `file_id`s aufgerufen (Betas: `files-api-2025-04-14`).
5. Antwort enthält Fließtext und Zitier-Metadaten (`citations_found`, `missing_citations`, `warnings`, `word_count`). Der Entwurf erscheint im Modal mit Hinweisboxen und j-lawyer Formular.
6. `GET /jlawyer/templates` liefert verfügbare ODT-Templates aus dem Standardordner, das Modal bietet Dropdown + Freitext.
7. `POST /send-to-jlawyer` überträgt den Text in die gewählte ODT-Vorlage (Placeholder `JLAWYER_PLACEHOLDER_KEY`, default `HAUPTTEXT`) und benennt die resultierende Datei pro Eingabe (`file_name`).

### OCR (Text Extraction) Flow
1. User clicks "OCR" button on classified document (Anhörung or Bescheid)
2. Frontend sends `POST /documents/{document_id}/ocr` request
3. Backend checks for cached OCR results in `processed_documents` table
4. If not cached:
   - Calls OCR service via Tailscale (`http://desktop:8004/extract-text`)
   - OCR service uses PaddleOCR for text extraction
   - Handles both native text PDFs and scanned images
   - Returns extracted text with metadata
5. Results stored in `processed_documents` table with `ocr_applied=True`
6. Frontend displays extracted text in modal
7. Text available for subsequent anonymization

**OCR Service:**
- External microservice running on `desktop:8004` (accessed via Tailscale)
- Uses PaddleOCR (open-source multilingual OCR toolkit)
- Handles multi-page PDFs
- Supports both machine-readable and scanned documents
- Optimized for German language documents
- Endpoint: `POST http://desktop:8004/extract-text`

### Anonymization Flow
1. User clicks "Anonymisieren" button on classified document (Anhörung or Bescheid)
2. Frontend sends `POST /documents/{document_id}/anonymize` request
3. Backend checks for cached anonymization in `processed_documents` table
4. If not cached:
   - Extracts text via OCR service (if not already extracted)
   - Sends first 10,000 characters to anonymization service via Tailscale
   - Anonymization service (`http://desktop:8004/anonymize-document`)
     - Uses NER (Named Entity Recognition) to identify plaintiff names
     - Detects personal information (names, addresses, dates of birth, etc.)
     - Replaces names with "Kläger/Klägerin" and family members with "Kind 1", "Kind 2", etc.
     - Returns anonymized text with confidence scores and extracted names
5. Backend stitches anonymized section with remaining text:
   - First 10k chars: anonymized
   - Remainder: original text appended
6. Results stored in `processed_documents` table:
   - `is_anonymized=True`
   - `anonymization_metadata` contains: plaintiff names, confidence score, full anonymized text
7. Document's `to_dict()` method returns `anonymized=True` flag
8. Frontend displays:
   - Anonymized text in modal
   - Green "Anonymisiert ✓" badge on document card (via SSE auto-update)
   - Extracted plaintiff names for verification

**Anonymization Service:**
- External microservice running on `desktop:8004` (accessed via Tailscale)
- Uses NER models trained on German legal documents
- Focuses on plaintiff identification in asylum law contexts
- Returns confidence scores for quality assessment
- Handles first 10k characters to prevent timeout (legal docs frontload personal info)
- Endpoint: `POST http://desktop:8004/anonymize-document`

**Text Stitching:**
- Anonymization service processes only first 10,000 characters (plaintiff names typically in document header)
- Backend combines anonymized section with untouched remainder
- Prevents timeout issues with large PDFs
- Leverages legal document structure (personal info at beginning)

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

**Document Generation with Claude Sonnet 4.5:**
```python
uploaded_files = []
for doc in selected_documents:
    uploaded = client.beta.files.upload(
        file=(doc.filename, open(doc.file_path, "rb"), "application/pdf"),
        betas=["files-api-2025-04-14"],
    )
    uploaded_files.append(uploaded.id)

content = [{"type": "text", "text": user_prompt}]
for file_id in uploaded_files:
    content.append({
        "type": "document",
        "source": {"type": "file", "file_id": file_id},
    })

response = client.beta.messages.create(
    model="claude-sonnet-4-5",
    system=system_prompt,
    max_tokens=4000,
    messages=[{"role": "user", "content": content}],
    betas=["files-api-2025-04-14"],
    temperature=0.2,
)

draft_text = "\n\n".join(
    block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
    for block in response.content
    if (isinstance(block, dict) and block.get("type") == "text") or getattr(block, "type", None) == "text"
).strip()
```

**j-lawyer Upload Pattern:**
```python
payload = [
    {
        "placeHolderKey": JLAWYER_PLACEHOLDER_KEY,
        "placeHolderValue": generated_text,
    }
]

async with httpx.AsyncClient(timeout=30) as client:
    response = await client.put(
        f"{JLAWYER_BASE_URL}/v6/templates/documents/{folder}/{template}/{case_id}/{file_name}",
        auth=(JLAWYER_USERNAME, JLAWYER_PASSWORD),
        json=payload,
    )
response.raise_for_status()
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
- `JLAWYER_BASE_URL`, `JLAWYER_USERNAME`, `JLAWYER_PASSWORD`, `JLAWYER_TEMPLATE_FOLDER`, `JLAWYER_PLACEHOLDER_KEY` - Configuration for j-lawyer template export

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
- `GET /documents/stream` - SSE stream for real-time document and source updates
- `DELETE /documents/{filename}` - Delete document and file

### Document Processing
- `POST /documents/{document_id}/ocr` - Extract text from PDF via OCR service (rate limit: 5/hour)
- `POST /documents/{document_id}/anonymize` - Anonymize document text (rate limit: 5/hour)
- `POST /anonymize-file` - Anonymize uploaded file without storing (rate limit: 5/hour)

### Research
- `POST /research` - Web research with Gemini + asyl.net (rate limit: 10/hour)

### Saved Sources
- `POST /sources` - Add source to saved collection
- `GET /sources` - Get all saved sources
- `GET /sources/download/{source_id}` - Download saved PDF
- `DELETE /sources/{source_id}` - Delete specific source
- `DELETE /sources` - Delete all sources

### Document Generation
- `POST /generate` - Generate legal draft with Claude Sonnet 4.5 + Files API (rate limit: 10/hour)
- `GET /jlawyer/templates` - List available ODT templates from the configured j-lawyer folder
- `POST /send-to-jlawyer` - Populate a j-lawyer template with the generated text

### System
- `GET /` - Main HTML interface
- `GET /health` - Health check
- `DELETE /reset` - Clear all documents, sources, and processed data (rate limit: 10/hour)

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
   - No direct ODT/PDF export yet (j-lawyer is the primary export path)
   - Relies on provided metadata for date/Az detection; inaccurate metadata can yield warning flags

5. **PDF Segmentation:**
   - Only processes Anhörung and Bescheid documents from Akte files
   - Rechtsprechung and other document types are not extracted
   - Segmentation failures are logged but do not block the original Akte classification

## Future Development

Planned features (see `plan.md` for full roadmap):
- Export drafts to ODT/PDF format (currently exports to j-lawyer only)
- Multi-user authentication and role-based access
- Svelte frontend (currently embedded HTML/CSS/JS)
- Full-text search in processed documents
- Alembic migrations for database schema changes
- Batch processing for multiple documents
- Advanced anonymization rules and custom entity recognition

## Reference Implementation

- **Classification**: Originally based on `/home/jay/GHPL/meta_ghpl_gpt5.py` (outside this repo), now migrated to Gemini with structured JSON output
- **Segmentation**: `kanzlei_gemini.py` provides the core PDF segmentation logic used when Akte files are uploaded
- **Standalone Tool**: `kanzlei-gemini.py` can be used independently for batch PDF segmentation outside the web interface
- **External Services**:
  - OCR and Anonymization services running on `desktop:8004` (via Tailscale)
  - Both services share the same endpoint host, different paths
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

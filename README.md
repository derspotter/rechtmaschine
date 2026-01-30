# Rechtmaschine

AI-powered legal document classification, research, and generation tool for German asylum law.

## Overview

Rechtmaschine assists German asylum lawyers by automatically classifying legal documents, conducting intelligent web research, and generating structured legal drafts. The system leverages multiple AI models and features automatic document segmentation for complex case files (Akten), OCR text extraction, automated anonymization, and direct integration with j-lawyer practice management software.

**Live at:** https://rechtmaschine.de

## Current Status

Fully functional production system with:
- Gemini-powered document classification with structured output validation
- Automatic PDF segmentation for Akte files (extracts Anhörung and Bescheid documents)
- Web research with Google Search grounding + asyl.net legal database integration
- Saved sources management with automatic PDF downloads
- Document generation via Claude Sonnet 4.5 with Files API
- OCR text extraction via external microservice
- Automated anonymization with NER-based plaintiff identification
- j-lawyer template integration for direct export
- Real-time UI updates via Server-Sent Events (SSE)

## Tech Stack

### Core Infrastructure
- **Frontend:** Embedded HTML/CSS/JS with real-time SSE updates (Svelte migration planned)
- **Backend:** FastAPI (Python 3.11) with modular endpoint architecture running in Docker
- **Database:** PostgreSQL (latest alpine) with LISTEN/NOTIFY for real-time events
- **Reverse Proxy:** Caddy (HTTPS with automatic certificates)
- **Deployment:** Docker Compose on self-hosted server

### AI Models
- **Google Gemini 2.5 Flash**: Document classification with structured JSON output
- **Google Gemini 2.5 Flash**: Automatic PDF segmentation for Akte files
- **Google Gemini 2.5 Flash**: Web research with Google Search grounding
- **Anthropic Claude Sonnet 4.5**: Structured document generation via Files API

### External Services
- **OCR Service** (`desktop:8004` via Tailscale): PaddleOCR-based text extraction for scanned documents
- **Anonymization Service** (`desktop:8004` via Tailscale): NER-based plaintiff identification and text anonymization
- **Playwright**: Web scraping for asyl.net search and automatic PDF detection

## Features

### Document Management
- **Intelligent Classification**: Automatically categorizes uploaded PDFs into:
  - Anhörung (hearing protocols)
  - Bescheid (administrative decisions)
  - Akte (complete BAMF case files)
  - Rechtsprechung (case law)
  - Sonstiges (other documents)
- **Automatic Segmentation**: Akte files are automatically split into individual Anhörung and Bescheid documents
- **OCR Text Extraction**: Extract text from scanned PDFs using PaddleOCR
- **Automated Anonymization**: NER-based identification and replacement of plaintiff names with "Kläger/Klägerin"
- **Real-time Updates**: Instant UI updates via SSE (no polling required)

### Research & Sources
- **Dual Web Research**:
  - Gemini with Google Search grounding (official sources, courts, government)
  - asyl.net database scraping with keyword suggestions
- **Automatic PDF Detection**: First 10 sources analyzed for PDF availability
- **Saved Sources**: Organize research with automatic PDF downloads and status tracking
- **Download Management**: Background PDF downloads with real-time status updates

### Document Generation
- **Structured Drafts**: Generate Klagebegründung or Schriftsatz using Claude Sonnet 4.5
- **Multi-document Context**: Reference multiple uploaded PDFs (Anhörung, Bescheid, Rechtsprechung, saved sources)
- **Citation Analysis**: Automatic detection of citations with quality warnings
- **j-lawyer Integration**:
  - Direct export to ODT templates
  - List available templates from configured folder
  - Populate placeholders with generated text
  - Custom file naming support

### Real-time Architecture
- **PostgreSQL LISTEN/NOTIFY**: Dual channels for document and source updates
- **Unified SSE Stream**: Single connection for all entity types
- **Zero Polling**: All updates delivered via push notifications
- **Instant UI Updates**: Sub-100ms latency for all operations

## Database Schema

**PostgreSQL Tables:**
1. **documents** - Classified legal documents with category, confidence, and file metadata
2. **research_sources** - Saved legal research sources with PDF download tracking
3. **processed_documents** - Text extraction and anonymization results linked to documents

## API Endpoints

### Document Operations
- `POST /classify` - Upload and classify PDF (auto-segments Akte files)
- `GET /documents` - Retrieve all documents grouped by category
- `GET /documents/stream` - SSE stream for real-time updates
- `DELETE /documents/{filename}` - Delete document and associated files

### Document Processing
- `POST /documents/{document_id}/ocr` - Extract text via OCR service
- `POST /documents/{document_id}/anonymize` - Anonymize document with NER
- `POST /anonymize-file` - Anonymize uploaded file without storing

### Research & Sources
- `POST /research` - Web research (Gemini + asyl.net)
- `POST /sources` - Save source to collection
- `GET /sources` - List all saved sources
- `GET /sources/download/{source_id}` - Download saved PDF
- `DELETE /sources/{source_id}` - Delete specific source

### Document Generation
- `POST /generate` - Generate legal draft with Claude Files API
- `GET /jlawyer/templates` - List available ODT templates
- `POST /send-to-jlawyer` - Populate j-lawyer template with generated text

### System
- `GET /` - Main HTML interface
- `GET /health` - Health check
- `DELETE /reset` - Clear all data (documents, sources, processed data)

## Development

### Quick Start
```bash
cd /var/opt/docker/rechtmaschine/app
docker compose up -d
docker compose logs -f app
```

### Environment Variables
Create `.env` file with:
```
DATABASE_URL=postgresql://rechtmaschine:password@postgres:5432/rechtmaschine_db
GOOGLE_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_claude_api_key
JLAWYER_BASE_URL=http://jlawyer-server:8080
JLAWYER_USERNAME=username
JLAWYER_PASSWORD=password
```

Optional external service manager (Wake-on-LAN + remote start):
```
SERVICE_MANAGER_HEALTH_URL=http://desktop:8004/health
SERVICE_MANAGER_SSH_HOST=desktop
SERVICE_MANAGER_SSH_USER=jayjag
SERVICE_MANAGER_START_CMD=tmux new -d -s service_manager 'cd ~/rechtmaschine && source .venv/bin/activate && python service_manager.py'
SERVICE_MANAGER_START_TIMEOUT_SEC=180
SERVICE_MANAGER_POLL_INTERVAL_SEC=5
SERVICE_MANAGER_SSH_TIMEOUT_SEC=20

WOL_SSH_HOST=osmc
WOL_SSH_USER=osmc
WOL_COMMAND=wakeonlan 9c:6b:00:8d:51:d2
WOL_MAC=9c:6b:00:8d:51:d2
```

### Database Access
```bash
docker exec -it rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db
```

### Restart After Changes
```bash
docker compose restart app
```

Note: Hot reload is enabled for Docker volumes; if changes do not appear, restart the app container.

## Architecture Details

See [CLAUDE.md](CLAUDE.md) for comprehensive technical documentation including:
- Detailed implementation patterns
- SSE architecture and design decisions
- External service integration
- Module documentation
- Known issues and limitations
- Future development roadmap

## Security & Privacy

- All processing happens on self-hosted infrastructure
- OCR and anonymization run on isolated Tailscale-connected services
- No client data sent to external services except AI API providers (Google, Anthropic)
- Automatic anonymization for plaintiff identification before draft generation
- PostgreSQL for secure persistent storage

---

Developed and deployed on self-hosted infrastructure for maximum security and control.

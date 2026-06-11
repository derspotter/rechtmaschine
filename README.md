# Rechtmaschine

AI-powered legal document classification, research, and generation tool for German asylum law.

## Overview

Rechtmaschine assists German asylum lawyers by classifying legal documents, segmenting complex case files (Akten), extracting and anonymizing text, conducting AI web research, and generating structured legal drafts with verified citations. It integrates directly with j-lawyer practice management software and runs entirely on self-hosted infrastructure.

**Live at:** https://rechtmaschine.de

## Features

### Document Management
- **Classification** (Gemini, structured JSON output): Anhörung, Bescheid, Akte, Rechtsprechung, Sonstiges, plus client-document types
- **Akte segmentation**: BAMF case files are automatically split into individual child documents, which are auto-classified and processed
- **OCR**: PaddleOCR-based text extraction for scanned PDFs (GPU worker)
- **Anonymization**: NER + local Qwen identifier extraction replaces plaintiff names before any text leaves the infrastructure
- **Translations**: foreign-language documents translated to German via local Qwen
- **Real-time UI**: instant updates via Server-Sent Events (PostgreSQL LISTEN/NOTIFY, zero polling)

### Research & Sources
- Web research via Gemini (Google Search grounding) and Grok (web_search tool)
- asyl.net legal database integration with AI keyword extraction
- Automatic extraction of relevant legal provisions (§§ AsylG, AufenthG, GG) from local law texts
- Saved sources with background PDF download and status tracking
- Persistent research runs per case

### Generation & Drafts
- Klagebegründung / Schriftsatz drafts via Claude Opus 4.7 (Files API), GPT-5.5, or Gemini
- Deterministic citation verification with optional local-Qwen semantic checks
- Persisted drafts per case
- j-lawyer export: populate ODT templates and file documents directly into cases

### Multi-User & Cases
- JWT login plus long-lived API tokens for scripted access
- All documents, sources, drafts, and research scoped per user and per case
- Durable case memory: case briefs, case strategies, and reviewable memory update proposals
- Long-running AI work (generation, query, research) executed asynchronously by a dedicated job worker

## Tech Stack

- **Backend:** FastAPI (Python 3.11), modular endpoint routers, Docker Compose (app + job-worker + PostgreSQL)
- **Database:** PostgreSQL with LISTEN/NOTIFY for real-time events; idempotent SQL migrations applied at startup
- **Frontend:** Embedded HTML/CSS/vanilla JS with SSE updates
- **Reverse Proxy:** Caddy (automatic HTTPS)

### AI Models
- **Google Gemini** (3.1 Pro / 3.5 Flash): classification, Akte segmentation, research grounding, document query, generation option
- **Anthropic Claude Opus 4.7**: default draft generation via Files API
- **OpenAI GPT-5.5**: generation option and citation review (optional Azure routing)
- **xAI Grok 4.3**: web research
- **Local Qwen3.6 27B** (llama-server/Ollama on own GPU hardware): anonymization, identifier extraction, vision PDF segmentation, citation verification, translations

### Infrastructure (three machines, one codebase)
- **server** — main app stack (this repo's Docker Compose)
- **desktop** — Qwen GPU worker: anonymization, extraction, segmentation (`service_manager.py`, role `anonymization`)
- **debian** — OCR/RAG GPU worker: PaddleOCR, embeddings, reranking, RAG store (`service_manager.py`, role `ocr`)

Workers are reached via Tailscale; the app can wake them over SSH on demand. `service_manager.py` loads/unloads models to fit available VRAM.

## Development

### Quick Start
```bash
cd /var/opt/docker/rechtmaschine
docker compose up -d
docker compose logs -f app
```

Configuration lives in `app/.env` (see `app/.env.example`). Required keys include `DATABASE_URL`, `SECRET_KEY` (app refuses to start without it), `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `JLAWYER_*`, and the `OCR_*` / `ANONYMIZATION_*` worker settings.

### Restart After Changes
Code is volume-mounted; restart both Python consumers:
```bash
docker compose restart app
docker compose restart job-worker
```

### Database Access
```bash
docker exec -it rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db
```

### API Access from the Terminal
The full HTTP API (login, cases, documents, generation, research) can be driven with curl — see `.claude/skills/rechtmaschine-api-cli/` for recipes. Interactive API docs are available at `/docs` when `ENABLE_API_DOCS=true`.

## Documentation

- [CLAUDE.md](CLAUDE.md) — architecture reference (containers, migrations, data model, AI responsibilities)
- [AGENTS.md](AGENTS.md) — conventions for agentic coding assistants
- `docs/` — design and implementation plans (RAG ingestion, multi-user, anonymization, etc.)

## Security & Privacy

- Self-hosted deployment; OCR and anonymization run on own GPU hardware over Tailscale
- Documents are anonymized locally before text is used in cloud AI calls
- Client data is sent only to the configured AI API providers, never to third parties
- Multi-user authentication with per-user data isolation

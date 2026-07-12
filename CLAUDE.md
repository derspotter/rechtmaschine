# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rechtmaschine is an AI-powered legal document classification, research, and generation tool for German asylum lawyers. Production deployment at https://rechtmaschine.de behind a Caddy reverse proxy. Multi-user with JWT auth; all data is scoped per user (`owner_id`) and per case (`case_id`).

All UI text, prompts, and document categories are German: Anhörung, Bescheid, Akte, Rechtsprechung, Sonstiges (plus client-document types).

## Three-Machine Architecture

One pullable codebase runs on three machines; roles are assigned via env, not code:

- **server** — runs the main app stack (this Docker Compose: postgres + app + job-worker).
- **desktop** — Qwen3.6 GPU worker: anonymization, identifier extraction, vision segmentation (`service_manager.py` with default role `all`, port 8004). Also owns Nextcloud/j-lawyer datasource export for RAG.
- **debian** — OCR/RAG GPU worker: PaddleOCR (`service_manager.py` with `SERVICE_MANAGER_ROLE=ocr`, port 8004, systemd unit `rechtmaschine-ocr`), plus the Docker RAG stack (`rag/docker-compose.debian.yml`: pgvector store, TEI embed/rerank, RAG API on port 8090).

Workers are reached via Tailscale. The app can wake a sleeping worker over SSH (`*_SERVICE_MANAGER_SSH_*` env vars; SSH keys are bind-mounted into the containers). **Do not infer which machine you are on from this file — check `hostname` first** before making machine-specific changes.

`service_manager.py` (repo root) is the GPU worker supervisor: it loads/unloads OCR and llama-server/Ollama Qwen models to fit VRAM, queues requests per service, and is API-key protected.

## Commands

```bash
# Start / rebuild / stop (use `docker compose`, never `docker-compose`)
docker compose up -d
docker compose build
docker compose down

# Restart after Python changes (code is volume-mounted; restart both consumers)
docker compose restart app
docker compose restart job-worker

# Logs
docker compose logs -f app
docker compose logs -f job-worker

# PostgreSQL shell
docker exec -it rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db

# Caddy config changes (in /var/opt/docker/caddy/Caddyfile):
# `caddy reload` is unreliable with bind-mounted config — always restart
docker restart caddy
```

Environment lives in `app/.env` (see `app/.env.example`). `SECRET_KEY` is mandatory — the app refuses to start without it. API docs (`/docs`) are disabled unless `ENABLE_API_DOCS=true`.

### Tests

`tests/` is a pytest suite (~300 tests, injected I/O, no live calls) plus some ad-hoc benchmark/shell scripts that need live GPU workers or API keys (check headers). Run on the host, not in Docker:

```bash
.venv/bin/python -m pytest tests/ -q               # full suite
.venv/bin/python -m pytest tests/ -q -m "not slow" # what the pre-push hook runs
```

A pre-push hook runs the suite minus `slow` marks — install via `cp scripts/git-hooks/pre-push .git/hooks/`. Gotcha (see `tests/conftest.py`): tests that stub `sys.modules` entries must pop them afterwards, or they poison later test files in the same run.

## Architecture

### Containers

Two containers are built from the same image:
- **app** (`rechtmaschine-app`) — FastAPI on port 8000 (exposed only to the `proxy-rechtmaschine` external Docker network; Caddy terminates TLS).
- **job-worker** (`rechtmaschine-job-worker`) — runs `app/job_worker.py`, which polls six job tables (`generation_jobs`, `query_jobs`, `research_jobs`, `ocr_jobs`, `anonymize_jobs`, `memory_reflection_jobs`) with claim/heartbeat/stale-requeue semantics and executes them via `_execute_*_request` functions imported from the endpoint modules. Long-running AI work goes through these job tables; the HTTP endpoints enqueue and clients poll/stream results.

### Backend layout (`app/`)

- `main.py` — FastAPI wiring, TrustedHost middleware, startup events, **and schema migrations** (see below).
- `shared.py` (~1850 lines) — shared Pydantic models, AI client factories (`get_gemini_client`, `get_anthropic_client`, `get_openai_client`, `get_xai_client`), directory constants, document-text storage helpers, SSE snapshot broadcasting.
- `models.py` — SQLAlchemy ORM models with `to_dict()` helpers.
- `events.py` — PostgreSQL LISTEN/NOTIFY → `BroadcastHub` → SSE.
- `auth.py` — JWT (OAuth2 password flow at `POST /token`) plus long-lived API tokens (`api_tokens` table, managed via `/api-tokens` endpoints).
- `endpoints/` — one router per concern: auth, cases, classification, documents, ocr, anonymization, translations, segmentation, research_sources (+ `research/` subpackage with gemini/grok/asylnet/specialized providers), generation, query, drafts, rechtsprechung_playbook, agent_memory, pattern_wiki, jurisprudence, doktrin, workflow, rag, root, system.
- `kanzlei_gemini.py` — Gemini-based Akte PDF segmentation (used on upload when a file classifies as Akte; segments are auto-classified as child documents).
- `legal_texts/` — local copies of AsylG/AufenthG/GG/AsylbLG (from bundestag/gesetze) with provision extraction; research auto-extracts relevant §§ via structured Gemini output.
- `citation_verifier.py` / `citation_qwen.py` — deterministic citation checks on generated drafts, with optional local-Qwen semantic verification for ambiguous findings.
- Case-memory extensions (see `docs/agent-memory-and-case-law-plan.md`): `endpoints/pattern_wiki.py` — anonymized cross-case argument/risk patterns, distilled via memory_reflection jobs, entries stay `pending` until lawyer review; `endpoints/jurisprudence.py` — freshness-gated per-fingerprint case-law packs injected into prompts (`JURIS_*` env vars); `doktrin_sync.py` — nightly mirror of wiki.aufentha.lt (runs via `docker exec rechtmaschine-app python /app/doktrin_sync.py`, `doktrin-sync.timer`), with read-only observability in `endpoints/doktrin.py`.
- `qwen_tagger.py` / `gemma_tagger.py` — interchangeable `tag_document(text, vocab)` backends that tag anonymized documents against the controlled vocabulary (`rag_vocabulary.json`) for RAG facet search.

### Database migrations

**No Alembic.** Migrations are an ordered list of idempotent SQL statement groups in the `MIGRATIONS` constant in `app/main.py`, tracked in the `schema_migrations` table and applied on startup (both app and job-worker call `apply_schema_migrations()`). To change the schema: update the ORM model in `models.py` **and** append a new named entry to `MIGRATIONS`. Use `IF NOT EXISTS` / `IF EXISTS` so re-runs are safe.

Note: extracted/anonymized/translated text is stored **on disk** (`app/ocr_text/`, `app/anonymized_text/`, `app/translated_text/`), with only paths and flags on the `documents` row (`extracted_text_path`, `is_anonymized`, `ocr_applied`, …). The old `processed_documents` table was dropped. Use `store_document_text` / `load_document_text` from `shared.py`.

Key tables beyond `documents`/`research_sources`: `users`, `api_tokens`, `cases` (per-user, with `users.active_case_id`), `generated_drafts`, `research_runs`, the six job tables, `document_segments`, `document_translations`, `rechtsprechung_entries`, and the case-memory family (`case_briefs`, `case_strategies`, `*_sources`, `case_memory_revisions`, `case_document_extractions`, `memory_update_proposals`).

### Real-time updates

Zero polling in the UI. DB writes trigger `pg_notify` on the `documents_updates`/`sources_updates` channels; a `PostgresListener` thread feeds the `BroadcastHub`; a single SSE endpoint `/documents/stream` pushes `documents_snapshot`/`sources_snapshot` events to the frontend. Auth: the client first fetches a one-time ticket via `POST /documents/stream-ticket`, then opens `EventSource('/documents/stream?ticket=...')` — no JWT in the URL. After any mutation, call the `broadcast_*_snapshot` helpers from `shared.py`. Caddy needs `flush_interval -1` for SSE.

### AI model responsibilities

- **Gemini** (`gemini-3.1-pro-preview`, `gemini-3.5-flash`) — Akte segmentation, research with Google Search grounding, generation option, document query. (Classification moved to local Qwen; Gemini available via `CLASSIFICATION_BACKEND=gemini`.)
- **Claude Opus 4.8** — default draft generation via the Anthropic Files API (`betas=["files-api-2025-04-14"]`).
- **GPT-5.6 Sol** (`gpt-5.6-sol`, reasoning effort `high` via `OPENAI_GPT5_REASONING_EFFORT`) — generation option, expert-pipeline critique/finalize, document query; ChatGPT-Search research and case profiling use `gpt-5.6-terra` (`OPENAI_RESEARCH_MODEL`, `RESEARCH_CASE_PROFILE_MODEL`), meta relevance judging `gpt-5.6-sol` (`META_RELEVANCE_MODEL`). Optionally routed through Azure OpenAI via `OPENAI_PROVIDER=azure`.
- **Grok 4.3** — research with web_search tool (model via `XAI_MODEL`; grok-4.5 pending EU availability).
- **Local Qwen3.6 27B** (via service managers on desktop, llama-server/Ollama) — anonymization and plaintiff-identifier extraction, vision PDF segmentation, **document classification** (`CLASSIFICATION_BACKEND`, default), semantic citation verification, translations. Feature flags / model names via `CITATION_QWEN_*`, `DOCUMENT_SEGMENTATION_*`, `OLLAMA_MODEL_*` env vars.

Anonymization pipeline: OCR text → NER service (Flair-based, code in `anon/`, mounted read-only into the containers and imported by `endpoints/anonymization.py`) → local Qwen identifier extraction → regex replacement → stitched text stored under `anonymized_text/`.

### Frontend

Embedded HTML in `app/templates/` plus `app/static/js/app.js` (main UI + auth wrapper) and `drafts.js`. No framework. German category names with umlauts must map to DOM ids (`'Anhörung' → 'anhoerung-docs'`, etc.).

## Terminal API access

For driving the app from a shell instead of the browser, use the local skill `.claude/skills/rechtmaschine-api-cli/SKILL.md` (login, token handling, listing cases/documents/sources, generation, research; curl recipes in its `references/curl-recipes.md`).

## Repo Miscellany

- `docs/` — design/plan documents (RAG ingestion, multi-user, anonymization refactor, etc.). Check here before designing a new subsystem; a plan probably exists.
- `ocr/`, `anon/`, `rag/` — code for the GPU worker machines, kept in this repo so all machines pull one codebase.
- `legacy/`, `examples/`, root-level `verify_*.py` / `benchmark_*.py` scripts — historical/one-off; not part of the running app.

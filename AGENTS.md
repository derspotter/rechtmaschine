# AGENTS.md

This file guides agentic coding assistants working in this repo.

## Rule Sources
- No `.cursor` rules were found in this repo.
- No `.cursorrules` or Copilot instructions were found.
- Follow the project documentation in `CLAUDE.md` for architecture details.

## Repo Layout (high level)
- `app/` FastAPI backend + embedded frontend assets
- `app/endpoints/` API routers (auth, cases, classification, documents, research, generation, query, agent_memory, rag, etc.)
- `app/job_worker.py` background worker for generation/query/research job tables (runs as its own container)
- `app/static/` JS/CSS assets for the UI
- `app/templates/` HTML templates
- `app/legal_texts/` local law text extraction and lookup
- `tests/` ad-hoc Python and shell scripts (no pytest suite)
- `ocr/`, `anon/`, `rag/` code for the GPU worker machines
- `docs/` design/implementation plans — check here before designing a new subsystem

## Key Entry Points
- `app/main.py` FastAPI app wiring, schema migrations (`MIGRATIONS` list — no Alembic), startup events
- `app/shared.py` shared constants, Pydantic models, AI client factories, helpers
- `app/models.py` SQLAlchemy ORM models and `to_dict()` helpers
- `app/auth.py` JWT auth (`POST /token`) + API token verification
- `app/events.py` Postgres LISTEN/NOTIFY + SSE helpers
- `app/static/js/app.js` main UI logic and auth wrapper

## Build / Run
Primary runtime is Docker Compose (note: use **docker compose**, not docker-compose):
- Start stack (postgres + app + job-worker): `docker compose up -d`
- Build/rebuild: `docker compose build`
- View logs: `docker compose logs -f app` (or `job-worker`)
- Restart after Python changes (both consumers): `docker compose restart app job-worker`
- Stop stack: `docker compose down`
- Shell into app container: `docker compose exec app /bin/bash`

Database access:
- PSQL shell: `docker exec -it rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db`

Local (non-Docker) install commands (if needed for tooling):
- App deps: `python -m venv .venv && source .venv/bin/activate && pip install -r app/requirements.txt`
- OCR deps: `python -m venv .venv && source .venv/bin/activate && pip install -r ocr/requirements.txt`
- Anon deps: `python -m venv .venv && source .venv/bin/activate && pip install -r anon/requirements.txt`
- RAG deps: `python -m venv .venv && source .venv/bin/activate && pip install -r rag/requirements.txt`

Environment files:
- The app expects `.env` in `app/.env` with database and API keys (see `app/.env.example`).
- Example keys: `DATABASE_URL`, `SECRET_KEY` (mandatory), `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `JLAWYER_*`, `OCR_*`, `ANONYMIZATION_*`.

Service manager (OCR/anonymization supervisor):
- Install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run locally: `python service_manager.py`

Three-machine runtime note:
- `desktop` is the Qwen3.6 worker for anonymization, anonymized metadata extraction, and segmentation. It also owns datasource collection/export from Nextcloud and j-lawyer because those sources are available there.
- `debian` is the OCR/RAG worker for OCR, embedding, reranking, ingestion, and the RAG store/API. It imports desktop exports instead of discovering Nextcloud or j-lawyer directly.
- `server` runs the main Rechtmaschine app stack.
- Do not infer the current machine from this shared file; check `hostname` or local-only memory before making machine-specific changes.

Claude Code (CLI) usage:
- Non-interactive: `claude -p "Your prompt here"`
- Interactive session: `claude`

## Tests
There is no pytest/CI test suite configured. `tests/` holds ad-hoc Python scripts and shell scripts, run individually:
- Python scripts: `python tests/test_citations.py`, `python tests/test_citation_verifier.py`, etc.
- Shell scripts: `bash tests/test_sequential.sh` (OCR→Anonymize), `bash tests/test_concurrent.sh`, `bash tests/test_manager.sh`, `bash anon/test_anonymization.sh`

Test notes:
- Check each script header first: many require live GPU workers (OCR/anonymization via Tailscale), API keys, or local sample-file paths.
- These scripts are manual and not suitable for CI.

## Lint / Format
No lint/format tooling is configured in this repo.
- Do not add new tooling unless explicitly requested.
- Match existing formatting and style in the files you edit.
- Avoid reformatting unrelated code while making changes.

## Python Code Style
General:
- Use 4-space indentation and blank lines between top-level defs.
- Prefer `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Use type hints for public functions and Pydantic models.
- Prefer `Pathlib` (`Path`) for filesystem paths.
- Keep German text as UTF-8; use `ensure_ascii=False` when serializing JSON for UI.
- Use `datetime.utcnow()` for timestamps and `uuid.uuid4()` for IDs.
- Use f-strings for log messages and error context.
- Add docstrings to public modules, classes, and endpoints.
- Avoid one-letter variable names outside tight loops.

Imports:
- Group imports: standard library → third-party → local modules.
- Avoid duplicate imports and keep import lists sorted when touching a file.

FastAPI / endpoints:
- Use `APIRouter` and dependency injection via `Depends`.
- Keep request/response models in `app/shared.py` (Pydantic `BaseModel`).
- Raise `HTTPException` for API errors; include clear `detail` messages.
- Use rate limits via `limiter.limit(...)` where appropriate.
- Keep response payloads JSON-serializable; use `JSONResponse` for custom headers.
- Keep SSE payloads shaped like existing `documents_snapshot` / `sources_snapshot` events.
- Favor `StreamingResponse` for SSE endpoints and include cache-control headers.

Database / ORM:
- Use SQLAlchemy ORM models from `app/models.py`.
- Use UUID primary keys and `datetime.utcnow()` for timestamps.
- Maintain `to_dict()` helpers for response payloads.
- Access DB sessions via `Depends(get_db)`.

Error handling:
- Use `try/except` around filesystem and external service calls.
- Log non-fatal issues with `print` (existing convention) instead of raising.
- Ensure files/paths exist before deleting or reading.
- Reserve `HTTPException` for user-facing errors in API handlers.

JSON/serialization:
- Use `JSONResponse` for custom headers or cache-control.
- Use `json.dumps(..., ensure_ascii=False)` for German text.

Security/Secrets:
- Never commit `.env`, API keys, or local file paths.
- Keep secrets in environment variables or `app/.env`.

## Frontend (JS/CSS)
JavaScript:
- Use `const`/`let`, arrow functions, and `async/await` patterns.
- Indent with 4 spaces and keep semicolons.
- Prefer single quotes for strings (as in `app/static/js/app.js`).
- Wrap network calls in `try/catch` and surface readable UI errors.
- Respect the auth token wrapper around `fetch` in `app/static/js/app.js`.
- Use `debugLog`/`debugError` helpers for consistent console output.
- Avoid introducing new frameworks; stay with vanilla DOM manipulation.

HTML/CSS:
- Keep UI text in German.
- Match existing class naming and layout conventions in templates.
- Maintain the current layout before refactoring styles.

## Safety / Ops Notes
- The app expects `.env` in `app/.env` with database and API keys.
- Hot reload is enabled in Docker and should pick up code changes; restart the app container if updates do not apply.
- External services (OCR/anonymization) run via Tailscale; avoid changing URLs unless requested.
- File storage uses `/app/uploads` and `/app/downloaded_sources` volumes.

## When in Doubt
- Keep changes minimal and consistent with existing patterns.
- Ask before introducing new dependencies or structural refactors.
- Prefer readability over cleverness.

# Endpoint-Centric Refactor Plan (Revised)

## Key Findings from `app/app.py`
- ~4,200 lines mixing app startup, schema migrations, directory helpers, enums/models, third-party client factories, and every FastAPI endpoint.
- New database text-cache workflow (`store_document_text`, `load_document_text`, `delete_document_text`) reused in classification, OCR, and generation flows.
- SSE broadcasting helpers (`_emit_event`, `_broadcast_documents_snapshot`, `_broadcast_sources_snapshot`) support multiple endpoints.
- Research + sources management share extensive helpers (`search_asyl_net`, `download_and_update_source`, suggestion cache).
- Generation/J-Lawyer endpoints depend on large helper blocks (`_upload_documents_to_*`, `_build_generation_prompts`, citation verification).
- Startup applies SQL migrations via `apply_schema_migrations()` before building listeners.

## Target Layout (minimal file fan-out)
```
app/
  app.py                       # FastAPI factory, limiter, startup/shutdown, router inclusion
  endpoints/
    __init__.py
    root.py                    # GET /
    classification.py          # /classify, /upload-direct (+Akten segmentation helpers)
    documents.py               # /documents*, /documents/stream, delete/reset helpers, broadcast utils
    ocr.py                     # OCR + anonymization + text storage utilities
    research_sources.py        # /research, /sources*, /debug-research, download workers
    generation.py              # /generate, /jlawyer/templates, /send-to-jlawyer (+prompt + citation helpers)
    system.py                  # /reset, /health, /favicon (admin/maintenance utilities)
  shared.py                    # Common enums/models (DocumentCategory, Pydantic schemas),
                               # reusable helpers (store/load text, limiter instance export, notifier helpers)
```
- Keep cross-cutting helpers/models in `shared.py` to avoid circular imports without creating large service/schema trees.
- Reuse existing modules (`events.py`, `kanzlei_gemini.py`, `database.py`) as-is.

## Migration Steps
1. **Baseline & Safety**
   - Run smoke tests (or note existing failures) and `rg "@app"` to map endpoint groupings.
   - Mark blocks to move: classification (~`@app.post("/classify")` onwards), documents (SSE + CRUD), OCR/anonymization, research/sources, generation/J-Lawyer, system/reset.
   - Capture current docker compose volume mounts since they will be adjusted later.

2. **Introduce Shared Module**
   - Create `app/shared.py` exporting:
     - `limiter` (so routers can decorate endpoints without importing the FastAPI app),
     - enums + Pydantic models used across groups (`DocumentCategory`, `ClassificationResult`, `ResearchRequest`, `ResearchResult`, `SavedSource`, `AddSourceRequest`, `Generation*`, `JLawyer*`, `Anonymization*`),
     - document text helpers (`store_document_text`, `load_document_text`, `delete_document_text`, `_text_path_for_document`, `_ensure_directory`, directory constants),
     - broadcast utilities (`_emit_event`, `_broadcast_documents_snapshot`, `_broadcast_sources_snapshot`, `_build_*_snapshot`, `_group_documents`),
     - any other helpers referenced by more than one endpoint module (e.g. `check_and_update_ocr_status_bg` if needed by both classification + documents).
   - Replace in-file references with imports from `app.shared`.

3. **Create `app/endpoints` Package**
   - Add `app/endpoints/__init__.py` exposing a `routers` list or helper to register routers in `app/app.py`.
   - Ensure package-level imports avoid side effects (only import router instances when needed).

4. **Move Endpoint Families Incrementally**
   - For each module, create a `router = APIRouter()` and move endpoints plus tightly-coupled helpers that are unique to that domain:
     - `root.py`: `/` (serve UI), keep `TEMPLATES_DIR` constant if only used here.
     - `classification.py`: `/classify`, `/upload-direct`, `classify_document`, `sanitize_filename`, `process_akte_segmentation`, plus any classification-specific constants.
     - `documents.py`: `/documents/stream`, `/documents`, `/documents/{filename}`, `/reset`, SSE generator, and helper functions specific to documents (e.g. event generator). Import broadcast utilities from `shared`.
     - `ocr.py`: `/documents/{document_id}/ocr`, `/documents/{document_id}/anonymize`, `/anonymize-file`, OCR/anonymization helpers (e.g. `perform_ocr_on_pdf`, `check_pdf_needs_ocr`, `anonymize_document_text`, `stitch_anonymized_text`).
     - `research_sources.py`: `/research`, `/sources*`, `/debug-research`, `search_asyl_net`, Gemini/Grok adapters, `download_and_update_source`, suggestion cache loading (move `ASYL_NET_*` constants here).
     - `generation.py`: `/generate`, `/jlawyer/templates`, `/send-to-jlawyer`, document collection functions, prompt builders, LLM upload helpers, citation verification.
     - `system.py`: `/health`, `/favicon`, any residual admin endpoints (if `/reset` stays in documents module, keep only remainder here).
   - Update rate-limiter decorators to use `@limiter.limit` imported from `app.shared`.
   - Replace direct references to `app` state with `Request.app` where necessary (e.g. background hub access) since modules won’t have direct `app` import.

5. **Slim Down `app/app.py`**
   - Keep startup/shutdown handlers, limiter initialisation, static mount, Postgres listener setup, and migration calls.
   - Import routers from `app.endpoints.*` and include them with tags matching existing semantics.
   - Remove moved helpers and redundant imports.

6. **Adjust Imports Elsewhere**
   - Update modules (tests, background scripts) referencing moved classes/functions to import from `app.shared` or the new endpoint module.
   - Confirm there are no circular imports (e.g. avoid `app.shared` importing routers).

7. **Update Docker Compose Mounts**
   - Replace individual file bind mounts in `app/docker-compose.yml` with a single directory mount (e.g. `./app:/app:delegated`) plus any required read-only subdirectories.
   - Ensure mounted directories still include `data`, `static`, `templates`, and runtime write targets (`/app/downloaded_sources`, `/app/uploads`).
   - Re-run `docker compose up` (or equivalent) to validate the container picks up code changes correctly with the new mount structure.

8. **Validation Pass**
   - Run `pytest` (if present) or the existing shell scripts (`test_sequential.sh`, `test_concurrent.sh`).
   - Manual smoke tests:
     - Upload/classify PDF (including Akte segmentation).
     - Stream documents via `/documents/stream`.
     - Trigger OCR and anonymization flows.
     - Execute `/research` with manual query and Bescheid attachment.
     - Generate document via `/generate` and send to J-Lawyer.
     - Ensure `/reset`, `/health`, and `/debug-research` respond.
   - Watch logs to verify migrations apply once, Postgres listeners connect, and SSE broadcasts fire.

9. **Documentation & Cleanup**
   - Update README or add a short section documenting the new `app/endpoints` layout and `app/shared` usage.
   - Remove any leftover unused imports/variables (run `pyproject` lint if available or `python -m compileall` sanity check).
   - Note migration of text cache paths (`OCR_TEXT_DIR`) remains centralised in `app.shared`.

## Progress (2025-11-14)
- `app/shared.py` created with shared limiter, models, broadcast helpers, and text-cache utilities; `app/app.py` now imports from it.
- `app/endpoints/` package initialised. Routers implemented for:
  - `root.py` (`/`)
  - `system.py` (`/health`, `/favicon`)
  - `classification.py` (classification + Akte segmentation, OCR background scheduling)
  - `documents.py` (SSE stream, listings, deletion, reset)
  - `ocr.py` (manual OCR, anonymisation, upload anonymiser)
- `app/app.py` now registers the above routers and has the related endpoint implementations removed.
- Research/sources endpoints and generation/J-Lawyer flows now live in `app/endpoints/research_sources.py` and `app/endpoints/generation.py`; residual imports in `app/app.py` will be trimmed as we stabilise the split.
- Docker compose now mounts the full application directory (`./:/app:delegated`) so code changes propagate without enumerating individual files.
- Ran `./venv/bin/python -m compileall app` as a quick sanity check; no compilation errors.

## Risks & Mitigations
- **Circular imports**: confine shared utilities to `app.shared`; endpoint modules should depend only on `database`, `models`, `shared`, external libs.
- **Background task scheduling**: ensure any `asyncio.create_task` calls still have access to `Request.app.state` or other shared context.
- **Migration/state coupling**: keep `apply_schema_migrations` and listener setup in `app/app.py`; do not move to avoid early import execution.
- **Duplicate constants**: when moving `ASYL_NET_*`, `DOWNLOADS_DIR`, etc., ensure single source of truth (prefer `app.shared` for directory paths, keep research-specific constants in its module).

## Acceptance Checklist
- [ ] App starts with no import errors; migrations still run.
- [ ] Rate limits enforced after refactor.
- [ ] Documents + sources SSE stream remains functional.
- [ ] OCR/anonymization text caching still works (files stored under `/app/ocr_text`).
- [ ] Research + generation endpoints behave as before (LLM integrations untouched).
- [ ] Documentation reflects new file layout.
- [ ] No unused helper code left orphaned in `app/app.py`.

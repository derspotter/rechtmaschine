# Rechtmaschine API / CLI Readiness Plan

## Goal

Make Rechtmaschine usable as a first-class automation target for terminal workflows, scripts, Codex/Claude/Gemini CLI agents, and external integrations.

The current API works for scripting, but it still carries too much browser-session logic:
- implicit active-case state
- filename-based document selection
- mixed response formats
- long-running operations tied directly to UI streams
- weak machine-readable result payloads

The target state is:
- explicit, stateless request semantics
- stable identifiers
- long-running job support
- better auth for automation
- stronger endpoint responses
- a smaller, simpler CLI skill layer

## Current Pain Points

### 1. Active-case coupling

Many workflows assume the user's active case is already set in session state.
This is convenient in the browser, but bad for CLI use.

Current practical workflow:
1. `POST /token`
2. `POST /cases/{case_id}/activate`
3. `GET /documents`
4. `POST /generate`

This creates hidden state dependencies and makes concurrent automation harder.

### 2. Filename-based document addressing

Selections currently use filenames for most document categories.
Because uploaded files get timestamp prefixes, clients must first call `/documents` and then build payloads from unstable-looking filenames.

That is brittle for automation.

### 3. Streaming-only long-running operations

`/generate` and `/query-documents` are designed primarily for the browser.
Streaming is good for UI, but not ideal for resumable automation.

There is no proper job abstraction for:
- generation
- query
- research

### 4. Weak endpoint result payloads

Some endpoints do work but do not return enough machine-usable information.

Example:
- `/send-to-jlawyer` returns success/message, but not the created j-lawyer document id or resolved case id.

### 5. Password-based automation auth

CLI usage currently depends on `/token` with username/password.
This works, but it is not ideal for automation.

### 6. Mixed endpoint semantics and formats

Current response styles differ significantly:
- `/query-documents` streams plain text
- `/generate` streams NDJSON
- `/research` returns JSON

This increases complexity for CLI wrappers.

## Design Principles

1. Prefer explicit request context over implicit active session state.
2. Prefer stable ids over filenames.
3. Keep browser compatibility while improving API ergonomics.
4. Add new fields/endpoints before removing old behavior.
5. Make long-running work resumable and inspectable.
6. Return enough metadata for downstream automation.

## Phase 1: Explicit Case Context

Status: Partially implemented.

### Objective

Allow major endpoints to operate without prior `activate` calls.

### Changes

Add optional explicit `case_id` to:
- `POST /generate`
- `POST /query-documents`
- `POST /research`
- `GET /documents`
- `GET /sources`
- `GET /drafts`

Also added to:
- `GET /documents/{filename}`
- `DELETE /documents/{filename}`
- `POST /documents/from-url`
- `GET /research/history`
- `GET /sources/download/{source_id}`
- `DELETE /sources/{source_id}`
- `DELETE /sources`
- `POST /sources`

### Semantics

If `case_id` is provided:
- use that case directly
- ignore active session case for that request

If `case_id` is omitted:
- keep current behavior and fall back to active case

### Benefit

CLI can operate statelessly and concurrently without hidden session mutation.

## Phase 2: Stable Document Selection by ID

Status: Implemented for core request models and major generation/query/research flows.

### Objective

Replace filename-centric payload construction with document-id-centric selection.

### Changes

Extend selection payloads to accept ids in parallel to filenames.

Examples:
- `bescheid.primary_id`
- `bescheid.other_ids`
- `vorinstanz.primary_id`
- `vorinstanz.other_ids`
- `anhoerung_ids`
- `rechtsprechung_ids`
- `akte_ids`
- `sonstiges_ids`

Keep current filename support for backward compatibility during migration.

Implemented in:
- shared `SelectedDocuments` schema
- `POST /generate`
- `POST /query-documents`
- `POST /research`

### Benefit

- less brittle scripting
- less dependence on timestamped upload names
- easier integration with machine clients

## Phase 3: Job-Based Generation API

Status: Implemented in a first usable form.

### Objective

Decouple long-running generation from a single browser stream.

### New Endpoints

- `POST /generate/jobs`
- `GET /generate/jobs/{job_id}`
- `GET /generate/jobs/{job_id}/events`
- `GET /generate/jobs/{job_id}/result`

Implemented behavior:
- jobs are persisted in `generation_jobs`
- execution starts in-process via background task after submission
- results are persisted with:
  - status
  - error message
  - final result payload
  - generated `draft_id`

Current limitation:
- execution is persisted, but workers are still in-process
- jobs are therefore inspectable and pollable, but not resumed automatically after app restart
- stale `queued` / `running` jobs are now marked `failed` on startup with `Job interrupted by app restart`

### Suggested Flow

1. client submits generation request
2. server returns `job_id`
3. client either:
   - polls job status, or
   - consumes event stream
4. client fetches final result explicitly

### Result payload should include

- generated draft id
- case id
- model used
- token usage
- estimated cost
- warnings
- used document ids
- final text

### Benefit

- resumability
- better CLI experience
- easier recovery if UI modal closes
- safer background execution

## Phase 4: Better Query API for CLI

Status: Implemented in a first usable form with job endpoints.

### Objective

Make `/query-documents` easier to automate.

### Options

Option A:
- keep current streaming endpoint
- add `POST /query-documents/jobs`

Option B:
- keep stream for UI
- add non-streaming variant for CLI

### Benefit

- cleaner shell automation
- fewer custom stream parsers
- easier tool integration

Implemented:
- `POST /query-documents/jobs`
- `GET /query-documents/jobs/{job_id}`
- `GET /query-documents/jobs/{job_id}/result`
- `GET /query-documents/jobs/{job_id}/events`

Current behavior:
- the original streaming `/query-documents` endpoint remains unchanged for the browser
- background query execution is persisted in `query_jobs`
- results are pollable and inspectable

Current limitation:
- like generation jobs, execution is still in-process
- jobs are therefore persistent and pollable, but not automatically resumed after app restart
- stale `queued` / `running` jobs are now marked `failed` on startup with `Job interrupted by app restart`

## Phase 5: Better Authentication for Automation

Status: Implemented in a first usable form.

### Objective

Avoid routine password-based CLI auth.

### Changes

Add personal API tokens:
- user-scoped
- revocable
- optionally named
- optionally expiring

### Endpoints

- `POST /api-tokens`
- `GET /api-tokens`
- `DELETE /api-tokens/{id}`

Implemented behavior:
- personal API tokens are stored hashed in `api_tokens`
- bearer auth now accepts either:
  - existing JWT access tokens, or
  - personal API tokens
- token list returns metadata only
- token creation returns the raw token once
- revocation is soft via `revoked_at`

Current limitation:
- no token scopes yet
- no UI for token management yet

### Benefit

- safer automation
- better agent integration
- less shell credential handling

## Phase 6: Job-Based Research API

Status: Implemented in a first usable form.

### Objective

Make research automatable without tying the client to one long-running HTTP request.

### Endpoints

- `POST /research/jobs`
- `GET /research/jobs/{job_id}`
- `GET /research/jobs/{job_id}/result`
- `GET /research/jobs/{job_id}/events`

Implemented behavior:
- jobs are persisted in `research_jobs`
- execution starts in-process via background task after submission
- results are persisted with:
  - status
  - error message
  - final result payload
  - persisted `research_run_id` when available

Current limitation:
- like generation/query jobs, execution is still in-process
- long-running research can therefore still monopolize the main app worker during execution
- stale `queued` / `running` jobs are now marked `failed` on startup with `Job interrupted by app restart`

### Benefit

- resumable/pollable research for CLI clients
- no need to keep one browser-like request open
- consistent job semantics across generation, query, and research

## Phase 7: Stronger Endpoint Responses

Status: Partially implemented.

### Objective

Return machine-usable metadata from successful writes.

### Immediate candidates

#### `/send-to-jlawyer`
Return:
- resolved j-lawyer case id
- input case reference
- created j-lawyer document id
- created filename
- template name

Implemented response fields:
- `requested_case_reference`
- `resolved_case_id`
- `template_folder`
- `template_name`
- `file_name`
- `created_document_id` when returned by j-lawyer
- raw parsed `jlawyer_response` when JSON is available

#### `/upload-direct`
Return:
- document id
- normalized filename
- category
- case id

Implemented:
- `document_id`
- `original_filename`
- `filename`
- `category`
- `case_id`
- `processing_status`
- `needs_ocr`
- `ocr_applied`
- `message`

#### `/generate`
Return in final metadata / result:
- draft id
- case id
- model used
- token usage
- estimated cost

Implemented for the streaming completion event:
- `draft_id`
- `case_id`
- `document_type`
- `model_used`
- `resolved_legal_area`
- `token_usage`
- `estimated_cost_usd`
- `word_count`

### Benefit

Reduces the need for follow-up list queries.

## Phase 8: Consistent API Formats

Status: Partially implemented.

### Objective

Reduce unnecessary variation in response handling.

### Recommendation

- Use NDJSON or SSE consistently for streaming endpoints.
- Use stable JSON envelopes for non-streaming endpoints.
- Document response schemas clearly in OpenAPI.

Implemented:
- job-based generation and query endpoints now expose stable JSON status/result resources
- the new workflow helper endpoints use typed JSON response models
- `/generate` final NDJSON `done` payload carries stable machine-usable metadata

Current limitation:
- legacy browser endpoints still use mixed formats by design:
  - `/query-documents` streams plain text
  - `/generate` streams NDJSON
  - `/research` returns JSON
- no unified stream envelope has been imposed on all legacy endpoints yet

### Benefit

Simpler clients, simpler CLI wrapper, fewer custom parsers.

## Phase 9: Canonical OpenAPI Contract

Status: Partially implemented.

### Objective

Make the OpenAPI spec the real integration contract.

### Changes

- ensure `/openapi.json` is stable and accurate
- use precise request/response models
- keep endpoint docs in sync with actual behavior

Implemented:
- newly added helper endpoints use explicit request/response models:
  - `WorkflowInventoryResponse`
  - `WorkflowJLawyerCaseResolveResponse`
  - `JLawyerSendDraftRequest`
- existing automation-facing endpoints already expose typed models for:
  - generation jobs
  - query jobs
  - API tokens
  - j-lawyer send
  - direct upload

Current limitation:
- several older endpoints still return loosely shaped `Dict[str, Any]` responses
- OpenAPI is now a usable contract for the automation surface, but not every legacy browser endpoint is fully normalized yet

### Benefit

Agents and CLI wrappers can inspect the API directly instead of relying on handwritten curl recipes.

## Phase 10: Thin Official CLI Wrapper

Status: Implemented in a first usable form.

### Objective

Provide a small official CLI on top of the HTTP API.

### Example commands

- `rechtmaschine login`
- `rechtmaschine cases list`
- `rechtmaschine cases activate`
- `rechtmaschine docs list --case ...`
- `rechtmaschine query ...`
- `rechtmaschine generate ...`
- `rechtmaschine research ...`
- `rechtmaschine jlawyer templates`
- `rechtmaschine jlawyer send ...`

Implemented:
- `scripts/rechtmaschine_cli.py`
- commands:
  - `login`
  - `whoami`
  - `cases list`
  - `inventory`
  - `generate-job submit|status|result`
  - `query-job submit|status|result`
  - `research-job submit|status|result`
  - `research`
  - `jlawyer templates`
  - `jlawyer resolve-case`
  - `jlawyer send-draft`
  - `api-tokens list|create|revoke`

### Benefit

- lower friction for operators
- easier agent usage
- fewer ad-hoc curl scripts

## Phase 11: Server-Side Workflow Helpers

Status: Implemented in a first usable form.

### Objective

Move common manual client logic into the server.

### Candidates

- resolve case by file number
- list case-scoped document inventory with stable ids and categories
- generate from existing draft id + additional prompt
- send draft to j-lawyer by draft id
- query documents by ids only

Implemented:
- `GET /workflow/jlawyer/resolve-case`
  - resolves a j-lawyer file number / case reference to the internal j-lawyer case id
- `GET /workflow/inventory`
  - returns case-scoped documents, sources, and recent drafts in one call
- `POST /workflow/jlawyer/send-draft`
  - sends an already saved draft to j-lawyer by `draft_id`
- query/generate/research already accept document ids in the core request payloads

Current limitation:
- no dedicated helper yet for “generate from existing draft id + additional prompt”

### Benefit

Makes the CLI skill smaller and the API itself more ergonomic.

## Recommended First 4 Changes

If only a small first phase is possible, implement these first:

1. explicit `case_id` on all major read/write endpoints
2. document-id based selection in parallel to filenames
3. job-based `/generate`
4. richer `/send-to-jlawyer` response payload

These four changes would move Rechtmaschine from a browser-oriented API to a much more robust automation API.

## Notes on Backward Compatibility

- Keep existing browser behavior working.
- Add new id-based and case-explicit fields before deprecating old ones.
- Support both old and new selection payload styles during migration.
- Keep `activate` for UI convenience even after stateless request support exists.

## Notes on the Existing CLI Skill

The current Codex skill at:
- `/home/jay/.codex/skills/rechtmaschine/SKILL.md`

is useful, but it currently compensates for API limitations by documenting workarounds:
- activate a case first
- always fetch documents first
- use timestamped filenames
- handle mixed streaming formats manually

A more CLI-ready API should reduce the amount of workflow knowledge that has to live in the skill.

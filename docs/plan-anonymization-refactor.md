# Plan: Move Anonymization Logic from Desktop to Server

## Context

All code lives in the same git repo (`/var/opt/docker/rechtmaschine/`). Desktop pulls from the same repo. The `anon/anonymization_service.py` and `service_manager.py` are already here — no SSH needed for edits, just `git pull` on desktop after.

## What Changes

Three files touched, one new module:

### 1. New: `app/anonymization/` module (extract from `anon/anonymization_service.py`)

Move these into the server's Python path so the Docker app can import them:

- **`app/anonymization/replacer.py`** — `safe_replace`, `_escape_fuzzy`, `_person_term_variants`, `safe_replace_case_numbers`, `apply_regex_replacements` + constants (`HONORIFICS`, `OCR_CONFUSABLES`)
- **`app/anonymization/filters.py`** — `filter_bamf_addresses`, `filter_non_person_group_labels` + constants (`BAMF_OFFICE_DATA`, `NON_PERSON_GROUP_TERMS`, `GROUP_CONTEXT_KEYWORDS`)
- **`app/anonymization/prompt.py`** — extraction prompt template, JSON schema, `build_ollama_payload(text, model, options)` function

These are direct moves from `anon/anonymization_service.py` lines ~30-313. No rewriting needed.

### 2. Modify: `app/endpoints/anonymization.py`

Replace `anonymize_document_text()`:
- Instead of POST to `desktop:8004/anonymize` (which returns anonymized text),
  POST to `desktop:8004/extract-entities` with the Ollama payload built by `prompt.py`
- Parse raw entity JSON from response
- Apply `filter_bamf_addresses()` + `filter_non_person_group_labels()` locally
- Apply `apply_regex_replacements()` locally
- Return `AnonymizationResult`

Delete `stitch_anonymized_text()` and `apply_regex_anonymization()` (the old server-side stitching — no longer needed since server does full replacement).

### 3. Modify: `service_manager.py`

Add `POST /extract-entities` endpoint:
- Receives `{prompt, model, format, options}` (complete Ollama payload)
- Enqueues as "anon" (same VRAM switching)
- `do_extract()` calls `http://localhost:11434/api/generate` directly
- Returns raw Ollama JSON response

Change `SERVICES["anon"]` from `kind: "process"` to `kind: "ollama"`:
- `is_service_running("anon")` → check Ollama `/api/ps` for loaded model
- `start_service("anon")` → warm model with empty prompt via Ollama API
- `kill_service("anon")` → already calls `unload_ollama_model()`, keep as-is

No more `anonymization_service.py` subprocess.

### 4. `anon/anonymization_service.py` — leave as-is for now, delete later

Keep old `/anonymize` endpoint on service_manager as deprecated fallback during transition.

## Verification

1. `git pull` on desktop, restart service_manager
2. Run `tests/bench_anonymize_pdfs.sh --dir tmp/retest_batch` against the 7 worst-case files
3. All previously-fixed leaks must remain fixed
4. Verify VRAM switching: OCR request → anon request → OCR request

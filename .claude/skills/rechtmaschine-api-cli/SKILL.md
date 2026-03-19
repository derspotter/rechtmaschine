---
name: rechtmaschine-api-cli
description: Use when Claude needs to operate Rechtmaschine from a terminal or script via its HTTP API instead of the browser UI. Covers login, token handling, listing cases/documents/sources, document query, generation, and research requests with curl against the local or remote FastAPI app.
---

# Rechtmaschine API CLI

Use the HTTP API when browser interaction is unnecessary or when you want reproducible shell workflows.

## Core workflow

1. Log in via `POST /token` using form-encoded credentials.
2. Reuse the bearer token for all subsequent requests.
3. Fetch current state before mutating anything:
   - `GET /cases`
   - `GET /documents`
   - `GET /sources`
4. Build request payloads from real filenames and source IDs returned by the API.
5. For streaming endpoints, prefer `curl -N`.

## Important payload rules

- Document selections use **filenames** for `anhoerung`, `bescheid`, `vorinstanz`, `rechtsprechung`, `akte`, `sonstiges`.
- `saved_sources` uses **UUIDs**, not titles.
- `/token` expects `application/x-www-form-urlencoded`, not JSON.
- `/query-documents` streams plain text.
- `/generate` streams NDJSON events.
- `/research` returns JSON.

## Minimal endpoint map

- `POST /token`
- `GET /cases`
- `GET /documents`
- `GET /sources`
- `POST /query-documents`
- `POST /generate`
- `POST /research`
- `GET /research/history`
- `GET /rechtsprechung/playbook`

## When to load the reference file

Read `references/curl-recipes.md` when you need:
- copy-paste curl commands
- JSON payload templates
- streaming examples
- troubleshooting for auth and selection mismatches

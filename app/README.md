# Rechtmaschine Backend (`app/`)

FastAPI backend and embedded frontend for Rechtmaschine. See the [root README](../README.md) for the project overview and [CLAUDE.md](../CLAUDE.md) for the architecture reference.

## Layout

- `main.py` — FastAPI wiring, schema migrations, startup events
- `job_worker.py` — background worker for generation/query/research jobs (separate container)
- `shared.py` — shared Pydantic models, AI client factories, helpers
- `models.py` — SQLAlchemy ORM models
- `auth.py` — JWT + API token auth
- `events.py` — Postgres LISTEN/NOTIFY → SSE broadcasting
- `endpoints/` — one router per concern (auth, cases, classification, documents, ocr, anonymization, research, generation, query, drafts, agent_memory, rag, …)
- `legal_texts/` — local German law texts with provision extraction
- `static/`, `templates/` — embedded frontend (vanilla JS, German UI)

## Setup

Configuration lives in `.env` in this directory (see `.env.example`). `SECRET_KEY` is mandatory.

Run the stack from the repo root:

```bash
cd ..
docker compose up -d
```

The app listens on port 8000 inside the Docker network; public access goes through Caddy at https://rechtmaschine.de.

## API

API endpoints require a Bearer token (JWT from `POST /token` or a long-lived API token); only `/token`, `/health`, and the HTML/static pages are public. For curl recipes see `../.claude/skills/rechtmaschine-api-cli/`. Interactive docs are available at `/docs` when `ENABLE_API_DOCS=true`.

Example:

```bash
TOKEN=$(curl -s -X POST https://rechtmaschine.de/token \
  -d "username=$USER_EMAIL" -d "password=$PASSWORD" | jq -r .access_token)

curl -s https://rechtmaschine.de/documents -H "Authorization: Bearer $TOKEN"
```

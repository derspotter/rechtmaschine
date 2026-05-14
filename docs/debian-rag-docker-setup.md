# Debian RAG Docker Setup

This runbook is for the `debian` machine in the three-machine Rechtmaschine setup.
It keeps OCR outside Docker as the existing user systemd service and runs the RAG stack in Docker Compose.

## Branch

Work on the Debian branch, not `master`:

```bash
git fetch origin
git switch master
git pull --ff-only
git switch codex/debian-rag-ocr || git switch -c codex/debian-rag-ocr
```

Debian-owned areas are `ocr/`, `rag/`, and `docs/debian-*`.
Avoid editing shared app/server files unless coordinated.

## OCR Baseline

OCR is already provided by the user systemd service:

```bash
systemctl --user status rechtmaschine-ocr --no-pager
curl -fsS http://127.0.0.1:8004/health
```

Run once so the user service starts before login after reboot:

```bash
sudo loginctl enable-linger justus
```

## Local Env

Create a local, untracked env file:

```bash
cp rag/.env.debian.example rag/.env.debian
chmod 600 rag/.env.debian
```

Edit at least:

```env
RAG_POSTGRES_PASSWORD=<local-password>
RAG_SERVICE_API_KEY=<shared-secret-for-server>
```

`rag/.env.debian` is excluded via `.git/info/exclude`; do not commit it.

## Start Docker RAG Stack

```bash
docker compose --env-file rag/.env.debian -f rag/docker-compose.debian.yml up -d --build
```

Services and ports:

- `rechtmaschine-rag-postgres`: `127.0.0.1:5433`
- `rechtmaschine-rag-embed`: `127.0.0.1:8085`
- `rechtmaschine-rag-rerank`: `127.0.0.1:8086`
- `rechtmaschine-rag-api`: `0.0.0.0:8090`

Check status:

```bash
docker compose --env-file rag/.env.debian -f rag/docker-compose.debian.yml ps
curl -fsS http://127.0.0.1:8085/health
curl -fsS http://127.0.0.1:8086/health
curl -fsS -H "X-API-Key: $(grep '^RAG_SERVICE_API_KEY=' rag/.env.debian | cut -d= -f2-)" \
  http://127.0.0.1:8090/v1/rag/health
```

## Smoke Upsert/Retrieve

Use an anonymized test chunk only:

```bash
API_KEY=$(grep '^RAG_SERVICE_API_KEY=' rag/.env.debian | cut -d= -f2-)

curl -fsS -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:8090/v1/rag/chunks/upsert \
  -d '{
    "chunks": [{
      "chunk_id": "smoke-001",
      "text": "Anonymisierter Testtext zu § 60 Abs. 5 AufenthG und medizinischer Versorgung.",
      "context_header": "[Smoke | Aufenthaltsrecht | anonymisiert]",
      "metadata": {"section_type": "legal_argument", "statute": "AufenthG", "paragraph": "§ 60"},
      "provenance": ["smoke"]
    }]
  }'

curl -fsS -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:8090/v1/rag/retrieve \
  -d '{"query": "medizinische Versorgung § 60 Abs. 5", "limit": 3, "use_reranker": true}'
```

## Server Integration Later

After smoke tests pass, the server branch should use:

```env
RAG_SERVICE_URL=http://debian:8090
RAG_SERVICE_API_KEY=<same shared secret>
```

Do not point server RAG calls at desktop. Desktop remains Qwen/anonymization only.

## Shutdown

```bash
docker compose --env-file rag/.env.debian -f rag/docker-compose.debian.yml down
```

Keep volumes for persistent RAG data. Use `down -v` only when intentionally deleting the RAG database.

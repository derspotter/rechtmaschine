# Debian RAG Docker Setup

This runbook is for the `debian` machine in the three-machine Rechtmaschine setup.
It keeps OCR outside Docker as the existing user systemd service and runs the RAG stack in Docker Compose.

## Branch

The RAG stack lives on `master` (the former `codex/debian-rag-ocr` branch was folded
into master in June 2026; do not switch back to it — its `ocr/` files are stale):

```bash
git switch master
git pull --ff-only
```

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

## Smoke Test

Runs health, upsert with server-side embedding, hybrid retrieve, reranked
retrieve, and cleanup against a throwaway `smoke_test` collection:

```bash
python rag/test_rag_api_smoke.py
```

It reads `RAG_SERVICE_API_KEY`/`RAG_API_PORT` from `rag/.env.debian` automatically.

## Server Integration

After smoke tests pass, the server `app/.env` should use:

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

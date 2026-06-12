#!/usr/bin/env python
"""End-to-end smoke test for the Debian RAG API.

Run on debian after `docker compose --env-file rag/.env.debian -f rag/docker-compose.debian.yml up -d`:

    python rag/test_rag_api_smoke.py

Exercises health, upsert (with server-side embedding), hybrid retrieve,
reranked retrieve, and cleanup via the delete endpoint. Uses a throwaway
collection so it never touches production chunks.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx


COLLECTION = "smoke_test"

CHUNKS = [
    {
        "chunk_id": "smoke-wehrdienst-1",
        "text": (
            "Dem Kläger droht bei Rückkehr nach Syrien die Einberufung zum Wehrdienst. "
            "Die Verweigerung des Militärdienstes wird vom Regime als oppositionelle "
            "Haltung gewertet und mit unverhältnismäßiger Bestrafung geahndet."
        ),
        "context_header": "Klagebegründung | Asyl | Syrien | Wehrdienstentziehung",
        "metadata": {"document_role": "Klagebegruendung", "legal_area": "Asyl", "country": "Syrien"},
    },
    {
        "chunk_id": "smoke-abschiebungsverbot-1",
        "text": (
            "Beim Kläger liegt ein Abschiebungsverbot nach § 60 Abs. 7 AufenthG vor. "
            "Die schwere Erkrankung kann im Herkunftsland nicht behandelt werden, eine "
            "wesentliche Verschlechterung des Gesundheitszustands wäre die Folge."
        ),
        "context_header": "Schriftsatz | Abschiebungsverbot | § 60 Abs. 7 AufenthG",
        "metadata": {"document_role": "Schriftsatz", "legal_area": "Abschiebungsverbot"},
    },
    {
        "chunk_id": "smoke-dublin-1",
        "text": (
            "Die Überstellung nach Italien ist unzulässig, weil dort systemische Mängel "
            "des Asylverfahrens und der Aufnahmebedingungen bestehen. Die Zuständigkeit "
            "ist auf die Bundesrepublik übergegangen."
        ),
        "context_header": "Eilantrag | Dublin | Italien",
        "metadata": {"document_role": "Eilantrag", "legal_area": "Dublin"},
    },
]


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent / ".env.debian"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_env_file()
    parser = argparse.ArgumentParser(description="RAG API smoke test")
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAG_API_URL", f"http://127.0.0.1:{os.getenv('RAG_API_PORT', '8090')}"),
    )
    parser.add_argument("--api-key", default=os.getenv("RAG_SERVICE_API_KEY", ""))
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    with httpx.Client(base_url=args.base_url, headers=headers, timeout=120.0) as client:
        print(f"RAG API smoke test against {args.base_url}")

        resp = client.get("/v1/rag/health")
        health = resp.json()
        check("health reachable", resp.status_code == 200, f"status={health.get('status')}")
        check("database ok", bool(health.get("details", {}).get("database")))
        check("embedder ok", bool(health.get("details", {}).get("embedder")))

        resp = client.post(
            "/v1/rag/chunks/upsert",
            json={"collection": COLLECTION, "chunks": CHUNKS},
        )
        check(
            "upsert 3 chunks (server-side embedding)",
            resp.status_code == 200 and resp.json().get("upserted") == 3,
            resp.text[:200] if resp.status_code != 200 else "",
        )

        resp = client.post(
            "/v1/rag/retrieve",
            json={
                "query": "Wehrdienstverweigerung Syrien Bestrafung",
                "collection": COLLECTION,
                "limit": 3,
            },
        )
        data = resp.json() if resp.status_code == 200 else {}
        top = (data.get("chunks") or [{}])[0]
        check(
            "hybrid retrieve finds Wehrdienst chunk first",
            resp.status_code == 200 and top.get("chunk_id") == "smoke-wehrdienst-1",
            f"top={top.get('chunk_id')} dense={data.get('retrieval', {}).get('dense_count')} "
            f"sparse={data.get('retrieval', {}).get('sparse_count')}",
        )

        resp = client.post(
            "/v1/rag/retrieve",
            json={
                "query": "Krankheit Behandlung Herkunftsland § 60 Abs. 7",
                "collection": COLLECTION,
                "limit": 3,
                "use_reranker": True,
            },
        )
        data = resp.json() if resp.status_code == 200 else {}
        top = (data.get("chunks") or [{}])[0]
        check(
            "reranked retrieve finds § 60 Abs. 7 chunk first",
            resp.status_code == 200 and top.get("chunk_id") == "smoke-abschiebungsverbot-1",
            f"reranker_applied={data.get('retrieval', {}).get('reranker_applied')}",
        )
        check("reranker applied", bool(data.get("retrieval", {}).get("reranker_applied")))

        resp = client.post("/v1/rag/chunks/delete", json={"collection": COLLECTION})
        check(
            "cleanup deletes smoke collection",
            resp.status_code == 200 and resp.json().get("deleted") == 3,
            resp.text[:200],
        )

    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILURE(S): ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

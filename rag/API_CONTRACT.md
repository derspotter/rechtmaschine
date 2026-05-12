# RAG API Contract (Frozen v1)
**Date:** February 6, 2026  
**Status:** Frozen for v1 implementation  
**Scope:** Server <-> Desktop RAG integration

## 1. Architecture Split
### Desktop (`/home/jayjag/rechtmaschine/rag` + `service_manager.py`)
- GPU-only operations:
  - query/chunk embedding (BGE-M3 via TEI or local model service)
  - optional reranking
  - OCR/anonymization already managed here
- Exposes a private API over Tailscale/LAN.

### Server (`app/`)
- No GPU dependency.
- Runs retrieval orchestration and vector DB access (Qdrant).
- Calls Desktop API for embeddings (and optional rerank).
- Exposes app-facing retrieval endpoint(s).

## 2. Versioning and Compatibility
- All new endpoints are versioned under `/v1/...`.
- v1 is backward-compatible for additive changes only.
- Breaking changes require `/v2/...`.

## 3. Security
- Header auth: `X-API-Key: <secret>`.
- If key is configured on target service, missing/invalid key returns `401`.
- Transport: private network only (Tailscale); no public exposure.

## 4. Common Conventions
- Content type: `application/json` unless file upload.
- Optional tracing header: `X-Request-ID`; service echoes it when present.
- Timestamps in ISO-8601 UTC.
- Error body format (all non-2xx):

```json
{
  "error": {
    "code": "string_code",
    "message": "Human-readable message",
    "retryable": false,
    "details": {}
  }
}
```

Common status codes:
- `400` bad request
- `401` unauthorized
- `404` not found
- `409` conflict
- `422` validation error
- `429` rate-limited
- `500` internal error
- `502` upstream dependency failed
- `503` service unavailable / queue saturated
- `504` upstream timeout

---

## 5. Desktop GPU API (called by server)
Base URL example: `http://desktop:8004/v1`

### 5.1 `GET /v1/health`
Purpose: liveness/readiness for embedding/rerank capability.

Response `200`:
```json
{
  "status": "healthy",
  "service": "rag-gpu-gateway",
  "embed": {
    "enabled": true,
    "model": "BAAI/bge-m3"
  },
  "rerank": {
    "enabled": false,
    "model": "BAAI/bge-reranker-v2-m3"
  }
}
```

### 5.2 `GET /v1/status`
Purpose: queue + active service state for VRAM arbitration.

Response `200`:
```json
{
  "current_service": "embed",
  "queued": {
    "ocr": 0,
    "anon": 1,
    "embed": 0,
    "rerank": 0
  },
  "keep_services_running": false
}
```

### 5.3 `POST /v1/embed`
Purpose: embed one text (query or chunk).

Request:
```json
{
  "text": "§ 60 Abs. 5 AufenthG Afghanistan",
  "with_sparse": true,
  "input_type": "query"
}
```

Fields:
- `text` (string, required)
- `with_sparse` (bool, default `true`)
- `input_type` (`query` | `passage`, default `query`)

Response `200`:
```json
{
  "model": "BAAI/bge-m3",
  "dense": [0.012, -0.033],
  "sparse": {
    "indices": [101, 5021],
    "values": [0.78, 0.45]
  },
  "dimension": 1024,
  "elapsed_ms": 42
}
```

Notes:
- `sparse` may be omitted if `with_sparse=false`.

### 5.4 `POST /v1/embed-batch`
Purpose: embed multiple chunks for ingestion.

Request:
```json
{
  "items": [
    {"id": "cb_001", "text": "...", "input_type": "passage"},
    {"id": "cb_002", "text": "...", "input_type": "passage"}
  ],
  "with_sparse": true
}
```

Response `200`:
```json
{
  "model": "BAAI/bge-m3",
  "dimension": 1024,
  "results": [
    {
      "id": "cb_001",
      "dense": [0.1, 0.2],
      "sparse": {"indices": [1, 2], "values": [0.4, 0.8]}
    }
  ],
  "failed": [],
  "elapsed_ms": 210
}
```

### 5.5 `POST /v1/rerank` (optional, feature-flagged)
Purpose: rerank top-N retrieved passages.

Request:
```json
{
  "query": "Welche Argumente zu §60 Abs.5 bei PTSD?",
  "documents": [
    {"id": "cb_001", "text": "..."},
    {"id": "cb_002", "text": "..."}
  ],
  "top_k": 8
}
```

Response `200`:
```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "results": [
    {"id": "cb_002", "score": 0.93, "rank": 1},
    {"id": "cb_001", "score": 0.81, "rank": 2}
  ],
  "elapsed_ms": 180
}
```

---

## 6. Server RAG API (called by app/backend)
Base URL example: `http://server:8000/v1/rag`

### 6.1 `GET /v1/rag/health`
Purpose: server readiness for retrieval orchestration.

Response `200`:
```json
{
  "status": "healthy",
  "qdrant_ok": true,
  "desktop_embedder_ok": true
}
```

### 6.2 `POST /v1/rag/chunks/upsert`
Purpose: ingestion upsert into vector store (idempotent by `chunk_id`).

Request:
```json
{
  "chunks": [
    {
      "chunk_id": "cb_001",
      "text": "Original chunk text",
      "context_header": "[Klage | 24/014 | VG Düsseldorf]",
      "metadata": {
        "section_type": "legal_argument",
        "statute": "AufenthG",
        "paragraph": "§ 60",
        "applicant_origin": "Afghanistan",
        "court": "VG Düsseldorf",
        "date": "2024-10-13",
        "citations": ["BVerwG 1 C 10.16"]
      },
      "provenance": ["doc_014"],
      "dense": [0.1, 0.2],
      "sparse": {
        "indices": [1, 2],
        "values": [0.4, 0.8]
      }
    }
  ]
}
```

Response `200`:
```json
{
  "upserted": 1,
  "collection": "rag_chunks"
}
```

### 6.3 `POST /v1/rag/retrieve`
Purpose: hybrid retrieval for generation/query flows.

Request:
```json
{
  "query": "§ 60 Abs. 5 AufenthG PTSD Afghanistan",
  "limit": 8,
  "dense_top_k": 50,
  "sparse_top_k": 50,
  "use_reranker": false,
  "filters": {
    "section_type": ["medical_risk", "legal_argument"],
    "statute": "AufenthG",
    "paragraph": "§ 60",
    "applicant_origin": "Afghanistan",
    "court": "VG Düsseldorf",
    "date_from": "2022-01-01",
    "date_to": "2026-12-31",
    "citations": []
  }
}
```

Response `200`:
```json
{
  "query": "§ 60 Abs. 5 AufenthG PTSD Afghanistan",
  "retrieval": {
    "fusion": "rrf",
    "dense_top_k": 50,
    "sparse_top_k": 50,
    "limit": 8,
    "reranker_applied": false
  },
  "chunks": [
    {
      "chunk_id": "cb_001",
      "score": 0.87,
      "text": "...",
      "context_header": "[Klage | 24/014 | VG Düsseldorf]",
      "metadata": {
        "section_type": "legal_argument"
      },
      "provenance": ["doc_014"]
    }
  ]
}
```

---

## 7. Timeouts and Retries (v1)
- Server -> Desktop embed/rerank timeout:
  - query embed: 20s
  - batch embed: 120s
  - rerank: 30s
- Retry policy: max 2 retries, exponential backoff (`250ms`, `1000ms`) for `502/503/504`.
- No retry on `400/401/422`.

## 8. Idempotency
- `POST /v1/rag/chunks/upsert` is idempotent by `chunk_id`.
- `POST /v1/embed` and `/v1/embed-batch` are pure compute (safe to retry).

## 9. Deferred to v2 (Not in this freeze)
- Query expansion (HyDE/multi-query)
- GraphRAG/citation graph traversal
- Streaming retrieval responses
- Multi-tenant collections

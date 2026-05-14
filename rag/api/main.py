"""Debian-local RAG API for anonymized chunk storage and retrieval."""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from typing import Any, Iterable, Optional

import httpx
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field


DATABASE_URL = os.getenv(
    "RAG_DATABASE_URL",
    "postgresql://rechtmaschine:rechtmaschine@rag-postgres:5432/rechtmaschine_rag",
)
EMBED_URL = os.getenv("RAG_EMBED_URL", "http://rag-embed/embed").rstrip("/")
RERANK_URL = os.getenv("RAG_RERANK_URL", "http://rag-rerank/rerank").rstrip("/")
RAG_SERVICE_API_KEY = os.getenv("RAG_SERVICE_API_KEY", "").strip()
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
EMBED_DIMENSION = int(os.getenv("RAG_EMBED_DIMENSION", "1024"))
EMBED_TIMEOUT_SECONDS = float(os.getenv("RAG_EMBED_TIMEOUT_SECONDS", "60"))
RERANK_TIMEOUT_SECONDS = float(os.getenv("RAG_RERANK_TIMEOUT_SECONDS", "30"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))

app = FastAPI(title="Rechtmaschine Debian RAG API", version="0.1.0")


class RagUpsertChunk(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)
    context_header: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[str] = Field(default_factory=list)
    dense: list[float] = Field(default_factory=list)
    sparse: Optional[dict[str, Any]] = None


class RagUpsertRequest(BaseModel):
    chunks: list[RagUpsertChunk] = Field(min_length=1, max_length=128)
    collection: str = "rag_chunks"


class RagFilters(BaseModel):
    section_type: Optional[list[str]] = None
    statute: Optional[str] = None
    paragraph: Optional[str] = None
    applicant_origin: Optional[str] = None
    court: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    citations: Optional[list[str]] = None


class RagRetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=12)
    dense_top_k: int = Field(default=50, ge=1, le=200)
    sparse_top_k: int = Field(default=50, ge=1, le=200)
    use_reranker: bool = False
    filters: Optional[RagFilters] = None
    collection: str = "rag_chunks"


class RerankDocument(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: list[RerankDocument]
    top_k: int = 8


def _error(code: str, message: str, status_code: int, retryable: bool = False, details: Optional[dict[str, Any]] = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "details": details or {},
            }
        },
    )


def _validate_api_key(x_api_key: Optional[str]) -> None:
    if not RAG_SERVICE_API_KEY:
        return
    if x_api_key == RAG_SERVICE_API_KEY:
        return
    raise _error("unauthorized", "Missing or invalid X-API-Key", 401)


@contextmanager
def _db_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _json(value: Any) -> psycopg2.extras.Json:
    return psycopg2.extras.Json(value)


def _vector_literal(vector: list[float]) -> str:
    if len(vector) != EMBED_DIMENSION:
        raise _error(
            "invalid_embedding_dimension",
            f"Expected dense embedding dimension {EMBED_DIMENSION}, got {len(vector)}",
            422,
        )
    if any(not math.isfinite(float(value)) for value in vector):
        raise _error("invalid_embedding", "Dense embedding contains non-finite values", 422)
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    payload = {"inputs": texts}
    try:
        with httpx.Client(timeout=EMBED_TIMEOUT_SECONDS) as client:
            response = client.post(EMBED_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        raise _error("embed_timeout", f"Embedding request timed out: {exc}", 504, retryable=True)
    except Exception as exc:
        raise _error("embed_failed", f"Embedding request failed: {exc}", 502, retryable=True)

    vectors = _parse_embedding_response(data)
    if len(vectors) != len(texts):
        raise _error(
            "embed_bad_response",
            f"Embedding service returned {len(vectors)} vectors for {len(texts)} texts",
            502,
            retryable=True,
        )
    return vectors


def _parse_embedding_response(data: Any) -> list[list[float]]:
    if isinstance(data, list):
        if data and all(isinstance(value, (int, float)) for value in data):
            return [[float(value) for value in data]]
        vectors: list[list[float]] = []
        for item in data:
            if isinstance(item, list):
                vectors.append([float(value) for value in item])
            elif isinstance(item, dict) and isinstance(item.get("embedding"), list):
                vectors.append([float(value) for value in item["embedding"]])
        if vectors:
            return vectors
    if isinstance(data, dict):
        if isinstance(data.get("embedding"), list):
            return [[float(value) for value in data["embedding"]]]
        if isinstance(data.get("embeddings"), list):
            return _parse_embedding_response(data["embeddings"])
        if isinstance(data.get("data"), list):
            return [
                [float(value) for value in item["embedding"]]
                for item in data["data"]
                if isinstance(item, dict) and isinstance(item.get("embedding"), list)
            ]
    raise _error("embed_bad_response", "Cannot parse embedding response", 502, retryable=True)


def _embedding_input(chunk: RagUpsertChunk) -> str:
    return f"{chunk.context_header}\n\n{chunk.text}" if chunk.context_header else chunk.text


def _filter_sql(filters: Optional[RagFilters]) -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    metadata_fields = {
        "statute": filters.statute,
        "paragraph": filters.paragraph,
        "applicant_origin": filters.applicant_origin,
        "court": filters.court,
    }
    for key, value in metadata_fields.items():
        if value:
            clauses.append("metadata ->> %s = %s")
            params.extend([key, value])
    if filters.section_type:
        clauses.append("metadata ->> 'section_type' = ANY(%s)")
        params.append(filters.section_type)
    if filters.date_from:
        clauses.append("metadata ->> 'date' >= %s")
        params.append(filters.date_from)
    if filters.date_to:
        clauses.append("metadata ->> 'date' <= %s")
        params.append(filters.date_to)
    if filters.citations:
        clauses.append("metadata -> 'citations' ?| %s")
        params.append(filters.citations)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(f"({clause})" for clause in clauses), params


def _rrf_merge(dense_rows: Iterable[dict[str, Any]], sparse_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source, rows in (("dense", dense_rows), ("sparse", sparse_rows)):
        for rank, row in enumerate(rows, start=1):
            chunk_id = row["chunk_id"]
            existing = merged.setdefault(chunk_id, dict(row, score=0.0, ranks={}))
            existing["score"] += 1.0 / (RRF_K + rank)
            existing["ranks"][source] = rank
            if source == "dense":
                existing["dense_score"] = float(row.get("dense_score") or 0.0)
            if source == "sparse":
                existing["sparse_score"] = float(row.get("sparse_score") or 0.0)
    return merged


def _rerank(query: str, candidates: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], bool]:
    if not candidates:
        return candidates, False
    payload = {
        "query": query,
        "texts": [candidate["text"] for candidate in candidates],
        "top_n": min(limit, len(candidates)),
    }
    try:
        with httpx.Client(timeout=RERANK_TIMEOUT_SECONDS) as client:
            response = client.post(RERANK_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return candidates[:limit], False

    scores = _parse_rerank_response(data, candidates)
    if not scores:
        return candidates[:limit], False
    ordered = sorted(candidates, key=lambda item: scores.get(item["chunk_id"], float("-inf")), reverse=True)
    for item in ordered:
        if item["chunk_id"] in scores:
            item["score"] = float(scores[item["chunk_id"]])
    return ordered[:limit], True


def _parse_rerank_response(data: Any, candidates: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    if isinstance(data, dict):
        data = data.get("results") or data.get("data") or data
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            index = item.get("index", item.get("rank", idx))
            score = item.get("score", item.get("relevance_score"))
            if score is None:
                continue
            try:
                candidate = candidates[int(index)]
            except Exception:
                continue
            scores[candidate["chunk_id"]] = float(score)
    return scores


@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    response = await call_next(request)
    request_id = request.headers.get("X-Request-ID")
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


@app.get("/v1/rag/health")
def health() -> dict[str, Any]:
    details: dict[str, Any] = {
        "database": False,
        "embedder": False,
        "reranker": False,
        "embed_url": EMBED_URL,
        "rerank_url": RERANK_URL,
    }
    status = "healthy"
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            details["database"] = True
    except Exception as exc:
        details["database_error"] = str(exc)
        status = "degraded"

    for key, url in (("embedder", EMBED_URL), ("reranker", RERANK_URL)):
        health_url = url.rsplit("/", 1)[0] + "/health"
        try:
            response = httpx.get(health_url, timeout=3.0)
            details[key] = response.status_code < 500
        except Exception as exc:
            details[f"{key}_error"] = str(exc)
            if key == "embedder":
                status = "degraded"

    return {
        "status": status,
        "qdrant_ok": details["database"],
        "desktop_embedder_ok": details["embedder"],
        "details": details,
    }


@app.post("/v1/rag/chunks/upsert")
def upsert_chunks(body: RagUpsertRequest, x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _validate_api_key(x_api_key)
    chunks = body.chunks
    missing_embedding = [chunk for chunk in chunks if not chunk.dense]
    if missing_embedding:
        vectors = _embed_texts([_embedding_input(chunk) for chunk in missing_embedding])
        for chunk, vector in zip(missing_embedding, vectors):
            chunk.dense = vector

    rows = [
        (
            chunk.chunk_id,
            body.collection,
            chunk.text,
            chunk.context_header,
            _json(chunk.metadata),
            _json(chunk.provenance),
            _vector_literal(chunk.dense),
            _json(chunk.sparse) if chunk.sparse is not None else None,
        )
        for chunk in chunks
    ]

    with _db_conn() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO rag_chunks (
                chunk_id, collection, text, context_header, metadata, provenance, dense, sparse
            ) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                collection = EXCLUDED.collection,
                text = EXCLUDED.text,
                context_header = EXCLUDED.context_header,
                metadata = EXCLUDED.metadata,
                provenance = EXCLUDED.provenance,
                dense = EXCLUDED.dense,
                sparse = EXCLUDED.sparse,
                updated_at = now()
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s::vector, %s)",
        )
    return {"upserted": len(rows), "collection": body.collection, "warnings": []}


@app.post("/v1/rag/retrieve")
def retrieve(body: RagRetrieveRequest, x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _validate_api_key(x_api_key)
    started = time.time()
    query_vector = _embed_texts([body.query])[0]
    vector_literal = _vector_literal(query_vector)
    filter_clause, filter_params = _filter_sql(body.filters)

    with _db_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT chunk_id, text, context_header, metadata, provenance,
                   1 - (dense <=> %s::vector) AS dense_score
            FROM rag_chunks
            WHERE collection = %s {filter_clause}
            ORDER BY dense <=> %s::vector
            LIMIT %s
            """,
            [vector_literal, body.collection, *filter_params, vector_literal, body.dense_top_k],
        )
        dense_rows = list(cur.fetchall())

        cur.execute(
            f"""
            SELECT chunk_id, text, context_header, metadata, provenance,
                   ts_rank_cd(search_text, websearch_to_tsquery('german', %s)) AS sparse_score
            FROM rag_chunks
            WHERE collection = %s
              AND search_text @@ websearch_to_tsquery('german', %s)
              {filter_clause}
            ORDER BY sparse_score DESC
            LIMIT %s
            """,
            [body.query, body.collection, body.query, *filter_params, body.sparse_top_k],
        )
        sparse_rows = list(cur.fetchall())

    merged = _rrf_merge(dense_rows, sparse_rows)
    candidates = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    reranker_applied = False
    if body.use_reranker:
        candidates, reranker_applied = _rerank(body.query, candidates, body.limit)
    else:
        candidates = candidates[: body.limit]

    chunks = [
        {
            "chunk_id": item["chunk_id"],
            "score": float(item["score"]),
            "text": item["text"],
            "context_header": item.get("context_header"),
            "metadata": item.get("metadata") or {},
            "provenance": item.get("provenance") or [],
        }
        for item in candidates
    ]
    return {
        "query": body.query,
        "retrieval": {
            "fusion": "rrf",
            "dense_top_k": body.dense_top_k,
            "sparse_top_k": body.sparse_top_k,
            "limit": body.limit,
            "reranker_applied": reranker_applied,
            "dense_count": len(dense_rows),
            "sparse_count": len(sparse_rows),
            "elapsed_ms": int((time.time() - started) * 1000),
        },
        "chunks": chunks,
    }

"""Simple retrieval smoke test for RAG-style chunk ranking.

This script builds a query embedding, embeds candidate chunk texts, computes cosine
similarity, and returns top-k matches. It is intentionally lightweight and does
not require pgvector/Qdrant to validate the retrieval math locally.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx


RAG_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNKS_DIR = RAG_DIR / "data" / "chunks"
DEFAULT_OUT_DIR = RAG_DIR / "data" / "embeddings"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG retrieval smoke test (local cosine).")
    parser.add_argument(
        "--query",
        required=True,
        help="Free-text retrieval query.",
    )
    parser.add_argument(
        "--chunk-file",
        type=Path,
        default=None,
        help="JSONL chunk file. If omitted uses latest file in rag/data/chunks.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help="Fallback chunk directory used when --chunk-file is not set.",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=25,
        help="Maximum number of chunks considered in retrieval.",
    )
    parser.add_argument(
        "--hard-limit",
        type=int,
        default=400,
        help="Hard safety cap when selecting chunks from file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many results to return.",
    )
    parser.add_argument(
        "--provider",
        choices=["tei", "openai"],
        default="tei",
        help=(
            "Endpoint style. 'tei' posts {'inputs': [...]} to /embed; "
            "'openai' posts {'input': [...], 'model': ...} to /v1/embeddings"
        ),
    )
    parser.add_argument(
        "--url",
        default=os.getenv("RAG_EMBED_URL", "http://127.0.0.1:8085/embed"),
        help="Embedding endpoint URL.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3"),
        help="Model name for OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("RAG_EMBED_API_KEY"),
        help="Optional API key header (Rag API: X-API-Key, OpenAI-style can use Authorization).",
    )
    parser.add_argument(
        "--include-context-header",
        action="store_true",
        help="Prepend chunk context_header to text before embedding",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Maximum texts per batch request to embedding endpoint.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for output summary JSON.",
    )
    parser.add_argument(
        "--out-prefix",
        default="retrieval_smoke",
        help="Filename prefix for summary output.",
    )
    parser.add_argument(
        "--timeout-per-item",
        action="store_true",
        help="Record per-item latency approximation from batched requests.",
    )
    return parser


def _latest_file(chunks_dir: Path) -> Path:
    candidates = sorted(
        chunks_dir.glob("chunks_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No chunks_*.jsonl in {chunks_dir}")
    return candidates[0]


def _iter_chunks(path: Path, hard_limit: int):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if hard_limit and count >= hard_limit:
                break

            payload = json.loads(line)
            text = (payload.get("text") or "").strip()
            if not text:
                continue

            chunk_id = str(
                payload.get("chunk_id")
                or payload.get("source_rel_path")
                or f"row_{count}"
            )
            context_header = (payload.get("context_header") or "").strip()
            metadata = payload.get("metadata") or {}
            yield {
                "chunk_id": chunk_id,
                "text": text,
                "context_header": context_header,
                "metadata": metadata,
            }
            count += 1


def _select_text(item: dict[str, Any], include_context: bool) -> str:
    text = str(item["text"]).strip()
    if include_context:
        header = item["context_header"].strip()
        if header:
            return f"{header}\n\n{text}"
    return text


def _build_payload_provider(payload_texts: list[str], provider: str, model: str) -> dict[str, Any]:
    if provider == "openai":
        return {
            "model": model,
            "input": payload_texts,
            "encoding_format": "float",
        }
    return {"inputs": payload_texts}


def _build_headers(provider: str, api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if not api_key:
        return headers
    if provider == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["X-API-Key"] = api_key
    return headers


def _extract_vectors(payload: Any) -> list[list[float]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            vectors = []
            for item in data:
                if isinstance(item, list):
                    vectors.append([float(v) for v in item])
                elif isinstance(item, dict):
                    if "embedding" in item and isinstance(item["embedding"], list):
                        vectors.append([float(v) for v in item["embedding"]])
                    elif "vector" in item and isinstance(item["vector"], list):
                        vectors.append([float(v) for v in item["vector"]])
                    else:
                        raise ValueError("Unsupported item format in data[]")
                else:
                    raise ValueError("Unsupported item type in data[]")
            return vectors

        if "embedding" in payload and isinstance(payload["embedding"], list):
            return [[float(v) for v in payload["embedding"]]]
        if "vector" in payload and isinstance(payload["vector"], list):
            return [[float(v) for v in payload["vector"]]]

    if isinstance(payload, list):
        vectors = []
        for item in payload:
            if isinstance(item, list):
                vectors.append([float(v) for v in item])
            elif isinstance(item, dict) and "embedding" in item and isinstance(item["embedding"], list):
                vectors.append([float(v) for v in item["embedding"]])
            else:
                raise ValueError("Unsupported top-level payload list format")
        return vectors

    raise ValueError(f"Cannot parse embedding response of type {type(payload).__name__}")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for a, b in zip(left, right, strict=False):
        dot += a * b
        n1 += a * a
        n2 += b * b

    denom = math.sqrt(n1) * math.sqrt(n2)
    if denom <= 0:
        return 0.0
    return dot / denom


@dataclass(frozen=True)
class ScoredResult:
    chunk_id: str
    score: float
    text_preview: str
    metadata: dict[str, Any]
    elapsed_ms: float


def _chunk_embeddings(
    chunks: list[dict[str, Any]],
    endpoint: str,
    headers: dict[str, str],
    provider: str,
    model: str,
    batch_size: int,
    timeout: float,
    include_context: bool,
    timeout_per_item: bool,
) -> tuple[list[list[float]], list[float]]:
    if batch_size <= 0:
        batch_size = 1

    vectors: list[list[float]] = []
    request_times: list[float] = []

    with httpx.Client(timeout=timeout) as client:
        for idx in range(0, len(chunks), batch_size):
            batch = chunks[idx : idx + batch_size]
            texts = [_select_text(item, include_context) for item in batch]
            payload = _build_payload_provider(texts, provider, model)
            start = time.perf_counter()
            response = client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000
            request_times.append(elapsed_ms)

            batch_vectors = _extract_vectors(response.json())
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} vectors, got {len(batch_vectors)}"
                )
            vectors.extend(batch_vectors)

    if not request_times:
        return vectors, []

    avg_request = mean(request_times)
    if timeout_per_item:
        if not vectors:
            per_item = 0.0
        else:
            per_item = avg_request / len(vectors)
    else:
        per_item = avg_request
    return vectors, [per_item] * len(vectors)


def _iter_chunks_limited(chunk_file: Path, hard_limit: int) -> list[dict[str, Any]]:
    chunks = list(_iter_chunks(chunk_file, hard_limit))
    return chunks


def _find_chunk_file(chunk_file: Path | None, chunks_dir: Path) -> Path:
    if chunk_file:
        return chunk_file
    return _latest_file(chunks_dir)


def main() -> int:
    args = build_parser().parse_args()

    if args.top_k < 1:
        print("--top-k must be >= 1")
        return 1

    if args.batch_size < 1:
        args.batch_size = 1

    endpoint = f"{args.url.rstrip('/')}/embeddings" if args.provider == "openai" and not args.url.rstrip("/").endswith("/embeddings") else args.url

    chunk_file = _find_chunk_file(args.chunk_file, args.chunks_dir)
    all_chunks = _iter_chunks_limited(chunk_file, args.hard_limit)
    if not all_chunks:
        print(f"No usable chunks in {chunk_file}")
        return 1

    chunks = all_chunks[: args.chunk_limit] if args.chunk_limit else all_chunks
    if not chunks:
        print("No chunks selected after limit filters.")
        return 1

    headers = _build_headers(args.provider, args.api_key)

    try:
        query_vector, _ = _chunk_embeddings(
            [{"text": args.query, "context_header": "", "metadata": {}}],
            endpoint,
            headers,
            args.provider,
            args.model,
            args.batch_size,
            args.timeout,
            False,
            args.timeout_per_item,
        )
        if not query_vector:
            print("No query vector returned")
            return 1
        qv = query_vector[0]

        chunk_vectors, per_item_ms = _chunk_embeddings(
            chunks,
            endpoint,
            headers,
            args.provider,
            args.model,
            args.batch_size,
            args.timeout,
            args.include_context_header,
            args.timeout_per_item,
        )
    except httpx.HTTPStatusError as exc:
        print(f"HTTP error: {exc.response.status_code} - {exc.response.text}")
        return 1
    except httpx.RequestError as exc:
        print(f"Request error: {exc}")
        return 1
    except ValueError as exc:
        print(f"Embedding response parsing error: {exc}")
        return 1

    if len(chunk_vectors) != len(chunks):
        print(
            f"Embedding size mismatch: got {len(chunk_vectors)} vectors for {len(chunks)} chunks"
        )
        return 1

    results: list[ScoredResult] = []
    for idx, chunk in enumerate(chunks):
        score = _cosine_similarity(qv, chunk_vectors[idx])
        text_preview = str(chunk.get("text", "")).strip().replace("\n", " ")
        if len(text_preview) > 220:
            text_preview = text_preview[:217] + "..."
        results.append(
            ScoredResult(
                chunk_id=chunk["chunk_id"],
                score=score,
                text_preview=text_preview,
                metadata=chunk.get("metadata") or {},
                elapsed_ms=(per_item_ms[idx] if per_item_ms else 0.0),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    top = results[: args.top_k]

    print("Query:", args.query)
    print("Chunk file:", chunk_file)
    print("Matches:")
    for rank, hit in enumerate(top, start=1):
        meta = hit.metadata
        file_info = meta.get("source_rel_path") or meta.get("filename") or ""
        if file_info:
            file_info = f" | {file_info}"
        print(
            f"  {rank:>2}. score={hit.score:.4f} id={hit.chunk_id}{file_info} "
            f"(approx_ms={hit.elapsed_ms:.2f})"
        )
        print(f"     {hit.text_preview}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    summary = {
        "provider": args.provider,
        "endpoint": endpoint,
        "query": args.query,
        "chunk_file": str(chunk_file),
        "chunk_candidates": len(chunks),
        "top_k": args.top_k,
        "top": [
            {
                "rank": idx,
                "chunk_id": hit.chunk_id,
                "score": hit.score,
                "preview": hit.text_preview,
                "metadata": hit.metadata,
            }
            for idx, hit in enumerate(top, start=1)
        ],
        "query_embedding_dim": len(qv),
    }

    out_path = out_dir / f"{args.out_prefix}_{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary: {out_path}")
    print("Retrieval smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

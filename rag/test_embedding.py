"""Smoke test for RAG embedding services.

This utility tries a small batch embedding request against a configured endpoint
(OpenAI-compatible or TEI-compatible) and prints a compact validation report.
"""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Test embedding endpoint with sample chunk text.")
    parser.add_argument(
        "--chunk-file",
        type=Path,
        default=None,
        help=(
            "JSONL chunk file. If omitted and --use-text-sample is not set, uses the "
            "latest file in rag/data/chunks."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help="Fallback chunk directory used when --chunk-file is not set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "How many chunks to send in the embedding request. "
            "0 means all available chunks in file (capped by --hard-limit)."
        ),
    )
    parser.add_argument(
        "--hard-limit",
        type=int,
        default=100,
        help="Hard safety cap when no explicit limit is set."
        )
    parser.add_argument(
        "--use-text-sample",
        action="store_true",
        help="Use built-in sample texts instead of chunk file",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("RAG_EMBED_URL", "http://127.0.0.1:8085/embed"),
        help="Embedding endpoint URL.",
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
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to store raw response JSON and summary report.",
    )
    parser.add_argument(
        "--include-context-header",
        action="store_true",
        help="Prepend chunk context_header to text before embedding",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Send embeddings in single requests per item instead of one batch request",
    )
    return parser


@dataclass(frozen=True)
class EmbeddingResult:
    """Single embedding result record for reporting."""

    source_id: str
    vector_len: int
    elapsed_ms: float
    has_sparse: bool
    sparse_len: int | None


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
            chunk_id = str(payload.get("chunk_id") or payload.get("source_rel_path") or f"row_{count}")
            context_header = (payload.get("context_header") or "").strip()
            yield chunk_id, text, context_header
            count += 1


def _default_samples() -> list[tuple[str, str, str]]:
    return [
        (
            "sample-001",
            "§ 60 Abs. 5 AufenthG bietet Schutz bei erheblicher Gefahr für Leib und Leben.",
            "[sample | asylum | retrieval]",
        ),
        (
            "sample-002",
            "Die Rückführung in Afghanistan ist unzumutbar bei schwerer Erkrankung oder fehlender medizinischer Versorgung.",
            "[sample | medical risk]",
        ),
        (
            "sample-003",
            "Unzumutbarkeit des Wegfalls der inländischen Fluchtalternative bei individuellen Gefährdungslagen.",
            "[sample | internal protection alternative]",
        ),
    ]


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


def _extract_vectors(payload: Any) -> tuple[list[list[float]], list[Any] | None]:
    """Return dense vectors and optional sparse payload list."""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            vectors: list[list[float]] = []
            sparse: list[Any] = []
            for item in data:
                if isinstance(item, dict):
                    if "embedding" in item and isinstance(item["embedding"], list):
                        vectors.append([float(v) for v in item["embedding"]])
                    elif "vector" in item and isinstance(item["vector"], list):
                        vectors.append([float(v) for v in item["vector"]])
                    elif isinstance(item, list):
                        vectors.append([float(v) for v in item])
                    else:
                        raise ValueError("Unsupported item format in data[]")

                    if (
                        "sparse_indices" in item
                        or "sparse_values" in item
                        or "sparse" in item
                    ):
                        sparse.append(
                            {
                                "indices": item.get("sparse_indices", []),
                                "values": item.get("sparse_values", item.get("sparse", [])),
                            }
                        )
                else:
                    raise ValueError("Unsupported item type in data[]")
            return vectors, sparse or None

        if "embedding" in payload and isinstance(payload["embedding"], list):
            return [[float(v) for v in payload["embedding"]]], None
        if "vector" in payload and isinstance(payload["vector"], list):
            return [[float(v) for v in payload["vector"]]], None

    if isinstance(payload, list):
        vectors = []
        for item in payload:
            if isinstance(item, list):
                vectors.append([float(v) for v in item])
            elif isinstance(item, dict) and "embedding" in item:
                vectors.append([float(v) for v in item["embedding"]])
            else:
                raise ValueError("Unsupported top-level payload list format")
        return vectors, None

    raise ValueError(f"Cannot parse embedding response of type {type(payload).__name__}")


def _report_results(results: list[EmbeddingResult]) -> str:
    if not results:
        return "No embedding vectors returned."

    lens = [r.vector_len for r in results]
    elapsed = [r.elapsed_ms for r in results]
    with_sparse = sum(1 for r in results if r.has_sparse)

    return (
        f"items: {len(results)}\n"
        f"dim:    {min(lens)}..{max(lens)} (sampled unique={len(set(lens))})\n"
        f"avg_ms: {mean(elapsed):.2f}\n"
        f"min_ms: {min(elapsed):.2f}\n"
        f"max_ms: {max(elapsed):.2f}\n"
        f"sparse: {with_sparse}/{len(results)} with sparse payload"
    )


def _select_input_text(item: tuple[str, str, str], include_context: bool) -> tuple[str, str]:
    chunk_id, text, context_header = item
    if include_context and context_header:
        return chunk_id, f"{context_header}\n\n{text}"
    return chunk_id, text


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.provider == "openai":
        endpoint = f"{args.url.rstrip('/')}/embeddings" if not args.url.endswith("/embeddings") else args.url
    else:
        endpoint = args.url

    if args.use_text_sample:
        items = _default_samples()
    else:
        chunk_file = args.chunk_file or _latest_file(args.chunks_dir)
        print(f"Reading chunks from: {chunk_file}")
        items = list(_iter_chunks(chunk_file, args.hard_limit))
        if args.limit:
            items = items[: args.limit]

    if not items:
        print("No input texts available for embedding test.")
        return 1

    payload_pairs = [(_select_input_text(item, args.include_context_header)) for item in items]
    texts = [text for _chunk_id, text in payload_pairs]
    source_ids = [chunk_id for chunk_id, _ in payload_pairs]

    headers = _build_headers(args.provider, args.api_key)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    request_payload = _build_payload_provider(texts, args.provider, args.model)

    results: list[EmbeddingResult] = []
    raw_response: list[dict[str, Any]] = []

    print("Connecting to:", endpoint)

    try:
        start = time.perf_counter()
        with httpx.Client(timeout=args.timeout) as client:
            if args.no_batch:
                for chunk_id, text in payload_pairs:
                    request_item = _build_payload_provider([text], args.provider, args.model)
                    response = client.post(endpoint, json=request_item, headers=headers)
                    response.raise_for_status()
                    vectors, sparse = _extract_vectors(response.json())
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    start = time.perf_counter()
                    if len(vectors) != 1:
                        raise ValueError("Single-request expected one vector")

                    vector = vectors[0]
                    result = EmbeddingResult(
                        source_id=chunk_id,
                        vector_len=len(vector),
                        elapsed_ms=elapsed_ms,
                        has_sparse=bool(sparse and sparse[0].get("indices") or sparse[0].get("values")),
                        sparse_len=len((sparse[0].get("indices") or [])) if sparse else None,
                    )
                    results.append(result)
                    raw_response.append({
                        "source_id": chunk_id,
                        "vector_len": result.vector_len,
                        "sparse_len": result.sparse_len,
                    })
            else:
                response = client.post(endpoint, json=request_payload, headers=headers)
                response.raise_for_status()
                vectors, sparse = _extract_vectors(response.json())
                batch_elapsed_ms = (time.perf_counter() - start) * 1000

                if len(vectors) != len(texts):
                    raise ValueError(
                        f"Expected {len(texts)} vectors, got {len(vectors)}"
                    )

                per_item_ms = batch_elapsed_ms / max(len(vectors), 1)
                for idx, chunk_id in enumerate(source_ids):
                    vector = vectors[idx]
                    sparse_item = sparse[idx] if sparse else None
                    result = EmbeddingResult(
                        source_id=chunk_id,
                        vector_len=len(vector),
                        elapsed_ms=per_item_ms,
                        has_sparse=bool(sparse_item and (sparse_item.get("indices") or sparse_item.get("values"))),
                        sparse_len=(
                            len(sparse_item.get("indices", []))
                            if sparse_item and isinstance(sparse_item.get("indices"), list)
                            else None
                        ),
                    )
                    results.append(result)
                    raw_response.append({
                        "source_id": chunk_id,
                        "vector_len": result.vector_len,
                        "sparse_len": result.sparse_len,
                    })

                print(f"Request elapsed: {batch_elapsed_ms:.2f} ms")

                out_resp = args.out_dir / f"embed_request_{run_id}.json"
                with out_resp.open("w", encoding="utf-8") as f:
                    json.dump(response.json(), f, ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as exc:
        print(f"HTTP error: {exc.response.status_code} - {exc.response.text}")
        return 1
    except httpx.RequestError as exc:
        print(f"Request error: {exc}")
        return 1
    except ValueError as exc:
        print(f"Parsing error: {exc}")
        return 1

    if not results:
        print("No vectors produced.")
        return 1

    print(_report_results(results))

    summary = {
        "provider": args.provider,
        "endpoint": endpoint,
        "count": len(results),
        "run_id": run_id,
        "dimensions": sorted(set(r.vector_len for r in results)),
        "with_sparse": sum(1 for r in results if r.has_sparse),
        "avg_ms": mean(r.elapsed_ms for r in results),
        "min_ms": min(r.elapsed_ms for r in results),
        "max_ms": max(r.elapsed_ms for r in results),
        "items": [
            {
                "source_id": r.source_id,
                "vector_len": r.vector_len,
                "elapsed_ms": r.elapsed_ms,
                "has_sparse": r.has_sparse,
                "sparse_len": r.sparse_len,
            }
            for r in results
        ],
    }

    out_summary = args.out_dir / f"embed_summary_{run_id}.json"
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved raw response: {args.out_dir / f'embed_request_{run_id}.json' if not args.no_batch else 'N/A (single mode)' }")
    print(f"Saved summary:     {out_summary}")

    # Persist concise diagnostics for quick diffability.
    out_plain = args.out_dir / f"embed_results_{run_id}.json"
    with out_plain.open("w", encoding="utf-8") as f:
        json.dump(raw_response, f, ensure_ascii=False, indent=2)
    print(f"Saved compact:    {out_plain}")

    print("Embedding smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

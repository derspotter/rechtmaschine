#!/usr/bin/env python
"""
Chunking v0: transform ingested JSONL entries into retrieval chunks.

Input:
- JSONL produced by ingest_v0.py

Output:
- JSONL chunks under rag/data/chunks/

Design goals:
- Deterministic chunk ids
- Simple, inspectable splitting (paragraph-first, then token-based)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import tiktoken


DEFAULT_INGESTED_DIR = Path(__file__).resolve().parent / "data" / "ingested"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "chunks"


def _latest_ingested(ingested_dir: Path) -> Path:
    candidates = sorted(
        ingested_dir.glob("ingested_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No ingested_*.jsonl found in {ingested_dir}")
    return candidates[0]


def _normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _split_paragraphs(text: str) -> list[str]:
    # Normalize newlines then split on blank lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    # Remove ultra-short junk paragraphs
    return [p for p in parts if len(p) >= 10]


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _make_context_header(metadata: dict[str, Any]) -> str:
    """
    Simple context header used later for embeddings.
    """
    case_folder = (metadata.get("case_folder") or "").strip()
    date_prefix = (metadata.get("date_prefix") or "").strip()
    court = (metadata.get("court_token") or "").strip()
    doc_token = (metadata.get("doc_token") or "").strip()
    filename = (metadata.get("filename") or "").strip()
    signals = metadata.get("signal_codes") or []

    doc_parts = [p for p in [case_folder, date_prefix, court, doc_token] if p]
    if not doc_parts:
        doc_parts = [filename] if filename else []

    sig = "+".join(signals) if isinstance(signals, list) and signals else ""
    lines: list[str] = []
    if doc_parts:
        lines.append(f"[{' | '.join(doc_parts)}]")
    if sig:
        lines.append(f"[Type: {sig}]")
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_rel_path: str
    source_sha256: str
    chunk_index: int
    text: str
    token_count: int
    context_header: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_rel_path": self.source_rel_path,
            "source_sha256": self.source_sha256,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "context_header": self.context_header,
            "metadata": self.metadata,
            "text": self.text,
        }


def _chunk_by_tokens(
    enc: tiktoken.Encoding,
    parts: list[str],
    chunk_tokens: int,
    overlap_tokens: int,
) -> Iterable[str]:
    """
    Paragraph-first then token windowing on the concatenated stream.
    """
    # Build one combined string with hard paragraph separators.
    combined = "\n\n".join(_normalize_ws(p) for p in parts if p.strip())
    if not combined:
        return []

    tokens = enc.encode(combined)
    if not tokens:
        return []

    start = 0
    n = len(tokens)
    step = max(1, chunk_tokens - overlap_tokens)
    while start < n:
        end = min(n, start + chunk_tokens)
        chunk_text = enc.decode(tokens[start:end]).strip()
        if chunk_text:
            yield chunk_text
        if end == n:
            break
        start += step


def _iter_ingested(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk ingested JSONL into chunks JSONL.")
    parser.add_argument(
        "--ingested",
        type=Path,
        default=None,
        help=f"Ingested JSONL. Default: latest in {DEFAULT_INGESTED_DIR}",
    )
    parser.add_argument("--ingested-dir", type=Path, default=DEFAULT_INGESTED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--min-chunk-tokens", type=int, default=80)
    parser.add_argument("--max-docs", type=int, default=0, help="0 = no limit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ingested_path = args.ingested or _latest_ingested(args.ingested_dir)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("cl100k_base")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"chunks_{run_id}.jsonl"

    print("=" * 80)
    print("RAG CHUNK V0")
    print("=" * 80)
    print(f"Ingested: {ingested_path}")
    print(f"Output:   {out_path}")
    print(f"Chunk:    {args.chunk_tokens} tokens, overlap {args.overlap_tokens} tokens")
    print()

    doc_count = 0
    chunk_count = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in _iter_ingested(ingested_path):
            if args.max_docs and doc_count >= args.max_docs:
                break
            doc_count += 1

            if not rec.get("ok", False):
                skipped += 1
                continue

            text = str(rec.get("text") or "").strip()
            if not text:
                skipped += 1
                continue

            source_rel_path = str(rec.get("source_rel_path") or "")
            source_sha = str(rec.get("sha256") or _sha256_hex(text))
            metadata = rec.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            # Keep provenance minimal but useful
            metadata = {
                **metadata,
                "source_rel_path": source_rel_path,
                "source_sha256": source_sha,
            }

            context_header = _make_context_header(metadata)
            parts = _split_paragraphs(text)
            if not parts:
                skipped += 1
                continue

            chunks_for_doc: list[str] = list(
                _chunk_by_tokens(
                    enc,
                    parts=parts,
                    chunk_tokens=args.chunk_tokens,
                    overlap_tokens=args.overlap_tokens,
                )
            )
            if not chunks_for_doc:
                skipped += 1
                continue

            for idx, chunk_text in enumerate(chunks_for_doc):
                tok = len(enc.encode(chunk_text))
                if tok < args.min_chunk_tokens:
                    continue
                chunk_body = chunk_text.strip()
                chunk_id = f"ch_{source_sha[:16]}_{idx:04d}"
                chunk = Chunk(
                    chunk_id=chunk_id,
                    source_rel_path=source_rel_path,
                    source_sha256=source_sha,
                    chunk_index=idx,
                    text=chunk_body,
                    token_count=tok,
                    context_header=context_header,
                    metadata=metadata,
                )
                out_f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                chunk_count += 1

            if doc_count % 10 == 0:
                print(f"Docs: {doc_count} | Chunks: {chunk_count}", end="\r")

    print()
    print(f"Docs processed: {doc_count}")
    print(f"Chunks written: {chunk_count}")
    print(f"Docs skipped:   {skipped}")
    print(f"Output:         {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


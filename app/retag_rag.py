"""Uniform retag pass over an existing RAG collection: scroll chunks from the
store, attach controlled-vocabulary tags, and re-upsert by the same chunk_id
(sending empty dense so the API re-embeds with the new context_header). No
re-extraction, re-download, or re-anonymization.

Subcommands:
  jurisprudence  tags come from the chunk's RechtsprechungEntry (DB), normalized.
  kanzlei        tags come from one desktop-Qwen call per document (added later).

Run inside the app container, e.g.:
  docker exec rechtmaschine-app python retag_rag.py jurisprudence --dry-run
  docker exec rechtmaschine-app python retag_rag.py jurisprudence
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Any, Optional

import httpx

from rag_vocabulary import (
    load_vocabulary, normalize_themen, normalize_country, normalize_normen,
    tag_line, facet_metadata,
)
import gemma_tagger
import qwen_tagger


def _rag_base() -> str:
    return os.environ["RAG_SERVICE_URL"].rstrip("/")


def _rag_headers() -> dict[str, str]:
    key = os.environ.get("RAG_SERVICE_API_KEY") or os.environ.get("RAG_API_KEY")
    return {"X-API-Key": key} if key else {}


def scroll_all(client: httpx.Client, collection: str, page: int = 256):
    cursor: Optional[str] = None
    while True:
        r = client.post(f"{_rag_base()}/v1/rag/chunks/scroll",
                        json={"collection": collection, "cursor": cursor, "limit": page},
                        headers=_rag_headers(), timeout=60)
        r.raise_for_status()
        data = r.json()
        for chunk in data["chunks"]:
            yield chunk
        cursor = data["next_cursor"]
        if cursor is None:
            break


def _base_header(context_header: Optional[str]) -> str:
    """Strip any previously-appended tag segments so reruns are idempotent."""
    if not context_header:
        return ""
    keep = []
    for seg in context_header.split(" | "):
        s = seg.strip()
        if s.startswith(("Schlagwörter:", "Herkunftsland:", "Normen:")):
            continue
        keep.append(s)
    return " | ".join([s for s in keep if s])


def build_retagged_chunk(chunk: dict[str, Any], themen, country, normen) -> dict[str, Any]:
    base = _base_header(chunk.get("context_header"))
    suffix = tag_line(themen, country, normen)
    header = " | ".join([s for s in (base, suffix) if s])
    metadata = {**(chunk.get("metadata") or {}), **facet_metadata(themen, country, normen)}
    return {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "context_header": header,
        "metadata": metadata,
        "provenance": chunk.get("provenance") or [],
        "dense": [],  # force re-embed with the new context_header
    }


def upsert_batch(client: httpx.Client, collection: str, batch: list[dict[str, Any]]) -> int:
    if not batch:
        return 0
    # The RAG upsert occasionally drops the connection mid-stream under load;
    # retry transient errors so a multi-hour pass isn't killed by one blip.
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        try:
            r = client.post(f"{_rag_base()}/v1/rag/chunks/upsert",
                            json={"collection": collection, "chunks": batch},
                            headers=_rag_headers(), timeout=180)
            r.raise_for_status()
            return int(r.json().get("upserted", 0))
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"[retag] upsert retry {attempt + 1}/4 after {type(exc).__name__}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"upsert failed after retries: {last_exc}")


def run_jurisprudence(args) -> int:
    from database import SessionLocal
    from models import RechtsprechungEntry

    vocab = load_vocabulary()
    db = SessionLocal()
    # Cache the entry's raw fields by id so we hit the DB once per entry, not
    # once per chunk.
    cache: dict[str, dict] = {}

    def raw_for_entry(entry_id: str) -> dict:
        if entry_id in cache:
            return cache[entry_id]
        e = db.query(RechtsprechungEntry).filter(RechtsprechungEntry.id == entry_id).first()
        cache[entry_id] = {
            "schlagworte": (e.schlagworte or []) if e else [],
            "country": (e.country if e else None),
            "normen": (e.normen or []) if e else [],
        }
        return cache[entry_id]

    upserted = scanned = skipped = 0
    batch: list[dict[str, Any]] = []
    try:
        with httpx.Client() as client:
            for chunk in scroll_all(client, "jurisprudence"):
                scanned += 1
                md = chunk.get("metadata") or {}
                entry_id = md.get("rechtsprechung_entry_id")
                if not entry_id:
                    skipped += 1
                    continue
                raw = raw_for_entry(str(entry_id))
                # Fold the chunk's own Gemini issue_tags (free-text) in with the
                # curated asyl.net schlagworte; the normalizer keeps only in-vocab.
                themen = normalize_themen(vocab, raw["schlagworte"] + (md.get("issue_tags") or []))
                country = normalize_country(vocab, raw["country"])
                normen = normalize_normen(vocab, raw["normen"])
                new_chunk = build_retagged_chunk(chunk, themen, country, normen)
                if args.dry_run:
                    if scanned <= 5:
                        print(f"  {chunk['chunk_id']}: {new_chunk['metadata'].get('schlagworte')} "
                              f"/ {new_chunk['metadata'].get('applicant_origin')}")
                    continue
                batch.append(new_chunk)
                if len(batch) >= 16:
                    upserted += upsert_batch(client, "jurisprudence", batch)
                    batch = []
            if not args.dry_run:
                upserted += upsert_batch(client, "jurisprudence", batch)
    finally:
        db.close()
    print(f"jurisprudence: scanned={scanned} skipped(no entry)={skipped} re-upserted={upserted} "
          f"{'(dry-run)' if args.dry_run else ''}")
    return 0


def _doc_id(chunk: dict[str, Any]) -> str:
    """Group key for a kanzlei document: the sha16 in chunk_id 'nc-<sha16>-<idx>'."""
    cid = chunk["chunk_id"]
    parts = cid.split("-")
    return parts[1] if len(parts) >= 3 and parts[0] == "nc" else cid


async def run_kanzlei(args) -> int:
    vocab = load_vocabulary()
    # Group all chunks by document first (scroll is cheap; tagging is the cost).
    docs: dict[str, list[dict[str, Any]]] = {}
    with httpx.Client() as client:
        for chunk in scroll_all(client, "kanzlei"):
            docs.setdefault(_doc_id(chunk), []).append(chunk)

    doc_ids = sorted(docs)
    if args.limit_docs:
        doc_ids = doc_ids[: args.limit_docs]

    def _already_tagged(chunks: list[dict[str, Any]]) -> bool:
        # A doc is considered done if any of its chunks already carries
        # schlagworte — lets a re-run resume after an interruption (e.g. a
        # debian reboot) without re-spending Qwen calls. --force re-tags all.
        return any((c.get("metadata") or {}).get("schlagworte") for c in chunks)

    tagger = gemma_tagger if getattr(args, "tagger", "gemma") == "gemma" else qwen_tagger
    print(f"kanzlei: {len(docs)} documents, tagging {len(doc_ids)} with {tagger.__name__}")

    tagged_docs = upserted = resumed = 0
    with httpx.Client() as client:
        for n, did in enumerate(doc_ids, 1):
            chunks = sorted(docs[did], key=lambda c: (c.get("metadata") or {}).get("chunk_index", 0))
            if not args.dry_run and not args.force and _already_tagged(chunks):
                resumed += 1
                continue
            text = "\n\n".join(c["text"] for c in chunks)
            facets = await tagger.tag_document(text, vocab)
            themen, country, normen = facets["schlagworte"], facets["herkunftsland"], facets["normen"]
            if args.dry_run:
                if n <= 5:
                    print(f"  {did}: {themen} / {country} / {normen}")
                continue
            batch = [build_retagged_chunk(c, themen, country, normen) for c in chunks]
            for start in range(0, len(batch), 16):
                upserted += upsert_batch(client, "kanzlei", batch[start:start + 16])
            tagged_docs += 1
            if n % 50 == 0:
                print(f"  ... {n}/{len(doc_ids)} docs, {upserted} chunks re-upserted")
    print(f"kanzlei: tagged_docs={tagged_docs} resumed(skipped already-tagged)={resumed} "
          f"re-upserted={upserted} {'(dry-run)' if args.dry_run else ''}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    jp = sub.add_parser("jurisprudence")
    jp.add_argument("--dry-run", action="store_true")
    kp = sub.add_parser("kanzlei")
    kp.add_argument("--dry-run", action="store_true")
    kp.add_argument("--limit-docs", type=int, default=0, help="Tag at most N docs (0 = all).")
    kp.add_argument("--tagger", choices=["gemma", "qwen"], default="gemma",
                    help="Tagging backend: gemma (debian:8011) or qwen (desktop).")
    kp.add_argument("--force", action="store_true",
                    help="Re-tag every doc, even ones already carrying schlagworte.")
    args = ap.parse_args()
    if args.cmd == "jurisprudence":
        return run_jurisprudence(args)
    if args.cmd == "kanzlei":
        return asyncio.run(run_kanzlei(args))  # defined in a later task
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

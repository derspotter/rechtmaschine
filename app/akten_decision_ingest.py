"""Bulk-ingest collected court decisions from the firm's Akten into the
global Rechtsprechung store (source_type="jlawyer_akten"/"nextcloud_akten").

The colleagues collect reference case law in two places the store never saw:
j-lawyer Akten (decision-named documents, split FREMD/EIGEN by the 2026-07
sweep) and Nextcloud (999-foo Sammelordner, 00 AT/lib, anlage_k* files).
This script reads a prepared manifest (one JSON object per line) of locally
staged PDFs and runs the shared jurisprudence pipeline: local-Qwen
extract_tags -> persist_entry -> RAG chunk upsert, mirroring
wiki_media_ingest.py.

Manifest line:
    {"path": "/app/downloaded_sources/akten/<key>.pdf",
     "source_type": "jlawyer_akten" | "nextcloud_akten",
     "source_ref": "jlawyer:<documentId>" | "nextcloud:<sha1(path)[:16]>",
     "origin": "<Akte 097/26 | /kanzlei/...>",
     "fallback_az": "10 K 2107/20.A"}   # first Az the deterministic sweep
                                        # regex found in the text, or ""

Az rule (same as wiki ingest): the LLM-extracted Az must be found
deterministically in the PDF text. If not, fall back to the sweep Az (which
by construction came from the text). No verifiable Az -> entry INAKTIV.

Run inside the app container:
    docker exec rechtmaschine-app python /app/akten_decision_ingest.py \
        --manifest /app/downloaded_sources/akten/manifest.jsonl \
        [--dry-run] [--limit N]

Idempotent: pre-ingest dedup on source_ref, then content sha256, then
cross-source Az (find_active_by_az). SHORT scans are reported for a later
OCR round, never silently dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys

import fitz  # PyMuPDF

from cited_ingest import find_active_by_az
from database import SessionLocal
from jurisprudence_ingest import (
    chunk_text,
    extract_model_label,
    extract_tags,
    persist_entry,
    upsert,
    _norm_az,
)
from models import RechtsprechungEntry
from rag_vocabulary import (
    facet_metadata,
    load_vocabulary,
    normalize_country,
    normalize_normen,
    normalize_themen,
    tag_line,
)

MIN_TEXT_CHARS = 400


def local_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    try:
        return "\n\n".join((page.get_text() or "").strip() for page in doc)
    finally:
        doc.close()


def repair_chunks(args) -> int:
    """Rebuild RAG chunks for entries whose upsert failed in an earlier run
    (entry persisted, debian/RAG unreachable). No LLM calls: text comes from
    the staged PDF, metadata from the stored entry. Idempotent (chunk_id
    upsert)."""
    items = {}
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                it = json.loads(line)
                items[it["source_ref"]] = it
    db = SessionLocal()
    try:
        vocab = load_vocabulary()
        entries = db.query(RechtsprechungEntry).filter(
            RechtsprechungEntry.source_type.in_(("jlawyer_akten", "nextcloud_akten"))
        ).all()
        repaired = 0
        for entry in entries:
            it = items.get(entry.source_ref)
            if it is None:
                print(f"  SKIP  {entry.source_ref} — nicht im Manifest")
                continue
            text = local_pdf_text(it["path"])
            full_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if full_sha != entry.content_sha256:
                print(f"  SKIP  {entry.source_ref} — content_sha weicht ab")
                continue
            sha16 = full_sha[:16]
            _themen = normalize_themen(vocab, entry.tags or [])
            _country = normalize_country(vocab, entry.country)
            _normen = normalize_normen(vocab, [])
            header_bits = ["Rechtsprechung", entry.court or "", entry.court_level or "",
                           str(entry.decision_date or ""), entry.country or "",
                           tag_line(_themen, _country, _normen)]
            context_header = " | ".join(b for b in header_bits if b)
            metadata = {
                "source_system": entry.source_type,
                "rechtsprechung_entry_id": str(entry.id),
                "akten_origin": it.get("origin", ""),
                "country": entry.country,
                "court": entry.court,
                "court_level": entry.court_level,
                "outcome": entry.outcome,
                "decision_date": str(entry.decision_date or ""),
                "aktenzeichen": entry.aktenzeichen,
                "issue_tags": entry.tags or [],
                **facet_metadata(_themen, _country, _normen),
                "instance_weight": entry.instance_weight,
                "language": "de",
            }
            provenance = [f"{entry.source_type}:{entry.source_ref}",
                          f"origin:{it.get('origin', '')}",
                          f"entry:{entry.id}", f"sha256:{sha16}"]
            payload = [
                {
                    "chunk_id": f"juris-{sha16}-{idx:03d}",
                    "text": chunk,
                    "context_header": context_header,
                    "metadata": {**metadata, "chunk_index": idx},
                    "provenance": provenance,
                }
                for idx, chunk in enumerate(chunk_text(text))
            ]
            upserted = upsert(payload, args.collection)
            repaired += 1
            print(f"  REPAIR {entry.source_ref} — {upserted} chunks "
                  f"({entry.court} {entry.aktenzeichen})", flush=True)
        print(f"\nrepaired {repaired}/{len(entries)} entries")
        return 0
    finally:
        db.close()


async def ingest(args) -> int:
    items = []
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    db = SessionLocal()
    try:
        new_items = []
        dup_ref = 0
        for it in items:
            if db.query(RechtsprechungEntry.id).filter(
                RechtsprechungEntry.source_ref == it["source_ref"]
            ).first():
                dup_ref += 1
                continue
            new_items.append(it)
        if args.limit:
            new_items = new_items[: args.limit]
        print(f"{len(items)} in manifest, {dup_ref} already stored, "
              f"{len(new_items)} to ingest\n", flush=True)

        vocab = load_vocabulary()
        ingested = dup_content = short = failed = inactive = chunk_total = 0
        rag_deferred = 0
        short_refs: list[str] = []
        for it in new_items:
            ref, origin = it["source_ref"], it.get("origin", "")
            try:
                text = local_pdf_text(it["path"])
                if len(text) < MIN_TEXT_CHARS:
                    short += 1
                    short_refs.append(f"{ref} ({origin})")
                    print(f"  SHORT {ref} — {len(text)} chars, OCR-Runde [{origin}]",
                          flush=True)
                    continue
                full_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                sha16 = full_sha[:16]
                if db.query(RechtsprechungEntry.id).filter(
                    RechtsprechungEntry.content_sha256 == full_sha
                ).first():
                    dup_content += 1
                    print(f"  DUP   {ref} (content) [{origin}]", flush=True)
                    continue

                tags = await extract_tags(text)
                if tags is None:
                    failed += 1
                    print(f"  FAIL  {ref} — LLM extraction unavailable [{origin}]",
                          flush=True)
                    continue

                warnings = list(tags.warnings or [])
                deactivate = False
                az_ok = bool(tags.aktenzeichen) and _norm_az(tags.aktenzeichen) in _norm_az(text)
                if not az_ok:
                    fb = (it.get("fallback_az") or "").strip()
                    if fb and _norm_az(fb) in _norm_az(text):
                        tags.aktenzeichen = fb
                        warnings.append("Az aus deterministischem Sweep-Regex "
                                        "(LLM-Extraktion ohne verifizierbares Az)")
                    else:
                        deactivate = True
                        warnings.append("Kein verifizierbares Az im PDF-Text - "
                                        "inaktiv, manuell prüfen")
                tags.warnings = warnings

                dup_entry = find_active_by_az(db, tags.aktenzeichen)
                if dup_entry is not None:
                    dup_content += 1
                    print(f"  DUP   {ref} (Az {tags.aktenzeichen} bereits aktiv als "
                          f"{dup_entry.source_type}:{dup_entry.id}) [{origin}]",
                          flush=True)
                    continue

                _themen = normalize_themen(vocab, tags.tags or [])
                _country = normalize_country(vocab, tags.country)
                _normen = normalize_normen(vocab, [])
                header_bits = ["Rechtsprechung", tags.court or "", tags.court_level or "",
                               tags.decision_date or "", tags.country or "",
                               tag_line(_themen, _country, _normen)]
                context_header = " | ".join(b for b in header_bits if b)

                if args.dry_run:
                    ingested += 1
                    print(f"  OK*   {tags.court} {tags.aktenzeichen} ({tags.country}, "
                          f"{tags.decision_date}) — {len(text)}c, "
                          f"{len(chunk_text(text))} chunks [dry-run, {origin}]",
                          flush=True)
                    continue

                entry = persist_entry(
                    db, tags, source_type=it["source_type"], source_url=it["path"],
                    source_ref=ref, content_sha256=full_sha,
                    model_label=f"{it['source_type']}+{extract_model_label()}",
                )
                metadata = {
                    "source_system": it["source_type"],
                    "rechtsprechung_entry_id": str(entry.id),
                    "akten_origin": origin,
                    "country": tags.country,
                    "court": tags.court,
                    "court_level": tags.court_level,
                    "outcome": tags.outcome,
                    "decision_date": tags.decision_date,
                    "aktenzeichen": tags.aktenzeichen,
                    "issue_tags": tags.tags or [],
                    **facet_metadata(_themen, _country, _normen),
                    "instance_weight": entry.instance_weight,
                    "language": "de",
                }
                provenance = [f"{it['source_type']}:{ref}", f"origin:{origin}",
                              f"entry:{entry.id}", f"sha256:{sha16}"]
                payload = [
                    {
                        "chunk_id": f"juris-{sha16}-{idx:03d}",
                        "text": chunk,
                        "context_header": context_header,
                        "metadata": {**metadata, "chunk_index": idx},
                        "provenance": provenance,
                    }
                    for idx, chunk in enumerate(chunk_text(text))
                ]
                if deactivate:
                    entry.is_active = False
                    db.commit()
                    inactive += 1
                if args.no_rag:
                    upserted = 0
                    rag_deferred += 1
                else:
                    try:
                        upserted = upsert(payload, args.collection)
                    except Exception:
                        # RAG service unreachable (debian asleep): without
                        # chunks the entry would be a dedup-blocking orphan —
                        # roll it back so a rerun ingests the document cleanly.
                        db.delete(entry)
                        db.commit()
                        if deactivate:
                            inactive -= 1
                        raise
                chunk_total += upserted
                ingested += 1
                flag = " INAKTIV" if deactivate else ""
                print(f"  OK{flag} {tags.court} {tags.aktenzeichen} ({tags.country}, "
                      f"{tags.decision_date}, {tags.outcome}, w{entry.instance_weight}) "
                      f"— {len(text)}c -> {upserted} chunks [{origin}]", flush=True)
                for w in warnings:
                    print(f"  WARN  {ref}: {w}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAIL  {ref} — {exc} [{origin}]", flush=True)

        verb = "would ingest" if args.dry_run else "ingested"
        print(f"\n{verb} {ingested} ({inactive} inaktiv), dup-ref {dup_ref}, "
              f"dup-content {dup_content}, short {short}, failed {failed}"
              + ("" if args.dry_run else f"; {chunk_total} chunks into '{args.collection}'."))
        if rag_deferred:
            print(f"RAG DEFERRED: {rag_deferred} Einträge ohne Chunks (--no-rag) — "
                  f"nachziehen mit --repair, sobald der RAG-Service erreichbar ist.")
        if short_refs:
            print("SHORT (Scan, vor Ingest OCRen): " + ", ".join(short_refs))
        return 0 if failed == 0 else 1
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--collection", default="jurisprudence")
    parser.add_argument("--limit", type=int, help="max new PDFs to ingest this run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repair", action="store_true",
                        help="nur fehlende RAG-Chunks für bestehende Einträge nachziehen")
    parser.add_argument("--no-rag", action="store_true",
                        help="Chunk-Upsert überspringen (RAG down) — später --repair")
    args = parser.parse_args()
    if args.repair:
        return repair_chunks(args)
    return asyncio.run(ingest(args))


if __name__ == "__main__":
    sys.exit(main())

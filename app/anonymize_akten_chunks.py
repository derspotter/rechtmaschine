"""Nachanonymisierung der Akten-Rechtsprechung im RAG (Jay-Go 27.07.2026).

Entscheidungen aus eigenen Verfahren (source_type jlawyer_akten /
nextcloud_akten) tragen Mandantennamen im Rubrum — die RAG-Chunks und die
Entry-Summaries damit auch. Dieser Pass zieht die produktive
Anonymisierungs-Pipeline (Entity-Extraktion Desktop-LLM + lokale
Regex-Ersetzung, endpoints/anonymization.anonymize_document_text) über jeden
Eintrag: Chunks werden aus dem anonymisierten Text neu gebaut und unter
denselben chunk_ids (juris-<sha16 des ORIGINALtexts>-NNN) re-upserted,
überzählige Alt-Chunks via /v1/rag/chunks/delete entfernt, und die
Namens-Entities werden auch aus summary/key_facts/key_holdings/
argument_patterns der Entries getilgt. Die Quell-PDFs bleiben unverändert
(verify-source braucht den echten Volltext).

Idempotent über einen State-File (source_ref -> done); neue Einträge einer
späteren OCR-Runde einfach durch erneuten Lauf nachziehen.

    docker exec rechtmaschine-app python /app/anonymize_akten_chunks.py \
        --manifest /app/downloaded_sources/akten/manifest.jsonl [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys

import fitz  # PyMuPDF

from database import SessionLocal
from endpoints.anonymization import (
    anonymize_document_text,
    resolve_anonymization_engine,
)
from akten_decision_ingest import upsert_retry
from jurisprudence_ingest import chunk_text, upsert
from models import RechtsprechungEntry
from rag_vocabulary import (
    facet_metadata,
    load_vocabulary,
    normalize_country,
    normalize_normen,
    normalize_themen,
    tag_line,
)

STATE_PATH = "/app/downloaded_sources/akten/anon_state.json"


def local_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    try:
        return "\n\n".join((page.get_text() or "").strip() for page in doc)
    finally:
        doc.close()


def delete_chunk_range(sha16: str, start: int, end: int, collection: str) -> None:
    """Delete juris-<sha16>-NNN for NNN in [start, end) via the RAG API."""
    import httpx

    base = os.getenv("RAG_SERVICE_URL", "").strip().rstrip("/")
    key = os.getenv("RAG_API_KEY") or os.getenv("RAG_SERVICE_API_KEY")
    ids = [f"juris-{sha16}-{i:03d}" for i in range(start, end)]
    if not ids:
        return
    with httpx.Client(timeout=60) as client:
        r = client.post(f"{base}/v1/rag/chunks/delete",
                        headers={"X-API-Key": key} if key else {},
                        json={"chunk_ids": ids, "collection": collection})
        r.raise_for_status()


def scrub(value, replacements: list[tuple[str, str]]):
    """Replace every entity string in str/list/dict values."""
    if isinstance(value, str):
        for needle, repl in replacements:
            if needle and needle in value:
                value = value.replace(needle, repl)
        return value
    if isinstance(value, list):
        return [scrub(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v, replacements) for k, v in value.items()}
    return value


async def run(args) -> int:
    items = {}
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                it = json.loads(line)
                items[it["source_ref"]] = it
    state = {}
    if os.path.exists(STATE_PATH):
        state = json.load(open(STATE_PATH))
    engine = resolve_anonymization_engine(None)
    db = SessionLocal()
    try:
        vocab = load_vocabulary()
        entries = db.query(RechtsprechungEntry).filter(
            RechtsprechungEntry.source_type.in_(("jlawyer_akten", "nextcloud_akten")),
            RechtsprechungEntry.is_active.is_(True),
        ).order_by(RechtsprechungEntry.created_at).all()
        todo = [e for e in entries if state.get(e.source_ref) != "done"]
        if args.limit:
            todo = todo[: args.limit]
        print(f"{len(entries)} aktive Akten-Einträge, {len(todo)} zu anonymisieren "
              f"(engine={engine})\n", flush=True)
        done = failed = 0
        for entry in todo:
            ref = entry.source_ref
            try:
                it = items.get(ref)
                path = (it or {}).get("path") or entry.source_url
                text = local_pdf_text(path)
                full_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if full_sha != entry.content_sha256:
                    print(f"  SKIP  {ref} — content_sha weicht ab", flush=True)
                    state[ref] = "sha_mismatch"
                    continue
                sha16 = full_sha[:16]
                result = await anonymize_document_text(
                    text, document_type="Rechtsprechung", engine=engine)
                if result is None or not (result.anonymized_text or "").strip():
                    failed += 1
                    print(f"  FAIL  {ref} — Anonymisierung lieferte nichts", flush=True)
                    continue
                anon_text = result.anonymized_text

                old_n = len(chunk_text(text))
                new_chunks = chunk_text(anon_text)

                _themen = normalize_themen(vocab, entry.tags or [])
                _country = normalize_country(vocab, entry.country)
                _normen = normalize_normen(vocab, [])
                header_bits = ["Rechtsprechung", entry.court or "",
                               entry.court_level or "", str(entry.decision_date or ""),
                               entry.country or "", tag_line(_themen, _country, _normen)]
                context_header = " | ".join(b for b in header_bits if b)
                metadata = {
                    "source_system": entry.source_type,
                    "rechtsprechung_entry_id": str(entry.id),
                    "akten_origin": (it or {}).get("origin", ""),
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
                    "anonymized": True,
                }
                provenance = [f"{entry.source_type}:{ref}",
                              f"origin:{(it or {}).get('origin', '')}",
                              f"entry:{entry.id}", f"sha256:{sha16}", "anonymized"]
                payload = [
                    {
                        "chunk_id": f"juris-{sha16}-{idx:03d}",
                        "text": chunk,
                        "context_header": context_header,
                        "metadata": {**metadata, "chunk_index": idx},
                        "provenance": provenance,
                    }
                    for idx, chunk in enumerate(new_chunks)
                ]
                upsert_retry(payload, args.collection)
                if old_n > len(new_chunks):
                    delete_chunk_range(sha16, len(new_chunks), old_n, args.collection)

                # Entry-Felder von denselben Entities befreien.
                repl = [(n, "[PERSON]") for n in (result.plaintiff_names or [])]
                repl += [(d, "[DATUM]") for d in (result.birth_dates or [])]
                repl += [(a, "[ADRESSE]") for a in (result.addresses or [])]
                repl = [(n, r) for n, r in repl if n and len(n) >= 4]
                entry.summary = scrub(entry.summary, repl)
                entry.key_facts = scrub(entry.key_facts or [], repl)
                entry.key_holdings = scrub(entry.key_holdings or [], repl)
                entry.argument_patterns = scrub(entry.argument_patterns or [], repl)
                db.commit()

                state[ref] = "done"
                done += 1
                print(f"  OK    {ref} — {len(new_chunks)} Chunks anonymisiert, "
                      f"{len(result.plaintiff_names or [])} Namen "
                      f"({entry.court} {entry.aktenzeichen})", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                # Exception-TYP mitdrucken, s. akten_decision_ingest.py:
                # leeres str() bei httpx-Timeouts macht Logs unauswertbar.
                print(f"  FAIL  {ref} — {type(exc).__name__}: {exc}", flush=True)
            finally:
                json.dump(state, open(STATE_PATH, "w"))
        print(f"\nanonymisiert {done}, failed {failed}, "
              f"state: {STATE_PATH}")
        return 0 if failed == 0 else 1
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="/app/downloaded_sources/akten/manifest.jsonl")
    parser.add_argument("--collection", default="jurisprudence")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

"""Generic ingest of individually cited court decisions into the global
Rechtsprechung store (source_type="cited" by default).

jurisprudence_ingest.py covers asyl.net, wiki_media_ingest.py covers the
doktrin wiki's media PDFs — but decisions that get CITED in drafts (and are
then verified against bverwg.de, EUR-Lex, a Landesportal, …) had no path
into the store. This script closes that gap: give it the fulltext PDF
(URL or file already inside the container) and it runs the shared pipeline
(extract_tags: Qwen -> Gemini fallback, persist_entry, RAG chunk upsert)
with the SAME metadata shape as the asylnet and wiki flows.

Az rule (mirrors wiki_media_ingest): the Aktenzeichen comes from the PDF
text via LLM extraction and is cross-checked deterministically against the
PDF. An operator-supplied --az/--court/--date is a CLAIM from the citation,
not proof: it is used as fallback metadata only if it is deterministically
found in the PDF text, otherwise the entry is stored inactive with a
warning ("vor Zitierung Volltext besorgen").

Run inside the app container:
    docker exec rechtmaschine-app python /app/cited_ingest.py \
        https://www.bverwg.de/entscheidungen/pdf/270410U10C5.09.0.pdf \
        [more URLs or /app/... paths] [--dry-run] [--source-type cited]

Single-source cross-check (order-independent flags apply to ALL sources,
so pass them only with exactly one source):
    docker exec rechtmaschine-app python /app/cited_ingest.py <url> \
        --az "10 C 5.09" --court BVerwG --date 27.04.2010

Idempotent: dedup on content sha256 AND on normalized Aktenzeichen of
active entries (the same decision fetched from another portal has a
different byte stream but the same Az).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

from database import SessionLocal
from jurisprudence_ingest import (
    chunk_text,
    download_pdf_text,
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

MIN_TEXT_CHARS = 400  # below this the PDF is a scan without text layer


def load_source_text(source: str) -> str:
    """Fulltext from a URL or a file path inside the container."""
    if source.startswith(("http://", "https://")):
        return download_pdf_text(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"not a file inside the container: {source}")
    import fitz  # same extraction as download_pdf_text

    doc = fitz.open(str(path))
    try:
        return "\n\n".join((page.get_text() or "").strip() for page in doc)
    finally:
        doc.close()


def find_active_by_az(db, aktenzeichen: str):
    want = _norm_az(aktenzeichen or "")
    if not want:
        return None
    for entry in db.query(RechtsprechungEntry).filter(
        RechtsprechungEntry.is_active.is_(True),
        RechtsprechungEntry.aktenzeichen.isnot(None),
    ).all():
        if _norm_az(entry.aktenzeichen) == want:
            return entry
    return None


async def main_async(args) -> int:
    if (args.az or args.court or args.date) and len(args.sources) != 1:
        print("--az/--court/--date describe ONE citation; pass exactly one source with them.",
              file=sys.stderr)
        return 2

    db = SessionLocal()
    try:
        vocab = load_vocabulary()
        ingested = dup = short = failed = chunk_total = 0
        for source in args.sources:
            try:
                text = load_source_text(source)
            except Exception as exc:  # noqa: BLE001 - per-source, reported
                failed += 1
                print(f"  FAIL  {source} — {exc}")
                continue
            if len(text) < MIN_TEXT_CHARS:
                short += 1
                print(f"  SHORT {source} — only {len(text)} chars (scan without text layer?)")
                continue

            full_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            sha16 = full_sha[:16]
            if db.query(RechtsprechungEntry.id).filter(
                RechtsprechungEntry.content_sha256 == full_sha
            ).first():
                dup += 1
                print(f"  DUP   {source} (content)")
                continue

            tags = await extract_tags(text)
            if tags is None:
                failed += 1
                print(f"  FAIL  {source} — LLM extraction unavailable")
                continue

            warnings = list(tags.warnings or [])
            deactivate = False
            az_ok = bool(tags.aktenzeichen) and _norm_az(tags.aktenzeichen) in _norm_az(text)
            if not az_ok:
                if args.az:
                    tags.aktenzeichen = args.az
                    if args.court:
                        tags.court = args.court
                        tags.court_level = None
                    if args.date:
                        tags.decision_date = args.date
                    if _norm_az(args.az) in _norm_az(text):
                        warnings.append("Az aus Zitat übernommen, deterministisch im "
                                        "PDF bestätigt (kein Rubrum im Auszug)")
                    else:
                        deactivate = True
                        warnings.append("Metadaten aus dem Zitat; Az im PDF-Text nicht "
                                        "auffindbar - vor Zitierung Volltext besorgen")
                else:
                    deactivate = True
                    warnings.append("Kein verifizierbares Az im PDF und kein --az "
                                    "übergeben - inaktiv, manuell prüfen")
            elif args.az and _norm_az(args.az) != _norm_az(tags.aktenzeichen):
                deactivate = True
                warnings.append(f"Az-Konflikt: Zitat '{args.az}' vs PDF "
                                f"'{tags.aktenzeichen}' - inaktiv, manuell prüfen")
            tags.warnings = warnings

            dup_entry = find_active_by_az(db, tags.aktenzeichen)
            if dup_entry is not None:
                dup += 1
                print(f"  DUP   {source} (Az {tags.aktenzeichen} bereits aktiv als "
                      f"{dup_entry.source_type}:{dup_entry.id})")
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
                      f"{len(chunk_text(text))} chunks [dry-run]")
                continue

            source_url = source if source.startswith(("http://", "https://")) else ""
            entry = persist_entry(
                db, tags, source_type=args.source_type, source_url=source_url,
                source_ref=f"cited:{sha16}", content_sha256=full_sha,
                model_label=f"cited+{extract_model_label()}",
            )
            metadata = {
                "source_system": args.source_type,
                "rechtsprechung_entry_id": str(entry.id),
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
            provenance = [f"{args.source_type}:{source_url or source}",
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
            upserted = upsert(payload, args.collection)
            chunk_total += upserted
            ingested += 1
            flag = " INAKTIV" if deactivate else ""
            print(f"  OK{flag} {tags.court} {tags.aktenzeichen} ({tags.country}, "
                  f"{tags.decision_date}, {tags.outcome}, w{entry.instance_weight}) "
                  f"— {len(text)}c -> {upserted} chunks")

        print(f"Done: {ingested} ingested, {dup} duplicate, {short} short, {failed} failed"
              + ("" if args.dry_run else f"; {chunk_total} chunks into '{args.collection}'."))
        return 0 if not failed else 1
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+",
                        help="PDF URLs or file paths inside the container")
    parser.add_argument("--source-type", default="cited",
                        help="RechtsprechungEntry.source_type (default: cited)")
    parser.add_argument("--az", help="Aktenzeichen laut Zitat (nur mit genau einer Quelle)")
    parser.add_argument("--court", help="Gericht laut Zitat")
    parser.add_argument("--date", help="Entscheidungsdatum laut Zitat (TT.MM.JJJJ)")
    parser.add_argument("--collection", default="jurisprudence")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Ingest court-decision PDFs from wiki.aufentha.lt media into the global
Rechtsprechung store (source_type="kanzlei_wiki").

doktrin_sync.py syncs page TEXT only, and the markup cleaning strips media
links — the decision PDFs in the wiki's media manager (often UNPUBLISHED
decisions from the firm's own practice) were invisible to Rechtmaschine.
This script closes that gap: it crawls the raw markup of all known doktrin
pages, extracts ``{{...pdf}}`` media links, downloads new PDFs and runs the
shared jurisprudence pipeline (extract_tags: Qwen -> Gemini fallback,
persist_entry, RAG chunk upsert) — mirroring the asylnet flow in
jurisprudence_ingest.py.

Az rule (2026-07-21, case 097/26): the Aktenzeichen comes from the PDF text
via LLM extraction and is cross-checked deterministically against the PDF.
The wiki page heading is NEVER the source — headings drop Länder-Ortskürzel
(wiki "3 L 4061/25.A" vs amtlich "3 L 4061/25.F.A").

Run inside the app container:
    docker exec rechtmaschine-app python /app/wiki_media_ingest.py \
        [--dry-run] [--limit N] [--pages id1,id2] [--delay 0.1]

Nightly: chained after doktrin_sync in scripts/doktrin_sync.sh.
Idempotent: pre-download dedup on source_ref ("wiki:<media_id>"), then
post-download dedup on content sha256 (same decision under another name or
already ingested from asyl.net). Scanned PDFs without a text layer are
reported as SHORT and skipped — no silent drops, the summary names them so
they can be OCR'd and re-ingested deliberately.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import sys
import time
from typing import Optional

from cited_ingest import find_active_by_az
from database import SessionLocal
from doktrin_sync import DEFAULT_BASE_URL, _normalize_base_url, _page_url, fetch_page
from jurisprudence_ingest import (
    chunk_text,
    download_pdf_text,
    extract_model_label,
    extract_tags,
    persist_entry,
    upsert,
    _norm_az,
)
from models import DoktrinPage, RechtsprechungEntry
from rag_vocabulary import (
    facet_metadata,
    load_vocabulary,
    normalize_country,
    normalize_normen,
    normalize_themen,
    tag_line,
)

#: DokuWiki media embeds/links: {{:file.pdf|Label}}, {{ns:sub:file.pdf?size}}, …
_MEDIA_PDF_RE = re.compile(r"\{\{\s*:?([^|}\s?]+?\.pdf)\s*(?:[?|][^}]*)?\}\}", re.IGNORECASE)

#: Marcel's decision headings: "=== - VG Minden, Beschluss vom 19.11.2025, 12 L 1178/25.A ==="
_HEADING_RE = re.compile(
    r"^=+\s*-?\s*(?P<court>[^,=]+?),\s*(?:Beschluss|Urteil|Gerichtsbescheid)\s+vom\s+"
    r"(?P<date>\d{1,2}\.\d{1,2}\.\d{4})\s*,\s*(?P<az>[^=]+?)\s*=+\s*$",
    re.MULTILINE,
)

MIN_TEXT_CHARS = 400  # below this the PDF is a scan without text layer


def media_url(base_url: str, media_id: str) -> str:
    """DokuWiki media id (``ns:sub:file.pdf``) -> fetch URL under /_media/.

    Namespace colons stay colons: this DokuWiki serves
    ``/_media/0:file.pdf`` (200) but 404s on ``/_media/0/file.pdf``."""
    return f"{base_url}/_media/{media_id}"


def nearest_heading(markup: str, pos: int) -> Optional[dict[str, str]]:
    """Last decision heading above pos — Marcel's curated Gericht/Datum/Az.
    Untrusted (headings drop Ortskürzel, see 3 L 4061/25.F.A), used only as
    fallback metadata when the PDF itself has no verifiable Rubrum."""
    best = None
    for m in _HEADING_RE.finditer(markup, 0, pos):
        best = m
    if not best:
        return None
    d, mth, y = best.group("date").split(".")
    return {
        "court": best.group("court").strip(),
        "date": f"{y}-{int(mth):02d}-{int(d):02d}",
        "az": best.group("az").strip().rstrip("-–— ").strip(),
    }


def collect_media_refs(base_url: str, page_ids: list[str], delay: float,
                       timeout: float = 30.0) -> dict[str, tuple[str, Optional[dict]]]:
    """Crawl raw markup, return {media_id: (first page_id, heading or None)}."""
    refs: dict[str, tuple[str, Optional[dict]]] = {}
    fetched = failed = 0
    for pid in page_ids:
        try:
            page = fetch_page(base_url, pid, timeout)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FETCH-FAIL {pid} — {exc}")
            continue
        fetched += 1
        for m in _MEDIA_PDF_RE.finditer(page.text or ""):
            media_id = m.group(1).strip().lstrip(":").lower()
            refs.setdefault(media_id, (pid, nearest_heading(page.text, m.start())))
        if delay:
            time.sleep(delay)
    print(f"crawled {fetched} pages ({failed} fetch failures), "
          f"{len(refs)} distinct media PDFs referenced\n")
    return refs


async def ingest(args) -> int:
    base_url = _normalize_base_url(args.base_url)
    db = SessionLocal()
    try:
        q = db.query(DoktrinPage.page_id)
        if args.pages:
            wanted = {p.strip() for p in args.pages.split(",") if p.strip()}
            q = q.filter(DoktrinPage.page_id.in_(wanted))
        page_ids = [row[0] for row in q.order_by(DoktrinPage.page_id).all()]
        if not page_ids:
            print("no doktrin pages found (run doktrin_sync.py first)")
            return 1

        refs = collect_media_refs(base_url, page_ids, args.delay)

        # Pre-download dedup by media id (mirrors asylnet's M-number dedup).
        new_refs = []
        dup_ref = 0
        for media_id, (page_id, heading) in sorted(refs.items()):
            if db.query(RechtsprechungEntry.id).filter(
                RechtsprechungEntry.source_ref == f"wiki:{media_id}"
            ).first():
                dup_ref += 1
                continue
            new_refs.append((media_id, page_id, heading))
        if args.limit:
            new_refs = new_refs[: args.limit]
        print(f"{dup_ref} already stored, {len(new_refs)} new to ingest\n")

        vocab = load_vocabulary()
        ingested = dup_content = short = failed = chunk_total = 0
        short_ids: list[str] = []
        for media_id, page_id, heading in new_refs:
            url = media_url(base_url, media_id)
            try:
                text = download_pdf_text(url)
                if len(text) < MIN_TEXT_CHARS:
                    print(f"  SHORT {media_id} — only {len(text)} chars (scan without text layer?)")
                    short += 1
                    short_ids.append(media_id)
                    continue
                full_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                sha16 = full_sha[:16]
                if db.query(RechtsprechungEntry.id).filter(
                    RechtsprechungEntry.content_sha256 == full_sha
                ).first():
                    print(f"  DUP   {media_id} (content)")
                    dup_content += 1
                    continue

                tags = await extract_tags(text)
                if tags is None:
                    # Unlike asylnet there is no curated metadata fallback —
                    # without extraction the entry would be unusable. Fail
                    # closed and report.
                    failed += 1
                    print(f"  FAIL  {media_id} — LLM extraction unavailable")
                    continue

                warnings = list(tags.warnings or [])
                deactivate = False
                az_ok = bool(tags.aktenzeichen) and _norm_az(tags.aktenzeichen) in _norm_az(text)
                if not az_ok:
                    # No verifiable Az in the PDF itself (partial excerpt or
                    # extraction miss — the LLM otherwise grabs cited courts
                    # from Vgl. blocks). Fall back to the wiki heading, which
                    # is a CLAIM, not proof: entry stays active only if the
                    # heading Az is deterministically found in the PDF text.
                    if heading:
                        tags.court = heading["court"]
                        tags.court_level = None
                        tags.decision_date = heading["date"]
                        tags.aktenzeichen = heading["az"]
                        if _norm_az(heading["az"]) in _norm_az(text):
                            warnings.append("Az aus Wiki-Überschrift, deterministisch im "
                                            "PDF bestätigt (kein Rubrum im Auszug)")
                        else:
                            deactivate = True
                            warnings.append("Metadaten aus Wiki-Überschrift; PDF-Teilauszug "
                                            "ohne Rubrum, Az dort nicht auffindbar - vor "
                                            "Zitierung Volltext besorgen")
                    else:
                        deactivate = True
                        warnings.append("Kein verifizierbares Az im PDF und keine "
                                        "Wiki-Überschrift - inaktiv, manuell prüfen")
                tags.warnings = warnings

                # Cross-source Az dedup: the same decision ingested from
                # asyl.net (or cited_ingest) has different bytes but the
                # same Aktenzeichen — one deduped store, not per-source silos.
                dup_entry = find_active_by_az(db, tags.aktenzeichen)
                if dup_entry is not None:
                    dup_content += 1
                    print(f"  DUP   {media_id} (Az {tags.aktenzeichen} bereits aktiv als "
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
                          f"{len(chunk_text(text))} chunks [dry-run, from {page_id}]")
                    continue

                entry = persist_entry(
                    db, tags, source_type="kanzlei_wiki", source_url=url,
                    source_ref=f"wiki:{media_id}", content_sha256=full_sha,
                    model_label=f"kanzlei-wiki+{extract_model_label()}",
                )
                metadata = {
                    "source_system": "kanzlei_wiki",
                    "rechtsprechung_entry_id": str(entry.id),
                    "wiki_page_id": page_id,
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
                provenance = [f"kanzlei_wiki:{_page_url(base_url, page_id)}",
                              f"media:{url}", f"entry:{entry.id}", f"sha256:{sha16}"]
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
                      f"— {len(text)}c -> {upserted} chunks [from {page_id}]")
                for w in warnings:
                    print(f"  WARN  {media_id}: {w}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAIL  {media_id} — {exc}")

        verb = "would ingest" if args.dry_run else "ingested"
        print(f"\n{verb} {ingested}, dup-ref {dup_ref}, dup-content {dup_content}, "
              f"short {short}, failed {failed}"
              + ("" if args.dry_run else f"; {chunk_total} chunks into '{args.collection}'."))
        if short_ids:
            print("SHORT (scan, needs OCR before ingest): " + ", ".join(short_ids))
        return 0 if failed == 0 else 1
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--collection", default="jurisprudence")
    parser.add_argument("--pages", help="comma-separated page ids (default: all doktrin pages)")
    parser.add_argument("--limit", type=int, help="max new PDFs to ingest this run")
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between page fetches")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(ingest(args))


if __name__ == "__main__":
    sys.exit(main())

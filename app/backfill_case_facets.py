"""One-time backfill: derive cases.facets_json for existing cases from their
anonymized Bescheid text (Pillar 4).

Skips cases that already have matchable facets, so re-runs are cheap and a
manual override is never touched (fill-only merge inside the hook).

Run inside the app container (needs DB + the local Qwen worker):
    docker exec rechtmaschine-app python backfill_case_facets.py --dry-run
    docker exec rechtmaschine-app python backfill_case_facets.py [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio

from database import SessionLocal
from facets import has_matchable_facets
from models import Case, Document
from shared import load_document_text


def _case_material(db, case: Case) -> str:
    """Anonymized text of the case's documents, Bescheide first."""
    docs = (
        db.query(Document)
        .filter(Document.case_id == case.id)
        .order_by(Document.created_at.desc())
        .limit(20)
        .all()
    )
    parts: list[str] = []
    for doc in sorted(docs, key=lambda d: 0 if (d.category or "").casefold() == "bescheid" else 1):
        try:
            text = load_document_text(doc) or ""
        except Exception:
            continue
        if text.strip():
            parts.append(f"### Dokument: {doc.outline_title or doc.filename}\nKategorie: {doc.category}\n{text}")
    return "\n\n".join(parts)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Nur die ersten N Kandidaten-Fälle.")
    ap.add_argument("--dry-run", action="store_true", help="Extrahieren und anzeigen, nichts speichern.")
    args = ap.parse_args()

    from facet_extraction import extract_facets_from_text, merge_facets_fill_only

    db = SessionLocal()
    done = skipped = failed = 0
    try:
        cases = (
            db.query(Case)
            .filter(Case.archived == False)  # noqa: E712
            .order_by(Case.updated_at.desc())
            .all()
        )
        for case in cases:
            if has_matchable_facets(case.facets_json or {}):
                skipped += 1
                continue
            if args.limit and done + failed >= args.limit:
                break
            material = _case_material(db, case)
            if not material.strip():
                print(f"—  {case.name or case.id}: kein Dokumenttext")
                failed += 1
                continue
            extracted = await extract_facets_from_text(material)
            if not extracted:
                print(f"—  {case.name or case.id}: Extraktion leer")
                failed += 1
                continue
            merged = merge_facets_fill_only(case.facets_json or {}, extracted)
            print(f"OK {case.name or case.id}: {merged}")
            if not args.dry_run:
                case.facets_json = merged
                db.add(case)
                db.commit()
            done += 1
    finally:
        db.close()

    print(f"\nbackfill: {done} gefüllt, {skipped} übersprungen (bereits matchbar), {failed} ohne Ergebnis"
          + (" [DRY-RUN]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""Aggregate the curated asyl.net tags already on RechtsprechungEntry rows into
the shared controlled vocabulary (app/rag_vocabulary.json).

themen  = schlagworte, frequency-ranked, singletons dropped (min_count).
laender = distinct country values (kept even if rare — countries are a closed set).
normen  = distinct normen, frequency-ranked, singletons dropped.

Alias maps start empty; curate them by hand afterwards (e.g. add
"wehrdienstverweigerung" -> "wehrdienstentziehung"). Re-running preserves any
hand-edited alias maps already present in the existing JSON.

Run: docker exec rechtmaschine-app python build_vocabulary.py [--min-count 2]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

from database import SessionLocal
from models import RechtsprechungEntry

_WS = re.compile(r"\s+")
OUT_PATH = os.path.join(os.path.dirname(__file__), "rag_vocabulary.json")


def _clean(value: str) -> str:
    return _WS.sub(" ", (value or "").strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=2,
                    help="Drop themen/normen seen fewer than this many times.")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        rows = db.query(
            RechtsprechungEntry.schlagworte,
            RechtsprechungEntry.normen,
            RechtsprechungEntry.country,
        ).all()
    finally:
        db.close()

    themen_counter: Counter[str] = Counter()
    normen_counter: Counter[str] = Counter()
    laender: dict[str, str] = {}  # lowercased -> display form (first seen)

    for schlagworte, normen, country in rows:
        for sw in (schlagworte or []):
            c = _clean(sw)
            if c:
                themen_counter[c.lower()] += 1
        for n in (normen or []):
            c = _clean(n)
            if c:
                normen_counter[c] += 1
        c = _clean(country or "")
        if c:
            laender.setdefault(c.lower(), c)

    themen = sorted([t for t, n in themen_counter.items() if n >= args.min_count])
    normen = sorted([t for t, n in normen_counter.items() if n >= args.min_count])
    laender_list = sorted(laender.values())

    # Preserve hand-curated alias maps across re-runs.
    existing = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as fh:
            existing = json.load(fh)

    data = {
        "themen": themen,
        "themen_aliases": existing.get("themen_aliases", {}),
        "laender": laender_list,
        "laender_aliases": existing.get("laender_aliases", {}),
        "normen": normen,
        "normen_aliases": existing.get("normen_aliases", {}),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=False)

    print(f"themen={len(themen)} laender={len(laender_list)} normen={len(normen)} "
          f"(from {len(rows)} entries, min_count={args.min_count})")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

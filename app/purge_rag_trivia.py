"""Trivial-Korrespondenz aus der kanzlei-RAG-Collection räumen.

Hintergrund (30.07.2026): Der Korpus-Filter (rag/filter_corpus.py) arbeitet auf
Datei-/Pfadnamen und konnte gerichts-adressierte Kurzbriefe (Akteneinsicht,
PKH-Nachreichung, Terminsanträge, Vollmachts-Übersendungen) nicht von
substanziellen Schriftsätzen unterscheiden — ~2.000 der 2.932 Dokumente sind
solche Zwerge. Nach dem Volltext-Retag (retag_pass) tragen alle Dokumente
aussagekräftige Tags, damit ist eine Inhalts-Entscheidung möglich.

Konservative Regel: GELÖSCHT wird nur, was kurz ist UND ausschließlich
Verfahrens-Boilerplate-Tags trägt (keine Normen, kein Herkunftsland).
Alles Unklare landet im REVIEW-Bucket für Jays Blick.

Läuft im App-Container:
  docker exec rechtmaschine-app python purge_rag_trivia.py --dry-run
  docker exec rechtmaschine-app python purge_rag_trivia.py --apply  # nach Review
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any, Optional

import httpx

# Verfahrens-Boilerplate: Tags, die reine Prozess-Korrespondenz beschreiben.
# Bewusst OHNE inhaltliche Verfahrensarten (untätigkeitsklage, dublin-verfahren,
# vorläufiger rechtsschutz …) — die deuten auf Substanz.
PROC_BOILERPLATE = {
    "verwaltungsgericht",
    "bundesverwaltungsgericht",
    "akteneinsicht",
    "prozesskostenhilfe",
    "dolmetscher",
    "mündliche verhandlung",
    "verzicht auf mündliche verhandlung",
    "bundesamt für migration und flüchtlinge",
    "ausländerbehörde",
    "rechtsmittel",
    "asylverfahren",
    "asylantrag",
    "aufenthaltsrecht",
}

PURGE_MAX_CHARS = 1500
REVIEW_MAX_CHARS = 3000


def purge_decision(
    total_chars: int,
    schlagworte: list[str],
    normen: list[str],
    herkunftsland: Optional[str],
) -> tuple[str, str]:
    """Return (bucket, reason) — bucket in {PURGE, REVIEW, KEEP}."""
    tags = set(schlagworte or [])
    pure_boilerplate = bool(tags) and tags <= PROC_BOILERPLATE and not normen and not herkunftsland

    if total_chars < PURGE_MAX_CHARS:
        if pure_boilerplate:
            return "PURGE", "kurz + nur boilerplate-tags"
        if not tags and not normen:
            return "REVIEW", "kurz + ungetaggt (kein Inhalts-Signal)"
        return "REVIEW", "kurz, aber spezifisches Signal (tags/normen/land)"
    if total_chars < REVIEW_MAX_CHARS and pure_boilerplate:
        return "REVIEW", "mittellang + nur boilerplate-tags"
    return "KEEP", "substanz oder lang"


def _rag_base() -> str:
    return os.environ["RAG_SERVICE_URL"].rstrip("/")


def _rag_headers() -> dict[str, str]:
    key = os.environ.get("RAG_SERVICE_API_KEY") or os.environ.get("RAG_API_KEY")
    return {"X-API-Key": key} if key else {}


def scroll_docs(client: httpx.Client, collection: str) -> dict[str, list[dict[str, Any]]]:
    docs: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    cursor: Optional[str] = None
    while True:
        r = client.post(f"{_rag_base()}/v1/rag/chunks/scroll",
                        json={"collection": collection, "cursor": cursor, "limit": 256},
                        headers=_rag_headers(), timeout=60)
        r.raise_for_status()
        data = r.json()
        for chunk in data["chunks"]:
            cid = chunk["chunk_id"]
            did = cid.split("-")[1] if cid.startswith("nc-") else cid
            docs[did].append(chunk)
        cursor = data["next_cursor"]
        if cursor is None:
            break
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="kanzlei")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true",
                    help="PURGE-Bucket wirklich löschen (nur nach Review des Dry-Run-Reports).")
    ap.add_argument("--report", default="rag/data/purge_reports/purge_kanzlei.jsonl",
                    help="Pfad (relativ zum Repo-Root /app) für den JSONL-Report.")
    ap.add_argument("--require-pass", default=None,
                    help="Nur Dokumente mit diesem metadata.retag_pass bewerten; "
                         "andere landen in REVIEW (Tags evtl. veraltet).")
    args = ap.parse_args()
    if not args.dry_run and not args.apply:
        args.dry_run = True

    with httpx.Client() as client:
        docs = scroll_docs(client, args.collection)

        buckets = collections.Counter()
        rows = []
        purge_chunk_ids: list[str] = []
        for did in sorted(docs):
            chunks = docs[did]
            md = (chunks[0].get("metadata") or {})
            total_chars = sum(len(c.get("text") or "") for c in chunks)
            tags = md.get("schlagworte") or []
            normen = md.get("normen") or []
            country = md.get("applicant_origin") or md.get("herkunftsland")

            if args.require_pass and md.get("retag_pass") != args.require_pass:
                bucket, reason = "REVIEW", f"ohne retag_pass={args.require_pass}"
            else:
                bucket, reason = purge_decision(total_chars, tags, normen, country)

            buckets[bucket] += 1
            rows.append({
                "doc": did,
                "bucket": bucket,
                "reason": reason,
                "chars": total_chars,
                "chunks": len(chunks),
                "schlagworte": tags,
                "normen": normen,
                "herkunftsland": country,
                "preview": (chunks[0].get("text") or "").replace("\n", " ")[:180],
            })
            if bucket == "PURGE":
                purge_chunk_ids.extend(c["chunk_id"] for c in chunks)

        report_path = os.path.join("/app", args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"{args.collection}: {len(docs)} Dokumente bewertet")
        for bucket in ("PURGE", "REVIEW", "KEEP"):
            print(f"  {bucket:6s} {buckets.get(bucket, 0):5d}")
        print(f"Report: {report_path}")

        if args.apply:
            deleted = 0
            for start in range(0, len(purge_chunk_ids), 512):
                batch = purge_chunk_ids[start:start + 512]
                r = client.post(f"{_rag_base()}/v1/rag/chunks/delete",
                                json={"collection": args.collection, "chunk_ids": batch},
                                headers=_rag_headers(), timeout=120)
                r.raise_for_status()
                deleted += int(r.json().get("deleted", 0))
            print(f"GELÖSCHT: {deleted} Chunks aus {buckets.get('PURGE', 0)} Dokumenten")
        else:
            print(f"Dry-Run — würde {len(purge_chunk_ids)} Chunks aus {buckets.get('PURGE', 0)} Dokumenten löschen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

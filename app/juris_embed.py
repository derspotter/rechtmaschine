"""Embedding-Sicht für RechtsprechungEntry-Zeilen ohne Volltext.

Der asyl.net-Ingest (jurisprudence_ingest) embeddet den Entscheidungs-
VOLLTEXT in die jurisprudence-Collection. Einträge aus anderen Pfaden —
research_verified-Write-back, manuelle Playbook-Einträge — hatten bisher
keine Chunks: semantisch unauffindbar, und sobald sie aus dem
Recent-200-SQL-Fenster der Pack-Assembly altern, verschwinden sie ganz.

Dieses Modul synthetisiert einen Embedding-Text aus den Entry-Feldern
(Gericht/Az/Leitsatz/Holdings/Argumentationsmuster) und upsertet ihn mit
deterministischen chunk_ids (``juris-entry-<id>-NNN``) — idempotent, ein
Re-Run überschreibt nur die eigenen Chunks und berührt Volltext-Chunks
des asyl.net-Ingests nie.

Backfill (im app-Container, IDs eine pro Zeile):
    docker exec -i rechtmaschine-app python /app/juris_embed.py --ids-file - < ids.txt
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from juris_facets import freeform_text
from rag_vocabulary import (
    facet_metadata,
    load_vocabulary,
    normalize_country,
    normalize_normen,
    normalize_themen,
    tag_line,
)

JURIS_COLLECTION = os.getenv("JURIS_COLLECTION", "jurisprudence")

_CHUNK_TARGET = 1800
_CHUNK_HARD = 2400


def _fmt_date(d: Any) -> Optional[str]:
    return d.strftime("%d.%m.%Y") if d else None


def entry_embed_text(entry: Any) -> str:
    """Prosa-Synthese der Entry-Felder — das, was semantisch auffindbar
    sein soll, wenn kein Volltext existiert."""
    head_bits = [
        entry.court or "",
        f"Entscheidung vom {_fmt_date(entry.decision_date)}" if entry.decision_date else "",
        f"Az. {entry.aktenzeichen}" if entry.aktenzeichen else "",
        f"Ergebnis: {entry.outcome}" if entry.outcome else "",
    ]
    lines: List[str] = [", ".join(b for b in head_bits if b)]
    if entry.country:
        lines.append(f"Herkunftsland: {entry.country}")
    if getattr(entry, "leitsatz", None):
        lines.append(f"Leitsatz: {entry.leitsatz}")
    for holding in entry.key_holdings or []:
        text = freeform_text(holding)
        if text:
            lines.append(text)
    for pattern in entry.argument_patterns or []:
        text = freeform_text(pattern)
        if text:
            lines.append(f"Argumentationsmuster — {text}")
    for fact in entry.key_facts or []:
        text = freeform_text(fact)
        if text:
            lines.append(text)
    if entry.summary:
        lines.append(entry.summary)
    if getattr(entry, "normen", None):
        lines.append("Normen: " + ", ".join(str(n) for n in entry.normen))
    schlagworte = list(getattr(entry, "schlagworte", None) or [])
    if schlagworte:
        lines.append("Schlagwörter: " + ", ".join(str(s) for s in schlagworte))
    return "\n".join(line for line in lines if line)


def _split(text: str, target: int = _CHUNK_TARGET, hard: int = _CHUNK_HARD) -> List[str]:
    """Zeilenweiser Split (Synthese-Text ist zeilenstrukturiert); eine
    überlange Einzelzeile wird hart geschnitten."""
    chunks: List[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > hard:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:hard].strip())
            line = line[hard:]
        candidate = f"{cur}\n{line}" if cur else line
        if len(candidate) > target and cur:
            chunks.append(cur)
            cur = line
        else:
            cur = candidate
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [""]


def entry_chunks(entry: Any, vocab: Any = None) -> List[Dict[str, Any]]:
    """Upsert-Payload für einen Entry — Metadaten-Schema wie der
    asyl.net-Ingest, damit Facetten-Filter beide Herkünfte gleich sehen."""
    if vocab is None:
        vocab = load_vocabulary()
    themen = normalize_themen(vocab, list(getattr(entry, "schlagworte", None) or []) + list(entry.tags or []))
    country_n = normalize_country(vocab, entry.country)
    normen_n = normalize_normen(vocab, list(getattr(entry, "normen", None) or []))

    header_bits = [
        "Rechtsprechung",
        entry.court or "",
        entry.court_level or "",
        entry.decision_date.isoformat() if entry.decision_date else "",
        entry.country or "",
        tag_line(themen, country_n, normen_n),
    ]
    context_header = " | ".join(b for b in header_bits if b)

    metadata = {
        "source_system": entry.source_type or "manual",
        "rechtsprechung_entry_id": str(entry.id),
        "country": entry.country,
        "court": entry.court,
        "court_level": entry.court_level,
        "outcome": entry.outcome,
        "decision_date": entry.decision_date.isoformat() if entry.decision_date else None,
        "aktenzeichen": entry.aktenzeichen,
        "issue_tags": list(entry.tags or []),
        **facet_metadata(themen, country_n, normen_n),
        "instance_weight": getattr(entry, "instance_weight", 0) or 0,
        "language": "de",
    }
    entry_key = str(entry.id).replace("-", "")
    provenance = [f"entry:{entry.id}", f"source:{entry.source_type or 'manual'}"]
    return [
        {
            "chunk_id": f"juris-entry-{entry_key}-{idx:03d}",
            "text": chunk,
            "context_header": context_header,
            "metadata": {**metadata, "chunk_index": idx},
            "provenance": provenance,
        }
        for idx, chunk in enumerate(_split(entry_embed_text(entry)))
    ]


def _default_post(url: str, json: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    import httpx

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=json, headers=headers)
        resp.raise_for_status()
        return resp.json()


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    collection: str = JURIS_COLLECTION,
    post_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> int:
    """Idempotenter Upsert (Batches à 16, wie jurisprudence_ingest)."""
    base = os.getenv("RAG_SERVICE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError("RAG_SERVICE_URL nicht gesetzt")
    key = os.getenv("RAG_API_KEY") or os.getenv("RAG_SERVICE_API_KEY")
    headers = {"X-API-Key": key} if key else {}
    post_fn = post_fn or _default_post
    total = 0
    for start in range(0, len(chunks), 16):
        result = post_fn(
            f"{base}/v1/rag/chunks/upsert",
            json={"collection": collection, "chunks": chunks[start : start + 16]},
            headers=headers,
        )
        total += int(result.get("upserted", 0))
    return total


def embed_entry(entry: Any, vocab: Any = None) -> int:
    """Ein Entry → Chunks → Upsert. Wirft bei RAG-Problemen (Caller
    entscheidet über fail-open vs. fail-closed)."""
    return upsert_chunks(entry_chunks(entry, vocab=vocab))


def _main() -> int:
    import argparse
    import sys
    import uuid as _uuid

    parser = argparse.ArgumentParser(description="Backfill: Entries ohne RAG-Chunks embedden")
    parser.add_argument("--ids-file", required=True, help="Datei mit Entry-UUIDs (eine pro Zeile), '-' für stdin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = sys.stdin.read() if args.ids_file == "-" else open(args.ids_file).read()
    ids = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(_uuid.UUID(line))
        except ValueError:
            print(f"SKIP  ungültige UUID: {line}")

    from database import SessionLocal
    from models import RechtsprechungEntry

    vocab = load_vocabulary()
    db = SessionLocal()
    ok = failed = 0
    try:
        for eid in ids:
            entry = db.query(RechtsprechungEntry).filter(RechtsprechungEntry.id == eid).first()
            if entry is None or not entry.is_active:
                print(f"SKIP  {eid} — nicht gefunden oder inaktiv")
                continue
            if args.dry_run:
                n = len(entry_chunks(entry, vocab=vocab))
                print(f"OK*   {entry.court} {entry.aktenzeichen} — {n} chunks [dry-run]")
                ok += 1
                continue
            try:
                n = embed_entry(entry, vocab=vocab)
                ok += 1
                print(f"OK    {entry.court} {entry.aktenzeichen} ({entry.source_type}) — {n} chunks")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {eid} — {exc}")
    finally:
        db.close()
    print(f"\nembedded {ok}, failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())

"""Nightly jurisprudence enrichment (Pillar 4, hook 2).

Caches two things ONCE per decision on RechtsprechungEntry — computed
case-independently, so the generation hot path stays at zero model calls:

  profil    applicant profile axes recovered from key_facts/summary/leitsatz
            (the asyl.net back-catalog has no structured applicant data)
  reliance  per axis: traegt | erwaehnt | irrelevant — does the decision
            REST on the trait (Regensburg: geschlecht/bildung tragen)

Qwen rules per plan sign-off: advisory, never blocking; small flat JSON;
model label stored for later re-judging (a model change re-enriches).
distinguish_risk stays "ungeprueft" for entries not yet enriched.

CLI (inside the app container, nightly via systemd user timer):
    docker exec rechtmaschine-app python juris_enrichment.py [--limit 50] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from typing import Any, Dict, Tuple

from facets import _normalize_profil

ENRICHMENT_MODEL = (
    os.getenv(
        "JURIS_ENRICHMENT_MODEL",
        os.getenv(
            "MEMORY_EXTRACTION_MODEL",
            os.getenv("LLAMA_SERVER_MODEL", "qwen3.6-27b-udq5xl-vision"),
        ),
    ).strip()
    or "qwen3.6-27b-udq5xl-vision"
)
ENRICHMENT_RETRIES = int((os.getenv("JURIS_ENRICHMENT_RETRIES", "2") or "2").strip())
ENRICHMENT_MAX_CHARS = int((os.getenv("JURIS_ENRICHMENT_MAX_CHARS", "12000") or "12000").strip())

_RELIANCE_VALUES = {"traegt", "erwaehnt", "irrelevant"}
_AXES = ("alter", "geschlecht", "gesundheit", "familienstand", "netzwerk_im_herkunftsland")

# Flat on purpose (local-model schema lessons); reliance axes are prefixed.
_ENRICHMENT_JSON_SPEC = """{
  "alter": 0,
  "geschlecht": "m|w|d|null",
  "gesundheit": "string|null",
  "familienstand": "string|null",
  "netzwerk_im_herkunftsland": true,
  "reliance_alter": "traegt|erwaehnt|irrelevant",
  "reliance_geschlecht": "traegt|erwaehnt|irrelevant",
  "reliance_gesundheit": "traegt|erwaehnt|irrelevant",
  "reliance_familienstand": "traegt|erwaehnt|irrelevant",
  "reliance_netzwerk_im_herkunftsland": "traegt|erwaehnt|irrelevant"
}"""

_ENRICHMENT_RULES = """Du analysierst eine deutsche Asyl-/Migrationsrechts-Entscheidung.

Aufgabe 1 — Profil der Klagepartei aus dem Text rekonstruieren (nur belegte Angaben, sonst null):
alter (Zahl), geschlecht (m/w/d), gesundheit (knapp), familienstand (knapp),
netzwerk_im_herkunftsland (true/false — Familie/tragfähige Kontakte im Herkunftsland?).

Aufgabe 2 — Für JEDE Achse beurteilen, welche Rolle sie für das Ergebnis spielt:
- "traegt": Die Entscheidung stützt sich tragend auf dieses Merkmal (es ist entscheidungserheblich).
- "erwaehnt": Das Merkmal wird erwähnt, trägt die Entscheidung aber nicht.
- "irrelevant": Das Merkmal spielt keine Rolle oder kommt nicht vor.

Antworte NUR mit diesem JSON-Objekt:
""" + _ENRICHMENT_JSON_SPEC


def enrichment_from_flat(parsed: Any) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Split + validate the flat enrichment JSON into (profil, reliance).
    Profil axes run through the same typing as case facets; reliance values
    outside the closed vocabulary are dropped."""
    if not isinstance(parsed, dict):
        return {}, {}
    profil = _normalize_profil({k: parsed.get(k) for k in _AXES})
    reliance = {
        axis: str(parsed.get(f"reliance_{axis}")).strip().lower()
        for axis in _AXES
        if str(parsed.get(f"reliance_{axis}") or "").strip().lower() in _RELIANCE_VALUES
    }
    return profil, reliance


def _entry_material(entry: Any) -> str:
    parts = []
    if getattr(entry, "leitsatz", None):
        parts.append(f"Leitsatz: {entry.leitsatz}")
    for fact in (getattr(entry, "key_facts", None) or []):
        parts.append(f"- {fact}")
    if getattr(entry, "summary", None):
        parts.append(str(entry.summary))
    return "\n".join(p for p in parts if str(p).strip())


def needs_enrichment(entry: Any, model_label: str) -> bool:
    """Enrich when never enriched, or when the judging model changed
    (re-judging on upgrade is deliberate), and there is text to read."""
    if not _entry_material(entry).strip():
        return False
    if getattr(entry, "enriched_at", None) is None:
        return True
    return (getattr(entry, "enrichment_model", None) or "") != model_label


async def enrich_entry_fields(entry: Any) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """One small Qwen call for one decision. ({}, {}) on failure — advisory."""
    from citation_qwen import call_qwen_json
    from shared import ensure_anonymization_service_ready

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    material = _entry_material(entry)
    if not service_url or not material.strip():
        return {}, {}

    prompt = f"{_ENRICHMENT_RULES}\n\nENTSCHEIDUNG:\n{material[:ENRICHMENT_MAX_CHARS]}"
    for attempt in range(1, ENRICHMENT_RETRIES + 1):
        try:
            await ensure_anonymization_service_ready()
            parsed = await call_qwen_json(
                service_url,
                prompt,
                model=ENRICHMENT_MODEL,
                num_predict=800,
                temperature=0.0,
            )
            if parsed:
                return enrichment_from_flat(parsed)
            print(f"[WARN] Enrichment empty/invalid JSON (attempt {attempt})")
        except Exception as exc:
            print(f"[WARN] Enrichment call failed (attempt {attempt}): {exc}")
        if attempt < ENRICHMENT_RETRIES:
            await asyncio.sleep(2 * attempt)
    return {}, {}


async def run_enrichment(limit: int = 50, dry_run: bool = False) -> Dict[str, int]:
    """Enrich up to ``limit`` active entries that need it, newest first.
    Per-entry commit so a crash mid-run loses nothing."""
    from database import SessionLocal
    from models import RechtsprechungEntry

    db = SessionLocal()
    done = skipped = failed = 0
    try:
        entries = (
            db.query(RechtsprechungEntry)
            .filter(RechtsprechungEntry.is_active == True)  # noqa: E712
            .order_by(RechtsprechungEntry.decision_date.desc().nullslast())
            .limit(2000)
            .all()
        )
        for entry in entries:
            if done + failed >= limit:
                break
            if not needs_enrichment(entry, ENRICHMENT_MODEL):
                skipped += 1
                continue
            profil, reliance = await enrich_entry_fields(entry)
            if not profil and not reliance:
                failed += 1
                continue
            print(f"OK {entry.court} {entry.aktenzeichen}: profil={profil} reliance={reliance}")
            if not dry_run:
                entry.profil = profil or None
                entry.reliance = reliance or None
                entry.enriched_at = datetime.utcnow()
                entry.enrichment_model = ENRICHMENT_MODEL
                entry.updated_at = datetime.utcnow()
                db.add(entry)
                db.commit()
            done += 1
    finally:
        db.close()
    return {"enriched": done, "skipped": skipped, "failed": failed}


async def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="Max. Entscheidungen pro Lauf.")
    ap.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts speichern.")
    args = ap.parse_args()
    stats = await run_enrichment(limit=args.limit, dry_run=args.dry_run)
    print(f"\nenrichment: {stats['enriched']} angereichert, {stats['skipped']} aktuell, "
          f"{stats['failed']} ohne Ergebnis" + (" [DRY-RUN]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

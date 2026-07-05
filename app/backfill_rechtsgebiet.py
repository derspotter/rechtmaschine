"""One-time backfill: cases.rechtsgebiet für Bestandsfälle klassifizieren
(Ombudsstelle-Plan Stufe 0).

Fill-only: nur Fälle mit rechtsgebiet IS NULL werden klassifiziert; ein
manuell (PUT) gesetzter Wert wird nie überschrieben. Advisory: bleibt die
Klassifikation leer/unbekannt, bleibt der Fall NULL (= Legacy-Verhalten,
Migrationsrecht).

Run inside the app container:
    docker exec rechtmaschine-app python backfill_rechtsgebiet.py --dry-run
    docker exec rechtmaschine-app python backfill_rechtsgebiet.py [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Optional

from rechtsgebiete import RECHTSGEBIETE, normalize_rechtsgebiet

BACKFILL_MODEL = (
    os.getenv(
        "FACET_EXTRACTION_MODEL",
        os.getenv(
            "MEMORY_EXTRACTION_MODEL",
            os.getenv("LLAMA_SERVER_MODEL", "qwen3.6-27b-udq5xl-vision"),
        ),
    ).strip()
    or "qwen3.6-27b-udq5xl-vision"
)
MAX_CHARS = 20000

_KEYS = "|".join(RECHTSGEBIETE)
_PROMPT = f"""Du ordnest einen deutschen Rechtsfall genau einem Rechtsgebiet zu.

Wähle NUR aus diesen Schlüsseln: {_KEYS}
- asyl: Asylverfahren (BAMF-Bescheid, Klage/Eilverfahren AsylG, Dublin, Widerruf).
- aufenthalt: Aufenthaltsrecht ohne Asylverfahren (Aufenthaltstitel, Duldung, Einbürgerung, Ausweisung).
- sozial: Sozialleistungen (Jobcenter/Bürgergeld, Krankenkasse, Rente, Sozialamt, Wohngeld, BAföG).
- miete: Wohnraummiete (Nebenkosten, Mieterhöhung, Kündigung, Mängel).
- inkasso: Inkasso-/Verbraucherforderungen.
- arbeit: Arbeitsverhältnis (Kündigung, Lohn, Abmahnung).
- sonstiges: nichts davon.

Antworte NUR mit diesem JSON-Objekt:
{{"rechtsgebiet": "{_KEYS}", "begruendung": "string"}}"""


def rechtsgebiet_from_flat(parsed: Any) -> Optional[str]:
    """Kanonischer Key aus der flachen Qwen-Antwort, sonst None."""
    if not isinstance(parsed, dict):
        return None
    return normalize_rechtsgebiet(parsed.get("rechtsgebiet"))


def _case_material(db, case) -> str:
    from models import Document
    from shared import load_document_text

    docs = (
        db.query(Document)
        .filter(Document.case_id == case.id)
        .order_by(Document.created_at.desc())
        .limit(10)
        .all()
    )
    parts = [f"Fallname: {case.name or '(ohne Namen)'}"]
    for doc in docs:
        try:
            text = load_document_text(doc) or ""
        except Exception:
            continue
        if text.strip():
            parts.append(f"### {doc.filename} ({doc.category})\n{text}")
        if sum(len(p) for p in parts) > MAX_CHARS:
            break
    return "\n\n".join(parts)[:MAX_CHARS]


async def _classify(material: str) -> Optional[str]:
    from citation_qwen import call_qwen_json
    from shared import ensure_anonymization_service_ready

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url or not material.strip():
        return None
    try:
        await ensure_anonymization_service_ready()
        parsed = await call_qwen_json(
            service_url,
            f"{_PROMPT}\n\nFALL:\n{material}",
            model=BACKFILL_MODEL,
            num_predict=300,
            temperature=0.0,
        )
        return rechtsgebiet_from_flat(parsed)
    except Exception as exc:
        print(f"[WARN] Rechtsgebiet-Klassifikation fehlgeschlagen: {exc}")
        return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Nur die ersten N Kandidaten.")
    ap.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts speichern.")
    args = ap.parse_args()

    from database import SessionLocal
    from models import Case

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
            if case.rechtsgebiet:
                skipped += 1
                continue
            if args.limit and done + failed >= args.limit:
                break
            gebiet = await _classify(_case_material(db, case))
            if not gebiet:
                failed += 1
                print(f"—  {case.name}: keine Klassifikation")
                continue
            print(f"OK {case.name}: {gebiet}")
            if not args.dry_run:
                case.rechtsgebiet = gebiet
                db.add(case)
                db.commit()
            done += 1
    finally:
        db.close()
    print(f"\nbackfill: {done} klassifiziert, {skipped} übersprungen (bereits gesetzt), "
          f"{failed} ohne Ergebnis" + (" [DRY-RUN]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

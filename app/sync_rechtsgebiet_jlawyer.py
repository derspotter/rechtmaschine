"""Rechtsgebiets-Sync aus j-lawyer (Ombudsstelle Stufe 0).

j-lawyer ist die maßgebliche Quelle: das reason-Feld ("wegen ...") wird
über gebiete_from_reason in kanonische Gebietslisten übersetzt und ADDITIV
in cases.rechtsgebiete gemergt — nie wird ein Gebiet entfernt, manuell
Gesetztes bleibt immer erhalten; das Primärgebiet wird nur gesetzt, wenn
noch keins existiert. Unerkannter reason-Freitext lässt den Fall
unangetastet.

Zuordnung RM-Fall ↔ j-lawyer-Akte über das Aktenzeichen im Fallnamen
(jlawyer_reader.extract_file_number, z.B. "008/26 Lisouskaya" → "008/26").

Run inside the app container:
    docker exec rechtmaschine-app python sync_rechtsgebiet_jlawyer.py --dry-run
    docker exec rechtmaschine-app python sync_rechtsgebiet_jlawyer.py
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Dict, Optional

import httpx

from rechtsgebiete import gebiete_from_reason


def merge_gebiete(existing: Optional[list], mapped: list) -> list:
    """Additiv, dedupliziert, bestehende Reihenfolge zuerst."""
    out = list(existing or [])
    for g in mapped:
        if g not in out:
            out.append(g)
    return out


async def _jlawyer_reason_index() -> Dict[str, str]:
    """fileNumber → reason über die volle Aktenliste (aktive + archivierte,
    damit auch ältere RM-Fälle matchen)."""
    from jlawyer_reader import _api_base, _auth, is_configured

    if not is_configured():
        raise SystemExit("j-lawyer ist nicht konfiguriert (JLAWYER_* env fehlt)")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(f"{_api_base()}/v1/cases/list", auth=_auth())
        response.raise_for_status()
        cases = response.json() or []
    return {
        (c.get("fileNumber") or "").strip(): (c.get("reason") or "").strip()
        for c in cases
        if (c.get("fileNumber") or "").strip()
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts speichern.")
    args = ap.parse_args()

    from database import SessionLocal
    from jlawyer_reader import extract_file_number
    from models import Case

    reasons = await _jlawyer_reason_index()
    db = SessionLocal()
    done = skipped = unmatched = 0
    try:
        cases = (
            db.query(Case)
            .filter(Case.archived == False)  # noqa: E712
            .order_by(Case.updated_at.desc())
            .all()
        )
        for case in cases:
            file_number = extract_file_number(case.name or "")
            reason = reasons.get(file_number or "")
            if not file_number or reason is None:
                unmatched += 1
                print(f"—  {case.name}: keine j-lawyer-Akte gefunden")
                continue
            mapped = gebiete_from_reason(reason)
            if not mapped:
                skipped += 1
                print(f"—  {case.name}: reason {reason!r} nicht erkannt — unangetastet")
                continue
            existing = list(case.rechtsgebiete or ([case.rechtsgebiet] if case.rechtsgebiet else []))
            merged = merge_gebiete(existing, mapped)
            if merged == existing and case.rechtsgebiet:
                skipped += 1
                continue
            print(f"OK {case.name}: {existing or '—'} + {reason!r} → {merged}")
            if not args.dry_run:
                case.rechtsgebiete = merged
                if not case.rechtsgebiet:
                    case.rechtsgebiet = merged[0]
                db.add(case)
                db.commit()
            done += 1
    finally:
        db.close()
    print(f"\nsync: {done} aktualisiert, {skipped} unverändert, {unmatched} ohne j-lawyer-Match"
          + (" [DRY-RUN]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

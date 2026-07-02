"""Write-back of verified research decisions into the jurisprudence store (P3-D).

The gate is strict by design: only decisions that are BOTH deterministically
verified against their fetched page AND grounding-complete become
RechtsprechungEntry rows. Because the jurisprudence pack injects only store
content into drafting context, this gate IS the drafter guard — unverified
research can never reach a Schriftsatz through the pack.

Pure module (stdlib only): the converter is host-testable; the DB hook lives
in research_sources.py.
"""
import hashlib
import re
from typing import Dict, Optional

#: A decision must carry all of these (verified) to enter the store.
WRITEBACK_REQUIRED = ("gericht", "datum", "aktenzeichen", "ergebnis", "zitat")

_WS_RE = re.compile(r"\s+")
_GERMAN_DATE_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")


def _iso_date(value: str) -> str:
    """Normalize '04.11.2025' → '2025-11-04'; ISO input passes through."""
    value = (value or "").strip()
    match = _GERMAN_DATE_RE.match(value)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"
    return value


def writeback_identity_sha(grounding: Dict) -> str:
    """Stable identity for dedupe: sha256 over normalized Gericht|Datum|Az."""
    gericht = _WS_RE.sub("", (grounding.get("gericht") or "").lower())
    datum = _iso_date(grounding.get("datum") or "")
    az = _WS_RE.sub("", (grounding.get("aktenzeichen") or "").lower())
    return hashlib.sha256(f"{gericht}|{datum}|{az}".encode("utf-8")).hexdigest()


def grounding_to_extraction_fields(source: Dict, country: str) -> Optional[Dict]:
    """RechtsprechungExtraction-shaped dict for a verified decision source,
    or None when the source may not enter the store."""
    grounding = source.get("grounding") if isinstance(source, dict) else None
    if not isinstance(grounding, dict):
        return None
    if grounding.get("quelle_typ", "entscheidung") != "entscheidung":
        return None
    if grounding.get("verifiziert") is not True:
        return None
    if any(not grounding.get(f) for f in WRITEBACK_REQUIRED):
        return None

    tags = []
    if grounding.get("lager"):
        tags.append(f"lager:{grounding['lager']}")

    return {
        "country": (country or "").strip() or "Unbekannt",
        "tags": tags,
        "court": grounding["gericht"],
        "court_level": grounding.get("ebene"),
        "decision_date": _iso_date(grounding["datum"]),
        "aktenzeichen": grounding["aktenzeichen"],
        "outcome": grounding["ergebnis"],
        "key_holdings": [grounding["zitat"]],
        "summary": source.get("description") or grounding.get("fit"),
        "confidence": 1.0,
    }

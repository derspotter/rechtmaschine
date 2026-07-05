"""Dünne Rechtsgebiets-Registry (Ombudsstelle-Plan, Stufe 0).

Ein Eintrag pro Gebiet — nur die Felder, die die aktuelle Ausbaustufe
braucht (Spec §4: keine vorauseilende Generalisierung). Die Asyl-Schichten
(Facetten-Hook, Jurisprudenz-Pack, COI-Research) hängen an
``uses_asyl_layers``; NULL/None bedeutet Legacy-Fall und verhält sich wie
bisher (Migrationsrecht).
"""
from typing import Any, Dict, Optional

RECHTSGEBIETE: Dict[str, Dict[str, Any]] = {
    "asyl":       {"label": "Asylrecht", "migrationsrecht": True},
    "aufenthalt": {"label": "Aufenthaltsrecht", "migrationsrecht": True},
    "sozial":     {"label": "Sozialrecht", "migrationsrecht": False},
    "miete":      {"label": "Mietrecht", "migrationsrecht": False},
    "inkasso":    {"label": "Inkasso/Verbraucher", "migrationsrecht": False},
    "arbeit":     {"label": "Arbeitsrecht", "migrationsrecht": False},
    "sonstiges":  {"label": "Sonstiges", "migrationsrecht": False},
}

_ALIASES = {
    "asylrecht": "asyl",
    "auslaenderrecht": "aufenthalt",
    "ausländerrecht": "aufenthalt",
    "aufenthaltsrecht": "aufenthalt",
    "migrationsrecht": "aufenthalt",
    "sozialrecht": "sozial",
    "sgb ii": "sozial",
    "sgb 2": "sozial",
    "buergergeld": "sozial",
    "bürgergeld": "sozial",
    "jobcenter": "sozial",
    "mietrecht": "miete",
    "wohnraummietrecht": "miete",
    "inkassorecht": "inkasso",
    "verbraucherrecht": "inkasso",
    "arbeitsrecht": "arbeit",
}


def normalize_rechtsgebiet(value: Any) -> Optional[str]:
    """Kanonischer Registry-Key oder None (unbekannt/leer)."""
    if not isinstance(value, str):
        return None
    key = value.strip().casefold()
    if not key:
        return None
    if key in RECHTSGEBIETE:
        return key
    return _ALIASES.get(key)


def uses_asyl_layers(rechtsgebiet: Optional[str]) -> bool:
    """Gate für die asylgebundenen Schichten (Facetten, Jurisprudenz-Pack).
    None = Legacy-Fall (Bestand vor Stufe 0) = Migrationsrecht."""
    if rechtsgebiet is None:
        return True
    entry = RECHTSGEBIETE.get(rechtsgebiet)
    return bool(entry and entry["migrationsrecht"])

"""Fenster-Tagging für die Dokument-Tagger (gemma_tagger / qwen_tagger).

Hintergrund: Der Gemma-Slot hat 16k Tokens Kontext — ein ganzes Dokument passt
nicht in einen Call. Statt nur die ersten N Zeichen zu taggen (ergibt generische
Rubrum-Tags), wird das Dokument in Fenster geschnitten, jedes Fenster getaggt
und die Facetten vereinigt: Schlagwörter/Normen nach Häufigkeit über die
Fenster, Herkunftsland per Mehrheit. So erreichen auch die spezifischen Themen
aus der Tiefe der Begründung die Chunk-Metadaten."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def split_windows(text: str, window_chars: int, max_windows: int = 24) -> List[str]:
    """Split text into consecutive windows; cap the count so ein pathologisch
    langes Dokument den Tagging-Batch nicht dominiert."""
    if len(text) <= window_chars:
        return [text]
    windows = [
        text[i:i + window_chars]
        for i in range(0, len(text), window_chars)
    ]
    return windows[:max_windows]


def merge_window_facets(
    results: List[Dict[str, Any]],
    max_themen: int = 12,
    max_normen: int = 12,
) -> Dict[str, Any]:
    """Union der Fenster-Facetten, sortiert nach Häufigkeit (bei Gleichstand
    Reihenfolge des ersten Auftretens). Herkunftsland: häufigster non-null-Wert."""
    themen_counts: Dict[str, int] = {}
    normen_counts: Dict[str, int] = {}
    country_counts: Dict[str, int] = {}

    for result in results:
        for thema in result.get("schlagworte") or []:
            themen_counts[thema] = themen_counts.get(thema, 0) + 1
        for norm in result.get("normen") or []:
            normen_counts[norm] = normen_counts.get(norm, 0) + 1
        country = result.get("herkunftsland")
        if country:
            country_counts[country] = country_counts.get(country, 0) + 1

    def _ranked(counts: Dict[str, int], cap: int) -> List[str]:
        # dict hält Einfüge-Reihenfolge — stable sort nach -count erhält sie
        # als Tiebreaker.
        return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:cap]

    best_country: Optional[str] = None
    if country_counts:
        best_country = sorted(country_counts.items(), key=lambda kv: -kv[1])[0][0]

    return {
        "schlagworte": _ranked(themen_counts, max_themen),
        "herkunftsland": best_country,
        "normen": _ranked(normen_counts, max_normen),
    }

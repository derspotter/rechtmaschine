"""Typed case facets (Pillar 4) — the single source of truth for matching a
case against the jurisprudence store.

A facet block is extracted from the Bescheid at document intake, stored on
``cases.facets_json``, and consumed by ``derive_fingerprint``/scoring. Every
field is normalized into the SAME canonical dialect the store's curated
columns use (rag_vocabulary: laender / normen Gesetz-first / themen), so
matching is a plain field-to-field comparison — never a free-text scrape.
"""

from typing import Any, Dict, Optional

from rag_vocabulary import (
    Vocabulary,
    load_vocabulary,
    normalize_country,
    normalize_normen,
    normalize_themen,
)

FACET_VERFAHRENSARTEN = {
    "asyl_klage",
    "asyl_eilverfahren",
    "asyl_folgeantrag",
    "dublin",
    "widerruf",
    "aufenthaltsrecht",
    "einbuergerung",
    "abschiebungshaft",
    "sonstiges",
}

_VERFAHRENSART_ALIASES = {
    "asylklage": "asyl_klage",
    "klage asyl": "asyl_klage",
    "asylverfahren": "asyl_klage",
    "eilverfahren": "asyl_eilverfahren",
    "eilantrag": "asyl_eilverfahren",
    "folgeantrag": "asyl_folgeantrag",
    "dublin-verfahren": "dublin",
    "dublinverfahren": "dublin",
    "widerrufsverfahren": "widerruf",
    "aufenthg": "aufenthaltsrecht",
    "aufenthalt": "aufenthaltsrecht",
    "aufenthaltserlaubnis": "aufenthaltsrecht",
    "einbürgerung": "einbuergerung",
}

_GESCHLECHT_ALIASES = {
    "m": "m", "w": "w", "d": "d",
    "männlich": "m", "maennlich": "m", "mann": "m", "male": "m",
    "weiblich": "w", "frau": "w", "female": "w",
    "divers": "d",
}

_BOOL_WORDS = {
    "true": True, "ja": True, "yes": True, "vorhanden": True,
    "false": False, "nein": False, "no": False, "nicht vorhanden": False,
    "kein": False, "keins": False, "keines": False,
}


def _clean_str(value: Any) -> str:
    return str(value).strip() if isinstance(value, (str, int, float)) else ""


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return _BOOL_WORDS.get(_clean_str(value).casefold())


def _coerce_alter(value: Any) -> Optional[int]:
    try:
        alter = int(str(value).strip())
    except (ValueError, TypeError):
        return None
    # 0 is the flat-spec type example Qwen echoes for "unbekannt", never a
    # real age — storing it would fill-only-block the true value forever.
    return alter if 1 <= alter <= 120 else None


def _normalize_profil(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    profil: Dict[str, Any] = {}

    alter = _coerce_alter(raw.get("alter"))
    if alter is not None:
        profil["alter"] = alter

    geschlecht = _GESCHLECHT_ALIASES.get(_clean_str(raw.get("geschlecht")).casefold())
    if geschlecht:
        profil["geschlecht"] = geschlecht

    for key in ("gesundheit", "familienstand"):
        value = _clean_str(raw.get(key))
        if value:
            profil[key] = value.casefold()

    netzwerk = _coerce_bool(raw.get("netzwerk_im_herkunftsland"))
    if netzwerk is not None:
        profil["netzwerk_im_herkunftsland"] = netzwerk

    besonderheiten = [
        _clean_str(b) for b in (raw.get("besonderheiten") or [])
        if _clean_str(b)
    ] if isinstance(raw.get("besonderheiten"), list) else []
    if besonderheiten:
        profil["besonderheiten"] = besonderheiten

    return profil


def normalize_facets(raw: Any, vocab: Optional[Vocabulary] = None) -> Dict[str, Any]:
    """Validate + canonicalize a raw (LLM-extracted or hand-edited) facet block.

    Unknown/unmappable values are dropped, never passed through — a facet
    field either speaks the store's canonical dialect or it is absent.
    Returns {} when nothing usable remains.
    """
    if not isinstance(raw, dict):
        return {}
    if vocab is None:
        vocab = load_vocabulary()

    facets: Dict[str, Any] = {}

    herkunftsland = normalize_country(vocab, _clean_str(raw.get("herkunftsland")))
    if herkunftsland:
        facets["herkunftsland"] = herkunftsland

    staatsangehoerigkeit = _clean_str(raw.get("staatsangehoerigkeit"))
    if staatsangehoerigkeit:
        facets["staatsangehoerigkeit"] = staatsangehoerigkeit.casefold()

    verfahrensart_raw = _clean_str(raw.get("verfahrensart")).casefold()
    verfahrensart = (
        verfahrensart_raw if verfahrensart_raw in FACET_VERFAHRENSARTEN
        else _VERFAHRENSART_ALIASES.get(verfahrensart_raw)
    )
    if verfahrensart:
        facets["verfahrensart"] = verfahrensart

    schutzgruende = raw.get("schutzgruende")
    if isinstance(schutzgruende, list):
        normen = normalize_normen(vocab, [_clean_str(n) for n in schutzgruende])
        if normen:
            facets["schutzgruende"] = normen

    themen_raw = raw.get("themen")
    if isinstance(themen_raw, list):
        themen = normalize_themen(vocab, [_clean_str(t) for t in themen_raw])
        if themen:
            facets["themen"] = themen

    region = _clean_str(raw.get("region"))
    if region:
        facets["region"] = region

    profil = _normalize_profil(raw.get("profil"))
    if profil:
        facets["profil"] = profil

    return facets


def has_matchable_facets(facets: Any) -> bool:
    """True when the block carries enough signal to match the store
    (country, normen, or themen — the three curated store columns)."""
    if not isinstance(facets, dict):
        return False
    return bool(
        facets.get("herkunftsland")
        or facets.get("schutzgruende")
        or facets.get("themen")
    )


def facets_complete(facets: Any) -> bool:
    """True when nothing substantial is left for extraction to fill: country,
    normen, themen AND a profil. Intake extraction keeps running (fill-only)
    until a case reaches this — a sparse first hit (e.g. herkunftsland from a
    cover letter) must not freeze the block before the Bescheid arrives."""
    if not isinstance(facets, dict):
        return False
    return bool(
        facets.get("herkunftsland")
        and facets.get("schutzgruende")
        and facets.get("themen")
        and facets.get("profil")
    )


def apply_facets_update(
    existing: Optional[Dict[str, Any]],
    update: Any,
    vocab: Optional[Vocabulary] = None,
) -> Dict[str, Any]:
    """Manual-override semantics for PUT: sent keys overwrite (normalized),
    absent keys survive, an explicit null deletes a key. The profil block
    merges per axis. A partial correction must never wipe extracted fields."""
    merged = dict(existing or {})
    if not isinstance(update, dict):
        return merged
    normalized = normalize_facets(update, vocab)
    for key, value in update.items():
        if value is None:
            merged.pop(key, None)
        elif key == "profil":
            profil = dict(merged.get("profil") or {})
            profil.update(normalized.get("profil") or {})
            for axis, axis_value in (value or {}).items() if isinstance(value, dict) else []:
                if axis_value is None:
                    profil.pop(axis, None)
            if profil:
                merged["profil"] = profil
            else:
                merged.pop("profil", None)
        elif key in normalized:
            merged[key] = normalized[key]
        # sent but unmappable (normalize dropped it): keep the existing value
    return merged

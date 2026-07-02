"""Facet-primary case fingerprinting + jurisprudence-store matching (Pillar 4).

Pure module (no DB/fastapi) so the matching semantics are unit-testable.
``endpoints/jurisprudence.py`` delegates here.

Two dialects meet in the store:
  - case side: ``cases.facets_json`` — canonical rag_vocabulary values
    (laender display form, Gesetz-first normen, canonical themen)
  - store side: the curated ``country`` / ``normen`` / ``schlagworte``
    columns on RechtsprechungEntry, which use the SAME canonical values.

When a case has matchable facets they are the single source of truth and
matching is field-to-field over those curated columns. The prose scrape of
the rendered case memory (countries/§-sections/keywords) remains only as
fallback for cases without facets — it matches against the free-form
``tags`` column, as before.
"""

import re
from datetime import date
from typing import Any, Dict, List, Optional

from facets import has_matchable_facets

# Country names that appear in asylum memory; used for the prose fallback.
_COUNTRY_WORDS = [
    "afghanistan", "syrien", "irak", "iran", "türkei", "tuerkei", "tadschikistan",
    "russland", "somalia", "eritrea", "äthiopien", "aethiopien", "nigeria", "guinea",
    "pakistan", "bangladesch", "georgien", "armenien", "aserbaidschan", "ukraine",
    "libanon", "ägypten", "aegypten", "tunesien", "marokko", "algerien", "gambia",
    "kamerun", "kongo", "sudan", "südsudan", "sri lanka", "indien", "china",
]

# Legal-area signals → (area label, urgency for freshness policy).
_AREA_SIGNALS = [
    ("ausbildungsduldung", "Ausbildungsduldung § 60c AufenthG", "urgent"),
    ("§ 60c", "Ausbildungsduldung § 60c AufenthG", "urgent"),
    ("beschäftigungsduldung", "Beschäftigungsduldung § 60d AufenthG", "urgent"),
    ("widerruf", "Widerrufsverfahren", "normal"),
    ("§ 25b", "Aufenthalt § 25b AufenthG", "normal"),
    ("§ 25a", "Aufenthalt § 25a AufenthG", "normal"),
    ("eilantrag", "Eilrechtsschutz § 123 VwGO", "urgent"),
    ("§ 123", "Eilrechtsschutz § 123 VwGO", "urgent"),
    ("dublin", "Dublin-Verfahren", "urgent"),
    ("familiennachzug", "Familiennachzug", "normal"),
    ("einbürgerung", "Einbürgerung", "evergreen"),
    ("asyl", "Asylverfahren", "normal"),
]

_SECTION_RE = re.compile(r"§\s?\d+[a-z]?(?:\s?abs\.?\s?\d+)?", re.IGNORECASE)

# verfahrensart (facets.py enum) → (legal_area label, urgency)
_VERFAHRENSART_AREA = {
    "asyl_klage": ("Asylverfahren", "normal"),
    "asyl_eilverfahren": ("Eilrechtsschutz § 123 VwGO", "urgent"),
    "asyl_folgeantrag": ("Asylfolgeverfahren", "normal"),
    "dublin": ("Dublin-Verfahren", "urgent"),
    "widerruf": ("Widerrufsverfahren", "normal"),
    "aufenthaltsrecht": ("Aufenthaltsrecht", "normal"),
    "einbuergerung": ("Einbürgerung", "evergreen"),
    "abschiebungshaft": ("Abschiebungshaft", "urgent"),
    "sonstiges": (None, "normal"),
}


def _norm(value: str) -> str:
    value = (value or "").casefold()
    for u, a in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        value = value.replace(u, a)
    return value


def _prose_fingerprint(case_memory_text: str) -> Dict[str, Any]:
    """Heuristic fingerprint from the rendered case memory: countries, legal
    area + urgency, and issue tags (statute sections + area keywords)."""
    hay = case_memory_text or ""
    hay_n = _norm(hay)

    countries = sorted({c for c in _COUNTRY_WORDS if _norm(c) in hay_n})

    legal_area = None
    urgency = "normal"
    for needle, area, urg in _AREA_SIGNALS:
        if _norm(needle) in hay_n:
            legal_area, urgency = area, urg
            break

    sections = sorted({re.sub(r"\s+", " ", m.group(0)).strip().lower() for m in _SECTION_RE.finditer(hay)})
    area_tags = sorted({area_kw for area_kw, _, _ in _AREA_SIGNALS if _norm(area_kw) in hay_n})
    issue_tags = sorted(set(sections) | set(area_tags))

    return {
        "countries": countries,
        "legal_area": legal_area,
        "urgency": urgency,
        "issue_tags": issue_tags,
    }


def derive_fingerprint(case_memory_text: str, facets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Case fingerprint: facets primary, prose scrape as fallback only.

    With matchable facets the fingerprint is derived exclusively from them —
    stable across memory rewrites and available on day one, straight from the
    Bescheid (kills the empty-memory gate). The facet block itself rides along
    under ``fp["facets"]`` for field-to-field store matching.
    """
    if not has_matchable_facets(facets):
        return _prose_fingerprint(case_memory_text)

    legal_area, urgency = _VERFAHRENSART_AREA.get(
        (facets or {}).get("verfahrensart") or "", (None, "normal")
    )
    normen = list(facets.get("schutzgruende") or [])
    themen = list(facets.get("themen") or [])
    countries = [facets["herkunftsland"]] if facets.get("herkunftsland") else []

    return {
        "countries": countries,
        "legal_area": legal_area,
        "urgency": urgency,
        "issue_tags": sorted(set(normen) | set(themen)),
        "facets": facets,
    }


def fingerprint_key(fp: Dict[str, Any]) -> str:
    parts = [
        ",".join(sorted(fp.get("countries") or [])),
        (fp.get("legal_area") or "").strip().lower(),
        ",".join(sorted(fp.get("issue_tags") or [])),
    ]
    return _norm("|".join(parts))[:255]


def _field(entry: Any, name: str) -> Any:
    if isinstance(entry, dict):
        return entry.get(name)
    return getattr(entry, name, None)


def entry_matches(fp: Dict[str, Any], entry: Any) -> bool:
    """Does a RechtsprechungEntry (ORM row or plain dict) match the case?

    Facet path: canonical field-to-field — country equality plus overlap on
    the curated normen/schlagworte columns. The free-form ``tags`` column is
    deliberately ignored here (the old matcher compared against it and never
    intersected the curated columns).

    Prose-fallback path: the legacy semantics — normalized country containment
    and issue_tags vs. free-form tags.
    """
    facets = fp.get("facets")
    if facets:
        herkunftsland = facets.get("herkunftsland")
        if herkunftsland and _norm(_field(entry, "country") or "") != _norm(herkunftsland):
            return False
        want_normen = {_norm(n) for n in (facets.get("schutzgruende") or [])}
        want_themen = {_norm(t) for t in (facets.get("themen") or [])}
        if not want_normen and not want_themen:
            return True
        have_normen = {_norm(str(n)) for n in (_field(entry, "normen") or [])}
        have_themen = {_norm(str(s)) for s in (_field(entry, "schlagworte") or [])}
        return bool(want_normen & have_normen) or bool(want_themen & have_themen)

    countries_n = {_norm(c) for c in (fp.get("countries") or [])}
    tags_n = {_norm(t) for t in (fp.get("issue_tags") or [])}
    country_ok = (not countries_n) or (_norm(_field(entry, "country") or "") in countries_n)
    etags = {_norm(str(t)) for t in (_field(entry, "tags") or [])}
    tag_ok = bool(tags_n & etags) if tags_n else True
    return country_ok and tag_ok


# ---------------------------------------------------------------------------
# Deterministic scoring: fit / lager / distinguish_risk
# ---------------------------------------------------------------------------

# The store speaks two outcome dialects: asyl.net ingest emits
# grant|partial|deny|remand|unknown, the research write-back emits
# stattgegeben|abgelehnt. We always act for the applicant, so pro/gegen is
# fixed relative to plaintiff success.
_OUTCOME_PRO = {"grant", "partial", "stattgegeben", "teilweise stattgegeben"}
_OUTCOME_GEGEN = {"deny", "abgelehnt"}

# Age gap (years) beyond which the alter axis counts as mismatched.
_ALTER_MISMATCH_YEARS = 10

# Profile axes compared when both sides carry a value.
_PROFIL_AXES = ("alter", "geschlecht", "gesundheit", "familienstand", "netzwerk_im_herkunftsland")


def outcome_lager(outcome: Optional[str]) -> str:
    o = _norm(outcome or "").strip()
    if o in {_norm(x) for x in _OUTCOME_PRO}:
        return "pro"
    if o in {_norm(x) for x in _OUTCOME_GEGEN}:
        return "gegen"
    return "neutral"


def _fit(fp: Dict[str, Any], entry: Any) -> float:
    facets = fp.get("facets") or {}
    score = 0.0

    herkunftsland = facets.get("herkunftsland")
    if herkunftsland and _norm(_field(entry, "country") or "") == _norm(herkunftsland):
        score += 0.35

    want_normen = {_norm(n) for n in (facets.get("schutzgruende") or [])}
    if want_normen:
        have = {_norm(str(n)) for n in (_field(entry, "normen") or [])}
        score += 0.30 * (len(want_normen & have) / len(want_normen))

    want_themen = {_norm(t) for t in (facets.get("themen") or [])}
    if want_themen:
        have = {_norm(str(s)) for s in (_field(entry, "schlagworte") or [])}
        score += 0.15 * (len(want_themen & have) / len(want_themen))

    score += min(int(_field(entry, "instance_weight") or 0), 3) * 0.05

    decision_date = _field(entry, "decision_date")
    if isinstance(decision_date, date):
        age_days = (date.today() - decision_date).days
        score += max(0.0, 0.05 * (1 - age_days / 3650))

    return round(min(score, 1.0), 3)


def _axis_mismatch(axis: str, ours: Any, theirs: Any) -> bool:
    if ours is None or theirs is None:
        return False
    if axis == "alter":
        try:
            return abs(int(ours) - int(theirs)) > _ALTER_MISMATCH_YEARS
        except (TypeError, ValueError):
            return False
    if isinstance(ours, bool) or isinstance(theirs, bool):
        return bool(ours) != bool(theirs)
    return _norm(str(ours)) != _norm(str(theirs))


def score_entry(fp: Dict[str, Any], entry: Any) -> Dict[str, Any]:
    """Deterministic per-decision score against the case facets.

    fit               0..1 — country/normen/themen overlap + instance + recency
    lager             pro | gegen | neutral (plaintiff perspective)
    distinguish_risk  ungeprueft (no comparable profiles) | niedrig |
                      moeglich (axis mismatch, reliance unknown/erwaehnt) |
                      hoch (mismatch on an axis the decision RESTS on)
    Zero model calls; the reliance judgment is read from the nightly
    enrichment cache on the entry, never computed here.
    """
    facets = fp.get("facets") or {}
    ours = facets.get("profil") or {}
    theirs = _field(entry, "profil") or {}
    reliance = _field(entry, "reliance") or {}

    mismatch_axes = [
        axis for axis in _PROFIL_AXES
        if _axis_mismatch(axis, ours.get(axis), theirs.get(axis))
    ]
    compared = [
        axis for axis in _PROFIL_AXES
        if ours.get(axis) is not None and theirs.get(axis) is not None
    ]
    tragende = [a for a in mismatch_axes if _norm(str(reliance.get(a) or "")) == "traegt"]

    if not compared:
        risk = "ungeprueft"
    elif tragende:
        risk = "hoch"
    elif mismatch_axes:
        risk = "moeglich"
    else:
        risk = "niedrig"

    return {
        "fit": _fit(fp, entry),
        "lager": outcome_lager(_field(entry, "outcome")),
        "distinguish_risk": risk,
        "mismatch_axes": mismatch_axes,
        "tragende_achsen": tragende,
    }


# ---------------------------------------------------------------------------
# Pack rendering: STÜTZEND / STÜTZEND MIT VORSICHT / GEGEN UNS
# ---------------------------------------------------------------------------

_AXIS_LABELS = {
    "alter": "Alter",
    "geschlecht": "Geschlecht",
    "gesundheit": "Gesundheit",
    "familienstand": "Familienstand",
    "netzwerk_im_herkunftsland": "Netzwerk im Herkunftsland",
}


def _decision_head(d: Dict[str, Any]) -> str:
    return " · ".join(
        str(p) for p in [d.get("court"), d.get("decision_date"), d.get("aktenzeichen"), d.get("outcome")] if p
    )


def _decision_lines(d: Dict[str, Any], with_kernaussage: bool) -> List[str]:
    lines = [f"### {_decision_head(d) or 'Entscheidung'}"]
    if with_kernaussage:
        kern = (d.get("leitsatz") or "").strip() or next(iter(d.get("holdings") or []), "")
        if kern:
            lines.append(f"- Kernaussage: {kern}")
    else:
        for h in (d.get("holdings") or [])[:3]:
            lines.append(f"- {h}")
        for a in (d.get("argument_patterns") or [])[:2]:
            lines.append(f"- Argumentationsmuster: {a}")
    return lines


def _risk_note(d: Dict[str, Any]) -> str:
    axes = d.get("tragende_achsen") or d.get("mismatch_axes") or []
    labels = ", ".join(_AXIS_LABELS.get(a, a) for a in axes)
    if d.get("lager") == "gegen":
        # On a contrary decision a profile mismatch is our CHANCE to
        # distinguish it away ("dort Netzwerk vorhanden"), not a risk.
        return f"- Unterscheidbar: dort abweichend — {labels}" if labels else ""
    if d.get("distinguish_risk") == "hoch":
        return f"- RISIKO: leicht zu unterscheiden — Entscheidung trägt auf: {labels}"
    if d.get("lager") == "neutral":
        return "- Hinweis: Ergebnis unklar/ungeprüft — nicht ungeprüft zitieren"
    if labels:
        return f"- Hinweis: Profil weicht ab ({labels})"
    return ""


def render_scored_block(scored: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    """Sectioned prompt block. Supporting decisions first, then supported-but-
    distinguishable (the Regensburg trap, automated), then contrary decisions
    with their Kernaussage — the drafter must know the best authority against
    us, not just our own."""
    if not scored:
        return ""

    stuetzend: List[Dict[str, Any]] = []
    vorsicht: List[Dict[str, Any]] = []
    gegen: List[Dict[str, Any]] = []
    for d in scored:
        lager = d.get("lager")
        if lager == "gegen":
            gegen.append(d)
        elif lager == "pro" and d.get("distinguish_risk") in {"niedrig", "ungeprueft"}:
            stuetzend.append(d)
        else:  # pro with mismatch risk, or neutral/unknown outcome
            vorsicht.append(d)

    for bucket in (stuetzend, vorsicht, gegen):
        bucket.sort(key=lambda d: d.get("fit") or 0.0, reverse=True)

    sections = [
        ("## STÜTZEND", stuetzend, False),
        ("## STÜTZEND MIT VORSICHT (Distinguish-Risiko)", vorsicht, False),
        ("## GEGEN UNS", gegen, True),
    ]

    parts: List[str] = []
    remaining = max_chars
    for title, bucket, with_kernaussage in sections:
        if not bucket:
            continue
        chunk_lines = [title]
        for d in bucket:
            entry_lines = _decision_lines(d, with_kernaussage)
            note = _risk_note(d)
            if note and (title.startswith("## STÜTZEND MIT") or with_kernaussage):
                entry_lines.append(note)
            entry = "\n".join(entry_lines)
            if len(entry) + len(chunk_lines[0]) > remaining:
                continue
            chunk_lines.append(entry)
            remaining -= len(entry)
        if len(chunk_lines) > 1:
            parts.append("\n\n".join(chunk_lines))
            remaining -= len(title)
    return "\n\n".join(parts)

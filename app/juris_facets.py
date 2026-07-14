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


# Urgency severity for combining the facet- and prose-derived signal.
_URGENCY_RANK = {"evergreen": 0, "normal": 1, "urgent": 2}


def derive_fingerprint(case_memory_text: str, facets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Case fingerprint: facets primary, prose scrape as complement.

    With matchable facets, countries/issue_tags (and thus the fingerprint
    key) come exclusively from the facets — stable across memory rewrites and
    available on day one, straight from the Bescheid (kills the empty-memory
    gate). The prose scrape still contributes what facets cannot lose:
    urgency escalation (a later Eilantrag lives only in the memory), a
    legal_area fallback, and ``prose_tags`` for matching cases whose facets
    carry no normen/themen yet. The facet block itself rides along under
    ``fp["facets"]`` for field-to-field store matching.
    """
    prose = _prose_fingerprint(case_memory_text)
    if not has_matchable_facets(facets):
        return prose

    legal_area, urgency = _VERFAHRENSART_AREA.get(
        (facets or {}).get("verfahrensart") or "", (None, "normal")
    )
    if _URGENCY_RANK.get(prose["urgency"], 1) > _URGENCY_RANK.get(urgency, 1):
        urgency = prose["urgency"]
    if not legal_area:
        legal_area = prose["legal_area"]

    normen = list(facets.get("schutzgruende") or [])
    themen = list(facets.get("themen") or [])
    countries = [facets["herkunftsland"]] if facets.get("herkunftsland") else []

    return {
        "countries": countries,
        "legal_area": legal_area,
        "urgency": urgency,
        "issue_tags": sorted(set(normen) | set(themen)),
        "prose_tags": prose["issue_tags"],
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

    Facet path: canonical field-to-field. Country: equality, but only when
    BOTH sides carry one — country-less leading decisions (BVerwG/EuGH) are
    never rejected on it. Topic: the curated normen/schlagworte columns
    decide when populated (the old matcher's bug was consulting only the
    free-form tags); for older entries whose curated columns are empty, the
    free-form tags remain the fallback. Case side: facet normen/themen when
    present, else the prose issue tags from the case memory — country-only
    facets must not degrade to a match-everything wildcard while a rich
    memory exists.

    Prose-fallback path: the legacy semantics — normalized country
    containment and issue_tags vs. free-form tags.
    """
    facets = fp.get("facets")
    if facets:
        herkunftsland = facets.get("herkunftsland")
        entry_country = _field(entry, "country")
        if herkunftsland and entry_country and _norm(entry_country) != _norm(herkunftsland):
            return False
        # GEAS-Brücke: neue VO-Zitate und alte AsylG-§§ zur selben
        # Rechtsfrage matchen über gemeinsame Gruppen-Tokens.
        want = _expand_with_bridges(facets.get("schutzgruende"))
        want |= {_norm(t) for t in (facets.get("themen") or [])}
        if not want:
            want = {_norm(t) for t in (fp.get("prose_tags") or [])}
        if not want:
            return True
        have_curated = _expand_with_bridges(_field(entry, "normen"))
        have_curated |= {_norm(str(s)) for s in (_field(entry, "schlagworte") or [])}
        if have_curated:
            return bool(want & have_curated)
        have_tags = {_norm(str(t)) for t in (_field(entry, "tags") or [])}
        return bool(want & have_tags)

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

# Lage-Umbrüche pro Herkunftsland (Jay-approved policy, 2026-07-05). A decision
# from before the cutoff assessed a country situation that no longer exists —
# its Lagebewertung is überholt and it stays citable for doctrine only. Keys
# are _norm()-normalized country names; extend when the next country flips.
_LAGE_CUTOFFS = {
    "syrien": date(2024, 12, 8),       # Sturz des Assad-Regimes
    "afghanistan": date(2021, 8, 15),  # Machtübernahme der Taliban
}


def _parse_decision_date(value: Any) -> Optional[date]:
    if isinstance(value, str):
        # Pack decisions store the date as ISO string; re-scoring at render
        # time must keep date-dependent components.
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return value if isinstance(value, date) else None


def lage_stale(entry: Any) -> bool:
    """True when the decision predates the Lage-Umbruch of its country."""
    cutoff = _LAGE_CUTOFFS.get(_norm(str(_field(entry, "country") or "")))
    if not cutoff:
        return False
    decided = _parse_decision_date(_field(entry, "decision_date"))
    return decided is not None and decided < cutoff


# GEAS-Rechts-Umbruch (Jay-approved policy, 2026-07-14): das GEAS-
# Anpassungsgesetz (in Kraft 12.06.2026) hat zentrale AsylG-Normen auf die
# EU-Verordnungen umgestellt — § 3 ff. (Status) auf VO (EU) 2024/1347,
# §§ 29/30/36 (Verfahren/Zulässigkeit) auf VO (EU) 2024/1348, Dublin III
# auf die AMMR. Ältere Entscheidungen zu diesen Normen sind dogmatisch nur
# eingeschränkt übertragbar; die Tatsachen-Rechtsprechung (§ 60 Abs. 5/7
# AufenthG, EMRK Art. 3, GR-Charta) bleibt bewusst unberührt.
_GEAS_CUTOFF = date(2026, 6, 12)

#: GEAS-geänderte AsylG-§§ → Brücken-Token je RECHTSFRAGE (nicht je Gesetz:
#: § 3 und § 4 sind verschiedene Schutzformen und dürfen einander nicht
#: matchen — das kanonische Matching hielt sie schon immer auseinander).
_GEAS_ASYLG_TOKEN = {
    "3": "geas:fluechtling", "3a": "geas:fluechtling", "3b": "geas:fluechtling",
    "3c": "geas:fluechtling", "3d": "geas:fluechtling", "3e": "geas:fluechtling",
    "4": "geas:subsidiaer",
    "29": "geas:unzulaessig", "29a": "geas:unzulaessig",
    "30": "geas:offensichtlich", "30a": "geas:offensichtlich",
    "36": "geas:ou-verfahren",
}

#: VO-Artikel → Token. Qualifikations-VO 2024/1347: Art. 5–14 Flüchtling,
#: Art. 15–17 subsidiär. Verfahrens-VO 2024/1348: Art. 38 Unzulässigkeit,
#: Art. 39/42 offensichtlich unbegründet, Art. 67/68 o.u.-Rechtsschutz.
_ASYLG_PARA_RE = re.compile(r"§\s*(\d+[a-z]?)")
_VO_ART_RE = re.compile(r"art(?:ikel|\.)?\s*(\d+)")


def _geas_bridge_tokens(norm: str) -> set:
    """Brücken-Tokens für eine Norm-Angabe (beide Dialekte: "AsylG § 3 Abs. 1"
    und "VO (EU) 2024/1347 Art. 9"). Leer für GEAS-unberührte Normen. Ein
    VO-Zitat ohne erkennbaren Artikel spannt alle Tokens seiner VO auf."""
    n = _norm(str(norm or ""))
    if "asylg" in n:
        match = _ASYLG_PARA_RE.search(n)
        token = _GEAS_ASYLG_TOKEN.get(match.group(1) if match else "")
        return {token} if token else set()
    if "2024/1347" in n:
        match = _VO_ART_RE.search(n)
        if match:
            art = int(match.group(1))
            if 15 <= art <= 17:
                return {"geas:subsidiaer"}
            return {"geas:fluechtling"}
        return {"geas:fluechtling", "geas:subsidiaer"}
    if "2024/1348" in n:
        match = _VO_ART_RE.search(n)
        if match:
            art = int(match.group(1))
            if art == 38:
                return {"geas:unzulaessig"}
            if art in (39, 42):
                return {"geas:offensichtlich"}
            if art in (67, 68):
                return {"geas:ou-verfahren"}
            return set()
        return {"geas:unzulaessig", "geas:offensichtlich", "geas:ou-verfahren"}
    if "dublin" in n or "604/2013" in n or "2024/1351" in n:
        return {"geas:zustaendigkeit"}
    return set()


def _geas_affected(norm: str) -> bool:
    return bool(_geas_bridge_tokens(norm))


def recht_stale(entry: Any) -> bool:
    """True when the decision predates GEAS AND rests on a GEAS-geänderte
    Norm — dogmatisch nur eingeschränkt übertragbar, Tatsachenwürdigung
    bleibt verwertbar (advisory, nie blockierend)."""
    decided = _parse_decision_date(_field(entry, "decision_date"))
    if decided is None or decided >= _GEAS_CUTOFF:
        return False
    return any(_geas_affected(str(n)) for n in (_field(entry, "normen") or []))


def _expand_with_bridges(values) -> set:
    """Normalisierte Norm-Strings plus ihre GEAS-Brücken-Tokens."""
    out = set()
    for value in values or []:
        out.add(_norm(str(value)))
        out |= _geas_bridge_tokens(str(value))
    return out

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

    want_normen = [_norm(n) for n in (facets.get("schutzgruende") or [])]
    if want_normen:
        have = _expand_with_bridges(_field(entry, "normen"))
        # Eine Fall-Norm zählt als getroffen, wenn sie direkt ODER über die
        # GEAS-Brücke im Eintrag vorkommt; Nenner bleibt die rohe Fall-Liste.
        matched = sum(
            1 for w in want_normen if w in have or (_geas_bridge_tokens(w) & have)
        )
        score += 0.30 * (matched / len(want_normen))

    want_themen = {_norm(t) for t in (facets.get("themen") or [])}
    if want_themen:
        have = {_norm(str(s)) for s in (_field(entry, "schlagworte") or [])}
        score += 0.15 * (len(want_themen & have) / len(want_themen))

    score += min(int(_field(entry, "instance_weight") or 0), 3) * 0.05

    decision_date = _parse_decision_date(_field(entry, "decision_date"))
    if isinstance(decision_date, date):
        age_days = (date.today() - decision_date).days
        score += max(0.0, 0.08 * (1 - age_days / 1825))

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
        "lage_stale": lage_stale(entry),
        "recht_stale": recht_stale(entry),
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


def freeform_text(v: Any) -> str:
    """Freiform-Einträge (holdings/argument_patterns) können dicts sein —
    das Enrichment liefert Argumentationsmuster als use_when/rebuttal/notes.
    Als Prosa rendern, nie als roher Python-dict im Prompt-Block."""
    if isinstance(v, dict):
        use_when = str(v.get("use_when") or "").strip().rstrip(".")
        core = str(v.get("rebuttal") or v.get("notes") or "").strip()
        if use_when and core:
            return f"{use_when}: {core}"
        if core or use_when:
            return core or use_when
        return " — ".join(s for s in (str(x).strip() for x in v.values()) if s)
    return str(v)


def _decision_lines(d: Dict[str, Any], with_kernaussage: bool) -> List[str]:
    lines = [f"### {_decision_head(d) or 'Entscheidung'}"]
    if with_kernaussage:
        kern = (d.get("leitsatz") or "").strip() or freeform_text(
            next(iter(d.get("holdings") or []), "")
        )
        if kern:
            lines.append(f"- Kernaussage: {kern}")
    else:
        for h in (d.get("holdings") or [])[:3]:
            lines.append(f"- {freeform_text(h)}")
        for a in (d.get("argument_patterns") or [])[:2]:
            lines.append(f"- Argumentationsmuster: {freeform_text(a)}")
    return lines


def _lage_note(d: Dict[str, Any]) -> str:
    if not d.get("lage_stale"):
        return ""
    if d.get("lager") == "gegen":
        # A contrary decision resting on the pre-Umbruch Lage is a gift:
        # its factual basis is gone.
        return "- LAGE ÜBERHOLT: Entscheidung vor dem Umbruch im Herkunftsland — Lagebewertung trägt nicht mehr (starkes Distinguishing)"
    return "- LAGE ÜBERHOLT: Entscheidung vor dem Umbruch im Herkunftsland — nicht für die aktuelle Lage zitieren, allenfalls dogmatisch verwertbar"


def _recht_note(d: Dict[str, Any]) -> str:
    if not d.get("recht_stale"):
        return ""
    if d.get("lager") == "gegen":
        # Eine Gegen-Entscheidung auf Vor-GEAS-Rechtsgrundlage ist angreifbar:
        # ihre Norm existiert so nicht mehr.
        return (
            "- GEAS: Entscheidung zur Vor-GEAS-Rechtslage (vor 12.06.2026) — "
            "Rechtsgrundlage geändert (starkes Distinguishing)"
        )
    return (
        "- GEAS: Entscheidung zur Vor-GEAS-Rechtslage (vor 12.06.2026) — "
        "dogmatisch nur eingeschränkt übertragbar, Tatsachenwürdigung weiter verwertbar"
    )


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
        elif (
            lager == "pro"
            and d.get("distinguish_risk") in {"niedrig", "ungeprueft"}
            and not d.get("lage_stale")  # überholte Lage never leads STÜTZEND
            and not d.get("recht_stale")  # Vor-GEAS-Rechtslage ebenso wenig
        ):
            stuetzend.append(d)
        else:  # pro with mismatch risk, stale Lage/Recht, or neutral/unknown
            vorsicht.append(d)

    for bucket in (stuetzend, vorsicht, gegen):
        bucket.sort(
            key=lambda d: (
                not (d.get("lage_stale") or d.get("recht_stale")),
                d.get("fit") or 0.0,
            ),
            reverse=True,
        )

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
            lage = _lage_note(d)
            if lage:
                entry_lines.append(lage)
            recht = _recht_note(d)
            if recht:
                entry_lines.append(recht)
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

"""Pillar 4: facet-primary fingerprint + store matching (pure, no DB/fastapi).

Encodes the two 242/25 failure modes:
  - gate: fresh case with Bescheid facets but empty memory must fingerprint
  - matcher: issue_tags were compared against free-form e.tags, never the
    curated schlagworte/normen columns → canonical facets must match
    country + normen + schlagworte field-to-field.

Run: .venv/bin/python -m pytest tests/test_juris_facets.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from juris_facets import (  # noqa: E402
    derive_fingerprint,
    entry_matches,
    fingerprint_key,
    render_scored_block,
    score_entry,
)

FACETS = {
    "herkunftsland": "Syrien",
    "verfahrensart": "asyl_klage",
    "schutzgruende": ["AufenthG § 60 Abs. 5", "AsylG § 4"],
    "themen": ["existenzminimum", "netzwerk"],
    "profil": {"alter": 21, "geschlecht": "m"},
}

PROSE = "Kläger aus Syrien, Asylverfahren, § 60 Abs. 5 AufenthG, Eilantrag"


# --- fingerprint ---

def test_prose_only_fingerprint_legacy_behavior():
    fp = derive_fingerprint(PROSE)
    assert "syrien" in fp["countries"], fp
    assert fp["legal_area"], fp
    assert any("60" in t for t in fp["issue_tags"]), fp
    assert "facets" not in fp, fp


def test_facets_fingerprint_primary():
    fp = derive_fingerprint("", facets=FACETS)
    assert fp["countries"] == ["Syrien"], fp
    assert fp["legal_area"] == "Asylverfahren", fp
    assert fp["urgency"] == "normal", fp
    assert "AufenthG § 60 Abs. 5" in fp["issue_tags"], fp
    assert "existenzminimum" in fp["issue_tags"], fp
    assert fp["facets"] == FACETS, fp


def test_facets_win_over_prose():
    fp = derive_fingerprint("Kläger aus Afghanistan, Dublin", facets=FACETS)
    assert fp["countries"] == ["Syrien"], fp
    assert fp["legal_area"] == "Asylverfahren", fp


def test_eilverfahren_is_urgent():
    fp = derive_fingerprint("", facets={"herkunftsland": "Syrien", "verfahrensart": "asyl_eilverfahren"})
    assert fp["urgency"] == "urgent", fp


def test_prose_urgency_survives_facets():
    # Bescheid facets say asyl_klage (normal); später kommt ein Eilantrag in
    # den Fall-Speicher — die Dringlichkeit darf nicht verloren gehen.
    fp = derive_fingerprint("Eilantrag § 123 VwGO gestellt", facets=FACETS)
    assert fp["urgency"] == "urgent", fp


def test_prose_legal_area_fills_missing_verfahrensart():
    fp = derive_fingerprint("Dublin-Überstellung", facets={"herkunftsland": "Syrien"})
    assert fp["legal_area"] == "Dublin-Verfahren", fp
    assert fp["urgency"] == "urgent", fp


def test_prose_tags_carried_alongside_facets():
    fp = derive_fingerprint(PROSE, facets={"herkunftsland": "Syrien"})
    assert any("60" in t for t in fp.get("prose_tags") or []), fp
    # fingerprint_key stays facet-driven: prose_tags are not part of issue_tags
    assert fp["issue_tags"] == [], fp


def test_unmatchable_facets_fall_back_to_prose():
    fp = derive_fingerprint(PROSE, facets={"region": "Daraa"})
    assert "syrien" in fp["countries"], fp
    assert "facets" not in fp, fp


def test_fingerprint_key_stable_for_facets():
    k1 = fingerprint_key(derive_fingerprint("", facets=FACETS))
    k2 = fingerprint_key(derive_fingerprint("anderer Memory-Text", facets=FACETS))
    assert k1 == k2, (k1, k2)
    assert k1.strip("|")


# --- matching ---

def _entry(**overrides):
    e = {
        "country": "Syrien",
        "normen": ["AufenthG § 60 Abs. 5"],
        "schlagworte": ["existenzminimum"],
        "tags": [],
    }
    e.update(overrides)
    return e


def test_entry_matches_on_normen():
    fp = derive_fingerprint("", facets=FACETS)
    assert entry_matches(fp, _entry(schlagworte=[]))


def test_entry_matches_on_schlagworte():
    fp = derive_fingerprint("", facets=FACETS)
    assert entry_matches(fp, _entry(normen=[]))


def test_entry_rejected_on_country_mismatch():
    fp = derive_fingerprint("", facets=FACETS)
    assert not entry_matches(fp, _entry(country="Afghanistan"))


def test_entry_rejected_without_field_overlap():
    fp = derive_fingerprint("", facets=FACETS)
    assert not entry_matches(fp, _entry(normen=["AsylG § 3"], schlagworte=["dublin"]))


def test_entry_curated_columns_beat_freeform_tags():
    # When the curated columns are populated they decide; free-form tags must
    # not rescue an entry whose curated normen/schlagworte don't overlap.
    fp = derive_fingerprint("", facets=FACETS)
    assert not entry_matches(fp, _entry(normen=["AsylG § 3"], schlagworte=["dublin"],
                                        tags=["existenzminimum"]))


def test_entry_tags_fallback_when_curated_empty():
    # Older store entries carry only free-form tags — they must stay reachable.
    fp = derive_fingerprint("", facets=FACETS)
    assert entry_matches(fp, _entry(normen=[], schlagworte=[], tags=["existenzminimum"]))


def test_entry_without_country_not_rejected():
    # BVerwG/EuGH leading decisions often have no country — country equality
    # only applies when both sides carry one.
    fp = derive_fingerprint("", facets=FACETS)
    assert entry_matches(fp, _entry(country=None))


def test_entry_matches_case_without_normen_or_themen():
    # Country-only facets, no memory: country is all the signal there is.
    fp = derive_fingerprint("", facets={"herkunftsland": "Syrien", "verfahrensart": "asyl_klage"})
    assert entry_matches(fp, _entry(normen=[], schlagworte=[]))


def test_country_only_facets_use_prose_tags_not_wildcard():
    # Country-only facets but a rich case memory: the prose issue tags must
    # keep filtering — NOT every Syria decision matches.
    fp = derive_fingerprint(PROSE, facets={"herkunftsland": "Syrien"})
    assert not entry_matches(fp, _entry(normen=[], schlagworte=[], tags=["dublin"]))
    assert entry_matches(fp, _entry(normen=[], schlagworte=[], tags=["§ 60 abs. 5"]))


def test_legacy_prose_matching_still_works():
    fp = derive_fingerprint(PROSE)
    entry = _entry(country="syrien", normen=[], schlagworte=[], tags=["§ 60 abs. 5"])
    assert entry_matches(fp, entry)


# --- Lage cutoffs + recency (score_entry / render_scored_block) ---

def _scored_entry(**overrides):
    """Decision dict as the pack render path sees it (ISO-string dates)."""
    e = {
        "country": "Syrien",
        "decision_date": "2025-03-01",
        "court": "VG Test",
        "aktenzeichen": "1 K 1/25",
        "outcome": "stattgegeben",
        "normen": ["AufenthG § 60 Abs. 5"],
        "schlagworte": ["existenzminimum"],
        "instance_weight": 1,
        "profil": None,
        "reliance": None,
        "holdings": ["Tragende Erwägung."],
        "leitsatz": "Leitsatz.",
    }
    e.update(overrides)
    return e


def _fp():
    return derive_fingerprint("", facets=FACETS)


def test_lage_stale_pre_assad_syria():
    score = score_entry(_fp(), _scored_entry(decision_date="2024-12-07"))
    assert score["lage_stale"] is True, score


def test_lage_fresh_from_cutoff_day():
    score = score_entry(_fp(), _scored_entry(decision_date="2024-12-08"))
    assert score["lage_stale"] is False, score


def test_lage_stale_afghanistan_taliban():
    e = _scored_entry(country="Afghanistan", decision_date="2021-08-14")
    assert score_entry(_fp(), e)["lage_stale"] is True
    e = _scored_entry(country="Afghanistan", decision_date="2021-08-15")
    assert score_entry(_fp(), e)["lage_stale"] is False


def test_lage_stale_only_for_cutoff_countries():
    e = _scored_entry(country="Türkei", decision_date="2019-01-01")
    assert score_entry(_fp(), e)["lage_stale"] is False


def test_lage_stale_without_date_is_false():
    score = score_entry(_fp(), _scored_entry(decision_date=None))
    assert score["lage_stale"] is False, score


def test_recency_counts_in_fit():
    fp = _fp()
    new = score_entry(fp, _scored_entry(decision_date="2026-06-01"))
    old = score_entry(fp, _scored_entry(decision_date="2019-06-01"))
    assert new["fit"] > old["fit"], (new, old)


def _rendered(*entries):
    fp = _fp()
    return render_scored_block([{**e, **score_entry(fp, e)} for e in entries])


def test_stale_pro_demoted_to_vorsicht():
    block = _rendered(_scored_entry(decision_date="2023-05-10"))
    assert "## STÜTZEND MIT VORSICHT" in block, block
    assert "LAGE ÜBERHOLT" in block, block
    assert "## STÜTZEND\n" not in block, block


def test_fresh_pro_stays_stuetzend():
    block = _rendered(_scored_entry(decision_date="2025-06-01"))
    assert block.startswith("## STÜTZEND"), block
    assert "LAGE ÜBERHOLT" not in block, block


def test_stale_gegen_flagged_as_distinguishing():
    block = _rendered(_scored_entry(outcome="abgelehnt", decision_date="2023-05-10"))
    assert "## GEGEN UNS" in block, block
    assert "LAGE ÜBERHOLT" in block, block


def test_stale_sorts_after_fresh_within_bucket():
    stale = _scored_entry(outcome="abgelehnt", decision_date="2023-05-10", aktenzeichen="ALT 1/23")
    fresh = _scored_entry(outcome="abgelehnt", decision_date="2025-06-01", aktenzeichen="NEU 1/25")
    block = _rendered(stale, fresh)
    assert block.index("NEU 1/25") < block.index("ALT 1/23"), block


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

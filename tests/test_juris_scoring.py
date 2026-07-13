"""Pillar 4: deterministic decision scoring (fit / lager / distinguish_risk)
and the sectioned pack rendering (STÜTZEND / MIT VORSICHT / GEGEN UNS).

Encodes the Regensburg trap from 242/25: a decision that supports the claim
but rests on profile traits our client does not share (geschlecht/bildung
tragen dort) must surface as STÜTZEND MIT VORSICHT, not as plain support.

Run: .venv/bin/python -m pytest tests/test_juris_scoring.py -q
"""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from juris_facets import derive_fingerprint, render_scored_block, score_entry  # noqa: E402

FACETS = {
    "herkunftsland": "Syrien",
    "verfahrensart": "asyl_klage",
    "schutzgruende": ["AufenthG § 60 Abs. 5", "AsylG § 4"],
    "themen": ["existenzminimum", "netzwerk"],
    "profil": {"alter": 21, "geschlecht": "m", "netzwerk_im_herkunftsland": False},
}

FP = derive_fingerprint("", facets=FACETS)


def _entry(**overrides):
    e = {
        "country": "Syrien",
        "normen": ["AufenthG § 60 Abs. 5"],
        "schlagworte": ["existenzminimum"],
        "outcome": "stattgegeben",
        "instance_weight": 1,
        "decision_date": date.today() - timedelta(days=200),
        "profil": None,
        "reliance": None,
    }
    e.update(overrides)
    return e


# --- lager (outcome dialects) ---

def test_lager_pro_both_dialects():
    assert score_entry(FP, _entry(outcome="stattgegeben"))["lager"] == "pro"
    assert score_entry(FP, _entry(outcome="grant"))["lager"] == "pro"
    assert score_entry(FP, _entry(outcome="partial"))["lager"] == "pro"


def test_lager_gegen_both_dialects():
    assert score_entry(FP, _entry(outcome="abgelehnt"))["lager"] == "gegen"
    assert score_entry(FP, _entry(outcome="deny"))["lager"] == "gegen"


def test_lager_neutral_for_unknown():
    assert score_entry(FP, _entry(outcome="unknown"))["lager"] == "neutral"
    assert score_entry(FP, _entry(outcome=None))["lager"] == "neutral"


# --- fit ---

def test_fit_full_match_beats_partial():
    full = score_entry(FP, _entry())["fit"]
    no_country = score_entry(FP, _entry(country="Afghanistan"))["fit"]
    no_normen = score_entry(FP, _entry(normen=["AsylG § 3"]))["fit"]
    assert full > no_country, (full, no_country)
    assert full > no_normen, (full, no_normen)
    assert 0.0 <= full <= 1.0


def test_fit_rewards_instance_weight():
    vg = score_entry(FP, _entry(instance_weight=1))["fit"]
    bverwg = score_entry(FP, _entry(instance_weight=3))["fit"]
    assert bverwg > vg


def test_fit_accepts_iso_date_string():
    # Pack decisions store decision_date as ISO string; render-time re-scoring
    # must not lose the recency component on them.
    iso = score_entry(FP, _entry(decision_date=(date.today() - timedelta(days=30)).isoformat()))["fit"]
    native = score_entry(FP, _entry(decision_date=date.today() - timedelta(days=30)))["fit"]
    assert iso == native, (iso, native)


def test_fit_rewards_recency():
    fresh = score_entry(FP, _entry(decision_date=date.today() - timedelta(days=30)))["fit"]
    old = score_entry(FP, _entry(decision_date=date.today() - timedelta(days=3000)))["fit"]
    assert fresh > old


# --- distinguish_risk ---

def test_risk_ungeprueft_without_entry_profil():
    s = score_entry(FP, _entry(profil=None))
    assert s["distinguish_risk"] == "ungeprueft", s


def test_risk_niedrig_when_profil_matches():
    s = score_entry(FP, _entry(profil={"alter": 24, "geschlecht": "m",
                                       "netzwerk_im_herkunftsland": False}))
    assert s["distinguish_risk"] == "niedrig", s
    assert s["mismatch_axes"] == [], s


def test_risk_moeglich_on_mismatch_without_reliance():
    s = score_entry(FP, _entry(profil={"geschlecht": "w"}))
    assert s["distinguish_risk"] == "moeglich", s
    assert "geschlecht" in s["mismatch_axes"], s


def test_risk_hoch_when_mismatched_axis_traegt():
    # The Regensburg trap: decision RESTS on a trait we do not share.
    s = score_entry(FP, _entry(profil={"geschlecht": "w"},
                               reliance={"geschlecht": "traegt"}))
    assert s["distinguish_risk"] == "hoch", s
    assert s["tragende_achsen"] == ["geschlecht"], s


def test_risk_alter_bucket_mismatch():
    s = score_entry(FP, _entry(profil={"alter": 55}))
    assert "alter" in s["mismatch_axes"], s


def test_risk_no_case_profil_is_ungeprueft():
    fp = derive_fingerprint("", facets={"herkunftsland": "Syrien", "schutzgruende": ["AsylG § 4"]})
    s = score_entry(fp, _entry(profil={"geschlecht": "w"}))
    assert s["distinguish_risk"] == "ungeprueft", s


# --- rendering ---

def _scored(**overrides):
    d = {
        "court": "VG Regensburg",
        "decision_date": "2025-10-01",
        "aktenzeichen": "RO 13 K 24.32549",
        "outcome": "stattgegeben",
        "holdings": ["Existenzminimum nicht gesichert"],
        "leitsatz": "Leitsatz-Text",
        "fit": 0.8,
        "lager": "pro",
        "distinguish_risk": "niedrig",
        "mismatch_axes": [],
        "tragende_achsen": [],
    }
    d.update(overrides)
    return d


def test_render_sections():
    block = render_scored_block([
        _scored(),
        _scored(court="VG München", aktenzeichen="M 1 K 1", distinguish_risk="hoch",
                mismatch_axes=["geschlecht"], tragende_achsen=["geschlecht"]),
        _scored(court="VG Köln", aktenzeichen="20 K 2", outcome="abgelehnt", lager="gegen",
                mismatch_axes=["netzwerk_im_herkunftsland"]),
    ])
    assert "STÜTZEND" in block
    assert "MIT VORSICHT" in block
    assert "GEGEN UNS" in block
    assert block.index("STÜTZEND") < block.index("MIT VORSICHT") < block.index("GEGEN UNS")
    assert "Geschlecht" in block  # tragende Achse benannt (Anzeige-Label)
    assert "Leitsatz-Text" in block  # Kernaussage bei GEGEN UNS


def test_render_gegen_mismatch_is_chance_not_risiko():
    # Auf einer GEGEN-UNS-Entscheidung ist ein Profil-Mismatch eine
    # Unterscheidbarkeits-CHANCE ("dort Netzwerk vorhanden"), kein Risiko.
    block = render_scored_block([
        _scored(outcome="abgelehnt", lager="gegen",
                mismatch_axes=["netzwerk_im_herkunftsland"],
                tragende_achsen=["netzwerk_im_herkunftsland"]),
    ])
    assert "Unterscheidbar" in block, block
    assert "RISIKO" not in block, block


def test_render_neutral_goes_to_vorsicht():
    block = render_scored_block([_scored(lager="neutral", outcome="unknown")])
    assert "MIT VORSICHT" in block
    assert "STÜTZEND\n" not in block.replace("STÜTZEND MIT VORSICHT", "X")


_PATTERN_DICT = {
    "use_when": "Prüfung einer internen Fluchtalternative im Jemen",
    "rebuttal": "Eine interne Fluchtalternative scheidet aus, da der Konflikt das gesamte Staatsgebiet erfasst.",
    "notes": "Gestützt auf UNHCR-Lageberichte.",
}


def test_render_dict_argument_pattern_as_prose():
    # Enrichment liefert Argumentationsmuster als dict (use_when/rebuttal/notes) —
    # die dürfen nicht als roher Python-dict im Prompt-Block landen.
    block = render_scored_block([_scored(argument_patterns=[_PATTERN_DICT])])
    assert "{" not in block, block
    assert "rebuttal" not in block, block
    assert _PATTERN_DICT["rebuttal"] in block
    assert _PATTERN_DICT["use_when"] in block


def test_render_dict_holding_as_prose():
    block = render_scored_block([_scored(holdings=[{"notes": "Existenzminimum nicht gesichert."}])])
    assert "{" not in block, block
    assert "Existenzminimum nicht gesichert." in block


def test_render_kernaussage_from_dict_holding():
    block = render_scored_block([
        _scored(outcome="abgelehnt", lager="gegen", leitsatz="", holdings=[_PATTERN_DICT]),
    ])
    assert "{" not in block, block
    assert _PATTERN_DICT["rebuttal"] in block


def test_render_empty():
    assert render_scored_block([]) == ""


def test_render_respects_char_budget():
    many = [_scored(aktenzeichen=f"Az {i}", holdings=["x" * 300]) for i in range(50)]
    block = render_scored_block(many, max_chars=1000)
    assert len(block) <= 1200, len(block)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

"""Unit tests for the Pillar-4 case-facet normalizer. Pure Python, no DB/GPU.
Run: python tests/test_facets.py"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag_vocabulary import Vocabulary
from facets import (
    apply_facets_update,
    facets_complete,
    has_matchable_facets,
    normalize_facets,
)

VOCAB = Vocabulary(
    themen=["existenzminimum", "abschiebungsverbot", "netzwerk", "rückkehr"],
    themen_aliases={"soziale bindungen": "netzwerk", "rückkehrsituation": "rückkehr"},
    laender=["Syrien", "Afghanistan", "Jordanien"],
    laender_aliases={"arabische republik syrien": "Syrien"},
    normen=["AufenthG § 60 Abs. 5", "AsylG § 4", "AsylG § 3"],
    normen_aliases={},
)


def test_full_block_normalizes():
    raw = {
        "herkunftsland": "arabische republik syrien",
        "staatsangehoerigkeit": "Syrisch",
        "verfahrensart": "asyl_klage",
        "schutzgruende": ["aufenthg § 60 abs. 5", "AsylG § 4", "§ 99 GibtEsNicht"],
        "themen": ["Soziale Bindungen", "Existenzminimum", "unbekanntes thema"],
        "region": "Daraa",
        "profil": {"alter": 21, "geschlecht": "m", "gesundheit": "gesund",
                   "familienstand": "ledig", "netzwerk_im_herkunftsland": False,
                   "besonderheiten": ["ausreise_als_kind", "11 jahre jordanien"]},
    }
    out = normalize_facets(raw, VOCAB)
    assert out["herkunftsland"] == "Syrien", out
    assert out["staatsangehoerigkeit"] == "syrisch", out
    assert out["verfahrensart"] == "asyl_klage", out
    assert out["schutzgruende"] == ["AufenthG § 60 Abs. 5", "AsylG § 4"], out
    assert out["themen"] == ["netzwerk", "existenzminimum"], out
    assert out["region"] == "Daraa", out
    assert out["profil"]["alter"] == 21, out
    assert out["profil"]["netzwerk_im_herkunftsland"] is False, out
    assert out["profil"]["besonderheiten"] == ["ausreise_als_kind", "11 jahre jordanien"], out


def test_unknown_country_dropped():
    out = normalize_facets({"herkunftsland": "Atlantis"}, VOCAB)
    assert "herkunftsland" not in out, out


def test_invalid_verfahrensart_dropped():
    out = normalize_facets({"verfahrensart": "quatschverfahren", "herkunftsland": "Syrien"}, VOCAB)
    assert "verfahrensart" not in out, out


def test_verfahrensart_aliases():
    assert normalize_facets({"verfahrensart": "Eilverfahren"}, VOCAB).get("verfahrensart") == "asyl_eilverfahren"
    assert normalize_facets({"verfahrensart": "Dublin-Verfahren"}, VOCAB).get("verfahrensart") == "dublin"
    assert normalize_facets({"verfahrensart": "AufenthG"}, VOCAB).get("verfahrensart") == "aufenthaltsrecht"


def test_profil_typing_coercion():
    out = normalize_facets({"profil": {"alter": "21", "geschlecht": "männlich",
                                       "netzwerk_im_herkunftsland": "nein"}}, VOCAB)
    p = out["profil"]
    assert p["alter"] == 21, p
    assert p["geschlecht"] == "m", p
    assert p["netzwerk_im_herkunftsland"] is False, p


def test_profil_invalid_values_dropped():
    out = normalize_facets({"profil": {"alter": 500, "geschlecht": "xyz",
                                       "netzwerk_im_herkunftsland": "vielleicht",
                                       "besonderheiten": ["", "  ", "echt"]}}, VOCAB)
    p = out.get("profil") or {}
    assert "alter" not in p, p
    assert "geschlecht" not in p, p
    assert "netzwerk_im_herkunftsland" not in p, p
    assert p.get("besonderheiten") == ["echt"], p


def test_empty_and_none_input():
    assert normalize_facets(None, VOCAB) == {}
    assert normalize_facets({}, VOCAB) == {}
    assert normalize_facets({"herkunftsland": "", "themen": [], "profil": {}}, VOCAB) == {}


def test_non_dict_tolerated():
    assert normalize_facets("kaputt", VOCAB) == {}
    assert normalize_facets({"profil": "kein dict", "herkunftsland": "Syrien"}, VOCAB) == {"herkunftsland": "Syrien"}


def test_has_matchable_facets():
    assert has_matchable_facets({"herkunftsland": "Syrien"})
    assert has_matchable_facets({"schutzgruende": ["AsylG § 4"]})
    assert has_matchable_facets({"themen": ["netzwerk"]})
    assert not has_matchable_facets({"region": "Daraa", "profil": {"alter": 21}})
    assert not has_matchable_facets({})
    assert not has_matchable_facets(None)


def test_facets_complete():
    assert not facets_complete({"herkunftsland": "Syrien"})
    assert not facets_complete({"herkunftsland": "Syrien", "schutzgruende": ["AsylG § 4"]})
    assert facets_complete({"herkunftsland": "Syrien", "schutzgruende": ["AsylG § 4"],
                            "themen": ["netzwerk"], "profil": {"alter": 21}})
    assert not facets_complete(None)


def test_apply_facets_update_merges_not_wipes():
    existing = {"herkunftsland": "Syrien", "schutzgruende": ["AsylG § 4"],
                "profil": {"alter": 21, "geschlecht": "m"}}
    # Partial correction: only the country is sent — the rest must survive.
    out = apply_facets_update(existing, {"herkunftsland": "Afghanistan"}, VOCAB)
    assert out["herkunftsland"] == "Afghanistan", out
    assert out["schutzgruende"] == ["AsylG § 4"], out
    assert out["profil"]["alter"] == 21, out


def test_apply_facets_update_null_deletes_key():
    existing = {"herkunftsland": "Syrien", "region": "Daraa"}
    out = apply_facets_update(existing, {"region": None}, VOCAB)
    assert "region" not in out, out
    assert out["herkunftsland"] == "Syrien", out


def test_apply_facets_update_profil_axis_merge():
    existing = {"herkunftsland": "Syrien", "profil": {"alter": 21, "geschlecht": "m"}}
    out = apply_facets_update(existing, {"profil": {"alter": 22}}, VOCAB)
    assert out["profil"] == {"alter": 22, "geschlecht": "m"}, out


def test_apply_facets_update_normalizes_values():
    out = apply_facets_update({}, {"herkunftsland": "arabische republik syrien"}, VOCAB)
    assert out.get("herkunftsland") == "Syrien", out


def test_alter_zero_is_unknown_not_an_age():
    # Qwen echoes the flat-spec type example for unknown ages; 0 is never a
    # real Klagepartei age and must not be stored — fill-only merge would
    # otherwise block the true value forever and 0 poisons alter-mismatch
    # scoring (|0 - x| > 10).
    out = normalize_facets({"profil": {"alter": 0, "geschlecht": "m"}})
    assert "alter" not in out.get("profil", {}), out
    assert out["profil"]["geschlecht"] == "m", out

    out = normalize_facets({"profil": {"alter": "0"}})
    assert "alter" not in out.get("profil", {}), out

    out = normalize_facets({"profil": {"alter": -3}})
    assert "alter" not in out.get("profil", {}), out

    out = normalize_facets({"profil": {"alter": 1}})
    assert out["profil"]["alter"] == 1, out


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

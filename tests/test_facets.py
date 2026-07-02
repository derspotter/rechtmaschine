"""Unit tests for the Pillar-4 case-facet normalizer. Pure Python, no DB/GPU.
Run: python tests/test_facets.py"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag_vocabulary import Vocabulary
from facets import normalize_facets, has_matchable_facets

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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

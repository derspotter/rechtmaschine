"""Pillar 4 intake: flat Qwen facet-extraction schema → canonical facet block,
and the fill-only merge policy (manual PUT overrides are never clobbered).
Pure Python — the LLM call itself is not exercised here.

Run: .venv/bin/python -m pytest tests/test_facet_extraction.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from facet_extraction import facets_from_flat, merge_facets_fill_only  # noqa: E402


def test_flat_reshapes_profil():
    flat = {
        "herkunftsland": "Syrien",
        "verfahrensart": "asyl_klage",
        "schutzgruende": ["AsylG § 4"],
        "themen": ["existenzminimum"],
        "region": "Daraa",
        "alter": 21,
        "geschlecht": "m",
        "gesundheit": "gesund",
        "familienstand": "ledig",
        "netzwerk_im_herkunftsland": False,
        "besonderheiten": ["ausreise als kind"],
    }
    raw = facets_from_flat(flat)
    assert raw["herkunftsland"] == "Syrien"
    assert raw["profil"]["alter"] == 21
    assert raw["profil"]["netzwerk_im_herkunftsland"] is False
    assert raw["profil"]["besonderheiten"] == ["ausreise als kind"]
    assert "alter" not in raw


def test_flat_nulls_dropped():
    raw = facets_from_flat({"herkunftsland": None, "alter": None, "geschlecht": None})
    assert raw == {}, raw


def test_flat_non_dict():
    assert facets_from_flat(None) == {}
    assert facets_from_flat("kaputt") == {}


def test_merge_fill_only_keeps_existing():
    existing = {"herkunftsland": "Syrien", "themen": ["netzwerk"]}
    extracted = {"herkunftsland": "Afghanistan", "themen": ["existenzminimum"],
                 "verfahrensart": "asyl_klage"}
    merged = merge_facets_fill_only(existing, extracted)
    assert merged["herkunftsland"] == "Syrien"
    assert merged["themen"] == ["netzwerk"]
    assert merged["verfahrensart"] == "asyl_klage"


def test_merge_fill_only_profil_axes():
    existing = {"profil": {"alter": 21}}
    extracted = {"profil": {"alter": 50, "geschlecht": "m"}}
    merged = merge_facets_fill_only(existing, extracted)
    assert merged["profil"]["alter"] == 21
    assert merged["profil"]["geschlecht"] == "m"


def test_merge_with_empty_existing():
    extracted = {"herkunftsland": "Syrien"}
    assert merge_facets_fill_only(None, extracted) == extracted
    assert merge_facets_fill_only({}, extracted) == extracted


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

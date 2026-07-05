"""Pillar 4 increment 2: nightly Qwen enrichment cache on RechtsprechungEntry
(profil backfill + per-axis reliance). Pure parts only — no DB/GPU here.

Run: .venv/bin/python -m pytest tests/test_juris_enrichment.py -q
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from juris_enrichment import _entry_material, enrichment_from_flat, needs_enrichment  # noqa: E402


def test_flat_splits_profil_and_reliance():
    parsed = {
        "alter": 25, "geschlecht": "w", "gesundheit": "ptbs",
        "familienstand": "ledig", "netzwerk_im_herkunftsland": True,
        "reliance_alter": "irrelevant",
        "reliance_geschlecht": "traegt",
        "reliance_gesundheit": "erwaehnt",
        "reliance_netzwerk_im_herkunftsland": "traegt",
    }
    profil, reliance = enrichment_from_flat(parsed)
    assert profil["geschlecht"] == "w", profil
    assert profil["alter"] == 25, profil
    assert reliance == {
        "alter": "irrelevant",
        "geschlecht": "traegt",
        "gesundheit": "erwaehnt",
        "netzwerk_im_herkunftsland": "traegt",
    }, reliance


def test_flat_invalid_reliance_dropped():
    profil, reliance = enrichment_from_flat({
        "geschlecht": "m",
        "reliance_geschlecht": "vielleicht",
        "reliance_quatsch": "traegt",
    })
    assert profil == {"geschlecht": "m"}, profil
    assert reliance == {}, reliance


def test_flat_geschlecht_alias_normalized():
    profil, _ = enrichment_from_flat({"geschlecht": "weiblich", "alter": "30"})
    assert profil == {"geschlecht": "w", "alter": 30}, profil


def test_flat_empty():
    assert enrichment_from_flat(None) == ({}, {})
    assert enrichment_from_flat({}) == ({}, {})


def _entry(**kw):
    class E:
        enriched_at = None
        enrichment_model = None
        key_facts = ["Klägerin, 25, aus Syrien"]
        summary = "Text"
        leitsatz = None
    e = E()
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def test_material_includes_key_holdings():
    e = _entry(key_holdings=["Zumutbar, da jung, gesund und arbeitsfähig."])
    material = _entry_material(e)
    assert "Tragende Erwägungen:" in material, material
    assert "jung, gesund und arbeitsfähig" in material, material


def test_material_without_key_holdings_unchanged():
    material = _entry_material(_entry())
    assert "Tragende Erwägungen:" not in material, material


def test_needs_enrichment_when_never_enriched():
    assert needs_enrichment(_entry(), "qwen-x")


def test_needs_enrichment_on_model_change():
    e = _entry(enriched_at=datetime(2026, 7, 1), enrichment_model="qwen-old")
    assert needs_enrichment(e, "qwen-x")


def test_no_enrichment_when_current():
    e = _entry(enriched_at=datetime(2026, 7, 1), enrichment_model="qwen-x")
    assert not needs_enrichment(e, "qwen-x")


def test_no_enrichment_without_material():
    e = _entry(key_facts=[], summary="", leitsatz=None)
    assert not needs_enrichment(e, "qwen-x")


def test_profil_alter_zero_dropped():
    # 0 = "unbekannt" echoed from the flat-spec example, not an age.
    profil, _ = enrichment_from_flat({"alter": 0, "geschlecht": "w"})
    assert profil == {"geschlecht": "w"}, profil


def test_spec_does_not_teach_alter_zero():
    from juris_enrichment import _ENRICHMENT_JSON_SPEC
    assert '"alter": 0' not in _ENRICHMENT_JSON_SPEC, _ENRICHMENT_JSON_SPEC


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

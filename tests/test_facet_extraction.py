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


def test_hook_gate_is_completeness_not_matchability():
    # A sparse first extraction (herkunftsland from a cover letter) must NOT
    # freeze the block — the hook keeps extracting until facets_complete.
    import asyncio
    import facet_extraction as fx

    calls = []

    async def fake_extract(text):
        calls.append(text)
        return {"schutzgruende": ["AsylG § 4"]}

    class FakeDB:
        def add(self, x): pass
        def commit(self): pass
        def rollback(self): pass

    class FakeCase:
        id = "c1"
        facets_json = {"herkunftsland": "Syrien"}  # matchable but incomplete

    orig = fx.extract_facets_from_text
    fx.extract_facets_from_text = fake_extract
    try:
        result = asyncio.run(fx.maybe_update_case_facets(FakeDB(), FakeCase(), "Bescheidtext"))
    finally:
        fx.extract_facets_from_text = orig
    assert calls, "Hook hat trotz unvollständiger Facetten nicht extrahiert"
    assert result and result["herkunftsland"] == "Syrien" and result["schutzgruende"] == ["AsylG § 4"], result


def test_spec_does_not_teach_alter_zero():
    from facet_extraction import _FACET_JSON_SPEC
    assert '"alter": 0' not in _FACET_JSON_SPEC, _FACET_JSON_SPEC


def test_empty_answer_is_final_no_retry():
    # A valid-but-empty Qwen answer ("{}" — e.g. a Jobcenter Akte with no
    # asylum facets) is an answer, not a failure: at temperature 0 with a
    # warm prompt cache the retry returns the identical emptiness. Only
    # transport/service errors are worth retrying.
    import asyncio
    import facet_extraction as fx

    calls = []

    async def fake_qwen(prompt):
        calls.append(prompt)
        return {}

    os.environ.setdefault("ANONYMIZATION_SERVICE_URL", "http://stub:8004")
    orig = fx._qwen_json
    fx._qwen_json = fake_qwen
    try:
        out = asyncio.run(fx.extract_facets_from_text("Entziehungsbescheid Jobcenter"))
    finally:
        fx._qwen_json = orig
    assert out == {}, out
    assert len(calls) == 1, f"leere Antwort wurde {len(calls)}x angefragt"


def test_service_error_still_retries():
    import asyncio
    import facet_extraction as fx

    calls = []

    async def broken_qwen(prompt):
        calls.append(prompt)
        raise RuntimeError("service down")

    os.environ.setdefault("ANONYMIZATION_SERVICE_URL", "http://stub:8004")
    orig = fx._qwen_json
    fx._qwen_json = broken_qwen
    try:
        out = asyncio.run(fx.extract_facets_from_text("Bescheidtext"))
    finally:
        fx._qwen_json = orig
    assert out == {}, out
    assert len(calls) == fx.FACET_EXTRACTION_RETRIES, calls


def test_hook_skips_non_migration_rechtsgebiet():
    # Stufe 0: ein Sozialrechtsfall darf den Asyl-Facetten-Hook nicht mehr
    # triggern (kein naechtliches Anklopfen a la 008/26, keine Asyl-Packs).
    import asyncio
    import facet_extraction as fx

    calls = []

    async def fake_extract(text):
        calls.append(text)
        return {"herkunftsland": "Syrien"}

    class FakeCase:
        id = "c3"
        facets_json = {}
        rechtsgebiet = "sozial"

    orig = fx.extract_facets_from_text
    fx.extract_facets_from_text = fake_extract
    try:
        result = asyncio.run(fx.maybe_update_case_facets(None, FakeCase(), "Jobcenter Bescheid"))
    finally:
        fx.extract_facets_from_text = orig
    assert not calls and result is None


def test_hook_multi_gebiet_migration_strang_bleibt_an():
    # 008/26: aufenthalt + sozial -> der Fall IST Migrationsfall, der
    # Facetten-Hook laeuft weiter (die Liste schlaegt das Einzelfeld).
    import asyncio
    import facet_extraction as fx

    calls = []

    async def fake_extract(text):
        calls.append(text)
        return {"herkunftsland": "Belarus"}

    class FakeDB:
        def add(self, x): pass
        def commit(self): pass
        def rollback(self): pass

    class FakeCase:
        id = "c4"
        facets_json = {}
        rechtsgebiet = "sozial"  # Primaerfeld allein wuerde gaten —
        rechtsgebiete = ["sozial", "aufenthalt"]  # die Liste ist die Wahrheit.

    orig = fx.extract_facets_from_text
    fx.extract_facets_from_text = fake_extract
    try:
        result = asyncio.run(fx.maybe_update_case_facets(FakeDB(), FakeCase(), "Anhoerung"))
    finally:
        fx.extract_facets_from_text = orig
    assert calls, "Multi-Gebiet-Migrationsfall wurde faelschlich gegated"
    assert result == {"herkunftsland": "Belarus"}, result


def test_hook_skips_when_complete():
    import asyncio
    import facet_extraction as fx

    calls = []

    async def fake_extract(text):
        calls.append(text)
        return {"themen": ["netzwerk"]}

    class FakeCase:
        id = "c2"
        facets_json = {"herkunftsland": "Syrien", "schutzgruende": ["AsylG § 4"],
                       "themen": ["existenzminimum"], "profil": {"alter": 21}}

    orig = fx.extract_facets_from_text
    fx.extract_facets_from_text = fake_extract
    try:
        result = asyncio.run(fx.maybe_update_case_facets(None, FakeCase(), "Text"))
    finally:
        fx.extract_facets_from_text = orig
    assert not calls and result is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

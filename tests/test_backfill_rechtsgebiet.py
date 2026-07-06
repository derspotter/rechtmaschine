"""Stufe 0: Rechtsgebiets-Backfill — flaches Qwen-JSON → kanonischer Key.
Pure Python; der LLM-Call selbst wird nicht getestet.
Run: .venv/bin/python tests/test_backfill_rechtsgebiet.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from backfill_rechtsgebiet import rechtsgebiet_from_flat  # noqa: E402


def test_flat_canonical_key():
    assert rechtsgebiet_from_flat({"rechtsgebiet": "sozial"}) == "sozial"


def test_flat_alias_normalized():
    assert rechtsgebiet_from_flat({"rechtsgebiet": "Sozialrecht"}) == "sozial"
    assert rechtsgebiet_from_flat({"rechtsgebiet": "Asylrecht"}) == "asyl"


def test_flat_unknown_or_broken():
    assert rechtsgebiet_from_flat({"rechtsgebiet": "steuerrecht"}) is None
    assert rechtsgebiet_from_flat({"rechtsgebiet": None}) is None
    assert rechtsgebiet_from_flat({}) is None
    assert rechtsgebiet_from_flat(None) is None
    assert rechtsgebiet_from_flat("kaputt") is None


def test_compose_material_samples_all_docs():
    # Klassifikation braucht Breite: ein riesiges Einzeldokument (Belarus-
    # Report, 008/26) darf die restlichen Dokumente nicht aus dem Fenster
    # druecken — jedes Dokument wird gekappt, alle kommen vor.
    from backfill_rechtsgebiet import compose_case_material
    named = [
        ("riesig.pdf", "X" * 50000),
        ("jobcenter-bescheid.pdf", "Entziehung Buergergeld"),
        ("brief.pdf", "Anhoerung"),
    ]
    material = compose_case_material("008/26 Test", named, per_doc_cap=3000, total_cap=20000)
    assert "jobcenter-bescheid.pdf" in material, material[:200]
    assert "Entziehung Buergergeld" in material
    assert "brief.pdf" in material
    assert len(material) <= 20000


def test_compose_material_name_only():
    from backfill_rechtsgebiet import compose_case_material
    material = compose_case_material("114/26 Hajstifi", [], per_doc_cap=3000, total_cap=20000)
    assert material == "Fallname: 114/26 Hajstifi", material


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

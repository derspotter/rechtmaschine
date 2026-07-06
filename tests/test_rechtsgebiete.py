"""Stufe 0: Rechtsgebiets-Registry — Normalisierung + Asyl-Schichten-Gate.
Run: .venv/bin/python tests/test_rechtsgebiete.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rechtsgebiete import (  # noqa: E402
    RECHTSGEBIETE,
    normalize_rechtsgebiet,
    uses_asyl_layers,
)


def test_registry_has_all_spec_keys():
    assert set(RECHTSGEBIETE) == {
        "asyl", "aufenthalt", "sozial", "miete", "inkasso", "arbeit", "sonstiges"
    }, set(RECHTSGEBIETE)
    for key, entry in RECHTSGEBIETE.items():
        assert entry.get("label"), f"{key}: label fehlt"
        assert isinstance(entry.get("migrationsrecht"), bool), key


def test_normalize_canonical_keys_pass_through():
    for key in RECHTSGEBIETE:
        assert normalize_rechtsgebiet(key) == key, key


def test_normalize_aliases_and_case():
    assert normalize_rechtsgebiet("Asylrecht") == "asyl"
    assert normalize_rechtsgebiet("Ausländerrecht") == "aufenthalt"
    assert normalize_rechtsgebiet("auslaenderrecht") == "aufenthalt"
    assert normalize_rechtsgebiet("Aufenthaltsrecht") == "aufenthalt"
    assert normalize_rechtsgebiet("Sozialrecht") == "sozial"
    assert normalize_rechtsgebiet("SGB II") == "sozial"
    assert normalize_rechtsgebiet("Mietrecht") == "miete"
    assert normalize_rechtsgebiet("Arbeitsrecht") == "arbeit"
    assert normalize_rechtsgebiet("  MIETE  ") == "miete"


def test_normalize_unknown_and_empty():
    assert normalize_rechtsgebiet("steuerrecht") is None
    assert normalize_rechtsgebiet("") is None
    assert normalize_rechtsgebiet(None) is None
    assert normalize_rechtsgebiet(42) is None


def test_uses_asyl_layers_gate():
    # NULL = Legacy = Migrationsrecht: Bestandsfälle verhalten sich wie bisher.
    assert uses_asyl_layers(None) is True
    assert uses_asyl_layers("asyl") is True
    assert uses_asyl_layers("aufenthalt") is True
    assert uses_asyl_layers("sozial") is False
    assert uses_asyl_layers("miete") is False
    assert uses_asyl_layers("inkasso") is False
    assert uses_asyl_layers("arbeit") is False
    assert uses_asyl_layers("sonstiges") is False
    # Unbekannte Strings fallen sicher: kein Asyl-Kontext für Nicht-Migrationsfälle.
    assert uses_asyl_layers("steuerrecht") is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

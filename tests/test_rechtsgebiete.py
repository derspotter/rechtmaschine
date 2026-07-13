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


def test_normalize_rechtsgebiete_list():
    from rechtsgebiete import normalize_rechtsgebiete
    # String, Liste, Aliase, Dedup (Reihenfolge bleibt), Unbekanntes fliegt.
    assert normalize_rechtsgebiete("sozial") == ["sozial"]
    assert normalize_rechtsgebiete(["Aufenthaltsrecht", "sozial"]) == ["aufenthalt", "sozial"]
    assert normalize_rechtsgebiete(["sozial", "Sozialrecht", "steuerrecht"]) == ["sozial"]
    assert normalize_rechtsgebiete([]) == []
    assert normalize_rechtsgebiete(None) == []
    assert normalize_rechtsgebiete(42) == []


def test_uses_asyl_layers_accepts_lists():
    # Ein Fall ist Migrationsfall, wenn IRGENDEIN Gebiet Migrationsrecht ist
    # (008/26: aufenthalt + sozial -> Asyl-Schichten bleiben an).
    assert uses_asyl_layers(["aufenthalt", "sozial"]) is True
    assert uses_asyl_layers(["sozial", "miete"]) is False
    assert uses_asyl_layers(["asyl"]) is True
    # Leere Liste = wie NULL = Legacy = Migrationsrecht.
    assert uses_asyl_layers([]) is True


def test_gebiete_from_reason_jlawyer():
    from rechtsgebiete import gebiete_from_reason
    # Die realen j-lawyer "wegen ..."-Werte (Sign-off-Mapping 2026-07-06).
    assert gebiete_from_reason("Asylrechts") == ["asyl"]
    assert gebiete_from_reason("Asyls") == ["asyl"]
    assert gebiete_from_reason("Aslyrechts") == ["asyl"]  # realer Tippfehler im Bestand
    assert gebiete_from_reason("Asyls/Aufenthalts") == ["asyl", "aufenthalt"]
    assert gebiete_from_reason("Asyls / Aufenthalts") == ["asyl", "aufenthalt"]
    assert gebiete_from_reason("Aufenthalts") == ["aufenthalt"]
    assert gebiete_from_reason("Einbürgerung") == ["aufenthalt"]
    assert gebiete_from_reason("Niederlassungserlaubnis") == ["aufenthalt"]
    assert gebiete_from_reason("Aufenthalts/Visums") == ["aufenthalt"]
    assert gebiete_from_reason("Arbeitserlaubnis") == ["aufenthalt"]
    assert gebiete_from_reason("Ausländerrecht") == ["aufenthalt"]
    assert gebiete_from_reason("Ausbildungsduldung") == ["aufenthalt"]
    assert gebiete_from_reason("Aufenthalts/Wohnsitzregelung") == ["aufenthalt"]
    assert gebiete_from_reason("Sozialleistungen") == ["sozial"]
    assert gebiete_from_reason("zu Unrecht erhaltene Sozialleistungen") == ["sozial"]
    assert gebiete_from_reason("AsylbLG") == ["asyl"]
    assert gebiete_from_reason("Aufenthalts/Leistungen nach dem AsylbLG") == ["aufenthalt", "asyl"]
    assert gebiete_from_reason("Mietvertrages") == ["miete"]
    assert gebiete_from_reason("Forderung vom 23.11.2021") == ["inkasso"]
    assert gebiete_from_reason("Kaufvertrag v. 28.03.2022") == ["inkasso"]
    assert gebiete_from_reason("Straftat vom 16.02.2022") == ["sonstiges"]
    assert gebiete_from_reason("Ordnungswidrigkeit vom 14.11.2021") == ["sonstiges"]
    assert gebiete_from_reason("Besitz von Betäubungsmitteln") == ["sonstiges"]
    assert gebiete_from_reason("unerlaubte Einreise u. a.") == ["sonstiges"]
    # Nachtrag 2026-07-13: reale Bestand-Werte, die der Sync bisher ausließ.
    assert gebiete_from_reason("Aufenth") == ["aufenthalt"]  # abgekürzt (127/26)
    assert gebiete_from_reason("Staatsangehörigkeit") == ["aufenthalt"]  # 076/26
    assert gebiete_from_reason("Sozialrechts") == ["sozial"]  # 098/25, 046/26
    assert gebiete_from_reason("Rentens") == ["sozial"]  # 009/26
    # Strafsache nach AufenthG ist KEIN Aufenthaltsmandat — bleibt leer.
    assert gebiete_from_reason("Verstoß ./. AufenthG") == []
    # Unerkannter Freitext -> leer -> KEIN Sync (Fall bleibt unangetastet).
    assert gebiete_from_reason("foo") == []
    assert gebiete_from_reason("Vorfall am 18.12.2023") == []
    assert gebiete_from_reason("Patientenrecht") == []
    assert gebiete_from_reason("") == []
    assert gebiete_from_reason(None) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

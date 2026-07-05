"""Stufe 0 Storage: cases.rechtsgebiet — Modellspalte, Migration, to_dict.
Run: .venv/bin/python tests/test_rechtsgebiet_storage.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")


def test_case_model_has_rechtsgebiet():
    from models import Case
    col = Case.__table__.columns.get("rechtsgebiet")
    assert col is not None, "cases.rechtsgebiet fehlt im Modell"
    assert col.nullable, "rechtsgebiet muss nullable sein (NULL = Legacy)"


def test_case_to_dict_includes_rechtsgebiet():
    from models import Case
    assert Case(name="x", rechtsgebiet="sozial").to_dict().get("rechtsgebiet") == "sozial"
    assert Case(name="y").to_dict().get("rechtsgebiet") is None


def test_migration_adds_rechtsgebiet():
    with open(os.path.join(APP_DIR, "main.py"), encoding="utf-8") as fh:
        src = fh.read()
    assert "2026-07-05_case_rechtsgebiet" in src, "Migrations-Eintrag fehlt in main.py"
    assert "ADD COLUMN IF NOT EXISTS rechtsgebiet" in src, "ALTER TABLE fehlt in main.py"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

"""Pillar 4 storage: cases.facets_json column + migration + endpoint helper.
Pure Python (no DB connection) — model metadata and migration SQL only.
Run: python tests/test_case_facets_storage.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")


def test_case_model_has_facets_json():
    from models import Case
    col = Case.__table__.columns.get("facets_json")
    assert col is not None, "cases.facets_json fehlt im Modell"


def test_case_to_dict_includes_facets():
    from models import Case
    case = Case(name="x", facets_json={"herkunftsland": "Syrien"})
    assert case.to_dict().get("facets") == {"herkunftsland": "Syrien"}


def test_migration_adds_facets_json():
    with open(os.path.join(APP_DIR, "main.py"), encoding="utf-8") as fh:
        src = fh.read()
    assert "facets_json" in src, "Migration für cases.facets_json fehlt in main.py"
    assert "ALTER TABLE cases" in src, "ALTER TABLE cases fehlt in main.py"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

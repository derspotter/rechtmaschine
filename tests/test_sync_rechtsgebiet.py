"""Stufe 0: j-lawyer-Rechtsgebiets-Sync — additiver Merge.
(Das reason-Mapping selbst wird in test_rechtsgebiete.py getestet.)
Run: .venv/bin/python tests/test_sync_rechtsgebiet.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


def test_sync_merge_gebiete_additive():
    from sync_rechtsgebiet_jlawyer import merge_gebiete
    # Additiv: nie entfernen, Bestand zuerst, dedupliziert.
    assert merge_gebiete(["aufenthalt", "sozial"], ["sozial"]) == ["aufenthalt", "sozial"]
    assert merge_gebiete(["sozial"], ["asyl", "aufenthalt"]) == ["sozial", "asyl", "aufenthalt"]
    assert merge_gebiete(None, ["asyl"]) == ["asyl"]
    assert merge_gebiete([], []) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

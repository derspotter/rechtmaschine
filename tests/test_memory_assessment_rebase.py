"""Gutachten-level rebase: conflicts are per id, not per field.

    .venv/bin/python -m pytest tests/test_memory_assessment_rebase.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import rebase_assessment_ops, validate_assessment_content  # noqa: E402


def _entry(gid):
    return {
        "id": gid,
        "rechtsfrage": "Frage?",
        "ergebnis": "Antwort.",
        "stand": "2026-09-03",
        "fundstellen": [
            {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"}
        ],
    }


def _content(*ids):
    return validate_assessment_content(
        {"gutachten": [_entry(i) for i in ids], "notizen": ""}
    )


def test_op_on_untouched_id_survives():
    ops = [{"op": "set", "path": "/gutachten/by-id/bb", "value": _entry("bb")}]
    kept = rebase_assessment_ops(ops, _content("aa", "bb"), conflict_ids={"aa"})
    assert kept == ops


def test_op_on_conflicting_id_is_dropped():
    ops = [{"op": "set", "path": "/gutachten/by-id/aa", "value": _entry("aa")}]
    kept = rebase_assessment_ops(ops, _content("aa"), conflict_ids={"aa"})
    assert kept == []


def test_append_of_now_existing_id_is_dropped():
    ops = [{"op": "append", "path": "/gutachten/-", "value": _entry("aa")}]
    kept = rebase_assessment_ops(ops, _content("aa"), conflict_ids=set())
    assert kept == []


def test_op_on_removed_id_is_dropped():
    ops = [{"op": "set", "path": "/gutachten/by-id/gone", "value": _entry("gone")}]
    kept = rebase_assessment_ops(ops, _content("aa"), conflict_ids=set())
    assert kept == []


def test_kept_ops_must_apply_together():
    # Both ops individually valid against the new content, but the second
    # depends on the first having added "cc" -- applied together they are fine.
    ops = [
        {"op": "append", "path": "/gutachten/-", "value": _entry("cc")},
        {"op": "set", "path": "/gutachten/by-id/cc", "value": _entry("cc")},
    ]
    kept = rebase_assessment_ops(ops, _content("aa"), conflict_ids=set())
    assert len(kept) == 2


def test_conflicting_second_op_drops_only_itself():
    ops = [
        {"op": "append", "path": "/gutachten/-", "value": _entry("cc")},
        {"op": "set", "path": "/gutachten/by-id/aa", "value": _entry("aa")},
    ]
    kept = rebase_assessment_ops(ops, _content("aa"), conflict_ids={"aa"})
    assert [op["op"] for op in kept] == ["append"]

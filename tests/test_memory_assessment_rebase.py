"""Gutachten-level rebase: conflicts are per id, not per field.

    .venv/bin/python -m pytest tests/test_memory_assessment_rebase.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    rebase_assessment_ops,
    touched_gutachten_ids,
    validate_assessment_content,
)


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


def test_conflicting_op_anywhere_in_batch_drops_the_whole_batch():
    """Alles oder nichts: a conflict on any op supersedes the whole
    proposal -- it no longer drops only the conflicting op."""
    ops = [
        {"op": "append", "path": "/gutachten/-", "value": _entry("cc")},
        {"op": "set", "path": "/gutachten/by-id/aa", "value": _entry("aa")},
    ]
    kept = rebase_assessment_ops(ops, _content("aa"), conflict_ids={"aa"})
    assert kept == []


def test_rebase_remove_plus_append_survives_or_dies_together(gutachten_factory):
    base = {"gutachten": [gutachten_factory("aa"), gutachten_factory("bb")], "notizen": ""}
    ops = [{"op": "remove", "path": "/gutachten/by-id/aa"},
           {"op": "append", "path": "/gutachten/-", "value": gutachten_factory("aa", ergebnis="neu")}]
    kept = rebase_assessment_ops(ops, base, conflict_ids={"bb"})
    assert [o["op"] for o in kept] == ["remove", "append"]
    assert rebase_assessment_ops(ops, base, conflict_ids={"aa"}) == []


def test_rebase_notizen_conflict_supersedes_whole_proposal(gutachten_factory):
    base = {"gutachten": [gutachten_factory("aa")], "notizen": "neu"}
    ops = [{"op": "set", "path": "/notizen", "value": "alt"},
           {"op": "append", "path": "/gutachten/-", "value": gutachten_factory("bb")}]
    assert rebase_assessment_ops(ops, base, conflict_ids={"notizen"}) == []
    assert len(rebase_assessment_ops(ops, base, conflict_ids=set())) == 2


def test_touched_gutachten_ids():
    ops = [{"op": "set", "path": "/gutachten/by-id/aa", "value": {}},
           {"op": "append", "path": "/gutachten/-", "value": {"id": "cc"}},
           {"op": "set", "path": "/notizen", "value": "x"}]
    assert touched_gutachten_ids(ops) == {"aa", "cc"}

"""Id-addressed patch ops for case_assessment.

    .venv/bin/python -m pytest tests/test_memory_assessment_ops.py -q
"""

import copy
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    apply_assessment_ops,
    sanitize_assessment_ops,
    validate_assessment_content,
)


def _entry(gid="a-frage"):
    return {
        "id": gid,
        "rechtsfrage": "Frage?",
        "ergebnis": "Antwort.",
        "stand": "2026-09-03",
        "fundstellen": [
            {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"}
        ],
    }


def _content(*entries):
    return validate_assessment_content({"gutachten": list(entries), "notizen": ""})


def test_append_adds_entry():
    out = apply_assessment_ops(
        _content(), [{"op": "append", "path": "/gutachten/-", "value": _entry()}]
    )
    assert [g["id"] for g in out["gutachten"]] == ["a-frage"]


def test_append_with_existing_id_is_rejected():
    with pytest.raises(ValueError):
        apply_assessment_ops(
            _content(_entry()),
            [{"op": "append", "path": "/gutachten/-", "value": _entry()}],
        )


def test_set_by_id_replaces_the_right_entry():
    content = _content(_entry("a-frage"), _entry("b-frage"))
    updated = _entry("b-frage")
    updated["ergebnis"] = "Neue Antwort."
    out = apply_assessment_ops(
        content,
        [{"op": "set", "path": "/gutachten/by-id/b-frage", "value": updated}],
    )
    assert out["gutachten"][0]["ergebnis"] == "Antwort."
    assert out["gutachten"][1]["ergebnis"] == "Neue Antwort."


def test_set_by_id_with_mismatched_value_id_is_rejected():
    with pytest.raises(ValueError):
        apply_assessment_ops(
            _content(_entry("a-frage")),
            [{"op": "set", "path": "/gutachten/by-id/a-frage", "value": _entry("b-frage")}],
        )


def test_set_by_id_unknown_is_rejected():
    with pytest.raises(ValueError):
        apply_assessment_ops(
            _content(_entry()),
            [{"op": "set", "path": "/gutachten/by-id/zzz", "value": _entry("zzz")}],
        )


def test_remove_by_id():
    content = _content(_entry("a-frage"), _entry("b-frage"))
    out = apply_assessment_ops(
        content, [{"op": "remove", "path": "/gutachten/by-id/a-frage"}]
    )
    assert [g["id"] for g in out["gutachten"]] == ["b-frage"]


def test_index_path_is_rejected():
    with pytest.raises(ValueError):
        apply_assessment_ops(
            _content(_entry()),
            [{"op": "set", "path": "/gutachten/0", "value": _entry()}],
        )


def test_scalar_set_on_notizen():
    out = apply_assessment_ops(_content(), [{"op": "set", "path": "/notizen", "value": "x"}])
    assert out["notizen"] == "x"


def test_set_by_id_resets_server_fields():
    content = _content(_entry())
    content["gutachten"][0]["fundstellen"][0]["store"] = "verified"
    updated = _entry()
    out = apply_assessment_ops(
        content, [{"op": "set", "path": "/gutachten/by-id/a-frage", "value": updated}]
    )
    assert out["gutachten"][0]["fundstellen"][0]["store"] == "unchecked"


def test_sanitize_strips_client_supplied_store_values():
    ops = [
        {
            "op": "append",
            "path": "/gutachten/-",
            "value": {
                **_entry(),
                "fundstellen": [
                    {
                        "gericht": "OVG NRW",
                        "datum": "2012-06-18",
                        "az": "18 E 491/12",
                        "store": "verified",
                    }
                ],
            },
        }
    ]
    original = copy.deepcopy(ops)
    cleaned = sanitize_assessment_ops(ops)
    assert "store" not in cleaned[0]["value"]["fundstellen"][0]
    assert ops == original, "sanitize must not mutate its input"

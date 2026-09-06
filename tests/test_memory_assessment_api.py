"""Route wiring for case_assessment: payload shape, target allowlist, recheck.

Source-level assertions plus pure-function tests -- the live API is covered by
the acceptance run in Task 12.

    .venv/bin/python -m pytest tests/test_memory_assessment_api.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

ENDPOINT = (APP_DIR / "endpoints" / "agent_memory.py").read_text(encoding="utf-8")


def test_create_route_validates_the_target_against_the_registry():
    assert "_KNOWN_TARGETS" in ENDPOINT
    assert "Unbekanntes Memory-Target" in ENDPOINT


def test_combined_payload_carries_the_third_block():
    assert '"case_assessment": assessment_payload' in ENDPOINT


def test_frontend_payload_has_an_assessment_section():
    assert '"assessment"' in ENDPOINT
    assert "Gutachten-Vorschlag" in ENDPOINT


def test_accept_route_surfaces_warnings():
    assert "assessment_warnings" in ENDPOINT


def test_recheck_route_exists():
    assert '@router.post("/cases/{case_id}/assessment/recheck")' in ENDPOINT


def test_recheck_counts_only_changed_fundstellen():
    from assessment_memory import reconcile_store, validate_assessment_content

    content = validate_assessment_content(
        {
            "gutachten": [
                {
                    "id": "aa",
                    "rechtsfrage": "F?",
                    "ergebnis": "E.",
                    "stand": "2026-09-03",
                    "fundstellen": [
                        {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"},
                        {"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20"},
                    ],
                }
            ],
            "notizen": "",
        }
    )
    store_map = {"18e491/12": [{"id": "entry-1", "decision_date": "2012-06-18"}]}
    once, _ = reconcile_store(content, store_map, None, "2026-09-04T10:00:00")
    twice, _ = reconcile_store(once, store_map, None, "2026-09-04T10:00:00")
    states_once = [f["store"] for f in once["gutachten"][0]["fundstellen"]]
    states_twice = [f["store"] for f in twice["gutachten"][0]["fundstellen"]]
    assert states_once == ["verified", "not_in_store"]
    assert states_once == states_twice, "recheck must be idempotent"


def _gutachten_entry(gid, fundstellen):
    return {
        "id": gid,
        "rechtsfrage": "F?",
        "ergebnis": "E.",
        "stand": "2026-09-03",
        "fundstellen": fundstellen,
    }


def _fundstelle(az, store="unchecked", store_entry_id=None):
    return {
        "gericht": "OVG NRW",
        "datum": "2012-06-18",
        "az": az,
        "store": store,
        "store_entry_id": store_entry_id,
    }


def _assessment(entries):
    from assessment_memory import validate_assessment_content

    return validate_assessment_content({"gutachten": entries, "notizen": ""})


def test_count_store_changes_counts_newly_verified_and_not_in_store():
    from assessment_memory import count_store_changes

    previous = _assessment(
        [_gutachten_entry("aa", [_fundstelle("18 E 491/12"), _fundstelle("7 L 7/20")])]
    )
    new = _assessment(
        [
            _gutachten_entry(
                "aa",
                [
                    _fundstelle("18 E 491/12", store="verified", store_entry_id="entry-1"),
                    _fundstelle("7 L 7/20", store="not_in_store"),
                ],
            )
        ]
    )
    assert count_store_changes(previous, new) == (2, 1)


def test_count_store_changes_is_zero_when_nothing_changed():
    from assessment_memory import count_store_changes

    content = _assessment(
        [
            _gutachten_entry(
                "aa", [_fundstelle("18 E 491/12", store="verified", store_entry_id="entry-1")]
            )
        ]
    )
    assert count_store_changes(content, content) == (0, 0)


def test_count_store_changes_only_flags_the_gutachten_that_actually_changed():
    from assessment_memory import count_store_changes

    previous = _assessment(
        [
            _gutachten_entry(
                "aa", [_fundstelle("18 E 491/12", store="verified", store_entry_id="entry-1")]
            ),
            _gutachten_entry("bb", [_fundstelle("7 L 7/20")]),
        ]
    )
    new = _assessment(
        [
            _gutachten_entry(
                "aa", [_fundstelle("18 E 491/12", store="verified", store_entry_id="entry-1")]
            ),
            _gutachten_entry("bb", [_fundstelle("7 L 7/20", store="verified", store_entry_id="entry-2")]),
        ]
    )
    assert count_store_changes(previous, new) == (1, 1)


def test_count_store_changes_flags_a_store_entry_id_change_alone():
    from assessment_memory import count_store_changes

    previous = _assessment(
        [
            _gutachten_entry(
                "aa", [_fundstelle("18 E 491/12", store="verified", store_entry_id="entry-1")]
            )
        ]
    )
    new = _assessment(
        [
            _gutachten_entry(
                "aa", [_fundstelle("18 E 491/12", store="verified", store_entry_id="entry-2")]
            )
        ]
    )
    assert count_store_changes(previous, new) == (1, 1)

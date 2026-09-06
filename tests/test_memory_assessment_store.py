"""Store reconciliation for Gutachten-Fundstellen.

    .venv/bin/python -m pytest tests/test_memory_assessment_store.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    changed_gutachten_ids,
    citation_lines,
    reconcile_store,
    validate_assessment_content,
)

NOW = "2026-09-04T10:00:00"


def _content(*fundstellen):
    return validate_assessment_content(
        {
            "gutachten": [
                {
                    "id": "a-frage",
                    "rechtsfrage": "Frage?",
                    "ergebnis": "Antwort.",
                    "stand": "2026-09-03",
                    "fundstellen": list(fundstellen),
                }
            ],
            "notizen": "",
        }
    )


def _store_map():
    return {
        "18e491/12": [{"id": "entry-1", "decision_date": "2012-06-18"}],
        "1k2/24": [{"id": "entry-2", "decision_date": "2024-01-01"}],
        "9x9/99": [
            {"id": "entry-3", "decision_date": "1999-01-01"},
            {"id": "entry-4", "decision_date": "1999-06-06"},
        ],
    }


def test_matching_az_and_date_is_verified():
    content = _content({"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"})
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    fundstelle = out["gutachten"][0]["fundstellen"][0]
    assert fundstelle["store"] == "verified"
    assert fundstelle["store_entry_id"] == "entry-1"
    assert fundstelle["store_checked_at"] == NOW
    assert warnings == []


def test_wrong_date_is_date_mismatch():
    content = _content({"gericht": "VG X", "datum": "2024-02-02", "az": "1 K 2/24"})
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    assert out["gutachten"][0]["fundstellen"][0]["store"] == "date_mismatch"
    assert warnings == [{"gutachten_id": "a-frage", "az": "1 K 2/24", "store": "date_mismatch"}]


def test_unknown_az_is_not_in_store():
    content = _content({"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20"})
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    assert out["gutachten"][0]["fundstellen"][0]["store"] == "not_in_store"
    assert warnings[0]["store"] == "not_in_store"


def test_multiple_store_hits_one_matching_date_is_verified():
    content = _content({"gericht": "VG Z", "datum": "1999-06-06", "az": "9 X 9/99"})
    out, _ = reconcile_store(content, _store_map(), None, NOW)
    fundstelle = out["gutachten"][0]["fundstellen"][0]
    assert fundstelle["store"] == "verified"
    assert fundstelle["store_entry_id"] == "entry-4"


def test_changed_ids_limits_reconciliation():
    content = _content({"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"})
    out, _ = reconcile_store(content, _store_map(), set(), NOW)
    assert out["gutachten"][0]["fundstellen"][0]["store"] == "unchecked"


def test_changed_gutachten_ids_ignores_server_fields():
    before = _content({"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"})
    after, _ = reconcile_store(before, _store_map(), None, NOW)
    assert changed_gutachten_ids(before, after) == set()
    after["gutachten"][0]["ergebnis"] = "Andere Antwort."
    assert changed_gutachten_ids(before, after) == {"a-frage"}


def test_citation_lines_use_the_parser_format():
    content = _content(
        {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12", "art": "Beschluss"},
        {"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20"},
    )
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    lines = citation_lines(warnings, out)
    assert lines == ["VG Y, Beschluss vom 01.01.2020 – 7 L 7/20"]


def test_parser_accepts_the_generated_line():
    from draft_citation_ingest import parse_decision_citations

    content = _content({"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20"})
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    line = citation_lines(warnings, out)[0]
    parsed = parse_decision_citations(line)
    assert parsed and parsed[0]["az"] == "7 L 7/20"

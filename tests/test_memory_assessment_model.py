"""Pure-logic tests for the case_assessment content models.

No DB, no containers. Run from the repo root:

    .venv/bin/python -m pytest tests/test_memory_assessment_model.py -q
"""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    default_case_assessment_json,
    strip_server_fields,
    validate_assessment_content,
)


def _gutachten(**overrides):
    base = {
        "id": "gueb-statt-duldung",
        "rechtsfrage": "Darf die ABH statt einer Duldung nur eine GUEB ausstellen?",
        "ergebnis": "Nein, wenn der Abschiebungszeitpunkt ungewiss ist.",
        "stand": "2026-09-03",
        "status": "aktiv",
        "pruefung": [
            {
                "these": "Kein Raum fuer ungeregelten Aufenthalt",
                "bewertung": "Traegt, solange die Behoerde keine Prognose belegt.",
                "fundstellen": ["18 E 491/12"],
            }
        ],
        "fundstellen": [
            {
                "gericht": "OVG NRW",
                "datum": "2012-06-18",
                "az": "18 E 491/12",
                "art": "Beschluss",
                "aussage": "Die GUEB ersetzt die Duldung nicht.",
                "richtung": "pro",
            }
        ],
        "risiken": ["Behoerde belegt eine zeitnahe Abschiebung."],
        "quelle": "Vermerk_2026-09-02_Recherche.pdf",
    }
    base.update(overrides)
    return base


def test_default_is_empty_and_valid():
    assert default_case_assessment_json() == {"gutachten": [], "notizen": ""}


def test_valid_content_round_trips():
    content = validate_assessment_content({"gutachten": [_gutachten()], "notizen": ""})
    assert content["gutachten"][0]["id"] == "gueb-statt-duldung"
    # server fields are materialized with their default
    assert content["gutachten"][0]["fundstellen"][0]["store"] == "unchecked"


def test_unknown_field_at_any_level_is_rejected():
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [_gutachten(unbekannt="x")], "notizen": ""})
    bad = _gutachten()
    bad["fundstellen"][0]["unbekannt"] = "x"
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [bad], "notizen": ""})


def test_invalid_slug_is_rejected():
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [_gutachten(id="GUEB Statt")], "notizen": ""})


def test_duplicate_id_is_rejected():
    with pytest.raises(ValueError):
        validate_assessment_content(
            {"gutachten": [_gutachten(), _gutachten()], "notizen": ""}
        )


def test_pruefung_reference_to_unknown_az_is_rejected():
    bad = _gutachten()
    bad["pruefung"][0]["fundstellen"] = ["9 X 1/99"]
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [bad], "notizen": ""})


def test_oversized_gutachten_is_rejected():
    huge = _gutachten(risiken=["x" * 300 for _ in range(10)])
    huge["pruefung"] = [
        {"these": "t" * 300, "bewertung": "b" * 800, "fundstellen": ["18 E 491/12"]}
        for _ in range(12)
    ]
    huge["fundstellen"] = [
        {
            "gericht": "OVG NRW",
            "datum": "2012-06-18",
            "az": "18 E 491/12",
            "aussage": "a" * 400,
            "richtung": "pro",
        }
        for _ in range(25)
    ]
    # still within per-field limits but over the 24 kB per-Gutachten cap
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [huge], "notizen": ""})


def test_strip_server_fields_removes_only_server_owned_keys():
    value = _gutachten()
    value["fundstellen"][0]["store"] = "verified"
    value["fundstellen"][0]["store_entry_id"] = "abc"
    cleaned = strip_server_fields(value)
    assert "store" not in cleaned["fundstellen"][0]
    assert cleaned["fundstellen"][0]["az"] == "18 E 491/12"


def test_oversized_content_is_rejected():
    # Build multiple valid Gutachten, each under 24 kB, that together exceed 200 kB
    entries = []
    for i in range(200):
        entry = _gutachten(
            id=f"gutachten-{i:03d}",
            rechtsfrage=f"Frage {i}: " + "x" * 200,
            ergebnis=f"Ergebnis {i}: " + "y" * 500,
        )
        entries.append(entry)

    # should raise ValueError due to exceeding MAX_CONTENT_BYTES
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": entries, "notizen": ""})

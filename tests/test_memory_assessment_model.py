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
    MAX_GUTACHTEN_BYTES,
    _json_size,
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
            # unique az per Fundstelle (index 0 keeps the id the pruefung
            # point above references) -- 25x the SAME az would now also
            # trip the duplicate-canonical-az check, which is not what
            # this test is about.
            "az": "18 E 491/12" if i == 0 else f"{i} X {i}/99",
            "aussage": "a" * 400,
            "richtung": "pro",
        }
        for i in range(25)
    ]
    # still within per-field limits but over the 24 kB per-Gutachten cap --
    # pin the message so this cannot pass on the duplicate-Az error instead.
    with pytest.raises(ValueError, match="überschreitet"):
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


def test_calendar_invalid_dates_are_rejected():
    """Final review 7a: die ISO-Regex allein laesst 2026-02-31 durch. Ein
    solches Datum kann der Store-Abgleich nie treffen und stuende im Prompt
    als echtes Entscheidungsdatum."""
    with pytest.raises(ValueError):
        validate_assessment_content(
            {"gutachten": [_gutachten(stand="2026-02-31")], "notizen": ""}
        )
    bad = _gutachten()
    bad["fundstellen"][0]["datum"] = "2012-13-01"
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [bad], "notizen": ""})
    leap = _gutachten()
    leap["fundstellen"][0]["datum"] = "2026-02-29"
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [leap], "notizen": ""})


def test_real_leap_day_is_accepted():
    ok = _gutachten()
    ok["fundstellen"][0]["datum"] = "2024-02-29"
    content = validate_assessment_content({"gutachten": [ok], "notizen": ""})
    assert content["gutachten"][0]["fundstellen"][0]["datum"] == "2024-02-29"


def test_size_limit_ignores_server_fields(gutachten_factory):
    """Brief's literal example pads Fundstelle.aussage to ~24 kB, which
    violates aussage's own max_length=400 before the size check is even
    reached. Pad with many max-length Fundstellen/Pruefungspunkte instead
    (each within its own field cap) to land just under the per-Gutachten
    budget -- close enough that the three server fields, once populated on
    all 25 Fundstellen, would tip a naive (unstripped) size measurement
    over the cap."""
    fundstellen = [
        {
            "gericht": "OVG NRW",
            "datum": "2012-06-18",
            "az": f"{i} X {i}/99",
            "art": "Beschluss",
            "aussage": "x" * 400,
            "richtung": "pro",
        }
        for i in range(25)
    ]
    pruefung = [
        {"these": "x" * 300, "bewertung": "y" * 800, "fundstellen": []}
        for _ in range(9)
    ]
    entry = gutachten_factory(
        "aa",
        rechtsfrage="Frage?",
        ergebnis="Antwort.",
        quelle="",
        risiken=[],
        pruefung=pruefung,
        fundstellen=fundstellen,
    )
    assert _json_size(strip_server_fields(entry)) < MAX_GUTACHTEN_BYTES
    validate_assessment_content({"gutachten": [entry], "notizen": ""})
    for f in entry["fundstellen"]:
        f.update(
            {
                "store": "verified",
                "store_entry_id": "e" * 36,
                "store_checked_at": "2026-09-07T12:00:00.000000",
            }
        )
    validate_assessment_content({"gutachten": [entry], "notizen": ""})  # darf nicht werfen


def test_duplicate_canonical_az_rejected(gutachten_factory):
    entry = gutachten_factory("aa")
    dup = dict(entry["fundstellen"][0])
    # "VG " strips cleanly to the same canonical Az; "OVG NRW " would not
    # (canonical_az only drops the court token, canonical_az("OVG NRW 18 E
    # 491/12") == "nrw18e491/12" != canonical_az("18 E 491/12")).
    dup["az"] = "VG " + dup["az"]
    entry["fundstellen"].append(dup)
    with pytest.raises(ValueError, match="doppelt"):
        validate_assessment_content({"gutachten": [entry], "notizen": ""})

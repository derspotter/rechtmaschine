"""Prompt block rendering, budget stages and the blocked-Az split.

    .venv/bin/python -m pytest tests/test_memory_assessment_render.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    render_assessment_block,
    render_assessment_for_wiki,
    validate_assessment_content,
    verified_az_whitelist,
)


def _entry(gid, stand, store="verified", risiken=None, pruefung=None):
    return {
        "id": gid,
        "rechtsfrage": f"Frage {gid}?",
        "ergebnis": f"Ergebnis {gid}.",
        "stand": stand,
        "status": "aktiv",
        "pruefung": pruefung
        if pruefung is not None
        else [
            {
                "these": f"These {gid}",
                "bewertung": f"Bewertung {gid}",
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
                "store": store,
            }
        ],
        "risiken": risiken if risiken is not None else [f"Risiko {gid}"],
        "quelle": "Vermerk.pdf",
    }


def _content(*entries):
    return validate_assessment_content({"gutachten": list(entries), "notizen": ""})


def test_verified_fundstelle_is_quotable():
    block, used, blocked = render_assessment_block(_content(_entry("aa", "2026-09-03")))
    assert "OVG NRW 18 E 491/12 (18.06.2012, pro)" in block
    assert used == ["aa"]
    assert blocked == []


def test_unverified_fundstelle_lands_in_the_blocklist_only():
    block, _, blocked = render_assessment_block(
        _content(_entry("aa", "2026-09-03", store="not_in_store"))
    )
    assert "Nicht zitierfähig" in block
    assert "18 E 491/12" in block
    assert "(18.06.2012, pro)" not in block
    assert blocked == ["18 E 491/12"]


def test_only_active_gutachten_are_rendered():
    overholt = _entry("bb", "2026-09-04")
    overholt["status"] = "ueberholt"
    _, used, _ = render_assessment_block(_content(_entry("aa", "2026-09-03"), overholt))
    assert used == ["aa"]


def test_newest_stand_first():
    _, used, _ = render_assessment_block(
        _content(_entry("alt", "2026-01-01"), _entry("neu", "2026-09-03"))
    )
    assert used == ["neu", "alt"]


def test_budget_drops_risiken_then_pruefung_then_whole_gutachten():
    entries = [_entry(f"g{i}", f"2026-0{i+1}-01") for i in range(4)]
    block, used, _ = render_assessment_block(_content(*entries), max_chars=300)
    assert "Risiko" not in block
    assert len(used) < 4
    assert "weitere Gutachten gekürzt" in block
    for gid in used:
        assert f"Frage {gid}?" in block
        assert f"Ergebnis {gid}." in block


def test_empty_content_renders_nothing():
    block, used, blocked = render_assessment_block(_content())
    assert block == ""
    assert used == []
    assert blocked == []


def test_render_for_wiki_returns_rendered_ids_and_whitelist_follows(gutachten_factory):
    """Brief's literal example sets ergebnis to 5000 'x' -- ergebnis's own
    max_length=800 rejects that before rendering is ever reached. Pad with
    risiken (each within its own 300-char cap) instead to make the "aa"
    block alone exceed the wiki budget, and give "aa" and "bb" distinct Az
    so an unfiltered whitelist would visibly differ from the filtered one."""
    big = gutachten_factory(
        "aa",
        ergebnis="x" * 800,
        risiken=["y" * 300 for _ in range(10)],
        pruefung=[],
        fundstellen=[
            {
                "gericht": "VG X",
                "datum": "2020-01-01",
                "az": "5 K 1/20",
                "art": "Beschluss",
                "aussage": "a",
                "richtung": "pro",
                "store": "verified",
            }
        ],
    )
    small = gutachten_factory("bb")
    small["fundstellen"][0]["store"] = "verified"
    content = {"gutachten": [big, small], "notizen": ""}

    text, ids = render_assessment_for_wiki(content, max_chars=3000)
    assert "GUTACHTEN aa" not in text and ids == {"bb"}
    assert verified_az_whitelist(content, only_ids=ids) == verified_az_whitelist(
        {"gutachten": [content["gutachten"][1]], "notizen": ""}
    )

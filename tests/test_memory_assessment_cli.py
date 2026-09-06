"""CLI projection and subcommands for case_assessment.

    .venv/bin/python -m pytest tests/test_memory_assessment_cli.py -q
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rm_cli", REPO / "scripts" / "rechtmaschine_cli.py"
)
cli = importlib.util.module_from_spec(SPEC)
sys.modules["rm_cli"] = cli
SPEC.loader.exec_module(cli)


def _payload():
    return {
        "case_brief": {"content_json": {"verfahrensstand": ["Etwas passiert."], "notizen": ""}},
        "case_strategy": {"content_json": {"kernstrategie": "Linie A", "offene_fragen": []}},
        "case_assessment": {
            "content_json": {
                "gutachten": [
                    {
                        "id": "gueb-statt-duldung",
                        "rechtsfrage": "GUEB statt Duldung?",
                        "ergebnis": "Nein.",
                        "stand": "2026-09-03",
                        "status": "aktiv",
                        "pruefung": [
                            {"these": "Kein ungeregelter Aufenthalt", "bewertung": "Traegt.",
                             "fundstellen": ["18 E 491/12"]}
                        ],
                        "fundstellen": [
                            {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12",
                             "aussage": "Ersetzt die Duldung nicht.", "richtung": "pro",
                             "store": "verified"}
                        ],
                        "risiken": [],
                        "quelle": "Vermerk.pdf",
                    }
                ],
                "notizen": "",
            }
        },
    }


def test_assessment_section_is_known():
    assert cli._MEMORY_SECTIONS["assessment"] == "case_assessment"


def test_entries_flatten_gutachten_for_grep():
    entries = cli._memory_entries(_payload(), "assessment")
    texts = " ".join(e["text"] for e in entries)
    assert "GUEB statt Duldung?" in texts
    assert "18 E 491/12" in texts
    assert "Kein ungeregelter Aufenthalt" in texts


def test_grep_finds_an_aktenzeichen():
    entries = cli._memory_entries(_payload(), None)
    hits = [e for e in entries if "18 E 491/12" in e["text"]]
    assert hits and hits[0]["section"] == "assessment"


def test_proposal_summary_line_for_gutachten_ops():
    op = {
        "op": "append",
        "path": "/gutachten/-",
        "value": {"id": "aa", "rechtsfrage": "F?", "fundstellen": [{"az": "1 A 1/24"}]},
    }
    assert cli._summarize_op(op) == "append /gutachten/-: aa | F? | 1 Fundstellen"


def test_recheck_subcommand_is_registered():
    parser = cli.build_parser()
    args = parser.parse_args(["memory", "assessment", "recheck", "--case-id", "x"])
    assert args.func is cli.cmd_memory_assessment_recheck

"""verify_facts: the blocked_az split (Task 6, fix round 1, Minor #2).

Blocked Az (Fundstellen the assessment's own store check marked "Nicht
zitierfähig") must not count as evidence even though they still appear in
the memory text (in the blocklist line) -- citing one in the draft must be
flagged, not silently accepted because the corpus happens to mention it.

Pure logic only -- no DB, no containers. citation_verifier has no heavy
top-level imports (fitz is imported lazily inside a function), so it is
imported directly, same as tests/test_citation_verifier.py. Run from the
repo root:

    .venv/bin/python -m pytest tests/test_memory_assessment_facts.py -q
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from citation_verifier import verify_facts  # noqa: E402

_AZ = "18 E 491/12"
_MEMORY_TEXT = (
    "RECHTLICHE WÜRDIGUNG DER KANZLEI (Fundstellen mit Store-Abgleich, Stand je Gutachten):\n"
    "[a1, Stand 03.09.2026] Rechtsfrage: Frage? Ergebnis: Ergebnis.\n"
    f"  Nicht zitierfähig (nicht im Bestand): {_AZ}"
)
_DRAFT = f"Das Gericht hat dies bereits entschieden ({_AZ})."


def test_blocked_az_in_draft_is_flagged_as_blocked_citation():
    result = verify_facts(
        _DRAFT,
        selected_documents={},
        memory_text=_MEMORY_TEXT,
        blocked_az={_AZ},
    )
    blocked = [c for c in result["fact_checks"] if c.get("status") == "blocked_citation"]
    assert len(blocked) == 1
    check = blocked[0]
    assert check["type"] == "aktenzeichen"
    assert check["status"] == "blocked_citation"
    assert check["severity"] == "high"


def test_same_az_without_blocked_az_produces_no_blocked_citation_check():
    result = verify_facts(
        _DRAFT,
        selected_documents={},
        memory_text=_MEMORY_TEXT,
    )
    blocked = [c for c in result["fact_checks"] if c.get("status") == "blocked_citation"]
    assert blocked == []


def test_blocked_bavarian_and_eugh_az_are_flagged():
    corpus = "Nicht zitierfähig: M 10 K 21.3767, C-151/22"
    res = verify_facts(
        "Vgl. M 10 K 21.3767 und EuGH C-151/22.", {}, memory_text=corpus,
        blocked_az={"m10k21.3767", "c-151/22"},
    )
    assert sorted(
        c["value"] for c in res["fact_checks"] if c["status"] == "blocked_citation"
    ) == ["C-151/22", "M 10 K 21.3767"]


def test_verify_facts_with_sources_forwards_blocklist_and_strips_blocklist_lines():
    from draft_context import verify_facts_with_sources

    memory = "Fakt: 01.02.2020\nNicht zitierfähig: 18 E 491/12\n"
    res = verify_facts_with_sources("Vgl. 18 E 491/12.", memory, [], blocked_az={"18e491/12"})
    assert res["fact_checks"][0]["status"] == "blocked_citation"


def test_blocked_check_runs_on_empty_corpus():
    res = verify_facts("Vgl. 18 E 491/12.", {}, memory_text="", blocked_az={"18e491/12"})
    assert res["fact_checks"] and res["fact_checks"][0]["status"] == "blocked_citation"

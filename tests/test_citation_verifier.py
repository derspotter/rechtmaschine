import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from citation_verifier import (  # noqa: E402
    AMBIGUOUS,
    FOUND_ON_DIFFERENT_PAGE,
    NO_PAGE_TEXT_AVAILABLE,
    NOT_FOUND,
    VERIFIED_ON_CITED_PAGE,
    verify_page_citations,
)


def _primary_bescheid(page_texts=None):
    entry = {
        "id": "doc-1",
        "filename": "bescheid.pdf",
        "category": "bescheid",
        "role": "primary",
    }
    if page_texts is not None:
        entry["page_texts"] = page_texts
    return {"bescheid": [entry], "internal_notes": []}


def test_verifies_claim_on_correct_cited_page():
    draft = "Der Kläger wurde am 1. Januar angehört (Anlage K2, S. 2)."
    selected = _primary_bescheid({
        1: "Andere Inhalte.",
        2: "Der Kläger wurde am 1. Januar angehört.",
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == VERIFIED_ON_CITED_PAGE
    assert result["checks"][0]["matched_pages"] == [2]


def test_finds_claim_on_different_page():
    draft = "Der Kläger wurde am 1. Januar angehört (Anlage K2, S. 2)."
    selected = _primary_bescheid({
        1: "Der Kläger wurde am 1. Januar angehört.",
        2: "Andere Inhalte.",
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == FOUND_ON_DIFFERENT_PAGE
    assert result["checks"][0]["matched_pages"] == [1]


def test_wrong_page_is_ambiguous_when_cited_page_has_near_fuzzy_evidence():
    draft = "Die Rückkehrgefährdung ergibt sich aus der Sicherheitslage (Anlage K2, S. 2)."
    selected = _primary_bescheid({
        1: "Die Rückkehrgefährdung ergibt sich aus der allgemeinen Sicherheitslage.",
        2: "Die Sicherheitslage wird allgemein dargestellt.",
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == AMBIGUOUS


def test_weak_fuzzy_cited_page_evidence_is_ambiguous_not_missing():
    draft = "Die Schwester lebt nach Darstellung des Bescheids in Syrien (Anlage K2, S. 2)."
    selected = _primary_bescheid({
        2: "In Deutschland leben mehrere Angehörige. Eine Schwester des Antragstellers wurde weiterhin in Syrien leben.",
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == AMBIGUOUS


def test_verifies_paraphrased_claim_with_shared_anchors():
    draft = "Der Antragsteller nahm am Integrationskurs teil und bestand die B1-Prüfung am 12.03.2026 (Anlage K2, S. 2)."
    selected = _primary_bescheid({
        2: "Teilnahmebescheinigung Integrationskurs. Deutsch-Test B1 bestanden am 12.03.2026.",
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == VERIFIED_ON_CITED_PAGE
    assert result["checks"][0]["match"]["method"] == "fuzzy_token_set"


def test_verifies_local_clause_in_long_sentence():
    draft = (
        "Der Bescheid enthält mehrere Ausführungen zur Identität des Antragstellers, "
        "außerdem ist die B1-Prüfung am 12.03.2026 bestanden worden (Anlage K2, S. 2)."
    )
    selected = _primary_bescheid({
        2: "Deutsch-Test B1: Prüfung bestanden am 12.03.2026.",
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == VERIFIED_ON_CITED_PAGE


def test_verifies_morphological_anchor_variants():
    draft = (
        "Familienmitglieder werden verfolgt, um gesuchte Personen einzuschüchtern "
        "oder Informationen über deren Verbleib zu erhalten (Anlage K2, S. 9)."
    )
    selected = _primary_bescheid({
        9: (
            "Die Verfolgung von Familienmitgliedern dient zur Druckausübung, "
            "Einschüchterung oder zum Erhalt von Informationen über den Verbleib "
            "gesuchter Personen."
        ),
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == VERIFIED_ON_CITED_PAGE
    assert result["checks"][0]["match"]["method"] == "fuzzy_token_set"


def test_verifies_short_claim_with_distinctive_rare_anchors():
    draft = (
        "Das Bundesamt beschreibt eine Übergangsregierung um den früheren "
        "HTS-Anführer Ahmad al-Shar'a (Anlage K2, S. 6)."
    )
    selected = _primary_bescheid({
        6: "Aktuelle Lage: Übergangsregierung um den HTS-Anführer Ahmad al-Shar'a.",
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == VERIFIED_ON_CITED_PAGE


def test_keeps_enumeration_tail_with_surrounding_clause():
    draft = (
        "Für Raqqa beschreibt der Bescheid fortdauernde Dynamiken zwischen SDF, "
        "Übergangsregierung, SNA, IS-Resten und lokalen Akteuren (Anlage K2, S. 9)."
    )
    selected = _primary_bescheid({
        9: (
            "Raqqa: fortdauernde Dynamiken zwischen SDF, Übergangsregierung, SNA, "
            "IS-Resten und lokalen Akteuren."
        ),
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == VERIFIED_ON_CITED_PAGE


def test_reports_missing_page_text():
    draft = "Der Kläger wurde am 1. Januar angehört (Anlage K2, S. 2)."
    selected = _primary_bescheid()

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == NO_PAGE_TEXT_AVAILABLE


def test_reports_missing_cited_page_text_even_when_other_pages_exist():
    draft = "Der Kläger wurde am 1. Januar angehört (Anlage K2, S. 2)."
    selected = _primary_bescheid({
        1: "Der Kläger wurde am 1. Januar angehört.",
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == NO_PAGE_TEXT_AVAILABLE
    assert result["checks"][0]["missing_cited_pages"] == [2]


def test_reports_not_found_only_for_reliable_hard_anchor_miss():
    draft = "Die B1-Prüfung wurde am 12.03.2026 bestanden (Anlage K2, S. 2)."
    selected = _primary_bescheid({
        2: "Die Seite enthält andere Angaben ohne das zitierte Prüfungsdatum.",
    })

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == NOT_FOUND


def test_warns_about_internal_note_citation():
    draft = "Die Aktennotiz vom 1. Januar bestätigt den Vortrag."
    selected = {
        "bescheid": [],
        "internal_notes": [
            {
                "id": "note-1",
                "filename": "Aktennotiz vom 01.01.2026.txt",
                "page_texts": {1: "Interner Inhalt."},
            }
        ],
    }

    result = verify_page_citations(draft, selected)

    assert result["warnings"]
    assert "interne Notizen" in result["warnings"][0]


def test_ignores_statutory_sentence_reference():
    draft = "Nach § 77 Abs. 2 AsylG, S. 2 ist die Sachlage maßgeblich."
    selected = _primary_bescheid({1: "Nachweise."})

    result = verify_page_citations(draft, selected)

    assert result["checks"] == []


def test_uses_original_page_range_from_segment_filename_for_blatt_citation():
    draft = "Der Kläger schilderte die Verfolgung wegen seiner Religion (Bl. 93 d.A.)."
    selected = {
        "anhoerung": [
            {
                "id": "doc-2",
                "filename": "Beiakte_Anhörung_p90-95.pdf",
                "page_texts": {
                    1: "Andere Seite.",
                    2: "Andere Seite.",
                    3: "Andere Seite.",
                    4: "Der Kläger schilderte die Verfolgung wegen seiner Religion.",
                },
            }
        ],
        "bescheid": [],
        "internal_notes": [],
    }

    result = verify_page_citations(draft, selected)

    assert result["checks"][0]["status"] == VERIFIED_ON_CITED_PAGE
    assert result["checks"][0]["matched_pages"] == [93]

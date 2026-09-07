"""Assessment-first FALL-SPEICHER and the Az whitelist for wiki entries.

    .venv/bin/python -m pytest tests/test_pattern_wiki_distill_assessment.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    render_assessment_for_wiki,
    validate_assessment_content,
    verified_az_whitelist,
)
from draft_citation_ingest import iter_decision_citations  # noqa: E402


def _content():
    return validate_assessment_content(
        {
            "gutachten": [
                {
                    "id": "aa",
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
                         "art": "Beschluss", "aussage": "Ersetzt die Duldung nicht.",
                         "richtung": "pro", "store": "verified"},
                        {"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20",
                         "art": "Beschluss", "aussage": "Unklar.", "richtung": "neutral",
                         "store": "not_in_store"},
                    ],
                    "risiken": [],
                    "quelle": "Vermerk.pdf",
                }
            ],
            "notizen": "",
        }
    )


def test_wiki_rendering_uses_the_parser_format_and_only_verified():
    text, rendered_ids = render_assessment_for_wiki(_content())
    assert rendered_ids == {"aa"}
    assert "OVG NRW, Beschluss vom 18.06.2012 – 18 E 491/12" in text
    assert "7 L 7/20" not in text
    parsed = list(iter_decision_citations(text))
    assert [p["az"] for p in parsed] == ["18 E 491/12"]


def test_whitelist_contains_only_verified_az():
    assert verified_az_whitelist(_content()) == {"18e491/12"}


def test_iter_yields_spans_that_slice_the_raw_citation():
    text = "Siehe OVG NRW, Beschluss vom 18.06.2012 – 18 E 491/12 dazu."
    hit = next(iter(iter_decision_citations(text)))
    assert text[hit["start"]:hit["end"]] == hit["raw"]
    assert hit["raw"].startswith("OVG NRW")


def test_strip_removes_foreign_citation_phrase_and_reports_it():
    from endpoints.pattern_wiki import strip_foreign_citations

    text = (
        "Argument traegt (OVG NRW, Beschluss vom 18.06.2012 – 18 E 491/12), "
        "anders VG Z, Urteil vom 01.02.2023 – 5 K 9/23."
    )
    cleaned, stripped = strip_foreign_citations(text, {"18e491/12"})
    assert "18 E 491/12" in cleaned
    assert "5 K 9/23" not in cleaned
    assert [s["az"] for s in stripped] == ["5 K 9/23"]


def test_strip_also_catches_a_bare_foreign_az():
    from endpoints.pattern_wiki import strip_foreign_citations

    cleaned, stripped = strip_foreign_citations("Vergleiche 5 K 9/23 hierzu.", {"18e491/12"})
    assert "5 K 9/23" not in cleaned
    assert stripped and stripped[0]["az"] == "5 K 9/23"


def test_whitelisted_bare_az_survives():
    from endpoints.pattern_wiki import strip_foreign_citations

    cleaned, stripped = strip_foreign_citations("Vergleiche 18 E 491/12 hierzu.", {"18e491/12"})
    assert "18 E 491/12" in cleaned
    assert stripped == []


def test_strip_foreign_citations_keeps_eu_directive_citation():
    from endpoints.pattern_wiki import strip_foreign_citations

    text = "Anspruch aus Art. 14 Abs. 2 RL 2008/115/EG und Art. 3 RL 2011/95/EU."
    cleaned, stripped = strip_foreign_citations(text, {"18e491/12"})
    assert cleaned == text
    assert stripped == []


def test_strip_foreign_citations_keeps_whitelisted_bavarian_prefix_az():
    from endpoints.pattern_wiki import strip_foreign_citations

    text = "(VG München, Beschluss vom 17.03.2022 – M 10 K 21.3767)"
    cleaned, stripped = strip_foreign_citations(text, {"m10k21.3767"})
    assert cleaned == text
    assert stripped == []


def test_strip_foreign_citations_strips_bavarian_prefix_with_az():
    from endpoints.pattern_wiki import strip_foreign_citations

    cleaned, stripped = strip_foreign_citations("(VG München, M 12 E 21.6201) trägt.", {"m10k21.3767"})
    assert "21.6201" not in cleaned
    assert "M )" not in cleaned and "M)" not in cleaned
    assert [s["az"] for s in stripped] == ["M 12 E 21.6201"]


def test_forbidden_tokens_ignore_eu_directive_citations():
    from types import SimpleNamespace
    from endpoints.pattern_wiki import _forbidden_tokens

    strategy = {"argumentationslinien": ["Anspruch aus Art. 14 Abs. 2 RL 2008/115/EG und Art. 3 VO (EU) Nr. 604/2013"]}
    tokens = _forbidden_tokens(SimpleNamespace(name="157/26 Testfall"), {"beteiligte": []}, strategy)
    assert not any("2008/115" in t or "604/2013" in t for t in tokens)


def test_entry_violations_exempt_bavarian_decision_citation():
    from endpoints.pattern_wiki import PatternWikiExtractionEntry, _entry_violations

    entry = PatternWikiExtractionEntry(
        title="Passvorlage heilt § 60b-Zusatz",
        argument_patterns=[
            "[bestätigt] Nach Vorlage des Passes entfällt der Zusatz (VGH Bayern, Beschluss vom 07.09.2022 – 10 ZB 22.1187).",
            "[bestätigt] GÜB ist keine Duldung (OVG NRW, Beschluss vom 18.06.2012 – 18 E 491/12).",
        ],
    )
    assert _entry_violations(entry, {"07.09.2022", "18.06.2012"}) == []
    assert _entry_violations(entry, {"07.09.2022", "Passvorlage"}) == ["Passvorlage"]

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


def test_strip_keeps_eu_norms_and_whitelisted_bavarian_az():
    from endpoints.pattern_wiki import strip_foreign_citations

    text = "Art. 3 RL 2011/95 und (VG München, Beschluss vom 17.03.2022 – M 10 K 21.3767)"
    cleaned, stripped = strip_foreign_citations(text, {"m10k21.3767"})
    assert cleaned == text and stripped == []


def test_strip_no_prefix_core_equivalence():
    from endpoints.pattern_wiki import strip_foreign_citations

    cleaned, stripped = strip_foreign_citations("(M 10 K 21.3767)", {"10k21.3767"})
    assert "21.3767" not in cleaned and stripped[0]["az"] == "M 10 K 21.3767"


def test_strip_handles_eugh_egmr_and_compact():
    from endpoints.pattern_wiki import strip_foreign_citations

    cleaned, stripped = strip_foreign_citations("EuGH C-151/22, EGMR Nr. 12345/19, 18E491/12", set())
    assert [s["az"] for s in stripped] == ["C-151/22", "Nr. 12345/19", "18E491/12"]


def test_entry_violations_name_inside_citation_span_is_still_caught():
    from endpoints.pattern_wiki import PatternWikiExtractionEntry, _entry_violations

    entry = PatternWikiExtractionEntry(title="x", argument_patterns=[
        "VG Teststadt, Frau Mustermann, Urteil vom 01.02.2020 – 18 E 491/12"])
    assert _entry_violations(entry, {"Mustermann"}) == ["Mustermann"]
    assert _entry_violations(entry, {"01.02.2020", "18 E 491/12"}) == []


def test_entry_violations_casefold():
    from endpoints.pattern_wiki import PatternWikiExtractionEntry, _entry_violations

    entry = PatternWikiExtractionEntry(title="mustermann klagt")
    assert _entry_violations(entry, {"Mustermann"}) == ["Mustermann"]


def test_forbidden_tokens_keep_client_name_when_memory_cites_an_eu_norm():
    """Die Normspannen-Sperre gilt nur den Treffern aus dem Blob -- die
    Namenswörter kommen aus case.name und stehen dort überhaupt nicht."""
    from types import SimpleNamespace
    from endpoints.pattern_wiki import _forbidden_tokens

    strategy = {"argumentationslinien": ["Anspruch aus Art. 3 RL 2011/95/EU"]}
    tokens = _forbidden_tokens(
        SimpleNamespace(name="157/26 Mustermann"), {"beteiligte": []}, strategy
    )
    assert "Mustermann" in tokens
    assert not any("2011/95" in t for t in tokens)


def test_forbidden_tokens_include_assessment_but_not_stand_or_decision_dates(gutachten_factory):
    from types import SimpleNamespace
    from endpoints.pattern_wiki import _forbidden_tokens

    entry = gutachten_factory("aa")
    entry["rechtsfrage"] = "Geburt am 18.10.1995, Az 9 K 1/26"
    tokens = _forbidden_tokens(
        SimpleNamespace(name="157/26 X"), {}, {},
        assessment_content={"gutachten": [entry], "notizen": ""},
    )
    assert "18.10.1995" in tokens and "9 K 1/26" in tokens


# --- Distill-Harness: Qwen und DB gestubbt, alles andere echt ----------------

def _distill(monkeypatch, entries, assessment_content=None):
    """Run _execute_pattern_wiki_distillation with a fake DB and a fake Qwen.

    Returns (result, rows) where rows are the PatternWikiEntry objects the
    job added."""
    import asyncio
    import uuid
    from types import SimpleNamespace

    import agent_memory_service as ams
    import citation_qwen
    import shared as shared_mod
    from endpoints import agent_memory as am
    from endpoints.pattern_wiki import _execute_pattern_wiki_distillation
    from models import PatternWikiEntry

    content = assessment_content if assessment_content is not None else _content()

    class _Query:
        def join(self, *a, **k):
            return self

        def filter(self, *a, **k):
            return self

        def all(self):
            return []

    class _DB:
        def __init__(self):
            self.added = []

        def query(self, *a, **k):
            return _Query()

        def add(self, row):
            self.added.append(row)

        def flush(self):
            for row in self.added:
                if getattr(row, "id", None) is None:
                    row.id = uuid.uuid4()

        def commit(self):
            pass

    monkeypatch.setenv("ANONYMIZATION_SERVICE_URL", "http://anonymization.invalid")
    monkeypatch.setattr(ams, "get_or_create_case_brief", lambda db, o, c: SimpleNamespace(content_json={}))
    monkeypatch.setattr(ams, "get_or_create_case_strategy", lambda db, o, c: SimpleNamespace(content_json={}))
    monkeypatch.setattr(
        ams, "get_or_create_case_assessment", lambda db, o, c: SimpleNamespace(content_json=content)
    )
    # Fall-Speicher muss über der 200-Zeichen-Schwelle liegen, geht aber nur
    # in den (gefakten) Qwen-Prompt.
    monkeypatch.setattr(ams, "render_case_brief_compact", lambda c: "Fallbrief: " + "Sachverhalt. " * 20)
    monkeypatch.setattr(ams, "render_case_strategy_compact", lambda c: "Fallstrategie: Duldung erzwingen.")

    async def _ready():
        return None

    monkeypatch.setattr(shared_mod, "ensure_anonymization_service_ready", _ready)
    monkeypatch.setattr(am, "_notify_memory_changed", lambda *a, **k: None)

    async def _fake_qwen(service_url, prompt, **kwargs):
        return {"entries": entries, "warnings": []}

    monkeypatch.setattr(citation_qwen, "call_qwen_json", _fake_qwen)

    db = _DB()
    result = asyncio.run(
        _execute_pattern_wiki_distillation(
            db, SimpleNamespace(id="owner-1"), "case-1", SimpleNamespace(name="157/26 Testfall")
        )
    )
    return result, [row for row in db.added if isinstance(row, PatternWikiEntry)]


def test_distill_strips_title_tags_fingerprint(monkeypatch):
    entry = {
        "title": "GÜB-Muster nach 5 K 9/23",
        "summary": "Trägt (OVG NRW, Beschluss vom 18.06.2012 – 18 E 491/12), anders 5 K 9/23.",
        "fingerprint": {"verfahrensgegenstand": "Duldung 5 K 9/23", "themen": ["Bezug auf 5 K 9/23"]},
        "tags": ["Linie 5 K 9/23"],
        "argument_patterns": ["Kein ungeregelter Aufenthalt."],
        "risk_patterns": [],
        "evidence_patterns": [],
        "recommended_next_steps": [],
        "confidence": 0.6,
    }
    result, rows = _distill(monkeypatch, [entry])

    assert result["created"] == 1
    row = rows[0]
    assert "9/23" not in row.title
    assert not any("9/23" in tag for tag in row.tags)
    assert "9/23" not in row.fingerprint["verfahrensgegenstand"]
    assert not any("9/23" in item for item in row.fingerprint["themen"])
    # Die verifizierte Fundstelle des Gutachtens bleibt stehen.
    assert "18 E 491/12" in row.summary

    # Buchführung: jede entfernte Fundstelle wird gemeldet, unter dem
    # bereinigten Titel -- der Titel wird zuerst gestrippt.
    reported = result["stripped_citations"]
    assert reported and all(
        set(item) == {"az", "citation", "entry_title"} for item in reported
    )
    assert {item["az"] for item in reported} == {"5 K 9/23"}
    assert {item["entry_title"] for item in reported} == {row.title}
    assert any("5 K 9/23" in warning for warning in result["warnings"]) is False
    assert f"{len(reported)} Fundstelle(n) außerhalb des Gutachtens entfernt" in result["warnings"]

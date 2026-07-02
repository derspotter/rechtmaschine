"""Tests for the structured, page-grounded Grok research output (Pillar 1).

Pure Python — no xai_sdk, no DB. Run:
    .venv/bin/python -m pytest tests/test_grok_structured.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from endpoints.research.structured import (  # noqa: E402
    StructuredSource,
    missing_grounding_fields,
)


def _decision_source(**overrides):
    base = dict(
        url="https://nrwe.justiz.nrw.de/ovgs/vg_duesseldorf/j2025/17_L_3613_25_A_Beschluss_20251104.html",
        title="VG Düsseldorf, Beschluss vom 04.11.2025 – 17 L 3613/25.A",
        description="§ 60 Abs. 5 AufenthG, Rückkehr Syrien",
        quelle_typ="entscheidung",
        gericht="Verwaltungsgericht Düsseldorf",
        datum="2025-11-04",
        aktenzeichen="17 L 3613/25.A",
        ebene="VG",
        ergebnis="abgelehnt",
        profil="gesunder Mann, 10 Jahre eigener Laden, volljähriger Sohn kehrt mit zurück",
        zitat="Weshalb er nicht an diese Berufstätigkeiten wird anknüpfen können, ist nicht ersichtlich",
        fit="§ 60 Abs. 5 Syrien-Rückkehrer, aber Netzwerk vorhanden",
        lager="gegen",
    )
    base.update(overrides)
    return StructuredSource(**base)


def test_decision_source_carries_grounding_fields():
    src = _decision_source()
    assert src.aktenzeichen == "17 L 3613/25.A"
    assert src.ergebnis == "abgelehnt"
    assert src.lager == "gegen"
    assert src.ebene == "VG"


def test_minimal_source_still_valid_for_backward_compat():
    # Old-style three-field sources (title/url/description) must keep validating,
    # so existing callers and cached payloads do not break.
    src = StructuredSource(
        url="https://example.org/x", title="T", description="D"
    )
    assert src.quelle_typ == "entscheidung"
    assert src.aktenzeichen is None


def test_invalid_lager_rejected():
    import pytest

    with pytest.raises(Exception):
        _decision_source(lager="vielleicht")


def test_missing_grounding_fields_lists_gaps_for_decisions():
    src = _decision_source(aktenzeichen=None, zitat=None)
    missing = missing_grounding_fields(src)
    assert "aktenzeichen" in missing
    assert "zitat" in missing
    assert "gericht" not in missing


def test_grounded_decision_has_no_gaps():
    assert missing_grounding_fields(_decision_source()) == []


def test_coi_sources_need_no_decision_grounding():
    src = StructuredSource(
        url="https://coi.euaa.europa.eu/syria-country-focus.pdf",
        title="EUAA COI Syria Country Focus 2025",
        description="Sicherheits- und Versorgungslage",
        quelle_typ="coi",
    )
    assert missing_grounding_fields(src) == []


# ---------------------------------------------------------------------------
# Citation gating: a source may only be emitted if grok actually retrieved its
# URL via web_search (response.citations). Grok prose is not trusted.
# ---------------------------------------------------------------------------
from endpoints.research.structured import (  # noqa: E402
    gate_sources_by_citations,
    normalize_citation_url,
)


def test_url_normalization_tolerates_cosmetic_differences():
    assert normalize_citation_url("http://openjur.de/u/2545074.html/") == \
        normalize_citation_url("https://openjur.de/u/2545074.html")
    assert normalize_citation_url("https://nrwe.justiz.nrw.de/x?a=1#gruende") == \
        normalize_citation_url("https://nrwe.justiz.nrw.de/x?a=1")


def test_url_normalization_keeps_meaningful_differences():
    assert normalize_citation_url("https://openjur.de/u/1.html") != \
        normalize_citation_url("https://openjur.de/u/2.html")
    assert normalize_citation_url("https://a.de/x?az=17L3613") != \
        normalize_citation_url("https://a.de/x?az=27K4231")


def test_gate_keeps_only_sources_grok_actually_retrieved():
    retrieved = [
        "https://nrwe.justiz.nrw.de/decision-a.html",
        "https://www.asyl.net/rsdb/m33861/",
    ]
    kept_src = _decision_source(url="http://nrwe.justiz.nrw.de/decision-a.html/")
    hallucinated = _decision_source(url="https://openjur.de/u/9999999.html")
    coi = StructuredSource(
        url="https://www.asyl.net/rsdb/m33861",
        title="asyl.net Eintrag", description="", quelle_typ="coi",
    )

    kept, dropped = gate_sources_by_citations(
        [kept_src, hallucinated, coi], retrieved
    )

    assert [s.url for s in kept] == [kept_src.url, coi.url]
    assert [s.url for s in dropped] == [hallucinated.url]


def test_gate_with_no_citations_drops_nothing_but_flags():
    # Some SDK responses may omit citations entirely; in that case gating is
    # inapplicable and everything passes through (fail-open, verifier catches it).
    src = _decision_source()
    kept, dropped = gate_sources_by_citations([src], [])
    assert kept == [src]
    assert dropped == []


# ---------------------------------------------------------------------------
# Prompt contract: the grounding rules the model must see. If these phrases
# leave the prompt, page-grounding silently degrades — hence contract tests.
# ---------------------------------------------------------------------------
from endpoints.research.prompting import build_research_priority_prompt  # noqa: E402


def test_grounded_prompt_contains_grounding_contract():
    prompt = build_research_priority_prompt(grounded=True).lower()
    # Az/Datum/Ergebnis verbatim from the OPENED page, never from memory:
    assert "wörtlich" in prompt
    assert "geöffneten" in prompt
    # Un-openable / un-quotable decisions must be omitted:
    assert "weglassen" in prompt or "weggelassen" in prompt
    # Outcome and camp are mandatory (contrary findings welcome but labelled):
    assert "ergebnis" in prompt
    assert "lager" in prompt or "stuetzt" in prompt
    # Honest negative result is a valid answer:
    assert "keine passende entscheidung" in prompt or "negativergebnis" in prompt


def test_ungrounded_prompt_unchanged_for_other_engines():
    # gemini/openai paths call the builder without the flag — their prompt
    # must not suddenly carry grok-schema field names.
    prompt = build_research_priority_prompt().lower()
    assert "lager" not in prompt
    assert "negativergebnis" not in prompt


# ---------------------------------------------------------------------------
# Round engine: SDK-free loop that grok.py wires to the real client. A fake
# chat exercises parse + citation-gating + dedup + early-stop behavior.
# ---------------------------------------------------------------------------
import asyncio  # noqa: E402

from endpoints.research.structured import (  # noqa: E402
    GrokResearchOutput,
    run_structured_research_rounds,
)


class _FakeResponse:
    def __init__(self, citations):
        self.citations = citations


class _FakeChat:
    def __init__(self, parsed, citations, log):
        self._parsed = parsed
        self._citations = citations
        self._log = log

    def append(self, message):
        self._log.append(("message", message))

    def parse(self, model_cls):
        assert model_cls is GrokResearchOutput
        return _FakeResponse(self._citations), self._parsed


def _run_engine(rounds, max_rounds=4, max_duration_sec=999):
    """rounds: list of (GrokResearchOutput, citations). Returns (result, log)."""
    log = []

    def create_chat(round_index):
        log.append(("round", round_index))
        parsed, citations = rounds[min(round_index, len(rounds) - 1)]
        return _FakeChat(parsed, citations, log)

    result = asyncio.run(
        run_structured_research_rounds(
            create_chat=create_chat,
            make_user_message=lambda text: text,
            base_message="BASE",
            max_rounds=max_rounds,
            max_duration_sec=max_duration_sec,
        )
    )
    return result, log


def _out(*sources, summary="S"):
    return GrokResearchOutput(summary=summary, sources=list(sources))


def test_engine_gates_ungrounded_sources_and_keeps_retrieved():
    real = _decision_source(url="https://nrwe.justiz.nrw.de/real.html")
    fake = _decision_source(url="https://openjur.de/u/999.html")
    result, _ = _run_engine(
        [(_out(real, fake), ["https://nrwe.justiz.nrw.de/real.html"])],
        max_rounds=1,
    )
    assert [s.url for s in result["sources"]] == [real.url]
    assert [s.url for s in result["dropped"]] == [fake.url]


def test_engine_dedups_across_rounds_and_stops_on_low_gain():
    a = _decision_source(url="https://a.de/1")
    round1 = (_out(a), ["https://a.de/1"])
    round2 = (_out(_decision_source(url="http://a.de/1/")), ["https://a.de/1"])  # same, cosmetic
    result, log = _run_engine([round1, round2], max_rounds=4)
    assert len(result["sources"]) == 1
    rounds_run = [r for kind, r in log if kind == "round"]
    assert rounds_run == [0, 1]  # early stop: round 2 brought <2 new sources


def test_engine_tells_later_rounds_what_it_already_has():
    a = _decision_source(url="https://a.de/1")
    b = _decision_source(url="https://b.de/2")
    c = _decision_source(url="https://c.de/3")
    result, log = _run_engine(
        [(_out(a, b), ["https://a.de/1", "https://b.de/2"]),
         (_out(c), ["https://c.de/3"])],
        max_rounds=2,
    )
    round2_messages = [m for kind, m in log if kind == "message"][1]
    assert "https://a.de/1" in round2_messages
    assert result["rounds"] == 2


def test_engine_keeps_longest_summary():
    a = _decision_source(url="https://a.de/1")
    b = _decision_source(url="https://b.de/2")
    c = _decision_source(url="https://c.de/3")
    result, _ = _run_engine(
        [(_out(a, b, summary="kurz"), ["https://a.de/1", "https://b.de/2"]),
         (_out(c, summary="deutlich ausführlicher"), ["https://c.de/3"])],
        max_rounds=2,
    )
    assert result["summary"] == "deutlich ausführlicher"


# ---------------------------------------------------------------------------
# The ranker rebuilds source dicts key-by-key; the page-grounding block must
# survive it, or Az/Ergebnis/Zitat silently vanish before the verifier runs.
# ---------------------------------------------------------------------------
from endpoints.research.source_quality import normalize_source_entry  # noqa: E402


def test_ranker_preserves_grounding_block():
    entry = normalize_source_entry(
        {
            "url": "https://nrwe.justiz.nrw.de/real.html",
            "title": "VG Düsseldorf, 17 L 3613/25.A",
            "description": "x",
            "grounding": {
                "aktenzeichen": "17 L 3613/25.A",
                "ergebnis": "abgelehnt",
                "lager": "gegen",
                "zitat": "wörtliche Passage",
            },
        },
        provider="Grok",
    )
    assert entry["grounding"]["aktenzeichen"] == "17 L 3613/25.A"
    assert entry["grounding"]["lager"] == "gegen"


# ---------------------------------------------------------------------------
# SDK 1.11 returns structured output as response.content (response_format=
# model at create-time; no chat.parse method). The content parser must accept
# clean JSON and defensively strip code fences.
# ---------------------------------------------------------------------------
from endpoints.research.structured import parse_structured_content  # noqa: E402


def test_parse_structured_content_plain_json():
    out = parse_structured_content(
        '{"summary": "S", "sources": []}', GrokResearchOutput
    )
    assert out.summary == "S"


def test_parse_structured_content_fenced_json():
    out = parse_structured_content(
        'Vorwort\n```json\n{"summary": "F", "sources": []}\n```\nNachwort',
        GrokResearchOutput,
    )
    assert out.summary == "F"


# ---------------------------------------------------------------------------
# Stress-test findings (2026-07-01): a mid-round failure must not destroy the
# results of earlier rounds, and default ports must not defeat URL gating.
# ---------------------------------------------------------------------------

def test_engine_survives_round_exception_and_keeps_prior_results():
    good = _decision_source(url="https://a.de/1")

    class OkChat:
        def append(self, m): pass
        def parse(self, cls):
            class R:
                citations = ["https://a.de/1"]
            return R(), _out(good)

    class BoomChat:
        def append(self, m): pass
        def parse(self, cls):
            raise ValueError("schema violation from API")

    log = []

    def create_chat(i):
        log.append(i)
        return OkChat() if i == 0 else BoomChat()

    result = asyncio.run(
        run_structured_research_rounds(
            create_chat=create_chat, make_user_message=str,
            base_message="B", max_rounds=3, max_duration_sec=999,
        )
    )
    assert [s.url for s in result["sources"]] == [good.url]
    assert result["errors"], "round error must be recorded, not swallowed silently"
    # after a failed round there is no point hammering the API again
    assert log == [0, 1]


def test_url_normalization_strips_default_ports():
    assert normalize_citation_url("https://a.de:443/x") == normalize_citation_url("https://a.de/x")
    assert normalize_citation_url("http://a.de:80/x") == normalize_citation_url("http://a.de/x")
    # non-default port stays significant
    assert normalize_citation_url("https://a.de:8443/x") != normalize_citation_url("https://a.de/x")

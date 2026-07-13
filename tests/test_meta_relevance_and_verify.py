"""Tests for the 2026-07-13 meta fixes: truncation-tolerant relevance parsing,
model fallback, and reduced verification for meta/web sources.

Encodes the live failure: gpt-5.6-sol's answer hit the 2400-token ceiling,
the JSON array was cut mid-object, and the whole evaluation silently fell
back to unranked (25 unfiltered sources instead of a curated set).

Run: .venv/bin/python -m pytest tests/test_meta_relevance_and_verify.py -q
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from endpoints.research.retrieval import FetchResult  # noqa: E402
from endpoints.research.verify import verify_meta_sources  # noqa: E402
from endpoints.research.store_writeback import grounding_to_extraction_fields  # noqa: E402
from endpoints.research import meta  # noqa: E402


def _item(i, score=7):
    return {"index": i, "score": score, "reasoning": "passt", "keep": True,
            "match_notes": [], "outcome_relevance": "claimant_positive",
            "authority_level": "high"}


# --- _extract_json_array: parsing and salvage ---

def test_extract_json_array_plain_and_fenced():
    arr = [_item(0), _item(1)]
    assert meta._extract_json_array(json.dumps(arr)) == arr
    assert meta._extract_json_array(f"```json\n{json.dumps(arr)}\n```") == arr


def test_extract_json_array_salvages_truncated_output():
    """Token-limit truncation mid-object keeps every complete item."""
    arr = [_item(0), _item(1), _item(2)]
    full = json.dumps(arr, indent=1)
    truncated = full[: full.rindex('"reasoning"')]  # cut inside item 3
    salvaged = meta._extract_json_array("Here is the ranking:\n" + truncated)
    assert salvaged == arr[:2]


def test_extract_json_array_no_array_returns_none():
    assert meta._extract_json_array("Ich kann leider kein JSON liefern.") is None
    assert meta._extract_json_array("") is None


# --- evaluate_relevance: fallback model ---

def test_evaluate_relevance_falls_back_to_second_model(monkeypatch):
    calls = []

    async def fake_call(prompt, sources_text, model):
        calls.append(model)
        if len(calls) == 1:
            return "REASONING ONLY — no json"  # primary unparseable
        return json.dumps([_item(0, score=9)])

    monkeypatch.setattr(meta, "_call_meta_relevance_model", fake_call)
    monkeypatch.setattr(meta, "META_RELEVANCE_FALLBACK_MODEL", "claude-opus-4-8")
    sources = [{"title": "VG X", "url": "https://x", "court": "VG X",
                "case_number": "1 K 1/26", "evidence_type": "decision_like"}]
    ranked, discarded = asyncio.run(meta.evaluate_relevance("query", sources))
    assert len(calls) == 2 and calls[1] == "claude-opus-4-8"
    assert len(ranked) == 1 and ranked[0]["relevance_score"] == 9


def test_evaluate_relevance_both_models_fail_uses_unranked_fallback(monkeypatch):
    async def fake_call(prompt, sources_text, model):
        return "kaputt"

    monkeypatch.setattr(meta, "_call_meta_relevance_model", fake_call)
    sources = [{"title": "A", "url": "https://a", "evidence_type": "decision_like"}]
    ranked, discarded = asyncio.run(meta.evaluate_relevance("query", sources))
    assert len(ranked) == 1  # graceful degradation keeps sources
    assert "Fallback" in ranked[0]["relevance_reason"]


# --- verify_meta_sources: reduced Az-level verification ---

def _fetch_fn(pages):
    async def fetch(url):
        return pages[url]
    return fetch


def test_verify_meta_sources_marks_az_hits_and_misses():
    sources = [
        {"title": "NRWE", "url": "https://nrwe/1", "court": "VG Köln",
         "case_number": "20 K 2991/24", "evidence_type": "decision_like"},
        {"title": "falsch", "url": "https://nrwe/2", "court": "VG Köln",
         "case_number": "9 L 999/99", "evidence_type": "decision_like"},
    ]
    pages = {
        "https://nrwe/1": FetchResult(status="ok", resolved_url="https://nrwe/1",
                                      text="Urteil VG Köln, 20 K 2991/24, Tenor ..."),
        "https://nrwe/2": FetchResult(status="ok", resolved_url="https://nrwe/2",
                                      text="Ganz andere Entscheidung 1 K 1/20"),
    }
    stats = asyncio.run(verify_meta_sources(sources, _fetch_fn(pages)))
    assert stats == {"verified": 1, "unverified": 1, "skipped": 0}
    assert sources[0]["grounding"]["verifiziert"] is True
    assert sources[0]["grounding"]["verify_level"] == "aktenzeichen"
    assert sources[1]["grounding"]["verifiziert"] is False
    assert "nicht auf der Seite" in sources[1]["grounding"]["verify_notes"]


def test_verify_meta_sources_skips_structured_and_azless_sources():
    sources = [
        {"title": "grok", "url": "https://g", "case_number": "1 K 1/26",
         "grounding": {"verifiziert": True}},           # structured: skip
        {"title": "artikel", "url": "https://t"},        # no Az: skip
    ]
    stats = asyncio.run(verify_meta_sources(sources, _fetch_fn({})))
    assert stats == {"verified": 0, "unverified": 0, "skipped": 2}
    assert sources[0]["grounding"] == {"verifiziert": True}  # untouched


def test_verify_meta_sources_fetch_error_is_unverified_not_crash():
    async def boom(url):
        raise OSError("timeout")
    sources = [{"title": "x", "url": "https://x", "court": "VG",
                "case_number": "1 K 1/26"}]
    stats = asyncio.run(verify_meta_sources(sources, boom))
    assert stats["unverified"] == 1
    assert sources[0]["grounding"]["verifiziert"] is False


def test_az_only_grounding_never_enters_the_store():
    """Reduced verification must not qualify for store writeback."""
    source = {"grounding": {"quelle_typ": "entscheidung", "gericht": "VG Köln",
                            "aktenzeichen": "20 K 2991/24", "verifiziert": True,
                            "verify_level": "aktenzeichen"}}
    assert grounding_to_extraction_fields(source, "Syrien") is None

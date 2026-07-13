"""verify_qwen: Qwen-Verifikation der Research-Quellen (Spec 2026-07-13).

Pure Tests mit injizierten Qwen-Antworten — kein Live-Call, kein Netz.
Fail-Richtung ist überall closed: was Qwen nicht positiv bestätigt,
bekommt kein verifiziert=True und damit keinen Store-Write-back.

Run: .venv/bin/python -m pytest tests/test_research_verify_qwen.py -q
"""
import asyncio

import pytest

from endpoints.research.retrieval import FetchResult
from endpoints.research.verify_qwen import (
    build_verify_prompt,
    parse_verdict,
    select_page_excerpt,
    verify_source_qwen,
)

ZITAT = (
    "Ob ein Rückkehrer nach Syrien wegen der dortigen schlechten humanitären "
    "Lage mit beachtlicher Wahrscheinlichkeit einen ernsthaften Schaden erleidet"
)

PAGE = (
    "Verwaltungsgericht Köln, Urteil vom 03.09.2025 — 27 K 4231/25.A\n"
    "Tenor: Die Klage wird abgewiesen.\n"
    "Gründe: ... " + ZITAT + " ...\n"
)


def _grounding(**overrides):
    g = {
        "quelle_typ": "entscheidung",
        "gericht": "VG Köln",
        "datum": "03.09.2025",
        "aktenzeichen": "27 K 4231/25.A",
        "ergebnis": "abgelehnt",
        "zitat": ZITAT,
        "lager": "gegen",
    }
    g.update(overrides)
    return g


def _fetch(text=PAGE, status="ok", ocr=False):
    return FetchResult(
        status=status, resolved_url="https://nrwe.example/x", text=text, ocr_applied=ocr
    )


def _verdict_json(**overrides):
    v = {
        "aktenzeichen_bestaetigt": True,
        "zitat_bestaetigt": True,
        "ergebnis_bestaetigt": True,
        "lager_plausibel": True,
        "verifiziert": True,
        "confidence": 0.92,
        "begruendung": "Tenor weist die Klage ab, Zitat steht wörtlich in den Gründen.",
    }
    v.update(overrides)
    return v


def _qwen_stub(verdict_json):
    calls = []

    async def call(prompt: str):
        calls.append(prompt)
        return verdict_json

    call.calls = calls
    return call


def _run(coro):
    return asyncio.run(coro)


# --- select_page_excerpt ------------------------------------------------------


def test_excerpt_short_page_passes_through():
    assert select_page_excerpt(PAGE, ZITAT, limit=20_000) == PAGE


def test_excerpt_long_page_keeps_head_and_quote_region():
    long_page = PAGE + ("Füllmaterial. " * 8000) + "IM HINTEREN TEIL: " + ZITAT + " Ende."
    excerpt = select_page_excerpt(long_page, ZITAT, limit=12_000)
    assert len(excerpt) <= 13_000
    assert "Tenor: Die Klage wird abgewiesen." in excerpt  # Kopf bleibt
    assert "IM HINTEREN TEIL" in excerpt  # Zitat-Region kommt mit


# --- build_verify_prompt ------------------------------------------------------


def test_prompt_contains_claim_hints_and_excerpt():
    det = {"aktenzeichen": True, "zitat": False}
    prompt = build_verify_prompt(_grounding(), PAGE, det)
    assert "27 K 4231/25.A" in prompt
    assert ZITAT[:40] in prompt
    assert "abgelehnt" in prompt
    assert "gegen" in prompt
    assert "Tenor: Die Klage wird abgewiesen." in prompt
    # Deterministische String-Befunde als Fakten-Anker
    assert "String-Suche" in prompt


# --- parse_verdict ------------------------------------------------------------


def test_parse_verdict_happy_path():
    verdict = parse_verdict(_verdict_json())
    assert verdict.verifiziert is True
    assert verdict.checks == {
        "aktenzeichen": True,
        "zitat": True,
        "ergebnis": True,
        "lager": True,
    }
    assert "Tenor" in verdict.begruendung


def test_parse_verdict_requires_all_core_checks():
    # Qwen behauptet verifiziert, bestätigt aber das Ergebnis nicht → gate zu.
    verdict = parse_verdict(_verdict_json(ergebnis_bestaetigt="unklar"))
    assert verdict.verifiziert is False
    assert verdict.checks["ergebnis"] is None


def test_parse_verdict_garbage_raises():
    with pytest.raises(ValueError):
        parse_verdict({"quatsch": 1})
    with pytest.raises(ValueError):
        parse_verdict(None)


# --- verify_source_qwen -------------------------------------------------------


def test_fetch_failure_is_unverified_without_qwen_call():
    qwen = _qwen_stub(_verdict_json())
    result = _run(verify_source_qwen(_grounding(), _fetch(status="blocked"), qwen))
    assert result.verifiziert is False
    assert "nicht verifizierbar" in result.notes
    assert qwen.calls == []


def test_coi_source_needs_reachability_only():
    qwen = _qwen_stub(_verdict_json())
    result = _run(verify_source_qwen(_grounding(quelle_typ="coi"), _fetch(), qwen))
    assert result.verifiziert is True
    assert qwen.calls == []


def test_confirmed_verdict_verifies_source():
    qwen = _qwen_stub(_verdict_json())
    result = _run(verify_source_qwen(_grounding(), _fetch(), qwen))
    assert result.verifiziert is True
    assert "Tenor" in result.notes
    assert len(qwen.calls) == 1


def test_refuted_ergebnis_fails_closed():
    qwen = _qwen_stub(
        _verdict_json(ergebnis_bestaetigt=False, verifiziert=False,
                      begruendung="Tenor gibt der Klage statt, Behauptung sagt abgelehnt.")
    )
    result = _run(verify_source_qwen(_grounding(), _fetch(), qwen))
    assert result.verifiziert is False
    assert "statt" in result.notes


def test_qwen_exception_fails_closed():
    async def broken(prompt):
        raise RuntimeError("service down")

    result = _run(verify_source_qwen(_grounding(), _fetch(), broken))
    assert result.verifiziert is False
    assert "Qwen-Verifikation fehlgeschlagen" in result.notes


def test_implausible_lager_is_removed_and_noted():
    grounding = _grounding()
    qwen = _qwen_stub(_verdict_json(lager_plausibel=False))
    result = _run(verify_source_qwen(grounding, _fetch(), qwen))
    assert result.verifiziert is True  # lager gated die Verifikation nicht
    assert "lager" not in grounding
    assert "lager" in result.notes.lower()


# --- grok.py-Wiring (Backend-Auswahl + Unavailable-Pfad) ----------------------


def test_grok_backend_defaults_to_qwen(monkeypatch):
    from endpoints.research import grok

    monkeypatch.delenv("RESEARCH_VERIFY_BACKEND", raising=False)
    assert grok._verify_backend() == "qwen"
    monkeypatch.setenv("RESEARCH_VERIFY_BACKEND", "deterministic")
    assert grok._verify_backend() == "deterministic"


def test_grok_qwen_unavailable_fails_closed(monkeypatch):
    # Desktop nicht weckbar → jede Quelle unverifiziert mit Notiz, kein
    # stiller Fallback auf den deterministischen Gate.
    import shared
    from endpoints.research import grok

    async def boom():
        raise RuntimeError("kein wake")

    monkeypatch.setattr(shared, "ensure_anonymization_service_ready", boom)
    verify_fn = _run(grok._prepare_qwen_verify_fn())
    result = _run(verify_fn(_grounding(), _fetch()))
    assert result.verifiziert is False
    assert "nicht verfügbar" in result.notes


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

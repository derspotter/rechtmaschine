"""Tests for the research retrieval module (Pillar 2).

Fixtures under tests/fixtures/retrieval/ are REAL pages captured 2026-07-01:
  - nrwe_decision.html          VG Düsseldorf 17 L 3613/25.A (full decision)
  - gesetzebayern_decision.html VG Regensburg RN 11 K 25.33928 (full decision)
  - asylnet_entry.html          asyl.net rsdb M33861 (entry w/ Leitsatz + Az)
  - voris_403.html              voris — serves full decision to a browser UA
                                despite login/purchase chrome (VG Osnabrück 7 B 19/26)
  - openjur_captcha.html        openJur JS slider wall: tiny page, NO 'captcha'
                                keyword, no decision content
  - asylnet_decision.pdf        20-page decision PDF (17 L 3613/25)

Run: .venv/bin/python -m pytest tests/test_retrieval.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import httpx  # noqa: E402
import pytest  # noqa: E402

from endpoints.research.retrieval import (  # noqa: E402
    FetchResult,
    classify_page,
    extract_pdf_text,
    extract_visible_text,
    fetch_source,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "retrieval")


def _fixture(name: str) -> bytes:
    with open(os.path.join(FIXTURES, name), "rb") as fh:
        return fh.read()


def _fixture_text(name: str) -> str:
    return _fixture(name).decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Visible-text extraction
# ---------------------------------------------------------------------------

def test_visible_text_strips_markup_but_keeps_decision_content():
    text = extract_visible_text(_fixture_text("nrwe_decision.html"))
    assert "17 L 3613/25" in text
    assert "Tenor" in text
    assert "<div" not in text
    assert "function" not in text.lower()[:2000]  # no JS leaked


# ---------------------------------------------------------------------------
# Classification — the heart of Pillar 2: walls must stop masquerading as
# successful fetches, and real decisions must classify ok even with login
# chrome around them.
# ---------------------------------------------------------------------------

def test_real_decisions_classify_ok():
    for name in ("nrwe_decision.html", "gesetzebayern_decision.html", "asylnet_entry.html"):
        status = classify_page(200, extract_visible_text(_fixture_text(name)))
        assert status == "ok", f"{name} -> {status}"


def test_voris_with_login_chrome_but_full_decision_is_ok():
    # Login/Kaufen buttons in the chrome must NOT outweigh actual decision text.
    status = classify_page(200, extract_visible_text(_fixture_text("voris_403.html")))
    assert status == "ok"


def test_openjur_js_wall_is_blocked_despite_http_200():
    # 3.6k page, no Az, no Tenor/Gründe, no 'captcha' keyword anywhere.
    status = classify_page(200, extract_visible_text(_fixture_text("openjur_captcha.html")))
    assert status == "blocked"


def test_http_errors_classify_by_status_code():
    assert classify_page(404, "irrelevant") == "notfound"
    assert classify_page(410, "irrelevant") == "notfound"
    assert classify_page(403, "") == "blocked"
    assert classify_page(429, "") == "blocked"


def test_long_prose_without_decision_signals_is_not_decision():
    prose = "Die Lage in Syrien ist weiterhin angespannt. " * 200
    assert classify_page(200, prose) == "not_decision"


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def test_pdf_text_extraction_contains_aktenzeichen():
    text = extract_pdf_text(_fixture("asylnet_decision.pdf"))
    assert "17 L 3613/25" in text


# ---------------------------------------------------------------------------
# fetch_source — transport injected via httpx.MockTransport, no network.
# ---------------------------------------------------------------------------

def _client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


@pytest.mark.anyio
async def test_fetch_source_html_decision_ok():
    async def handler(request):
        return httpx.Response(200, content=_fixture("nrwe_decision.html"),
                              headers={"content-type": "text/html; charset=utf-8"})

    async with _client_for(handler) as client:
        result = await fetch_source("https://nrwe.justiz.nrw.de/x.html", client=client)
    assert isinstance(result, FetchResult)
    assert result.status == "ok"
    assert "17 L 3613/25" in result.text
    assert result.resolved_url.endswith("/x.html")


@pytest.mark.anyio
async def test_fetch_source_detects_pdf_and_extracts_text():
    async def handler(request):
        return httpx.Response(200, content=_fixture("asylnet_decision.pdf"),
                              headers={"content-type": "application/pdf"})

    async with _client_for(handler) as client:
        result = await fetch_source("https://asyl.net/x.pdf", client=client)
    assert result.status == "ok"
    assert "17 L 3613/25" in result.text
    assert result.is_pdf


@pytest.mark.anyio
async def test_fetch_source_wall_reports_blocked_not_success():
    async def handler(request):
        return httpx.Response(200, content=_fixture("openjur_captcha.html"),
                              headers={"content-type": "text/html"})

    async with _client_for(handler) as client:
        result = await fetch_source("https://openjur.de/u/999.html", client=client)
    assert result.status == "blocked"
    assert result.text == ""


@pytest.mark.anyio
async def test_fetch_source_follows_redirects_and_reports_resolved_url():
    async def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200, content=_fixture("nrwe_decision.html"),
                              headers={"content-type": "text/html"})

    async with _client_for(handler) as client:
        result = await fetch_source("https://nrwe.justiz.nrw.de/old", client=client)
    assert result.status == "ok"
    assert result.resolved_url.endswith("/new")


@pytest.mark.anyio
async def test_fetch_source_network_error_is_error_status():
    async def handler(request):
        raise httpx.ConnectError("boom")

    async with _client_for(handler) as client:
        result = await fetch_source("https://unreachable.example/x", client=client)
    assert result.status == "error"


@pytest.fixture
def anyio_backend():
    return "asyncio"

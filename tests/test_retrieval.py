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

def _pdf_fixture_or_skip() -> bytes:
    # The repo-wide *.pdf gitignore (post data-incident guard) keeps this
    # fixture out of git on purpose. Regenerate locally with:
    #   curl -sL -A "Mozilla/5.0" -o tests/fixtures/retrieval/asylnet_decision.pdf \
    #     https://www.asyl.net/fileadmin/user_upload/33812.pdf
    path = os.path.join(FIXTURES, "asylnet_decision.pdf")
    if not os.path.exists(path):
        pytest.skip("PDF fixture not present (gitignored); see comment for regeneration")
    with open(path, "rb") as fh:
        return fh.read()


def test_pdf_text_extraction_contains_aktenzeichen():
    text = extract_pdf_text(_pdf_fixture_or_skip())
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
    pdf_bytes = _pdf_fixture_or_skip()

    async def handler(request):
        return httpx.Response(200, content=pdf_bytes,
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


# ---------------------------------------------------------------------------
# Review findings (2026-07-01) — Az coverage and short-decision thresholds.
# ---------------------------------------------------------------------------

def test_top_court_aktenzeichen_formats_classify_ok():
    # Finding #3: BVerfG ("2 BvR 1845/18"), EuGH ("C-123/22") and ECLI ids
    # must count as decision signals.
    base = "Tenor Gründe Randnummer. " + ("Entscheidungstext. " * 60)
    for az in ("2 BvR 1845/18", "1 BvL 7/16", "C-123/22", "ECLI:DE:BVERFG:2019:rs20191105"):
        assert classify_page(200, f"Beschluss {az} {base}") == "ok", az


def test_short_but_real_decision_is_ok_not_blocked():
    # Finding #5: a Tenor-only Beschluss of ~900 chars is a real decision,
    # not a wall.
    text = ("Beschluss 7 B 19/26. Tenor: Der Antrag wird abgelehnt. "
            "Die Kosten des Verfahrens trägt der Antragsteller. ") * 8
    assert 200 < len(text) < 1500
    assert classify_page(200, text) == "ok"


def test_tiny_snippet_with_az_is_still_not_ok():
    assert classify_page(200, "VG Köln Beschluss 27 K 4231/25.A") == "blocked"


@pytest.mark.anyio
async def test_fetch_source_sets_browser_ua_even_on_injected_client():
    # Finding #10: voris gates on UA; an injected (pooled) client must still
    # send the browser UA per-request.
    seen = {}

    async def handler(request):
        seen["ua"] = request.headers.get("user-agent", "")
        return httpx.Response(200, content=_fixture("nrwe_decision.html"),
                              headers={"content-type": "text/html"})

    async with _client_for(handler) as client:
        await fetch_source("https://nrwe.justiz.nrw.de/x.html", client=client)
    from endpoints.research.retrieval import BROWSER_UA
    assert seen["ua"] == BROWSER_UA


# ---------------------------------------------------------------------------
# OCR fallback for scanned decision PDFs (Pillar 3 prerequisite).
# A scan (pages but no text layer) must go through the injected ocr_fn; with
# no OCR available it degrades to 'scan_unocred' — never fake-blocked/ok.
# ---------------------------------------------------------------------------

def _scanned_pdf_bytes() -> bytes:
    """Programmatic 2-page PDF with NO text layer (a 'scan')."""
    import fitz
    doc = fitz.open()
    doc.new_page(); doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


DECISION_TEXT = ("Beschluss 7 B 19/26. Tenor: Der Antrag wird abgelehnt. "
                 "Gründe: " + "Entscheidungstext. " * 30)


@pytest.mark.anyio
async def test_scanned_pdf_goes_through_ocr_fn():
    called = {}

    async def fake_ocr(data: bytes) -> str:
        called["bytes"] = len(data)
        return DECISION_TEXT

    async def handler(request):
        return httpx.Response(200, content=_scanned_pdf_bytes(),
                              headers={"content-type": "application/pdf"})

    async with _client_for(handler) as client:
        result = await fetch_source("https://asyl.net/scan.pdf", client=client, ocr_fn=fake_ocr)

    assert called, "ocr_fn was not invoked for a scanned PDF"
    assert result.status == "ok"
    assert result.ocr_applied is True
    assert "7 B 19/26" in result.text


@pytest.mark.anyio
async def test_scanned_pdf_without_ocr_degrades_honestly():
    async def handler(request):
        return httpx.Response(200, content=_scanned_pdf_bytes(),
                              headers={"content-type": "application/pdf"})

    async with _client_for(handler) as client:
        result = await fetch_source("https://asyl.net/scan.pdf", client=client)

    assert result.status == "scan_unocred"
    assert result.ocr_applied is False
    assert result.text == ""


@pytest.mark.anyio
async def test_ocr_failure_degrades_to_scan_unocred_not_error():
    async def broken_ocr(data: bytes) -> str:
        raise RuntimeError("desktop asleep, WOL timeout")

    async def handler(request):
        return httpx.Response(200, content=_scanned_pdf_bytes(),
                              headers={"content-type": "application/pdf"})

    async with _client_for(handler) as client:
        result = await fetch_source("https://asyl.net/scan.pdf", client=client, ocr_fn=broken_ocr)

    assert result.status == "scan_unocred"
    assert "asleep" in result.notes or "ocr" in result.notes.lower()


@pytest.mark.anyio
async def test_text_layer_pdf_does_not_invoke_ocr():
    pdf_bytes = _pdf_fixture_or_skip()
    called = {}

    async def fake_ocr(data: bytes) -> str:
        called["hit"] = True
        return "should not be used"

    async def handler(request):
        return httpx.Response(200, content=pdf_bytes,
                              headers={"content-type": "application/pdf"})

    async with _client_for(handler) as client:
        result = await fetch_source("https://asyl.net/x.pdf", client=client, ocr_fn=fake_ocr)

    assert not called
    assert result.status == "ok"
    assert result.ocr_applied is False


# ---------------------------------------------------------------------------
# render_fn fallback (Addendum 2026-07-14): JS-Walls bekommen genau EINEN
# Playwright-Nachversuch; PDFs und ok-Seiten nie.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_blocked_wall_recovers_via_render_fn():
    decision_html = _fixture_text("nrwe_decision.html")

    async def handler(request):
        return httpx.Response(200, content=_fixture("openjur_captcha.html"),
                              headers={"content-type": "text/html"})

    async def render_fn(url):
        return decision_html

    async with _client_for(handler) as client:
        result = await fetch_source("https://openjur.de/u/x.html", client=client, render_fn=render_fn)

    assert result.status == "ok"
    assert "17 L 3613/25" in result.text
    assert "Playwright" in result.notes


@pytest.mark.anyio
async def test_render_fn_failure_keeps_original_status():
    async def handler(request):
        return httpx.Response(200, content=_fixture("openjur_captcha.html"),
                              headers={"content-type": "text/html"})

    async def render_fn(url):
        raise RuntimeError("browser crashed")

    async with _client_for(handler) as client:
        result = await fetch_source("https://openjur.de/u/x.html", client=client, render_fn=render_fn)

    assert result.status == "blocked"


@pytest.mark.anyio
async def test_render_fn_wall_behind_wall_stays_blocked():
    wall = _fixture_text("openjur_captcha.html")

    async def handler(request):
        return httpx.Response(200, content=wall.encode(),
                              headers={"content-type": "text/html"})

    async def render_fn(url):
        return wall  # auch gerendert nur die Wall

    async with _client_for(handler) as client:
        result = await fetch_source("https://openjur.de/u/x.html", client=client, render_fn=render_fn)

    assert result.status == "blocked"
    assert result.text == ""


@pytest.mark.anyio
async def test_ok_page_never_invokes_render_fn():
    called = {}

    async def handler(request):
        return httpx.Response(200, content=_fixture("nrwe_decision.html"),
                              headers={"content-type": "text/html"})

    async def render_fn(url):
        called["hit"] = True
        return ""

    async with _client_for(handler) as client:
        result = await fetch_source("https://nrwe.justiz.nrw.de/x.html", client=client, render_fn=render_fn)

    assert result.status == "ok"
    assert not called


@pytest.mark.anyio
async def test_blocked_pdf_never_invokes_render_fn():
    called = {}

    async def handler(request):
        return httpx.Response(403, content=b"%PDF-1.4 denied",
                              headers={"content-type": "application/pdf"})

    async def render_fn(url):
        called["hit"] = True
        return ""

    async with _client_for(handler) as client:
        result = await fetch_source("https://x.de/u.pdf", client=client, render_fn=render_fn)

    assert not called

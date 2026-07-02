"""Tests for the research citation verifier (Pillar 3, deterministic tier).

Encodes the 242/25 audit failure modes: wrong Aktenzeichen on a real page
(Wiesbaden "m33861" vs real "6 L 3034/25.WI.A"), quotes that are not on the
page, outcomes misreported (denials framed as support), unfetchable pages.

Run: .venv/bin/python -m pytest tests/test_research_verify.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from endpoints.research.retrieval import FetchResult  # noqa: E402
from endpoints.research.verify import (  # noqa: E402
    VerifyResult,
    check_aktenzeichen,
    check_ergebnis,
    check_zitat,
    verify_source,
)

PAGE = (
    "Verwaltungsgericht Düsseldorf Beschluss vom 04.11.2025 – 17 L 3613/25.A "
    "Tenor: Der Antrag wird abgelehnt. Der Antragsteller trägt die Kosten des Verfahrens. "
    "Gründe: Syrern droht im Falle einer Rückkehr generell keine allgemeine Notlage "
    "im Sinne des § 60 Abs. 5 AufenthG i.V.m. Art. 3 EMRK. "
    "Weshalb er nicht an diese Berufstätigkeiten wird anknüpfen können, ist nicht ersichtlich."
)


# ---------------------------------------------------------------------------
# Aktenzeichen containment (normalization-tolerant)
# ---------------------------------------------------------------------------

def test_az_found_verbatim():
    assert check_aktenzeichen("17 L 3613/25.A", PAGE) is True


def test_az_found_despite_spacing_differences():
    assert check_aktenzeichen("17 L 3613/25.A", PAGE.replace("17 L 3613/25.A", "17  L  3613 / 25.A")) is True
    assert check_aktenzeichen("17L3613/25.A", PAGE) is True


def test_wrong_az_fails():
    # The Wiesbaden failure: claimed id not on the page.
    assert check_aktenzeichen("m33861", PAGE) is False
    assert check_aktenzeichen("6 L 3034/25.WI.A", PAGE) is False


def test_empty_az_fails():
    assert check_aktenzeichen("", PAGE) is False
    assert check_aktenzeichen(None, PAGE) is False


# ---------------------------------------------------------------------------
# Verbatim quote (fuzzy — OCR/whitespace tolerant)
# ---------------------------------------------------------------------------

def test_exact_quote_passes():
    assert check_zitat("Syrern droht im Falle einer Rückkehr generell keine allgemeine Notlage", PAGE) is True


def test_quote_with_minor_ocr_noise_passes_with_ocr_threshold():
    noisy = "Syrern droht im FaIle einer Rúckkehr generell keine allgemeine Notlage"
    assert check_zitat(noisy, PAGE, ocr_applied=True) is True


def test_fabricated_quote_fails():
    assert check_zitat("Dem Kläger ist ein Abschiebungsverbot zwingend zuzuerkennen", PAGE) is False


def test_short_or_empty_quote_fails():
    assert check_zitat("", PAGE) is False
    assert check_zitat("Tenor", PAGE) is False  # too short to prove anything


# ---------------------------------------------------------------------------
# Ergebnis vs Tenor (rule-based; Qwen semantic layer is a later, advisory tier)
# ---------------------------------------------------------------------------

def test_abgelehnt_consistent_with_tenor():
    assert check_ergebnis("abgelehnt", PAGE) is True


def test_stattgegeben_contradicts_denial_tenor():
    # The Düsseldorf failure: a denial must not verify as stattgegeben.
    assert check_ergebnis("stattgegeben", PAGE) is False


def test_stattgegeben_consistent_with_granting_tenor():
    granting = ("Urteil RN 11 K 25.33928 Tenor: Die Beklagte wird verpflichtet festzustellen, "
                "dass ein Abschiebungsverbot nach § 60 Abs. 5 AufenthG vorliegt. Der Klage wird "
                "insoweit stattgegeben. Gründe: ...")
    assert check_ergebnis("stattgegeben", granting) is True


def test_unklar_or_missing_ergebnis_is_not_checked():
    # 'unklar' makes no claim — nothing to falsify.
    assert check_ergebnis("unklar", PAGE) is None
    assert check_ergebnis(None, PAGE) is None


# ---------------------------------------------------------------------------
# verify_source — the combined gate
# ---------------------------------------------------------------------------

def _fetch_ok(text=PAGE, ocr=False):
    return FetchResult(status="ok", resolved_url="https://x.de/d", text=text,
                       http_status=200, notes="", is_pdf=False)


def _grounding(**overrides):
    g = {
        "quelle_typ": "entscheidung",
        "gericht": "Verwaltungsgericht Düsseldorf",
        "datum": "2025-11-04",
        "aktenzeichen": "17 L 3613/25.A",
        "ergebnis": "abgelehnt",
        "zitat": "Syrern droht im Falle einer Rückkehr generell keine allgemeine Notlage",
        "lager": "gegen",
    }
    g.update(overrides)
    return g


def test_verify_source_all_checks_pass():
    r = verify_source(_grounding(), _fetch_ok())
    assert isinstance(r, VerifyResult)
    assert r.verifiziert is True
    assert r.checks["aktenzeichen"] is True
    assert r.checks["zitat"] is True
    assert r.checks["ergebnis"] is True


def test_verify_source_wrong_az_fails_hard():
    r = verify_source(_grounding(aktenzeichen="6 L 3034/25.WI.A"), _fetch_ok())
    assert r.verifiziert is False
    assert "aktenzeichen" in r.notes.lower()


def test_verify_source_unfetchable_page_is_unverified_not_refuted():
    blocked = FetchResult(status="blocked", resolved_url="https://openjur.de/u/9.html", text="")
    r = verify_source(_grounding(), blocked)
    assert r.verifiziert is False
    assert "blocked" in r.notes


def test_verify_source_scan_quote_miss_reports_scan_not_refuted():
    # Failed quote match on OCR'd text reads "nicht bestätigt (Scan)", not refuted.
    scan = FetchResult(status="ok", resolved_url="https://x.de/scan.pdf",
                       text=PAGE.replace(
                           "Syrern droht im Falle einer Rückkehr generell keine allgemeine Notlage",
                           "#### vollständig unleserlicher OCR-Bereich ####"),
                       is_pdf=True, notes="", http_status=200)
    scan.ocr_applied = True
    r = verify_source(_grounding(), scan)
    assert r.verifiziert is False
    assert "scan" in r.notes.lower()


def test_verify_source_coi_needs_only_reachability():
    coi = {"quelle_typ": "coi"}
    r = verify_source(coi, _fetch_ok(text="EUAA Country Focus Syria 2025 " * 100))
    assert r.verifiziert is True

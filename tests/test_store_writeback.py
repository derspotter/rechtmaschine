"""Tests for verified-decision write-back into the jurisprudence store (P3-D).

Only decisions that are BOTH deterministically verified AND grounding-complete
may enter RechtsprechungEntry — that is the drafter guard: the pack only ever
injects store content, so unverified research can never reach a Schriftsatz
through it.

Run: .venv/bin/python -m pytest tests/test_store_writeback.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from endpoints.research.store_writeback import (  # noqa: E402
    grounding_to_extraction_fields,
    writeback_identity_sha,
)


def _verified_source(**overrides):
    g = {
        "quelle_typ": "entscheidung",
        "gericht": "Verwaltungsgericht Düsseldorf",
        "datum": "04.11.2025",
        "aktenzeichen": "17 L 3613/25.A",
        "ebene": "VG",
        "ergebnis": "abgelehnt",
        "zitat": "Syrern droht im Falle einer Rückkehr generell keine allgemeine Notlage",
        "lager": "gegen",
        "verifiziert": True,
        "verify_notes": "",
    }
    g.update(overrides)
    return {"url": "https://nrwe.justiz.nrw.de/x.html", "title": "T",
            "description": "Rückkehr Syrien, § 60 Abs. 5", "grounding": g}


def test_verified_complete_source_maps_to_extraction_fields():
    fields = grounding_to_extraction_fields(_verified_source(), country="Syrien")
    assert fields["country"] == "Syrien"
    assert fields["court"] == "Verwaltungsgericht Düsseldorf"
    assert fields["aktenzeichen"] == "17 L 3613/25.A"
    assert fields["outcome"] == "abgelehnt"
    assert fields["decision_date"] == "2025-11-04"  # normalized to ISO
    assert fields["key_holdings"] == ["Syrern droht im Falle einer Rückkehr generell keine allgemeine Notlage"]
    assert "lager:gegen" in fields["tags"]


def test_iso_date_stays_iso():
    fields = grounding_to_extraction_fields(_verified_source(datum="2025-11-04"), country="Syrien")
    assert fields["decision_date"] == "2025-11-04"


def test_unverified_source_is_rejected():
    assert grounding_to_extraction_fields(_verified_source(verifiziert=False), country="Syrien") is None
    assert grounding_to_extraction_fields(_verified_source(verifiziert=None), country="Syrien") is None


def test_incomplete_grounding_is_rejected():
    assert grounding_to_extraction_fields(_verified_source(aktenzeichen=None), country="Syrien") is None
    assert grounding_to_extraction_fields(_verified_source(zitat=""), country="Syrien") is None


def test_coi_source_is_rejected():
    assert grounding_to_extraction_fields(_verified_source(quelle_typ="coi"), country="Syrien") is None


def test_source_without_grounding_is_rejected():
    assert grounding_to_extraction_fields({"url": "https://x.de"}, country="Syrien") is None


def test_identity_sha_stable_across_cosmetics():
    a = writeback_identity_sha({"gericht": "VG Düsseldorf", "datum": "04.11.2025", "aktenzeichen": "17 L 3613/25.A"})
    b = writeback_identity_sha({"gericht": "vg  düsseldorf", "datum": "2025-11-04", "aktenzeichen": "17L3613/25.A"})
    assert a == b
    c = writeback_identity_sha({"gericht": "VG Köln", "datum": "2025-09-03", "aktenzeichen": "27 K 4231/25.A"})
    assert a != c


def test_identity_sha_canonicalizes_court_name_variants():
    # Die realen Duplikate im Bestand (2026-07-13): Langform vs. Kurzform.
    def _sha(gericht):
        return writeback_identity_sha(
            {"gericht": gericht, "datum": "16.02.2024", "aktenzeichen": "3 K 320/22"}
        )

    assert _sha("VG Düsseldorf") == _sha("Verwaltungsgericht Düsseldorf")
    assert _sha("Verwaltungsgericht Bremen") == _sha(
        "Verwaltungsgericht der Freien Hansestadt Bremen"
    )
    assert _sha("OVG NRW") == _sha("Oberverwaltungsgericht NRW")
    assert _sha("BVerwG") == _sha("Bundesverwaltungsgericht")
    assert _sha("VGH Baden-Württemberg") == _sha("Verwaltungsgerichtshof Baden-Württemberg")
    # Verschiedene Gerichte mit gleichem Az bleiben verschieden.
    assert _sha("VG Bremen") != _sha("VG Köln")
    # OVG ist nicht VG.
    assert _sha("OVG Bremen") != _sha("VG Bremen")

"""GII-Quelle (gesetze-im-internet.de XML) für Normtexte.

Fixtures sind die ECHTEN xml.zip-Inhalte (GG, AsylbLG, Stand 2026) —
der Parser muss daraus Extractor-kompatibles Markdown erzeugen, das das
looks_valid_law_markdown-Gate des Downloaders besteht.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from legal_texts.downloader import looks_valid_law_markdown  # noqa: E402
from legal_texts.gii import law_xml_to_markdown  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "legal_texts"
GG_XML = (FIXTURES / "gii_gg.xml").read_text(encoding="utf-8")
ASYLBLG_XML = (FIXTURES / "gii_asylblg.xml").read_text(encoding="utf-8")


def test_gg_passes_downloader_gate():
    markdown, version = law_xml_to_markdown(GG_XML, "GG")
    assert looks_valid_law_markdown("GG", markdown)
    assert version == "2026-05-06"  # builddate der Fixture


def test_gg_art_16a_with_body():
    markdown, _ = law_xml_to_markdown(GG_XML, "GG")
    assert "\n### Art 16a\n" in markdown
    assert "Politisch Verfolgte genießen Asylrecht." in markdown


def test_gg_front_matter():
    markdown, _ = law_xml_to_markdown(GG_XML, "GG")
    head = markdown[:600]
    assert "jurabk: GG" in head
    assert "stand: 2026-05-06" in head
    assert "quelle: GII" in head
    # Änderungsstand aus der standangabe — anwaltlich relevante Info
    assert "geändert" in head.lower()


def test_gg_skips_non_provisions():
    markdown, _ = law_xml_to_markdown(GG_XML, "GG")
    assert "### Eingangsformel" not in markdown
    assert "### Präambel" not in markdown
    assert "### Anhang EV" not in markdown


def test_asylblg_passes_downloader_gate():
    markdown, version = law_xml_to_markdown(ASYLBLG_XML, "AsylbLG")
    assert looks_valid_law_markdown("AsylbLG", markdown)
    assert version == "2026-06-11"


def test_asylblg_geas_folgeaenderung_im_stand():
    _, version = law_xml_to_markdown(ASYLBLG_XML, "AsylbLG")
    markdown, _ = law_xml_to_markdown(ASYLBLG_XML, "AsylbLG")
    assert "23.4.2026" in markdown[:600]  # zuletzt geändert (GEAS-Folge)


def test_asylblg_numbered_lists_flattened():
    markdown, _ = law_xml_to_markdown(ASYLBLG_XML, "AsylbLG")
    assert "### § 1 Leistungsberechtigte" in markdown
    body = markdown.split("### § 1 Leistungsberechtigte")[1].split("###")[0]
    assert "(1) Leistungsberechtigt" in body
    assert "1. eine Aufenthaltsgestattung nach dem Asylgesetz besitzen," in body
    # verschachtelte alpha-Liste unter Nummer 3
    assert "b) nach § 25 Absatz 4 Satz 1 des Aufenthaltsgesetzes oder" in body
    assert "<" not in body  # kein Roh-XML durchgesickert


def test_wrong_root_raises():
    import pytest

    with pytest.raises(Exception):
        law_xml_to_markdown("<html><body>CAPTCHA</body></html>", "GG")


def test_entity_declaration_rejected():
    import pytest

    evil = (
        '<?xml version="1.0"?><!DOCTYPE dokumente [<!ENTITY a "aaaa">]>'
        "<dokumente builddate=\"20260101000000\"><norm/></dokumente>"
    )
    with pytest.raises(ValueError, match="ENTITY"):
        law_xml_to_markdown(evil, "GG")

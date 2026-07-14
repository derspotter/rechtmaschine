"""NeuRIS-Quelle für legal_texts (Spec-Addendum 2026-07-14).

Fixtures sind ECHTE API-Antworten vom 14.07.2026:
  - asylg_neuris_excerpt.html — Kopf + §§ 1, 3, 30 der AsylG-Fassung
    2026-07-10 (GEAS: § 3 verweist auf VO (EU) 2024/1347, § 30 auf 2024/1348)
  - search_asylg.json — /v1/legislation-Suchtreffer inkl. Rausch-Treffer

Run: .venv/bin/python -m pytest tests/test_legal_texts_neuris.py -q
"""
import json
import os

import pytest

from legal_texts.neuris import (
    html_law_to_markdown,
    pick_law_from_search,
    version_date_from_eli,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "legal_texts")


def _fixture_text(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


# --- Auflösung Suche → Gesetz ---------------------------------------------------


def test_pick_law_matches_exact_abbreviation_only():
    data = json.loads(_fixture_text("search_asylg.json"))
    hit = pick_law_from_search(data, "AsylG")
    assert hit["legislationIdentifier"] == "eli/bund/bgbl-1/1992/s1126/2026-07-10/1/deu"
    # Rausch-Treffer (AZRG-DV) und fehlende Gesetze geben None.
    assert pick_law_from_search(data, "GG") is None
    assert pick_law_from_search({}, "AsylG") is None


def test_version_date_from_eli():
    assert version_date_from_eli("eli/bund/bgbl-1/1992/s1126/2026-07-10/1/deu") == "2026-07-10"
    assert version_date_from_eli("kaputt") == ""


# --- HTML → Markdown -------------------------------------------------------------


@pytest.fixture(scope="module")
def markdown():
    return html_law_to_markdown(
        _fixture_text("asylg_neuris_excerpt.html"),
        title="Asylgesetz",
        abbreviation="AsylG",
        eli="eli/bund/bgbl-1/1992/s1126/2026-07-10/1/deu",
    )


def test_markdown_has_provision_headings(markdown):
    assert "### § 1 Geltungsbereich" in markdown
    assert "### § 3 Zuerkennung des internationalen Schutzes" in markdown
    assert "### § 30 Offensichtlich unbegründete Asylanträge" in markdown


def test_markdown_carries_geas_text_and_unescapes_entities(markdown):
    assert "Verordnung (EU) 2024/1347" in markdown  # § 3 n.F.
    assert "Verordnung (EU) 2024/1348" in markdown  # § 30 n.F.
    assert "&nbsp;" not in markdown
    assert "§ 60 Absatz 8" in markdown  # nbsp-Entities zu normalen Spaces


def test_markdown_front_matter_carries_version(markdown):
    head = markdown[:400]
    assert "jurabk: AsylG" in head
    assert "2026-07-10" in head
    assert "eli/bund/bgbl-1/1992/s1126" in head


def test_markdown_has_no_leftover_markup(markdown):
    assert "<div" not in markdown
    assert "akn-" not in markdown
    assert "Inhaltsübersicht" not in markdown  # Präambel/ToC wird übersprungen


# --- Integration: bestehender Extractor liest das neue Markdown ------------------


def test_extractor_reads_converted_markdown(markdown, tmp_path, monkeypatch):
    import legal_texts.downloader as dl
    from legal_texts.extractor import extract_provision

    (tmp_path / "asylg.md").write_text(markdown, encoding="utf-8")
    monkeypatch.setattr(dl, "LEGAL_TEXTS_DIR", tmp_path)

    text = extract_provision("AsylG", "30")
    assert "2024/1348" in text
    assert text.startswith("### § 30")
    assert "[FEHLER]" in extract_provision("AsylG", "99")

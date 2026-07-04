import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from dokuwiki_markup import (  # noqa: E402
    Section,
    chunk_section,
    clean_markup,
    context_header,
    extract_normen,
    split_sections,
)
from rag_vocabulary import Vocabulary  # noqa: E402


# --- clean_markup -----------------------------------------------------------


def test_headings_convert_to_markdown_levels():
    raw = "====== Afghanistan ======\ntext\n===== Lage =====\nmehr\n== tief ==\n"
    cleaned = clean_markup(raw)
    assert "# Afghanistan" in cleaned
    assert "## Lage" in cleaned
    assert "##### tief" in cleaned


def test_internal_link_uses_label_or_page_tail():
    raw = "Siehe [[laender:afghanistan|die Länderseite]] und [[asylgesetz:par_3_asylg]]."
    cleaned = clean_markup(raw)
    assert "die Länderseite" in cleaned
    assert "par 3 asylg" in cleaned
    assert "[[" not in cleaned


def test_external_link_keeps_url():
    raw = "Vgl. [[https://www.bverwg.de/de/urteil|BVerwG-Urteil]]."
    cleaned = clean_markup(raw)
    assert "BVerwG-Urteil (https://www.bverwg.de/de/urteil)" in cleaned


def test_formatting_markers_stripped_url_slashes_survive():
    raw = "Das ist **wichtig** und //kursiv//, Quelle: https://example.org/pfad"
    cleaned = clean_markup(raw)
    assert "wichtig" in cleaned and "**" not in cleaned
    assert "kursiv" in cleaned
    assert "https://example.org/pfad" in cleaned


def test_note_becomes_hinweis_and_table_flattens():
    raw = (
        "<note important>Frist beachten</note>\n"
        "^ Land ^ Frist ^\n"
        "| Syrien | 2 Wochen |\n"
    )
    cleaned = clean_markup(raw)
    assert "Hinweis: Frist beachten" in cleaned
    assert "Land | Frist" in cleaned
    assert "Syrien | 2 Wochen" in cleaned


def test_footnotes_media_and_unknown_tags():
    raw = (
        "Text mit Fußnote((BVerwG, Urt. v. 01.01.2020)).\n"
        "{{ :bild.png?200 |Karte}}\n"
        "<wrap hi>markiert</wrap>\n"
        "<code>§ 60 AufenthG bleibt</code>\n"
    )
    cleaned = clean_markup(raw)
    assert "(Fn: BVerwG, Urt. v. 01.01.2020)" in cleaned
    assert "bild.png" not in cleaned and "Karte" in cleaned
    assert "markiert" in cleaned and "<wrap" not in cleaned
    assert "§ 60 AufenthG bleibt" in cleaned and "<code>" not in cleaned


# --- split_sections ---------------------------------------------------------


def test_split_sections_builds_heading_paths():
    cleaned = (
        "Einleitung vor allem.\n"
        "# Afghanistan\n"
        "Allgemeines zum Land.\n"
        "## Abschiebungsverbote\n"
        "### § 60 Abs. 5 AufenthG\n"
        "Details zur Norm.\n"
        "## Rückkehr\n"
        "Rückkehrfragen.\n"
    )
    sections = split_sections(cleaned, "Afghanistan")
    paths = [s.heading_path for s in sections]
    assert paths[0] == "Afghanistan"  # lead
    assert "Afghanistan > Abschiebungsverbote > § 60 Abs. 5 AufenthG" in paths
    assert "Afghanistan > Rückkehr" in paths
    # H1 equal to the page title must not duplicate ("Afghanistan > Afghanistan").
    assert all(not p.startswith("Afghanistan > Afghanistan") for p in paths)
    norm_section = next(s for s in sections if "§ 60" in s.heading_path)
    assert norm_section.text == "Details zur Norm."


def test_split_sections_distinct_title_prefixes_page_title():
    sections = split_sections("# Lage\nText.", "Eritrea")
    assert sections == [Section(heading_path="Eritrea > Lage", text="Text.")]


# --- chunk_section ----------------------------------------------------------


def test_chunk_section_deterministic_and_merges_tail():
    paras = "\n\n".join(f"Absatz {i} " + "x" * 400 for i in range(10))
    first = chunk_section(paras)
    second = chunk_section(paras)
    assert first == second
    assert all(len(c) <= 2400 for c in first)
    assert len(first[-1]) >= 200 or len(first) == 1


# --- context_header ---------------------------------------------------------


def test_context_header_format():
    header = context_header(
        "Afghanistan",
        "Afghanistan > Abschiebungsverbote",
        "https://wiki.aufentha.lt/laender/afghanistan",
    )
    assert header.splitlines() == [
        "[Aufenthaltswiki | Afghanistan | Afghanistan > Abschiebungsverbote]",
        "[Source: public legal wiki]",
        "[URL: https://wiki.aufentha.lt/laender/afghanistan]",
    ]


def test_context_header_skips_redundant_path():
    header = context_header("Afghanistan", "Afghanistan", "")
    assert header.splitlines() == [
        "[Aufenthaltswiki | Afghanistan]",
        "[Source: public legal wiki]",
    ]


# --- extract_normen ---------------------------------------------------------


def test_extract_normen_reorders_to_gesetz_first():
    text = (
        "Ein Abschiebungsverbot nach § 60 Abs. 5 AufenthG i.V.m. Art. 3 EMRK "
        "kommt in Betracht. Daneben ist § 3 AsylG zu prüfen."
    )
    normen = extract_normen(text)
    assert "AufenthG § 60 Abs. 5" in normen
    assert "EMRK Art. 3" in normen
    assert "AsylG § 3" in normen


def test_extract_normen_accepts_gesetz_first_prose_and_drops_unknown():
    vocab = Vocabulary(normen=["AsylG § 3"], normen_aliases={})
    text = "AsylG § 3 sowie § 999 PhantasieG gelten."
    assert extract_normen(text, vocab=vocab) == ["AsylG § 3"]

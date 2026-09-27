"""Zentrale PDF-Textextraktion (app/pdf_text.py).

Bis 2026-09 extrahierten vierzehn Stellen PDF-Text je selbst, in zwei
unterschiedlichen Formaten: die Store-Pfade trennten Seiten mit einer
Leerzeile und trimmten jede Seite, Verifikation und Research verbanden die
Seiten ungetrimmt mit einem einfachen Umbruch. Das zentrale Modul legt das
Store-Format fest, weil dieses Format im Rechtsprechungsstore und in den
Chunks persistiert ist und sich nicht ändern darf.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

from pdf_text import extract_text, extract_text_and_pages

APP = Path(__file__).resolve().parent.parent / "app"


def _pdf(*pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        y = 72
        for line in (text or "").split("\n"):
            if line:
                page.insert_text((72, y), line)
            y += 14
    return doc.tobytes()


def _legacy_store_text(data: bytes) -> str:
    """Exakte Nachbildung von jurisprudence_ingest.pdf_bytes_text vor 2026-09-27."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(data)
        tmp.flush()
        doc = pymupdf.open(tmp.name)
        try:
            return "\n\n".join((page.get_text() or "").strip() for page in doc)
        finally:
            doc.close()


def test_pages_are_stripped_and_joined_with_blank_line():
    data = _pdf("Seite eins", "Seite zwei\nzweite Zeile", None)
    assert extract_text(data) == "Seite eins\n\nSeite zwei\nzweite Zeile\n\n"


def test_matches_legacy_store_extraction_byte_for_byte():
    data = _pdf("Erste Seite", "", "Dritte Seite\n\nnach Leerzeile", "Vierte")
    assert extract_text(data) == _legacy_store_text(data)


def test_path_and_bytes_give_identical_text(tmp_path):
    data = _pdf("A", "B")
    f = tmp_path / "x.pdf"
    f.write_bytes(data)
    assert extract_text(f) == extract_text(str(f)) == extract_text(data)


def test_extract_text_and_pages_reports_page_count():
    data = _pdf("A", None, "C")
    text, pages = extract_text_and_pages(data)
    assert pages == 3
    assert text == extract_text(data)


def test_import_writes_nothing_to_stdout():
    proc = subprocess.run(
        [sys.executable, "-c", "import pdf_text"],
        cwd=APP,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout == ""


# --- Aufrufer, die bis 2026-09-27 ein abweichendes Format lieferten ---------
#
# Research und Verifikation verbanden die Seiten ungetrimmt mit einem
# einfachen Umbruch. Dieses PDF trennt beide Formate: führender Einzug auf
# Seite 1 und ein Schlussumbruch, den nur das alte Format behält.


def _pdf_where_formats_differ():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "   eingerueckt")
    page = doc.new_page()
    page.insert_text((72, 72), "oben")
    page.insert_text((72, 700), "unten")
    return doc.tobytes()


def test_research_extract_pdf_text_uses_store_format():
    from endpoints.research.retrieval import extract_pdf_text

    data = _pdf_where_formats_differ()
    assert extract_pdf_text(data) == extract_text(data)


def test_research_extract_pdf_text_with_pages_uses_store_format():
    from endpoints.research.retrieval import extract_pdf_text_with_pages

    data = _pdf_where_formats_differ()
    assert extract_pdf_text_with_pages(data) == extract_text_and_pages(data)


def test_verify_source_reads_local_file_in_store_format(tmp_path):
    from types import SimpleNamespace

    import verify_source

    # fetch_fulltext liest lokale Dateien erst ab 200 Zeichen, darunter
    # fällt es auf den Download zurück.
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "   eingerueckt " + "x" * 60)
    for _ in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), "oben " + "y" * 60)
        page.insert_text((72, 700), "unten")
    f = tmp_path / "entscheidung.pdf"
    f.write_bytes(doc.tobytes())

    entry = SimpleNamespace(source_url=str(f), source_type="cited", source_ref=None)
    text, used = verify_source.fetch_fulltext(entry)
    assert used == str(f)
    assert text == extract_text(f)

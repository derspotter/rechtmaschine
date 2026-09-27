"""Zentrale PDF-Textextraktion.

Das Format ist das des Rechtsprechungsstores: jede Seite getrimmt, Seiten mit
einer Leerzeile getrennt. Es ist in den gespeicherten Einträgen und Chunks
persistiert und darf sich nicht ändern.

pymupdf wird als ``pymupdf`` importiert, nicht als ``fitz``: das fitz-Paket
ist nur noch eine Kompatibilitätsschicht, die beim Import eine
Deprecation-Warnung auf stdout schreibt und damit JSON-Ausgaben bricht.
"""

from __future__ import annotations

import os
from typing import Union

import pymupdf

PdfSource = Union[str, os.PathLike, bytes, bytearray]

PAGE_SEPARATOR = "\n\n"


def _open(source: PdfSource) -> pymupdf.Document:
    if isinstance(source, (bytes, bytearray)):
        return pymupdf.open(stream=bytes(source), filetype="pdf")
    return pymupdf.open(os.fspath(source))


def extract_text_and_pages(source: PdfSource) -> tuple[str, int]:
    """Text eines PDFs samt Seitenzahl. Quelle ist ein Pfad oder die Bytes."""
    with _open(source) as doc:
        text = PAGE_SEPARATOR.join((page.get_text() or "").strip() for page in doc)
        return text, doc.page_count


def extract_text(source: PdfSource) -> str:
    """Text eines PDFs. Quelle ist ein Pfad oder die Bytes."""
    return extract_text_and_pages(source)[0]

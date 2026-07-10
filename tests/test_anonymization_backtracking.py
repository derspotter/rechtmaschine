"""Regression: katastrophales Backtracking in der Fuzzy-Ersetzung.

2026-07-10 hing die App >2h in einem einzigen re.sub (safe_replace):
Ein extrahierter Begriff mit internem Whitespace-Lauf (z.B. eine Straße,
die Qwen wortgetreu inkl. OCR-Spacing zurückgibt) erzeugt in _escape_fuzzy
pro Whitespace-Zeichen je ein \\s* — benachbarte \\s*-Quantifier explodieren
kombinatorisch, sobald der Präfix im Text matcht, dahinter ein langer
Whitespace-Lauf steht und der Suffix nicht matcht.
"""

import re
import sys
from multiprocessing import Process
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "anon"))

from anonymization_service import _escape_fuzzy, safe_replace  # noqa: E402


def _run_safe_replace(text, terms, placeholder):
    safe_replace(text, terms, placeholder)


def _finishes_within(seconds, target, args):
    p = Process(target=target, args=args)
    p.start()
    p.join(seconds)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    return True


def test_safe_replace_terminates_on_term_with_internal_whitespace_run():
    # Text: Ankerwort kommt mehrfach vor, gefolgt von langen Whitespace-Läufen
    # (typisches OCR-Layout von Formularen/Tabellen).
    text = ("Aufnahmeeinrichtung" + " " * 60 + "Feldwert\n") * 8 + "Fließtext " * 500
    # Begriff wortgetreu aus dem OCR-Text extrahiert: internes Spacing,
    # Endstück weicht ab (OCR-Drift) -> Suffix matcht nie.
    term = "Aufnahmeeinrichtung" + " " * 10 + "Musterheim"

    assert _finishes_within(
        5, _run_safe_replace, (text, [term], "[ADRESSE]")
    ), "safe_replace hängt: katastrophales Backtracking bei Begriff mit Whitespace-Lauf"


def test_escape_fuzzy_collapses_adjacent_whitespace_wildcards():
    pattern = _escape_fuzzy("Muster" + " " * 10 + "Stadt")
    assert r"\s*\s*" not in pattern, f"benachbarte \\s* nicht zusammengefasst: {pattern}"


def test_escape_fuzzy_separator_mix_has_no_adjacent_wildcards():
    # " - " und ", " erzeugen \s*-\s* / \s*,\s* plus Nachbar-\s* -> muss verschmelzen.
    for term in ("Haupt - Straße", "Meier ,  Anna", "a  ,-  b"):
        pattern = _escape_fuzzy(term)
        assert r"\s*\s*" not in pattern, f"{term!r} -> {pattern}"


def test_safe_replace_still_replaces_across_spacing_drift():
    # Fachlicher Erhalt: Begriff mit internem Lauf matcht Vorkommen
    # mit anderem Spacing (das ist der Zweck der Fuzzy-Escapes).
    term = "Berliner" + " " * 6 + "Allee 12"
    text = "wohnhaft Berliner Allee 12 in X, sowie Berliner\nAllee 12 (Kopie)"
    out = safe_replace(text, [term], "[ADRESSE]")
    assert out.count("[ADRESSE]") == 2, out

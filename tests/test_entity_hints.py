"""Tests für die Aktendaten-Hinweise (entity_hints) der Anonymisierung.

Anlass (30.07.2026, Akte 034/26): Die Qwen-Extraktion lieferte auf einem
zweiseitigen Notizdokument NULL Entitäten, obwohl der Nachname des Mandanten
fünfmal im Text stand. Das Null-Gate verwarf das Ergebnis zu Recht, aber die
Anonymisierung war damit blockiert. `known_entities` (LLM-Extraktion komplett
ersetzen) war der Notnagel — dauerhaft richtig ist der Hinweis-Pfad: was
j-lawyer über die Beteiligten der Akte weiß, stützt die Extraktion, ersetzt
sie aber nicht.

`_entity_hints_present_in_text` ist das Sicherheitsnetz: ein Hinweis wird nur
dann zwingend ersetzt, wenn er auch WIRKLICH im Text steht. Damit kann die
Aktendaten-Quelle nichts schwärzen, was gar nicht vorkommt, und ein Aussetzer
des Modells kann bekannte Beteiligte nicht mehr durchrutschen lassen.

Reine Logik, keine DB, kein Netz. Das tests-Verzeichnis ist nicht in den
Container gemountet, deshalb hineinkopieren:

    docker cp tests/test_entity_hints.py rechtmaschine-app:/tmp/
    docker exec rechtmaschine-app python3 /tmp/test_entity_hints.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from endpoints.anonymization import _entity_hints_present_in_text  # noqa: E402


def test_name_present_in_text_is_kept():
    hints = {"names": ["Li"]}
    text = "Besprechung mit Herrn Li am 31.03.2026."
    assert _entity_hints_present_in_text(hints, text) == {"names": ["Li"]}


def test_name_absent_from_text_is_dropped():
    # Der Mandant der Akte kommt in DIESEM Dokument nicht vor -> nichts schwärzen.
    hints = {"names": ["Tiancheng Li"]}
    text = "Aktenvermerk ohne Namensnennung."
    assert _entity_hints_present_in_text(hints, text) == {}


def test_substring_match_does_not_count():
    # "Li" steckt in "Polizei" und "politisch" — ohne Wortgrenzen würde der
    # halbe Text geschwärzt.
    hints = {"names": ["Li"]}
    text = "Die Polizei hat den politischen Hintergrund geprüft."
    assert _entity_hints_present_in_text(hints, text) == {}


def test_match_is_case_insensitive():
    hints = {"names": ["Tiancheng Li"]}
    text = "Unterzeichnet von TIANCHENG LI."
    assert _entity_hints_present_in_text(hints, text) == {"names": ["Tiancheng Li"]}


def test_identifiers_with_punctuation_are_found():
    # Az und Geschäftszeichen enthalten Sonderzeichen, an denen ein naives
    # \b-Muster scheitert.
    hints = {
        "court_aktenzeichen": ["41 K 1420/26.A"],
        "bamf_geschaeftszeichen": ["10916327-479"],
        "birth_dates": ["19.11.1997"],
    }
    text = "Verfahren 41 K 1420/26.A, Geschäftszeichen 10916327-479, geb. 19.11.1997."
    result = _entity_hints_present_in_text(hints, text)
    assert result["court_aktenzeichen"] == ["41 K 1420/26.A"]
    assert result["bamf_geschaeftszeichen"] == ["10916327-479"]
    assert result["birth_dates"] == ["19.11.1997"]


def test_unknown_keys_are_ignored():
    hints = {"names": ["Li"], "lieblingsfarbe": ["blau"]}
    text = "Herr Li mag blau."
    assert _entity_hints_present_in_text(hints, text) == {"names": ["Li"]}


def test_empty_and_blank_values_are_ignored():
    hints = {"names": ["", "   ", "Li"], "cities": []}
    text = "Herr Li."
    assert _entity_hints_present_in_text(hints, text) == {"names": ["Li"]}


def test_none_hints_yield_empty_dict():
    assert _entity_hints_present_in_text(None, "beliebiger Text") == {}


def test_duplicates_collapse():
    hints = {"names": ["Li", "li", "Li"]}
    text = "Herr Li und nochmals Li."
    assert _entity_hints_present_in_text(hints, text) == {"names": ["Li"]}


def main():
    tests = [
        test_name_present_in_text_is_kept,
        test_name_absent_from_text_is_dropped,
        test_substring_match_does_not_count,
        test_match_is_case_insensitive,
        test_identifiers_with_punctuation_are_found,
        test_unknown_keys_are_ignored,
        test_empty_and_blank_values_are_ignored,
        test_none_hints_yield_empty_dict,
        test_duplicates_collapse,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"OK   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
        sys.exit(1)
    print("\nAll tests passed")


if __name__ == "__main__":
    main()

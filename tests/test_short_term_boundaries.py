"""Kurze Begriffe dürfen nicht mitten in fremden Wörtern ersetzt werden.

Gefunden 30.07.2026 (Akte 034/26): Der Nachname "Li" wurde auch innerhalb von
"Polizei" und "politisch" ersetzt ("Po[PERSON]zei"), eine Kennung zerlegte
"passiert" zu "pass[DOKUMENT-ID]". Für ORTSnamen war das längst gelöst
(`_wrap_alpha_boundaries`, Kommentar "Essen -> Interessen"), für Personen und
Kennungen nicht.

Der Fix ist bewusst auf KURZE Begriffe beschränkt (<= 4 alphanumerische
Zeichen). Lange Namen behalten die bisherige, bewusst großzügige Ersetzung:
Sie fängt OCR-Verschmelzungen wie "HerrMüller" ab, und dort wäre eine
Wortgrenze eine Unter-Schwärzung — also genau der Fehler, der weh tut. Bei
kurzen Begriffen kippt die Abwägung, weil ein Treffer mitten im Wort dort fast
immer ein Fehltreffer ist und den Text für das nachgelagerte Modell zerstört.

    docker cp tests/test_short_term_boundaries.py rechtmaschine-app:/tmp/
    docker exec -e PYTHONPATH=/app/anon rechtmaschine-app python3 /tmp/test_short_term_boundaries.py
"""

import sys

from anonymization_service import (  # noqa: E402
    safe_replace,
    safe_replace_case_numbers,
)


def test_short_name_no_longer_hits_inside_other_words():
    text = "Die Polizei war politisch motiviert."
    assert safe_replace(text, ["Li"], "[PERSON]") == text


def test_short_name_is_still_replaced_standalone():
    out = safe_replace("Besprechung mit Herrn Li heute.", ["Li"], "[PERSON]")
    assert out == "Besprechung mit Herrn [PERSON] heute."


def test_short_name_with_german_inflection_is_replaced():
    # "Lis Antrag" — Genitiv-s darf die Ersetzung nicht verhindern.
    out = safe_replace("Lis Antrag ging ein.", ["Li"], "[PERSON]")
    assert out.startswith("[PERSON]")
    assert "Lis" not in out


def test_short_name_does_not_swallow_longer_words_with_same_start():
    # Linke Wortgrenze allein würde "Liste" treffen — die rechte muss auch sitzen.
    text = "Die Liste liegt vor, ebenso die Lieferung."
    assert safe_replace(text, ["Li"], "[PERSON]") == text


def test_long_name_keeps_the_permissive_match():
    # Regressionsschutz: OCR-Verschmelzung bleibt erfasst.
    out = safe_replace("Termin mit HerrMüller heute.", ["Müller"], "[PERSON]")
    assert out == "Termin mit Herr[PERSON] heute."


def test_long_name_in_compound_still_replaced():
    out = safe_replace("Wohnhaft Müllerstraße 4.", ["Müller"], "[PERSON]")
    assert "[PERSON]" in out


def test_short_identifier_no_longer_splits_words():
    text = "Das ist passiert."
    assert safe_replace_case_numbers(text, ["iert"], "[DOKUMENT-ID]") == text


def test_identifier_is_still_replaced_standalone():
    out = safe_replace_case_numbers("Nummer 7364-KLE-2026 vergeben.", ["7364-KLE-2026"], "[REFERENZNUMMER]")
    assert out == "Nummer [REFERENZNUMMER] vergeben."


def test_short_identifier_standalone_is_still_replaced():
    out = safe_replace_case_numbers("Zeichen 4711 vergeben.", ["4711"], "[REFERENZNUMMER]")
    assert out == "Zeichen [REFERENZNUMMER] vergeben."


def test_short_name_does_not_eat_digits_via_ocr_confusables():
    # "Li" wurde ueber die Verwechslungsklasse [IiLl1|] auf "11" abgebildet und
    # zerlegte damit jedes Datum: "19.11.1997" -> "19.[PERSON].1997".
    out = safe_replace("Geboren am 19.11.1997 in Chengdu.", ["Li"], "[PERSON]")
    assert out == "Geboren am 19.11.1997 in Chengdu."


def test_long_name_keeps_digit_confusables():
    # Regressionsschutz: bei langen Namen bleibt die OCR-Ziffern-Toleranz.
    out = safe_replace("Frau Schm1tt erschien.", ["Schmitt"], "[PERSON]")
    assert out == "Frau [PERSON] erschien."


def test_short_name_keeps_letter_confusables():
    # Buchstaben-Verwechslung bleibt auch bei kurzen Namen erhalten (I/l/i).
    out = safe_replace("Herr Ll erschien.", ["Li"], "[PERSON]")
    assert out == "Herr [PERSON] erschien."


def test_place_behaviour_unchanged():
    # War schon vorher wortgrenzengebunden, darf sich nicht ändern.
    out = safe_replace("Interessen in Essen.", ["Essen"], "[ORT]")
    assert out == "Interessen in [ORT]."


def main():
    tests = [
        test_short_name_no_longer_hits_inside_other_words,
        test_short_name_is_still_replaced_standalone,
        test_short_name_with_german_inflection_is_replaced,
        test_short_name_does_not_swallow_longer_words_with_same_start,
        test_long_name_keeps_the_permissive_match,
        test_long_name_in_compound_still_replaced,
        test_short_identifier_no_longer_splits_words,
        test_identifier_is_still_replaced_standalone,
        test_short_identifier_standalone_is_still_replaced,
        test_short_name_does_not_eat_digits_via_ocr_confusables,
        test_long_name_keeps_digit_confusables,
        test_short_name_keeps_letter_confusables,
        test_place_behaviour_unchanged,
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

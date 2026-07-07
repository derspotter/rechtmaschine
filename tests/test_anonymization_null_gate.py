"""Ad-hoc test for the Task 2 Null-Ersetzungs-Gate in
endpoints.anonymization._anonymization_gate_failed.

Verifies that a degenerate anonymization result (empty output, byte-identical
output, or all entity/replacement lists empty) is rejected instead of being
persisted as a successfully anonymized document. Also checks that a genuine
anonymization result (text changed, at least one entity list non-empty)
passes the gate.

Pure logic only -- no DB, no containers, no network. Run inside the app
container (deps like `fastapi`/`sqlalchemy`/`pikepdf` are only installed
there):

    docker exec rechtmaschine-app python3 /app/../tests/test_anonymization_null_gate.py

or, if run from the repo root with the app's Python deps on PYTHONPATH:

    python3 tests/test_anonymization_null_gate.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from endpoints.anonymization import _anonymization_gate_failed  # noqa: E402


def test_empty_anonymized_text_triggers_gate():
    assert _anonymization_gate_failed(
        "Herr Ali Hassan wohnt in der Musterstraße 1.",
        "",
        ["Ali Hassan"],
        [],
        ["Musterstraße 1"],
    ) is True


def test_byte_identical_output_triggers_gate():
    original = "Herr Ali Hassan wohnt in der Musterstraße 1."
    assert _anonymization_gate_failed(
        original,
        original,
        [],
        [],
        [],
    ) is True


def test_all_entity_lists_empty_triggers_gate_even_if_text_changed():
    # Degenerate case: text differs (e.g. whitespace normalization by the
    # service) but no actual entity was ever extracted/replaced.
    assert _anonymization_gate_failed(
        "Herr Ali Hassan wohnt in der Musterstraße 1.",
        "Herr Ali Hassan wohnt in der Musterstraße  1.",
        [],
        [],
        [],
    ) is True


def test_genuine_anonymization_passes_gate():
    assert _anonymization_gate_failed(
        "Herr Ali Hassan wohnt in der Musterstraße 1.",
        "Herr [NAME] wohnt in der [ADRESSE].",
        ["Ali Hassan"],
        [],
        ["Musterstraße 1"],
    ) is False


def test_only_birth_date_extracted_still_passes_gate():
    # Any one of the three entity lists being non-empty is enough, as long as
    # the text actually changed.
    assert _anonymization_gate_failed(
        "Geboren am 01.01.1990 in Kabul.",
        "Geboren am [GEBURTSDATUM] in Kabul.",
        [],
        ["01.01.1990"],
        [],
    ) is False


def main():
    tests = [
        test_empty_anonymized_text_triggers_gate,
        test_byte_identical_output_triggers_gate,
        test_all_entity_lists_empty_triggers_gate_even_if_text_changed,
        test_genuine_anonymization_passes_gate,
        test_only_birth_date_extracted_still_passes_gate,
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

"""Ad-hoc test for the Task 1 privacy gate in shared.get_document_for_upload.

Verifies that a document flagged is_anonymized=True never silently falls back
to raw/OCR text when the anonymized text file is missing or unreadable, and
instead raises AnonymizedTextMissingError. Also checks the happy path (a
readable anonymized file is preferred over extracted/raw text) still works.

Pure logic only -- no DB, no containers, no network. Run inside the app
container (deps like `anthropic`/`google-genai` are only installed there):

    docker exec rechtmaschine-app python3 /app/../tests/test_anonymized_text_gate.py

or, if run from the repo root with the app's Python deps on PYTHONPATH:

    python3 tests/test_anonymized_text_gate.py
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from shared import AnonymizedTextMissingError, get_document_for_upload  # noqa: E402


def test_missing_anonymized_file_raises():
    entry = {
        "filename": "bescheid.pdf",
        "is_anonymized": True,
        "anonymization_metadata": {"anonymized_text_path": "/app/anonymized_text/does-not-exist.txt"},
        "extracted_text_path": "/app/ocr_text/does-not-exist.txt",
        "file_path": "/app/uploads/does-not-exist.pdf",
    }
    try:
        get_document_for_upload(entry)
    except AnonymizedTextMissingError:
        pass
    else:
        raise AssertionError("expected AnonymizedTextMissingError when anonymized_text_path is missing")


def test_missing_anonymized_path_key_raises():
    # is_anonymized True but metadata has no anonymized_text_path at all
    # (the exact bug this task fixes: metadata got replaced, dropping the key
    # while is_anonymized stayed True).
    entry = {
        "filename": "bescheid.pdf",
        "is_anonymized": True,
        "anonymization_metadata": {"ocr_text_length": 123},
        "extracted_text_path": "/app/ocr_text/does-not-exist.txt",
        "file_path": "/app/uploads/does-not-exist.pdf",
    }
    try:
        get_document_for_upload(entry)
    except AnonymizedTextMissingError:
        pass
    else:
        raise AssertionError("expected AnonymizedTextMissingError when anonymized_text_path key is absent")


def test_unreadable_anonymized_file_raises():
    with tempfile.TemporaryDirectory() as tmp_dir:
        anon_path = Path(tmp_dir) / "anon.txt"
        anon_path.write_text("anonymisierter text", encoding="utf-8")
        os.chmod(anon_path, 0)  # remove all permissions -> unreadable
        try:
            entry = {
                "filename": "bescheid.pdf",
                "is_anonymized": True,
                "anonymization_metadata": {"anonymized_text_path": str(anon_path)},
            }
            if os.access(anon_path, os.R_OK):
                # Running as root (e.g. inside the container) bypasses chmod 0;
                # skip this specific assertion in that environment.
                print("[SKIP] test_unreadable_anonymized_file_raises (running as root, chmod 0 has no effect)")
                return
            try:
                get_document_for_upload(entry)
            except AnonymizedTextMissingError:
                pass
            else:
                raise AssertionError("expected AnonymizedTextMissingError for unreadable anonymized file")
        finally:
            os.chmod(anon_path, 0o644)


def test_valid_anonymized_file_is_used():
    with tempfile.TemporaryDirectory() as tmp_dir:
        anon_path = Path(tmp_dir) / "anon.txt"
        anon_path.write_text("anonymisierter text", encoding="utf-8")
        entry = {
            "filename": "bescheid.pdf",
            "is_anonymized": True,
            "anonymization_metadata": {"anonymized_text_path": str(anon_path)},
        }
        selected_path, mime_type, needs_cleanup = get_document_for_upload(entry)
        assert selected_path == str(anon_path), selected_path
        assert mime_type == "text/plain", mime_type
        assert needs_cleanup is False


def test_non_anonymized_document_still_falls_back_normally():
    # is_anonymized False (or absent) -- existing OCR/raw fallback chain must
    # keep working unchanged.
    with tempfile.TemporaryDirectory() as tmp_dir:
        ocr_path = Path(tmp_dir) / "ocr.txt"
        ocr_path.write_text("ocr text", encoding="utf-8")
        entry = {
            "filename": "bescheid.pdf",
            "is_anonymized": False,
            "anonymization_metadata": None,
            "extracted_text_path": str(ocr_path),
        }
        selected_path, mime_type, _ = get_document_for_upload(entry)
        assert selected_path == str(ocr_path)
        assert mime_type == "text/plain"


def main():
    tests = [
        test_missing_anonymized_file_raises,
        test_missing_anonymized_path_key_raises,
        test_unreadable_anonymized_file_raises,
        test_valid_anonymized_file_is_used,
        test_non_anonymized_document_still_falls_back_normally,
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

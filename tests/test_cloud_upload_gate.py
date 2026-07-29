"""Category policy of the cloud upload gate (Datenschutzkonzept M2c).

get_document_for_upload is the single choke point for every cloud upload.
These tests pin the policy:
  - Akte never goes to the cloud.
  - Bescheid/Anhörung/Mandantenunterlagen/Vorinstanz only pseudonymized.
  - Sonstige Quellen raw only with the explicit per-request opt-in flag.
  - Rechtsprechung and category-less entries (ResearchSources) pass raw.
  - is_anonymized always wins: the anonymized text is served, no category check.
"""
import os

import pytest

from shared import (
    AnonymizedTextMissingError,
    CloudUploadBlockedError,
    get_document_for_upload,
)


def _entry(tmp_path, category=None, **overrides):
    text_file = tmp_path / "doc.txt"
    text_file.write_text("OCR-Text", encoding="utf-8")
    entry = {
        "filename": "doc.pdf",
        "file_path": None,
        "extracted_text_path": str(text_file),
        "anonymization_metadata": None,
        "is_anonymized": False,
        "category": category,
    }
    entry.update(overrides)
    return entry


def test_akte_is_always_blocked(tmp_path):
    with pytest.raises(CloudUploadBlockedError, match="Akten"):
        get_document_for_upload(_entry(tmp_path, category="Akte"))
    # even the Sonstiges opt-in flag must not unlock an Akte
    with pytest.raises(CloudUploadBlockedError):
        get_document_for_upload(
            _entry(tmp_path, category="Akte", allow_unanonymized_sonstiges=True)
        )


@pytest.mark.parametrize(
    "category", ["Bescheid", "Anhörung", "Mandantenunterlagen", "Vorinstanz"]
)
def test_client_categories_require_anonymization(tmp_path, category):
    with pytest.raises(CloudUploadBlockedError, match="anonymisiert"):
        get_document_for_upload(_entry(tmp_path, category=category))
    # the Sonstiges opt-in flag does not apply to these categories either
    with pytest.raises(CloudUploadBlockedError):
        get_document_for_upload(
            _entry(tmp_path, category=category, allow_unanonymized_sonstiges=True)
        )


def test_sonstiges_blocked_without_flag_allowed_with_flag(tmp_path):
    category = "Sonstige gespeicherte Quellen"
    with pytest.raises(CloudUploadBlockedError, match="allow_unanonymized_sonstiges"):
        get_document_for_upload(_entry(tmp_path, category=category))

    path, mime, _ = get_document_for_upload(
        _entry(tmp_path, category=category, allow_unanonymized_sonstiges=True)
    )
    assert mime == "text/plain"
    assert os.path.basename(path) == "doc.txt"


def test_rechtsprechung_and_uncategorized_pass_raw(tmp_path):
    for category in ("Rechtsprechung", None, ""):
        path, mime, _ = get_document_for_upload(_entry(tmp_path, category=category))
        assert mime == "text/plain"


def test_anonymized_version_wins_over_category(tmp_path):
    anon_file = tmp_path / "anon.txt"
    anon_file.write_text("[PERSON] hat Klage erhoben", encoding="utf-8")
    entry = _entry(
        tmp_path,
        category="Bescheid",
        is_anonymized=True,
        anonymization_metadata={"anonymized_text_path": str(anon_file)},
    )
    path, mime, _ = get_document_for_upload(entry)
    assert path == str(anon_file)
    assert mime == "text/plain"


def test_anonymized_flag_without_file_still_hard_fails(tmp_path):
    entry = _entry(
        tmp_path,
        category="Bescheid",
        is_anonymized=True,
        anonymization_metadata={"anonymized_text_path": str(tmp_path / "missing.txt")},
    )
    with pytest.raises(AnonymizedTextMissingError):
        get_document_for_upload(entry)

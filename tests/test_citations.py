import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

import sqlalchemy

sqlalchemy.create_engine = MagicMock(return_value=MagicMock())

import app as app_module  # type: ignore  # noqa: E402

verify_citations = app_module.verify_citations


def test_verify_citations_recognises_date_and_az():
    entry = {
        "filename": "bescheid1.pdf",
        "category": "Bescheid",
        "explanation": "Bescheid vom 15.03.2024, Az. M 1 K 24.123",
        "role": "primary",
    }
    collected = {"bescheid": [entry]}
    generated_text = (
        "Der Bescheid vom 15.03.2024, Az. M 1 K 24.123, Anlage K2, bestätigt das Vorbringen."
    )

    result = verify_citations(generated_text, collected)

    assert result["missing"] == []
    assert any("Bescheid" in label for label in result["cited"])
    assert all("Anlage K2" not in warn for warn in result["warnings"])


def test_verify_citations_flags_missing_reference():
    entry = {
        "filename": "bescheid1.pdf",
        "category": "Bescheid",
        "explanation": "Bescheid vom 15.03.2024, Az. M 1 K 24.123",
        "role": "primary",
    }
    collected = {"bescheid": [entry]}
    generated_text = "Der Bescheid wurde angegriffen, Anlage K2."  # lacks date/Az/filename

    result = verify_citations(generated_text, collected)

    assert result["missing"] == []
    assert any("Bescheid" in warn for warn in result["warnings"])

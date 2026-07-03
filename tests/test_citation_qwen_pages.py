import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

# citation_qwen -> shared -> database would spin up the real engine; stub it.
import types  # noqa: E402

sys.modules.setdefault(
    "shared",
    types.SimpleNamespace(ensure_anonymization_service_ready=lambda *a, **k: None),
)

from citation_qwen import expand_pages_for_document  # noqa: E402


def _large_report_document():
    """Inventory entry of a 405-page report: the prompt list is truncated to
    the first 80 pages, which must not limit page expansion."""
    return {
        "id": "doc-arc",
        "label": "Burundi Country Report ARCECOI Jan 2023.pdf",
        "category": "sonstiges",
        "pages": list(range(1, 81)),
    }


def _all_pages():
    return {page: f"Text {page}" for page in range(1, 406)}


def test_folgende_expansion_beyond_truncated_inventory():
    pages = expand_pages_for_document(
        [113], "ARC, Burundi Country Report, Januar 2023, S. 113 f.",
        _large_report_document(), available_pages=sorted(_all_pages()),
    )
    assert pages == [113, 114]


def test_fortfolgende_expansion_beyond_truncated_inventory():
    pages = expand_pages_for_document(
        [100], "ARC, Burundi Country Report, Januar 2023, S. 100 ff.",
        _large_report_document(), available_pages=sorted(_all_pages()),
    )
    assert pages and pages[0] == 100
    assert 113 in pages


def test_expansion_falls_back_to_inventory_pages_without_lookup():
    pages = expand_pages_for_document(
        [42], "S. 42 f.", _large_report_document(),
    )
    assert pages == [42, 43]

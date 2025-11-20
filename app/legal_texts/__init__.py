"""
Legal texts module - download, extract, and manage German legal provisions.
"""

from .models import LegalProvision, ProvisionsExtractionResult
from .extractor import extract_provision, parse_provision_reference
from .downloader import download_all_laws, update_law, get_law_path

__all__ = [
    "LegalProvision",
    "ProvisionsExtractionResult",
    "extract_provision",
    "parse_provision_reference",
    "download_all_laws",
    "update_law",
    "get_law_path",
]

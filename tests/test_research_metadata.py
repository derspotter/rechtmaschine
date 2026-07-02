"""Tests for per-engine metadata passthrough in aggregated research results.

The aggregate path rebuilds final metadata from a fixed literal, dropping
provider-specific health signals (structured_v2, verification, round_errors,
dropped_ungrounded). collect_engine_metadata preserves them under
metadata["engines"] without touching any existing key.

Run: .venv/bin/python -m pytest tests/test_research_metadata.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from endpoints.research.metadata import collect_engine_metadata  # noqa: E402


class _R:
    def __init__(self, metadata):
        self.metadata = metadata


def test_engines_keyed_by_provider_with_full_child_metadata():
    grok = _R({"provider": "grok-4.3", "structured_v2": True,
               "verification": {"verified": 3, "unverified": 0},
               "round_errors": [], "dropped_ungrounded": 1})
    asyl = _R({"provider": "asyl.net", "query_count": 1})

    engines = collect_engine_metadata([grok, asyl])

    assert engines["grok-4.3"]["verification"] == {"verified": 3, "unverified": 0}
    assert engines["grok-4.3"]["structured_v2"] is True
    assert engines["grok-4.3"]["dropped_ungrounded"] == 1
    assert engines["asyl.net"]["query_count"] == 1


def test_provider_collisions_and_missing_metadata_are_safe():
    a = _R({"provider": "grok-4.3", "x": 1})
    b = _R({"provider": "grok-4.3", "x": 2})
    c = _R(None)

    engines = collect_engine_metadata([a, b, c])

    assert engines["grok-4.3"]["x"] == 1
    assert engines["grok-4.3#2"]["x"] == 2
    assert len(engines) == 3  # the None-metadata engine still gets a slot


def test_search_engine_key_used_as_fallback_name():
    engines = collect_engine_metadata([_R({"search_engine": "chatgpt-search"})])
    assert "chatgpt-search" in engines

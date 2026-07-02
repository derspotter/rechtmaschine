"""Per-engine metadata passthrough for aggregated research results.

The multi-engine aggregate path rebuilds final_result.metadata from a fixed
literal, which silently dropped provider-specific health signals
(structured_v2, verification, round_errors, dropped_ungrounded — see plan
rollout log 2026-07-01). This helper preserves each child result's full
metadata under a per-provider key; the orchestrator nests it as
metadata["engines"] without changing any existing key. Pure module.
"""
from typing import Any, Dict, List


def collect_engine_metadata(results: List[Any]) -> Dict[str, Dict]:
    """{provider: child_metadata} for every child result, collision-safe."""
    engines: Dict[str, Dict] = {}
    for index, result in enumerate(results, start=1):
        child = getattr(result, "metadata", None) or {}
        provider = str(child.get("provider") or child.get("search_engine") or f"engine{index}").strip()
        key = provider
        suffix = 2
        while key in engines:
            key = f"{provider}#{suffix}"
            suffix += 1
        engines[key] = child
    return engines

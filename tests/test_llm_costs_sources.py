"""num_sources_used und server_side_tool_calls müssen aus der xAI
SamplingUsage ins Kosten-Log gelangen, damit cost_in_usd_ticks pro Run
dekomponierbar ist (Token-Anteil vs. web_search-Gebühr).
"""
from types import SimpleNamespace

from endpoints.research.llm_costs import (
    begin_collection,
    end_collection,
    record_grok_usage,
    usage_summary,
)


def _fake_sampling_usage(**overrides):
    base = dict(
        prompt_tokens=3144,
        completion_tokens=155,
        reasoning_tokens=251,
        cached_prompt_text_tokens=128,
        num_sources_used=3,
        server_side_tools_used=[1, 1],
        # 1 USD = 10^10 Ticks (amtlich, docs.x.ai) — NICHT 1e9
        cost_in_usd_ticks=85_100_000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_grok_record_carries_num_sources_used():
    token = begin_collection()
    try:
        record_grok_usage(_fake_sampling_usage(), "grok-4.5", source="grok-structured")
        summary = usage_summary()
    finally:
        end_collection(token)

    call = summary["calls"][0]
    assert call["num_sources_used"] == 3
    assert call["server_side_tool_calls"] == 2
    assert call["cost_usd"] == 0.0085
    assert summary["by_provider"]["xai"]["num_sources_used"] == 3
    assert summary["by_provider"]["xai"]["server_side_tool_calls"] == 2


def test_num_sources_aggregates_across_calls_and_defaults_to_zero():
    token = begin_collection()
    try:
        record_grok_usage(_fake_sampling_usage(num_sources_used=170), "grok-4.5", source="grok-structured")
        record_grok_usage(_fake_sampling_usage(num_sources_used=3), "grok-4.5", source="grok-search")
        # Usage-Objekt ohne das Feld (ältere SDKs) darf nicht crashen
        legacy = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost_in_usd_ticks=1000)
        record_grok_usage(legacy, "grok-4.3", source="grok-search")
        summary = usage_summary()
    finally:
        end_collection(token)

    assert summary["by_provider"]["xai"]["num_sources_used"] == 173
    assert summary["calls"][2]["num_sources_used"] == 0
    assert summary["calls"][2]["server_side_tool_calls"] == 0

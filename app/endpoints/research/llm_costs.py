"""Per-research-run LLM usage/cost collection.

Research runs fan out over several engines (chatgpt-search, grok, gemini,
meta relevance evaluation), none of which reported token usage into the
persisted result. This module collects every model call of one research run
via a ContextVar and summarizes it into `metadata.llm_usage`, so cost
questions are answerable from research_jobs.result_payload with SQL.

Collection is scoped by `begin_collection()`/`end_collection()` around
`_execute_research_request`. Engine call sites report through the
`record_*` helpers; outside an active collection they are no-ops, so the
helpers are safe to call from any other path.
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

_records: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "research_llm_usage_records", default=None
)

# Same env names/defaults as app/endpoints/generation.py so pricing is
# configured in one place (the environment), not two code constants.
_OPENAI_IN = float((os.getenv("OPENAI_GPT5_INPUT_PRICE_PER_1M", "2.5") or "2.5").strip())
_OPENAI_OUT = float((os.getenv("OPENAI_GPT5_OUTPUT_PRICE_PER_1M", "15") or "15").strip())
_GEMINI_PRO_IN = float((os.getenv("GEMINI_PRO_INPUT_PRICE_PER_1M", "2.5") or "2.5").strip())
_GEMINI_PRO_OUT = float((os.getenv("GEMINI_PRO_OUTPUT_PRICE_PER_1M", "15") or "15").strip())
_GEMINI_FLASH_IN = float((os.getenv("GEMINI_FLASH_INPUT_PRICE_PER_1M", "0.3") or "0.3").strip())
_GEMINI_FLASH_OUT = float((os.getenv("GEMINI_FLASH_OUTPUT_PRICE_PER_1M", "2.5") or "2.5").strip())


def begin_collection():
    """Start collecting for the current context. Returns a reset token."""
    return _records.set([])


def end_collection(token) -> None:
    try:
        _records.reset(token)
    except Exception:
        pass


def _estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    if provider == "openai":
        return input_tokens * _OPENAI_IN / 1e6 + output_tokens * _OPENAI_OUT / 1e6
    if provider == "google":
        if "flash" in (model or ""):
            return input_tokens * _GEMINI_FLASH_IN / 1e6 + output_tokens * _GEMINI_FLASH_OUT / 1e6
        return input_tokens * _GEMINI_PRO_IN / 1e6 + output_tokens * _GEMINI_PRO_OUT / 1e6
    return None  # anthropic/xai: xai reports its own cost, anthropic left token-only


def record_usage(
    provider: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    cost_usd: Optional[float] = None,
    source: str = "",
) -> None:
    records = _records.get()
    if records is None:
        return
    if cost_usd is None:
        cost_usd = _estimate_cost(provider, model, input_tokens, output_tokens)
    records.append(
        {
            "provider": provider,
            "model": model,
            "source": source,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "cached_tokens": int(cached_tokens or 0),
            "reasoning_tokens": int(reasoning_tokens or 0),
            "cost_usd": round(cost_usd, 4) if cost_usd is not None else None,
        }
    )


def record_openai_response(response: Any, model: str, source: str) -> None:
    """Record usage from an OpenAI Responses API response object."""
    try:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        details_in = getattr(usage, "input_tokens_details", None)
        details_out = getattr(usage, "output_tokens_details", None)
        record_usage(
            "openai",
            model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_tokens=getattr(details_in, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(details_out, "reasoning_tokens", 0) or 0,
            source=source,
        )
    except Exception as exc:  # usage tracking must never break research
        print(f"[LLM-COSTS] failed to record openai usage: {exc}")


def record_anthropic_response(response: Any, model: str, source: str) -> None:
    try:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        record_usage(
            "anthropic",
            model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            source=source,
        )
    except Exception as exc:
        print(f"[LLM-COSTS] failed to record anthropic usage: {exc}")


def record_gemini_response(response: Any, model: str, source: str) -> None:
    try:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return
        record_usage(
            "google",
            model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0)
            + (getattr(usage, "thoughts_token_count", 0) or 0),
            reasoning_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
            source=source,
        )
    except Exception as exc:
        print(f"[LLM-COSTS] failed to record gemini usage: {exc}")


def record_grok_usage(usage: Any, model: str, source: str) -> None:
    """xai_sdk SamplingUsage: token counts plus authoritative cost in USD
    ticks (1 tick = 1e-9 USD; 703278500 ticks ≈ $0.70, matches grok-4.3
    token+live-search pricing)."""
    try:
        if not usage:
            return
        ticks = getattr(usage, "cost_in_usd_ticks", 0) or 0
        record_usage(
            "xai",
            model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=getattr(usage, "cached_prompt_text_tokens", 0) or 0,
            reasoning_tokens=getattr(usage, "reasoning_tokens", 0) or 0,
            cost_usd=(ticks / 1e9) if ticks else None,
            source=source,
        )
    except Exception as exc:
        print(f"[LLM-COSTS] failed to record grok usage: {exc}")


def usage_summary() -> Optional[Dict[str, Any]]:
    """Summarize the current collection (None if no calls were recorded)."""
    records = _records.get()
    if not records:
        return None
    by_provider: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        agg = by_provider.setdefault(
            rec["provider"],
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        agg["calls"] += 1
        agg["input_tokens"] += rec["input_tokens"]
        agg["output_tokens"] += rec["output_tokens"]
        agg["cost_usd"] = round(agg["cost_usd"] + (rec["cost_usd"] or 0.0), 4)
    return {
        "calls": records,
        "by_provider": by_provider,
        "total_cost_usd": round(
            sum(rec["cost_usd"] or 0.0 for rec in records), 4
        ),
    }

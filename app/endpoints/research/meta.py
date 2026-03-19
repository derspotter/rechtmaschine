"""
Meta-search aggregation and relevance evaluation.
"""

import asyncio
import os
import json
import traceback
from typing import Dict, List, Any, Optional

from google.genai import types
from shared import (
    ResearchCaseProfile,
    ResearchResult,
    get_anthropic_client,
    get_gemini_client,
    get_openai_client,
)
from .source_quality import canonical_url


RELEVANCE_MIN_SCORE = 5
META_RELEVANCE_MODEL = (
    os.getenv("META_RELEVANCE_MODEL", "gemini-3.1-pro-preview").strip()
    or "gemini-3.1-pro-preview"
)
META_RELEVANCE_MAX_OUTPUT_TOKENS = 2400


def _is_generate_sota_model(model: str) -> bool:
    return model in {
        "claude-opus-4-6",
        "gpt-5.4",
        "gpt-5.2",
        "gemini-3-pro-preview",
        "gemini-3.1-pro-preview",
    }


def _resolve_meta_model() -> str:
    model = META_RELEVANCE_MODEL
    if _is_generate_sota_model(model):
        return model
    if model and model.lower().startswith("claude"):
        return model
    if model and model.lower().startswith("gpt"):
        return model
    if model and model.lower().startswith("gemini"):
        return model
    return "claude-opus-4-6"


def _to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _extract_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    if not raw:
        return None

    content = (raw or "").strip()
    if not content:
        return None

    content = content.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(content[start : end + 1])
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        return None


def _truncate_text(value: Any, max_length: int = 320) -> str:
    text = _to_str(value)
    if len(text) <= max_length:
        return text
    return f"{text[:max_length].rstrip()}…"


def _normalize_source_for_meta(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": _to_str(source.get("title"), "No Title"),
        "url": _to_str(source.get("url"), "No URL"),
        "original_url": _to_str(source.get("original_url"), ""),
        "resolved_url": _to_str(source.get("resolved_url"), ""),
        "description": _truncate_text(source.get("description"), 280),
        "summary": _truncate_text(source.get("summary") or source.get("description"), 320),
        "provider": _to_str(source.get("provider") or source.get("source")),
        "source": _to_str(source.get("source") or source.get("provider")),
        "document_type": _to_str(source.get("document_type"), "Rechtsprechung"),
        "evidence_type": _to_str(source.get("evidence_type"), "web"),
        "publication_year": source.get("publication_year"),
        "case_number": source.get("case_number"),
        "case_numbers": source.get("case_numbers"),
        "court": source.get("court"),
        "keywords": source.get("keywords"),
        "paragraphs": source.get("paragraphs"),
        "grounding_segments": source.get("grounding_segments"),
        "search_queries": source.get("search_queries"),
    }


def _normalize_evaluation_item(raw_item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_item, dict):
        return None

    idx = raw_item.get("index")
    if idx is None:
        return None
    if isinstance(idx, str) and not idx.isdigit():
        return None

    keep = raw_item.get("keep", False)
    score = _to_int(raw_item.get("score"), 0)
    reasoning = _to_str(raw_item.get("reasoning"), "")
    match_notes = raw_item.get("match_notes")
    if not isinstance(match_notes, list):
        match_notes = []
    normalized_match_notes = [_to_str(item) for item in match_notes if _to_str(item)]
    outcome_relevance = _to_str(raw_item.get("outcome_relevance"), "")
    authority_level = _to_str(raw_item.get("authority_level"), "")
    return {
        "index": int(idx),
        "keep": bool(keep),
        "score": score,
        "reasoning": reasoning,
        "match_notes": normalized_match_notes[:6],
        "outcome_relevance": outcome_relevance,
        "authority_level": authority_level,
    }


def _extract_anthropic_text(response: Any) -> str:
    content_parts: List[str] = []
    for item in getattr(response, "content", []) or []:
        if getattr(item, "type", None) == "text":
            content_parts.append(getattr(item, "text", "") or "")
    return "".join(content_parts).strip()


def _extract_openai_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if text:
        return text

    output = getattr(response, "output", None) or []
    parts: List[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type in ("output_text", "text"):
                parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


async def _call_meta_relevance_model(prompt: str, sources_text: str, model: str) -> str:
    normalized_model = (model or "").strip()
    payload = prompt + "\n\nCandidates JSON:\n" + sources_text
    if normalized_model.startswith("claude"):
        client = get_anthropic_client()
        response = await asyncio.to_thread(
            client.messages.create,
            model=normalized_model,
            max_tokens=META_RELEVANCE_MAX_OUTPUT_TOKENS,
            temperature=0.0,
            messages=[{"role": "user", "content": payload}],
        )
        return _extract_anthropic_text(response)

    if normalized_model.startswith("gpt"):
        client = get_openai_client()
        response = await asyncio.to_thread(
            client.responses.create,
            model=normalized_model,
            input=payload,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
            max_output_tokens=META_RELEVANCE_MAX_OUTPUT_TOKENS,
        )
        return _extract_openai_text(response)

    if normalized_model.startswith("gemini"):
        client = get_gemini_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=normalized_model,
            contents=[payload],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        return getattr(response, "text", "") or ""

    # Fallback to current default Opus if a custom model is configured incorrectly.
    client = get_anthropic_client()
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-opus-4-6",
        max_tokens=META_RELEVANCE_MAX_OUTPUT_TOKENS,
        temperature=0.0,
        messages=[{"role": "user", "content": payload}],
    )
    return _extract_anthropic_text(response)

async def evaluate_relevance(
    query: str,
    sources: List[Dict[str, Any]],
    case_profile: Optional[ResearchCaseProfile] = None,
    min_score: int = RELEVANCE_MIN_SCORE,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluate and rank research sources using one of the generation SOTA models.
    """
    if not sources:
        return [], []

    print(f"[META] Evaluating relevance for {len(sources)} sources...")

    candidate_sources: List[Dict[str, Any]] = []
    for idx, src in enumerate(sources):
        source_payload = _normalize_source_for_meta(src)
        source_payload["index"] = idx
        candidate_sources.append(source_payload)

    if not candidate_sources:
        return [], []

    sources_text = json.dumps(candidate_sources, ensure_ascii=False, indent=2)
    case_profile_text = json.dumps(
        case_profile.model_dump() if case_profile else {},
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""You are a senior legal research assistant for German Asylum Law.
Your task is to evaluate the relevance of the following source candidates for the user's query and the extracted case profile.
Use the structured metadata to avoid duplicate recommendations.
Pick only high-quality, recent German-relevant court decisions when available, but prioritize factual fit over generic topical overlap.

User Query: "{query}"

Case Profile / Search Plan / Ranking Profile:
{case_profile_text}

Each candidate has:
- index (for linking back to the original list)
- metadata: court, case_number, publication_year, keywords, paragraphs, evidence_type, document_type

Analyze each candidate and assign a relevance score (0-10) based on:
1. Direct fit to the concrete fact pattern and legal question
2. Fit to the risk / attribution / conflict mechanism described in the case profile
3. Fit to the procedural posture
4. Authority of the source (Courts > Official Agencies > Kommentierungen)
5. Timeliness (Newer is better, especially for case law)
6. Practical usefulness for claimant-side drafting or unavoidable counter-argument handling

For each source, provide:
- "score": Integer 0-10
- "reasoning": Brief explanation (1 sentence)
- "keep": Boolean (true if source should survive to final result set)
- "match_notes": list of short, concrete match or mismatch notes
- "outcome_relevance": claimant_positive | negative_if_authoritative | mixed | unclear
- "authority_level": low | medium | high

Deduplicate:
- If multiple candidates clearly refer to the same case, keep at most one.
- Prefer decisions with explicit court/aktenzeichen + publication_year.
- Negative decisions should only survive if they are authoritative or practically unavoidable.
- Downgrade sources that are only generally on-topic but miss the concrete fact pattern.

Output solely a JSON array of objects, one for each source in the input order:
[
  {{
    "index": 0,
    "score": 8,
    "reasoning": "...",
    "keep": true,
    "match_notes": ["...", "..."],
    "outcome_relevance": "claimant_positive",
    "authority_level": "high"
  }},
  ...
]
"""
    
    try:
        model = _resolve_meta_model()
        print(f"[META] Evaluating with model={model}")
        raw_model_output = await _call_meta_relevance_model(prompt, sources_text, model)

        raw_evaluations = _extract_json_array(raw_model_output)
        if raw_evaluations is None:
            raise ValueError("Model output could not be parsed as JSON array")

        parsed_evaluations: List[Dict[str, Any]] = []
        for raw_item in raw_evaluations:
            normalized = _normalize_evaluation_item(raw_item)
            if normalized is None:
                continue
            parsed_evaluations.append(normalized)

        if not parsed_evaluations:
            raise ValueError("No valid evaluation items from model")

        seen_duplicates: set[tuple] = set()
        ranked_sources: List[Dict[str, Any]] = []
        discarded_sources: List[Dict[str, Any]] = []
        for item in sorted(
            parsed_evaluations,
            key=lambda x: (x["keep"], x["score"]),
            reverse=True,
        ):
            idx = item["index"]
            if not (0 <= idx < len(sources)):
                continue

            source = dict(sources[idx])
            score = max(0, min(10, _to_int(item["score"], 0)))
            keep = bool(item.get("keep", False))
            match_notes = item.get("match_notes") or []
            outcome_relevance = _to_str(item.get("outcome_relevance"), "")
            authority_level = _to_str(item.get("authority_level"), "")

            evidence_key = (
                str(source.get("case_number") or "").strip().lower(),
                str(source.get("court") or "").strip().lower(),
                _to_int(source.get("publication_year"), 0),
                _to_str(source.get("url"), "").lower().strip(),
            )
            if evidence_key in seen_duplicates:
                keep = False
            elif score >= min_score:
                seen_duplicates.add(evidence_key)
            else:
                keep = False

            source["relevance_score"] = score
            source["relevance_reason"] = _to_str(item.get("reasoning"), "")
            source["match_notes"] = match_notes
            source["outcome_relevance"] = outcome_relevance
            source["authority_level"] = authority_level
            if keep:
                ranked_sources.append(source)
            else:
                discarded_sources.append(source)

        if not ranked_sources and parsed_evaluations:
            fallback_limit = min(3, len(parsed_evaluations))
            selected_for_fallback = []
            for item in sorted(parsed_evaluations, key=lambda x: x["score"], reverse=True)[:fallback_limit]:
                idx = item["index"]
                if not (0 <= idx < len(sources)):
                    continue
                source = dict(sources[idx])
                source["relevance_score"] = max(0, min(10, _to_int(item["score"], 0)))
                source["relevance_reason"] = _to_str(item.get("reasoning"), "Fallback kept due to low keep signal.")
                source["match_notes"] = item.get("match_notes") or []
                source["outcome_relevance"] = _to_str(item.get("outcome_relevance"), "")
                source["authority_level"] = _to_str(item.get("authority_level"), "")
                selected_for_fallback.append(source)
            ranked_sources.extend(selected_for_fallback)

        ranked_sources.sort(
            key=lambda source: (
                _to_int(source.get("relevance_score"), 0),
                _to_int(source.get("publication_year"), 0),
                _to_str(source.get("court"), ""),
            ),
            reverse=True,
        )

        print(f"[META] Kept {len(ranked_sources)}/{len(sources)} sources after evaluation.")
        return ranked_sources, discarded_sources

    except Exception as e:
        print(f"[META] Relevance evaluation failed: {e}")
        traceback.print_exc()
        # Fallback: validation failure, return all original sources
        fallback_sources = [dict(source) for source in sources]
        for item in fallback_sources:
            item["relevance_score"] = RELEVANCE_MIN_SCORE if item.get("evidence_type") == "decision_like" else 0
            item["relevance_reason"] = "Fallback ranking used (model parse failed)."
            item["match_notes"] = []
            item["outcome_relevance"] = ""
            item["authority_level"] = ""
        fallback_sources.sort(
            key=lambda source: (
                1 if source.get("evidence_type") == "decision_like" else 0,
                _to_int(source.get("publication_year"), 0),
                _to_str(source.get("court"), ""),
            ),
            reverse=True,
        )
        return fallback_sources, []

async def aggregate_search_results(
    query: str,
    results: List[ResearchResult],
    case_profile: Optional[ResearchCaseProfile] = None,
) -> ResearchResult:
    """
    Combine multiple ResearchResult objects into one, de-duplicate, and re-rank.
    """
    all_sources = []
    seen_urls = set()

    # Collect all sources
    for res in results:
        for src in res.sources:
            url = canonical_url(src.get("url") or "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append(src)

    # Evaluate relevance
    ranked, discarded = await evaluate_relevance(
        query,
        all_sources,
        case_profile=case_profile,
    )
    evaluation_model = _resolve_meta_model()

    # Combine summaries (naively for now, or use a generation model to synthesize)
    # Use the first available rich summary if present.
    combined_summary = ""
    for res in results:
        if res.summary and len(res.summary) > len(combined_summary):
            combined_summary = res.summary
    
    if not combined_summary:
        combined_summary = "**Meta-Search Results**\n\nAggregated results from multiple engines."

    # Collect all suggestions
    all_suggestions = []
    for res in results:
        if res.suggestions:
            all_suggestions.extend(res.suggestions)
    
    # Deduplicate suggestions
    unique_suggestions = list(dict.fromkeys(all_suggestions))

    query_count = 0
    filtered_count = 0
    reranked_count = 0
    duration_ms = 0
    for res in results:
        metadata = res.metadata or {}
        query_count += int(metadata.get("query_count", 0) or 0)
        filtered_count += int(metadata.get("filtered_count", 0) or 0)
        reranked_count += int(metadata.get("reranked_count", len(res.sources)) or 0)
        duration_ms += int(metadata.get("duration_ms", 0) or 0)

    metadata = {
        "provider": "meta",
        "model": "meta-aggregate",
        "evaluation_model": evaluation_model,
        "search_mode": "balanced",
        "max_sources": len(ranked),
        "domain_policy": "legal_balanced",
        "jurisdiction_focus": "de_eu",
        "recency_years": 6,
        "query_count": query_count,
        "filtered_count": filtered_count,
        "reranked_count": reranked_count,
        "source_count": len(ranked),
        "duration_ms": duration_ms,
    }

    return ResearchResult(
        query=query,
        summary=combined_summary,
        sources=ranked,
        discarded_sources=discarded,
        suggestions=unique_suggestions,
        metadata=metadata,
        case_profile=case_profile.model_dump() if case_profile else None,
    )

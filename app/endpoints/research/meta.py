"""
Meta-search aggregation and relevance evaluation.
"""

import asyncio
import json
import traceback
from typing import Dict, List, Any

from google.genai import types
from shared import get_gemini_client, ResearchResult

async def evaluate_relevance(
    query: str,
    sources: List[Dict[str, str]],
    min_score: int = 5
) -> List[Dict[str, Any]]:
    """
    Evaluate and rank research sources using Gemini 3.
    """
    if not sources:
        return []

    print(f"[META] Evaluating relevance for {len(sources)} sources...")

    # Prepare sources for the prompt
    sources_text = ""
    for idx, src in enumerate(sources):
        sources_text += f"SOURCE #{idx}:\n"
        sources_text += f"Title: {src.get('title', 'No Title')}\n"
        sources_text += f"URL: {src.get('url', 'No URL')}\n"
        sources_text += f"Description: {src.get('description', '')[:300]}\n"
        sources_text += "---\n"

    prompt = f"""You are a senior legal research assistant for German Asylum Law.
Your task is to evaluate the relevance of the following search results for the user's query.

User Query: "{query}"

Analyze each source and assign a relevance score (0-10) based on:
1. Direct relevance to the legal question (e.g. correct topic, country, legal issue)
2. Authority of the source (Courts > Official Agencies > NGOs > News)
3. Timeliness (Newer is better, especially for case law)

For each source, provide:
- "score": Integer 0-10
- "reasoning": Brief explanation (1 sentence)
- "keep": Boolean (true if score >= {min_score})

Output solely a JSON array of objects, one for each source in the input order:
[
  {{ "index": 0, "score": 8, "reasoning": "...", "keep": true }},
  ...
]
"""
    
    try:
        client = get_gemini_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash-preview-09-2025", # Using 2.5 Flash for speed/cost effectiveness dealing with many tokens, or switch to 3 if required for reasoning
            # model="gemini-3-pro-preview", # User requested Gemini 3 for relevance evaluation.
            contents=[prompt + "\n\n" + sources_text],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )

        try:
            evaluations = json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback for plain text response
            text = response.text.replace("```json", "").replace("```", "")
            evaluations = json.loads(text)

        # Map back to sources
        ranked_sources = []
        for eval_item in evaluations:
            idx = eval_item.get("index")
            if idx is not None and 0 <= idx < len(sources):
                source = sources[idx].copy()
                source["relevance_score"] = eval_item.get("score", 0)
                source["relevance_reason"] = eval_item.get("reasoning", "")
                
                if eval_item.get("keep", False):
                    ranked_sources.append(source)

        # Sort by score descending
        ranked_sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        print(f"[META] Kept {len(ranked_sources)}/{len(sources)} sources after evaluation.")
        return ranked_sources

    except Exception as e:
        print(f"[META] Relevance evaluation failed: {e}")
        traceback.print_exc()
        # Fallback: validation failure, return all original sources
        return sources

async def aggregate_search_results(
    query: str,
    results: List[ResearchResult]
) -> ResearchResult:
    """
    Combine multiple ResearchResult objects into one, de-duplicate, and re-rank.
    """
    all_sources = []
    seen_urls = set()

    # Collect all sources
    for res in results:
        for src in res.sources:
            url = src.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append(src)

    # Evaluate relevance
    ranked = await evaluate_relevance(query, all_sources)

    # Combine summaries (naively for now, or use Gemini to synthesize)
    # Using the summary from the first result (usually Grok/Gemini) if available
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

    return ResearchResult(
        query=query,
        summary=combined_summary,
        sources=ranked,
        suggestions=unique_suggestions
    )

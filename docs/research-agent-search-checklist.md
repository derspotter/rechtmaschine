# Research Endpoint Agent-Search Checklist

This checklist translates current agentic web-search best practices into concrete work for this codebase.

## P0: Reliability and Evidence Quality

- [ ] Add a shared source validation module for all providers.
  - Target: `app/endpoints/research/source_quality.py` (new), used by `app/endpoints/research/openai_search.py`, `app/endpoints/research/gemini.py`, `app/endpoints/research/grok.py`.
  - Scope: URL normalization, decision-signal scoring, off-topic penalties, duplicate clustering (by canonical URL + title fingerprint), and provenance fields (`provider`, `retrieved_at`, `evidence_type`).
  - Acceptance: all providers return sources with the same minimum schema and ranking behavior.

- [ ] Enforce citation provenance in all provider outputs.
  - Target: `app/endpoints/research/openai_search.py`, `app/endpoints/research/gemini.py`, `app/endpoints/research/grok.py`.
  - Scope: every source must include direct URL and non-empty description/snippet; summary text should only claim facts that are source-backed.
  - Acceptance: no source without URL; no empty source list when summary claims concrete decisions.

- [ ] Add a configurable domain policy (allowlist + blocklist) for high-stakes legal research.
  - Target: `app/endpoints/research/source_quality.py` (new), wired in `app/endpoints/research_sources.py`.
  - Scope: prioritize official court/legal portals; demote or block blogs/news/marketing pages.
  - Acceptance: ranking consistently prefers official/legal primary sources over secondary commentary.

## P0: Query Planning and Search Loop

- [ ] Standardize a 3-step retrieval loop across providers: analyze case -> generate search queries -> retrieve and rerank.
  - Target: `app/endpoints/research/openai_search.py`, `app/endpoints/research/gemini.py`, `app/endpoints/research/grok.py`.
  - Scope: provider prompt should explicitly produce/execute multiple focused queries (issue, jurisdiction, recency) before final summary.
  - Acceptance: logs show multiple focused searches for non-trivial cases; results improve on recency + relevance.

- [ ] Add explicit stop criteria for iterative search.
  - Target: provider modules above.
  - Scope: stop when additional query rounds do not add new high-quality sources (marginal gain threshold).
  - Acceptance: bounded latency with stable source quality.

## P0: Request and Response Contract

- [ ] Extend research request options with explicit search controls.
  - Target: `app/shared.py` (request model), `app/endpoints/research_sources.py` (routing).
  - Proposed fields: `search_mode` (`fast|balanced|deep`), `max_sources`, `domain_policy`, `jurisdiction_focus`, `recency_years`.
  - Acceptance: same request controls work for OpenAI, Grok, and Gemini paths.

- [ ] Return run metadata for reproducibility.
  - Target: `app/shared.py` response model and provider returns.
  - Proposed metadata: provider/model, search_mode, query_count, retrieval_duration_ms, filtered_count, reranked_count.
  - Acceptance: each research response includes enough metadata for benchmark and audit.

## P1: Ranking and Recency Improvements

- [ ] Improve decision-date extraction and use it in ranking.
  - Target: `app/endpoints/research/source_quality.py` (new), replace duplicated year/date logic in providers.
  - Scope: parse explicit dates from URL/title/snippet; fallback to year; reward newer decisions.
  - Acceptance: top results prefer recent, comparable decisions when relevance is equal.

- [ ] Keep diversity as soft reranking only (no hard per-court drops).
  - Target: `app/endpoints/research/openai_search.py`, `app/endpoints/research/gemini.py`, `app/endpoints/research/grok.py`.
  - Scope: diversity penalties should never remove otherwise relevant sources.
  - Acceptance: important repeated-court decisions still appear when they materially add value.

## P1: Evaluation Harness (Regression)

- [ ] Add benchmark dataset and scoring rubric for legal decision retrieval.
  - Target: `tests/research_compare.py` and `tests/fixtures/research_cases.json` (new).
  - Metrics: precision@k for court decisions, recency score, source authority score, duplicate ratio, off-topic ratio.
  - Acceptance: one command outputs per-provider comparison with stable metrics.

- [ ] Add adversarial/noise test cases.
  - Target: same fixture.
  - Scope: ambiguous facts, noisy OCR, misleading keywords, mixed jurisdictions.
  - Acceptance: providers avoid obvious off-topic/news/blog drift.

## P2: Performance and Cost Controls

- [ ] Add provider-specific timeout and retry policy by search mode.
  - Target: provider modules and `app/endpoints/research_sources.py`.
  - Scope: `fast` mode with strict latency; `deep` mode allows extra retrieval rounds.
  - Acceptance: predictable latency bands by mode.

- [ ] Add optional response/source caching for repeated queries.
  - Target: `app/endpoints/research_sources.py` and data layer (if enabled).
  - Scope: cache by normalized query + document fingerprint + mode + provider.
  - Acceptance: repeated identical runs are faster and reproducible.

## Suggested Implementation Order

- [ ] Step 1: Build `source_quality.py` and migrate common ranking/normalization there.
- [ ] Step 2: Add request controls and run metadata plumbing in `app/shared.py` and `app/endpoints/research_sources.py`.
- [ ] Step 3: Standardize provider search loop prompts and stop criteria.
- [ ] Step 4: Add benchmark fixtures and regression reporting in `tests/research_compare.py`.

## Quick Validation Command (after implementation)

```bash
python3 tests/research_compare.py --seed tests/fixtures/research_cases.json --providers grok,openai,gemini
```

## External Basis (best-practice sources)

- OpenAI web search + tools docs.
- Anthropic tool-use + citations docs.
- Google Gemini grounding docs.
- ReAct and WebGPT lines of work.
- BrowseComp-style evaluation framing.

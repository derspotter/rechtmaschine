#!/usr/bin/env python3
"""Benchmark research providers with strict dual-judge regression gating."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SEC = 180
CURRENT_YEAR = datetime.utcnow().year

DECISION_HINTS = (
    "entscheid",
    "urteil",
    "beschluss",
    "aktenzeichen",
    "ecli",
    "bverwg",
    "bverfg",
    "ovg",
    "verwaltungsgericht",
    "egmr",
    "eugh",
)

OFFICIAL_HINTS = (
    "nrwe.justiz.nrw.de",
    "justiz.nrw.de",
    "justiz.de",
    "bverwg.de",
    "bverfg.de",
    "juris.de",
    "dejure.org",
    "curia.europa",
    "hudoc.echr",
)

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass
class ProviderCaseResult:
    provider: str
    case_id: str
    query: str
    status: str
    elapsed_sec: float
    source_count: int
    decision_like_count: int
    heuristic_score: float
    recency_score: float
    judge_openai: Optional[float]
    judge_claude: Optional[float]
    consensus_score: float
    top_decisions: List[Dict[str, str]]
    sources: List[Dict[str, Any]] = field(default_factory=list)
    meta_source_coverage: Optional[Dict[str, int]] = None
    dedup_ratio: Optional[float] = None
    error: Optional[str] = None


def request_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    data: Optional[bytes] = None
    req_headers = dict(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = Request(url, method=method, data=data, headers=req_headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if not raw:
                return {}
            return json.loads(raw)
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def login(base_url: str, email: str, password: str, timeout: int) -> str:
    req = Request(
        f"{base_url.rstrip('/')}/token",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=urlencode({"username": email, "password": password}).encode("utf-8"),
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"Login failed ({exc.code}): {body}") from exc

    token = payload.get("access_token")
    if not token:
        raise RuntimeError("No access_token in /token response")
    return token


def _blob(source: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(source.get("url", "") or ""),
            str(source.get("title", "") or ""),
            str(source.get("description", "") or ""),
        ]
    ).lower()


def _is_official(source: Dict[str, Any]) -> bool:
    text = _blob(source)
    return any(token in text for token in OFFICIAL_HINTS)


def _decision_signal_count(source: Dict[str, Any]) -> int:
    text = _blob(source)
    return sum(1 for token in DECISION_HINTS if token in text)


def _extract_year(source: Dict[str, Any]) -> Optional[int]:
    text = _blob(source)
    years = [int(y) for y in YEAR_RE.findall(text)]
    years = [y for y in years if 2000 <= y <= CURRENT_YEAR + 1]
    if not years:
        return None
    return max(years)


def _url_key(source: Dict[str, Any]) -> str:
    return str(source.get("url", "") or "").strip().lower()


def _dedupe_with_provider_sources(
    results: List["ProviderCaseResult"],
) -> Tuple[List[Dict[str, Any]], Dict[str, Set[str]]]:
    by_url: Dict[str, Dict[str, Any]] = {}
    provider_coverage: Dict[str, Set[str]] = {}

    for item in results:
        if item.status != "ok":
            continue
        for source in item.sources:
            url = _url_key(source)
            if not url:
                continue
            if url not in by_url:
                by_url[url] = source
            provider_coverage.setdefault(url, set()).add(item.provider)

    deduped = [by_url[key] for key in by_url]
    return deduped, provider_coverage


def _build_meta_local(
    case_id: str,
    query: str,
    provider_results: List["ProviderCaseResult"],
    provider_count: int,
) -> ProviderCaseResult:
    deduped, provider_coverage = _dedupe_with_provider_sources(provider_results)
    source_count = len(deduped)
    heuristic, recency, decision_like, top_decisions = _heuristic_score(deduped)

    judge_score_openai = judge_openai(query, "meta-local", top_decisions)
    judge_score_claude = judge_claude(query, "meta-local", top_decisions)

    if judge_score_openai is not None and judge_score_claude is not None:
        consensus = (judge_score_openai + judge_score_claude) / 2.0
    elif judge_score_openai is not None:
        consensus = judge_score_openai
    elif judge_score_claude is not None:
        consensus = judge_score_claude
    else:
        consensus = heuristic

    total_urls = len(provider_coverage)
    dedup_ratio = None
    if provider_count > 0 and total_urls >= 0:
        raw_count = sum(len(item.sources) for item in provider_results if item.status == "ok")
        if raw_count:
            dedup_ratio = raw_count / total_urls if total_urls else None
    coverage_counter: Dict[str, int] = {}
    for providers in provider_coverage.values():
        coverage_counter[str(len(providers))] = coverage_counter.get(str(len(providers)), 0) + 1

    return ProviderCaseResult(
        provider="meta-local",
        case_id=case_id,
        query=query,
        status="ok",
        elapsed_sec=0.0,
        source_count=source_count,
        decision_like_count=decision_like,
        heuristic_score=heuristic,
        recency_score=recency,
        judge_openai=judge_score_openai,
        judge_claude=judge_score_claude,
        consensus_score=consensus,
        top_decisions=top_decisions,
        sources=deduped,
        meta_source_coverage=coverage_counter,
        dedup_ratio=dedup_ratio,
        error=None,
    )


def _heuristic_score(sources: List[Dict[str, Any]]) -> Tuple[float, float, int, List[Dict[str, str]]]:
    if not sources:
        return 0.0, 0.0, 0, []

    unique = []
    seen = set()
    for source in sources:
        url = str(source.get("url", "") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(source)

    scored = []
    decision_like_count = 0
    years = []
    for source in unique:
        decision_count = _decision_signal_count(source)
        official = _is_official(source)
        year = _extract_year(source)
        if decision_count > 0 or official:
            decision_like_count += 1
        if year:
            years.append(year)

        score = 0.0
        score += min(4.0, decision_count * 0.8)
        score += 2.5 if official else 0.0
        score += 1.5 if str(source.get("url", "")).lower().endswith(".pdf") else 0.0
        if year:
            score += min(2.0, max(0, year - (CURRENT_YEAR - 6)) * 0.35)
        scored.append((score, source, year))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:10]
    avg_score = statistics.mean(item[0] for item in top) if top else 0.0
    recency = statistics.mean(years) if years else 0.0
    top_decisions = [
        {
            "title": str(item[1].get("title", "") or "")[:160],
            "url": str(item[1].get("url", "") or "")[:240],
            "year": str(item[2] or ""),
        }
        for item in top[:5]
    ]
    return avg_score, recency, decision_like_count, top_decisions


def _judge_payload(query: str, provider: str, top_decisions: List[Dict[str, str]]) -> str:
    return (
        "Bewerte die Qualität von Suchergebnissen für juristische Primärentscheidungen.\n"
        f"Anfrage: {query}\n"
        f"Provider: {provider}\n"
        "Top-Quellen (Titel/URL/Jahr):\n"
        f"{json.dumps(top_decisions, ensure_ascii=False)}\n\n"
        "Antworte nur als JSON: {\"score\": <0-10 Zahl>, \"reason\": \"kurz\"}.\n"
        "Kriterien: Relevanz, Aktualität, Primärquellen-Qualität, wenig Off-Topic."
    )


def _extract_score(raw_text: str) -> Optional[float]:
    if not raw_text:
        return None
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict) and "score" in obj:
            return float(obj["score"])
    except Exception:
        pass

    match = re.search(r"(-?\d+(?:\.\d+)?)", raw_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def judge_openai(query: str, provider: str, top_decisions: List[Dict[str, str]]) -> Optional[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.2-mini")
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": _judge_payload(query, provider, top_decisions)}],
            }
        ],
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 300,
    }
    try:
        result = request_json(
            method="POST",
            url="https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload=payload,
            timeout=60,
        )
        output_text = result.get("output_text") or ""
        return _extract_score(str(output_text))
    except Exception:
        return None


def judge_claude(query: str, provider: str, top_decisions: List[Dict[str, str]]) -> Optional[float]:
    cli_timeout = int(os.getenv("CLAUDE_JUDGE_CLI_TIMEOUT_SEC", "120") or "120")
    prompt = _judge_payload(query, provider, top_decisions)

    # Prefer Claude CLI (signed-in session) for judging.
    try:
        completed = subprocess.run(
            [
                "env",
                "-u",
                "ANTHROPIC_API_KEY",
                "-u",
                "CLAUDE_API_KEY",
                "claude",
                "-p",
                prompt,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=cli_timeout,
            check=False,
        )
        if completed.returncode == 0:
            score = _extract_score((completed.stdout or "").strip())
            if score is not None:
                return score
    except Exception:
        pass

    # Fallback to Anthropic API if CLI is unavailable or failed.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    model = os.getenv("CLAUDE_JUDGE_MODEL", "claude-opus-4-7")
    payload = {
        "model": model,
        "max_tokens": 220,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": _judge_payload(query, provider, top_decisions)}],
    }
    try:
        result = request_json(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout=60,
        )
        text = ""
        for item in result.get("content", []) or []:
            if item.get("type") == "text":
                text += item.get("text", "")
        return _extract_score(text)
    except Exception:
        return None


def run_case(
    *,
    base_url: str,
    token: str,
    case: Dict[str, Any],
    provider: str,
    timeout: int,
    search_mode: str,
    max_sources: int,
    domain_policy: str,
    jurisdiction_focus: str,
    recency_years: int,
) -> ProviderCaseResult:
    payload: Dict[str, Any] = {
        "query": case.get("query", ""),
        "search_engine": provider,
        "search_mode": search_mode,
        "max_sources": max_sources,
        "domain_policy": domain_policy,
        "jurisdiction_focus": jurisdiction_focus,
        "recency_years": recency_years,
    }
    reference_document_id = case.get("reference_document_id")
    if reference_document_id:
        payload["reference_document_id"] = reference_document_id

    start = time.perf_counter()
    response = None
    error = None
    try:
        response = request_json(
            method="POST",
            url=f"{base_url.rstrip('/')}/research",
            headers={"Authorization": f"Bearer {token}"},
            payload=payload,
            timeout=timeout,
        )
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - start
    case_id = str(case.get("id", case.get("query", "case"))).strip() or "case"
    query = str(case.get("query", "") or "").strip()
    response_query = query

    if error:
        return ProviderCaseResult(
            provider=provider,
            case_id=case_id,
            query=response_query,
            status="error",
            elapsed_sec=elapsed,
            source_count=0,
            decision_like_count=0,
            heuristic_score=0.0,
            recency_score=0.0,
            judge_openai=None,
            judge_claude=None,
            consensus_score=0.0,
            top_decisions=[],
            sources=[],
            error=error,
        )

    if not response_query and isinstance(response, dict):
        response_query = str(response.get("query", query) or "").strip()

    sources = response.get("sources") if isinstance(response, dict) else []
    if not isinstance(sources, list):
        sources = []

    heuristic, recency, decision_like, top_decisions = _heuristic_score(sources)
    score_openai = judge_openai(query, provider, top_decisions)
    score_claude = judge_claude(query, provider, top_decisions)

    if score_openai is not None and score_claude is not None:
        consensus = (score_openai + score_claude) / 2.0
    elif score_openai is not None:
        consensus = score_openai
    elif score_claude is not None:
        consensus = score_claude
    else:
        consensus = heuristic

    return ProviderCaseResult(
        provider=provider,
        case_id=case_id,
        query=response_query,
        status="ok",
        elapsed_sec=elapsed,
        source_count=len(sources),
        decision_like_count=decision_like,
        heuristic_score=heuristic,
        recency_score=recency,
        judge_openai=score_openai,
        judge_claude=score_claude,
        consensus_score=consensus,
        top_decisions=top_decisions,
        sources=sources,
        error=None,
    )


def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and "cases" in raw:
        raw = raw.get("cases")
    if not isinstance(raw, list):
        raise RuntimeError(f"Seed file {path} must contain a list of cases")
    return [
        item for item in raw
        if isinstance(item, dict) and (
            isinstance(item.get("query"), str) and bool(item.get("query", "").strip())
            or item.get("reference_document_id")
        )
    ]


def result_key(item: ProviderCaseResult) -> str:
    return f"{item.provider}::{item.case_id}"


def build_baseline_payload(results: List[ProviderCaseResult]) -> Dict[str, Any]:
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "items": {
            result_key(item): {
                "provider": item.provider,
                "case_id": item.case_id,
                "query": item.query,
                "judge_openai": item.judge_openai,
                "judge_claude": item.judge_claude,
                "consensus_score": item.consensus_score,
            }
            for item in results
            if item.status == "ok"
        },
    }


def evaluate_regression(
    results: List[ProviderCaseResult],
    baseline: Dict[str, Any],
    threshold: float,
) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    baseline_items = baseline.get("items", {}) if isinstance(baseline, dict) else {}
    for item in results:
        if item.status != "ok":
            continue
        key = result_key(item)
        prev = baseline_items.get(key)
        if not isinstance(prev, dict):
            continue

        prev_openai = prev.get("judge_openai")
        prev_claude = prev.get("judge_claude")
        if (
            prev_openai is None
            or prev_claude is None
            or item.judge_openai is None
            or item.judge_claude is None
        ):
            continue

        delta_openai = float(item.judge_openai) - float(prev_openai)
        delta_claude = float(item.judge_claude) - float(prev_claude)
        if delta_openai < -threshold and delta_claude < -threshold:
            failures.append(
                f"{key} regressed: OpenAI {delta_openai:.2f}, Claude {delta_claude:.2f}"
            )

    return (len(failures) == 0, failures)


def print_table(results: List[ProviderCaseResult]) -> None:
    print(
        f"{'Provider':<16} {'Case':<18} {'Status':<8} {'Zeit':>6} {'Src':>4} "
        f"{'Dec':>4} {'Heur':>6} {'OAI':>5} {'CLD':>5} {'Cons':>6}"
    )
    print("-" * 96)
    for item in sorted(results, key=lambda r: r.consensus_score, reverse=True):
        print(
            f"{item.provider:<16} {item.case_id[:18]:<18} {item.status:<8} {item.elapsed_sec:>5.1f}s "
            f"{item.source_count:>4} {item.decision_like_count:>4} {item.heuristic_score:>6.2f} "
            f"{(item.judge_openai if item.judge_openai is not None else float('nan')):>5.2f} "
            f"{(item.judge_claude if item.judge_claude is not None else float('nan')):>5.2f} "
            f"{item.consensus_score:>6.2f}"
        )
        if item.error:
            print(f"  -> {item.error}")
        if item.top_decisions:
            for idx, source in enumerate(item.top_decisions[:3], start=1):
                title = source.get("title", "")[:96]
                year = source.get("year", "")
                url = source.get("url", "")
                year_label = f" ({year})" if year else ""
                print(f"     {idx}. {title}{year_label}")
                print(f"        {url}")
        if item.meta_source_coverage:
            breakdown = ", ".join(
                f"{providers} providers: {count}"
                for providers, count in sorted(item.meta_source_coverage.items(), key=lambda item: int(item[0]))
            )
            print(f"     Meta coverage: {breakdown}")
            if item.dedup_ratio is not None:
                print(f"     Dedup ratio (raw/dedup): {item.dedup_ratio:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Research provider benchmark with dual-judge gating")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=None)
    parser.add_argument("--email", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--seed", default="tests/fixtures/research_cases.json")
    parser.add_argument("--baseline", default="tests/fixtures/research_baseline.json")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--strict-gate", dest="strict_gate", action="store_true", default=True)
    parser.add_argument("--no-strict-gate", dest="strict_gate", action="store_false")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--provider", action="append", default=[])
    parser.add_argument("--parallel-per-case", dest="parallel_per_case", action="store_true", default=True)
    parser.add_argument("--no-parallel-per-case", dest="parallel_per_case", action="store_false")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--search-mode", default="balanced")
    parser.add_argument("--max-sources", type=int, default=12)
    parser.add_argument("--domain-policy", default="legal_balanced")
    parser.add_argument("--jurisdiction-focus", default="de_eu")
    parser.add_argument("--recency-years", type=int, default=6)
    parser.add_argument(
        "--meta-approach",
        choices=["api", "local"],
        default="api",
        help=(
            "How to evaluate meta mode: call /research meta directly (api) or "
            "aggregate provider rows locally (local)."
        ),
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    providers = args.provider or ["grok-4-1-fast", "chatgpt-search", "gemini"]
    cases = load_cases(args.seed)
    if not cases:
        print(f"No cases found in seed file: {args.seed}")
        return 2

    if args.token:
        token = args.token
    else:
        if not args.email or not args.password:
            print("Provide --token or --email + --password")
            return 2
        token = login(args.base_url, args.email, args.password, args.timeout)

    results: List[ProviderCaseResult] = []
    for case in cases:
        case_id = str(case.get("id", "case"))
        print(f"\nCase: {case_id} | {case.get('query','')[:120]}")

        run_targets = list(providers)
        local_meta = args.meta_approach == "local" and "meta" in providers
        if local_meta:
            run_targets = [provider for provider in providers if provider != "meta"]

        case_results: List[ProviderCaseResult] = []

        if args.parallel_per_case and len(run_targets) > 1:
            workers = max(1, min(args.max_workers, len(run_targets)))
            print(f"  -> running {len(run_targets)} providers concurrently (workers={workers})")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for provider in run_targets:
                    print(f"  - queued {provider}")
                    future = executor.submit(
                        run_case,
                        base_url=args.base_url,
                        token=token,
                        case=case,
                        provider=provider,
                        timeout=args.timeout,
                        search_mode=args.search_mode,
                        max_sources=args.max_sources,
                        domain_policy=args.domain_policy,
                        jurisdiction_focus=args.jurisdiction_focus,
                        recency_years=args.recency_years,
                    )
                    futures[future] = provider

                for future in as_completed(futures):
                    provider = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # defensive wrapper
                        result = ProviderCaseResult(
                            provider=provider,
                            case_id=case_id,
                            query=str(case.get("query", "")),
                            status="error",
                            elapsed_sec=0.0,
                            source_count=0,
                            decision_like_count=0,
                            heuristic_score=0.0,
                            recency_score=0.0,
                            judge_openai=None,
                            judge_claude=None,
                            consensus_score=0.0,
                            top_decisions=[],
                            sources=[],
                            error=str(exc),
                        )
                    case_results.append(result)
                    print(f"  - done {provider}: {result.status} ({result.elapsed_sec:.1f}s)")
        else:
            for provider in run_targets:
                print(f"  - {provider} ... ", end="", flush=True)
                result = run_case(
                    base_url=args.base_url,
                    token=token,
                    case=case,
                    provider=provider,
                    timeout=args.timeout,
                    search_mode=args.search_mode,
                    max_sources=args.max_sources,
                    domain_policy=args.domain_policy,
                    jurisdiction_focus=args.jurisdiction_focus,
                    recency_years=args.recency_years,
                )
                case_results.append(result)
                print(f"{result.status} ({result.elapsed_sec:.1f}s)")

        if local_meta:
            local_meta_generated = False
            if run_targets:
                success_targets = [item for item in case_results if item.provider in run_targets and item.status == "ok"]
                if success_targets:
                    print("  -> building local meta aggregation from provider results")
                    meta_result = _build_meta_local(
                        case_id=case_id,
                        query=str(case.get("query", "")),
                        provider_results=case_results,
                        provider_count=len(run_targets),
                    )
                    if meta_result.source_count:
                        local_meta_generated = True
                        case_results.append(meta_result)
                        local_meta_elapsed = max(item.elapsed_sec for item in success_targets)
                        meta_result.elapsed_sec = local_meta_elapsed
                else:
                    print("  -> no provider success for local meta, falling back to API meta call")

            if not local_meta_generated and "meta" in providers:
                try:
                    fallback_meta = run_case(
                        base_url=args.base_url,
                        token=token,
                        case=case,
                        provider="meta",
                        timeout=args.timeout,
                        search_mode=args.search_mode,
                        max_sources=args.max_sources,
                        domain_policy=args.domain_policy,
                        jurisdiction_focus=args.jurisdiction_focus,
                        recency_years=args.recency_years,
                    )
                    case_results.append(fallback_meta)
                    print(f"  - done meta (fallback): {fallback_meta.status} ({fallback_meta.elapsed_sec:.1f}s)")
                except Exception as exc:
                    case_results.append(
                        ProviderCaseResult(
                            provider="meta",
                            case_id=case_id,
                            query=str(case.get("query", "")),
                            status="error",
                            elapsed_sec=0.0,
                            source_count=0,
                            decision_like_count=0,
                            heuristic_score=0.0,
                            recency_score=0.0,
                            judge_openai=None,
                            judge_claude=None,
                            consensus_score=0.0,
                            top_decisions=[],
                            sources=[],
                            error=str(exc),
                        )
                )

            if run_targets and not local_meta_generated:
                print("  -> local meta had no deduplicated sources; fallback to API meta was attempted.")

        if not local_meta:
            results.extend(case_results)
            continue

        # For local meta, ensure source coverage is only built if meta row exists and keep it in order.
        case_results_sorted = sorted(
            case_results,
            key=lambda item: (0 if item.provider == "meta-local" else 1, item.provider),
        )
        results.extend(case_results_sorted)

    print("\n=== Benchmark Results ===")
    print_table(results)

    grouped: Dict[str, List[float]] = {}
    for item in results:
        grouped.setdefault(item.provider, []).append(item.consensus_score)
    ranking = [
        (provider, statistics.mean(scores), len(scores))
        for provider, scores in grouped.items()
    ]
    ranking.sort(key=lambda item: item[1], reverse=True)

    print("\n=== Provider Ranking (mean consensus) ===")
    for idx, (provider, score, count) in enumerate(ranking, start=1):
        print(f"{idx:>2}. {provider:<16} {score:.2f} (n={count})")

    baseline_payload = build_baseline_payload(results)
    if args.update_baseline:
        with open(args.baseline, "w", encoding="utf-8") as handle:
            json.dump(baseline_payload, handle, ensure_ascii=False, indent=2)
        print(f"\nUpdated baseline: {args.baseline}")

    gate_ok = True
    gate_failures: List[str] = []
    if args.strict_gate and os.path.exists(args.baseline):
        with open(args.baseline, "r", encoding="utf-8") as handle:
            baseline_data = json.load(handle)
        gate_ok, gate_failures = evaluate_regression(
            results=results,
            baseline=baseline_data,
            threshold=args.threshold,
        )
        print("\n=== Strict Gate ===")
        if gate_ok:
            print("PASS")
        else:
            print("FAIL")
            for failure in gate_failures:
                print(f" - {failure}")

    if args.output_json:
        output_payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "providers": providers,
            "seed": args.seed,
            "results": [item.__dict__ for item in results],
            "ranking": [
                {"provider": provider, "mean_consensus": score, "count": count}
                for provider, score, count in ranking
            ],
            "strict_gate": {"enabled": args.strict_gate, "pass": gate_ok, "failures": gate_failures},
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON report: {args.output_json}")

    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

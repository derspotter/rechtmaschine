#!/usr/bin/env python3
"""Compare research providers across multiple search engines.

This script calls /research for each configured engine, scores returned sources,
and ranks providers by a simple blend of freshness + decision relevance.

Usage:
  python tests/research_compare.py \
    --base-url http://127.0.0.1:8000 \
    --email admin@example.com --password admin123 \
    --engine grok-4-1-fast --engine chatgpt-search \
    --query "Befangenheit in Asylverfahren und Berufungszulassung" \
    --query "Asylverfahren, neue Entscheidungen OVG"
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_TIMEOUT_SEC = 180
CURRENT_YEAR = datetime.utcnow().year

official_hints = (
    "nrwe.justiz.nrw.de",
    "justiz.nrw.de",
    "nrw.justiz.de",
    "justiz.de",
    "verwaltungsgericht",
    "berlin.de",
    "bverwg",
    "bverfg",
    "juris.de",
    "dejure.org",
    "hudoc.echr",
    "curia.europa",
    "ec.europa",
    "ec.eu",
    "eur-lex",
)

decision_keywords = (
    "entscheid",
    "beschluss",
    "urteil",
    "aktenzeichen",
    "ecli",
    "rechtsgrundlage",
)

court_weights = {
    "bverfg": 35,
    "bverwg": 34,
    "bverf": 30,
    "bgh": 20,
    "eugh": 18,
    "egmr": 18,
    "ovg": 12,
    "vg berlin": 12,
    "verwaltungsgericht": 10,
}

date_re = re.compile(r"\b(?:(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2})\b")
year_re = re.compile(r"\b((?:19|20)\d{2})\b")
word_re = re.compile(r"[a-zäöüß0-9]{3,}")


@dataclass
class EngineResult:
    engine: str
    query: str
    status: str
    elapsed: float
    error: Optional[str]
    source_count: int
    unique_sources: int
    official_count: int
    decision_count: int
    freshness_year: int
    freshness_score: float
    relevance_score: float
    composite_score: float
    top_sources: List[Dict[str, Any]]



def request_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = BASE_TIMEOUT_SEC,
) -> Dict[str, Any]:
    data: Optional[bytes]
    if payload is None:
        data = None
        request_headers = dict(headers)
    else:
        data = json.dumps(payload).encode("utf-8")
        request_headers = dict(headers)
        request_headers.setdefault("Content-Type", "application/json")

    req = Request(url, method=method, data=data, headers=request_headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if not raw:
                return {}
            return json.loads(raw)
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def login(base_url: str, email: str, password: str, timeout: int) -> str:
    req = Request(
        f"{base_url.rstrip('/')}" + "/token",
        data=urlencode({"username": email, "password": password}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with urlopen(req, timeout=timeout) as response:
            token_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"Login failed ({exc.code}): {body}") from exc

    token = token_payload.get("access_token")
    if not token:
        raise RuntimeError("No access_token in /token response")
    return token


def normalize_terms(query: str) -> List[str]:
    lowered = (query or "").lower()
    stop_words = {
        "die", "der", "das", "und", "oder", "in", "im", "am", "mit", "für", "von", "bei", "nach", "wie", "auch",
        "über", "auf", "zu", "des", "ein", "eine", "ist", "sind", "werden", "wird", "als", "an", "den", "dem", "zuletzt",
    }
    terms = [m.group(0) for m in word_re.finditer(lowered)]
    filtered = [t for t in terms if len(t) >= 4 and t not in stop_words]
    return sorted(set(filtered))


def source_blob(source: Dict[str, Any]) -> str:
    return " ".join(
        str(source.get(key, "") or "")
        for key in ("url", "title", "description")
    ).lower()


def extract_year(text: str) -> Optional[int]:
    candidates = []
    date_hit = date_re.search(text)
    if date_hit:
        for part in re.sub(r"[./-]", ".", date_hit.group(0)).split("."):
            if part.isdigit() and len(part) == 4:
                year = int(part)
                if 2000 <= year <= CURRENT_YEAR + 1:
                    candidates.append(year)

    for group in year_re.findall(text):
        year = int(group)
        if 2000 <= year <= CURRENT_YEAR + 1:
            candidates.append(year)

    if not candidates:
        return None
    return max(candidates)


def score_source(source: Dict[str, Any], query_terms: List[str]) -> Dict[str, Any]:
    blob = source_blob(source)
    url = (source.get("url") or "").strip()
    title = (source.get("title") or "").strip()

    official = any(token in blob for token in official_hints)
    decision_count = sum(1 for kw in decision_keywords if kw in blob)

    court_score = 0
    for token, weight in court_weights.items():
        if token in blob:
            court_score = max(court_score, weight)

    query_score = sum(1 for term in query_terms if term in blob)
    year = extract_year(f"{url} {title} {blob}")

    recency_points = min((year - 2010), 25) if year and year >= 2010 else 0
    score = (
        (6 if official else 0)
        + min(decision_count * 2, 10)
        + min(court_score, 35)
        + min(query_score * 1.5, 20)
        + recency_points
        + (2 if "pdf" in blob else 0)
    )

    return {
        "url": url,
        "title": title,
        "description": source.get("description") or "",
        "source": source.get("source") or "n/a",
        "official": official,
        "decision_count": decision_count,
        "court_score": court_score,
        "year": year,
        "query_score": query_score,
        "score": score,
    }


def evaluate(engine: str, query: str, result: Optional[Dict[str, Any]], elapsed: float, error: Optional[str]) -> EngineResult:
    if error:
        return EngineResult(
            engine=engine,
            query=query,
            status="error",
            elapsed=elapsed,
            error=error,
            source_count=0,
            unique_sources=0,
            official_count=0,
            decision_count=0,
            freshness_year=0,
            freshness_score=0.0,
            relevance_score=0.0,
            composite_score=0.0,
            top_sources=[],
        )

    if not isinstance(result, dict):
        return EngineResult(
            engine=engine,
            query=query,
            status="invalid_payload",
            elapsed=elapsed,
            error="Response is not a JSON object.",
            source_count=0,
            unique_sources=0,
            official_count=0,
            decision_count=0,
            freshness_year=0,
            freshness_score=0.0,
            relevance_score=0.0,
            composite_score=0.0,
            top_sources=[],
        )

    if not result.get("sources"):
        return EngineResult(
            engine=engine,
            query=query,
            status="no_sources",
            elapsed=elapsed,
            error=None,
            source_count=0,
            unique_sources=0,
            official_count=0,
            decision_count=0,
            freshness_year=0,
            freshness_score=0.0,
            relevance_score=0.0,
            composite_score=0.0,
            top_sources=[],
        )

    terms = normalize_terms(query or "")
    seen_urls = set()
    scored_sources: List[Dict[str, Any]] = []
    for source in result.get("sources") or []:
        if not isinstance(source, dict):
            continue

        scored = score_source(source, terms)
        url = scored.get("url", "")
        if not url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        scored_sources.append(scored)

    if not scored_sources:
        return EngineResult(
            engine=engine,
            query=query,
            status="no_valid_sources",
            elapsed=elapsed,
            error=None,
            source_count=len(result.get("sources") or []),
            unique_sources=0,
            official_count=0,
            decision_count=0,
            freshness_year=0,
            freshness_score=0.0,
            relevance_score=0.0,
            composite_score=0.0,
            top_sources=[],
        )

    scored_sources.sort(key=lambda s: s["score"], reverse=True)
    top10 = scored_sources[:10]
    years = [s["year"] for s in scored_sources if s.get("year")]
    decision_scores = [
        s for s in scored_sources if s["official"] or s["decision_count"] > 0
    ]
    freshness_year = max(years) if years else 0
    freshness_score = statistics.mean(years) if years else 0.0
    relevance_score = statistics.mean(s["score"] for s in top10)

    return EngineResult(
        engine=engine,
        query=query,
        status="ok",
        elapsed=elapsed,
        error=None,
        source_count=len(result.get("sources") or []),
        unique_sources=len(scored_sources),
        official_count=sum(1 for s in scored_sources if s["official"]),
        decision_count=len(decision_scores),
        freshness_year=freshness_year,
        freshness_score=freshness_score,
        relevance_score=relevance_score,
        composite_score=0.0,
        top_sources=top10[:5],
    )


def run_query(
    base_url: str,
    token: str,
    query: str,
    engine: str,
    timeout: int,
    reference_document_id: Optional[str] = None,
) -> EngineResult:
    payload: Dict[str, Any] = {
        "query": query,
        "search_engine": engine,
    }
    if reference_document_id:
        payload["reference_document_id"] = reference_document_id

    headers = {"Authorization": f"Bearer {token}"}
    start = time.perf_counter()
    error = None
    response: Optional[Dict[str, Any]] = None

    try:
        response = request_json(
            method="POST",
            url=f"{base_url.rstrip('/')}" + "/research",
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - start
    return evaluate(engine, query, response, elapsed, error)


def score_all(results: List[EngineResult]) -> None:
    valid = [r for r in results if r.status == "ok"]
    if not valid:
        for result in results:
            result.composite_score = 0.0
        return

    max_freshness = max((r.freshness_score for r in valid), default=1.0)
    max_relevance = max((r.relevance_score for r in valid), default=1.0)

    for result in results:
        if result.status != "ok":
            result.composite_score = 0.0
            continue

        freshness_norm = result.freshness_score / max_freshness if max_freshness else 0.0
        relevance_norm = result.relevance_score / max_relevance if max_relevance else 0.0

        official_ratio = (result.official_count / result.unique_sources) if result.unique_sources else 0.0
        decision_ratio = (result.decision_count / result.unique_sources) if result.unique_sources else 0.0
        quality_mix = max(official_ratio, decision_ratio)
        breadth = min(1.0, result.unique_sources / 12.0)

        result.composite_score = (
            0.42 * freshness_norm
            + 0.42 * relevance_norm
            + 0.10 * quality_mix
            + 0.06 * breadth
        )


def print_result_table(query: str, results: List[EngineResult]) -> List[EngineResult]:
    print(f"\n=== Query: {query} ===")
    print(
        f"{'Engine':<18} {'Status':<15} {'Zeit':>6} {'Src':>4} {'uniq':>4} "
        f"{'Off':>3} {'Ent':>3} {'Neu':>4} {'Fresh':>7} {'Rel':>7} {'Score':>6}"
    )
    print("-" * 90)

    for result in sorted(results, key=lambda r: r.composite_score, reverse=True):
        status = result.status
        status_label = status
        if result.error:
            status_label = f"{status} ERR"
        print(
            f"{result.engine:<18} {status_label:<15} "
            f"{result.elapsed:>5.1f}s "
            f"{result.source_count:>4} {result.unique_sources:>4} {result.official_count:>3} "
            f"{result.decision_count:>3} {result.freshness_year:>4} {result.freshness_score:>7.1f} "
            f"{result.relevance_score:>7.1f} {result.composite_score:>6.2f}"
        )
        if result.error:
            print(f"    -> {result.error}")

    for result in sorted(results, key=lambda r: r.composite_score, reverse=True):
        if not result.top_sources:
            continue
        print(f"\nTop {result.engine} sources:")
        for idx, source in enumerate(result.top_sources, start=1):
            year = source.get("year")
            year_text = f" ({year})" if year else ""
            print(f"  {idx:>2}. {source.get('title','').strip()[:95] or source.get('url','')}{year_text}")
            print(f"      {source.get('source','n/a')} :: {source.get('url','')[:120]}")

    return sorted(results, key=lambda r: r.composite_score, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare research engines by result freshness and relevance")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of Rechtmaschine")
    parser.add_argument("--token", default=None, help="Bearer token for /research")
    parser.add_argument("--email", default=None, help="Login email if token is not provided")
    parser.add_argument("--password", default=None, help="Login password if token is not provided")
    parser.add_argument("--query", action="append", default=[], help="Query to run (repeatable)")
    parser.add_argument(
        "--engine",
        action="append",
        default=[],
        help="Engine to benchmark (repeatable)",
    )
    parser.add_argument("--reference-document-id", dest="reference_document_id", default=None)
    parser.add_argument("--timeout", type=int, default=BASE_TIMEOUT_SEC)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    query_list = args.query or [
        "Befangenheit in Asylverfahren und Berufungszulassung",
        "Aktuelle Rechtsprechung zu Asylverfahren Bayerisches Migrationsrecht",
    ]
    requested_engines = args.engine or ["grok-4-1-fast", "chatgpt-search"]
    engines = []
    for engine in requested_engines:
        if engine and engine not in engines:
            engines.append(engine)

    if not args.token:
        if not args.email or not args.password:
            print("Fehler: kein token und keine login-daten gesetzt.")
            print("Nutze --token oder --email + --password")
            return 2
        try:
            token = login(args.base_url, args.email, args.password, args.timeout)
        except Exception as exc:
            print(f"Fehler beim Login: {exc}")
            return 2
    else:
        token = args.token

    all_results: List[EngineResult] = []
    for query in query_list:
        print(f"\nStarte Query: {query}")
        query_results: List[EngineResult] = []

        for engine in engines:
            print(f"  - {engine} ...", end=" ", flush=True)
            result = run_query(
                base_url=args.base_url,
                token=token,
                query=query,
                engine=engine,
                timeout=args.timeout,
                reference_document_id=args.reference_document_id,
            )
            query_results.append(result)
            all_results.append(result)
            print(f"{result.status} ({result.elapsed:.1f}s)")

        score_all(query_results)
        print_result_table(query, query_results)

    if all_results:
        engine_scores: Dict[str, List[float]] = {}
        for result in all_results:
            engine_scores.setdefault(result.engine, []).append(result.composite_score)

        total_ranking = [
            (engine, statistics.mean(scores), len(scores))
            for engine, scores in engine_scores.items()
        ]
        total_ranking.sort(key=lambda x: x[1], reverse=True)

        print("\n=== Gesamtwertung (Durchschnitt über Queries) ===")
        for idx, (engine, mean_score, count) in enumerate(total_ranking, start=1):
            print(f"{idx:>2}. {engine:<18} {mean_score:.3f} (n={count})")

        best = total_ranking[0][0] if total_ranking else "n/a"
        print(f"\nBeste Engine insgesamt: {best}")

        if args.output_json:
            payload = {
                "generated_at": datetime.utcnow().isoformat(),
                "query_count": len(query_list),
                "engines": engines,
                "results": [
                    {
                        "engine": r.engine,
                        "query": r.query,
                        "status": r.status,
                        "elapsed": r.elapsed,
                        "source_count": r.source_count,
                        "unique_sources": r.unique_sources,
                        "official_count": r.official_count,
                        "decision_count": r.decision_count,
                        "freshness_year": r.freshness_year,
                        "freshness_score": r.freshness_score,
                        "relevance_score": r.relevance_score,
                        "composite_score": r.composite_score,
                    }
                    for r in all_results
                ],
            }
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"JSON-Output gespeichert: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

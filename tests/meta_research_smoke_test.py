#!/usr/bin/env python3
"""Smoke-test meta research endpoint and print provider/model diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, Optional

from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _json_request(url: str, payload: Dict[str, Any], token: str, timeout: int) -> Dict[str, Any]:
    req = Request(
        f"{url.rstrip('/')}/research",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    start = time.time()
    with urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    elapsed_ms = int((time.time() - start) * 1000)
    print(f"HTTP status: 200 (time={elapsed_ms}ms)")
    return json.loads(raw)


def _http_error(e: HTTPError) -> Dict[str, Any]:
    body = e.read().decode("utf-8") if e.fp else ""
    return {
        "error": "http",
        "status": getattr(e, "code", None),
        "body": body,
    }


def _require_token(base_url: str) -> Optional[str]:
    token = os.getenv("RESEARCH_TOKEN", "").strip()
    if token:
        return token

    email = os.getenv("RESEARCH_EMAIL")
    password = os.getenv("RESEARCH_PASSWORD")
    if not email or not password:
        return None

    login_payload = {
        "username": email,
        "password": password,
    }
    req = Request(
        f"{base_url.rstrip('/')}/token",
        method="POST",
        data=urlencode(login_payload).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("access_token", "")
    except HTTPError as exc:
        print(f"Token request failed ({exc.code}): {exc.read().decode('utf-8') if exc.fp else ''}")
    except URLError as exc:  # pragma: no cover
        print(f"Token request network error: {exc}")

    return None


def _top_sources(data: Dict[str, Any], limit: int = 8) -> None:
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        return

    print("\nTop sources:")
    for idx, src in enumerate(sources[:limit], start=1):
        if not isinstance(src, dict):
            continue
        title = (src.get("title") or "").strip()
        court = (src.get("court") or "").strip()
        case_number = (src.get("case_number") or "").strip()
        year = src.get("publication_year")
        score = src.get("relevance_score")
        provider = (src.get("provider") or src.get("source") or "").strip()
        print(f"  {idx:02d}. {title[:95]} | {court} {case_number} | {provider} | score={score} | year={year}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("RESEARCH_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--query",
        default="Aktuelle Entscheidungen zur Asylablehnung bei militärischem Zwangsdienst",
    )
    parser.add_argument("--max-sources", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--jurisdiction", default="de_eu")
    args = parser.parse_args()

    token = _require_token(args.base_url)
    if not token:
        print(
            "No auth token available. Set RESEARCH_TOKEN or RESEARCH_EMAIL/RESEARCH_PASSWORD.",
            file=sys.stderr,
        )
        return 1

    payload = {
        "query": args.query,
        "search_engine": "meta",
        "max_sources": args.max_sources,
        "search_mode": "balanced",
        "domain_policy": "legal_balanced",
        "jurisdiction_focus": args.jurisdiction,
        "recency_years": 6,
    }

    start = time.time()
    try:
        data = _json_request(args.base_url, payload, token, timeout=args.timeout)
    except HTTPError as exc:
        print(json.dumps(_http_error(exc), ensure_ascii=False, indent=2))
        return 1
    except URLError as exc:
        print(f"Request failed: {exc}")
        return 1

    metadata = data.get("metadata") if isinstance(data, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    print(f"Search provider: {metadata.get('provider')}")
    print(f"Model: {metadata.get('model')}")
    print(f"Meta evaluation model: {metadata.get('evaluation_model', 'not-set')}")
    print(f"Search mode: {metadata.get('search_mode')} / recency={metadata.get('recency_years')} / max_sources={metadata.get('max_sources')}")

    provider_sources = metadata.get("provider_sources")
    if isinstance(provider_sources, list) and provider_sources:
        print(f"Provider providers: {', '.join(provider_sources)}")

    providers = Counter()
    sources = data.get("sources") or []
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            providers[(source.get("provider") or source.get("source") or "unknown")] += 1

    if providers:
        provider_counts = ", ".join(f"{name}:{count}" for name, count in sorted(providers.items()))
        print(f"Source providers: {provider_counts}")

    print(f"Source count (returned/all): {len(sources)} | metadata source_count={metadata.get('source_count')}")
    print(f"Query time: {int((time.time()-start)*1000)}ms")

    _top_sources(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

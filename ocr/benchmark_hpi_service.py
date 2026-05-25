#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark the current Debian HPI OCR service through the service manager.

The output intentionally excludes OCR text. It records request timing, page counts,
confidence, peak VRAM, and per-page timing metadata for regression checks.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


def _page_summary(page: dict[str, Any]) -> dict[str, Any]:
    metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
    return {
        "page_index": page.get("page_index"),
        "line_count": page.get("line_count", len(page.get("lines") or [])),
        "render_dpi": metadata.get("render_dpi"),
        "render_size": metadata.get("render_size"),
        "requested_render_size": metadata.get("requested_render_size"),
        "render_seconds": metadata.get("render_seconds"),
        "effective_dpi": metadata.get("effective_dpi"),
        "render_limited": metadata.get("render_limited"),
        "predict_seconds": metadata.get("predict_seconds"),
        "total_seconds": metadata.get("total_seconds"),
        "attempt": metadata.get("attempt"),
        "fallback_used": metadata.get("fallback_used"),
    }


def _summarize_response(
    pdf_path: Path,
    service_url: str,
    request_id: str,
    client_seconds: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
    page_summaries = [
        _page_summary(page)
        for page in pages
        if isinstance(page, dict)
    ]
    slowest_pages = sorted(
        page_summaries,
        key=lambda item: item.get("total_seconds") or 0,
        reverse=True,
    )[:10]
    fallback_pages = [
        item
        for item in page_summaries
        if item.get("fallback_used")
    ]

    return {
        "request_id": payload.get("request_id") or request_id,
        "service_url": service_url,
        "filename": pdf_path.name,
        "source_path": str(pdf_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "client_seconds": round(client_seconds, 3),
        "page_count": payload.get("page_count", len(page_summaries)),
        "avg_confidence": payload.get("avg_confidence"),
        "metadata": payload.get("metadata", {}),
        "total_lines": sum(int(item.get("line_count") or 0) for item in page_summaries),
        "fallback_page_count": len(fallback_pages),
        "fallback_pages": fallback_pages,
        "slowest_pages": slowest_pages,
        "pages": page_summaries,
    }


def benchmark_pdf(
    pdf_path: Path,
    service_url: str,
    request_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    with pdf_path.open("rb") as file_handle:
        files = {"file": (pdf_path.name, file_handle, "application/pdf")}
        headers = {"X-Request-ID": request_id}
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(f"{service_url.rstrip('/')}/ocr", files=files, headers=headers)
            response.raise_for_status()
    client_seconds = time.perf_counter() - start_time
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("OCR service returned a non-object JSON response")
    return _summarize_response(pdf_path, service_url, request_id, client_seconds, payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a PDF against the Debian HPI OCR service manager."
    )
    parser.add_argument("pdf", type=Path, help="PDF file to benchmark")
    parser.add_argument(
        "--service-url",
        default="http://127.0.0.1:8004",
        help="Service manager URL (default: http://127.0.0.1:8004)",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Request ID to pass through logs (default: generated)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="HTTP timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print(f"[ERROR] Expected a PDF file: {pdf_path}", file=sys.stderr)
        return 1

    request_id = args.request_id or f"ocr-bench-{uuid.uuid4().hex[:12]}"
    print(f"[INFO] Benchmarking {pdf_path}")
    print(f"[INFO] request_id={request_id}")
    summary = benchmark_pdf(pdf_path, args.service_url, request_id, args.timeout)

    output_text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
        print(f"[INFO] Summary written to {args.output}")

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

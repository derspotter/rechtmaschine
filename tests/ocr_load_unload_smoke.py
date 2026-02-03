#!/usr/bin/env python3
"""
Smoke check for OCR hibernate service:
- /load
- warm OCR
- /unload
- cold OCR (auto-load)
- warm OCR again
- /unload

Reports timings and saves JSON output.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from pikepdf import Pdf


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _split_first_page(pdf_path: Path) -> Path:
    with Pdf.open(pdf_path) as pdf:
        if len(pdf.pages) == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        single = Pdf.new()
        single.pages.append(pdf.pages[0])
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_path = Path(handle.name)
        handle.close()
        single.save(temp_path)
        return temp_path


def _timed_post(
    client: httpx.Client,
    url: str,
    files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.time()
    response = client.post(url, files=files)
    duration = time.time() - start
    body = response.text
    return {
        "status": response.status_code,
        "duration": duration,
        "body": body[:500],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR load/unload smoke check")
    parser.add_argument(
        "--ocr-url",
        default="http://127.0.0.1:9003",
        help="OCR service base URL",
    )
    parser.add_argument(
        "--pdf",
        default="test_files/anlage_k4_bescheid.pdf",
        help="PDF file for OCR requests",
    )
    parser.add_argument(
        "--output",
        default="ocr_load_unload_smoke.json",
        help="Where to write JSON results",
    )
    args = parser.parse_args()

    ocr_url = args.ocr_url.rstrip("/")
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    page_path = _split_first_page(pdf_path)

    results: dict[str, Any] = {
        "ocr_url": ocr_url,
        "pdf": str(pdf_path),
        "page": str(page_path),
        "timings": {},
    }

    timeout = httpx.Timeout(300.0)
    with httpx.Client(timeout=timeout) as client:
        # Warm load
        results["timings"]["load"] = _timed_post(client, f"{ocr_url}/load")
        print(
            f"/load: {_format_seconds(results['timings']['load']['duration'])} "
            f"status={results['timings']['load']['status']}"
        )

        # Warm OCR
        with page_path.open("rb") as handle:
            files = {"file": (page_path.name, handle, "application/pdf")}
            results["timings"]["warm_ocr_1"] = _timed_post(
                client, f"{ocr_url}/ocr", files=files
            )
        print(
            f"warm OCR: {_format_seconds(results['timings']['warm_ocr_1']['duration'])} "
            f"status={results['timings']['warm_ocr_1']['status']}"
        )

        # Unload
        results["timings"]["unload"] = _timed_post(client, f"{ocr_url}/unload")
        print(
            f"/unload: {_format_seconds(results['timings']['unload']['duration'])} "
            f"status={results['timings']['unload']['status']}"
        )

        # Cold OCR (auto-load)
        with page_path.open("rb") as handle:
            files = {"file": (page_path.name, handle, "application/pdf")}
            results["timings"]["cold_ocr"] = _timed_post(
                client, f"{ocr_url}/ocr", files=files
            )
        print(
            f"cold OCR: {_format_seconds(results['timings']['cold_ocr']['duration'])} "
            f"status={results['timings']['cold_ocr']['status']}"
        )

        # Warm OCR after cold
        with page_path.open("rb") as handle:
            files = {"file": (page_path.name, handle, "application/pdf")}
            results["timings"]["warm_ocr_2"] = _timed_post(
                client, f"{ocr_url}/ocr", files=files
            )
        print(
            f"warm OCR 2: {_format_seconds(results['timings']['warm_ocr_2']['duration'])} "
            f"status={results['timings']['warm_ocr_2']['status']}"
        )

        # Final unload
        results["timings"]["unload_2"] = _timed_post(client, f"{ocr_url}/unload")
        print(
            f"/unload (final): {_format_seconds(results['timings']['unload_2']['duration'])} "
            f"status={results['timings']['unload_2']['status']}"
        )

    warm_1 = results["timings"]["warm_ocr_1"]["duration"]
    cold = results["timings"]["cold_ocr"]["duration"]
    warm_2 = results["timings"]["warm_ocr_2"]["duration"]
    results["estimated_load"] = {
        "cold_minus_warm2": max(0.0, cold - warm_2),
        "cold_minus_warm1": max(0.0, cold - warm_1),
    }

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[INFO] Saved results to {args.output}")

    try:
        page_path.unlink()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

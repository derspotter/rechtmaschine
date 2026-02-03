#!/usr/bin/env python3
"""
Service manager queue benchmark.

Sends N alternating OCR/anonymize requests concurrently to the service manager
and reports completion order and timing. Useful for verifying batching/queue
behavior and load times.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pikepdf import Pdf


def _read_file_bytes(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"OCR file not found: {path}")
    return path.read_bytes()


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _expected_first_service(current_service: str | None, ocr_count: int, anon_count: int) -> str:
    if current_service in {"ocr", "anon"}:
        return current_service
    if ocr_count > 0:
        return "ocr"
    if anon_count > 0:
        return "anon"
    return "none"


def _try_get_status(url: str) -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{url}/status", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _split_pdf_to_pages(pdf_path: Path, max_pages: int) -> list[Path]:
    temp_pages: list[Path] = []
    with Pdf.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"[INFO] PDF has {total_pages} pages")

        for i, page in enumerate(list(pdf.pages)[:max_pages], 1):
            single_page_pdf = Pdf.new()
            single_page_pdf.pages.append(page)
            temp_file = Path(
                Path(os.getenv("TMPDIR", "/tmp"))
                / f"queue_benchmark_page_{i}.pdf"
            )
            single_page_pdf.save(temp_file)
            temp_pages.append(temp_file)

    return temp_pages


def _cleanup_pages(pages: list[Path]) -> None:
    for page in pages:
        try:
            if page.exists():
                page.unlink()
        except Exception:
            pass


def _read_journal(unit: str, since: datetime) -> str | None:
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    result = subprocess.run(
        ["journalctl", "-u", unit, "--since", since_str, "--no-pager"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _parse_load_times(logs: str) -> dict[str, list[float]]:
    pattern = re.compile(r"\\[(?:Manager|Queue)\\].*?(ocr|anon).*?ready after (\\d+)s")
    matches = pattern.findall(logs)
    parsed: dict[str, list[float]] = {"ocr": [], "anon": []}
    for service, seconds in matches:
        parsed[service].append(float(seconds))
    return parsed


def _try_kill_process(pattern: str) -> None:
    try:
        subprocess.run(["pkill", "-f", pattern], check=False)
    except Exception:
        pass


async def _run_per_page_benchmark(
    client: httpx.AsyncClient,
    manager_url: str,
    page_files: list[Path],
) -> list[float]:
    timings: list[float] = []
    for page in page_files:
        start = time.time()
        with page.open("rb") as handle:
            response = await client.post(
                f"{manager_url}/ocr",
                files={"file": (page.name, handle, "application/pdf")},
            )
        duration = time.time() - start
        if response.status_code != 200:
            raise RuntimeError(
                f"OCR page {page.name} failed ({response.status_code}): {response.text[:300]}"
            )
        timings.append(duration)
        print(f"[PER-PAGE] {page.name}: {_format_seconds(duration)}")
    return timings


async def _measure_service_load_times(
    manager_url: str,
    file_path: Path,
    file_bytes: bytes,
    anon_payload: dict[str, Any],
    anon_key: str | None,
    force_cold: bool,
    load_ocr_pages: int,
) -> dict[str, Any]:
    status = _try_get_status(manager_url) or {}
    ocr_running = status.get("ocr_running")
    anon_running = status.get("anon_running")

    if force_cold:
        _try_kill_process("ocr_service_hibernate.py")
        _try_kill_process("ocr_service.py")
        _try_kill_process("anonymization_service_flair.py")
        _try_kill_process("anonymization_service.py")
        time.sleep(1)

    load_file = file_path
    if load_ocr_pages > 0:
        pages = _split_pdf_to_pages(file_path, load_ocr_pages)
        if pages:
            load_file = pages[0]
    else:
        pages = []

    results: dict[str, Any] = {
        "force_cold": force_cold,
        "ocr_running_before": ocr_running,
        "anon_running_before": anon_running,
    }

    timeout = httpx.Timeout(600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        start = time.time()
        with load_file.open("rb") as handle:
            response = await client.post(
                f"{manager_url}/ocr",
                files={"file": (load_file.name, handle, "application/pdf")},
            )
        ocr_cold = time.time() - start
        results["ocr_cold"] = ocr_cold
        results["ocr_cold_ok"] = response.status_code == 200

        start = time.time()
        with load_file.open("rb") as handle:
            response = await client.post(
                f"{manager_url}/ocr",
                files={"file": (load_file.name, handle, "application/pdf")},
            )
        ocr_warm = time.time() - start
        results["ocr_warm"] = ocr_warm
        results["ocr_warm_ok"] = response.status_code == 200
        results["ocr_estimated_load"] = max(0.0, ocr_cold - ocr_warm)

        start = time.time()
        response = await client.post(
            f"{manager_url}/anonymize",
            json=anon_payload,
            headers={"X-API-Key": anon_key} if anon_key else {},
        )
        anon_cold = time.time() - start
        results["anon_cold"] = anon_cold
        results["anon_cold_ok"] = response.status_code == 200

        start = time.time()
        response = await client.post(
            f"{manager_url}/anonymize",
            json=anon_payload,
            headers={"X-API-Key": anon_key} if anon_key else {},
        )
        anon_warm = time.time() - start
        results["anon_warm"] = anon_warm
        results["anon_warm_ok"] = response.status_code == 200
        results["anon_estimated_load"] = max(0.0, anon_cold - anon_warm)

    _cleanup_pages(pages)
    return results


async def _send_ocr(
    client: httpx.AsyncClient,
    url: str,
    file_bytes: bytes,
    filename: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{url}/ocr",
        files={"file": (filename, file_bytes, "application/pdf")},
    )
    return {"status_code": response.status_code, "body": response.text}


async def _send_anon(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    api_key: str | None,
) -> dict[str, Any]:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    response = await client.post(
        f"{url}/anonymize",
        json=payload,
        headers=headers,
    )
    return {"status_code": response.status_code, "body": response.text}


async def _run_request(
    idx: int,
    service: str,
    client: httpx.AsyncClient,
    url: str,
    file_bytes: bytes,
    filename: str,
    anon_payload: dict[str, Any],
    anon_key: str | None,
) -> dict[str, Any]:
    start = time.time()
    try:
        if service == "ocr":
            result = await _send_ocr(client, url, file_bytes, filename)
        else:
            result = await _send_anon(client, url, anon_payload, anon_key)
        ok = result["status_code"] == 200
        error = None if ok else result["body"][:300]
    except Exception as exc:
        ok = False
        error = str(exc)
    end = time.time()
    return {
        "index": idx,
        "service": service,
        "start": start,
        "end": end,
        "duration": end - start,
        "ok": ok,
        "error": error,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Service manager queue benchmark")
    parser.add_argument(
        "--manager-url",
        default="http://127.0.0.1:8004",
        help="Service manager base URL",
    )
    parser.add_argument(
        "--ocr-file",
        default="test_files/anlage_k4_bescheid.pdf",
        help="PDF file to send to OCR",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=5,
        help="Total number of requests (alternating OCR/anon)",
    )
    parser.add_argument(
        "--per-page",
        action="store_true",
        help="Run per-page OCR timing after queue test",
    )
    parser.add_argument(
        "--measure-load",
        action="store_true",
        help="Measure cold vs warm load times via the service manager",
    )
    parser.add_argument(
        "--force-cold",
        action="store_true",
        help="Kill OCR/anon child processes before measuring load times",
    )
    parser.add_argument(
        "--load-ocr-pages",
        type=int,
        default=1,
        help="Number of pages to use for load-time OCR (default: 1)",
    )
    parser.add_argument(
        "--per-page-count",
        type=int,
        default=3,
        help="Number of pages to time in per-page benchmark",
    )
    parser.add_argument(
        "--journal-unit",
        default="service-manager@jayjag.service",
        help="Systemd unit name for service manager logs",
    )
    parser.add_argument(
        "--anon-text",
        default="Dies ist ein kurzer Testtext fuer die Anonymisierung.",
        help="Text payload for anonymization",
    )
    parser.add_argument(
        "--anon-document-type",
        default="test",
        help="Document type for anonymization payload",
    )
    parser.add_argument(
        "--save-json",
        default="queue_benchmark_results.json",
        help="Where to write JSON results",
    )

    args = parser.parse_args()

    manager_url = args.manager_url.rstrip("/")
    file_path = Path(args.ocr_file)
    file_bytes = _read_file_bytes(file_path)

    start_marker = datetime.now()
    status = _try_get_status(manager_url)
    if status:
        queue = status.get("queue", {})
        current = queue.get("current_service")
        ocr_pending = queue.get("queued", {}).get("ocr", 0)
        anon_pending = queue.get("queued", {}).get("anon", 0)
        expected_first = _expected_first_service(current, ocr_pending, anon_pending)
        print("[INFO] Initial status:")
        print(json.dumps(status, indent=2))
        print(
            f"[INFO] Expected first service based on queue: {expected_first} (current={current})"
        )
    else:
        print("[WARNING] Could not fetch /status from service manager")
        expected_first = "unknown"

    anon_payload = {
        "text": args.anon_text,
        "document_type": args.anon_document_type,
    }
    anon_key = os.getenv("ANON_API_KEY")

    services = ["ocr" if i % 2 == 0 else "anon" for i in range(args.requests)]

    timeout = httpx.Timeout(600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _run_request(
                idx=i + 1,
                service=services[i],
                client=client,
                url=manager_url,
                file_bytes=file_bytes,
                filename=file_path.name,
                anon_payload=anon_payload,
                anon_key=anon_key,
            )
            for i in range(args.requests)
        ]
        results = await asyncio.gather(*tasks)

    completion_order = sorted(results, key=lambda r: r["end"])

    print("\n[RESULTS] Completion order:")
    for item in completion_order:
        status_label = "ok" if item["ok"] else "fail"
        print(
            f"  {item['service']}#{item['index']} -> {status_label} "
            f"({_format_seconds(item['duration'])})"
        )

    ocr_times = [r["duration"] for r in results if r["service"] == "ocr" and r["ok"]]
    anon_times = [r["duration"] for r in results if r["service"] == "anon" and r["ok"]]

    print("\n[SUMMARY]")
    print(f"  Expected first service: {expected_first}")
    if ocr_times:
        stats = _summary_stats(ocr_times)
        print(
            "  OCR timings: "
            f"min={_format_seconds(stats['min'])}, "
            f"avg={_format_seconds(stats['avg'])}, "
            f"max={_format_seconds(stats['max'])}"
        )
    else:
        print("  OCR timings: no successful OCR responses")

    if anon_times:
        stats = _summary_stats(anon_times)
        print(
            "  Anon timings: "
            f"min={_format_seconds(stats['min'])}, "
            f"avg={_format_seconds(stats['avg'])}, "
            f"max={_format_seconds(stats['max'])}"
        )
    else:
        print("  Anon timings: no successful anonymization responses")

    errors = [r for r in results if not r["ok"]]
    if errors:
        print("\n[ERRORS]")
        for err in errors:
            print(f"  {err['service']}#{err['index']}: {err['error']}")

    load_times: dict[str, list[float]] = {"ocr": [], "anon": []}
    logs = _read_journal(args.journal_unit, start_marker)
    if logs:
        load_times = _parse_load_times(logs)
        if load_times["ocr"] or load_times["anon"]:
            print("\n[LOAD TIMES] (from service manager logs)")
            if load_times["ocr"]:
                print(
                    "  OCR load times: "
                    + ", ".join(_format_seconds(t) for t in load_times["ocr"])
                )
            if load_times["anon"]:
                print(
                    "  Anon load times: "
                    + ", ".join(_format_seconds(t) for t in load_times["anon"])
                )
        else:
            print("\n[LOAD TIMES] No load timing lines found in journal output.")
    else:
        print("\n[LOAD TIMES] Unable to read journal (insufficient permissions or unit missing).")

    per_page_timings: list[float] = []
    if args.per_page:
        print("\n[PER-PAGE] Running per-page OCR timing...")
        pages = _split_pdf_to_pages(file_path, args.per_page_count)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                per_page_timings = await _run_per_page_benchmark(
                    client, manager_url, pages
                )
        except Exception as exc:
            print(f"[PER-PAGE] Failed: {exc}")
        finally:
            _cleanup_pages(pages)

    load_measurements: dict[str, Any] | None = None
    if args.measure_load:
        print("\n[LOAD MEASURE] Measuring cold vs warm times...")
        load_measurements = await _measure_service_load_times(
            manager_url=manager_url,
            file_path=file_path,
            file_bytes=file_bytes,
            anon_payload=anon_payload,
            anon_key=anon_key,
            force_cold=args.force_cold,
            load_ocr_pages=args.load_ocr_pages,
        )
        print(
            f"  OCR cold: {_format_seconds(load_measurements['ocr_cold'])} "
            f"(ok={load_measurements['ocr_cold_ok']})"
        )
        print(
            f"  OCR warm: {_format_seconds(load_measurements['ocr_warm'])} "
            f"(ok={load_measurements['ocr_warm_ok']})"
        )
        print(
            "  OCR estimated load: "
            f"{_format_seconds(load_measurements['ocr_estimated_load'])}"
        )
        print(
            f"  Anon cold: {_format_seconds(load_measurements['anon_cold'])} "
            f"(ok={load_measurements['anon_cold_ok']})"
        )
        print(
            f"  Anon warm: {_format_seconds(load_measurements['anon_warm'])} "
            f"(ok={load_measurements['anon_warm_ok']})"
        )
        print(
            "  Anon estimated load: "
            f"{_format_seconds(load_measurements['anon_estimated_load'])}"
        )

    output = {
        "manager_url": manager_url,
        "expected_first": expected_first,
        "results": results,
        "completion_order": completion_order,
        "load_times": load_times,
        "per_page_timings": per_page_timings,
        "load_measurements": load_measurements,
    }
    Path(args.save_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[INFO] Saved results to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

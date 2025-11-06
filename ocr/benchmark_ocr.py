#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark script to compare OCR-VL performance with and without vLLM acceleration.
"""

import sys
import time
import argparse
from pathlib import Path
import tempfile
import json

import httpx
from pikepdf import Pdf


def split_pdf_to_pages(pdf_path: Path, max_pages: int = 3) -> list[Path]:
    """Split a PDF into individual page files."""
    temp_pages = []

    with Pdf.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"[INFO] PDF has {total_pages} pages")

        for i, page in enumerate(list(pdf.pages)[:max_pages], 1):
            single_page_pdf = Pdf.new()
            single_page_pdf.pages.append(page)

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f"_page_{i}.pdf",
                prefix="benchmark_"
            )
            temp_path = Path(temp_file.name)
            single_page_pdf.save(temp_path)
            temp_pages.append(temp_path)

    return temp_pages


def benchmark_ocr_service(service_url: str, page_files: list[Path]) -> dict:
    """Benchmark OCR service performance."""
    results = {
        "service_url": service_url,
        "total_pages": len(page_files),
        "page_times": [],
        "total_time": 0,
        "avg_time_per_page": 0,
        "success_count": 0,
        "error_count": 0,
        "total_blocks": 0,
    }

    print(f"\n{'='*60}")
    print(f"Benchmarking: {service_url}")
    print(f"{'='*60}\n")

    overall_start = time.time()

    for i, page_file in enumerate(page_files, 1):
        print(f"[{i}/{len(page_files)}] Processing page {i}...", end=" ", flush=True)

        page_start = time.time()

        try:
            with open(page_file, "rb") as f:
                files = {"file": (page_file.name, f, "application/pdf")}

                with httpx.Client(timeout=120.0) as client:
                    response = client.post(f"{service_url}/ocr", files=files)
                    response.raise_for_status()

            page_time = time.time() - page_start
            result = response.json()

            results["page_times"].append(page_time)
            results["success_count"] += 1

            # Handle both response formats:
            # VL services: {"block_count": N}
            # Simple OCR: {"pages": [{"lines": [...]}]}
            block_count = result.get("block_count", 0)
            if block_count == 0 and "pages" in result:
                # Count total lines from all pages for simple OCR
                block_count = sum(len(p.get("lines", [])) for p in result["pages"])

            results["total_blocks"] += block_count

            print(f"✓ {page_time:.2f}s ({block_count} lines/blocks)")

        except Exception as e:
            page_time = time.time() - page_start
            results["page_times"].append(page_time)
            results["error_count"] += 1
            print(f"✗ {page_time:.2f}s (error: {e})")

    results["total_time"] = time.time() - overall_start

    if results["success_count"] > 0:
        results["avg_time_per_page"] = results["total_time"] / results["success_count"]

    return results


def print_comparison(results1: dict, results2: dict = None):
    """Print benchmark results and comparison."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}\n")

    def print_result(label: str, res: dict):
        print(f"{label}:")
        print(f"  Service: {res['service_url']}")
        print(f"  Total pages: {res['total_pages']}")
        print(f"  Successful: {res['success_count']}")
        print(f"  Failed: {res['error_count']}")
        print(f"  Total blocks: {res['total_blocks']}")
        print(f"  Total time: {res['total_time']:.2f}s")
        print(f"  Avg per page: {res['avg_time_per_page']:.2f}s")
        print(f"  Min/Max: {min(res['page_times']):.2f}s / {max(res['page_times']):.2f}s")
        print()

    print_result("Setup 1", results1)

    if results2:
        print_result("Setup 2", results2)

        print(f"{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}\n")

        if results1['avg_time_per_page'] > 0 and results2['avg_time_per_page'] > 0:
            speedup = results2['avg_time_per_page'] / results1['avg_time_per_page']
            faster_label = "Setup 1" if speedup > 1 else "Setup 2"
            speedup_val = speedup if speedup > 1 else 1/speedup

            print(f"  {faster_label} is {speedup_val:.2f}x faster")
            print(f"  Time saved per page: {abs(results1['avg_time_per_page'] - results2['avg_time_per_page']):.2f}s")
            print(f"  Time saved total: {abs(results1['total_time'] - results2['total_time']):.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OCR-VL service performance"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to test PDF file"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Number of pages to test (default: 3)"
    )
    parser.add_argument(
        "--service1",
        type=str,
        default="http://localhost:9005",
        help="First service URL (default: http://localhost:9005)"
    )
    parser.add_argument(
        "--service2",
        type=str,
        default=None,
        help="Second service URL for comparison (optional)"
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    # Split PDF into pages
    print(f"[INFO] Preparing test: {pdf_path}")
    page_files = split_pdf_to_pages(pdf_path, args.max_pages)

    try:
        # Benchmark first service
        results1 = benchmark_ocr_service(args.service1, page_files)

        # Benchmark second service if provided
        results2 = None
        if args.service2:
            print("\n" + "="*60)
            input("Press Enter to continue with second service...")
            results2 = benchmark_ocr_service(args.service2, page_files)

        # Print comparison
        print_comparison(results1, results2)

        # Save results
        output_file = Path("benchmark_results.json")
        with open(output_file, "w") as f:
            json.dump({
                "setup1": results1,
                "setup2": results2,
            }, f, indent=2)
        print(f"\n[INFO] Results saved to: {output_file}")

    finally:
        # Cleanup
        print(f"\n[INFO] Cleaning up {len(page_files)} temp files...")
        for page_file in page_files:
            try:
                page_file.unlink()
            except Exception as e:
                print(f"[WARNING] Could not delete {page_file}: {e}")


if __name__ == "__main__":
    main()

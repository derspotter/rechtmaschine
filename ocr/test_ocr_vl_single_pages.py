#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for OCR-VL service - processes one PDF page at a time.
"""

import sys
import tempfile
import argparse
from pathlib import Path
import httpx
from pikepdf import Pdf

OCR_VL_URL = "http://localhost:9005/ocr"


def split_pdf_to_pages(pdf_path: Path) -> list[Path]:
    """
    Split a PDF into individual page files.
    Returns list of temp file paths.
    """
    temp_pages = []

    with Pdf.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"[INFO] PDF has {total_pages} pages")

        for i, page in enumerate(pdf.pages, 1):
            # Create a new PDF with just this page
            single_page_pdf = Pdf.new()
            single_page_pdf.pages.append(page)

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f"_page_{i}.pdf",
                prefix="ocr_test_"
            )
            temp_path = Path(temp_file.name)
            single_page_pdf.save(temp_path)
            temp_pages.append(temp_path)

            print(f"[INFO] Created temp file for page {i}: {temp_path}")

    return temp_pages


def send_page_to_ocr(page_path: Path, page_num: int) -> dict:
    """
    Send a single page PDF to the OCR-VL service.
    """
    print(f"\n{'='*60}")
    print(f"[INFO] Processing page {page_num}: {page_path.name}")
    print(f"{'='*60}")

    try:
        with open(page_path, "rb") as f:
            files = {"file": (page_path.name, f, "application/pdf")}

            with httpx.Client(timeout=120.0) as client:
                response = client.post(OCR_VL_URL, files=files)
                response.raise_for_status()

        result = response.json()

        # Display results
        print(f"\n[RESULT] Page {page_num}:")
        print(f"  - Model: {result.get('model')}")
        print(f"  - Block count: {result.get('block_count')}")
        print(f"  - Page count: {result.get('page_count')}")

        full_text = result.get('full_text', '')
        if full_text:
            print(f"\n[EXTRACTED TEXT]:")
            print("-" * 60)
            print(full_text[:500])  # Show first 500 chars
            if len(full_text) > 500:
                print(f"\n... ({len(full_text) - 500} more characters)")
            print("-" * 60)
        else:
            print("[WARNING] No text extracted!")

        return result

    except httpx.HTTPError as e:
        print(f"[ERROR] HTTP error for page {page_num}: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"[ERROR] Unexpected error for page {page_num}: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Test OCR-VL service with single pages from a PDF"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file to test"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to process (default: all)"
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.suffix.lower() == ".pdf":
        print(f"[ERROR] File must be a PDF: {pdf_path}")
        sys.exit(1)

    # Check if service is running
    print("[INFO] Checking OCR-VL service health...")
    try:
        with httpx.Client(timeout=5.0) as client:
            health_response = client.get("http://localhost:9005/health")
            health_response.raise_for_status()
            health = health_response.json()
            print(f"[OK] Service running: {health.get('model')}")
            print(f"     GPU available: {health.get('gpu_available')}")
    except Exception as e:
        print(f"[ERROR] OCR-VL service not available: {e}")
        print("[INFO] Start the service with: cd ocr && docker-compose up -d")
        sys.exit(1)

    # Split PDF into pages
    print(f"\n[INFO] Splitting PDF: {pdf_path}")
    page_files = split_pdf_to_pages(pdf_path)

    # Limit pages if requested
    if args.max_pages:
        page_files = page_files[:args.max_pages]
        print(f"[INFO] Limited to first {args.max_pages} pages")

    # Process each page
    results = []
    try:
        for i, page_file in enumerate(page_files, 1):
            result = send_page_to_ocr(page_file, i)
            results.append(result)
    finally:
        # Cleanup temp files
        print(f"\n[INFO] Cleaning up {len(page_files)} temp files...")
        for page_file in page_files:
            try:
                page_file.unlink()
            except Exception as e:
                print(f"[WARNING] Could not delete {page_file}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"[SUMMARY]")
    print(f"{'='*60}")
    print(f"Total pages processed: {len(results)}")

    successful = sum(1 for r in results if "error" not in r)
    failed = len(results) - successful

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        total_blocks = sum(r.get('block_count', 0) for r in results if "error" not in r)
        print(f"Total text blocks extracted: {total_blocks}")


if __name__ == "__main__":
    main()

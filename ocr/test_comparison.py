#!/usr/bin/env python3
"""
Compare performance of PaddleOCR vs PaddleOCR-VL
Tests both services with the same PDF and reports timing/accuracy
"""

import httpx
import time
import sys
from pathlib import Path

def test_ocr_service(url: str, file_path: str, service_name: str):
    """Test a single OCR service"""
    print(f"\n{'='*60}")
    print(f"Testing {service_name}")
    print(f"{'='*60}")

    try:
        # Check if service is running
        health_response = httpx.get(f"{url}/health", timeout=5.0)
        if health_response.status_code != 200:
            print(f"❌ {service_name} health check failed")
            return None

        health_data = health_response.json()
        print(f"✓ Service: {health_data.get('service')}")
        print(f"✓ GPU: {health_data.get('gpu_available')}")
        print(f"✓ Model: {health_data.get('model', 'N/A')}")

        # Test OCR
        print(f"\n📄 Processing: {Path(file_path).name}")
        start_time = time.time()

        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "application/pdf")}
            response = httpx.post(
                f"{url}/ocr",
                files=files,
                timeout=300.0
            )

        elapsed = time.time() - start_time

        if response.status_code != 200:
            print(f"❌ OCR failed: {response.status_code}")
            print(response.text)
            return None

        result = response.json()
        text_length = len(result.get("full_text", ""))

        print(f"\n⏱️  Time: {elapsed:.2f}s")
        print(f"📊 Text length: {text_length} chars")

        if "page_count" in result:
            print(f"📑 Pages: {result['page_count']}")
        if "element_count" in result:
            print(f"🔢 Elements: {result['element_count']}")
        if "avg_confidence" in result:
            print(f"💯 Avg confidence: {result['avg_confidence']:.2%}")

        # Show first 200 chars of text
        preview = result.get("full_text", "")[:200]
        print(f"\n📝 Preview:\n{preview}...")

        return {
            "service": service_name,
            "elapsed": elapsed,
            "text_length": text_length,
            "result": result
        }

    except Exception as e:
        print(f"❌ Error testing {service_name}: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_comparison.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print("🚀 PaddleOCR vs PaddleOCR-VL Comparison Test")
    print(f"📂 Test file: {pdf_path}")

    # Test both services
    results = []

    ocr_result = test_ocr_service(
        "http://localhost:9003",
        pdf_path,
        "PaddleOCR (Regular)"
    )
    if ocr_result:
        results.append(ocr_result)

    ocr_vl_result = test_ocr_service(
        "http://localhost:9005",
        pdf_path,
        "PaddleOCR-VL (Vision-Language)"
    )
    if ocr_vl_result:
        results.append(ocr_vl_result)

    # Summary
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("📊 COMPARISON SUMMARY")
        print(f"{'='*60}")

        faster = min(results, key=lambda x: x["elapsed"])
        slower = max(results, key=lambda x: x["elapsed"])
        speedup = (slower["elapsed"] / faster["elapsed"])

        print(f"🏆 Faster: {faster['service']} ({faster['elapsed']:.2f}s)")
        print(f"🐢 Slower: {slower['service']} ({slower['elapsed']:.2f}s)")
        print(f"⚡ Speedup: {speedup:.2f}x")

        print(f"\n📝 Text extraction:")
        for r in results:
            print(f"  {r['service']}: {r['text_length']} chars")


if __name__ == "__main__":
    main()

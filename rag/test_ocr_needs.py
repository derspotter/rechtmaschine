#!/usr/bin/env python3
"""
Test how many PDFs need OCR (no extractable text with PyMuPDF)
"""

import fitz  # PyMuPDF
from pathlib import Path
import random
from collections import defaultdict
import sys

CORPUS_DIR = Path("/run/media/jayjag/My Book1/RAG/kanzlei/")


def check_pdf_extractable(pdf_path):
    """
    Check if PDF has extractable text with PyMuPDF.
    Returns: (extractable: bool, char_count: int, page_count: int, error: str)
    """
    try:
        doc = fitz.open(pdf_path)
        total_chars = 0
        page_count = len(doc)

        for page in doc:
            text = page.get_text()
            total_chars += len(text.strip())

        doc.close()

        # Consider "extractable" if we got at least 100 chars total
        # (filters out PDFs with just a few stray characters)
        extractable = total_chars >= 100

        return extractable, total_chars, page_count, None

    except Exception as e:
        return False, 0, 0, str(e)


def analyze_ocr_needs(sample_size=None, verbose=False):
    """
    Analyze how many PDFs need OCR.

    Args:
        sample_size: If None, check all PDFs. If int, check random sample.
        verbose: Print details for each file
    """

    print("=" * 80)
    print("PDF OCR NEEDS ANALYSIS")
    print("=" * 80)
    print(f"Corpus: {CORPUS_DIR}")
    print()

    # Find all PDFs
    print("Finding all PDFs...")
    all_pdfs = list(CORPUS_DIR.glob("**/*.pdf")) + list(CORPUS_DIR.glob("**/*.PDF"))
    total_pdfs = len(all_pdfs)

    print(f"Found {total_pdfs:,} PDF files")

    # Sample if requested
    if sample_size and sample_size < total_pdfs:
        print(f"Sampling {sample_size:,} random PDFs for analysis...")
        pdfs_to_check = random.sample(all_pdfs, sample_size)
    else:
        print(f"Checking all {total_pdfs:,} PDFs (this will take a while)...")
        pdfs_to_check = all_pdfs

    print()

    # Analyze each PDF
    results = {
        'extractable': [],      # PDFs with text
        'needs_ocr': [],        # PDFs with no/minimal text (scanned)
        'errors': [],           # PDFs that failed to open
    }

    stats = {
        'total_chars': 0,
        'total_pages': 0,
    }

    for i, pdf_path in enumerate(pdfs_to_check, 1):
        if i % 100 == 0 or verbose:
            print(f"Progress: {i}/{len(pdfs_to_check)} ({i/len(pdfs_to_check)*100:.1f}%)", end='\r')

        extractable, chars, pages, error = check_pdf_extractable(pdf_path)

        if error:
            results['errors'].append((pdf_path, error))
            if verbose:
                print(f"❌ ERROR: {pdf_path.name} - {error}")
        elif extractable:
            results['extractable'].append((pdf_path, chars, pages))
            stats['total_chars'] += chars
            stats['total_pages'] += pages
            if verbose:
                print(f"✅ OK: {pdf_path.name} - {chars:,} chars, {pages} pages")
        else:
            results['needs_ocr'].append((pdf_path, chars, pages))
            stats['total_pages'] += pages
            if verbose:
                print(f"🔍 NEEDS OCR: {pdf_path.name} - only {chars} chars, {pages} pages")

    print()  # Clear progress line
    print()

    # Calculate statistics
    checked_count = len(pdfs_to_check)
    extractable_count = len(results['extractable'])
    needs_ocr_count = len(results['needs_ocr'])
    error_count = len(results['errors'])

    extractable_pct = (extractable_count / checked_count * 100) if checked_count > 0 else 0
    needs_ocr_pct = (needs_ocr_count / checked_count * 100) if checked_count > 0 else 0
    error_pct = (error_count / checked_count * 100) if checked_count > 0 else 0

    # Report
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nPDFs Analyzed: {checked_count:,}")
    if sample_size and sample_size < total_pdfs:
        print(f"(Sample of {total_pdfs:,} total PDFs)")
    print()

    print(f"✅ Extractable (PyMuPDF works):  {extractable_count:6,} ({extractable_pct:5.1f}%)")
    print(f"🔍 Need OCR (scanned/no text):   {needs_ocr_count:6,} ({needs_ocr_pct:5.1f}%)")
    print(f"❌ Errors (corrupted/protected): {error_count:6,} ({error_pct:5.1f}%)")

    if stats['total_pages'] > 0:
        avg_chars_per_page = stats['total_chars'] / stats['total_pages'] if stats['total_pages'] > 0 else 0
        print(f"\nAverage text density: {avg_chars_per_page:.0f} chars/page (extractable docs)")

    # Extrapolate to full corpus if sampling
    if sample_size and sample_size < total_pdfs:
        print(f"\n" + "-" * 80)
        print("EXTRAPOLATED TO FULL CORPUS")
        print("-" * 80)

        est_extractable = int(total_pdfs * extractable_pct / 100)
        est_needs_ocr = int(total_pdfs * needs_ocr_pct / 100)
        est_errors = int(total_pdfs * error_pct / 100)

        print(f"Estimated for all {total_pdfs:,} PDFs:")
        print(f"  ✅ Extractable: ~{est_extractable:,}")
        print(f"  🔍 Need OCR:    ~{est_needs_ocr:,}")
        print(f"  ❌ Errors:      ~{est_errors:,}")

    # Show some examples
    if needs_ocr_count > 0:
        print(f"\n" + "-" * 80)
        print(f"SAMPLE FILES NEEDING OCR (first 10)")
        print("-" * 80)
        for pdf_path, chars, pages in results['needs_ocr'][:10]:
            rel_path = pdf_path.relative_to(CORPUS_DIR)
            print(f"  {rel_path}")
            print(f"    └─ {pages} pages, only {chars} chars extractable")

    if error_count > 0:
        print(f"\n" + "-" * 80)
        print(f"FILES WITH ERRORS (first 10)")
        print("-" * 80)
        for pdf_path, error in results['errors'][:10]:
            rel_path = pdf_path.relative_to(CORPUS_DIR)
            print(f"  {rel_path}")
            print(f"    └─ {error}")

    # Recommendation
    print(f"\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if needs_ocr_pct < 5:
        print(f"\n✅ PyMuPDF is sufficient!")
        print(f"   Only {needs_ocr_pct:.1f}% of PDFs need OCR.")
        print(f"   Recommendation: Use PyMuPDF only, skip OCR for now.")
        print(f"   You can add OCR later for the few problematic files.")
    elif needs_ocr_pct < 15:
        print(f"\n⚠️  Small OCR requirement")
        print(f"   {needs_ocr_pct:.1f}% of PDFs need OCR.")
        print(f"   Recommendation: Use PyMuPDF as primary, add Docling/Tesseract as fallback.")
    else:
        print(f"\n🔍 Significant OCR requirement")
        print(f"   {needs_ocr_pct:.1f}% of PDFs need OCR.")
        print(f"   Recommendation: Use Docling (built-in OCR) or PyMuPDF + Tesseract.")

    print("\n" + "=" * 80)

    return results, stats


if __name__ == "__main__":
    # Check if sample size specified
    if len(sys.argv) > 1:
        sample_size = int(sys.argv[1])
        print(f"Using sample size: {sample_size}")
    else:
        # Default: sample 500 PDFs for quick analysis
        sample_size = 500
        print(f"Using default sample size: {sample_size}")
        print(f"(Run with argument to change, e.g., `python test_ocr_needs.py 1000`)")
        print(f"(Or use `python test_ocr_needs.py all` to check all PDFs)")
        print()

    if sys.argv[-1].lower() == 'all':
        sample_size = None

    try:
        analyze_ocr_needs(sample_size=sample_size, verbose=False)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

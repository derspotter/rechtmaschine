#!/usr/bin/env python3
"""
Detect Kanzlei Keienborg letterhead in documents
Based on actual template analysis from 00 AT/vorlagen/

Letterhead contains:
- Kanzlei Keienborg
- Marcel Keienborg / Christian Schotte
- Friedrich-Ebert-Straße 17
- 40210 Düsseldorf
"""

import fitz  # PyMuPDF
from pathlib import Path
import re
from collections import Counter
import sys

CORPUS_DIR = Path("/run/media/jayjag/My Book1/RAG/kanzlei/")

# Letterhead detection patterns (based on actual templates)
STRONG_PATTERNS = [
    r"keienborg",  # Unique surname - STRONGEST signal
    r"friedrich[-\s]ebert[-\s]str",  # Office address
]

SUPPORTING_PATTERNS = [
    r"40210\s+d[üu]sseldorf",  # ZIP + city
    r"marcel\s+keienborg",
    r"christian\s+schotte",
    r"kanzlei\s+keienborg",
]


def detect_letterhead(pdf_path, check_pages=1):
    """
    Detect if PDF contains Kanzlei Keienborg letterhead.

    Args:
        pdf_path: Path to PDF file
        check_pages: Number of pages to check (default: 1 - first page only)

    Returns:
        dict with:
            - is_kanzlei_doc: bool
            - confidence: float (0-1)
            - matched_patterns: list of matched pattern names
            - first_page_text: str (first 500 chars for debugging)
            - error: str or None
    """
    try:
        doc = fitz.open(pdf_path)

        if len(doc) == 0:
            return {
                'is_kanzlei_doc': False,
                'confidence': 0.0,
                'matched_patterns': [],
                'first_page_text': '',
                'error': 'Empty PDF'
            }

        # Extract text from first N pages
        text = ""
        for page_num in range(min(check_pages, len(doc))):
            text += doc[page_num].get_text()

        doc.close()

        # Normalize text for matching
        text_lower = text.lower()
        text_normalized = re.sub(r'\s+', ' ', text_lower)  # Collapse whitespace

        # Check for strong patterns
        strong_matches = []
        for pattern in STRONG_PATTERNS:
            if re.search(pattern, text_normalized):
                strong_matches.append(pattern)

        # Check for supporting patterns
        supporting_matches = []
        for pattern in SUPPORTING_PATTERNS:
            if re.search(pattern, text_normalized):
                supporting_matches.append(pattern)

        all_matches = strong_matches + supporting_matches

        # Calculate confidence
        confidence = 0.0

        if 'keienborg' in strong_matches:
            # "Keienborg" = almost certain
            confidence = 0.98
        elif 'friedrich[-\\s]ebert[-\\s]str' in strong_matches:
            # Office address = very likely (but could be copied)
            confidence = 0.85
        elif len(supporting_matches) >= 2:
            # Multiple supporting patterns = likely
            confidence = 0.70
        elif len(supporting_matches) >= 1:
            # Single supporting pattern = maybe
            confidence = 0.40

        is_kanzlei_doc = confidence >= 0.70

        return {
            'is_kanzlei_doc': is_kanzlei_doc,
            'confidence': confidence,
            'matched_patterns': all_matches,
            'first_page_text': text[:500],
            'error': None
        }

    except Exception as e:
        return {
            'is_kanzlei_doc': False,
            'confidence': 0.0,
            'matched_patterns': [],
            'first_page_text': '',
            'error': str(e)
        }


def analyze_corpus_letterhead(sample_size=500):
    """
    Analyze corpus to detect Kanzlei Keienborg documents.
    """
    print("=" * 80)
    print("KANZLEI KEIENBORG LETTERHEAD DETECTION")
    print("=" * 80)
    print(f"Corpus: {CORPUS_DIR}")
    print()
    print("Detection patterns based on actual templates:")
    print("  ✓ Strong: 'keienborg', 'friedrich-ebert-str'")
    print("  ✓ Supporting: '40210 düsseldorf', 'marcel keienborg', etc.")
    print()

    # Find all PDFs
    print("Finding PDFs...")
    all_pdfs = list(CORPUS_DIR.glob("**/*.pdf")) + list(CORPUS_DIR.glob("**/*.PDF"))

    print(f"Found {len(all_pdfs):,} PDF files")

    # Sample PDFs
    if sample_size and sample_size < len(all_pdfs):
        print(f"Sampling {sample_size:,} random PDFs...")
        import random
        pdfs_to_check = random.sample(all_pdfs, sample_size)
    else:
        print(f"Checking all {len(all_pdfs):,} PDFs...")
        pdfs_to_check = all_pdfs

    print()

    # Results
    results = {
        'kanzlei_docs': [],
        'external_docs': [],
        'errors': []
    }

    # Analyze each PDF
    for i, pdf_path in enumerate(pdfs_to_check, 1):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(pdfs_to_check)} ({i/len(pdfs_to_check)*100:.1f}%)", end='\r')

        result = detect_letterhead(pdf_path)

        if result['error']:
            results['errors'].append((pdf_path, result['error']))
        elif result['is_kanzlei_doc']:
            results['kanzlei_docs'].append((pdf_path, result['confidence'], result['matched_patterns']))
        else:
            results['external_docs'].append((pdf_path, result['confidence'], result['matched_patterns']))

    print()  # Clear progress line
    print()

    # Calculate statistics
    checked_count = len(pdfs_to_check)
    kanzlei_count = len(results['kanzlei_docs'])
    external_count = len(results['external_docs'])
    error_count = len(results['errors'])

    kanzlei_pct = (kanzlei_count / checked_count * 100) if checked_count > 0 else 0
    external_pct = (external_count / checked_count * 100) if checked_count > 0 else 0

    # Report
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nPDFs Analyzed: {checked_count:,}")
    print()

    print(f"📝 Kanzlei Keienborg docs:        {kanzlei_count:6,} ({kanzlei_pct:5.1f}%)")
    print(f"📄 External docs (BAMF/courts):   {external_count:6,} ({external_pct:5.1f}%)")
    print(f"❌ Errors:                        {error_count:6,}")

    # Extrapolate to full corpus
    if sample_size and sample_size < len(all_pdfs):
        print(f"\n" + "-" * 80)
        print("EXTRAPOLATED TO FULL CORPUS")
        print("-" * 80)

        total_pdfs = len(all_pdfs)
        est_kanzlei = int(total_pdfs * kanzlei_pct / 100)
        est_external = int(total_pdfs * external_pct / 100)

        print(f"Estimated for all {total_pdfs:,} PDFs:")
        print(f"  📝 Kanzlei docs:  ~{est_kanzlei:,}")
        print(f"  📄 External docs: ~{est_external:,}")

    # ODT analysis
    all_odts = list(CORPUS_DIR.glob("**/*.odt"))
    print(f"\n" + "-" * 80)
    print("ODT FILES (LibreOffice - almost certainly Kanzlei docs)")
    print("-" * 80)
    print(f"Total ODT files: {len(all_odts):,}")

    # Show sample Kanzlei docs
    if kanzlei_count > 0:
        print(f"\n" + "-" * 80)
        print(f"SAMPLE KANZLEI KEIENBORG DOCUMENTS (first 15)")
        print("-" * 80)
        for pdf_path, confidence, patterns in results['kanzlei_docs'][:15]:
            rel_path = pdf_path.relative_to(CORPUS_DIR)
            print(f"  {rel_path}")
            print(f"    └─ Confidence: {confidence:.2f}, Patterns: {patterns}")

    # Document type breakdown by filename
    if kanzlei_count > 0:
        print(f"\n" + "-" * 80)
        print("KANZLEI DOC TYPES (by filename pattern)")
        print("-" * 80)

        doc_type_counts = Counter()
        for pdf_path, _, _ in results['kanzlei_docs']:
            filename = pdf_path.name.lower()

            if 'klage' in filename:
                doc_type_counts['Klage'] += 1
            elif 'schriftsatz' in filename or re.match(r'^\d{6}_', filename):
                doc_type_counts['Schriftsatz'] += 1
            elif 'vollmacht' in filename:
                doc_type_counts['Vollmacht'] += 1
            elif 'pka' in filename or 'pkh' in filename or 'mittellos' in filename:
                doc_type_counts['PKH/Mittellosigkeit'] += 1
            elif 'betreuung' in filename:
                doc_type_counts['Betreuung'] += 1
            elif 'beschwerde' in filename:
                doc_type_counts['Beschwerde'] += 1
            elif 'eidesstattlich' in filename:
                doc_type_counts['Eidesstattliche Versicherung'] += 1
            else:
                doc_type_counts['Other'] += 1

        for doc_type, count in doc_type_counts.most_common():
            pct = count / kanzlei_count * 100
            print(f"  {doc_type:35s} {count:4,} ({pct:5.1f}%)")

    # Show sample external docs
    if external_count > 0 and external_count <= 20:
        print(f"\n" + "-" * 80)
        print(f"SAMPLE EXTERNAL DOCUMENTS (all {external_count})")
        print("-" * 80)
        for pdf_path, confidence, patterns in results['external_docs']:
            rel_path = pdf_path.relative_to(CORPUS_DIR)
            print(f"  {rel_path.name}")

    # Recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS FOR RAG PIPELINE")
    print("=" * 80)

    print(f"\n1. Document Classification:")
    print(f"   ✓ Add 'is_kanzlei_doc' boolean metadata field")
    print(f"   ✓ Add 'doc_origin' enum: 'internal' | 'external'")
    print(f"   ✓ Detect letterhead during PDF parsing (first page only)")

    print(f"\n2. Anonymization Strategy:")
    print(f"   ✓ Kanzlei docs ({kanzlei_pct:.0f}%): SKIP anonymization")
    print(f"     → Your briefs are already generic/anonymized")
    print(f"     → Saves processing time and API costs")
    print(f"   ✓ External docs ({external_pct:.0f}%): FULL anonymization")
    print(f"     → BAMF decisions contain client names/details")
    print(f"     → Court decisions may contain personal info")

    print(f"\n3. Section Detection:")
    print(f"   ✓ Kanzlei docs: Reliable structure (I., II., III., etc.)")
    print(f"   ✓ External docs: Varied formats (need robust patterns)")

    print(f"\n4. Deduplication Priority:")
    print(f"   ✓ Kanzlei docs: HIGH dedup potential")
    print(f"     → You reuse country condition arguments")
    print(f"     → Legal reasoning blocks are templated")
    print(f"   ✓ External docs: LOW dedup")
    print(f"     → Each BAMF decision is unique")

    print(f"\n5. Metadata Extraction:")
    print(f"   ✓ Kanzlei docs: Extract from filenames")
    print(f"     → Date: YYMMDD prefix (e.g., '251013_vg_klage.pdf')")
    print(f"     → Type: klage, schriftsatz, vollmacht, etc.")
    print(f"   ✓ External docs: Extract from content")
    print(f"     → BAMF Az: search for '\\d{{7}}-\\d{{3}}'")
    print(f"     → Court Az: search for 'Az\\.: .*K \\d+/\\d+\\.A'")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    sample_size = 500

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'all':
            sample_size = None
        else:
            sample_size = int(sys.argv[1])

    try:
        analyze_corpus_letterhead(sample_size=sample_size)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

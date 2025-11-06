#!/usr/bin/env python3
"""
Corpus Data Structure Analysis
Analyzes the legal document corpus to inform parser selection
"""

import os
from pathlib import Path
from collections import Counter, defaultdict
import json
from datetime import datetime

# Target directory
CORPUS_DIR = Path("/run/media/jayjag/My Book1/RAG/kanzlei/")


def analyze_corpus():
    """Analyze the complete corpus structure"""

    print("=" * 80)
    print("RAG CORPUS STRUCTURE ANALYSIS")
    print("=" * 80)
    print(f"Target: {CORPUS_DIR}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize counters
    total_files = 0
    total_dirs = 0
    extension_counts = Counter()
    size_by_extension = defaultdict(int)
    year_distribution = Counter()
    case_count = 0

    # Document type patterns
    bescheid_count = 0
    klage_count = 0
    schriftsatz_count = 0
    anhoerung_count = 0

    # Track ODT-PDF pairs
    odt_pdf_pairs = 0
    orphan_odts = []
    orphan_pdfs = []

    print("Scanning directory tree...")

    # Walk through all directories
    for root, dirs, files in os.walk(CORPUS_DIR):
        total_dirs += len(dirs)
        root_path = Path(root)

        # Count cases (directories with " vs " in name)
        if " vs " in root_path.name:
            case_count += 1

        # Year distribution
        if root_path.parent == CORPUS_DIR:
            year_distribution[root_path.name] = len(dirs)

        # Track files in this directory
        dir_files = set(files)

        for filename in files:
            total_files += 1
            file_path = root_path / filename

            # Get extension (case-insensitive)
            ext = file_path.suffix.lower().lstrip('.')
            if not ext and '.' in filename:
                ext = filename.split('.')[-1].lower()

            extension_counts[ext] += 1

            # Get file size
            try:
                size = file_path.stat().st_size
                size_by_extension[ext] += size
            except:
                pass

            # Detect document types by filename patterns
            fname_lower = filename.lower()

            if 'bescheid' in fname_lower or 'anlage_k2' in fname_lower:
                bescheid_count += 1
            if 'klage' in fname_lower:
                klage_count += 1
            if 'schriftsatz' in fname_lower or fname_lower.startswith(('24', '25')):
                schriftsatz_count += 1
            if 'anhoerung' in fname_lower or 'anhörung' in fname_lower:
                anhoerung_count += 1

            # Check for ODT-PDF pairs
            if ext == 'odt':
                pdf_equivalent = filename.rsplit('.', 1)[0] + '.pdf'
                if pdf_equivalent in dir_files:
                    odt_pdf_pairs += 1
                else:
                    orphan_odts.append(str(file_path.relative_to(CORPUS_DIR)))

    # Calculate statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTotal Files:       {total_files:,}")
    print(f"Total Directories: {total_dirs:,}")
    print(f"Case Folders:      {case_count:,}")

    print(f"\nYear Distribution:")
    for year in sorted(year_distribution.keys()):
        print(f"  {year}: {year_distribution[year]:,} case folders")

    print(f"\n" + "-" * 80)
    print("FILE TYPE BREAKDOWN")
    print("-" * 80)

    # Top file extensions
    print("\nTop File Types by Count:")
    for ext, count in extension_counts.most_common(15):
        size_mb = size_by_extension[ext] / (1024 * 1024)
        pct = (count / total_files) * 100
        print(f"  .{ext:10s} {count:6,} files ({pct:5.1f}%)  ~{size_mb:8.1f} MB total")

    # PDF/ODT analysis
    pdf_count = extension_counts.get('pdf', 0) + extension_counts.get('PDF', 0)
    odt_count = extension_counts.get('odt', 0)
    docx_count = extension_counts.get('docx', 0)

    print(f"\n" + "-" * 80)
    print("TEXT DOCUMENT ANALYSIS")
    print("-" * 80)
    print(f"\nTotal text documents: {pdf_count + odt_count + docx_count:,}")
    print(f"  PDF files:   {pdf_count:6,} ({pdf_count/(pdf_count+odt_count)*100:.1f}%)")
    print(f"  ODT files:   {odt_count:6,} ({odt_count/(pdf_count+odt_count)*100:.1f}%)")
    print(f"  DOCX files:  {docx_count:6,}")

    print(f"\nODT-PDF Pairing Analysis:")
    print(f"  Matched pairs: {odt_pdf_pairs:,} (ODT with corresponding PDF)")
    print(f"  Orphan ODTs:   {len(orphan_odts):,} (ODT without PDF)")
    print(f"  Pair coverage: {odt_pdf_pairs/odt_count*100:.1f}% of ODT files have PDF versions")

    print(f"\n" + "-" * 80)
    print("DOCUMENT TYPE DETECTION (by filename)")
    print("-" * 80)
    print(f"  Bescheide (BAMF decisions):     {bescheid_count:,}")
    print(f"  Klagen (lawsuits):              {klage_count:,}")
    print(f"  Schriftsätze (briefs):          {schriftsatz_count:,}")
    print(f"  Anhörungen (hearings):          {anhoerung_count:,}")

    # Non-text files
    print(f"\n" + "-" * 80)
    print("NON-TEXT FILES (to exclude from RAG)")
    print("-" * 80)

    exclude_exts = ['jpg', 'jpeg', 'png', 'heic', 'zip', '7z', 'mp4', 'xml', 'p7s']
    exclude_count = sum(extension_counts.get(ext, 0) for ext in exclude_exts)
    exclude_pct = (exclude_count / total_files) * 100

    print(f"  Images/videos: {extension_counts.get('jpg', 0) + extension_counts.get('jpeg', 0) + extension_counts.get('png', 0) + extension_counts.get('heic', 0):,}")
    print(f"  Archives:      {extension_counts.get('zip', 0) + extension_counts.get('7z', 0):,}")
    print(f"  Other:         {extension_counts.get('xml', 0) + extension_counts.get('p7s', 0) + extension_counts.get('mp4', 0):,}")
    print(f"  Total to exclude: {exclude_count:,} ({exclude_pct:.1f}% of all files)")

    # Recommendations
    print(f"\n" + "=" * 80)
    print("PARSER RECOMMENDATIONS")
    print("=" * 80)

    target_docs = pdf_count + odt_count
    print(f"\nTarget documents for RAG: {target_docs:,} (PDF + ODT)")
    print(f"\nStrategy:")
    print(f"  1. Convert ODT → PDF using LibreOffice headless")
    print(f"     • {odt_count:,} ODT files to convert")
    print(f"     • {odt_pdf_pairs:,} already have PDF versions (can skip if identical)")
    print(f"  2. Parse all PDFs with single parser")
    print(f"     • ~{pdf_count + odt_count:,} PDFs after conversion")

    print(f"\nParser choice:")
    if pdf_count > 5000:
        print(f"  ✅ PyMuPDF (FAST) - Recommended for large corpus ({pdf_count:,} PDFs)")
        print(f"     • ~0.2s per doc = {pdf_count * 0.2 / 60:.1f} minutes total")
        print(f"     • Good for well-structured legal PDFs")
    else:
        print(f"  ⚖️  Docling (ROBUST) - Acceptable for medium corpus ({pdf_count:,} PDFs)")
        print(f"     • ~2s per doc = {pdf_count * 2 / 60:.1f} minutes total")
        print(f"     • Better layout understanding, built-in OCR")

    print(f"\nEstimated processing time:")
    print(f"  PyMuPDF:  ~{target_docs * 0.2 / 60:.0f} minutes")
    print(f"  Docling:  ~{target_docs * 2 / 60:.0f} minutes")

    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "corpus_path": str(CORPUS_DIR),
        "summary": {
            "total_files": total_files,
            "total_directories": total_dirs,
            "case_folders": case_count,
            "target_documents": target_docs
        },
        "file_types": dict(extension_counts.most_common(20)),
        "year_distribution": dict(year_distribution),
        "document_types": {
            "bescheid": bescheid_count,
            "klage": klage_count,
            "schriftsatz": schriftsatz_count,
            "anhoerung": anhoerung_count
        },
        "odt_pdf_pairing": {
            "matched_pairs": odt_pdf_pairs,
            "orphan_odts": len(orphan_odts),
            "coverage_percent": round(odt_pdf_pairs/odt_count*100, 2) if odt_count > 0 else 0
        }
    }

    report_path = Path(__file__).parent / "data" / "corpus_analysis.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detailed report saved to: {report_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        analyze_corpus()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

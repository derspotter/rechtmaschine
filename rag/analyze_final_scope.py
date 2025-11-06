#!/usr/bin/env python3
"""
Final RAG Ingestion Scope Analysis
INCLUDES: Only Kanzlei Keienborg legal briefs (Schriftsätze, Klagen, etc.)
EXCLUDES: buchhaltung, 00 AT, Vollmacht, PKH/Mittellosigkeit
ANONYMIZATION: YES (via service_manager.py → anon service)
"""

import fitz
from pathlib import Path
import re
from collections import Counter
import sys

CORPUS_DIR = Path("/run/media/jayjag/My Book1/RAG/kanzlei/")

# Exclusion patterns
EXCLUDE_PATH_PATTERNS = [
    r'buchhalt',     # buchhaltung, buchhaltg folders
    r'rechnung',     # rechngen, rechnungen
    r'[/\\]00\s+at[/\\]',  # 00 AT folder (templates, admin)
    r'[/\\]00\s+at$',      # Match if path ends with /00 at
]

EXCLUDE_FILENAME_PATTERNS = [
    r'vollmacht',           # Vollmacht (power of attorney)
    r'pka',                 # PKH applications
    r'pkh',                 # PKH applications
    r'mittellos',           # Mittellosigkeit
]


def has_letterhead(pdf_path):
    """Quick letterhead check"""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return False
        text = doc[0].get_text().lower()
        doc.close()
        return 'keienborg' in text
    except:
        return False


def should_exclude(file_path):
    """Check if file should be excluded"""
    path_str = str(file_path).lower()
    filename = file_path.name.lower()

    # Check path patterns
    for pattern in EXCLUDE_PATH_PATTERNS:
        if re.search(pattern, path_str):
            return True, f"path:{pattern}"

    # Check filename patterns
    for pattern in EXCLUDE_FILENAME_PATTERNS:
        if re.search(pattern, filename):
            return True, f"filename:{pattern}"

    return False, None


def analyze_final_scope(sample_size=500):
    """
    Final analysis of documents to ingest for RAG.
    """
    print("=" * 80)
    print("RAG FINAL INGESTION SCOPE ANALYSIS")
    print("=" * 80)
    print(f"Corpus: {CORPUS_DIR}")
    print()
    print("✅ INCLUSION CRITERIA:")
    print("  • Has Kanzlei Keienborg letterhead")
    print("  • Legal briefs (Schriftsätze, Klagen, Beschwerden)")
    print()
    print("❌ EXCLUSION CRITERIA:")
    print("  • buchhaltung folders (accounting)")
    print("  • 00 AT folder (templates, admin)")
    print("  • Vollmacht files (power of attorney)")
    print("  • PKH/Mittellosigkeit files (legal aid)")
    print()
    print("🔒 ANONYMIZATION:")
    print("  • YES - via ../service_manager.py (port 8004)")
    print("  • Removes any remaining client names/details")
    print("  • Even Kanzlei docs may contain case-specific info")
    print()

    # Find all PDFs and ODTs
    print("Finding documents...")
    all_pdfs = list(CORPUS_DIR.glob("**/*.pdf")) + list(CORPUS_DIR.glob("**/*.PDF"))
    all_odts = list(CORPUS_DIR.glob("**/*.odt"))

    print(f"Found {len(all_pdfs):,} PDFs and {len(all_odts):,} ODTs")
    print()

    # Results
    results = {
        'include': [],
        'exclude_path': [],
        'exclude_filename': [],
        'exclude_no_letterhead': [],
        'errors': []
    }

    # Analyze ODTs
    print("Analyzing ODTs...")
    for odt_path in all_odts:
        excluded, reason = should_exclude(odt_path)
        if excluded:
            if reason.startswith('path'):
                results['exclude_path'].append(('odt', odt_path, reason))
            else:
                results['exclude_filename'].append(('odt', odt_path, reason))
        else:
            # ODTs are almost always Kanzlei docs (no letterhead check needed)
            results['include'].append(('odt', odt_path))

    # Sample PDFs
    if sample_size and sample_size < len(all_pdfs):
        print(f"Sampling {sample_size:,} random PDFs for analysis...")
        import random
        pdfs_to_check = random.sample(all_pdfs, sample_size)
    else:
        print(f"Checking all {len(all_pdfs):,} PDFs...")
        pdfs_to_check = all_pdfs

    print()

    # Analyze PDFs
    print("Analyzing PDFs...")
    for i, pdf_path in enumerate(pdfs_to_check, 1):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(pdfs_to_check)} ({i/len(pdfs_to_check)*100:.1f}%)", end='\r')

        # Check exclusions first
        excluded, reason = should_exclude(pdf_path)
        if excluded:
            if reason.startswith('path'):
                results['exclude_path'].append(('pdf', pdf_path, reason))
            elif reason.startswith('filename'):
                results['exclude_filename'].append(('pdf', pdf_path, reason))
            continue

        # Check letterhead
        if has_letterhead(pdf_path):
            results['include'].append(('pdf', pdf_path))
        else:
            results['exclude_no_letterhead'].append(('pdf', pdf_path))

    print()
    print()

    # Calculate statistics
    include_odt = sum(1 for t, _ in results['include'] if t == 'odt')
    include_pdf = sum(1 for t, _ in results['include'] if t == 'pdf')

    total_include = len(results['include'])
    total_exclude = (len(results['exclude_path']) +
                    len(results['exclude_filename']) +
                    len(results['exclude_no_letterhead']))

    # Report
    print("=" * 80)
    print("FINAL INGESTION SCOPE")
    print("=" * 80)
    print()

    print(f"✅ DOCUMENTS TO INGEST: {total_include:,}")
    print(f"   ├─ PDFs:  {include_pdf:,}")
    print(f"   └─ ODTs:  {include_odt:,}")
    print()

    print(f"❌ DOCUMENTS EXCLUDED: {total_exclude:,}")
    print(f"   ├─ Path exclusions (buchhaltung, 00 AT): {len(results['exclude_path']):,}")
    print(f"   ├─ Filename exclusions (vollmacht, pkh): {len(results['exclude_filename']):,}")
    print(f"   └─ No letterhead (external docs):        {len(results['exclude_no_letterhead']):,}")

    # Extrapolate if sampling
    if sample_size and sample_size < len(all_pdfs):
        print(f"\n" + "-" * 80)
        print("EXTRAPOLATED TO FULL CORPUS")
        print("-" * 80)

        total_pdfs = len(all_pdfs)
        sample_rate = len(pdfs_to_check) / total_pdfs

        est_include_pdf = int(include_pdf / sample_rate)
        est_exclude_path = int(sum(1 for t, _, _ in results['exclude_path'] if t == 'pdf') / sample_rate)
        est_exclude_filename = int(sum(1 for t, _, _ in results['exclude_filename'] if t == 'pdf') / sample_rate)
        est_exclude_external = int(len(results['exclude_no_letterhead']) / sample_rate)

        # Add back ODTs (already counted all)
        exclude_path_odt = sum(1 for t, _, _ in results['exclude_path'] if t == 'odt')
        exclude_filename_odt = sum(1 for t, _, _ in results['exclude_filename'] if t == 'odt')

        est_total_include = est_include_pdf + include_odt
        est_total_exclude = (est_exclude_path + exclude_path_odt +
                            est_exclude_filename + exclude_filename_odt +
                            est_exclude_external)

        print(f"\nEstimated for all {total_pdfs:,} PDFs + {len(all_odts):,} ODTs:")
        print(f"  ✅ To ingest:  ~{est_total_include:,}")
        print(f"     ├─ PDFs:   ~{est_include_pdf:,}")
        print(f"     └─ ODTs:    {include_odt:,}")
        print(f"  ❌ To exclude: ~{est_total_exclude:,}")
        print(f"     ├─ Paths:     ~{est_exclude_path + exclude_path_odt:,}")
        print(f"     ├─ Filenames: ~{est_exclude_filename + exclude_filename_odt:,}")
        print(f"     └─ External:  ~{est_exclude_external:,}")

    # Document type breakdown
    print(f"\n" + "-" * 80)
    print("INGESTED DOCUMENT TYPES")
    print("-" * 80)

    doc_type_counts = Counter()
    for doc_type, doc_path in results['include']:
        filename = doc_path.name.lower()

        if 'klage' in filename:
            doc_type_counts['Klage (lawsuit)'] += 1
        elif 'schriftsatz' in filename:
            doc_type_counts['Schriftsatz (explicit)'] += 1
        elif re.match(r'^\d{6}_', filename):
            # Date-prefixed files
            if 'vg' in filename or 'ovg' in filename or 'bverwg' in filename:
                doc_type_counts['Schriftsatz (court filing)'] += 1
            elif 'bamf' in filename:
                doc_type_counts['Schriftsatz (BAMF)'] += 1
            elif 'betreuung' in filename:
                doc_type_counts['Betreuung'] += 1
            elif 'beschwerde' in filename:
                doc_type_counts['Beschwerde (appeal)'] += 1
            else:
                doc_type_counts['Other dated docs'] += 1
        elif 'antrag' in filename:
            doc_type_counts['Antrag (application)'] += 1
        elif 'beschwerde' in filename:
            doc_type_counts['Beschwerde (appeal)'] += 1
        else:
            doc_type_counts['Other'] += 1

    for doc_type, count in doc_type_counts.most_common():
        pct = count / total_include * 100 if total_include > 0 else 0
        print(f"  {doc_type:40s} {count:5,} ({pct:5.1f}%)")

    # Sample documents
    print(f"\n" + "-" * 80)
    print("SAMPLE DOCUMENTS TO INGEST (first 25)")
    print("-" * 80)
    for doc_type, doc_path in results['include'][:25]:
        rel_path = doc_path.relative_to(CORPUS_DIR)
        print(f"  [{doc_type.upper()}] {rel_path}")

    # Pipeline details
    print(f"\n" + "=" * 80)
    print("RAG PIPELINE CONFIGURATION")
    print("=" * 80)

    if sample_size and sample_size < len(all_pdfs):
        est_total = est_total_include
    else:
        est_total = total_include

    print(f"\n📊 Corpus Size: ~{est_total:,} documents")

    print(f"\n🔄 Processing Steps:")
    print(f"  1. ODT→PDF conversion (LibreOffice headless)")
    print(f"     • {include_odt:,} ODT files")
    print(f"  2. PDF parsing (PyMuPDF + Docling fallback)")
    print(f"     • ~{est_total:,} PDFs after conversion")
    print(f"     • Extract text + structure")
    print(f"  3. Anonymization (service_manager.py → anon)")
    print(f"     • Remove client names, dates, places")
    print(f"     • Port: http://localhost:8004/anonymize")
    print(f"  4. Section classification")
    print(f"     • Detect: country_conditions, legal_argument, etc.")
    print(f"     • Regex patterns for German legal structure")
    print(f"  5. Deduplication (hash-based)")
    print(f"     • Expected dedup: 60-80% for legal sections")
    print(f"  6. Chunking (400-500 tokens, 100 overlap)")
    print(f"  7. Embedding (Qwen3-Embedding-4B on GPU)")
    print(f"  8. pgvector storage (PostgreSQL)")

    print(f"\n⏱️  Processing Time Estimate:")
    odt_time = include_odt * 2 / 60  # 2 sec per ODT
    pdf_time = est_total * 0.5 / 60  # 0.5 sec per PDF (hybrid)
    anon_time = est_total * 3 / 60   # 3 sec per doc (anonymization)
    chunk_time = est_total * 0.5 / 60  # 0.5 sec per doc
    embed_time = est_total * 2 / 60    # 2 sec per doc (GPU)
    total_time = odt_time + pdf_time + anon_time + chunk_time + embed_time

    print(f"  • ODT→PDF:       ~{int(odt_time)} min")
    print(f"  • PDF parsing:   ~{int(pdf_time)} min")
    print(f"  • Anonymization: ~{int(anon_time)} min")
    print(f"  • Chunking:      ~{int(chunk_time)} min")
    print(f"  • Embedding:     ~{int(embed_time)} min")
    print(f"  • Total:         ~{int(total_time)} min (~{total_time/60:.1f} hours)")

    print(f"\n💾 Storage Estimate:")
    avg_chunks_per_doc = 10  # Assuming ~5000 chars per doc, ~500 char chunks
    total_chunks = est_total * avg_chunks_per_doc
    dedup_rate = 0.7  # 70% deduplication for legal sections
    unique_chunks = int(total_chunks * (1 - dedup_rate))

    embedding_dim = 1024  # Qwen3-Embedding-4B reduced dimension
    bytes_per_embedding = embedding_dim * 4  # float32
    total_storage_mb = unique_chunks * bytes_per_embedding / (1024 * 1024)

    print(f"  • Total chunks (before dedup):  ~{total_chunks:,}")
    print(f"  • Unique chunks (after dedup):  ~{unique_chunks:,}")
    print(f"  • Vector storage (1024-dim):    ~{total_storage_mb:.0f} MB")
    print(f"  • Text storage:                 ~{unique_chunks * 500 / 1024 / 1024:.0f} MB")
    print(f"  • Total DB size:                ~{total_storage_mb + unique_chunks * 500 / 1024 / 1024:.0f} MB")

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
        analyze_final_scope(sample_size=sample_size)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

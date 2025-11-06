# RAG Corpus Analysis Summary

**Analysis Date:** 2025-10-31
**Corpus Location:** `/run/media/jayjag/My Book1/RAG/kanzlei/`

---

## 📊 Corpus Statistics

### Overall
- **Total Files:** 10,712
- **Total Directories:** 763 (433 case folders)
- **Years Covered:** 2023-2025
  - 2023: 144 cases
  - 2024: 206 cases
  - 2025: 98 cases

### Document Types
- **PDF Files:** 8,680 (76.2%)
- **ODT Files:** 1,671 (15.1%)
- **DOCX Files:** 38 (0.4%)
- **Other:** 897 (images, archives, etc.)

**Target for RAG:** 10,351 text documents (PDF + ODT + DOCX)

---

## 🔍 OCR Requirements Analysis

**Sample:** 500 PDFs analyzed

### Results:
- ✅ **Extractable (PyMuPDF works):** 69.8%
- 🔍 **Need OCR (scanned):** 30.0%
- ❌ **Errors (corrupted):** 0.2%

### Estimated for full corpus:
- ~5,694 PDFs with extractable text
- ~2,447 PDFs need OCR
- ~16 corrupted/empty files

### Files needing OCR:
- Scanned BAMF Bescheide
- Scanned Vollmacht forms
- Older court documents
- Foreign language documents (Arabic/Farsi)
- Mobile phone scans (kyoScan files)

**Recommendation:** Hybrid approach (PyMuPDF + Docling fallback for OCR)

---

## 📝 Letterhead Detection Analysis

**Sample:** 500 PDFs analyzed

### Detection Patterns:
**Strong indicators:**
- "keienborg" (unique surname)
- "friedrich-ebert-str" (office address)

**Supporting indicators:**
- "40210 düsseldorf"
- "marcel keienborg"
- "christian schotte"
- "kanzlei keienborg"

### Results:
- 📝 **Kanzlei Keienborg docs:** 47.0% (internal)
- 📄 **External docs:** 52.8% (BAMF, courts, etc.)
- ❌ **Errors:** 0.2%

### Estimated for full corpus:
- ~4,079 Kanzlei PDF docs
- ~1,671 Kanzlei ODT docs (almost all are internal)
- **Total internal:** ~5,750 documents
- **Total external:** ~4,583 documents

### Kanzlei Document Types (by filename):
- Schriftsätze (briefs): 26.8%
- Klagen (lawsuits): 8.5%
- Vollmacht: 4.3%
- PKH/Mittellosigkeit: 1.3%
- Other: 59.1%

---

## 🎯 Pipeline Recommendations

### 1. **Parser Strategy: Hybrid**

```
┌─────────────────────────────────────────────┐
│ PDF/ODT Ingestion                           │
│                                             │
│  ODT → LibreOffice → PDF (1,671 files)     │
│  ↓                                          │
│  Try PyMuPDF first (fast: ~70% success)    │
│  ↓                                          │
│  If text < 100 chars → Docling (OCR: ~30%) │
│  ↓                                          │
│  Extracted text + structure                 │
└─────────────────────────────────────────────┘
```

**Processing time estimate:**
- PyMuPDF fast path (70%): ~30 minutes
- Docling OCR path (30%): ~1.5 hours
- **Total: ~2 hours** (vs 5+ hours for Docling-only)

### 2. **Document Classification**

Add metadata fields:
- `is_kanzlei_doc`: boolean (detected via letterhead)
- `doc_origin`: enum ('internal' | 'external')
- `doc_type`: enum ('klage', 'schriftsatz', 'bescheid', 'vollmacht', etc.)
- `confidence`: float (letterhead detection confidence)

### 3. **Anonymization Strategy**

**Kanzlei docs (47%):**
- ✅ **SKIP anonymization**
- Already written generically
- Saves processing time and API costs
- ~5,750 documents

**External docs (53%):**
- ✅ **FULL anonymization required**
- Contains client names, addresses, birthdates
- Use existing anonymization service (port 9002)
- ~4,583 documents

**Estimated savings:**
- Skip anonymization for 47% of docs
- Reduces processing time by ~40%
- Reduces API costs by ~40%

### 4. **Section Detection**

**Kanzlei docs:**
- Reliable structure: "I.", "II.", "III."
- Standard headings: "Zur Lage in", "Rechtliche Würdigung"
- High regex success rate

**External docs:**
- Varied formats (BAMF vs court decisions)
- Need robust heading detection
- Fallback to paragraph-based splitting

### 5. **Deduplication Priority**

**Kanzlei docs (HIGH dedup potential):**
- Reuse country condition arguments
- Templated legal reasoning blocks
- Medical risk arguments (PTSD, suicide)
- Expected dedup rate: 60-80% for legal sections

**External docs (LOW dedup):**
- Unique BAMF decisions
- Unique court rulings
- Expected dedup rate: <10%

### 6. **Metadata Extraction**

**From Kanzlei filenames:**
- Date: YYMMDD prefix (e.g., `251013_vg_klage.pdf`)
- Type: klage, schriftsatz, vollmacht, etc.
- Court: vg, ovg, bverwg, etc.

**From external content:**
- BAMF Az: regex `\d{7}-\d{3}`
- Court Az: regex `K \d+/\d+\.A`
- Date: extract from "Beschluss vom DD.MM.YYYY"
- Country: extract from text content

---

## 📁 Corpus Structure

```
/run/media/jayjag/My Book1/RAG/kanzlei/
├── 23/                          # 2023 cases (144 folders)
│   ├── 001 NAME vs BRD/
│   │   ├── 230125_vg_klage.odt      # Kanzlei doc (date-prefixed)
│   │   ├── 230125_vg_klage.pdf      # PDF export
│   │   ├── anlage_k1_vollmacht.pdf  # Client power of attorney
│   │   ├── anlage_k2_bescheid.pdf   # BAMF decision (external)
│   │   └── doc*.pdf                 # Scanned documents (need OCR)
│   └── ...
├── 24/                          # 2024 cases (206 folders)
├── 25/                          # 2025 cases (98 folders)
└── 00 AT/                       # Templates and admin
    └── vorlagen/                # Kanzlei templates
        ├── Gericht_m.odt        # Court letter template
        ├── Mandant_m.odt        # Client letter template
        └── vollmacht_asyl.pdf   # Power of attorney template
```

---

## 🚀 Next Steps

1. **Finalize requirements.txt** with PyMuPDF, Docling, transformers, pgvector
2. **Implement hybrid parser** (PyMuPDF + Docling fallback)
3. **Implement ODT converter** (LibreOffice headless)
4. **Implement letterhead detection** (integrate into parser)
5. **Implement conditional anonymization** (external docs only)
6. **Implement section classifier** (German legal patterns)
7. **Implement deduplication** (hash-based for Kanzlei sections)
8. **Set up Postgres + pgvector**
9. **Implement embedding** (Qwen3-Embedding-4B)
10. **Build retrieval API**

---

## 💾 Analysis Files

- `corpus_analysis.json` - Detailed file type breakdown
- `analyze_corpus.py` - Corpus structure analyzer
- `test_ocr_needs.py` - OCR requirements tester
- `detect_letterhead.py` - Letterhead detection script

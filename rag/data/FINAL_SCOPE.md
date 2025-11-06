# RAG System - Final Ingestion Scope

**Analysis Date:** 2025-10-31
**Corpus:** `/run/media/jayjag/My Book1/RAG/kanzlei/`

---

## ✅ What Gets Ingested

**~3,122 documents** consisting of:
- **Kanzlei Keienborg legal briefs** (Schriftsätze, Klagen, Beschwerden)
- **Date-prefixed court filings** (e.g., `251013_vg_klage.odt`)
- **Case-specific legal documents**

### Document Types:
- 51.0% - Schriftsätze (court filings)
- 20.5% - Klagen (lawsuits)
- 17.8% - Other dated documents
-  9.2% - Other legal docs
-  1.0% - BAMF filings
-  0.3% - Beschwerden (appeals)

### File Formats:
- 1,718 PDFs
- 1,404 ODTs (will be converted to PDF)

---

## ❌ What Gets Excluded

**~7,227 documents** excluded:
- ~3,122 - Buchhaltung folders (accounting/billing)
- ~3,122 - 00 AT folder (templates, admin)
- ~703 - Vollmacht files (power of attorney)
- ~703 - PKH/Mittellosigkeit (legal aid applications)
- ~3,402 - External documents (BAMF decisions, court rulings without letterhead)

---

## 🔧 Pipeline Configuration

### 1. **Document Filtering**
```python
INCLUDE:
✓ Has "Kanzlei Keienborg" letterhead
✓ Located in case folders (23/*, 24/*, 25/*)

EXCLUDE:
✗ buchhaltung/* folders
✗ 00 AT/* folder
✗ *vollmacht* files
✗ *pkh* / *pka* / *mittellos* files
✗ No letterhead (external docs)
```

### 2. **Processing Steps**

```
Step 1: ODT→PDF Conversion (LibreOffice)
├─ Input:  1,404 ODT files
├─ Tool:   soffice --headless --convert-to pdf
└─ Time:   ~46 minutes

Step 2: PDF Parsing (Hybrid)
├─ Primary:  PyMuPDF (fast, 70% success)
├─ Fallback: Docling (OCR for scanned docs, 30%)
├─ Input:    ~3,122 PDFs
└─ Time:     ~26 minutes

Step 3: Anonymization (REQUIRED)
├─ Service: http://localhost:8004/anonymize
├─ Backend: Qwen3-14B via Ollama (port 9002)
├─ Purpose: Remove client names, case-specific details
├─ Note:    Even Kanzlei docs may contain personal info
└─ Time:    ~156 minutes (longest step)

Step 4: Section Classification
├─ Patterns: German legal structure
│   • I., II., III. (numbered sections)
│   • "Zur Lage in..." (country conditions)
│   • "Rechtliche Würdigung" (legal reasoning)
│   • "Sachverhalt" (facts)
├─ Types: country_conditions, legal_argument,
│         medical_risk, procedural_background
└─ Time:  ~10 minutes

Step 5: Deduplication (Hash-based)
├─ Strategy: SHA-256 hash of normalized text
├─ Scope:    Legal sections (not personal facts)
├─ Expected: 60-80% deduplication rate
│   • Country conditions blocks highly reused
│   • Legal reasoning templates common
│   • Medical risk arguments (PTSD) repeated
└─ Time:  ~15 minutes

Step 6: Chunking
├─ Size:    400-500 tokens per chunk
├─ Overlap: 100 tokens between chunks
├─ Total:   ~31,220 chunks (before dedup)
├─ Unique:  ~9,366 chunks (after dedup)
└─ Time:    ~26 minutes

Step 7: Embedding (GPU)
├─ Model:   Qwen3-Embedding-4B
├─ Device:  CUDA (RTX 3060 12GB VRAM)
├─ Dim:     1024 (reduced from 2560)
├─ Batch:   8 chunks per batch
└─ Time:    ~104 minutes

Step 8: Vector Database Storage
├─ DB:      PostgreSQL + pgvector
├─ Index:   HNSW or IVFFlat
├─ Columns: id, text, metadata (JSONB), embedding (vector(1024))
└─ Time:    ~10 minutes

Total Pipeline Time: ~6 hours
```

### 3. **Storage Estimates**

```
Vector Database (PostgreSQL + pgvector):
├─ Chunks:              ~9,366 unique
├─ Vector storage:      ~37 MB (1024-dim float32)
├─ Text storage:        ~4 MB
├─ Metadata (JSONB):    ~5 MB
└─ Total:               ~46 MB (very reasonable!)

Deduplication Savings:
├─ Before dedup:  31,220 chunks → ~122 MB
├─ After dedup:    9,366 chunks → ~37 MB
└─ Savings:        70% reduction (21,854 chunks eliminated)
```

### 4. **Metadata Fields**

Each chunk will have:

```json
{
  "chunk_id": "uuid",
  "source_file": "24/014 NOUR vs BRD/251013_vg_klage.pdf",
  "source_type": "odt",
  "doc_date": "2025-10-13",
  "court": "vg",
  "doc_type": "klage",
  "section_type": "legal_argument",
  "case_folder": "24/014 NOUR vs BRD",
  "anonymized": true,
  "anonymizer_version": "v1.0",
  "embedding_model": "Qwen3-Embedding-4B-dim1024",
  "canonical_hash": "sha256:...",
  "created_at": "2025-10-31T..."
}
```

---

## 🎯 Key Optimizations

### 1. **No External Documents**
- Only ingest your own legal work
- Excludes BAMF decisions, court rulings
- Focused corpus = better retrieval quality

### 2. **Smart Exclusions**
- Skip accounting docs (irrelevant)
- Skip templates (redundant with actual briefs)
- Skip Vollmacht (boilerplate)
- Skip PKH (administrative)

### 3. **High Deduplication**
- Legal arguments are templated
- Country condition sections reused extensively
- 70% deduplication rate = massive storage savings

### 4. **Anonymization Required**
- Even your own briefs may contain:
  - Client names (if not carefully written)
  - Case-specific dates/locations
  - Identifying details
- Better safe than sorry for GDPR compliance

---

## 📊 Example Documents

Sample ingested documents:
```
24/003 ALI vs BRD/240108_ag_beschwerde.odt
24/007 FARAZANEH vs BRD/240110_vg_klage_80v.odt
24/014 NOUR vs BRD/251013_vg_klage.odt
24/021 ALBUAWADH vs BRD/240205_vg_ae.odt
25/015 NAME vs BRD/250315_ovg_beschwerde.odt
```

Naming convention:
```
YYMMDD_court_doctype.odt
  ↓      ↓       ↓
 Date   VG/OVG  klage/schriftsatz/beschwerde
```

---

## 🚀 Next Implementation Steps

1. **Create requirements.txt** ✓ (partially done)
2. **Implement ODT converter** (LibreOffice subprocess)
3. **Implement hybrid PDF parser** (PyMuPDF + Docling)
4. **Implement letterhead filter** (exclude non-Kanzlei docs)
5. **Integrate anonymization** (service_manager.py client)
6. **Implement section classifier** (regex + heuristics)
7. **Implement deduplicator** (SHA-256 hash)
8. **Implement chunker** (tiktoken-based)
9. **Set up PostgreSQL + pgvector**
10. **Implement embedder** (Qwen3-Embedding-4B on GPU)
11. **Create ingestion orchestrator**
12. **Build retrieval API**
13. **Test with sample documents**

---

## ✅ Validation Checks

Before production deployment:

- [ ] Anonymization removes all client names
- [ ] No documents from buchhaltung/ ingested
- [ ] No documents from 00 AT/ ingested
- [ ] No Vollmacht files ingested
- [ ] No PKH files ingested
- [ ] All ingested docs have Kanzlei letterhead
- [ ] Deduplication working (check hash collisions)
- [ ] Vector search returns relevant results
- [ ] Metadata extraction accurate (dates, courts, types)
- [ ] Database size within expectations (~50 MB)

---

## 📝 Notes

- **Privacy:** All documents anonymized before embedding
- **GDPR:** No client names/PII in vector database
- **Performance:** GPU embedding on RTX 3060 (12GB VRAM)
- **Scalability:** System designed for ~3,000-5,000 documents
- **Future growth:** Can re-index if embedding model changes

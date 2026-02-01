# Anonymization Audit Plan (Kanzlei RAG, text-only)

Status: plan
Scope date: 2026-02-01
Owner: Rechtmaschine

## 0) Scope and constraints
- Corpus source: `/run/media/jayjag/My Book1/RAG/kanzlei/`
- Sample size: 200 documents
- Only Kanzlei documents (letterhead detected) from case folders (23/*, 24/*, 25/*)
- Text anonymization only (no PDF redaction in this audit)
- Local storage allowed; do not commit any PII or extracted text
- Use the same extraction logic as the production endpoint

## 1) Goals and success criteria
- Robustness: stable behavior under service down/timeout/5xx/corrupt input
- Speed: P50/P95 latency targets for each path
  - cached text + cached anonymization
  - fresh text + direct extraction
  - fresh text + OCR
- Quality: high PII recall with controlled over-redaction
  - Report recall/precision per label and per document class
  - Track "PII beyond first 5k chars" as explicit risk

## 2) Sampling strategy (200 Kanzlei docs)
Use a stratified sample to cover realistic variance:
- Years: 2023 / 2024 / 2025
- Doc types: Schriftsatz, Klage, Beschwerde, sonstige dated docs
- Input format: PDF (text), PDF (scan), ODT->PDF
- Length buckets: 1-5, 6-20, 20+ pages
- OCR needed vs not needed

Operational steps:
1) Build candidate list using existing scripts:
   - `rag/analyze_final_scope.py` and/or `rag/analyze_kanzlei_docs.py`
   - `rag/detect_letterhead.py` for letterhead validation
2) Exclude: buchhaltung, 00 AT, vollmacht, pkh/mittellos, non-letterhead
3) Stratify and sample 200 docs, store manifest with metadata

## 3) Audit dataset layout (local only)
```
/test_files/anonymization_audit/
  raw/               # copies or symlinks to selected PDFs
  extracted/         # extracted plain text snapshots
/tests/anonymization_audit/
  manifest.jsonl     # doc_id, path, year, doc_type, ocr_needed, length
  labels/            # gold spans (jsonl)
  results/           # metrics + failure cases
/reports/
  anonymization_audit_YYYYMMDD.md
```

Add .gitignore entries so raw/extracted data never commits.

## 4) Extraction snapshot (ground truth input)
- Use the same extraction path as the endpoint:
  - `extract_pdf_text` with quality threshold
  - `check_pdf_needs_ocr` to decide OCR
  - `perform_ocr_on_file` for scans
- Record for each doc:
  - extracted length, OCR used, time spent
  - quality heuristics (text length, strip length)
- Save extracted text to `test_files/anonymization_audit/extracted/`

## 5) Labeling (gold PII spans)
Label schema (minimum):
- PERSON, ADDRESS, BIRTH_DATE, ID (BAMF/AZR/Aktenzeichen), CONTACT

Process:
- Pre-label with regex for IDs, dates, phone, email
- Add case-specific protected terms where available (case folder names, doc headers)
- Human review and correction
- Double-annotate 10-20% for agreement

## 6) Evaluation harness
Batch runner should:
- Call `/documents/{id}/anonymize` and `/anonymize-file`
- Capture:
  - anonymized text
  - processed_characters
  - ocr_used
  - timing breakdown

Metrics:
- Recall/precision per label
- Over-redaction rate (removed non-PII)
- Leak checks:
  - known terms scan (protected terms)
  - second-pass regex/NER on output
- "Remainder risk": PII found after first 5k chars

Outputs:
- `tests/anonymization_audit/results/metrics.json`
- `tests/anonymization_audit/results/failures.jsonl`

## 7) Robustness tests
- Service down / timeouts / 5xx
- Corrupt PDF and missing file
- Large docs (20+ pages)
- Concurrency: 10-20 parallel requests

Record errors, status codes, retries, and latency impact.

## 8) Performance profiling
- Stage timings: extraction, OCR, anonymization call, stitching, IO, DB commit
- P50/P95 per path (cached, no OCR, OCR)
- Identify top 3 hotspots by time share

## 9) Reporting and gates
`reports/anonymization_audit_YYYYMMDD.md` should include:
- Summary table of latency/robustness/quality
- Leak rate and top failure patterns
- Remainder-risk stats
- Prioritized fix list

Define gates (example, adjust):
- 0 high-risk leaks in PERSON/ADDRESS
- >= 98% recall for PERSON/ADDRESS
- P95 latency targets by path

## 10) Compliance notes (text-only)
- Treat output as pseudonymized unless proven anonymized
- Keep audit artifacts local, access-restricted
- Document retention and deletion after audit


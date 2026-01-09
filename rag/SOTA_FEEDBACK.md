# RAG Plan SOTA Feedback Review

**Date:** 2026-01-10
**Reviewed by:** Codex (GPT-5.1), Gemini 2.5 Flash, Gemini 3 Pro
**Plan reviewed:** `gpt5plan.md`, `data/corpus_summary.md`, `data/FINAL_SCOPE.md`

---

## Executive Summary

All three models agree the current plan has a solid foundation for privacy and preprocessing, but relies on "Generation 1" retrieval (dense vector search only) which is insufficient for legal domains requiring high precision. Key upgrades needed:

1. **Hybrid Search** (Dense + Sparse/BM25)
2. **Re-ranking** (Cross-encoder)
3. **Contextual/Late Chunking**
4. **Evaluation Metrics**

---

## Codex (GPT-5.1) Feedback

### What's Strong
- Clear local-only processing with GDPR-conscious anonymization before embedding
- Solid hybrid parsing + OCR fallback, regex-based sectioning, metadata-rich chunk schema
- Deduplication plan to cut redundancy and track provenance
- Tight scope with explicit inclusions/exclusions and processing estimates

### Missing SOTA Techniques (2025-2026)

| Technique | Why It's Needed |
|-----------|----------------|
| **Hybrid Dense-Sparse** | BM25/SPLADE/BGE-M3 lexical side + ColBERT-v2.5 multi-vector late-interaction for citation-heavy text |
| **Query Expansion** | HyDE, multi-query generation, cross-lingual variants — current plan embeds raw query only |
| **Reranking Stage** | Cross-encoder or lightweight LLM reranker to reorder top-50 candidates (none planned) |
| **Late Chunking** | RAPTOR-style hierarchical retrieval, GraphRAG neighborhoods to combat chunk boundary issues |
| **Evaluation Harness** | Synthetic Q&A + human checks with nDCG/MRR/Recall@K — plan lacks measurement loop |
| **Knowledge-aware** | Cross-chunk citation linking, parent-child doc structure, contextual compression |

### German-Legal Recommendations
- German-legal-tuned embedding/reranker (multilingual BGE-M3 or de-legal DeBERTa/BERT)
- Parse legal cites, court/chamber, decision dates, BAMF/Court Az as first-class filters
- Keep citations un-anonymized but separate from PII to preserve retrieval fidelity
- Support bilingual queries (German/English) with translation-aware expansion and diacritic normalization
- Near-duplicate detection (MinHash/LCS) instead of strict hashing — slightly edited briefs shouldn't collapse
- Semantic splits at headings like "Sachverhalt"/"Rechtliche Würdigung" to maintain rhetorical roles

### Risks & Concerns
- **Anonymization policy conflict**: corpus_summary.md suggests skipping for internal docs, gpt5plan.md anonymizes everything — reconcile
- **Hash dedup loses updates**: May drop meaningful version changes (new dates/facts), lose temporal nuance
- **1024-dim reduction**: Could hurt nuance in long legal arguments — test native dimension or PQ/OPQ
- **pgvector IVFFlat only**: For 9k vectors it's fine, but add precise rerank and periodic vacuum/analyze
- **Excluding external docs**: Caps answer completeness for authority citations — allow optional external index
- **No monitoring**: Missing recall drift or anonymization failure alerts

---

## Gemini 2.5 Flash Feedback

### What's Good
- Privacy-first local execution addresses GDPR/Berufsgeheimnis
- Rigorous data filtering (letterhead detection, exclusions) — garbage-in-garbage-out is #1 RAG failure
- Qwen3 good choice for German/multilingual
- Storage efficiency via deduplication (70% reduction)

### Critical Missing Techniques

| Technique | Why |
|-----------|-----|
| **Hybrid Search** | Dense vectors miss exact matches like `24/014`, `§ 60 Abs. 5 AufenthG` |
| **Re-Ranking** | Cross-encoder or ColBERTv2 for precision — retrieve 50, rank to 8 |
| **Late Chunking** | 500-token chunks break legal argument flow ("The appeal is therefore unfounded..." loses WHY) |
| **Contextual Retrieval** | Prepend generated summary to each chunk (Anthropic approach) |

### German-Legal Recommendations
- Add `tsvector` column for PostgreSQL full-text search
- Use **Reciprocal Rank Fusion (RRF)** to merge vector + keyword results
- Provenance-aware dedup queries — ensure metadata filtering works across provenance array (JOIN not WHERE)
- Model recommendation: `BAAI/bge-reranker-v2-m3` for German

### Risks & Concerns
- **Dedup vs Context**: 70% dedup destroys case-specific "flavor" and tailoring
- **PDF Parsing**: PyMuPDF misses margin notes, stamps, multi-column layouts
- **Anonymization latency**: 156 min for 3k docs via 14B model is optimistic — ensure batching

### Suggested Todo Updates
1. Modify DB Schema: Add support for Hybrid Search (`tsvector` or sparse columns)
2. Update Retrieval Logic: Implement Hybrid Search (Keyword + Vector) + RRF
3. Add Re-ranker: Integrate a local Cross-Encoder service
4. Refine Dedup Query: Ensure metadata filtering works across `provenance` array

---

## Gemini 3 Pro Feedback

### What's Good
- **Privacy & Compliance First**: Mandatory anonymization before embedding aligns with GDPR/Berufsgeheimnis
- **Smart Pre-processing**: Hybrid Parser (PyMuPDF + Docling) optimizes ingestion — fast for 70%, heavy OCR only when needed
- **Canonical Deduplication**: Prevents retrieving 5 identical "Safety in Damascus" arguments
- **Scoped Ingestion**: Excluding Buchhaltung/templates reduces noise

### Critical Missing SOTA Techniques

| Technique | Issue | Fix |
|-----------|-------|-----|
| **Hybrid Search** | Pure vector fails on exact matches (`§ 60 Abs. 5`, case numbers `23/001`) | BM25 + Vector with RRF fusion |
| **Re-ranking** | Bi-encoders lose nuance between "General rejection" vs "Specific rejection based on health" | Cross-encoder (BGE-Reranker-v2) on top-50 → top-8 |
| **Contextual Retrieval** | Chunks like "application rejected due to credibility" lack context — WHICH application? WHY? | Prepend LLM-generated header to every chunk |
| **Late Chunking** | Cutting context loses meaning | Process full doc before chunking (Jina-v3 support) |

### German-Legal Recommendations
- **Enhanced metadata schema**: Separate fields for:
  - `statute` (e.g., "AufenthG")
  - `paragraph` (e.g., "§ 60")
  - `applicant_origin` (e.g., "Syria")
- **Hybrid DB schema**: Keep pgvector + add `TSVECTOR` column for German full-text search
- **GraphRAG Lite**: Extract citation graph — "Find cases citing BVerwG 1 C 10.16" becomes deterministic
- **Metadata-Filtered Search**: Allow queries like "Show me arguments about §60 for Syrians"

### Concerns

| Issue | Risk | Mitigation |
|-------|------|------------|
| **Qwen3-Embedding-4B too large** | 500ms+ query latency on RTX 3060, slow re-indexing | Switch to `bge-m3` (multilingual, dense+sparse native) or `jina-embeddings-v3` |
| **LLM anonymization probabilistic** | Will miss names occasionally, may hallucinate replacements that change legal meaning | Add regex safety net post-processing, flag high-entropy chunks for human review |

### Actionable Changes
1. **Hybrid Search**: Add `TSVECTOR` column, implement BM25+Vector with RRF
2. **Re-ranking**: Cross-encoder step (`bge-reranker-v2-m3`)
3. **Contextual Chunking**: Prepend `Doc Title + Section Header` before embedding
4. **Evaluate embedding model**: Benchmark Qwen3 vs `bge-m3` for latency

---

## Consensus: All Three Models Agree

| Upgrade | Codex | Gemini Flash | Gemini 3 Pro |
|---------|:-----:|:------------:|:------------:|
| Hybrid Dense+Sparse (BM25/SPLADE) | ✅ | ✅ | ✅ |
| Re-ranking (Cross-encoder) | ✅ | ✅ | ✅ |
| Contextual/Late Chunking | ✅ | ✅ | ✅ |
| German-specific metadata filtering | ✅ | ✅ | ✅ |
| Evaluation metrics (nDCG/MRR/Recall@K) | ✅ | — | — |
| bge-m3 as embedding alternative | ✅ | ✅ | ✅ |
| Provenance-aware dedup queries | ✅ | ✅ | — |
| Citation graph / GraphRAG lite | ✅ | — | ✅ |

---

## Recommended Architecture Updates

### 1. Retrieval Pipeline (SOTA 2025-2026)

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Query Processing                   │
│  - Query expansion (HyDE/multi-query)│
│  - Bilingual normalization          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Hybrid Retrieval (Top 50)          │
│  - Dense: pgvector cosine           │
│  - Sparse: TSVECTOR/BM25            │
│  - Fusion: Reciprocal Rank Fusion   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Re-ranking (Top 8)                 │
│  - Cross-encoder: bge-reranker-v2-m3│
│  - Or: ColBERTv2 late interaction   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Context Assembly                   │
│  - Metadata filtering               │
│  - Parent-child expansion           │
│  - Citation linking                 │
└─────────────────────────────────────┘
    │
    ▼
  LLM Generation
```

### 2. Updated Database Schema

```sql
CREATE TABLE rag_chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,

    -- Contextual header (NEW)
    context_header TEXT,  -- "Klage Nour vs BRD, Section: Rechtliche Würdigung"

    -- Rich metadata (ENHANCED)
    metadata JSONB NOT NULL,
    -- metadata should include:
    -- - section_type: legal_argument | country_conditions | medical_risk | procedural
    -- - statute: "AufenthG" | "AsylG" | "GG"
    -- - paragraph: "§ 60" | "§ 3"
    -- - applicant_origin: "Afghanistan" | "Syria" | "Iran"
    -- - court: "VG Düsseldorf" | "OVG NRW"
    -- - date: "2024-10-13"
    -- - citations: ["BVerwG 1 C 10.16", "EuGH C-91/20"]

    -- Embeddings
    embedding VECTOR(1024),

    -- Full-text search (NEW)
    text_search TSVECTOR GENERATED ALWAYS AS (to_tsvector('german', text)) STORED,

    -- Provenance for deduped blocks
    provenance JSONB,  -- Array of source doc_ids
    canonical_block_id TEXT,

    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX rag_chunks_embedding_idx ON rag_chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX rag_chunks_text_search_idx ON rag_chunks USING GIN (text_search);
CREATE INDEX rag_chunks_statute_idx ON rag_chunks ((metadata->>'statute'));
CREATE INDEX rag_chunks_origin_idx ON rag_chunks ((metadata->>'applicant_origin'));
CREATE INDEX rag_chunks_citations_idx ON rag_chunks USING GIN ((metadata->'citations'));
```

### 3. Embedding Model Comparison

| Model | Dimension | German Quality | Sparse Support | Latency (RTX 3060) |
|-------|-----------|----------------|----------------|-------------------|
| Qwen3-Embedding-4B | 1024 (reduced) | Good | No | ~500ms |
| bge-m3 | 1024 | Excellent | Native | ~50ms |
| jina-embeddings-v3 | 1024 | Excellent | No | ~80ms |
| E5-mistral-7b | 4096 | Very Good | No | ~800ms |

**Recommendation**: Switch to `bge-m3` for production — faster, native sparse vectors, excellent German support.

### 4. Contextual Chunking Format

Before embedding, prepend context header:

```
[Klage | 24/014 NOUR vs BRD | VG Düsseldorf | 2024-10-13]
[Section: Rechtliche Würdigung | §60 Abs. 5 AufenthG | Afghanistan]

{Original chunk text here...}
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
- [ ] Add `TSVECTOR` column to schema
- [ ] Implement hybrid retrieval (vector + BM25 + RRF)
- [ ] Add context header to chunks before embedding

### Phase 2: Quality Boost (3-5 days)
- [ ] Integrate cross-encoder reranker (`bge-reranker-v2-m3`)
- [ ] Benchmark Qwen3 vs bge-m3 embedding latency
- [ ] Enhanced metadata extraction (statute, paragraph, citations)

### Phase 3: Advanced (1-2 weeks)
- [ ] Build evaluation harness (synthetic Q&A, nDCG/MRR metrics)
- [ ] Citation graph extraction (GraphRAG lite)
- [ ] Query expansion (HyDE, multi-query generation)
- [ ] Near-duplicate detection (MinHash) for dedup

---

## References

- [ColBERT v2](https://github.com/stanford-futuredata/ColBERT)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [BGE Reranker v2](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [RAPTOR](https://arxiv.org/abs/2401.18059)
- [HyDE](https://arxiv.org/abs/2212.10496)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

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

---

## Implementation Decision: BGE-M3 Hybrid (January 2026)

Based on additional analysis from Gemini and GPT-5, we've selected **BAAI/bge-m3** as our embedding model for hybrid retrieval.

### Why BGE-M3?

| Property | Value |
|----------|-------|
| **Type** | Hybrid (Dense + Sparse + Multi-Vector) |
| **Parameters** | ~567M |
| **VRAM Usage** | ~1.2GB (FP16) |
| **Context Length** | 8192 tokens |
| **Dense Dimension** | 1024 |

**Key Advantage:** Unlike dense-only models (Qwen-Embedding, Nomic), BGE-M3 outputs **two signals simultaneously**:

1. **Dense Vector (1024 dim):** Semantic/concept search ("feline" matches "cat")
2. **Sparse Vector (Lexical):** Weighted keywords like smart BM25 - prevents missing exact matches like `§ 60 Abs. 5` or case numbers `24/014`

---

### VRAM Management: "Baton Pass" Strategy

With RTX 3060 12GB, we cannot load all models simultaneously:

| Component | VRAM |
|-----------|------|
| System Overhead | ~1.5GB |
| Qwen 14B (Q4_K_M) | ~9.0GB |
| PaddleOCR | ~1.0GB |
| BGE-M3 | ~1.2GB |
| **Total if simultaneous** | **~12.7GB** ❌ |

**Solution:** Never have Qwen and BGE-M3 loaded at the same moment.

```
┌─────────────────────────────────────────────────────────────┐
│  RAG Query Pipeline (VRAM-Aware)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: EMBED QUERY                                        │
│  ├── Load BGE-M3 (~300ms from NVMe)                        │
│  ├── Generate dense + sparse vectors                        │
│  └── Unload BGE-M3                                          │
│                                                             │
│  Step 2: HYBRID RETRIEVAL                                   │
│  ├── Vector DB search (no VRAM - runs on CPU/disk)         │
│  ├── Dense search → top 50                                  │
│  ├── Sparse search → top 50                                 │
│  └── RRF fusion → top 30                                    │
│                                                             │
│  Step 3: RERANK (Optional)                                  │
│  ├── Load cross-encoder (~500MB)                           │
│  ├── Rerank top 30 → top 8                                  │
│  └── Unload reranker                                        │
│                                                             │
│  Step 4: GENERATE                                           │
│  ├── Load Qwen 14B                                          │
│  ├── Generate answer with context                           │
│  └── Unload Qwen 14B                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Performance Note:** BGE-M3 loads in ~300-500ms from NVMe SSD. Negligible for RAG latency.

---

### Inference Server: Text Embeddings Inference (TEI)

HuggingFace TEI is the fastest option for hybrid embeddings with sparse vector support.

**Docker Command:**
```bash
docker run -d --name bge-m3 \
  -p 8085:80 \
  -v $HOME/.cache/huggingface:/data \
  --gpus all \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-m3 \
  --pooling cls \
  --dtype float16 \
  --max-batch-tokens 16384
```

**Python Client (Dense + Sparse):**
```python
import httpx

TEI_URL = "http://localhost:8085"

def embed_hybrid(text: str) -> tuple[list[float], dict]:
    """Get both dense and sparse embeddings from BGE-M3"""

    # Dense embedding
    dense_response = httpx.post(
        f"{TEI_URL}/embed",
        json={"inputs": text}
    ).json()
    dense_vector = dense_response[0]  # 1024-dim

    # Sparse embedding (lexical weights)
    sparse_response = httpx.post(
        f"{TEI_URL}/embed_sparse",
        json={"inputs": text}
    ).json()
    sparse_vector = sparse_response[0]  # {indices: [...], values: [...]}

    return dense_vector, sparse_vector

# Example usage
dense, sparse = embed_hybrid("§ 60 Abs. 5 AufenthG Abschiebungsverbot Afghanistan")
# dense: [0.023, -0.041, ...] (1024 floats)
# sparse: {'indices': [101, 2561, ...], 'values': [0.5, 0.8, ...]}
```

---

### Hybrid Retrieval with RRF Fusion

**Reciprocal Rank Fusion (RRF)** combines BM25/sparse and dense results:

```python
def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],  # [(doc_id, score), ...]
    sparse_results: list[tuple[str, float]],
    k: int = 60  # smoothing constant
) -> list[tuple[str, float]]:
    """
    Combine dense and sparse results using RRF.

    score(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)
    """
    scores = {}

    for rank, (doc_id, _) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Sort by combined score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

**Recommended Settings:**
```python
# Retrieval parameters
DENSE_TOP_K = 50      # Retrieve 50 from dense index
SPARSE_TOP_K = 50     # Retrieve 50 from sparse/BM25 index
FUSED_TOP_K = 30      # After RRF fusion, keep top 30
RERANK_TOP_K = 8      # After cross-encoder rerank, keep top 8

# Chunking parameters
CHUNK_SIZE = 500      # tokens (300-800 range)
CHUNK_OVERLAP = 50    # 10-20% overlap
```

---

### Vector Database: Qdrant (Recommended)

Qdrant supports native hybrid search with dense + sparse vectors in the same collection.

**Docker Setup:**
```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -v $HOME/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Collection Schema for Hybrid:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, Distance,
    PointStruct, SparseVector
)

client = QdrantClient("localhost", port=6333)

# Create collection with both dense and sparse vectors
client.create_collection(
    collection_name="rag_chunks",
    vectors_config={
        "dense": VectorParams(size=1024, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams()
    }
)

# Insert a chunk with both vectors
client.upsert(
    collection_name="rag_chunks",
    points=[
        PointStruct(
            id="chunk_001",
            vector={
                "dense": dense_vector,  # list of 1024 floats
                "sparse": SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
                )
            },
            payload={
                "text": "Original chunk text...",
                "context_header": "[Klage | 24/014 | VG Düsseldorf]",
                "section_type": "legal_argument",
                "statute": "AufenthG",
                "paragraph": "§ 60",
                "applicant_origin": "Afghanistan",
                "provenance": ["doc_001", "doc_042"]
            }
        )
    ]
)

# Hybrid search
results = client.query_points(
    collection_name="rag_chunks",
    prefetch=[
        # Dense search
        {"query": dense_query, "using": "dense", "limit": 50},
        # Sparse search
        {"query": SparseVector(indices=sparse_indices, values=sparse_values),
         "using": "sparse", "limit": 50}
    ],
    query={"fusion": "rrf"},  # RRF fusion built-in!
    limit=30,
    with_payload=True
)
```

---

### Cross-Encoder Reranker

After RRF fusion, rerank with a cross-encoder for precision.

**Model:** `BAAI/bge-reranker-v2-m3` (~500MB VRAM)

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")

def rerank(query: str, chunks: list[dict], top_k: int = 8) -> list[dict]:
    """Rerank chunks using cross-encoder"""
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    # Sort by reranker score
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]
```

---

### Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE (SOTA 2026)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐                                                       │
│  │  User Query  │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────┐                              │
│  │  1. EMBED QUERY (BGE-M3 via TEI)     │  ~50ms                       │
│  │     → dense vector (1024)            │                              │
│  │     → sparse vector (lexical)        │                              │
│  └──────────────┬───────────────────────┘                              │
│                 │                                                       │
│         ┌───────┴───────┐                                              │
│         │               │                                              │
│         ▼               ▼                                              │
│  ┌─────────────┐ ┌─────────────┐                                       │
│  │ Dense Search│ │Sparse Search│  Qdrant (CPU)                        │
│  │   Top 50    │ │   Top 50    │                                       │
│  └──────┬──────┘ └──────┬──────┘                                       │
│         │               │                                              │
│         └───────┬───────┘                                              │
│                 ▼                                                       │
│  ┌──────────────────────────────────────┐                              │
│  │  2. RRF FUSION                        │  ~5ms                       │
│  │     → Combined Top 30                 │                              │
│  └──────────────┬───────────────────────┘                              │
│                 │                                                       │
│                 ▼                                                       │
│  ┌──────────────────────────────────────┐                              │
│  │  3. RERANK (bge-reranker-v2-m3)      │  ~200ms                      │
│  │     → Top 8 most relevant            │                              │
│  └──────────────┬───────────────────────┘                              │
│                 │                                                       │
│                 ▼                                                       │
│  ┌──────────────────────────────────────┐                              │
│  │  4. CONTEXT ASSEMBLY                  │                              │
│  │     → Format chunks with headers     │                              │
│  │     → Add metadata context           │                              │
│  └──────────────┬───────────────────────┘                              │
│                 │                                                       │
│                 ▼                                                       │
│  ┌──────────────────────────────────────┐                              │
│  │  5. LLM GENERATION (Qwen 14B)        │  ~5-10s                      │
│  │     → Answer with citations          │                              │
│  └──────────────────────────────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Updated Implementation Priority

#### Phase 1: Infrastructure (Day 1-2)
- [ ] Deploy TEI with BGE-M3 (Docker)
- [ ] Deploy Qdrant (Docker)
- [ ] Create hybrid collection schema
- [ ] Test embedding + insertion for 10 sample chunks

#### Phase 2: Ingestion Pipeline (Day 3-5)
- [ ] Implement chunker with context headers
- [ ] Integrate anonymization service
- [ ] Batch embed with BGE-M3 (dense + sparse)
- [ ] Insert into Qdrant
- [ ] Process 100 test documents

#### Phase 3: Retrieval API (Day 6-8)
- [ ] Implement hybrid search endpoint
- [ ] Add RRF fusion (or use Qdrant built-in)
- [ ] Integrate cross-encoder reranker
- [ ] Test retrieval quality on sample queries

#### Phase 4: Full Pipeline (Day 9-12)
- [ ] Process full corpus (~3,122 docs → ~9k chunks)
- [ ] Build evaluation harness (synthetic Q&A)
- [ ] Integrate with main Rechtmaschine app
- [ ] VRAM orchestration with service_manager.py

---

### Service Manager Integration

Add BGE-M3 and reranker to the existing service manager queue:

```python
SERVICES = {
    "ocr": {...},
    "anon": {...},
    "embedder": {
        "port": 8085,
        "url": "http://localhost:8085",
        "process_name": "text-embeddings-inference",
        "docker_container": "bge-m3",
        "vram": 1.2,  # GB
        "load_time": 0.5  # seconds (fast!)
    },
    "reranker": {
        "port": 8086,
        "url": "http://localhost:8086",
        "process_name": "reranker_service.py",
        "vram": 0.5,
        "load_time": 1
    }
}
```

---

## References

- [ColBERT v2](https://github.com/stanford-futuredata/ColBERT)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [BGE Reranker v2](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [RAPTOR](https://arxiv.org/abs/2401.18059)
- [HyDE](https://arxiv.org/abs/2212.10496)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference)
- [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/hybrid-queries/)

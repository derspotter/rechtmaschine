# Specification: Controlled-Vocabulary Schlagwörter for Hybrid RAG Search

## Overview

**Goal:** Every chunk in every RAG collection (`jurisprudence`, `kanzlei`, future sources) carries canonical Schlagwörter (keywords), Herkunftsländer (origin countries), and Normen (statutes), written into both `context_header` (indexed by the existing dense+sparse hybrid channels) and `metadata` (stored for future hard faceting).

**v1 scope:** Tag all existing chunks, bake the tagging into forward ingest, verify retrieval quality. No query-side facet logic in v1 — the existing sparse and dense channels index the tags automatically once they're in `context_header`.

**Rationale:** Hybrid search = dense (semantic) + sparse (lexical). Sparse is already a full-text index; once tags live in `context_header`, both channels pick them up automatically. No third channel needed.

---

## 1. The Controlled Vocabulary

### 1.1 Source and Structure

**File:** `rag/vocabulary/schlagworte.json` (repo root)

**Build process:** Aggregate from the 473 existing `RechtsprechungEntry` rows:
- Deduplicate and normalize asyl.net's curated `schlagworte` field (already harvested in the prior session).
- Include herkunftsländer from `applicant_origin` in case memory + existing chunk metadata.
- Include canonical statutes from `Normen` (asyl.net) + the statutes in `legal_texts/` extraction (AsylG § numbers, etc.).
- Frequency-rank; drop singletons (appear in <2 entries).
- Alias map: `Wehrdienstverweigerung → Wehrdienstentziehung`, `syrisch → Syrien`, etc.

**Schema:**

```json
{
  "version": "2026-06-14",
  "meta": {
    "description": "Controlled vocabulary for RAG Schlagwörter across all collections",
    "canonical_count": {
      "themen": 87,
      "herkunftslaender": 42,
      "normen": 256
    }
  },
  "themen": [
    {
      "canonical": "Wehrdienstentziehung",
      "aliases": ["Wehrdienstverweigerung", "Kriegsdienstverweigerung"],
      "frequency": 34
    },
    {
      "canonical": "internationaler Schutz in EU-Staat",
      "aliases": ["Sekundärmigration"],
      "frequency": 75
    }
  ],
  "herkunftslaender": [
    {
      "canonical": "Syrien",
      "aliases": ["Arabische Republik Syrien", "syrisch"],
      "iso_3166_1_alpha_3": "SYR",
      "frequency": 118
    },
    {
      "canonical": "Afghanistan",
      "aliases": ["afghanisch"],
      "iso_3166_1_alpha_3": "AFG",
      "frequency": 156
    }
  ],
  "normen": [
    {
      "canonical": "§ 60 Abs. 7 AufenthG",
      "aliases": ["§ 60 VII AufenthG", "§60(7) AufenthG"],
      "statute_id": "AufenthG_60_7",
      "frequency": 112
    }
  ]
}
```

### 1.2 Normalization Functions

**File:** `app/rag_vocabulary.py` (new)

Functions used by both jurisprudence and kanzlei taggers to map raw tags → canonical:

```python
# app/rag_vocabulary.py (skeleton)

from typing import Set
import json
from pathlib import Path

class SchlagworteVocabulary:
    def __init__(self, vocab_json_path: Path = Path("rag/vocabulary/schlagworte.json")):
        with open(vocab_json_path) as f:
            self.vocab = json.load(f)
        self._build_alias_maps()
    
    def _build_alias_maps(self):
        """Invert: alias → canonical."""
        self.themen_map = {}
        self.land_map = {}
        self.normen_map = {}
        for item in self.vocab['themen']:
            self.themen_map[item['canonical'].lower()] = item['canonical']
            for alias in item.get('aliases', []):
                self.themen_map[alias.lower()] = item['canonical']
        # Same for herkunftslaender, normen
    
    def normalize_themen(self, raw_tags: list[str]) -> Set[str]:
        """Map a list of raw tags → canonical Schlagwörter (deduped)."""
        result = set()
        for tag in raw_tags:
            normalized = self.themen_map.get(tag.lower())
            if normalized:
                result.add(normalized)
        return result
    
    def normalize_country(self, country_str: str) -> str | None:
        """Single country → canonical or None."""
        return self.land_map.get(country_str.lower())
    
    def normalize_normen(self, raw_norm: str) -> str | None:
        """Statute string → canonical form or None."""
        return self.normen_map.get(raw_norm.lower())
```

---

## 2. Re-Tagging Mechanism

### 2.1 RAG API Scroll Endpoint

**Purpose:** Export chunks page-by-page so both retag passes can iterate without holding the entire store in RAM.

**Endpoint:** `GET /v1/rag/chunks/scroll` (RAG API, debian)

**Parameters:**
- `collection` (str): collection name
- `limit` (int, default 100): page size
- `offset` (int, default 0): starting position

**Response:**
```json
{
  "collection": "kanzlei",
  "limit": 100,
  "offset": 0,
  "total": 10660,
  "chunks": [
    {
      "chunk_id": "nc-abc123-0",
      "rechtsprechung_entry_id": null,
      "text": "...",
      "context_header": "...",
      "metadata": {...},
      "created_at": "2026-06-13T..."
    }
  ]
}
```

**Implementation:** Simple paginated `SELECT` from `rag_chunks` ordered by `created_at`, no re-embedding trigger.

### 2.2 Retag → Re-Upsert Pattern

Both jurisprudence and kanzlei follow the same flow:

1. **Scroll** chunks via `/scroll`.
2. **Enrich** with new tags (jurisprudence: normalize entry.schlagworte; kanzlei: call Qwen doc-tagger).
3. **Rewrite** `context_header` and `metadata` with canonical tags.
4. **Re-upsert** via `/v1/rag/chunks/upsert` with **`dense: null`** (signals "re-embed the new context_header").

This is **idempotent by `chunk_id`** — re-running the pass produces the same result.

---

## 3. Jurisprudence Re-Tagging (Immediate)

**Scope:** 473 entries × ~10,257 chunks

**Data source:** Already harvested in `RechtsprechungEntry.schlagworte`, `.normen`, `.leitsatz`. The entry is joined to chunks via `metadata.rechtsprechung_entry_id`.

**Process:**

```
for each page of chunks from /scroll:
  for each chunk:
    entry = lookup RechtsprechungEntry by metadata.rechtsprechung_entry_id
    canonical_themen = vocab.normalize_themen(entry.schlagworte)
    canonical_normen = {vocab.normalize_normen(n) for n in entry.normen}
    canonical_country = vocab.normalize_country(entry.country)  # if present
    
    # Rebuild context_header
    header_parts = [
      existing_case_ref_or_courtname,
      f"Schlagwörter: {', '.join(sorted(canonical_themen))}",
      f"Normen: {', '.join(sorted(canonical_normen))}",
    ]
    new_context_header = " | ".join(header_parts)
    
    # Metadata
    new_metadata = chunk.metadata.copy()
    new_metadata['schlagworte'] = list(canonical_themen)
    new_metadata['normen'] = list(canonical_normen)
    if canonical_country:
      new_metadata['applicant_origin'] = canonical_country
    
    # Re-upsert (dense=null forces re-embedding)
    upsert(chunk_id, text, new_context_header, new_metadata, dense=None)
```

**Script:** `rag/retag_jurisprudence.py` (new), runs via `docker exec rechtmaschine-app python rag/retag_jurisprudence.py --collection jurisprudence`.

**Cost:** One `/scroll` pass + one `/upsert` per chunk (~10k requests) + ~10k TEI re-embeds. Batch-friendly, runs on debian, no desktop GPU needed.

---

## 4. Kanzlei Tagging (The Main Work)

### 4.1 Document-Level Tagging

**Scope:** ~3,000 documents → re-tag their ~10,660 chunks

**Challenge:** Anonymized client documents have no prior tags and no external source. Need LLM tagging, but **must never send to cloud** — kanzlei docs carry client information.

**Solution:** Desktop-Qwen local tagging via `service_manager.py`'s existing `/anonymize` / `/prompt` endpoints (role `all`).

### 4.2 Qwen Tagging Prompt

A single call per **document** (not per chunk), on the document's already-anonymized concatenated text (from `ingest_runner.py`):

```
Du bist ein spezialisierter Legal-Assistant für deutsches Asylrecht.
Analysiere das folgende anonymisierte Schriftsatz-Dokument und antworte mit strukturiertem JSON.
Extrahiere nur aus dem kontrollierten Vokabular (siehe unten):

Kontrolliertes Vokabular (Themen):
- Wehrdienstentziehung
- internationaler Schutz in EU-Staat
- offensichtlich unbegründet
- Abschiebungsverbot
- Dublinverfahren
... (full list from schlagworte.json)

Herkunftsländer:
- Afghanistan
- Syrien
- Irak
... (full list from herkunftslaender)

Relevant Normen:
- § 60 Abs. 7 AufenthG
- § 3 AsylG
- Art. 3 EMRK
... (relevant sublist from normen)

Antworte mit:
{
  "themen": ["theme1", "theme2"],
  "applicant_origin": "country",
  "normen": ["statute1", "statute2"],
  "section_type": "Klage | Beschwerde | Schriftsatz | Stellungnahme | other",
  "confidence": 0.85
}

Dokument:
<anonymized_full_text>
```

**Structure:** `doc_sha256 → {themen, applicant_origin, normen, section_type, confidence}` (one response per document, cached so re-runs are fast).

### 4.3 Chunk-Level Re-Upsert

Once tags are computed for a document, apply them to **all chunks of that document** (identified by `metadata.sha256`):

```
for each page of chunks from /scroll (collection: kanzlei):
  for each chunk:
    doc_sha = chunk.metadata.sha256
    if doc_sha not in doc_tags_cache:
      doc_tags_cache[doc_sha] = call_qwen_for_doc(doc_sha)
    
    tags = doc_tags_cache[doc_sha]
    
    # Rebuild context_header
    new_context_header = f"{chunk.context_header} | Schlagwörter: {', '.join(tags['themen'])} | Normen: ..."
    
    # Metadata
    new_metadata = chunk.metadata.copy()
    new_metadata['schlagworte'] = tags['themen']
    new_metadata['applicant_origin'] = tags['applicant_origin']
    new_metadata['normen'] = tags['normen']
    new_metadata['section_type'] = tags['section_type']
    
    upsert(chunk_id, text, new_context_header, new_metadata, dense=None)
```

**Script:** `rag/retag_kanzlei.py` (new). Phases:
1. Scroll/group chunks by doc_sha, collect list of unique docs.
2. For each doc: `curl localhost:8004/prompt -d {structured prompt}` → cache the response.
3. Scroll again, re-tag + re-upsert per chunk.

**Cost:** ~3,000 Qwen calls (one per doc, ~10s ea via desktop GPU = ~8–10 hours), then ~10,660 TEI re-embeds. Runs on the server (via Tailscale to desktop) with kanzlei already anonymized.

---

## 5. Forward Integration (Live Ingest)

### 5.1 Jurisprudence (Live)

**File:** `rag/jurisprudence_ingest.py` (already has asyl.net metadata harvest)

**Change:** At upsert time, normalize harvested `schlagworte/normen` through the vocab before writing.

```python
# In main_async, around persist_entry() call:

from app.rag_vocabulary import SchlagworteVocabulary
vocab = SchlagworteVocabulary()

canonical_themen = vocab.normalize_themen(extracted_entry.schlagworte)
canonical_normen = {vocab.normalize_normen(n) for n in extracted_entry.normen}

# Pass to chunk builder:
build_rag_block(..., schlagworte=canonical_themen, normen=canonical_normen, ...)
```

### 5.2 Kanzlei (Live)

**File:** `rag/ingest_runner.py` (anonymizes, chunks, upsets)

**Change:** After anonymization completes per document, call desktop Qwen for tags before chunking:

```python
# Around line 455 (chunk loop):

# After anonymize_text() returns:
doc_tags = await call_qwen_tagging(anonymized_text)  # Desktop service-manager

# In chunk loop:
for chunk_text in chunks:
  # Rebuild header to include doc-level tags:
  context_header = f"{chunk.context_header} | Schlagwörter: {', '.join(doc_tags['themen'])}"
  
  # Metadata:
  chunk_metadata['schlagworte'] = doc_tags['themen']
  chunk_metadata['applicant_origin'] = doc_tags['applicant_origin']
  chunk_metadata['normen'] = doc_tags['normen']
  
  # Upsert to debian:
  upsert(context_header=context_header, metadata=chunk_metadata, ...)
```

---

## 6. Retrieval (No v1 Changes)

### 6.1 How Tags Flow Through Hybrid Search

**Dense channel:** Embedding input includes `context_header` (which now has `Schlagwörter: ...`). The tag text becomes part of the semantic vector.

**Sparse channel:** `search_text = to_tsvector('german', context_header || ' ' || text)`. Tags are full-text indexed and matched by German stemming.

**RRF fusion + rerank:** Both channels contribute scores independently; cross-encoder reranks the fused result.

**Result:** A query mentioning "Griechenland" or "Wehrdienstentziehung" matches chunks tagged that way via the sparse channel, and the semantic channel also sees the meaning. No special v1 query logic needed.

### 6.2 Future (Deferred, YAGNI)

- `RagFilters.schlagworte: list` + hard/soft facet filtering.
- Query-side country/topic extraction from case memory → automatic facet boost.
- UI facet pickers.

These are all cheap because `metadata` is already stored; they can be added on demand.

---

## 7. Build Sequence

### Phase 1: Setup (2–3 hours)
1. Aggregate `rag/vocabulary/schlagworte.json` from 473 entries + `legal_texts/` statutes.
2. Write `app/rag_vocabulary.py` normalizer + unit tests.
3. Add `GET /v1/rag/chunks/scroll` to RAG API.
4. Test with one jurisprudence chunk to confirm scroll + upsert round-trip works.

### Phase 2: Jurisprudence Retag (2–3 hours)
5. Write `rag/retag_jurisprudence.py` (scroll → normalize → upsert).
6. Dry-run on 10 chunks, verify context_header and metadata look right.
7. Run full pass (~10k chunks, ~10k TEI re-embeds, ~1 hour on debian GPU).

### Phase 3: Kanzlei Tagging (8–12 hours)
8. Write `rag/retag_kanzlei.py` phase 1 (collect docs).
9. Wire `call_qwen_tagging()` to desktop service-manager, test on 5 docs.
10. Run phase 2 (Qwen tag ~3,000 docs, desktop GPU — 8–10 hours).
11. Run phase 3 (retag + re-upsert ~10,660 chunks, ~1–2 hours).

### Phase 4: Live Integration (1–2 hours)
12. Bake tag normalization into `jurisprudence_ingest.py`.
13. Wire Qwen tagging into `ingest_runner.py` (before chunking).
14. Verify birth-tagged on next jurisprudence fetch + kanzlei ingest.

### Phase 5: Verification (2–3 hours)
15. Leak-safety check: scan a few anonymized kanzlei chunks for client PII names (should be all `[PERSON]`).
16. Retrieval benchmark: re-run the 18-query suite from the prior session, confirm no regression (tags should only help).
17. Spot-check: manually verify facets on a few entries (country + themen combo).

---

## 8. Success Criteria

- **All 473 jurisprudence chunks** have canonical `schlagworte`, `normen`, `applicant_origin` in both `context_header` and `metadata`.
- **All ~10,660 kanzlei chunks** carry document-level `schlagworte` (section_type, country, normen) in both layers.
- **Retrieval quality** unchanged or improved (benchmark pass).
- **Leak-safety:** zero client names in anonymized kanzlei tags (controlled categories only).
- **Forward:** new jurisprudence ingests + kanzlei docs are automatically born-tagged.
- **Ready for v2:** faceted filtering (deferred) can be added without re-ingesting.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Qwen tagging on kanzlei mislabels client info as a tag | Tags are limited to controlled vocab; no free-text extraction. Confidence threshold + spot-check samples. |
| Re-embedding 10k+ chunks overwhelms debian GPU | Batch the scroll/upsert in groups of 100, sleep between batches. TEI usually handles 10–50 concurrent. |
| Alias map omits a real tag variant | Start with asyl.net + `legal_texts/` as seeds; human review the vocab JSON before committing. |
| Schema change breaks existing queries | Only `context_header` and `metadata` change; chunk_id, text, embeddings unchanged. Old queries still work (tags just add signal). |

---

## 10. Implementation Notes

- **Vocab JSON is versioned** (`"version": "2026-06-14"`). Retag scripts check version and warn if stale.
- **Retag scripts are idempotent** — safe to re-run (same `chunk_id` + new tags → same result).
- **Desktop-Qwen calls must handle rate limits** — batch in 1-at-a-time if needed, with exponential backoff.
- **All retag scripts log progress and checkpoints** so failures can resume cleanly.
- **Migration:** mark existing chunks with a `tagged_vocab_version: null` to distinguish old-style tags from controlled tags.

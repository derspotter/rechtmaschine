# Controlled-vocabulary Schlagwörter as metadata across all RAG collections

Date: 2026-06-14
Status: Design approved, pending spec review

## Problem

The RAG store serves two populated collections — `kanzlei` (~10,660 chunks from
the firm's own anonymized filings) and `jurisprudence` (~10,257 chunks from 473
asyl.net decisions). Hybrid retrieval (dense bge-m3 + German full-text sparse,
RRF-fused, cross-encoder reranked) works on chunk **text** only.

We harvested asyl.net's curated Schlagwörter/Normen/Leitsatz onto the 473
`RechtsprechungEntry` rows, but those tags do **not** participate in retrieval:
they live on the DB entries, not in the chunks in the store, and nothing on the
query side engages them. The `kanzlei` collection has no topical tags at all.

Goal: give **every chunk in every collection** canonical Schlagwörter +
Herkunftsland + Normen as real metadata, so hybrid search benefits from them and
the same concept is the same token across sources (a court decision and one of
our own Schriftsätze about Greece/Dublin must match on the same `griechenland` /
`dublin` tokens).

## Key technical fact that shapes the design

Both retrieval signals already read `context_header`, not just `text`:

- **Sparse:** `search_text TSVECTOR GENERATED ALWAYS AS
  to_tsvector('german', context_header || ' ' || text)` (`rag/db/init.sql:12`).
- **Dense:** embedding input is `context_header + "\n\n" + text`
  (`rag/api/main.py:180`).
- **Hard faceting already exists:** `_filter_sql` filters on
  `metadata->>'statute'|'paragraph'|'applicant_origin'|'court'`, date ranges, and
  `metadata->'citations' ?| array` (`rag/api/main.py:184`).

Therefore: **tags written into `context_header` flow into both the embedding and
the German full-text index for free** — no schema change. Tags written into
`metadata` enable future hard/soft faceting via the existing `RagFilters`.

This is a data + wiring project, not a store rebuild.

## Scope decisions (made during brainstorming)

1. **Vocabulary: controlled and shared.** One canonical taxonomy across all
   collections, seeded from asyl.net's curated terms. Not free-form, not
   per-collection.
2. **Kanzlei granularity: document-level + section_type.** One LLM tagging call
   per document (~3,000), propagated to all of that document's chunks. (Chunk
   `section_type` is a cheap structural add, not the topical tagging unit.)
3. **Query-side retrieval: soft only, no query extraction (v1).** Tags go into
   `context_header` (auto-indexed by both existing hybrid channels) and
   `metadata` (stored for future faceting). No third RRF channel, no
   query→facet mapping, no hard filters in v1. Rationale: the existing sparse
   (German full-text) channel already does keyword matching on tags placed in
   `context_header`, and the dense channel covers their meaning — a separate
   tag-overlap channel would be largely redundant. The only genuine gap is
   query-side vocabulary normalization (controlled term not literally in the
   query, e.g. `Wehrdienstentziehung` vs "Wehrdienstverweigerer"), which the
   dense channel partly covers; defer building for it until benchmarks justify.

## Design

### 1. The controlled vocabulary

A single canonical taxonomy committed to the repo:
`rag/vocabulary/schlagworte.json`, with three facet types:

- **themen** — seeded from the asyl.net Schlagwörter already on the 473
  `RechtsprechungEntry` rows, deduplicated, frequency-ranked, long-tail
  singletons dropped, with an **alias map** mapping synonyms/variants to the
  canonical form (`Wehrdienstverweigerung → Wehrdienstentziehung`).
- **herkunftslaender** — canonical country names + aliases
  (`Arabische Republik Syrien → Syrien`).
- **normen** — canonical §§ from asyl.net Normen plus the `legal_texts/` corpus,
  normalized (`§ 60 Abs. 7 AufenthG`, `Art. 3 EMRK`).

Built once by an aggregation script that reads the 473 entries' existing
`schlagworte`/`normen`/`country`, ranks by frequency, and emits a
human-reviewable JSON (canonical list + alias map).

A small module `app/rag_vocabulary.py` loads the vocab and exposes:

- `normalize_themen(raw: list[str]) -> list[str]`
- `normalize_country(raw: str | None) -> str | None`
- `normalize_normen(raw: list[str]) -> list[str]`

Both taggers (jurisprudence + kanzlei) call the same normalizer, so all
collections land on the same tokens.

### 2. One uniform retag mechanism

Add a minimal paginated export endpoint to the RAG API:
`POST /v1/rag/chunks/scroll` (cursor over `(collection, chunk_id)`, returns
`chunk_id`, `text`, `context_header`, `metadata`, `provenance`). Authenticated
with the existing `X-API-Key`.

Both retag passes then follow the same shape and require **no re-extraction,
re-download, or re-anonymization**:

1. Scroll all chunks of a collection from the store.
2. Compute / look up canonical tags for each chunk.
3. Rewrite `context_header` (append the tag line) and `metadata`.
4. Re-upsert by the same `chunk_id` (idempotent `ON CONFLICT`), sending **empty
   `dense`** so the upsert endpoint re-embeds with the new `context_header`.

#### Jurisprudence retag (data exists — cheapest)

Each chunk's `metadata.rechtsprechung_entry_id` joins to its
`RechtsprechungEntry`. Normalize the entry's `schlagworte/normen/country`
through the vocab, rewrite header+metadata, re-upsert. No LLM, no network.

#### Kanzlei retag (the LLM work)

Group scrolled chunks by document (`chunk_id` prefix `nc-{sha16}-*`). For each
document, run **one desktop-Qwen tagging call** on the document's
already-anonymized chunk text (reconstructed from its chunks, or from the stored
`{sha16}.txt`), constrained to emit only canonical vocab terms → Schlagwörter +
country + §§. Apply to all of that document's chunks, re-upsert.

- Runs on the import host over Tailscale to desktop Qwen (`:8004`).
- Operates on anonymized text only; never sends client text to the cloud.
- Tags are controlled categories and cannot reintroduce PII.

### 3. What gets written to each chunk

- `context_header` gains a compact tag line, e.g.:
  `... | Schlagwörter: griechenland, internationaler schutz in eu-staat,
  offensichtlich unbegründet | Normen: § 29 AsylG, Art. 3 EMRK`
- `metadata` gains, aligned to existing `RagFilters` keys for free future
  faceting:
  - `schlagworte: [canonical themen...]`
  - `applicant_origin: <canonical country>` (jurisprudence already stores
    `country`; unify under `applicant_origin`)
  - `citations: [canonical §§...]`

### 4. Retrieval changes

**None required for v1.** Tags ride in `context_header`, so the existing
dense+sparse RRF+rerank path uses them. The jurisprudence pack
(`app/endpoints/jurisprudence.py`) and `app/rag_context.py` already surface
chunk metadata, so they will display the richer tags without changes.

### 5. Born-tagged going forward

Bake tag-writing into both live ingest paths so new documents are tagged at
ingest, not by a later retag:

- `rag/ingest_runner.py` (`context_header` at `:455`, `metadata` at `:457`):
  add the kanzlei Qwen doc-tag step + write tags into header/metadata.
- `app/jurisprudence_ingest.py` (`context_header` at `:386`, `metadata` at
  `:400`): normalize the asyl.net tags through the vocab and write them into the
  header (metadata already carries `issue_tags`; add normalized `schlagworte` +
  align `country`/`citations`).

Future wiki collections (aufenthaltswiki/dokuwiki) inherit the same contract;
being public, their tagging may use a cloud model.

## Build sequence

1. Aggregate the 473 entries → `rag/vocabulary/schlagworte.json`; add
   `app/rag_vocabulary.py` normalizer.
2. Add `POST /v1/rag/chunks/scroll` to `rag/api/main.py`.
3. Jurisprudence retag pass + bake tags into the live jurisprudence ingest.
4. Kanzlei Qwen doc-tagger + doc-level retag pass (~3,000 calls) + bake into the
   live `ingest_runner.py`.
5. Verify: leak-safe check on the new headers (tags are categories, no PII),
   re-run the retrieval benchmark for lift, spot-check facets. (~21k chunks
   re-embed through TEI bge-m3 on debian — a batch job, runtime noted.)

## Deferred (YAGNI for now)

- Third RRF tag-overlap channel.
- `RagFilters.schlagworte` field + hard/soft metadata filters.
- Query-side facet extraction (query → controlled terms) for the synonym-
  normalization gap. Add only if the benchmark shows it hurts.
- UI facet pickers.

The `metadata` stored in v1 makes all of these cheap to add later.

## Safety / constraints

- Kanzlei tagging runs only on already-anonymized text via desktop Qwen; no raw
  client text leaves the firm machines; controlled-vocab tags cannot reintroduce
  PII.
- Jurisprudence is court-published and pre-redacted; cloud tagging already in
  use there is unchanged.
- Re-upsert is idempotent by `chunk_id`; reruns are safe.

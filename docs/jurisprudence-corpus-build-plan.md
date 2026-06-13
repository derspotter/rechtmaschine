# Jurisprudence Corpus & Refresh — Build Plan

Concrete build spec for a standing, tagged German asylum/migration case-law
corpus with periodic refresh and hybrid retrieval. Turns §4 ("Jurisprudence
Corpus and Packs") of `agent-memory-and-case-law-plan.md` into something
implementable. Decided 2026-06-13: cover all five sources; write this spec
before code.

## Relationship to what already exists

Reuse, don't rebuild:

- **`RechtsprechungEntry`** (models.py) — already the tagging layer: `country`,
  `tags`, `court`, `court_level`, `decision_date`, `aktenzeichen`, `outcome`,
  `key_facts`, `key_holdings`, `argument_patterns`, `citations`, `summary`,
  `confidence`, `is_active`. Keep it as the per-decision metadata record.
- **Debian hybrid store** (`rag/docker-compose.debian.yml`, pgvector + TEI +
  rerank, API `:8090`) — add a **`jurisprudence` collection** beside `kanzlei`.
  Same upsert/retrieve API; tags become chunk metadata for filtered hybrid search.
- **`app/rag_context.py`** — the prompt-injection pattern. Jurisprudence gets a
  sibling block (see Retrieval).
- **`app/endpoints/research/asylnet.py`** — existing asyl.net scraper; the
  refresh job reuses its fetch/parse rather than starting over.
- Tag extraction already exists (research auto-extracts structured fields via
  Gemini); reuse it to populate `RechtsprechungEntry` from decision text.

## Key simplification: no anonymization

Court-published decisions are already redacted by the courts before publication
(names, birthdates removed). Unlike the `kanzlei` corpus, the jurisprudence
pipeline **skips the Qwen anonymization step**. Spot-check a sample per source
during build (some VG decisions leave partial initials); add a light scrub if
needed, but do not route through desktop Qwen. This makes jurisprudence a
public-source pipeline (like the planned DokuWiki source): fetch → tag → chunk
→ embed → upsert, no GPU anonymization, no desktop dependency.

## Where it runs

- **Fetch + parse + tag + chunk**: server (`app/`) — it has the research
  scrapers, Gemini keys, and network access to the sources. No GPU needed.
- **Embed + store + retrieve**: debian, via the existing `/v1/rag/chunks/upsert`
  and `/v1/rag/retrieve` (TEI embeds server-side on debian). The server POSTs
  tagged chunks to debian; no embedding on the server.
- **Schedule**: a periodic job (systemd timer on the server, or a job-table
  entry polled by job-worker). Cadence below.

## Data model

1. `RechtsprechungEntry` — unchanged; one row per decision (canonical metadata).
   Add if missing: `source_type` (asylnet|nrwe|rii|edal|openjur), `source_url`,
   `source_ref` (M-number / ECLI / court doc id), `last_fetched_at`,
   `content_sha256`, `instance_weight` (BVerfG/BVerwG/EuGH/EGMR > OVG > VG).
2. `jurisprudence_sources` (new, small) — refresh watermark per source:
   `source_type` (pk), `newest_decision_date`, `last_run_at`, `last_seen_ids`
   (for TOC-diff sources), `notes`.
3. Chunks live only in the debian `jurisprudence` collection (not a Postgres
   table on the server). Chunk metadata mirrors the retrieval-relevant tags:
   `country`, `court`, `court_level`, `outcome`, `decision_date`, `statutes`
   (e.g. `§ 60 Abs. 7 AufenthG`), `issue_tags`, `aktenzeichen`, `source_type`,
   `rechtsprechung_entry_id`, `chunk_index`. Provenance: `source_url`,
   `source_ref`.

## Sources (all five) — per-source adapter spec

Each adapter implements: `fetch_since(watermark) -> list[RawDecision]` with
`{source_url, source_ref, court, decision_date, aktenzeichen, raw_text|html}`.

| Source | Scope | Incremental strategy | Notes / ToS |
|---|---|---|---|
| **asyl.net** Entscheidungsdatenbank | Curated migration, all instances incl. EuGH/EGMR, editorial Leitsätze (M-numbers) | Date-sorted listing; take entries with publication/decision date > watermark | Reuse `asylnet.py`. Public; rate-limit, set UA, attribute M-number + URL. |
| **NRWE** nrwe.justiz.nrw.de | NRW courts (OVG Münster, VG Düsseldorf/Köln/…) — firm's jurisdiction | Search with date-range filter, page until watermark | Public state DB; polite scrape, store doc URL. |
| **rechtsprechung-im-internet.de** | Federal (BVerwG, BVerfG, …), binding top-instance | **`rii-toc.xml`** full index → diff against `last_seen_ids`; fetch new ECLIs | Official, explicit reuse permission. Cleanest/most stable. |
| **EDAL** asylumlawdatabase.eu | Curated EuGH/EGMR asylum (Dublin, GEAS, Art. 3) | Date-sorted listing; incremental by date | Public, attribute. English summaries + source links. |
| **openjur.de** | Broad free German case law (breadth/fallback) | Date-sorted listing; incremental by date | Public; higher volume, less curation → heavier dedup/tag load. Lowest priority of the five. |

Filter at fetch: migration/asylum relevance (statute hits AufenthG/AsylG/
Dublin/GEAS, or source already migration-scoped like asyl.net/EDAL). openjur
especially needs a relevance gate to avoid pulling unrelated law.

## Pipeline (per refresh run)

1. For each source: `fetch_since(watermark)`.
2. **Cross-source dedup**: a decision often appears in several sources
   (asyl.net republishes NRWE/BVerwG). Dedup key = normalized
   `(court, aktenzeichen, decision_date)`; fall back to `content_sha256`. Keep
   the most-curated copy (asyl.net/EDAL editorial > raw court text); record the
   other source URLs as additional provenance.
3. **Relevance gate** (drop non-migration).
4. **Tag/extract** → populate/refresh `RechtsprechungEntry` (Gemini structured
   extraction: country, issue_tags, holdings, argument_patterns, statutes,
   outcome, court_level). Set `instance_weight`.
5. **Chunk** the decision text (reuse the kanzlei chunker; case-law structure:
   keep Leitsätze/Tenor/Gründe boundaries where detectable). Prepend a context
   header: `Rechtsprechung | <court> | <decision_date> | <country> | <statutes>`.
6. **Embed + upsert** into the `jurisprudence` collection (debian), chunk ids
   deterministic: `juris-<entry_id>-<idx>`.
7. **Advance watermark** (`newest_decision_date`, `last_seen_ids`, `last_run_at`).

Idempotent: deterministic chunk ids + dedup key mean re-runs and overlapping
date windows don't duplicate. A decision that gets superseded/updated re-tags
the same `RechtsprechungEntry` and re-upserts the same chunk ids.

## Refresh cadence

- Backfill once (last ~3–5 years, matching the corpus recency stance — older
  asylum doctrine is pre-GEAS and lower value).
- Then **weekly** incremental (case law moves slower than the news cycle; daily
  is overkill). rii-toc.xml diff + asyl.net/NRWE/EDAL/openjur date windows since
  `last_run_at`. Run off-hours.
- Freshness watermark per source makes a missed run self-healing (next run
  covers the gap).

## Retrieval & jurisprudence packs

Two layers, mirroring §4:

- **Corpus retrieval** (base): hybrid search over the `jurisprudence`
  collection with tag filters (country, statute, issue_tags, court_level) +
  **freshness/instance ranking** — boost recent `decision_date` and higher
  `instance_weight` so a 2025 BVerwG ranks above a 2019 VG on equal relevance.
- **Jurisprudence pack** (compact, cached): per case fingerprint
  (`country + issue_tags`, e.g. `afghanistan + alleinstehende_frau`), a stored
  bundle of top recent holdings + argument patterns + source refs, with
  `last_refreshed_at` / `newest_decision_date` / `refresh_after_days`. The pack
  is what gets injected; the corpus is what builds/refreshes it.

**Prompt integration** (`rag_context.py` pattern): add a parallel block, e.g.
`EINSCHLÄGIGE RECHTSPRECHUNG (aktuell)`, beside the kanzlei-precedent block.
Query = `body.user_prompt`/`body.query` enriched with case `country`+issue tags.
Gate behind its own flag (`JURIS_RETRIEVAL_ENABLED`). Mind prompt budget: a
generation prompt would then carry case-memory + kanzlei-precedent +
jurisprudence + docs — keep each block small (top 3–5) and consider letting the
user toggle which layers are active per generation.

Distinction to keep clear in the prompt: kanzlei-precedent = *how we argued*
(firm's own filings, anonymized); jurisprudence = *what the courts held*
(binding/persuasive authority, cite-able). The jurisprudence block may be cited
in drafts; the kanzlei-precedent block may not.

## Build sequence

1. Schema: add the `RechtsprechungEntry` columns + `jurisprudence_sources`
   table (+ a `MIGRATIONS` entry).
2. One source vertical slice end-to-end: **asyl.net** (scraper exists) →
   tag → chunk → upsert `jurisprudence` → retrieve. Verify tagging + retrieval
   quality on real decisions; spot-check redaction.
3. Add NRWE, then rii-toc.xml, then EDAL, then openjur (each an adapter).
4. Cross-source dedup + relevance gate.
5. Backfill (3–5 years), inspect.
6. Weekly scheduled refresh.
7. Pack assembly + `rag_context` jurisprudence block + flag.

## Open questions

- Pack granularity: precompute packs for common fingerprints, or assemble
  on-the-fly from corpus retrieval and cache? (Lean: on-the-fly + cache.)
- Backfill depth per source (3 vs 5 years) — tune to volume after the asyl.net
  slice shows counts.
- Whether jurisprudence retrieval should be always-on for generation or an
  explicit toggle, given prompt-budget pressure with four context layers.
- openjur relevance gate precision — risk of pulling unrelated law; may defer
  openjur until the curated sources prove the pipeline.

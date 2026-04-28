# Agent Memory and Fresh Case-Law Plan

This document describes how to turn Rechtmaschine into a real agent that:

- preserves durable knowledge across sessions,
- learns case-specific strategy and user-specific preferences over time,
- accumulates anonymized cross-case legal patterns,
- and stays current on recent jurisdiction and case law.

The core design principle is:

- durable memory and fresh legal knowledge must be separate systems.

The model should not "learn" by keeping everything in chat history. It should:

1. write durable facts into explicit memory,
2. retrieve that memory when needed,
3. search old artifacts when needed,
4. refresh volatile legal knowledge through retrieval and research.

## Why This Split Matters

There are two fundamentally different knowledge types in this product:

- stable knowledge
  - case facts
  - lawyer preferences
  - strategy patterns
  - user corrections
- volatile knowledge
  - latest case law
  - recent court decisions
  - changing country information
  - new legal developments

Stable knowledge belongs in memory.
Volatile knowledge belongs in retrieval with freshness rules.
Old transcripts, drafts, and research runs belong in searchable history, not hot memory.

If both are mixed into one undifferentiated memory store, the agent will become unreliable.

## What OpenClaw Does Well

OpenClaw provides a strong memory pattern that should be adapted here.

- Memory is explicit and inspectable.
  - It uses Markdown files as the source of truth.
- Memory is retrieved, not assumed.
  - The agent uses `memory_search` and `memory_get` tools.
- Important facts are written before context loss.
  - OpenClaw uses a silent pre-compaction memory flush.
- Retrieval is hybrid.
  - Vector similarity for semantic recall.
  - BM25 / full-text for exact tokens.
- Retrieval quality is improved by:
  - temporal decay for recency,
  - MMR for result diversity.

The main lesson to copy is not "Markdown files specifically".
The main lesson is:

- memory must be explicit,
- memory must be auditable,
- memory must be searchable,
- memory must be separate from transient chat context.

## What Hermes Agent Adds To The Design

Hermes Agent contributes a second important lesson set.

- Keep always-on memory very small and curated.
  - Hermes injects only a bounded snapshot of persistent memory into the system prompt.
- Separate hot memory from searchable history.
  - Hermes keeps small persistent memory distinct from `session_search` over prior conversations.
- Trigger write-back explicitly.
  - Hermes uses periodic memory nudges and a pre-reset flush before context is lost.
- Treat scope as a first-class safety boundary.
  - Hermes has already hit real scope-leak issues when memory or search were too global across profiles or chats.

The main lesson to copy from Hermes is not "use two markdown files".
The main lessons are:

- always-on memory must stay compact,
- searchable history should be on-demand,
- memory write-back needs explicit lifecycle hooks,
- scope boundaries must be enforced rigorously.

## Combined Design Synthesis

The three strongest external influences play different roles.

They should not be copied literally.
They should be combined.

### OpenClaw: Session Memory Operations and Retrieval

OpenClaw is the strongest implementation reference for:

- explicit memory/search tools
- search-first, then exact-read retrieval
- hybrid recall over notes and optionally session transcripts
- pre-compaction memory flush before context loss
- background consolidation and promotion
- treating session memory and memory operations as a real runtime subsystem

This maps best to:

- document Q&A and draft-session recall behavior
- searchable artifact history
- silent reflector flushes before compaction or case/context loss
- retrieval mechanics such as search, exact read, recency handling, and diversity

OpenClaw should therefore influence:

- session and interaction memory mechanics
- retrieval APIs and ranking behavior
- pre-compaction / pre-reset write-back hooks

### Karpathy: Maintained Knowledge Base Layer

Karpathy is the strongest conceptual reference for:

- a persistent maintained synthesis layer between raw sources and answers
- raw sources remaining immutable
- compiled summaries becoming the main query surface
- schema-like guidance for how the knowledge base should be maintained
- index, log, lint, and contradiction tracking for maintained knowledge artifacts

This maps best to:

- `pattern_wiki_entries`
- provenance-rich maintained summaries
- page-like reusable legal knowledge organized by fingerprint and tags

Karpathy should therefore influence:

- the anonymized cross-case wiki
- contradiction handling
- freshness/linting for maintained knowledge pages
- the distinction between raw artifacts and compiled reusable knowledge

### Hermes Agent: Bounded Hot Memory and Continual Learning

Hermes is the strongest reference for:

- keeping always-on memory small
- separating hot memory from searchable history
- explicit memory nudges and flush behavior
- continual learning through repeated conservative write-back
- strict scope boundaries to avoid cross-session leakage

This maps best to:

- bounded injected case brief and strategy summaries
- conservative reflector-based write-back after meaningful turns
- continual learning from accepted drafts, corrections, and Q&A
- hard scoping between case-local memory and cross-case knowledge

Hermes should therefore influence:

- memory size discipline
- write-back timing and lifecycle hooks
- continual learning thresholds
- scope isolation rules

### Resulting Combined Architecture

Use the external systems for different layers:

- OpenClaw for memory operations, retrieval mechanics, and session-loss prevention
- Karpathy for the maintained cross-case wiki knowledge layer
- Hermes for bounded hot memory and continual write-back discipline

In Rechtmaschine, that becomes:

- raw layer
  - documents
  - drafts
  - research runs
  - `RechtsprechungEntry`
- case synthesis layer
  - `case_briefs`
  - `case_strategies`
- cross-case maintained knowledge layer
  - `pattern_wiki_entries`
- searchable history layer
  - prior drafts
  - Q&A transcripts
  - research runs
  - playbook-linked artifacts
- freshness layer
  - jurisprudence packs
  - live research

This split is preferable to treating "memory" as one undifferentiated store.

## Current Building Blocks Already Present In Rechtmaschine

The codebase already has most of the persistence needed for a first version.

- Case workspace state already exists in `Case.state`.
- Documents, saved sources, drafts, and research runs are already persisted.
- Curated extracted case-law entries already exist as `RechtsprechungEntry`.
- Research already has a multi-provider pipeline and stores `ResearchRun`.

Relevant code:

- `app/models.py`
- `app/endpoints/research_sources.py`
- `app/endpoints/generation.py`
- `app/endpoints/query.py`
- `app/static/js/app.js`

Current limitation:

- follow-up context in "Dokument befragen" is only browser-side working memory.
- it is not durable server-side agent memory.

## Target Architecture

### 1. Working Memory

This is the short-lived context of the current interaction.

Examples:

- current selected documents
- current draft
- current follow-up chat
- currently loaded research results

This should remain ephemeral and scoped to the active interaction.

### 2. Durable Structured Memory

This is the persistent memory the agent should learn from.

Do not start with one generic `agent_memories` bucket.
The product needs at least three distinct memory layers.

#### 2a. Case Brief Memory

Primary key:

- `owner_id`
- `case_id`

Purpose:

- the durable factual brief for one matter
- what the system should know instantly when re-entering a case

Examples:

- who the mandant is
- family relations
- procedural history
- what has already happened in the matter
- what the client wants
- what we are trying to achieve
- current status, risks, and next steps

Suggested model:

- `case_briefs`
- `id`
- `owner_id`
- `case_id`
- `content_json`
- `search_text`
- `version`
- `last_reflected_at`
- `created_at`
- `updated_at`

Source-level provenance belongs in `case_brief_sources`.
Revision history belongs in `case_memory_revisions`.

#### 2b. Case Strategy Memory

Primary key:

- `owner_id`
- `case_id`

Purpose:

- the distilled legal approach for one case
- not raw drafts, but the durable argumentation takeaways

Examples:

- strongest arguments
- weak points and evidentiary gaps
- what framing worked
- what failed
- which authorities mattered
- distilled versions of prior `Klagebegründungen`, `Schriftsätze`, and `AZBs`

Suggested model:

- `case_strategies`
- `id`
- `owner_id`
- `case_id`
- `content_json`
- `search_text`
- `version`
- `last_reflected_at`
- `created_at`
- `updated_at`

Source-level provenance belongs in `case_strategy_sources`.
Revision history belongs in `case_memory_revisions`.

#### 2c. Anonymized Cross-Case Pattern Wiki

This is the reusable learning layer across similar matters.

It should not be keyed by `case_id`.
It should be keyed by a normalized case fingerprint plus flexible tags.

Examples of fingerprint dimensions:

- gender
- country of origin
- type of `Rechtsbehelf`
- type of protection status
- type of persecution
- procedural posture
- issue tags such as `PTSD`, `Dublin`, `conversion`, `domestic violence`

Purpose:

- summarize what we have learned across similar cases
- summarize recurring legal patterns from prior work and processed Rechtsprechung
- speed up work on new but similar matters

Suggested model:

- `pattern_wiki_entries`
- `id`
- `owner_id` nullable
- `scope`
  - `private`
  - `firm`
- `fingerprint`
- `tags`
- `title`
- `summary`
- `argument_patterns`
- `risk_patterns`
- `evidence_patterns`
- `recommended_next_steps`
- `confidence`
- `last_used_at`
- `created_at`
- `updated_at`

With provenance in a separate join structure:

- `pattern_wiki_sources`
- `pattern_wiki_entry_id`
- `source_type`
  - `case_brief`
  - `case_strategy`
  - `draft`
  - `document`
  - `research_run`
  - `rechtsprechung_entry`
- `source_id`
- `anonymized_note`

Important design rules:

- do not let the model write arbitrary permanent free-form memory without structure
- do not duplicate raw source documents in memory
- store distilled, durable, higher-level information only
- every persisted memory item must remain traceable to a source

### 3. Searchable History

Separate durable hot memory from larger historical artifacts.

This layer should include:

- prior drafts
- document Q&A transcripts
- research runs
- promoted Rechtsprechung entries

This is the equivalent of Hermes' `session_search` idea:

- not always injected
- searched on demand
- summarized into compact context when needed

Suggested first implementation:

- PostgreSQL full-text over persisted artifacts and transcripts
- later optional hybrid retrieval

### 4. Jurisprudence Corpus and Packs

Create a separate store for volatile legal knowledge.

This should not live in the case brief or case strategy memory tables.

Possible model:

- `jurisprudence_chunks`
- `jurisprudence_documents`
- or an extended normalized structure around `RechtsprechungEntry`

Store:

- court
- decision date
- aktenzeichen
- country
- issue tags
- leitsaetze / holdings
- normalized text chunks
- source URL
- source type
- retrieval metadata
- optional embedding

This corpus should be refreshed continuously and searched with freshness-aware ranking.

On top of the corpus, build `jurisprudence packs`.

A jurisprudence pack is a compact, agent-facing bundle for a case fingerprint, for example:

- `woman + widerruf + afghanistan`
- `alleinstehende frau + afghanistan`
- `iran + konversion`

Suggested pack contents:

- normalized fingerprint / issue tags
- top recent decisions
- short holdings
- argument patterns
- source IDs
- `last_refreshed_at`
- `newest_decision_date`
- `refresh_after_days`
- `coverage_confidence`

Important distinction:

- the corpus is the searchable base layer
- the jurisprudence pack is the compact prompt payload the agent actually injects

## Retrieval Design

### Memory Retrieval

Before drafting or answering, retrieve:

- relevant case brief memory
- relevant case strategy memory
- relevant user preferences
- matching anonymized pattern wiki entries where appropriate
- old artifacts only when explicitly needed

First version:

- PostgreSQL full-text search plus metadata filters

Later:

- add vector search or hybrid search when corpus size justifies it

Reason:

- exact tokens like names, countries, Aktenzeichen, sections of law, and dates are often better served by structured filters and full-text first.

#### Retrieval Order By Task

For document Q&A inside an active case:

1. case brief memory
2. case strategy memory
3. only if needed, search prior drafts or Q&A transcripts

For draft generation:

1. case brief memory
2. case strategy memory
3. matching anonymized pattern wiki entries
4. freshness-checked jurisprudence pack
5. only if needed, search older drafts or research artifacts

For a new case with little prior history:

1. sparse case brief memory
2. matching anonymized pattern wiki entries
3. freshness-checked jurisprudence pack

#### Context Injection Cap

Memory retrieval must have a strict injection cap from the first version.

Recommended default:

- inject only compact synthesized blocks, not a large list of raw rows
- keep case brief memory to a small bounded brief
- keep case strategy memory to a small bounded strategy brief
- keep anonymized pattern wiki injection to a small bounded set of matching entries
- apply a token budget cap in addition to the row cap
- rank by relevance first, then recency

If an active case accumulates a large number of memory rows:

- trigger a memory compaction job
- synthesize older low-priority rows into compact case brief or strategy summaries
- keep the original rows archived and searchable

Important rule:

- compaction should reduce default prompt injection, not destroy source-traceable facts.

### Case-Law Retrieval

Before producing substantive legal output:

1. extract the legal issues from the prompt and selected documents,
2. load the matching jurisprudence pack if one exists,
3. check whether the pack is fresh and sufficiently covered,
4. if not, trigger fresh research,
5. update or rebuild the pack,
6. retrieve and rank the best matching current sources,
7. only then draft the final answer.

Fresh research should not run on every request.

It should run only when one of these is true:

- no matching jurisprudence pack exists
- the pack is older than its freshness threshold
- the pack has weak coverage for the current fingerprint
- the user explicitly asks for the latest decisions
- the case raises a new issue not represented in the pack

This should use:

- metadata filters
- full-text
- recency-aware ranking
- later hybrid search with vectors

### Jurisprudence Pack Lifecycle

Recommended lifecycle:

1. detect the case fingerprint
2. assemble candidate material from:
   - `RechtsprechungEntry`
   - saved sources
   - prior research runs
3. create or refresh the jurisprudence pack
4. inject only the compact pack into generation

This is how the system avoids both extremes:

- not searching live every time
- not relying on stale legal material forever

## Freshness Policy

Add a per-case or per-run freshness policy.

Suggested fields:

- `legal_area`
- `countries`
- `issue_tags`
- `courts`
- `max_age_days`

Examples:

- urgent asylum issue: `max_age_days = 7`
- normal jurisdiction check: `max_age_days = 30`
- evergreen doctrinal background: `max_age_days = 180`

Freshness rule:

- if the matching jurisprudence pack is older than the configured threshold, too thin, or missing, auto-run new research before answering.

## Recommended Source Strategy For Current Case Law

Use existing live research infrastructure as the main freshness layer.

Priority sources should be official or primary where available.

Candidate source set:

- Rechtsinformationen des Bundes / NeuRIS API docs
  - `https://docs.rechtsinformationen.bund.de`
- BVerwG RSS feeds
  - `https://www.bverwg.de/rss-feeds`
- asyl.net decision database
  - `https://www.asyl.net/recht/entscheidungsdatenbank/`
- existing provider-based web research already implemented in `app/endpoints/research_sources.py`

Source-priority nuance:

- NeuRIS is strategically important and should be treated as a strong federal source.
- It should not be treated as sufficient coverage for asylum-law freshness.
- In practical asylum work, VG and OVG level decisions remain critical.
- Therefore Asyl.net and state-level court sources remain essential, even after federal API integration.

Important rule:

- recent case law should be re-retrieved and re-verified.
- it should not be trusted simply because it was once stored in memory.

## Tie-In With The Existing Playbook

The current "Aktuelle Rechtsprechung" playbook is the right foundation.

Today it already stores extracted decision-level structure such as:

- `country`
- `tags`
- `court`
- `court_level`
- `decision_date`
- `aktenzeichen`
- `outcome`
- `key_facts`
- `key_holdings`
- `argument_patterns`
- `summary`

That makes it a strong curated corpus layer.

Recommended role split:

- `RechtsprechungEntry` / playbook
  - human-visible and editable decision records
  - per-decision extracted structure
  - curated source of truth for reusable jurisprudence material

- `jurisprudence pack`
  - agent-facing compact bundle for a case fingerprint
  - built from matching playbook entries plus recent research material
  - optimized for prompt injection, not for direct UI editing

Practical flow:

1. a Rechtsprechung document is promoted into the playbook
2. the playbook entry becomes searchable by country, tags, court level, and date
3. a later generation request detects a fingerprint such as `woman + widerruf + afghanistan`
4. the system builds or loads a jurisprudence pack from matching playbook entries and fresh research
5. only the compact pack is injected into the generation prompt

This means:

- the playbook remains the inspectable legal knowledge base
- the pack becomes the agent's compact working layer for drafting

The playbook should therefore evolve into:

- a curated jurisprudence corpus
- a seed layer for pack generation
- a place where useful argument patterns can be reviewed and corrected by the lawyer

## Memory Write Paths

### Case Brief Writes

The agent should be able to save durable case facts such as:

- plaintiff timeline facts
- client profile facts
- family relations
- procedural history
- current goals
- confirmed next steps

These writes should be triggered after:

- document analysis
- user correction
- accepted draft generation
- important chat clarification

### Case Strategy Writes

The agent should save durable strategic takeaways such as:

- strongest argument lines
- weak points and contradictions
- effective framing choices
- evidentiary gaps
- distilled argumentation from accepted filings

These writes should be triggered after:

- accepted draft generation
- high-signal user corrections to a draft
- important legal clarifications in chat
- research completion when it materially changes the approach

### User Memory Writes

The agent should save durable user preferences such as:

- preferred phrasing style
- preferred argument order
- preferred strategic framing
- rules the lawyer explicitly gives

This memory must be user-scoped, not case-scoped.

### Reflection / Wrap-Up

Inspired by OpenClaw:

- after a generation or chat turn, run a lightweight reflection step
- ask whether the turn produced durable facts or stable preferences
- write only high-confidence structured memories

Do not start by writing everything automatically.
Start with strict thresholds and conservative extraction.

### Reflector Pattern

The default write-back mechanism should use a separate reflector step, not direct mid-thought memory writes from the main agent.

Recommended flow:

1. the main agent completes the user-facing task
2. a background job picks up the completed transcript or draft interaction
3. a cheaper, tightly-constrained model extracts candidate memories
4. the backend validates and persists only structured, source-linked results

Why:

- keeps the main UX fast
- avoids prompt bloat in the main drafting/chat path
- reduces the risk of the main agent writing speculative or premature memory
- makes memory extraction easier to inspect, retry, and disable independently

Recommended reflector behavior:

- strict JSON schema output
- empty array when no durable memory is present
- extract only:
  - explicit user preferences
  - confirmed case facts
  - durable strategy takeaways
- anonymizable cross-case patterns
- do not allow free-form long-term memory invention

Recommended write-back routing:

- confirmed client/case facts -> `case_briefs`
- durable argumentation takeaways -> `case_strategies`
- reusable anonymized patterns -> `pattern_wiki_entries`

Direct memory writes by the main agent should be reserved for:

- explicit user instructions such as "remember this"
- very high-confidence structured writes where the source is immediate and unambiguous

## Why Not Start With pgvector

`pgvector` is useful, but it should not be step one.

Recommended order:

1. structured memory tables
2. metadata filters
3. PostgreSQL full-text search
4. retrieval injection into generation/query flows
5. reflection jobs
6. then vector search / hybrid retrieval

Reason:

- the current product already has enough structure for a strong non-vector first version,
- the main missing piece is durable memory logic, not embedding infrastructure,
- legal retrieval benefits heavily from exact filters and citations.

## Why FastAPI BackgroundTasks Are Not Enough

`BackgroundTasks` can be useful for low-stakes reflection, but not for the core memory and jurisprudence pipeline.

For production-grade agent behavior, use a durable queue for:

- memory extraction jobs
- jurisprudence ingestion jobs
- re-embedding jobs
- re-ranking / enrichment jobs

Otherwise memory writes and ingestion can be lost on restart or deploy.

## Queue Strategy

Start with a Postgres-backed queue.

Recommended implementation:

- a `job_queue` table
- worker polling with `FOR UPDATE SKIP LOCKED`
- job types such as:
  - `reflect_memory`
  - `refresh_case_law`
  - `ingest_jurisprudence`
  - `compact_memory`

Why this is the right first step:

- no extra infrastructure beyond Postgres
- queue state is naturally captured in database backups
- simpler operationally than adding Redis or Celery immediately

Minimum queue features required:

- retries
- backoff
- idempotency
- stuck-job recovery
- failure logging

Do not treat a Postgres queue as "free".
It is the right first move only if these reliability basics are implemented.

## Concrete Implementation Plan

### Phase 1: Maintained Case Pages

Start with one maintained brief and one maintained strategy per case.
Do not begin with many granular memory rows.

Add:

- `case_briefs`
- `case_strategies`
- add indexes for:
  - `owner_id`
  - `case_id`
  - `created_at`
  - `updated_at`
- add Pydantic models for read/update/render

Each table should store:

- `id`
- `owner_id`
- `case_id`
- `content_json`
- `search_text`
- `version`
- `last_reflected_at`
- `created_at`
- `updated_at`

Acceptance:

- each active case can have one inspectable, editable case brief and one inspectable, editable case strategy.
- empty sections are represented explicitly so gaps and open questions are visible.

### Phase 2: Provenance and Revisions

Add source and revision tracking before reflector automation.

Add:

- `case_brief_sources`
- `case_strategy_sources`
- `case_memory_revisions`

Source records should support:

- `source_type`
  - `document`
  - `draft`
  - `chat`
  - `research_run`
  - `rechtsprechung_entry`
  - `user_instruction`
- `source_id`
- `label`
- `excerpt`
- `metadata`

Revision records should store:

- target type
- target id
- previous `content_json`
- new `content_json`
- source refs
- actor
- created_at

Acceptance:

- case brief and strategy updates are traceable and reviewable.
- a bad update can be inspected and rolled back.

### Phase 3: Deterministic Retrieval Injection

Inject the maintained case pages into existing flows before adding full-text search.

- `app/endpoints/generation.py`
- `app/endpoints/query.py`

For document Q&A:

1. load the active case brief directly
2. load the active case strategy directly
3. inject compact rendered blocks before selected documents and chat history
4. keep total memory injection under a hard token budget

For draft generation:

1. load the active case brief directly
2. load the active case strategy directly
3. inject both before selected documents and drafting instructions
4. later add pattern wiki and jurisprudence pack injection

Injection limits:

- hard token budget cap for injected memory
- compact rendered sections only
- no dumping full revision history or all source excerpts into normal prompts
- no vector search in the MVP

Acceptance:

- drafts and document Q&A visibly use the maintained case brief and strategy.
- prompt payloads remain bounded and predictable.

### Phase 4: Manual UI Editing

Add frontend panels for maintaining the two case pages.

UI should allow:

- view/edit case brief
- view/edit case strategy
- show sources
- show revision history
- restore prior revision
- mark missing information / open questions

Acceptance:

- lawyers can correct memory before trusting automated updates.
- memory is visible, not hidden prompt state.

### Phase 5: Cheap LLM Case Extraction

Use a cheap long-context model to extract structured facts from case documents.
Do not use this as an unsupervised final legal strategy writer.

Recommended role:

- extractor for per-document facts
- synthesizer for proposed case brief updates
- assistant for proposed case strategy updates

Candidate models:

- Qwen3.5-35B-A3B
  - preferred first candidate for text-heavy files
  - long context and efficient MoE architecture make it attractive for structured extraction
- Gemma 4 31B
  - preferred candidate for scanned, visual, or layout-heavy documents
  - useful where OCR, images, handwriting, charts, or PDF layout matter

Extraction flow:

1. run per-document or per-Akte-segment extraction
2. emit strict JSON with source refs
3. store extraction artifacts separately from maintained memory
4. synthesize proposed updates to `case_brief` and `case_strategy`
5. route the result into `memory_update_proposals`

Per-document extraction should capture:

- people and roles
- family relations
- dates and procedural events
- client goals
- claims and feared harms
- contradictions and credibility issues
- evidence and missing evidence
- open questions
- page or document source refs

Important rule:

- long context is useful, but do not rely on one monolithic "read the whole case" prompt as the only mechanism.
- prefer hierarchical extraction: document extraction, then case-level synthesis, then reviewed patch proposal.

Acceptance:

- selected case documents can produce structured extraction artifacts with source refs.
- the system can propose initial case brief and strategy patches from those artifacts.
- no extraction result is applied without review in the MVP.

### Phase 6: Reviewed Reflector Proposals

Add structured write-back as proposed patches, not silent writes.

Add:

- `memory_update_proposals`

Proposal records should store:

- `target_type`
  - `case_brief`
  - `case_strategy`
- `target_id`
- `status`
  - `pending`
  - `accepted`
  - `rejected`
  - `superseded`
- `ops`
- `source_refs`
- `confidence`
- `model`
- `created_at`
- `reviewed_at`

Start with explicit triggers:

- "propose memory update from selected documents"
- "propose memory update from this draft"
- "propose memory update from this Q&A"
- "propose memory update from this correction"

Reflector output should be patch-like:

- `set` for scalar fields
- `append` for list sections
- `remove` only after explicit review

Backend validation must enforce:

- allowlisted paths
- valid JSON schema after patch
- source refs present
- version match before apply

Acceptance:

- reflector output can be reviewed, accepted, rejected, and audited.
- accepted proposals create revisions and source records.

### Phase 7: Controlled Auto-Apply

Only after reviewed proposals are reliable, allow auto-apply for narrow cases.

Allowed candidates:

- high-confidence low-risk additions to open questions
- explicit user preference updates
- explicit user instruction such as "remember this"

Not allowed for auto-apply initially:

- client identity or family facts
- deadlines
- legal strategy changes
- cross-case pattern wiki updates

Acceptance:

- auto-applied updates are rare, explainable, reversible, and source-linked.

### Phase 8: Pattern Wiki MVP

Only after case-local memory is working, add the Karpathy-style maintained cross-case wiki.

Add:

- `pattern_wiki_entries`
- `pattern_wiki_sources`

Scope:

- anonymized reusable argument patterns
- risk patterns
- evidence patterns
- next-step heuristics
- fingerprint/tag keyed retrieval

Acceptance:

- no case-specific or identifying facts enter the wiki.
- each wiki entry has provenance and anonymization notes.

### Phase 9: Freshness Gate and Jurisprudence Packs

Before final legal generation:

- extract issue tags
- inspect matching jurisprudence pack, `ResearchRun`, and related sources
- decide whether the pack is stale, thin, or missing
- run targeted research when needed
- rebuild the pack when needed

Acceptance:

- the system does not rely blindly on old legal search results and does not search live unnecessarily on every request.

### Phase 10: Jurisprudence Corpus

Build a dedicated ingestion pipeline for case law.

Suggested tasks:

- nightly or scheduled ingestion
- normalize source metadata
- extract holdings / proposition chunks
- store chunks in a searchable corpus

Use:

- exact filters first
- full-text second
- vectors later

Acceptance:

- the agent can retrieve current, issue-specific jurisprudence without depending only on generic live search.
- the playbook and the jurisprudence pack builder share the same normalized base records where possible.

### Phase 11: Hybrid Retrieval

Only after the above is stable:

- enable embeddings
- add hybrid search
- add temporal decay
- add MMR

Acceptance:

- retrieval quality improves for semantically phrased queries without losing exact citation precision.

## Concrete Hooks Into Current Codebase

### Add New Persistence

- `app/models.py`
  - add `CaseBrief`
  - add `CaseStrategy`
  - add `CaseBriefSource`
  - add `CaseStrategySource`
  - add `CaseMemoryRevision`
  - add `CaseDocumentExtraction`
  - add `MemoryUpdateProposal`
  - add `PatternWikiEntry`
  - add `PatternWikiSource`
  - add jurisprudence chunk models later
- `app/main.py`
  - add schema migration

### Add New Endpoints / Service Layer

- `app/endpoints/agent_memory.py`
  - get/update/render case brief
  - get/update/render case strategy
  - list sources and revisions
  - create/list document extraction artifacts
  - create/list/review memory update proposals
  - expose pattern wiki endpoints later
- or keep a small internal service module first if no UI is needed yet

### Wire Retrieval Into Existing Flows

- `app/endpoints/generation.py`
  - inject compact rendered case brief and case strategy before generation
  - add pattern wiki and freshness-checked jurisprudence pack later
- `app/endpoints/query.py`
  - inject compact rendered case brief and case strategy before document Q&A
- `app/endpoints/documents.py`
  - add extraction trigger for selected documents or a full case file
- `app/endpoints/research_sources.py`
  - add fingerprint extraction, freshness checks, and pack refresh logic later
- `app/endpoints/rechtsprechung_playbook.py`
  - treat playbook entries as a primary input source for jurisprudence pack assembly

### UI Evolution

Add inspectable case brief and case strategy panels so memory stays auditable.

That panel should show:

- what the system learned
- where it came from
- confidence
- revision history
- pending reflector proposals
- delete / correct controls

## Frontend Plan

The frontend should evolve incrementally from the existing layout.

The goal is not to build a new agent UI from scratch.
The goal is to make durable memory, jurisprudence packs, and freshness state visible and controllable within the current page.

### Guiding Rule

Keep transient interaction state in the browser, but move durable agent state to the backend.

In practice:

- browser state remains responsible for:
  - current input text
  - current stream rendering
  - temporary follow-up chat state
- backend state becomes responsible for:
  - durable case memory
  - user preferences
  - jurisprudence packs
  - freshness status
  - reflection / queue status where relevant

### 1. Add a Case Memory Panel

Add a new visible panel for case-level memory tied to the active case.

This panel should show:

- case brief summary
- timeline facts
- confirmed case facts
- client goals
- current status
- next steps
- confidence and source information

Recommended actions:

- edit
- delete
- pin / prioritize
- mark as incorrect

Why:

- memory must remain auditable
- lawyers need to see what the system has learned
- this prevents hidden, magical, and unreviewable memory accumulation

### 1b. Add a Case Strategy Panel

Add a separate but adjacent panel for strategy memory.

This panel should show:

- strongest arguments
- weak points and risks
- key authorities
- evidentiary gaps
- distilled takeaways from prior accepted filings

Why:

- factual brief and legal approach should not be conflated
- lawyers need to see whether the system is remembering facts or argument choices

### 2. Extend "Aktuelle Rechtsprechung" Into Playbook Plus Pack

The current playbook UI is the correct home for jurisprudence features.

Recommended structure:

- tab 1: `Entscheidungen / Playbook`
- tab 2: `Jurisprudenz-Pack`

Role of each:

- Playbook
  - human-visible, editable corpus of individual decision entries
- Jurisprudenz-Pack
  - compact, agent-facing legal bundle for the detected case fingerprint

The pack view should display:

- detected fingerprint
- freshness status
- `last_refreshed_at`
- `newest_decision_date`
- top decisions included
- compact holdings
- argument patterns
- whether live refresh was triggered

Recommended actions:

- refresh now
- rebuild pack
- inspect source decisions

### 3. Show Fingerprint Detection In Query and Drafting

When the user asks a document question or requests a draft, show the detected issue fingerprint.

Examples:

- `frau`
- `widerruf`
- `afghanistan`
- `alleinstehend`
- `§ 73 AsylG`

This should appear as chips or badges near the query/generation area.

Why:

- the user can quickly verify whether the agent classified the matter correctly
- the user can catch wrong routing before the system uses the wrong pack or memory

### 4. Show Freshness State Inline

The UI should expose whether the legal context used for generation is current.

Recommended statuses:

- `Rechtsprechung aktuell`
- `Rechtsprechung wird aktualisiert`
- `Kein passender Pack vorhanden`
- `Pack möglicherweise veraltet`

This should appear:

- in the drafting area before generation
- in the jurisprudence pack panel
- optionally in research history entries

Why:

- users should understand why a run was fast or why a refresh happened
- freshness checks should feel intentional, not mysterious

### 5. Make Draft Inputs Traceable

When a draft is generated, the draft modal should show what memory and legal context went into the prompt.

Recommended display blocks:

- `Verwendeter Fall-Überblick`
- `Verwendete Strategie-Merkpunkte`
- `Verwendete Nutzer-Präferenzen`
- `Verwendete anonymisierte Muster`
- `Verwendeter Jurisprudenz-Pack`
- `Frische Recherche ausgelöst: ja/nein`

This should be compact, not a full debug panel.

Why:

- increases trust
- makes debugging possible
- helps the lawyer understand why the wording or argument order changed

### 6. Keep the Current Query UX, But Upgrade the State Model

The existing "Dokument befragen" UX should stay conversational.

However:

- the browser should no longer be the owner of all long-term context
- durable takeaways from document Q&A should be reflected into backend memory
- jurisprudence-pack usage should be surfaced to the user when relevant

This means:

- keep frontend streaming and follow-up feel
- but back it with server-owned memory and pack state

### 7. Suggested Frontend Implementation Order

1. Add backend-backed `Case Memory` panel.
2. Extend `Aktuelle Rechtsprechung` to `Playbook + Jurisprudenz-Pack`.
3. Add fingerprint chips in query and draft flows.
4. Add freshness indicators.
5. Add draft provenance blocks for memory + jurisprudence-pack usage.

### 8. Non-Goals For The First Frontend Iteration

- no separate full-screen agent cockpit
- no complex graph view of memories
- no autonomous hidden UI changes
- no replacement of the existing drafting / research / query boxes

The first frontend iteration should make the agent's learned state visible and controllable without disrupting the current workflow.

## Non-Goals For The First Iteration

- fully autonomous self-modifying agent behavior
- opaque permanent memory writes
- storing all current jurisprudence as permanent memory
- heavy vector infrastructure before structured retrieval is working

## Recommended First Milestone

The highest-value first milestone is:

- one maintained `case_brief` and one maintained `case_strategy` per case,
- provenance and revision history from day one,
- compact deterministic injection into generation and document Q&A,
- visible UI panels for review and manual editing,
- cheap LLM extraction from selected documents into source-linked artifacts,
- reviewed `memory_update_proposals` from drafts, Q&A, and corrections.

This creates the memory substrate first.
Pattern wiki entries, jurisprudence packs, hybrid retrieval, and autonomous write-back should come after the case-local memory loop is reliable.

## External Basis

- OpenClaw memory concept:
  - `https://github.com/openclaw/openclaw/blob/main/docs/concepts/memory.md`
- OpenClaw session-memory hook:
  - `https://github.com/openclaw/openclaw/blob/main/src/hooks/bundled/session-memory/handler.ts`
- OpenClaw memory flush:
  - `https://github.com/openclaw/openclaw/blob/main/src/auto-reply/reply/memory-flush.ts`
- OpenClaw session and compaction design:
  - `https://github.com/openclaw/openclaw/blob/main/docs/reference/session-management-compaction.md`
- OpenClaw memory tools:
  - `https://github.com/openclaw/openclaw/blob/main/src/agents/tools/memory-tool.ts`
- Hermes Agent persistent memory:
  - `https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/memory.md`
- Hermes Agent session search:
  - `https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/sessions.md`
- Hermes Agent memory configuration:
  - `https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example`
- Hermes Agent scope-leak lessons:
  - `https://github.com/NousResearch/hermes-agent/issues/6320`
  - `https://github.com/NousResearch/hermes-agent/issues/10554`

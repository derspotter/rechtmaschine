# Agent Memory and Fresh Case-Law Plan

This document describes how to turn Rechtmaschine into a real agent that:

- preserves durable knowledge across sessions,
- learns case-specific and user-specific preferences over time,
- and stays current on recent jurisdiction and case law.

The core design principle is:

- durable memory and fresh legal knowledge must be separate systems.

The model should not "learn" by keeping everything in chat history. It should:

1. write durable facts into explicit memory,
2. retrieve that memory when needed,
3. refresh volatile legal knowledge through retrieval and research.

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

### 2. Durable Agent Memory

This is the persistent memory the agent can learn from.

Create a new structured table, for example `agent_memories`.

Suggested fields:

- `id`
- `owner_id`
- `case_id` nullable
- `scope`
  - `user`
  - `case`
- `kind`
  - `fact`
  - `timeline`
  - `strategy`
  - `style_preference`
  - `research_takeaway`
  - `risk`
  - `todo`
- `title`
- `body`
- `citation`
- `source_type`
  - `document`
  - `research_run`
  - `draft`
  - `chat`
  - `user_instruction`
- `source_id`
- `confidence`
- `last_used_at`
- `created_at`
- `updated_at`

Important design rule:

- do not let the model write arbitrary permanent free-form memory without structure.

Memory writes must be conservative, scoped, and traceable to a source.

### 3. Jurisprudence Corpus and Packs

Create a separate store for volatile legal knowledge.

This should not live in `agent_memories`.

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

- relevant case memories
- relevant user preferences
- recent strategy notes
- recent research takeaways

First version:

- PostgreSQL full-text search plus metadata filters

Later:

- add vector search or hybrid search when corpus size justifies it

Reason:

- exact tokens like names, countries, Aktenzeichen, sections of law, and dates are often better served by structured filters and full-text first.

#### Context Injection Cap

Memory retrieval must have a strict injection cap from the first version.

Recommended default:

- inject at most 20 memory rows
- apply a token budget cap in addition to the row cap
- rank by relevance first, then recency

If an active case accumulates a large number of memory rows:

- trigger a memory compaction job
- synthesize older low-priority rows into a single `case_background` style memory row
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

### Case Memory Writes

The agent should be able to save durable case facts such as:

- plaintiff timeline facts
- credibility disputes
- contradictions
- recurring strategic points
- accepted argument patterns for this case

These writes should be triggered after:

- document analysis
- user correction
- accepted draft generation
- important chat clarification

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
- do not allow free-form long-term memory invention

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

### Phase 1: Durable Memory Schema

- add `agent_memories` table
- add indexes for:
  - `owner_id`
  - `case_id`
  - `scope`
  - `kind`
  - `created_at`
  - full-text column / TSVECTOR if used
- add Pydantic models for memory create/search/read

Acceptance:

- memory rows can be written, listed, and filtered by case and user.

### Phase 2: Memory Retrieval Inference Path

Inject memory retrieval into:

- `app/endpoints/generation.py`
- `app/endpoints/query.py`

Retrieval order:

1. case memory
2. user memory
3. recent research runs
4. selected `RechtsprechungEntry` material

Injection limits:

- maximum 20 injected memory rows
- separate token budget cap for injected memory
- if a case exceeds a configured memory threshold, enqueue `compact_memory`

Acceptance:

- drafts and document Q&A visibly use prior case facts and user preferences.

### Phase 3: Memory Write-Back

Add structured write-back after:

- successful document Q&A sessions
- accepted draft generation
- user corrections
- research completion

Start with:

- reflector job instead of default direct tool writes
- strict JSON extraction schema
- conservative extraction prompt
- no arbitrary self-modification by the agent
- backend validation before persistence

Default execution model:

- enqueue `reflect_memory` after completed document Q&A sessions
- enqueue `reflect_memory` after accepted draft generation
- enqueue `reflect_memory` after high-signal user corrections

Acceptance:

- new durable facts appear in memory with source traceability.

### Phase 4: Freshness Gate and Jurisprudence Packs

Before final legal generation:

- extract issue tags
- inspect matching jurisprudence pack, `ResearchRun`, and related sources
- decide whether the pack is stale, thin, or missing
- run targeted research when needed
- rebuild the pack when needed

Acceptance:

- the system does not rely blindly on old legal search results and does not search live unnecessarily on every request.

### Phase 5: Jurisprudence Corpus

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

### Phase 6: Hybrid Retrieval

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
  - add `AgentMemory`
  - optionally add jurisprudence chunk models
- `app/main.py`
  - add schema migration

### Add New Endpoints / Service Layer

- `app/endpoints/agent_memory.py`
  - create/list/search memory
- or keep a small internal service module first if no UI is needed yet

### Wire Retrieval Into Existing Flows

- `app/endpoints/generation.py`
  - inject retrieved memory and the freshness-checked jurisprudence pack before generation
- `app/endpoints/query.py`
  - inject case memory and user memory before document Q&A
- `app/endpoints/research_sources.py`
  - add fingerprint extraction, freshness checks, and pack refresh logic
- `app/endpoints/rechtsprechung_playbook.py`
  - treat playbook entries as a primary input source for jurisprudence pack assembly

### UI Evolution

Later add an inspectable "Case Memory" panel so memory stays auditable.

That panel should show:

- what the system learned
- where it came from
- confidence
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

- timeline facts
- confirmed case facts
- strategy notes
- risks
- lawyer preferences for this case
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

- `Verwendete Fall-Merkpunkte`
- `Verwendete Nutzer-Präferenzen`
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

- durable structured case memory plus user preference memory,
- retrieval injection into generation and document Q&A,
- a hard context injection cap,
- reflector-based memory write-back,
- and jurisprudence packs with freshness gates on top of the existing research pipeline.

This will make the system feel meaningfully more agentic without overcomplicating the architecture too early.

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

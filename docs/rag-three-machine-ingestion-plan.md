# Three-Machine RAG Ingestion Plan

## Summary

This plan defines the production split for Rechtmaschine RAG ingestion and retrieval across three machines:

- `desktop`: dedicated Qwen3.6 worker for anonymization, anonymized metadata extraction, and segmentation. It also collects datasource manifests and selected staged source artifacts from Nextcloud and j-lawyer because those sources are available there.
- `debian`: dedicated RAG machine for OCR, embedding, reranking, persistent vector/full-text storage, retrieval API, and ingestion jobs. It imports datasource exports from desktop.
- `server`: main Rechtmaschine app host; calls Debian's private RAG API and does not run GPU/RAG worker services.

The key design decision is that Debian owns persistent RAG storage and retrieval. This avoids a query-time loop where the server retrieves candidate chunks and sends them back to Debian for reranking. With Debian owning the store, the server sends a query once and receives final anonymized chunks.

All stored and embedded content must be anonymized first. Raw source files remain in the Kanzlei Nextcloud corpus, j-lawyer, or desktop export staging area, not in the RAG store.

Operational note: if Codex is working in `/var/opt/docker/rechtmaschine`, it is on the server host unless explicitly verified otherwise. Server-side work may develop connectors and export public source material such as DokuWiki raw pages, but production chunking, embedding, vector/full-text storage, reranking, and retrieval must happen on Debian.

## Architecture

### Desktop: Qwen3.6 and datasource export

Desktop should reserve VRAM for Qwen3.6. It must not run OCR, embedding, reranking, or the RAG vector store. Desktop also owns datasource collection/export because it has the practical access to both Kanzlei Nextcloud and j-lawyer.

Responsibilities:

- Run `service_manager.py` as the anonymization/Qwen role.
- Route Qwen3.6 payloads to the local Ollama instance.
- Provide endpoints for:
  - anonymization,
  - anonymized metadata extraction,
  - segmentation if ingestion tests show it is useful.
- Build Nextcloud inventory/manifest exports.
- Build j-lawyer metadata manifests.
- Stage selected Nextcloud and j-lawyer source files needed for ingestion tests or selected ingestion batches.
- Write checksums and non-sensitive provenance for staged files.
- Keep export manifests path-stable and machine-portable.
- Do not embed, rerank, or store RAG vectors.

Recommended user service env file:

```env
# ~/.config/rechtmaschine/service-manager.env
ANON_BACKEND=qwen
OLLAMA_URL=http://127.0.0.1:11435/api/generate
OLLAMA_MODEL=qwen3.6:27b-q4_K_M
SERVICE_MANAGER_ROLE=anonymization
SERVICE_MANAGER_PORT=8002

# Desktop must not own OCR/RAG workloads.
RAG_EMBED_ENABLED=0
RAG_RERANK_ENABLED=0
```

Operational commands:

```bash
cd ~/rechtmaschine
git pull --ff-only
systemctl --user daemon-reload
systemctl --user restart service-manager
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8002/status
```

Expected state:

- anonymization/Qwen service manager open on the configured private port.
- Ollama Qwen endpoint reachable.
- `8085` and `8086` not required on desktop.
- No BGE embedding or reranker containers running on desktop.
- export root exists outside the repo at `/home/jayjag/rechtmaschine-rag-export`.

### Debian: RAG and OCR machine

Debian owns the full RAG pipeline after desktop export and Qwen anonymization.

Responsibilities:

- Pull the same Rechtmaschine repo.
- Pull exported manifests and selected staged source files from desktop.
- Validate imported checksums before processing.
- Run OCR locally.
- Run BGE-M3 embedding locally.
- Run BGE reranker locally.
- Run the persistent RAG store locally.
- Run the RAG retrieval API locally and expose it privately over Tailscale/LAN.
- Run ingestion jobs.

Default v1 store:

- Use local Postgres with `pgvector` for dense vectors.
- Use PostgreSQL German full-text search for sparse/keyword retrieval.
- Implement hybrid retrieval by combining vector and full-text results, then rerank locally on Debian.
- Use Qdrant only if Postgres/pgvector testing shows a concrete blocker.

Debian service layout:

- OCR service: local HTTP endpoint, likely `127.0.0.1:9003`.
- Embedding service: local BGE-M3 endpoint, likely `127.0.0.1:8085`.
- Reranker service: local BGE reranker endpoint, likely `127.0.0.1:8086`.
- RAG API: private endpoint, for example `http://debian:<rag-port>/v1/rag`.
- Qwen upstream: `http://desktop:8002`.

Recommended Debian env shape:

```env
# Debian RAG worker/API environment
DESKTOP_QWEN_URL=http://desktop:8002
RAG_DESKTOP_IMPORT_ROOT=/home/justus/rechtmaschine/rag/data/imports/desktop-export
OCR_SERVICE_URL=http://127.0.0.1:9003
RAG_EMBED_URL=http://127.0.0.1:8085
RAG_RERANK_URL=http://127.0.0.1:8086
RAG_DATABASE_URL=postgresql://rechtmaschine:<password>@127.0.0.1:5432/rechtmaschine_rag
RAG_API_HOST=0.0.0.0
RAG_API_PORT=<rag-port>
RAG_SERVICE_API_KEY=<shared-secret>
```

Operational commands:

```bash
cd ~/rechtmaschine
git pull --ff-only

# Pull desktop datasource export.
rsync -aH --info=progress2 \
  jayjag@desktop:/home/jayjag/rechtmaschine-rag-export/ \
  /home/justus/rechtmaschine/rag/data/imports/desktop-export/

# Start or restart local worker services.
bash ocr/run_hpi_service.sh
bash rag/run_bge_m3.sh
bash rag/run_bge_reranker.sh

# Start the Debian RAG API once implemented.
# The exact command should live in the RAG API implementation/runbook.
```

Health checks:

```bash
curl -fsS http://127.0.0.1:9003/health
curl -fsS http://127.0.0.1:8085/health
curl -fsS http://127.0.0.1:8086/health
curl -fsS http://desktop:8002/health
curl -fsS http://127.0.0.1:<rag-port>/v1/rag/health
```

### Server: app host only

The server runs the main Rechtmaschine app and delegates RAG retrieval to Debian.

Responsibilities:

- Keep the normal FastAPI app and database responsibilities.
- Configure the app to call Debian for RAG.
- Do not run OCR, embedding, reranking, or Qwen workloads.
- Do not persist the RAG vector store for this architecture.
- Do not run production chunking or embedding jobs.
- Public source gathering, such as exporting `https://wiki.aufentha.lt/` raw DokuWiki pages, is allowed on server only as connector/source-export work. The exported corpus must be handed to Debian for production chunking and indexing.

Recommended app env:

```env
RAG_SERVICE_URL=http://debian:<rag-port>
RAG_SERVICE_API_KEY=<shared-secret>
```

Retrieval flow:

1. User asks a retrieval-backed question in the app.
2. Server sends the query and filters to Debian's RAG API.
3. Debian embeds the query, performs hybrid dense/full-text search, reranks candidates locally, and returns final anonymized chunks.
4. Server uses those final chunks in the app response.

Server must not call Debian reranker directly and must not send retrieved candidate texts back to Debian.

## Export Boundary

Datasource exporters write source bundles outside the repo. Desktop remains the main export owner for private Nextcloud and j-lawyer material; server may export public DokuWiki pages.

```text
/home/<user>/rechtmaschine-rag-export/
  manifests/
    nextcloud_*.jsonl
    jlawyer_*.jsonl
    merged_*.jsonl
  dokuwiki/
    manifests/dokuwiki_*.json
    ingested/dokuwiki_*.jsonl
    raw/dokuwiki/*.txt
    checksums/dokuwiki_*.sha256
  staged_files/
    nextcloud/...
    jlawyer/...
  checksums/
    sha256sums.txt
```

Debian imports that export with:

```bash
rsync -aH --info=progress2 \
  jayjag@desktop:/home/jayjag/rechtmaschine-rag-export/ \
  /home/justus/rechtmaschine/rag/data/imports/desktop-export/
```

For a server-side public DokuWiki export, import the public bundle separately:

```bash
rsync -aH --info=progress2 \
  jay@server:/home/jay/rechtmaschine-rag-export/dokuwiki/ \
  /home/justus/rechtmaschine/rag/data/imports/server-export/dokuwiki/
```

Transfer rules:

- Export manifests should use stable source-relative paths and include the export root in the run manifest, not as retrieval provenance.
- `staged_files/...` paths in manifests must resolve after Debian pulls the export.
- DokuWiki raw exports are public source material and may keep canonical URLs as source provenance.
- Raw staged files and corpus exports are not committed to Git.
- Checksums cover every staged file that Debian may OCR or process.
- Git branches should stay split by ownership: desktop branch owns collection/export tooling; Debian branch owns import, processing, and RAG ingestion tooling.

## Ingestion Datasources

Before bulk ingestion starts, desktop must build a complete datasource export. Debian must validate the imported export before embedding a large batch.

Required sources:

- Kanzlei Nextcloud corpus: the already sorted local filesystem corpus available on desktop, using the existing `rag/data/filter_reports` and `rag/data/manifests` workflow where possible.
- j-lawyer: all relevant case documents for lawyers who do not use Nextcloud, discovered by desktop through j-lawyer metadata first and content download only after inclusion rules select a document.
- Aufenthaltswiki (`https://wiki.aufentha.lt/`): public curated legal wiki pages exported from DokuWiki raw markup. This is a reusable source of doctrine, statutes, country notes, and case-law summaries. It belongs in the cross-case legal knowledge/RAG layer, not in individual case memory.

Datasource requirements:

- Normalize all sources into one manifest/ingested shape before Debian OCR/anonymization/chunking.
- Keep `source_system` on every manifest item: `nextcloud`, `jlawyer`, or `dokuwiki`.
- For Nextcloud items, keep an export-relative source path and source hash.
- For j-lawyer items, keep non-sensitive provenance such as j-lawyer case id, document id, document name, creation/change date, size, and tags.
- For DokuWiki items, keep page id, title, canonical page URL, raw export URL, source hash, and export run id. Wiki content is public and does not need anonymization, but it still needs source provenance and refresh timestamps.
- Do not store j-lawyer raw file content in the RAG store; only store downloaded desktop staging files long enough for transfer, OCR/text extraction, and audit/debug artifacts.
- Use the j-lawyer read paths documented in `docs/jlawyer-agent-import-plan.md` as the connector reference.

The complete datasource export is a gate: ingestion can run small smoke tests before this, but production ingestion should wait until Nextcloud and j-lawyer manifests are merged or explicitly accounted for.

## Cross-Source Deduplication (Nextcloud vs j-lawyer)

Measured June 2026 on three cases: j-lawyer holds 3-10x more documents per case
than the Nextcloud folder, but the surplus is correspondence (beA, xjustiz,
inbound court/BAMF mail) that RAG excludes anyway. Authored filings live
authoritatively in Nextcloud; overlap with j-lawyer is either byte-identical
mirrors (same name and size) or same-content-different-bytes (authored ODT vs
filed PDF, template-created docs).

Dedup layers in the Debian ingestion runner, cheapest first:

1. Source byte hash: `sha256` from the export manifests is a unique key in the
   RAG store across source systems; identical bytes are never embedded twice.
2. Normalized text hash: after text extraction, hash the lowercased,
   whitespace-collapsed text and skip exact text duplicates. Catches ODT/PDF
   pairs and re-exports with differing bytes.
3. Provenance merge: a deduplicated document keeps the source pointers of all
   copies (nextcloud path and j-lawyer case/document ids).

Consequence for the j-lawyer connector: it is a targeted top-up, not a bulk
import. Per case, select own-authored documents (template-created ODTs,
Rechtmaschine drafts) whose normalized text hash is not already in the store.

## Ingestion Pipeline

Ingestion should build on the existing `rag/data/filter_reports` and `rag/data/manifests` work instead of starting from scratch, then extend the manifest layer with j-lawyer documents collected on desktop.

Default split pipeline:

1. Desktop generates or reuses a high-confidence manifest of Kanzlei Schriftsätze from the Nextcloud corpus.
2. Desktop generates a j-lawyer manifest from case/document metadata and applies the same inclusion intent: own Kanzlei Schriftsätze first, external documents only if explicitly selected later.
3. Desktop or server exports public DokuWiki raw pages with `rag/export_dokuwiki.py`; the output uses the ingested JSONL shape, but production chunking still happens on Debian.
4. Desktop stages selected Nextcloud and j-lawyer source files for ingestion tests or selected batches.
5. Desktop writes export-relative paths, hashes, and provenance into source manifests.
6. Desktop merges the Nextcloud, j-lawyer, and DokuWiki manifests/ingested records into one datasource-complete ingestion set.
7. Debian pulls the desktop export and verifies checksums.
8. Debian prefers filename/path/j-lawyer/wiki metadata heuristics over Qwen classification.
9. Debian OCRs only when extracted text is missing or poor.
10. Debian sends non-public extracted text to desktop Qwen for anonymization and anonymized metadata extraction. DokuWiki records skip anonymization because they are public source material.
11. Debian optionally asks Qwen for segmentation if tests show it improves chunk quality.
12. Debian chunks anonymized text and public DokuWiki text.
13. Debian prepends compact metadata headers before embedding.
14. Debian embeds chunks locally.
15. Debian stores anonymized chunks, public wiki chunks, metadata, vectors, keyword index, and provenance locally.

Qwen classification is optional. Use it only for files that remain ambiguous after path and filename heuristics.

## Anonymized Metadata

Metadata is part of retrieval quality and must be anonymized before storage.

Extract and store these fields where available:

- `document_role`: Klage, Klagebegruendung, Eilantrag, Schriftsatz, Stellungnahme, Beweisantrag, Zulassungsantrag, Beschwerde, other.
- `legal_area`: Asyl, Aufenthalt, Dublin, Abschiebungsverbot, Familiennachzug, Einbuergerung, other.
- `country`: Herkunftsland or relevant state, normalized but not personally identifying.
- `claim_themes`: generalized themes such as Wehrdienst, politische Opposition, Religion, Ethnie, LGBTQ, geschlechtsspezifische Gewalt, Krankheit, Familie, Minderheit, Sippenhaft.
- `legal_issues`: generalized legal issues such as Fluechtlingseigenschaft, subsidiarer Schutz, `§ 60 Abs. 5 AufenthG`, `§ 60 Abs. 7 AufenthG`, Glaubhaftigkeit, inlaendische Fluchtalternative, Dublin-Zustaendigkeit.
- `document_date`: document date when available.
- `anonymized_summary`: short retrieval-oriented summary without names, addresses, birth dates, case numbers, or identifying details.
- `citations`: statutes and non-personal legal citations where present.

Do not store:

- names,
- addresses,
- birth dates,
- phone numbers,
- email addresses,
- BAMF/client/court file numbers if they identify the person,
- unusually specific personal facts not needed for retrieval.

Store non-sensitive provenance:

- source system (`nextcloud` or `jlawyer`),
- source path relative to the desktop export root,
- source hash,
- j-lawyer case/document ids where applicable,
- manifest run id,
- ingestion timestamp,
- document date,
- chunk index.

## Retrieval Behavior

Debian should perform the full retrieval pipeline locally:

1. Receive query and optional filters from server.
2. Embed query locally.
3. Search dense vectors.
4. Search sparse/full-text index.
5. Fuse results.
6. Rerank top candidates locally.
7. Return only final selected anonymized chunks to server.

Hybrid search means:

- dense embeddings for semantic similarity,
- sparse/full-text search for exact legal terms, countries, statutes, and doctrine words.

Recency bias is not required for v1, but `document_date` must be stored now so later ranking can boost newer documents when otherwise similar documents concern the same legal area, country, and themes.

## Interfaces

Desktop Qwen interface:

- Base URL: `http://desktop:8002`.
- Existing endpoints may be used first:
  - `/anonymize`
  - `/extract-entities`
- Add a dedicated anonymized metadata endpoint only if the existing endpoints cannot reliably return anonymized text plus structured metadata.

Debian RAG API:

- `GET /v1/rag/health`
- `POST /v1/rag/chunks/upsert`
- `POST /v1/rag/retrieve`

Expected retrieval response:

```json
{
  "query": "string",
  "chunks": [
    {
      "chunk_id": "string",
      "text": "anonymized chunk text",
      "score": 0.0,
      "metadata": {},
      "provenance": {}
    }
  ],
  "retrieval": {
    "dense_count": 0,
    "sparse_count": 0,
    "reranked": true
  }
}
```

Server app interface:

- Existing app RAG proxy can continue using `RAG_SERVICE_URL`.
- `RAG_SERVICE_URL` must point to Debian, not desktop.

## Validation Plan

### Desktop validation

```bash
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8002/status
timeout 2 bash -c '</dev/tcp/127.0.0.1/8085' && echo "unexpected embed port open" || true
timeout 2 bash -c '</dev/tcp/127.0.0.1/8086' && echo "unexpected rerank port open" || true
test -d /home/jayjag/rechtmaschine-rag-export/manifests
```

Pass criteria:

- Qwen service manager is healthy.
- RAG embed/rerank are not required on desktop.
- GPU memory is reserved primarily for Qwen.
- Nextcloud and j-lawyer export manifests use export-relative paths.
- Staged files selected for transfer have checksums.

### Debian validation

```bash
curl -fsS http://127.0.0.1:9003/health
curl -fsS http://127.0.0.1:8085/health
curl -fsS http://127.0.0.1:8086/health
curl -fsS http://desktop:8002/health
curl -fsS http://127.0.0.1:<rag-port>/v1/rag/health
test -d /home/justus/rechtmaschine/rag/data/imports/desktop-export/manifests
```

Run smoke tests:

- Pull one desktop export with `rsync`.
- Verify staged file checksums.
- OCR one scanned PDF.
- Embed one German asylum-law query.
- Rerank three anonymized candidate chunks.
- Insert one test chunk into the RAG store.
- Retrieve that chunk through `/v1/rag/retrieve`.
- Run a 10-document ingestion sample from the Nextcloud manifest.
- Run a small j-lawyer metadata inventory sample and download only selected test documents.
- Run a merged-manifest ingestion sample containing both Nextcloud and j-lawyer documents.
- Manually inspect anonymized text and metadata for leaks.

### Server validation

```bash
curl -fsS http://debian:<rag-port>/v1/rag/health
```

Pass criteria:

- App can reach Debian RAG API.
- Retrieval returns final anonymized chunks.
- Server does not call desktop or Debian reranker directly.
- Server does not store RAG vectors in this architecture.

## Rollout Order

1. Update desktop env to run only the anonymization/Qwen role and no RAG workloads.
2. Create the desktop export root and build Nextcloud and j-lawyer manifests.
3. Stage selected Nextcloud and j-lawyer source files, then write checksums.
4. Pull the desktop export to Debian with `rsync`.
5. Configure Debian with repo, import path, OCR, embedder, reranker, and local RAG store.
6. Implement or adapt Debian RAG API.
7. Run ingestion smoke tests for Nextcloud-only, j-lawyer-only, and merged-manifest batches.
8. Inspect anonymization and metadata quality.
9. Ingest a larger datasource-complete batch.
10. Point server `RAG_SERVICE_URL` to Debian.
11. Validate retrieval through the app.

## Open Follow-Up Items

- ~~Decide final Debian RAG API process manager~~ — decided June 2026: Docker Compose (`rag/docker-compose.debian.yml`), API on port 8090, pgvector Postgres on 127.0.0.1:5433, DB `rechtmaschine_rag`. See `docs/debian-rag-docker-setup.md`.
- Decide RAG database backup path on Debian.
- Implement desktop export tooling for Nextcloud and j-lawyer manifests.
- Implement Debian import validation for export-relative paths and checksums.
- Decide whether segmentation materially improves chunks after a sample comparison.
- Add recency bias later using stored `document_date`.
- Add monitoring for anonymization failures and retrieval drift after first real usage.

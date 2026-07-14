# Draft-Context Pipeline: /generate-Parität für direktes Drafting

**Datum:** 2026-07-14 · **Status:** approved by Jay (design conversation 14.07.)

## Problem

Wenn der Assistent Schriftsätze/Stellungnahmen direkt im Terminal entwirft
(ODT via j-lawyer-Templates, gerichtsrubrum-Pipeline), fehlen ihm die
Kontext-Anreicherungen, die Rechtmaschines `/generate`-Endpoint automatisch
injiziert. Ergebnis: Drafts ohne Kanzlei-Präzedenzmuster, ohne
Gesetzestext-Block, mit nur zufällig gezogenem Fallgedächtnis und ohne
deterministische Faktenprüfung.

Was `/generate` heute injiziert (Code: `app/endpoints/generation.py`):

1. **RAG-Präzedenz** — `rag_context.build_rag_block(query, owner_id,
   case_name, limit, collect)`: bis zu 6 anonymisierte Argumentations-Chunks
   aus ANDEREN Kanzlei-Mandaten (debian RAG-Store, Collection `kanzlei`),
   eigene Akte per `case_hash` ausgeschlossen, Provenienz in `collect`.
   Gated per `RAG_RETRIEVAL_ENABLED`; degradiert zu `""`.
2. **Fallgedächtnis** — `agent_memory_service.get_case_memory_prompt_context`
   ("KOMPAKTES FALLGEDÄCHTNIS"-Block, Grounding in dict).
3. **Gesetzestexte** — `legal_context.build_statute_block`.
4. **Kanzlei-Stilregeln** — im System-Prompt von `/generate` verdrahtete
   Regeln (Mandanten-E-Mails nie als Anlage/Beweis zitieren, Parteivortrag
   vs. Beweis, Zitierregeln, attributionale Formeln).
5. **Fact-Verify** — `citation_verifier.verify_facts(draft_text,
   selected_documents, memory_text)`: deterministische Prüfung von Daten,
   Aktenzeichen, Beträgen gegen Memory + Quelldokumente (kein Modell-Call).

## Entscheidung (Jay, 14.07.)

- Scope: ALLE vier Anreicherungen + Fact-Verify.
- Architektur: **neue App-Endpoints**, die exakt dieselben Funktionen rufen
  wie `/generate` (Parität by construction, keine Logik-Duplikation).
- Packaging: **gerichtsrubrum-Skill wird zum Drafting-Pipeline-Skill und
  heißt fortan `drafting`** (Rename inkl. aller Querverweise; alte Trigger
  bleiben in der description erhalten).

## Komponenten

### 1. App: `POST /workflow/draft-context`

Request: `{ "case_id": "<uuid, optional>", "query": "<str, required>",
"rag_limit": <int, optional> }`

Response:

```json
{
  "rag_block": "…|''",
  "statute_block": "…|''",
  "case_memory_block": "…|''",
  "style_rules": "…",
  "grounding": { "rag_chunks": [...], "case_memory_text": "..." }
}
```

- Aufrufkette identisch zu `/generate`: `_rag_block_for_generation` (bzw.
  `build_rag_block` mit `exclude_case_hash` aus dem Case-Namen),
  `build_statute_block`, `get_case_memory_prompt_context`.
- `style_rules`: die heute in `/generate` inline verdrahteten Stilregel-
  Strings werden in eine Modul-Konstante extrahiert (z. B.
  `generation.DRAFTING_STYLE_RULES`), die `/generate` UND der neue Endpoint
  verwenden — eine Quelle, kein Drift.
- Ohne `case_id`: RAG + Gesetzestexte kommen trotzdem, Memory-Block leer.
- Jeder Block degradiert bei Fehlern zu `""` (gleiche Semantik wie in
  `/generate`); der Endpoint wirft nur bei fehlendem `query`.

### 2. App: `POST /workflow/verify-facts`

Request: `{ "text": "<str, required>", "case_id": "<uuid, optional>",
"sources": ["<str>", …] (optional) }`

- Ruft `citation_verifier.verify_facts(text, selected_documents, memory_text)`.
- `memory_text` aus dem Fallgedächtnis des `case_id` (leer ohne case_id);
  `sources` werden als Quelltexte in die `selected_documents`-Struktur
  gemappt (Kategorie "Sonstiges").
- Response: das `{"fact_checks": [...], "fact_summary": {...}}`-Dict
  unverändert.

### 3. CLI: zwei Subcommands in `rechtmaschine_cli.py` / Wrapper

- `rechtmaschine-cli draft-context "<case-ref-oder-uuid>" --query "…"
  [--rag-limit N] [--out ctx.md]` — löst Case-Referenz wie gehabt auf,
  druckt die Blöcke als Markdown (oder schreibt `--out`).
- `rechtmaschine-cli verify-facts --text-file draft.txt
  [--case-id …] [--source-file f1.txt …]` — druckt Findings tabellarisch,
  Exit-Code 1 bei high-severity-Findings (Datum/Az ohne Quelle).

### 4. Skill: Rename `gerichtsrubrum` → `drafting` + Pipeline-Abschnitt

- Verzeichnis-Rename in `~/.codex/skills` (canonical; `~/.claude/skills`
  ist Symlink aufs Repo — kein zweiter Rename nötig), `name:`-Frontmatter,
  Querverweise anpassen (`api/SKILL.md` referenziert gerichtsrubrum;
  Volltext-Grep über Skills + Auto-Memory).
- description: bisherige Trigger (Rubrum, rechtsbündig, Formatierung
  prüfen, …) BLEIBEN, ergänzt um "Schriftsatz/Stellungnahme/Klagebegründung
  entwerfen", "draft a filing".
- Neuer Abschnitt "Kontext-Pipeline (PFLICHT bei jedem Schriftsatz)":
  1. VOR dem Formulieren: `draft-context` ziehen und einweben. RAG-Chunks
     sind anonymisierte Argumentationsmuster aus FREMDEN Akten — Muster
     übernehmen, NIE Fakten.
  2. NACH dem Formulieren: `verify-facts` über den Drafttext; high-severity
     erst auflösen, dann weiter.
  3. Danach wie bisher: `rubrum-cli check` (bleibt Pflicht) → Upload/beA.
- rubrum-cli-Skripte bleiben unverändert im umbenannten Verzeichnis.

## Fehlerverhalten

- RAG-Store down / `RAG_RETRIEVAL_ENABLED` aus → `rag_block: ""`, CLI zeigt
  Hinweis "RAG leer (Service prüfen?)" statt zu scheitern.
- Unbekannte Case-Referenz → CLI-Fehler vor dem Request (wie andere
  Subcommands).
- App nicht erreichbar → normale CLI-Retry-Semantik (`RECHTMASCHINE_FORCE_IPV4`
  etc. gelten unverändert).

## Tests / Abnahme

- App-Tests: draft-context liefert alle vier Keys; Parität der style_rules
  mit dem /generate-Prompt (Konstante wird an beiden Stellen importiert —
  Test prüft Identität); Degradation (RAG aus → ""); verify-facts flaggt
  präpariertes falsches Datum/Az und lässt belegte Fakten durch.
- Smoke/Abnahme: `draft-context "044/26 …" --query "gewöhnlicher Aufenthalt
  § 10 StAG Auslandsstudium"` liefert nichtleere Blöcke; verify-facts über
  einen Absatz mit absichtlich falschem Aktenzeichen → high-severity-Finding.
- Deploy: App ist volume-gemountet (uvicorn-Reload); job-worker unberührt.

## Nicht-Ziele

- Kein neuer Generierungs-Endpoint, keine Änderung an `/generate` außer der
  Stilregel-Extraktion.
- Keine Collection-Erweiterung (jurisprudence/doktrin-Retrieval kann die CLI
  später als `--collection`-Flag bekommen; Default bleibt `kanzlei` wie in
  `/generate`).
- xlsx/Leads/Intake unberührt.

# Research Pipeline Upgrade Plan

Status: DRAFT (2026-07-01, Jay + Claude session on case 242/25)
Owner: Jay
Scope: `app/endpoints/research/*`, `app/endpoints/jurisprudence.py`, `app/agent_memory_service.py`, `app/rag_vocabulary.py`

## 0. Evidence / why

Research run for 242/25 (Syrer, § 60 Abs. 5 / Art. 3 EMRK, "kein Netzwerk"): grok-4.3
returned 8 "hochrelevante" Entscheidungen. Manual verification:

| Hit | Problem |
|---|---|
| VG Wiesbaden "m33861" | real decision, **wrong Az** (real: 6 L 3034/25.WI.A) — usable only after manual fetch |
| VG Regensburg RN 11 K 25.33928 | real, granted — but 18-jährige kaum alphabetisierte Frau; **trivially distinguishable**, summary hid that |
| VG Düsseldorf 17 L 3613/25.A | real — **denied** (contrary), summary framed as support |
| VG Köln 27 K 4231/25.A | real — **denied** (applicant HAD network), framed as support |
| VG Bremen "25 K 1569/25" | wrong Az (real: 3 V 1569/25), Widerrufs-Fall eines Totschlägers — **toxic**, contrary |
| BVerwG 1 C 14.25 | real — off-point (inlandsbezogene Belange), mischaracterized |
| VG Köln 27 K 8782/25.A | unverifiable (openJur CAPTCHA; not in NRWE) |
| VG Osnabrück 7 B 19/26 | unverifiable (voris 403); reportedly denied |

Net usable: 1/8 (+1 with caveat). Root causes traced below. Order of work agreed with Jay:
**better grok prompting/wiring → better fetching → verifier.** Plus the structural facet
fix that makes the jurisprudence store usable from day one of a case.

## 1. Current architecture (traced 2026-07-01, file:line)

### 1a. Grok research (`research/grok.py`)
- Uses `xai_sdk` 1.11.0 agentic `web_search()` tool — **live search, real URLs**. Good.
- BUT (grok.py:416–468): calls `chat.sample()`, requests JSON via **plain-text instruction**,
  extracts with **regex**, validates with Pydantic afterwards.
- **Never uses `chat.parse()`** (SDK-native structured output).
- **Never reads `response.citations` / `response.inline_citations`** — the URLs grok
  *actually retrieved*. Instead trusts grok's free-text `sources` array = grok's
  paraphrase from memory. → real-ish URL, wrong Az, wrong outcome.
- `StructuredSource` = title/url/description only. No Gericht/Datum/Az/Ergebnis fields,
  so nothing forces page-grounding.
- SDK facts (verified in-container): `web_search(allowed_domains|excluded_domains
  [max 5, mutually exclusive], enable_image_understanding, user_location_*)`.
  **No date parameter** — recency only via prompt. `Response.citations`,
  `Response.inline_citations`, `chat.parse(PydanticModel)` all exist.

### 1b. Fetching (`research/utils.py:download_source_as_pdf`)
- HEAD/Range-probe for direct PDFs, else **Playwright renders ANY HTML to PDF** and
  reports success. CAPTCHA walls (openJur), paywalls (voris/Wolters Kluwer, beck),
  soft-404s all come back as "downloaded PDFs".
- `research-job ingest-pdfs` preflight catches *fake PDF links* but nothing validates
  *content* (is this actually the decision text?).

### 1c. Jurisprudence store (`models.py:706 RechtsprechungEntry`)
- Rich typed schema: country, tags, schlagworte, normen, court, court_level,
  decision_date, aktenzeichen, **outcome**, key_facts, key_holdings, argument_patterns,
  leitsatz, **instance_weight** (BVerfG/EuGH/EGMR/BVerwG=3, OVG=2, VG=1),
  content_sha256 dedupe, source_type (asylnet|nrwe|rii|edal|openjur).
- Ingest: asyl.net Entscheidungsdatenbank via `jurisprudence_ingest.py`
  (cron: `scripts/jurisprudence_refresh.sh`, broaden script exists).
- **No embedding/vector column. Retrieval is keyword/tag matching only.**

### 1d. Pack injection (why it silently no-ops)
Chain: `generation.py:1347` → `get_case_memory_prompt_context()`
(`agent_memory_service.py:958`) → `maybe_render_jurisprudence_context()`
(`jurisprudence.py:410`) → block "AKTUELLE RECHTSPRECHUNG (Pack, frischegeprüft)".

Wired and enabled (`JURIS_PACK_INJECT_ENABLED` defaults true). Four silent gates:
1. active case required (generation.py:1347)
2. **case memory text non-empty** — empty brief/strategy → `return ""`
3. fingerprint must extract country/legal_area/issue_tags from the memory **prose**
   (`derive_fingerprint` = regex/keyword scrape: `_COUNTRY_WORDS`, `_AREA_SIGNALS`,
   `_SECTION_RE`)
4. corpus must contain tag-matching rows; thin pack → enqueue research, return ""

**242/25 failure: gate 2** — memory was "Keine gepflegten Inhalte" at generation time.
Fresh cases (the common state) never get case-law context.

### 1e. Schema mismatch (case memory ↔ jurisprudence)
- Case memory `content_json`: German **prose lists** (beteiligte, verfahrensstand,
  sachverhalt, …, kernstrategie, argumentationslinien, …). No typed fields.
- RechtsprechungEntry: fully typed.
- **Zero shared fields.** Bridge = `derive_fingerprint` regex over rendered prose. Lossy;
  fails on empty memory or unusual phrasing.
- Already available to build on: `rag_vocabulary.py` (`normalize_country`,
  `normalize_normen`, `normalize_themen`, `facet_metadata()` aligned to RagFilters keys
  `applicant_origin`/`citations`/`schlagworte`) and `CaseDocumentExtraction`
  (per-doc JSONB: facts/entities/dates/claims — free-form, not faceted).

## 2. Pillar 1 — Grok rewiring (highest quality gain, small diff)

File: `research/grok.py` (+ `prompting.py`)

1. **Replace text-JSON + regex with `chat.parse(GrokResearchOutput)`.**
2. **Read `response.citations`** after every round; a source is only emitted if its URL
   (after redirect normalization) is in the retrieved-citations set. Grok prose ≠ truth.
3. **Enrich `StructuredSource`** (all Optional, but prompt-required for decisions):
   ```python
   class StructuredSource(BaseModel):
       url: str
       title: str
       description: str = ""
       quelle_typ: Literal["entscheidung","coi","sonstiges"] = "entscheidung"
       gericht: Optional[str]       # verbatim from opened page
       datum: Optional[str]         # ISO, from page
       aktenzeichen: Optional[str]  # verbatim from page — NEVER from memory
       ebene: Optional[Literal["BVerfG","EuGH","EGMR","BVerwG","OVG","VG","sonstige"]]
       ergebnis: Optional[Literal["stattgegeben","abgelehnt","teilweise","unklar"]]
       profil: Optional[str]        # 1-line applicant profile (Alter/Geschlecht/Gesundheit/Netzwerk)
       zitat: Optional[str]         # VERBATIM passage from page supporting the fit claim
       fit: Optional[str]           # why it matches THIS case
       lager: Optional[Literal["stuetzt","gegen","neutral"]]   # pro/contra!
   ```
4. **Prompt hard rules** (prompting.py addition): Az/Datum/Ergebnis wörtlich von der
   geöffneten Seite; `zitat` muss wörtlich auf der Seite stehen; Entscheidungen, die
   nicht geöffnet und zitiert werden können, WEGLASSEN; `ergebnis` und `lager` immer
   angeben (Gegenrechtsprechung ist erwünscht, aber als `gegen` markiert); ehrliches
   Negativ-Ergebnis ("keine stützende Entscheidung gefunden") ist eine gültige Antwort.
5. **Domain whitelist rounds**: round 1 `allowed_domains=["nrwe.justiz.nrw.de",
   "openjur.de","gesetze-bayern.de","asyl.net","rechtsprechung-im-internet.de"]`;
   round 2 (balanced/deep) unrestricted for COI + long tail. Max 5 domains per call —
   rotate, don't squeeze.
6. Keep multi-round dedup; drop `_extract_sources_from_grok_response` regex fallback to
   citations-based fallback.

## 3. Pillar 2 — Retrieval function (`research/retrieval.py`, new)

Replace blind `download_source_as_pdf` in the research path with:

```python
@dataclass
class FetchResult:
    status: Literal["ok","captcha","paywall","notfound","error","not_decision"]
    resolved_url: str
    text: str            # extracted decision text ("" unless ok)
    title: str
    pdf_path: Optional[str]  # kept for ingest compatibility
```

- Fetch (httpx, redirects, UA) → extract text (pdftotext for PDFs / trafilatura-or-
  readability for HTML) → **validate**:
  - junk markers: CAPTCHA/consent/login/"Zugriff verweigert"/cookie walls → `captcha|paywall`
  - decision heuristic: must contain ≥2 of {Gericht-name, Az-pattern, "Urteil"/"Beschluss",
    "Tenor"/"Gründe"} → else `not_decision`
- **Per-domain adapters** (only the 5 we actually use): NRWE (clean HTML), openJur
  (`/u/<id>.print` bypasses CAPTCHA — verify), gesetze-bayern (BeckRS HTML), asyl.net
  (rsdb entry + attached PDF), bverwg.de (structured decision pages). Fallback: generic.
- Playwright only as last resort for `ok`-but-JS pages, never to "succeed" on walls.
- `ingest-pdfs` keeps working: adapter produces both text and (where possible) PDF.

## 4. Pillar 3 — Verifier (`research/verify.py`, new; after 1+2)

For each `StructuredSource` with `quelle_typ=="entscheidung"`:
1. `FetchResult = retrieve(url)`; status != ok → `verifiziert=False, grund=status`.
2. Az check: normalize (whitespace/dots) and require `aktenzeichen` ∈ text.
3. Quote check: fuzzy-find `zitat` in text (normalized, ≥0.9 partial ratio).
4. Outcome check (cheap LLM or rule on Tenor): klage abgewiesen / stattgegeben ↔ `ergebnis`.
5. Attach `verifiziert: bool` + `verify_notes`. **UI shows badge; drafter (gpt-5.5)
   prompt forbids citing `verifiziert=False` sources.**
5a. **OCR fallback for scanned decision PDFs** (esp. asyl.net rsdb uploads):
   if a PDF has pages but ~no text layer (<~50 chars/page), send it to the
   PaddleOCR service (`http://debian:8004/ocr`, X-API-Key; deterministic OCR,
   NOT a vision model) and verify against `full_text`; mark `ocr_applied`.
   Availability rule (like Qwen): OCR service down/desktop asleep -> degrade to
   `ok_scan_unocred` + note, never block research. Verifier tolerance: on
   `ocr_applied` sources drop the Zitat fuzzy threshold (~0.9 -> ~0.8) and
   report failed quote matches as "nicht bestätigt (Scan)", not "widerlegt";
   Az matching already normalizes whitespace/dots.
6. Verified decisions → **write back** into `RechtsprechungEntry` (all fields present:
   outcome, leitsatz←zitat, instance_weight←ebene; sha256 dedupe). The store compounds.

## 5. Pillar 4 — Case facets (fixes pack gates + schema mismatch)

New shared, typed facet block — single source of truth for matching:

```json
{
  "herkunftsland": "Syrien",              // rag_vocabulary.normalize_country
  "staatsangehoerigkeit": "syrisch",
  "verfahrensart": "asyl_klage",          // asyl_klage|eilverfahren|aufenthG|einbuergerung|...
  "schutzgruende": ["AufenthG § 60 Abs. 5", "AsylG § 4"],   // CANONICAL normalize_normen format (Gesetz first) — NOT the lowercase _SECTION_RE dialect
  "themen": ["existenzminimum","abschiebungsverbot"],        // restricted to canonical rag_vocabulary themen (373 entries)
  "region": "Daraa",
  "profil": { "alter": 21, "geschlecht": "m", "gesundheit": "gesund",
              "familienstand": "ledig", "netzwerk_im_herkunftsland": false,
              "besonderheiten": ["ausreise_als_kind","11 jahre jordanien"] }
}
```

- **Vocabulary alignment (verified 2026-07-01 against asyl.net ingest)**:
  - `herkunftsland` → `normalize_country` canonical `laender` (73 entries; matches
    `RechtsprechungEntry.country`). Aligned.
  - `schutzgruende` → MUST use `normalize_normen` canonical format
    (`"AufenthG § 60 Abs. 5"`, Gesetz first). The lowercase section-first strings that
    `derive_fingerprint`/`_SECTION_RE` produce are a **different dialect** and never
    intersect the store's `normen` column.
  - `themen` → restricted to canonical `rag_vocabulary.json` themen. `existenzminimum`,
    `abschiebungsverbot`, `subsidiärer schutz`, `krankheit` exist; **`netzwerk` and
    `rückkehr` do NOT** — add them to the vocabulary with aliases ("soziale bindungen",
    "familiäre unterstützung" → netzwerk; "rückkehrer", "rückkehrsituation" → rückkehr)
    as part of this pillar.
  - `profil` axes → net-new; not in asyl.net. Populated for verifier write-back entries;
    back-catalog optionally via LLM pass over `key_facts`/`summary`. Scoring degrades
    gracefully when absent.
  - **Matcher bug to fix while here**: `_assemble_contents` currently matches fingerprint
    `issue_tags` against the free-form `e.tags` column, ignoring the cleanly curated
    `schlagworte`/`normen` columns. The facet matcher must compare `country` +
    `normen` + `schlagworte` directly.
- **Storage**: `cases.facets_json` (new JSONB column) + mirror into
  `CaseDocumentExtraction.extraction_json["facets"]` per source document.
- **Population**: extend the existing Bescheid extraction (agent_memory_service
  extraction path) with a facet sub-schema; backfill CLI for existing cases from the
  anonymized Bescheid text. Manual override via API/CLI.
- **Matching** (`jurisprudence.py`): `derive_fingerprint(case_memory_text)` becomes
  `derive_fingerprint(facets, case_memory_text)` — facets primary, prose scrape as
  fallback only. Kills gate 2/3 for fresh cases: pack works on first generation,
  straight from the Bescheid.
- **Scoring** per matched RechtsprechungEntry:
  - `fit` = country == + normen ∩ + themen ∩, weighted by instance_weight + recency
  - `lager` = outcome vs. our Klageziel → pro/contra split in the pack block
  - `distinguish_risk` = profile-axis mismatches (geschlecht/alter/gesundheit/netzwerk)
    → "passt, aber leicht zu unterscheiden" warning (the Regensburg trap, automated)
- Pack block renders: STÜTZEND / GEGEN UNS (mit Kernaussage) / RISIKO-HINWEISE.

## 5b. Pillar 4 runtime design (agreed 2026-07-01, Jay sign-off on Qwen split)

Four hooks in existing flows; only one new scheduled job:

1. FACET EXTRACTION — at document intake (Bescheid processed -> facet
   sub-schema in the existing extraction -> cases.facets_json). Case is
   matchable from day one, before any case memory exists. One-time backfill
   CLI over existing Spott cases from anonymized Bescheide.
2. STORE ENRICHMENT — nightly (alongside 03:30 memory cron), Qwen on the
   desktop GPU, results CACHED on RechtsprechungEntry (computed once per
   decision, case-independent):
   - profil backfill for the asyl.net back-catalog (no structured applicant
     data upstream): axes from key_facts/summary
   - reliance judgment per axis: traegt / erwaehnt / irrelevant — does the
     decision REST on the trait (Regensburg: geschlecht/bildung tragen)
   Qwen rules: advisory, never blocking; auto-wake via service-manager; small
   flat JSON schema (Gemma-hang + truncation lessons); store model label for
   later re-judging. distinguish_risk = "ungeprueft" until enriched.
3. SCORING + PACK RENDERING — at every generation/query via the existing
   maybe_render_jurisprudence_context hook. Matching: facets vs
   country/normen/schlagworte (canonical dialect, field-to-field — fixes the
   matcher bug that ignores curated columns). Per matched decision:
   deterministic profil-mismatch x cached reliance = distinguish-risk.
   ZERO model calls / fetches in the hot path. Pack block renders:
   STUETZEND / STUETZEND MIT VORSICHT (Distinguish-Risiko + tragende Achse) /
   GEGEN UNS (mit Unterscheidbarkeits-Hinweis, z.B. "dort Netzwerk vorhanden").
4. LOOP CLOSURE — research write-back (live since 2026-07-01) delivers new
   decisions WITH profil; night enriches reliance; next generation on ANY
   case scores them.

Build order: facets + deterministic mismatch first (useful day one:
"profil weicht ab" warning), Qwen reliance/backfill as nightly enrichment
increment. Vocabulary additions required: themen 'netzwerk', 'rueckkehr'
(+ aliases) in rag_vocabulary.json.

## 6. Later / optional
- pgvector embedding column on RechtsprechungEntry; hybrid retrieval (facet prefilter
  → vector rank over leitsatz+key_holdings). Do after facets prove insufficient recall.
- Bescheid-citation harvesting ("BAMF zitiert selbst") as dedicated research mode —
  extract Fundstellen from the Bescheid, fetch, check whether they actually support
  the BAMF (OCHA 15.8% pattern). Manual version won 242/25.
- Instanzenzug check (appealed/overturned?) via openJur/RII backlinks.
- Adversarial "Gegenrecherche" prompt mode (explicit: find the BAMF's best authority).
- Separate COI lane (EUAA/AA/UNHCR, "newest version wins") from Rechtsprechung lane.

## 7. Sequencing

| Step | What | Size | Depends |
|---|---|---|---|
| 1 | Pillar 1 grok rewiring (parse + citations + schema + domains) | S–M | — |
| 2 | Pillar 2 retrieval module + adapters + tests (fixture pages: openJur CAPTCHA, NRWE decision, BeckRS, asyl.net, voris 403) | M | — |
| 3 | Pillar 3 verifier + UI badge + drafter guard + write-back | M | 1,2 |
| 4 | Pillar 4 facets: column + extraction + backfill + fingerprint/scoring | M–L | — (parallel) |
| 5 | Pack block pro/contra + distinguish-risk rendering | S | 4 |

Testing: TDD per module; golden tests replaying the 242/25 run (the 8 sources above as
fixtures) — the pipeline must end with: 1 verified-support, 1 verified-with-caveat
(flagged distinguishable), 4 contrary/off-point (labelled `gegen`/dropped), 2 unverifiable
(flagged). Work on branch `research-upgrade`; feature flags per pillar
(`RESEARCH_STRUCTURED_V2`, `RESEARCH_VERIFY_ENABLED`, `JURIS_FACETS_ENABLED`).

## 8. Out of scope (explicitly)
- Engine swap (grok-4.3 stays; failure mode was wiring, not search quality).
- Draft-internal citation-verifier (048/26 work) — different subsystem, keep separate.
- Muster-Wiki changes.

## 9. Rollout log

- 2026-07-01: Pillars 1+2 merged to master (ce88fdb) and activated (container
  restart after memory-job idle window). Acceptance via real /research/jobs
  path: grounding blocks intact end-to-end (Regensburg stuetzt/stattgegeben,
  Köln gegen/abgelehnt — contrary correctly labelled).
- KNOWN GAP (small follow-up): the multi-engine aggregate path in
  research_sources.py rebuilds final_result.metadata from a fixed literal,
  dropping provider-specific keys (structured_v2, dropped_ungrounded,
  round_errors). Single-engine path preserves them (spread). Fix: carry
  per-child metadata as metadata["engines"] = {provider: child_metadata}.
  Observability only — grounding data on sources is unaffected.
- 2026-07-01 (später): Pillar 3 implemented on branch research-pillar3 —
  verify.py (deterministic Az/Zitat/Ergebnis gate), OCR fallback for scanned
  PDFs (scan_unocred degradation, PaddleOCR bridge with auto-wake), grok.py
  verification pass (RESEARCH_VERIFY_ENABLED, default on), write-back of
  verified decisions into RechtsprechungEntry (source_type=research_verified,
  model=research-verified, sha256 dedupe over Gericht|Datum|Az).
  POLICY NOTE for sign-off: write-back entries are ACTIVE immediately
  (is_active=True) — rationale: deterministically verified from official
  portals, distinguishable and bulk-removable via source_type. Flip to a
  review-gated flow (is_active=False + accept step, like the Muster-Wiki)
  if preferred. Qwen semantic tier (Tenor/ratio judgment) deliberately
  deferred — deterministic tier alone caught 6/6 of the 242/25 failures.
- 2026-07-02: Pillar 4 implemented on branch research-pillar4 (TDD, 6 neue
  Test-Dateien, alle grün):
  - Vokabular: themen_extra (netzwerk, rückkehr) + Aliase; überlebt
    build_vocabulary-Reruns wie die Alias-Maps.
  - facets.py: kanonischer Facetten-Block (laender/Gesetz-first-normen/
    themen-Dialekt, verfahrensart-Enum, getypte profil-Achsen).
  - Storage: cases.facets_json (Migration 2026-07-02_case_facets),
    GET/PUT /cases/{id}/facets (PUT normalisiert).
  - juris_facets.py (pur): derive_fingerprint(text, facets) — Facetten
    primär, Prosa nur Fallback; entry_matches Feld-zu-Feld gegen
    country/normen/schlagworte (Matcher-Bug behoben: tags-Spalte wird im
    Facetten-Pfad ignoriert); score_entry (fit/lager/distinguish_risk,
    beide Outcome-Dialekte); render_scored_block STÜTZEND / STÜTZEND MIT
    VORSICHT / GEGEN UNS (mit Kernaussage). Flag JURIS_FACETS_ENABLED
    (default on). Empty-Memory-Gate fällt: Fall mit Facetten matcht ab
    Tag eins.
  - Intake: facet_extraction.py als EIGENER kleiner flacher Qwen-Call
    (ABWEICHUNG vom Sub-Schema-Plan — Truncation/Hang-Lektionen), Hook in
    documents-Reflection UND j-lawyer-Fold (Bescheide zuerst), nur solange
    der Fall keine matchbaren Facetten hat; Merge fill-only (manueller
    PUT-Override gewinnt immer). backfill_case_facets.py (--dry-run).
  - Increment 2: juris_enrichment.py — nächtlicher Qwen-Cache profil +
    reliance je Achse auf RechtsprechungEntry (Migration
    2026-07-02_rechtsprechung_enrichment; Modell-Label für Re-Judging);
    systemd-Units jurisprudence-enrichment.{service,timer} 03:50 (NICHT
    aktiviert — Rollout-Schritt).
  - AUSGELASSEN: Mirror der Facetten in CaseDocumentExtraction (Extraktion
    läuft über kombiniertes Material, per-Dokument-Mirror passt nicht;
    Provenienz steckt in den Reflection-Proposals).
  - POLICY NOTES für Sign-off: (1) Merge-Politik fill-only — späterer
    besserer Bescheid überschreibt nie, Korrektur nur via PUT;
    (2) lager=neutral (remand/unknown) rendert unter STÜTZEND MIT
    VORSICHT mit Hinweis "Ergebnis unklar"; (3) Facet-Pfad filtert
    Hybrid-Kandidaten strikt (Country-Mismatch fliegt raus, auch bei
    hoher semantischer Relevanz).
  - Review-Runde (high-effort, 26 Agents): 10 bestätigte Findings, alle
    gefixt (ae3d2bf) — u.a. Render-Time-Re-Scoring statt gecachter
    Profil-Risiken, Roh-Material statt Kontext-Block für den Intake-Hook,
    prose_tags gegen Country-Wildcard, facets_complete-Gate statt
    run-once, Hybrid-Fallback bei leerem Facet-Filter, PUT-Merge statt
    Wipe, Urgency-Maximum, Owner-Scoping, Cooldown-Vererbung über den
    Fingerprint-Dialektwechsel.
  - Rollout: merge → Container-Neustart (Migrationen laufen beim Start) →
    docker exec rechtmaschine-app python backfill_case_facets.py --dry-run,
    dann ohne --dry-run → systemctl --user enable --now
    jurisprudence-enrichment.timer → Akzeptanz: /cases/{id}/facets für
    242/25 prüfen, Pack-Block einer frischen Akte ohne Memory ansehen.

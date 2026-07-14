# Research-Verifikation durch lokales Qwen (statt Marker-Regeln)

Datum: 2026-07-13 · Status: approved (Justus: "ja go")

## Problem

Der deterministische Research-Verifier (`verify.py`, Pillar 3) ist beim
Ergebnis-Check fail-open: `verifiziert = az AND zitat AND ergebnis is not
False` — wenn die Tenor-Marker kein klares Signal geben, wird Groks
Ergebnis-Behauptung ungeprüft übernommen. Dazu ist "wird aufgehoben" als
Grant-Marker ambivalent (Berufungsaufhebung eines stattgebenden Urteils),
und das `lager:`-Tag ist reine Grok-Einschätzung. Ein falsches Ergebnis
sortiert die Entscheidung im Jurisprudenz-Pack in die falsche Sektion.
Separat: die Write-back-Dedupe-SHA bricht an Gerichtsnamen-Varianten
("VG Düsseldorf" ≠ "Verwaltungsgericht Düsseldorf") — 3 von 18
Bestand-Einträgen sind Duplikate.

## Entscheidung (Justus)

Komplette Verifikation der Grok-Quellen durch **lokales Qwen 27B**
(desktop/3090, bestehender citation_qwen-/service_manager-Pfad mit
Auto-Wake). Nicht Gemma (schwächer bei juristischem Deutsch,
Structured-Output-Hang, 24k-Cap), kein Gemma-first-Fallback (überdimensioniert
für Niedrigvolumen).

## Design

- **Neues Modul** `app/endpoints/research/verify_qwen.py`:
  - `select_page_excerpt(page_text, zitat)` — Seitenausschnitt: Kopf (Tenor)
    plus Fenster um den Zitat-Fundort, Gesamtbudget wie
    `CITATION_QWEN_PAGE_CHAR_LIMIT`.
  - `build_verify_prompt(grounding, page_text, det_checks)` — deutscher
    Prompt: Behauptung (Gericht, Datum, Az, Ergebnis, Zitat, lager) + die
    deterministischen String-Befunde als Fakten-Anker + Seitenausschnitt.
  - `parse_verdict(parsed)` — strukturiertes Urteil: Checks je Feld
    (az/zitat/ergebnis/lager), Gesamt-`verifiziert`, confidence, deutsche
    Begründung. Unparsebar → Exception → fail-closed.
  - `verify_source_qwen(grounding, fetch_result, qwen_call)` — kombiniert
    alles zu einem `VerifyResult`. Nicht-Entscheidungsquellen (COI):
    Erreichbarkeit genügt, kein LLM-Call (wie bisher). Widerspricht Qwen dem
    `lager`, wird das Tag entfernt und notiert.
- **Wiring**: `verify_ranked_sources` (verify.py, bleibt pur) nimmt eine
  injizierbare async `verify_fn`; grok.py wählt per
  `RESEARCH_VERIFY_BACKEND` (default `qwen`, Rollback `deterministic`) und
  reicht die impure Seam (ensure_anonymization_service_ready +
  call_qwen_json) hinein. Service-Readiness einmal pro Batch.
- **Fail-Richtung strikt closed**: Qwen nicht erreichbar / Timeout /
  unparsebar / Verdict unklar → `verifiziert=False` + Notiz, kein
  Store-Write-back. Desktop-Ausfall kostet Recall, nie Korrektheit.
- **Dedupe (deterministisch, kein Modell)**: `writeback_identity_sha`
  kanonisiert Gerichtsnamen (bverwg/ovg/vgh/vg-Langformen, Füllwörter wie
  "der freien hansestadt"); Bestand: SHAs der research_verified-Einträge
  neu berechnen, die 3 Duplikate löschen (Ops-Schritt via docker exec).
- **Badge-Semantik**: Meta-/Web-Quellen behalten den reduzierten
  deterministischen Az-Check (können nie in den Store).

## Tests

Pure Tests mit injizierten Qwen-Antworten (kein Live-Call in der Suite):
Prompt-Inhalt, Parser happy/garbage, fail-closed-Pfade, lager-Korrektur,
verify_fn-Injection, kanonische Court-SHA. Akzeptanz live: ein echter
Research-Lauf (Syrien-Query) mit Log-Kontrolle der Verdicts + Store-Diff.

## Rollout

Flag default an → Container-Restart → ersten (Auto-)Research-Lauf
beobachten. Rollback = `RESEARCH_VERIFY_BACKEND=deterministic`.

## Addendum 2026-07-14: Playwright-Fallback fürs Verify-Fetching (Justus: "ok")

Befund: Verifikations-Fetch ist reines httpx (kein JS) — 2 von 35
Grounding-Quellen scheiterten an JS-Walls (openJur-Slider), obwohl der
Playwright-Stack im selben Container liegt (asyl.net-Suche, HTML→PDF).
Playwright-FIRST wurde verworfen (94% der Quellen brauchen kein Rendering,
PDFs kann der Browser ohnehin nicht parsen, Playwright ist der flakigste
Baustein). Stattdessen:

- `fetch_source` bekommt ein injizierbares ``render_fn(url) -> html``
  (analog ``ocr_fn``, Modul bleibt pur). Nur wenn das httpx-Ergebnis
  `blocked`/`not_decision` UND kein PDF ist, ein einziger Render-Versuch;
  der gerenderte Text läuft erneut durch `classify_page`. Erfolg →
  status ok + note "via Playwright gerendert"; Misserfolg → ursprünglicher
  Status bleibt (ehrlich, fail-closed).
- `RenderSession` (grok.py-Seite): lazy Chromium-Launch beim ersten
  Bedarf, EIN Browser pro Verify-Batch, Semaphore(2), 30s-Timeout pro
  Seite, Cleanup nach verify_ranked_sources. Flag
  `RESEARCH_RENDER_FALLBACK` (default an).
- openJur→nrwe-Az-Mirror: ZURÜCKGESTELLT, bis der Render-Fallback
  gemessen ist (billigste Option zuerst; eigener Scraper wäre neue
  Flakiness).

## Addendum 2026-07-14 (3): GEAS-Rechts-Umbruch im Jurisprudenz-Store (Justus: "ok lets go")

Folgefrage zur GEAS-Entdeckung: mehrere hundert Store-Einträge hängen an
GEAS-geänderten AsylG-Normen (§ 3: 158 Nennungen, § 29: 110, § 3b: 106,
§ 4: 33, § 30: 29). Analog zu _LAGE_CUTOFFS (bewährtes Muster) in
juris_facets.py:

- `recht_stale`: Entscheidung vor dem 12.06.2026 UND Normen in der
  GEAS-Policy-Liste (AsylG §§ 3–3e, 4, 29/29a, 30/30a, 36, Dublin-Zitate;
  bewusst NICHT § 60 Abs. 5/7 AufenthG, EMRK, GRC — Tatsachen-Kern
  unverändert). Wirkung wie lage_stale: nie STÜTZEND-Führung, Bucket-Ende,
  Render-Note "GEAS: … dogmatisch nur eingeschränkt übertragbar,
  Tatsachenwürdigung weiter verwertbar" (bei GEGEN-UNS: "Rechtsgrundlage
  geändert — starkes Distinguishing"). Advisory, nie blockierend.
- Normen-Brücke je RECHTSFRAGE (nicht je Gesetz — § 3 und § 4 sind
  verschiedene Schutzformen und matchen einander weiterhin nicht):
  geas:fluechtling (§§ 3–3e ↔ VO 2024/1347 Art. ≤14), geas:subsidiaer
  (§ 4 ↔ Art. 15–17), geas:unzulaessig (§§ 29 ↔ 1348 Art. 38),
  geas:offensichtlich (§§ 30 ↔ Art. 39/42), geas:ou-verfahren (§ 36 ↔
  Art. 67/68), geas:zustaendigkeit (Dublin ↔ AMMR). VO-Zitat ohne
  Artikel spannt alle Tokens seiner VO auf. Brücke wirkt in
  entry_matches UND im Normen-Anteil des Fit-Scores (Nenner bleibt die
  rohe Fall-Liste).
- Live verifiziert: 79 § 30-Einträge matchen einen VO-Art.-42-Fall über
  die Brücke; Pro-Entscheidungen rendern MIT VORSICHT + GEAS-Note.

## Addendum 2026-07-14 (2): legal_texts auf NeuRIS umgestellt (Justus: "ja")

Messung zur Frage "brauchen wir Law-APIs wirklich" ergab: Verify-Mirror &
Co. nein (kein gemessener Bedarf) — aber die Normtext-Quelle war ein
realer Fehler: bundestag/gesetze (GitHub) ist tot (AsylG-Datei zuletzt
2018 committet, Inhalt "zuletzt geändert 9.7.2021"), die GEAS-Reform
(in Kraft 12.06.2026, Folgeänderung 10.07.2026) fehlte komplett — die
§§-Injektion in Research-Prompts lieferte einen Monat lang Vor-GEAS-Recht
(§ 3 und § 30 AsylG sind in der GEAS-Fassung komplett umgeschrieben,
Verweise auf VO (EU) 2024/1347/1348).

Umsetzung: legal_texts/neuris.py (Suche mit exaktem abbreviation-Match →
Expression-ELI → HTML-Encoding → Extractor-kompatibles Markdown, stdlib-
Parser gegen echte API-Fixtures getestet); downloader.py NeuRIS-first mit
GitHub-Fallback (GG/AsylbLG fehlen in der Testphase noch — selbstheilend,
sobald sie auftauchen) und looks_valid_law_markdown-Gate (Anker-§§ +
Mindestzahl, nie kaputte Konvertierung über gute Datei). Live-Refresh:
AsylG Fassung 2026-07-10, AufenthG 2026-07-01. Wöchentlicher Timer
legal-texts-refresh (Mo 05:45) mit Mail an Justus bei Fassungsänderung
(Gesetzesänderung = anwaltliche Nachricht, nicht nur Ops).

Live-Befund bei der Abnahme: die beiden historisch geblockten Quellen
(beide openjur.de) sind KEINE JS-Walls, sondern ein interaktives
Rotations-CAPTCHA — auch der echte Chromium sieht nur die 185-Zeichen-
Challenge. Der Render-Fallback bleibt (korrekt + kostenlos für Batches
ohne Walls, rettet echte JS-Seiten), aber für openJur ist die Gegenmaßnahme
prompt-seitig: RESEARCH_PRIORITY_BLOCK weist Grok jetzt an, openjur.de zu
meiden und dieselbe Entscheidung auf offiziellen Portalen zu verlinken.
Der Az-Mirror bleibt zurückgestellt, falls das nicht reicht.

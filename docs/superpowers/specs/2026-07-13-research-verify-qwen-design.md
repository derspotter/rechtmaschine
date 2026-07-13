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

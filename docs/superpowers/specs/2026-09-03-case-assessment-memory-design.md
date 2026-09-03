# Rechtliche Würdigung als drittes Memory-Target (`case_assessment`)

Stand: 03.09.2026. Entschieden mit Jay in der Session zu 157/26 (Khushvakhtov).

## Problem

Rechtsrecherchen einer Session enden heute als `Vermerk_*.pdf` in der
j-lawyer-Akte. Der nächtliche j-lawyer-Reflect liest den Vermerk zwar (Beleg:
Proposal 285579b5 vom 03.09.2026, Quelle `Vermerk_2026-09-02_Recherche_GÜB_statt_Duldung.pdf`),
destilliert ihn aber in Fakten-Listen von Brief und Strategie. Verloren gehen
Prüfungsstruktur, verifizierte Fundstellen mit tragender Aussage, Gegenlinie und
Bewertung. Die Generierung sieht davon nur zwei Sätze in
`rechtliche_ansatzpunkte`. Vier Seiten Recherche mit neun verifizierten
Entscheidungen sind im Prompt nicht vorhanden.

## Ziel

Ein drittes, versioniertes Memory-Target `case_assessment` neben `case_brief`
und `case_strategy`, das je Rechtsfrage ein strukturiertes Gutachten mit
Fundstellen hält, beim Accept deterministisch gegen den
Kanzlei-Rechtsprechungsstore geprüft wird, fehlende Entscheidungen über die
bestehende Zitat-Beschaffung nachlädt, in Generierung und Query als eigener
Block injiziert wird und die primäre Quelle für `wiki distill` ist.

Der Vermerk in j-lawyer bleibt Pflicht und bleibt das lesbare Langdokument. Das
Gutachten in der Memory ist die maschinenlesbare, zitierfähige Kurzform mit
Verweis auf den Vermerk.

## Entscheidungen

| Frage | Entscheidung |
|---|---|
| Granularität | Mehrere Gutachten je Akte, eines je Rechtsfrage, als Liste von Objekten in einem Target |
| Autor | Nur Sessions (Claude, Codex). Reflect und Consolidate fassen das Target nicht an |
| Verifikation beim Accept | Weich: deterministischer Store-Check setzt je Fundstelle ein Feld, Accept geht immer durch, Warnungen in der Antwort |
| Beschaffung fehlender Fundstellen | Bestehende Pipeline `draft_citation_ingest.spawn_for_text`, danach Recheck |
| Patch-Ops | `append`, `set`, `remove` wie bei den anderen Targets, zusätzlich Status `ueberholt` als sanfte Variante |
| Injektion | Eigener Block nach der Strategie, eigenes Budget 4.000 Zeichen, nur `verified`-Fundstellen zitierfähig |
| Pseudonymisierung | Wie Brief und Strategie (Namen, Geburtsdaten, Adressen der Akte), fail-closed |
| Muster-Wiki | `wiki distill` liest zuerst die aktiven Gutachten in voller Länge, Az-Whitelist aus den Gutachten |

## 1. Datenmodell

### Tabellen

`case_assessments` und `case_assessment_sources`, spiegelbildlich zu
`case_strategies` und `case_strategy_sources` (`content_json`, `search_text`,
`version`, `last_reflected_at`, Zeitstempel, Quellen-Tabelle mit
`case_assessment_id`). Migration `2026-09-03_case_assessments` in der
Migrationsliste in `app/main.py`, Indizes wie bei den Nachbartabellen.
`case_memory_revisions` und `memory_update_proposals` bleiben unverändert, sie
sind über `target_type` generisch.

### Inhalt

`content_json` hat genau ein Listenfeld `gutachten` und ein Skalarfeld
`notizen`. Ein Gutachten ist ein Objekt (Pydantic `GutachtenEntry`):

| Feld | Typ | Bedeutung |
|---|---|---|
| `id` | str, Pflicht, je Akte eindeutig | Kurzer stabiler Schlüssel, von der Session vergeben, z. B. `gueb-statt-duldung` |
| `rechtsfrage` | str, Pflicht | Ein Satz |
| `ergebnis` | str, Pflicht | Zwei bis drei Sätze |
| `stand` | Datum ISO, Pflicht | Stand der Würdigung |
| `status` | `aktiv` oder `ueberholt`, Default `aktiv` | Nur `aktiv` wird injiziert und destilliert |
| `pruefung` | Liste `PruefungsPunkt` | je `these`, `bewertung`, `fundstellen` (Liste von Az-Strings, die auf `fundstellen` verweisen) |
| `fundstellen` | Liste `Fundstelle` | je `gericht`, `datum` (ISO), `az` (Pflicht), `art` (`Urteil` oder `Beschluss`, optional), `aussage`, `richtung` (`pro`, `contra`, `neutral`), `store` (vom Server gesetzt) |
| `risiken` | Liste str | Kurze Sätze |
| `quelle` | str | Dateiname des Vermerks in j-lawyer plus Datum |

`store` je Fundstelle nimmt einen der Werte `verified`, `date_mismatch`,
`not_in_store`, `unchecked` an. Beim Create eines Proposals wird ein von der
Session mitgelieferter `store`-Wert ignoriert und auf `unchecked` gesetzt. Nur
der Server schreibt dieses Feld.

Konvention für die schreibende Session: keine Namen von Beteiligten im
Gutachten, sondern Rollen ("der Mandant", "die Freundin"). Die
Pseudonymisierung bleibt Sicherheitsnetz.

### Validierung

- Fremde Felder im Inhalt: 400 wie heute bei Brief und Strategie.
- Gutachten ohne `rechtsfrage`, `ergebnis`, `stand` oder `id`: 400.
- `status` außerhalb der zwei Werte, `richtung` außerhalb der drei Werte: 400.
- Fundstelle ohne `az`: 400.
- Doppelte `id` innerhalb von `gutachten` nach Anwendung der Ops: 400 beim
  Create und beim Accept.
- Az-Verweise in `pruefung[].fundstellen`, die in `fundstellen` des Gutachtens
  nicht vorkommen: 400.

### Patch-Ops

Unverändert gegenüber den anderen Targets: `append /gutachten/-` mit einem
vollständigen Gutachten, `set /gutachten/<index>` für die Fortschreibung,
`remove /gutachten/<index>`, `set /notizen`. Die Fortschreibung setzt `stand`
neu und lässt `store` aller Fundstellen auf `unchecked` zurückfallen, der
Accept prüft dann erneut.

### Register

`_target_spec` bekommt den dritten Eintrag. Die drei Stellen, die heute
`if brief else strategy` fest verdrahten (`_target_content`,
`_apply_patch_ops` am Ende, Validierung), werden auf eine Registerabfrage
umgestellt, damit die Validierungsfunktion aus dem Register kommt.
`MemoryTargetType` in `app/shared.py` wird um `case_assessment` erweitert,
ebenso die Response-Modelle (`CaseAssessmentResponse`) und der kombinierte
GET-Payload.

## 2. Verifikation und Beschaffung beim Accept

Beim Accept eines Proposals mit `target_type == case_assessment` läuft nach
dem Anwenden der Ops und vor dem Persistieren:

1. Für jede Fundstelle jedes Gutachtens, dessen Inhalt sich durch die Ops
   geändert hat, der deterministische Store-Check aus
   `verify_source.store_lookup` (Az normalisiert, whitespace- und
   bindestrichtolerant). Ergebnis:
   - Treffer und `decision_date` gleich `datum`: `verified`
   - Treffer, Datum abweichend oder im Store leer: `date_mismatch`
   - kein Treffer: `not_in_store`
   - technischer Fehler des Checks: `unchecked` für alle Fundstellen dieses
     Gutachtens, Accept geht trotzdem durch.
2. Für jede Fundstelle mit `not_in_store` wird eine Zitatzeile im Format des
   deterministischen Parsers erzeugt, `"<Gericht>, <art> vom <TT.MM.JJJJ> – <Az>"`.
   Fehlt `art`, wird `Beschluss` gesetzt. Das ist unschädlich, weil der Parser
   die Zeile nur für die Suche braucht und `cited_ingest` die Metadaten aus dem
   Volltext zieht. Alle Zeilen zusammen gehen an
   `draft_citation_ingest.spawn_for_text`. Scheitert der Spawn, nur Log.
3. Die Accept-Antwort enthält zusätzlich `assessment_warnings`: Liste von
   Objekten `{gutachten_id, az, store}` für alles außer `verified`. Die CLI
   druckt sie nach dem Proposal-Objekt.

Der Store-Check läuft ohne Qwen und ohne Netzwerk. Der Desktop muss nicht wach
sein.

### Recheck

`POST /memory/cases/{case_id}/assessment/recheck`, Owner-geprüft. Führt den
Store-Check über alle Fundstellen aller Gutachten erneut aus und schreibt die
`store`-Felder. Schreibt eine Revision mit `actor = "recheck"` und
`source_refs = []`, erhöht die Version nicht, erzeugt kein Proposal. Antwort:
`{changed: n, warnings: [...]}` mit derselben Warnungsstruktur wie beim Accept.
Ohne Gutachten: 200, `changed: 0`, keine Revision.

CLI: `rechtmaschine-cli memory assessment recheck [--case-id]`.

Der tägliche `memory-triage.timer` (05:00) ruft nach der Triage den Recheck für
jede eigene Akte auf, deren Gutachten mindestens eine Fundstelle mit
`not_in_store` oder `date_mismatch` hat. Fremde Akten werden wie bei der Triage
übersprungen. Was danach noch fehlt, ist Handarbeit über die Eskalationskette
des verify-source-Skills.

## 3. Prompt-Injektion

`get_case_memory_prompt_context` rendert nach Brief und Strategie einen
eigenen Block mit eigenem Budget (`max_assessment_chars`, Default 4.000). Das
bestehende Budget von 5.000 Zeichen für Brief plus Strategie bleibt unberührt.

```
RECHTLICHE WÜRDIGUNG DER KANZLEI (geprüft, Stand je Gutachten):
[gueb-statt-duldung, Stand 03.09.2026] Rechtsfrage: ... Ergebnis: ...
  Prüfung: <these> – <bewertung> – Fundstellen: OVG NRW 18 E 491/12 (18.06.2012, pro)
  Risiken: ...
  Nicht zitierfähig (nicht im Bestand): VG X 1 K 2/24
```

Regeln:

- Nur Gutachten mit `status == aktiv`, sortiert nach `stand` absteigend.
- `verified`-Fundstellen erscheinen mit Gericht, Datum, Az und Richtung an
  ihrem Prüfungspunkt. Alle anderen erscheinen nur in der Sperrliste "Nicht
  zitierfähig" mit Az.
- Kürzung bei Budgetüberschreitung in Stufen je Gutachten, beginnend beim
  ältesten: erst `risiken`, dann `pruefung`. `rechtsfrage`, `ergebnis` und die
  Sperrliste bleiben immer. Reicht das nicht, fallen ganze Gutachten von hinten
  weg, mit Hinweis `[weitere Gutachten gekürzt: n]`. Nie ein halbes Gutachten.
- Der Block läuft durch dieselbe Pseudonymisierung wie Brief und Strategie
  (`pseudonymize_case_text_for_cloud`), mit demselben Flag
  `pseudonymize_for_cloud`. Fail-closed wie heute: schlägt sie fehl, fällt der
  gesamte Memory-Text aus dem Cloud-Prompt.
- Der Block liegt vor Doktrin, Muster-Wiki und Rechtsprechungs-Pack. Das
  Matching dieser drei gegen `base_memory` schließt den Gutachten-Block ein,
  damit Fingerprints die Rechtsfragen sehen.
- `collect["assessment_used"] = True` und `collect["assessment_ids"]`
  für die Provenienzansicht des Entwurfs.

Generierung, Query-Job und Workflow-Endpunkt rufen dieselbe Funktion und
bekommen den Block ohne weitere Änderung.

## 4. Schnittstellen und Werkzeuge

### API

- `GET /memory/cases/{id}` liefert zusätzlich `case_assessment` mit `version`,
  `content_json`, `rendered`. `--versions` zeigt es mit.
- `POST /memory/cases/{id}/proposals` akzeptiert `target_type:
  "case_assessment"`. Accept und Reject unverändert, Accept-Antwort mit
  `assessment_warnings`. 409-Ordnungsblocker gilt je Target wie heute.
- `PUT /memory/cases/{id}` bleibt auf Brief und Strategie beschränkt. Kein
  manuelles Überschreiben des Gutachtens.
- Neu nur `POST /memory/cases/{id}/assessment/recheck`.

### CLI (`scripts/rechtmaschine_cli.py`)

- `_MEMORY_SECTIONS` um `"assessment": "case_assessment"` erweitern.
  `memory get --section assessment`, `--grep`, `--field` arbeiten über die
  bestehende Projektion. `_memory_entries` flacht Gutachten-Objekte zu
  Textzeilen (`rechtsfrage`, `ergebnis`, jede These, jede Fundstelle als
  "Gericht Datum Az Aussage"), damit `--grep` ein Az findet.
- `memory proposals list` zeigt bei Ops auf `/gutachten` statt des Rohobjekts
  `id | rechtsfrage | n Fundstellen`.
- `memory proposals accept` druckt `assessment_warnings`.
- `memory assessment recheck [--case-id]`.

### Triage und Stop-Hook

- `memory_triage.py`: Proposals mit `target_type == case_assessment` bekommen
  das Verdikt SESSION, unabhängig vom Modell. Kein Automatik-Accept. Der Timer
  ruft zusätzlich den Recheck wie in Abschnitt 2.
- `memory_hygiene_hook.py`: dritte Erinnerung. Hat die Session in diesem Turn
  eine Datei `Vermerk_*` in eine eigene Akte hochgeladen (Erkennung über die
  Upload-Zeile des jlawyer-cli im Transkript, wie der Hook heute Uploads
  erkennt) und kein Proposal mit `target_type == case_assessment` für diese
  Akte angelegt, druckt er `📚 Vermerk ohne Gutachten-Proposal: <Datei>`.

### Skill `rechtmaschine-memory`

Neuer Abschnitt "Gutachten (`case_assessment`)": Schema, Konvention ohne
Namen, Ablauf (Recherche, Vermerk in j-lawyer, Proposal, Accept, Warnungen
lesen, `cited_ingest` für Fehlende, Recheck), Beispiel-Payload, und der
Hinweis, dass Reflect und Consolidate das Target nicht anfassen.

### Reflect und Consolidate

Fassen `case_assessment` nicht an. Extraktionsschemata, Rolling Fold und
Konsolidierung bleiben auf Brief und Strategie beschränkt.

### Muster-Wiki (`wiki distill`)

`_execute_pattern_wiki_distillation` baut den FALL-SPEICHER neu:

1. Zuerst alle aktiven Gutachten in voller Länge, ungekürzt, mit ausschließlich
   `verified`-Fundstellen als "Gericht, Datum, Az, Richtung, Aussage".
2. Danach Brief und Strategie wie bisher.

`_DISTILL_RULES` bekommen zwei Sätze: Argumentationsmuster führen die tragende
Fundstelle in Klammern mit, Format "Gericht, Datum, Az". Es dürfen nur
Fundstellen aus dem FALL-SPEICHER verwendet werden, keine aus dem Modellwissen.

Nach der Extraktion prüft der Server mit `parse_decision_citations` aus
`draft_citation_ingest`, ob jedes im Eintrag genannte Az in den übergebenen
Gutachten-Fundstellen vorkommt (Az-Kern-Vergleich wie `cited_ingest._az_core`).
Fremde Az werden aus dem Text entfernt und im Job-Ergebnis als
`stripped_citations` gemeldet. Ohne aktives Gutachten läuft der Distill wie
heute, ohne Whitelist-Prüfung.

## 5. Fehlerfälle

| Fall | Verhalten |
|---|---|
| Ungültiges Gutachten-Objekt im Proposal | 400 beim Create |
| Doppelte Gutachten-`id` | 400 beim Create und beim Accept |
| Store-Check technisch gescheitert | Accept geht durch, `unchecked`, Warnung |
| `spawn_for_text` scheitert | nur Log, Accept unberührt |
| Recheck auf Akte ohne Gutachten | 200, `changed: 0`, keine Revision |
| Rendering über Budget | Kürzungsstufen, nie ein halbes Gutachten |
| Distill mit fremden Az | Az entfernt, Warnung, Eintrag landet als pending |
| Pseudonymisierung scheitert | ganzer Memory-Text fällt aus dem Cloud-Prompt (bestehendes Verhalten) |

## 6. Tests

Unter `tests/`, im Stil der bestehenden Memory-Tests:

- `test_memory_assessment_model.py`: Validierung, Default, Patch-Ops
  `append`/`set`/`remove` auf `/gutachten`, fremde Felder, doppelte `id`,
  Verweis auf unbekanntes Az in `pruefung`, `store` wird beim Create auf
  `unchecked` gesetzt.
- `test_memory_assessment_accept.py`: Store-Check gegen eine Test-DB mit zwei
  Entscheidungen setzt `verified`, `date_mismatch`, `not_in_store`;
  `assessment_warnings` in der Antwort; `spawn_for_text` als Mock mit
  erwarteter Zitatzeile; technischer Fehler ergibt `unchecked`.
- `test_memory_assessment_render.py`: Block-Format, Budgetstufen, nur aktive
  Gutachten, Sortierung nach `stand`, nicht verifizierte Az nur in der
  Sperrliste, Pseudonymisierung ersetzt einen Mandantennamen, `collect`-Felder.
- `test_memory_assessment_recheck.py`: Felder neu gesetzt, Revision
  geschrieben, Version unverändert, leere Akte ohne Revision.
- `test_memory_get_projection.py` erweitert: `--section assessment`, `--grep`
  trifft ein Az in einem Gutachten.
- `test_pattern_wiki_distill_assessment.py`: FALL-SPEICHER beginnt mit den
  Gutachten, fremde Az werden entfernt und gemeldet, ohne Gutachten kein
  Whitelist-Eingriff.
- Migration: Smoke-Test über den bestehenden Migrations-Testpfad.

## 7. Abnahme

Deploy über den Container wie üblich. Live-Test mit 157/26: die beiden
Gutachten vom 03.09.2026 (GÜB statt Duldung, Heirat und § 39 Nr. 5 AufenthV)
als Proposal schreiben, annehmen, Warnungen lesen, für fehlende Fundstellen
`cited_ingest`, Recheck, danach eine Test-Generierung mit Provenienzansicht,
in der `assessment_used` gesetzt ist und die verifizierten Az im Prompt
stehen. Anschließend `wiki distill` auf 157/26 und prüfen, dass die
Argumentationsmuster Fundstellen tragen.

## Nicht in diesem Schritt

- Qwen-Vorschläge für Gutachten aus Vermerken.
- Qwen-Aussageprüfung (`verify_claim`) beim Accept.
- Rückwirkendes Anlegen von Gutachten für alte Akten.
- Änderung des Budgets von Brief plus Strategie.

## Hinweis für die Umsetzung

Im Repo liegen zum Zeitpunkt der Spec uncommittete Änderungen anderer Sessions
in `app/agent_memory_service.py`, `app/endpoints/agent_memory.py` und
`scripts/rechtmaschine_cli.py`. Vor dem Bau `claim check` auf die
Tooling-Schlüssel und mit den Haltern abstimmen, in einem eigenen Worktree
arbeiten und auf den dann committeten Stand rebasen.

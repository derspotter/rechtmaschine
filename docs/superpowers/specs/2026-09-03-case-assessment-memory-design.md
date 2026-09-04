# Rechtliche Würdigung als drittes Memory-Target (`case_assessment`)

Stand: 04.09.2026, Fassung 2 nach Codex-Review (gpt-5.6-sol, 04.09.2026).
Entschieden mit Jay in der Session zu 157/26 (Khushvakhtov).

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
Kanzlei-Rechtsprechungsstore abgeglichen wird, fehlende Entscheidungen über die
bestehende Zitat-Beschaffung nachlädt, in Generierung und Query als eigener
Block injiziert wird und die primäre Quelle für `wiki distill` ist.

Der Vermerk in j-lawyer bleibt Pflicht und bleibt das lesbare Langdokument. Das
Gutachten in der Memory ist die maschinenlesbare, zitierfähige Kurzform mit
Verweis auf den Vermerk.

## Entscheidungen

| Frage | Entscheidung |
|---|---|
| Granularität | Mehrere Gutachten je Akte, eines je Rechtsfrage, als Liste von Objekten in einem Target |
| Autor | Nur Sessions (Claude, Codex) als Arbeitskonvention. Reflect und Consolidate fassen das Target nicht an. Serverseitig nicht erzwungen, siehe Abschnitt 4 |
| Verifikation beim Accept | Weich: deterministischer Store-Abgleich setzt je Fundstelle ein Feld, Accept geht immer durch, Warnungen in der Antwort |
| Beschaffung fehlender Fundstellen | Bestehende Pipeline `draft_citation_ingest.spawn_for_text`, danach Recheck |
| Patch-Ops | `append`, `set`, `remove`, Adressierung per Gutachten-`id`, nicht per Index. Status `ueberholt` als sanfte Variante |
| Injektion | Eigener Block nach Brief und Strategie, eigenes Budget 4.000 Zeichen, nur Fundstellen mit Store-Abgleich zitierfähig, gesperrte Az nicht im Fakten-Grounding |
| Pseudonymisierung | Wie Brief und Strategie (Namen, Geburtsdaten, Adressen der Akte) |
| Muster-Wiki | `wiki distill` liest zuerst die aktiven Gutachten, Az-Whitelist aus den Gutachten |
| Schnitt | Eigenes Domainmodul `app/assessment_memory.py` für Validierung, ID-Auflösung, Store-Abgleich, Rendering, Recheck. Die generische Memory-Schicht behält Persistierung, Revision, Version, Proposal-Lebenszyklus |

## 1. Datenmodell

### Tabellen

`case_assessments` und `case_assessment_sources`, spiegelbildlich zu
`case_strategies` und `case_strategy_sources`. ORM-Definition ist maßgeblich:
`apply_schema_migrations` in `app/main.py` ruft zuerst `Base.metadata.create_all()`
unter dem Advisory-Lock und danach die SQL-Migrationen. Auf einer frischen
Datenbank legt also das ORM die Tabellen an, die Migration
`2026-09-03_case_assessments` greift nur auf bestehenden Datenbanken.
Deshalb:

- `content_json` als `NOT NULL` mit serverseitigem JSONB-Default `'{}'`, im ORM
  `server_default` und `default`.
- Unique-Index auf `(owner_id, case_id)`, Indizes auf `owner_id`, `case_id`,
  `updated_at`, plus die Source-Indizes wie bei `case_strategy_sources`.
- `case_assessment_sources.case_assessment_id` mit `ON DELETE CASCADE`.
- Kein GIN-Index. Der Recheck lädt die Assessment-Zeilen je Owner und prüft in
  Python.

`case_memory_revisions` und `memory_update_proposals` bleiben unverändert, sie
sind über `target_type` generisch. Hinweis aus dem Review, nur zur Kenntnis:
`_create_revision` übergibt Felder (`revision_number`, `change_type`,
`summary`), die die Revisionstabelle nicht hat und die `_new_model` still
verwirft. Revisionen sind unnummeriert. Das wird hier nicht geändert.

### Inhalt

`content_json` hat genau ein Listenfeld `gutachten` und ein Skalarfeld
`notizen`. Ein Gutachten ist ein Objekt (Pydantic `GutachtenEntry`, alle
Untermodelle mit `extra="forbid"`, Datumsfelder als validierte ISO-Strings,
keine `date`-Objekte, weil `_model_dump` sonst nicht JSONB-serialisierbare
Werte liefert):

| Feld | Typ | Bedeutung |
|---|---|---|
| `id` | Slug `^[a-z0-9][a-z0-9-]{1,63}$`, Pflicht, je Akte eindeutig | Stabiler Schlüssel, von der Session vergeben, z. B. `gueb-statt-duldung` |
| `rechtsfrage` | str, Pflicht, max 300 Zeichen | Ein Satz |
| `ergebnis` | str, Pflicht, max 800 Zeichen | Zwei bis drei Sätze |
| `stand` | ISO-Datum, Pflicht | Stand der Würdigung |
| `status` | `aktiv` oder `ueberholt`, Default `aktiv` | Nur `aktiv` wird injiziert und destilliert |
| `pruefung` | Liste `PruefungsPunkt`, max 12 | je `these` (max 300), `bewertung` (max 800), `fundstellen` (Liste von Az-Strings, die auf `fundstellen` verweisen) |
| `fundstellen` | Liste `Fundstelle`, max 25 | je `gericht` (Pflicht), `datum` (ISO, Pflicht), `az` (Pflicht), `art` (`Urteil` oder `Beschluss`, optional), `aussage` (max 400), `richtung` (`pro`, `contra`, `neutral`), `store`, `store_entry_id`, `store_checked_at` (die drei letzten vom Server gesetzt) |
| `risiken` | Liste str, max 10, je max 300 | Kurze Sätze |
| `quelle` | str, max 200 | Dateiname des Vermerks in j-lawyer plus Datum |

Gesamtgröße eines Gutachtens nach JSON-Serialisierung: max 24 kB. Gesamtgröße
`content_json`: max 200 kB. Beides wird bei Create und Accept geprüft (400).

`store` je Fundstelle nimmt einen der Werte `verified`, `date_mismatch`,
`not_in_store`, `unchecked` an. `verified` bedeutet ausschließlich: ein aktiver
Store-Eintrag mit gleichem normalisierten Az existiert und sein
`decision_date` stimmt mit `datum` überein. Gericht und tragende Aussage werden
nicht geprüft. Bei mehreren Store-Treffern genügt ein Treffer mit passendem
Datum, dessen `id` landet in `store_entry_id`.

Die drei Server-Felder werden beim Create eines Proposals aus den Ops entfernt
und beim Accept neu gesetzt. Das erfordert eine Sanitizing-Stufe vor dem
Speichern der `ops_list` in `create_memory_update_proposal`, weil die Funktion
heute das Ergebnis des Probe-Laufs verwirft und die rohen Ops speichert.

Konvention für die schreibende Session: keine Namen von Beteiligten im
Gutachten, sondern Rollen ("der Mandant", "die Freundin"). Die
Pseudonymisierung bleibt Sicherheitsnetz.

### Validierung

- Fremde Felder auf jeder Ebene: 400 (`extra="forbid"`, nicht nur die oberste
  Ebene wie bei `_validate_brief_content`).
- Pflichtfelder fehlen, Aufzählungswerte falsch, Slug ungültig, Längen oder
  Anzahlen überschritten: 400.
- Doppelte `id` innerhalb von `gutachten` nach Anwendung der Ops: 400 beim
  Create und beim Accept.
- Az-Verweise in `pruefung[].fundstellen`, die in `fundstellen` des Gutachtens
  nicht vorkommen: 400.

### Patch-Ops

Adressierung per `id`, nicht per Index. Begründung aus dem Review: Bei
`[A, B, C]` mit einem pending `remove /gutachten/0` und einem pending
`set /gutachten/1` zeigt Index 1 nach dem ersten Accept auf C. Index-Ops sind
bei mehreren unabhängigen Objekten in einer Liste nicht rebasefähig.

| Op | Pfad | Bedeutung |
|---|---|---|
| `append` | `/gutachten/-` | neues Gutachten, `id` darf nicht existieren |
| `set` | `/gutachten/by-id/<id>` | Fortschreibung, `value.id` muss gleich `<id>` sein, `stand` neu, Server-Felder werden zurückgesetzt |
| `remove` | `/gutachten/by-id/<id>` | Entfernen |
| `set` | `/notizen` | Skalar |

400 bei unbekannter `id` in `set` oder `remove`, bei `id`-Abweichung zwischen
Pfad und Wert, und bei Index-Pfaden auf `/gutachten`. Die Auflösung liegt im
Domainmodul, `_apply_patch_ops` delegiert für dieses Target.

### Rebase pending Proposals

Der generische Rebase (`_rebase_pending_proposals`, `_changed_fields`,
`_normalize_rebase_value`) arbeitet feldweise über normalisierte Strings. Für
Gutachten-Objekte ist das die falsche Gleichheit: Das Feld `gutachten` ändert
sich bei jeder Store-Markierung, ein `set` für Gutachten B würde nach einem
Accept für Gutachten A verworfen, und zwei `append` derselben `id` gelten als
verschieden.

Deshalb bekommt das Target eine eigene Rebase-Strategie im Domainmodul, die
der generische Rebase für `case_assessment` aufruft:

- Identität eines Gutachtens ist seine normalisierte `id`.
- Konfliktmenge des Accepts ist die Menge der `id`s, deren Inhalt sich
  geändert hat, verglichen rekursiv und ohne die drei Server-Felder.
- Pending `set`/`remove` auf eine `id` in der Konfliktmenge werden
  `superseded`. Pending Ops auf andere `id`s bleiben.
- Pending `append` mit einer `id`, die inzwischen existiert, wird
  `superseded`, mit Hinweis im `reject_reason`, dass ein `set by-id` nötig ist.
- Nach dem Filtern werden alle verbleibenden Ops eines Proposals gemeinsam
  gegen den neuen Inhalt probeweise angewandt. Scheitert das, wird das ganze
  Proposal `superseded`.

Ein Recheck (Abschnitt 2) rebased pending Proposals nicht, weil er nur
Server-Felder ändert.

### Register

`_target_spec` liefert heute ein Sieben-Tupel, das positionsabhängig entpackt
wird. Vor der Erweiterung wird daraus ein benanntes `TargetSpec`
(`NamedTuple`) mit den Feldern `model`, `source_model`, `source_fk`,
`default_content`, `renderer`, `list_fields`, `scalar_fields`, `validate`,
`apply_ops`, `rebase`. Bestehende Aufrufer werden auf Attributzugriff
umgestellt. Damit entfallen die fest verdrahteten Brief-oder-Strategie-Zweige
in `_target_content`, `_write_target_content`, `_update_target_content`,
`_apply_patch_ops` und `memory_row_to_dict` (das heute jede Nicht-`CaseBrief`-
Zeile als Strategie klassifiziert).

`MemoryTargetType` in `app/shared.py` wird um `case_assessment` erweitert. Die
HTTP-Validierung der Proposal-Route läuft aber über das lokale
`MemoryProposalCreateRequest` in `endpoints/agent_memory.py` mit
`target_type: str`. Dort wird explizit gegen die Registerschlüssel geprüft
(400 bei unbekanntem Target). Der GET-Payload entsteht in `_combined_payload`,
nicht aus den Response-Modellen in `shared.py`. `_combined_payload` bekommt
den dritten Block `case_assessment` mit `version`, `content_json`, `rendered`.
`CaseAssessmentResponse` in `shared.py` wird ergänzt, ist aber nur Dokumentation.

## 2. Store-Abgleich und Beschaffung beim Accept

Einhängepunkt in `accept_memory_update_proposal`: nach
`new_content = _apply_patch_ops(...)` und vor `_create_revision(...)`.
Reihenfolge:

1. Ops anwenden und strukturell validieren.
2. Geänderte Gutachten bestimmen, Vergleich ohne Server-Felder.
3. Mitgelieferte Server-Felder verwerfen.
4. Store einmal laden als Map `normalisiertes Az -> [(entry_id, decision_date)]`
   (heute lädt `store_lookup` bei jedem Aufruf alle aktiven Einträge und
   normalisiert in Python, das darf pro Accept nur einmal passieren). Der
   Normalisierer wird als öffentliche Funktion aus `verify_source`
   herausgezogen und von Accept, Recheck und Wiki-Whitelist gemeinsam benutzt.
5. Für jede Fundstelle der geänderten Gutachten `store`, `store_entry_id`,
   `store_checked_at` setzen.
6. Das angereicherte `new_content` erneut validieren.
7. Genau dieses Objekt in Revision, `content_json` und `search_text` schreiben.
8. Version erhöhen.
9. Pending Proposals per Gutachten-Rebase behandeln.
10. Commit.
11. Erst nach erfolgreichem Commit die Zitatzeilen an `spawn_for_text`
    übergeben, damit ein gescheiterter Accept keinen Hintergrund-Ingest auslöst.

Der Store-Abgleich läuft in einer getrennten Read-Session, damit ein DB-Fehler
darin die Accept-Transaktion nicht unbrauchbar macht. Scheitert er technisch,
bekommen alle Fundstellen der geänderten Gutachten `unchecked`, der Accept
geht durch.

Zitatzeile für die Beschaffung im Format des deterministischen Parsers
(`DECISION_RE` in `draft_citation_ingest`):
`"<Gericht>, <art> vom <TT.MM.JJJJ> – <Az>"`. Fehlt `art`, wird `Beschluss`
gesetzt. Scheitert der Spawn, nur Log.

Rückgabe: `accept_memory_update_proposal` gibt heute nur das Proposal-Objekt
zurück, die Route baut daraus `_proposal_frontend_payload`. Neu: die Funktion
gibt `(proposal, warnings)` zurück, die Route hängt `assessment_warnings`
(Liste `{gutachten_id, az, store}` für alles außer `verified`) an den Payload.
`_proposal_frontend_payload` bekommt für dieses Target `section =
"assessment"`, `title = "Gutachten-Vorschlag"` und Inhaltszeilen aus
`id | rechtsfrage | n Fundstellen` statt Roh-JSON. Das Web-UI (`app.js`) zeigt
`assessment_warnings` nach dem Accept als Liste unter dem Proposal. Mehr UI
nicht in diesem Schritt.

### Recheck

`POST /memory/cases/{case_id}/assessment/recheck`, Owner-geprüft. Lädt den
Store einmal, prüft alle Fundstellen aller Gutachten, auch die bereits als
`verified` markierten (ein deaktivierter oder korrigierter Store-Eintrag darf
nicht dauerhaft zitierfähig bleiben), und schreibt die Server-Felder. Schreibt
eine Revision mit `actor = "recheck"` und `source_refs = []`, erhöht die
Version nicht, setzt `updated_at`, erzeugt kein Proposal, rebased nichts, löst
den bestehenden SSE-Refresh aus. Antwort: `{changed_fundstellen: n,
changed_gutachten: m, warnings: [...]}` mit derselben Warnungsstruktur wie beim
Accept. Ohne Gutachten: 200, Zähler 0, keine Revision.

Die Version unverändert zu lassen ist nur deshalb sicher, weil Proposal-Ops
die Server-Felder nie tragen (Sanitizing beim Create) und der Accept sie immer
neu setzt.

CLI: `rechtmaschine-cli memory assessment recheck [--case-id]`.

Der tägliche `memory-triage.timer` (05:00) ruft nach der Triage den Recheck für
jede eigene Akte mit mindestens einem aktiven Gutachten auf. Fremde Akten
werden wie bei der Triage übersprungen. Was danach noch fehlt, ist Handarbeit
über die Eskalationskette des verify-source-Skills.

## 3. Prompt-Injektion

`get_case_memory_prompt_context` rendert das Gutachten unabhängig von
`include_strategy` als eigenen Block. Reihenfolge: Brief, Strategie (falls
eingeschaltet), Gutachten, Doktrin, Muster-Wiki, Pack. Heute gibt es keinen
Aufrufer mit `include_strategy=False` und keinen mit `max_chars`-Override, die
Regel ist trotzdem Teil des Vertrags.

Budget: Erst Brief plus Strategie auf die bisherigen `max_chars` (5.000)
kürzen. Danach das Gutachten separat auf `max_assessment_chars` (4.000)
rendern. Erst dann beide Blöcke zusammenführen und pseudonymisieren. Das
Gutachten darf das alte Budget nicht anfressen.

```
RECHTLICHE WÜRDIGUNG DER KANZLEI (Fundstellen mit Store-Abgleich, Stand je Gutachten):
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
  Sperrliste bleiben. Die Feldlängen aus Abschnitt 1 garantieren, dass ein
  Gutachten ohne `risiken` und `pruefung` höchstens etwa 1.500 Zeichen
  braucht. Reicht das nicht, fallen ganze Gutachten von hinten weg, mit
  Hinweis `[weitere Gutachten gekürzt: n]`. Nie ein halbes Gutachten.
- `collect["assessment_used"]` und `collect["assessment_ids"]` enthalten nur
  Gutachten, die nach der Kürzung tatsächlich im Prompt stehen. Schlägt die
  Pseudonymisierung fehl, gilt `assessment_used=False`, `assessment_ids=[]`.
- Pseudonymisierung wie Brief und Strategie über
  `pseudonymize_case_text_for_cloud`. Genau: Sie ersetzt nur bekannte
  Anonymisierungsentitäten der Akte. Hat die Akte keine, geht der Text
  unverändert. Nur ein technischer Fehler führt zu leerem Text.
- Das Matching von Doktrin, Muster-Wiki und Pack gegen `base_memory` schließt
  den Gutachten-Block ein.

### Fakten-Grounding

Die Generierung übergibt `case_memory_text` an `_fact_corpus` in
`citation_verifier`, und der Faktenchecker behandelt jedes dort vorkommende Az
als Beleg. Stünde die Sperrliste im selben Text, würde ein Entwurf mit einer
gesperrten Fundstelle fälschlich keine Warnung erhalten. Deshalb liefert
`get_case_memory_prompt_context` über `collect` zusätzlich
`collect["assessment_blocked_az"]` (normalisierte Az der Sperrliste), und
`_fact_corpus` bekommt den Memory-Text ohne die Sperrlistenzeilen. Der
Faktenchecker meldet ein Az aus `assessment_blocked_az` im Entwurf als
Warnung "gesperrte Fundstelle verwendet".

## 4. Schnittstellen und Werkzeuge

### API

- `GET /memory/cases/{id}` liefert zusätzlich `case_assessment` mit `version`,
  `content_json`, `rendered` über `_combined_payload`. `--versions` zeigt es.
- `POST /memory/cases/{id}/proposals` akzeptiert `target_type:
  "case_assessment"`, geprüft gegen das Register. Accept und Reject
  unverändert, Accept-Antwort mit `assessment_warnings`. 409-Ordnungsblocker
  gilt je Target wie heute.
- `PUT /memory/cases/{id}` bleibt auf Brief und Strategie beschränkt.
- Neu nur `POST /memory/cases/{id}/assessment/recheck`.

Autorenschaft: "nur Sessions" ist Konvention, nicht Durchsetzung. Jeder
authentifizierte Nutzer kann ein Assessment-Proposal anlegen, `model` ist frei
wählbar. Der Server setzt beim Create zusätzlich `metadata.origin_user` aus dem
Token, damit die Herkunft nachvollziehbar bleibt. Mehr nicht in diesem Schritt.

### CLI (`scripts/rechtmaschine_cli.py`)

- `_MEMORY_SECTIONS` um `"assessment": "case_assessment"` erweitern.
  `memory get --section assessment`, `--grep`, `--field` arbeiten über die
  bestehende Projektion. `_memory_entries` flacht Gutachten-Objekte zu
  Textzeilen (`rechtsfrage`, `ergebnis`, jede These, jede Fundstelle als
  "Gericht Datum Az Aussage"), damit `--grep` ein Az findet.
- `memory proposals list` zeigt bei Ops auf `/gutachten` `id | rechtsfrage |
  n Fundstellen`.
- `memory proposals accept` druckt `assessment_warnings`.
- `memory assessment recheck [--case-id]`.

### Triage und Stop-Hook

- `memory_triage.py`: Proposals mit `target_type == case_assessment` bekommen
  das Verdikt SESSION, unabhängig vom Modell, geprüft vor `_op_values`, weil
  diese Funktion aus Dicts nur `name` liest und ein Gutachten sonst als FACT
  automatisch angenommen würde. Der Timer ruft zusätzlich den Recheck wie in
  Abschnitt 2.
- `memory_hygiene_hook.py` liest heute ausschließlich `claim-release`-Events
  aus der Activity-DB (SQLite unter `~/.local/state/claims`). Es gibt keine
  Upload-Erkennung im Transkript. Deshalb: `jlawyer-cli upload` schreibt für
  Dateien mit Präfix `Vermerk_` ein Event `kind = "vermerk-upload"` mit
  Akte und Dateiname in dieselbe Activity-DB (gleicher Pfad wie
  `claim-release`). Der Hook prüft beim Stop, ob diese Session seit Baseline
  ein solches Event geschrieben und für dieselbe Akte kein Proposal mit
  `target_type == case_assessment` angelegt hat (Abfrage über die
  Proposal-Liste der Akte, `model` in `claude`/`codex`, `created_at` nach
  Baseline). Dann druckt er `📚 Vermerk ohne Gutachten-Proposal: <Datei>`.

### Skill `rechtmaschine-memory`

Neuer Abschnitt "Gutachten (`case_assessment`)": Schema, Slug-Regel,
Konvention ohne Namen, `by-id`-Pfade, Ablauf (Recherche, Vermerk in j-lawyer,
Proposal, Accept, Warnungen lesen, `cited_ingest` für Fehlende, Recheck),
Beispiel-Payload, Bedeutung von `verified` als reiner Store- und
Datumsabgleich, und der Hinweis, dass Reflect und Consolidate das Target nicht
anfassen.

### Reflect und Consolidate

Fassen `case_assessment` nicht an. Extraktionsschemata, Rolling Fold und
Konsolidierung bleiben auf Brief und Strategie beschränkt.

### Muster-Wiki (`wiki distill`)

`_execute_pattern_wiki_distillation` baut den FALL-SPEICHER neu:

1. Zuerst alle aktiven Gutachten, gerendert mit ausschließlich
   `verified`-Fundstellen im parserfähigen Vollformat
   `"<Gericht>, <art> vom <TT.MM.JJJJ> – <Az>"` plus Richtung und Aussage.
   Nicht "ungekürzt" im Sinne von unbegrenzt: Die Größenlimits aus Abschnitt 1
   begrenzen das Volumen, zusätzlich gilt ein Gesamtlimit von 24.000 Zeichen
   für den Gutachten-Teil, damit `MEMORY_EXTRACTION_NUM_CTX` (32.768) mit
   Brief und Strategie nicht überläuft. Kürzung wie in Abschnitt 3.
2. Danach Brief und Strategie wie bisher.

`_DISTILL_RULES` bekommen: Argumentationsmuster führen die tragende Fundstelle
im Vollformat `"Gericht, Urteil oder Beschluss vom TT.MM.JJJJ – Az"` mit. Es
dürfen nur Fundstellen aus dem FALL-SPEICHER verwendet werden, keine aus dem
Modellwissen.

Whitelist-Prüfung nach der Pydantic-Extraktion, vor `_entry_violations` und
Persistierung:

1. Whitelist = normalisierte Az-Kerne aller `verified`-Fundstellen der
   aktiven Gutachten, mit demselben öffentlichen Normalisierer wie Accept und
   Recheck.
2. Der Parser in `draft_citation_ingest` bekommt eine Variante, die Rohtext
   und Start-End-Positionen je Treffer liefert.
3. Alle Textfelder des Eintrags (`summary` und sämtliche Pattern-Listen)
   werden geparst. Fremde Az: die gesamte Fundstellenphrase wird per Span
   entfernt, danach Klammern und Satzzeichen bereinigt.
4. Zusätzlich werden Az-artige Tokens außerhalb des Vollformats geprüft
   (Regex auf `\d{1,3}\s+[A-Z]{1,3}\s+\d+[./]\d+`), damit ein nacktes Az die
   Whitelist nicht umgeht. Fremde Tokens werden entfernt.
5. Entfernte Fundstellen werden als `stripped_citations`
   (`{entry_title, az, citation}`) im Job-Ergebnis und gespiegelt in
   `warnings` gemeldet.
6. `_forbidden_tokens` (PII-Gate) und die Whitelist benutzen denselben Parser
   und Normalisierer. Erlaubte Fundstellen-Daten und Az aus den Gutachten sind
   keine verbotenen Tokens.
7. `PatternWikiSource` bekommt bei Assessment als Primärquelle
   `source_type = "case_assessment"` und die Gutachten-`id`s im Label, statt
   heute pauschal `case_brief` mit "Brief+Strategie".

Ohne aktives Gutachten läuft der Distill wie heute, ohne Whitelist-Prüfung.

## 5. Fehlerfälle

| Fall | Verhalten |
|---|---|
| Ungültiges Gutachten-Objekt, Längen, Anzahlen, Slug | 400 beim Create |
| Doppelte Gutachten-`id` | 400 beim Create und beim Accept |
| `set`/`remove by-id` mit unbekannter `id`, `id`-Abweichung, Index-Pfad | 400 |
| Store-Abgleich technisch gescheitert | Accept geht durch, `unchecked`, Warnung, Accept-Transaktion bleibt intakt (getrennte Read-Session) |
| `spawn_for_text` scheitert | nur Log, nach Commit |
| Recheck auf Akte ohne Gutachten | 200, Zähler 0, keine Revision |
| Rendering über Budget | Kürzungsstufen, nie ein halbes Gutachten |
| Distill mit fremden Az | Az samt Phrase entfernt, `stripped_citations` und `warnings`, Eintrag landet als pending |
| Pseudonymisierung technisch gescheitert | ganzer Memory-Text fällt aus dem Cloud-Prompt, `assessment_used=False` |
| Paralleler Erstzugriff auf das Target | `_get_or_create_target` fängt den Unique-Index-Konflikt und liest erneut |

## 6. Tests

Unter `tests/`, im Stil der bestehenden Memory-Tests. Die bestehenden Tests,
die `models` und `shared` stubben (`test_memory_rebase_changed_fields.py`),
werden um die Assessment-Klassen ergänzt, sonst scheitert der Import.

- `test_memory_assessment_model.py`: Validierung inklusive `extra="forbid"`
  auf allen Ebenen, Slug, Längen, Anzahlen, Default, `by-id`-Ops, Index-Pfad
  400, fremde Felder, doppelte `id`, Verweis auf unbekanntes Az, Server-Felder
  werden beim Create entfernt.
- `test_memory_assessment_rebase.py`: `remove` plus `set` auf verschiedene
  `id`s bleiben beide gültig, `set` auf geänderte `id` wird superseded,
  `append` mit existierender `id` wird superseded, gemeinsame Probeanwendung,
  Recheck rebased nichts.
- `test_memory_assessment_accept.py`: Store-Map gegen eine Test-DB mit zwei
  Entscheidungen setzt `verified`, `date_mismatch`, `not_in_store`,
  `store_entry_id`; mehrfaches Az mit einem passenden Datum ist `verified`;
  `assessment_warnings`; `spawn_for_text` als Mock erst nach Commit mit
  erwarteter Zitatzeile; technischer Fehler ergibt `unchecked` und der Accept
  committet trotzdem.
- `test_memory_assessment_render.py`: Block-Format, Budgetstufen, Reihenfolge
  ohne Strategie, altes Budget unangetastet, nur aktive Gutachten, Sortierung,
  Sperrliste, `assessment_ids` nach Kürzung, Pseudonymisierung ersetzt einen
  Mandantennamen, Fehlerfall setzt `assessment_used=False`,
  `assessment_blocked_az` gefüllt.
- `test_memory_assessment_facts.py`: `_fact_corpus` ohne Sperrliste, Warnung
  bei gesperrtem Az im Entwurf.
- `test_memory_assessment_recheck.py`: Felder neu gesetzt auch bei vorher
  `verified`, Revision geschrieben, Version unverändert, `updated_at` gesetzt,
  leere Akte ohne Revision, pending Proposal bleibt unberührt.
- `test_memory_get_projection.py` und `test_memory_combined_payload.py`
  erweitert: drittes Target im GET, `--section assessment`, `--grep` trifft
  ein Az.
- `test_pattern_wiki_distill_assessment.py`: FALL-SPEICHER beginnt mit den
  Gutachten im Vollformat, fremde Az im Vollformat und als nacktes Token werden
  entfernt und gemeldet, erlaubte Az überleben das PII-Gate, Provenienz
  `case_assessment`, ohne Gutachten kein Whitelist-Eingriff.
- Migration: echter PostgreSQL-Smoke-Test gegen eine leere Datenbank
  (ORM legt an, Migration ist No-op) und gegen eine Datenbank mit angewandter
  `2026-04-28_case_memory_mvp` (Migration legt an). Läuft gegen den
  Test-Container, nicht als Source-Text-Prüfung.
- Integration: zwei parallele Erstzugriffe auf das Target derselben Akte.

## 7. Abnahme

Deploy über den Container wie üblich. Live-Test mit 157/26: die beiden
Gutachten vom 03.09.2026 (GÜB statt Duldung, Heirat und § 39 Nr. 5 AufenthV)
als Proposal schreiben, annehmen, Warnungen lesen, für fehlende Fundstellen
`cited_ingest`, Recheck, danach eine Test-Generierung mit Provenienzansicht,
in der `assessment_used` gesetzt ist und die verifizierten Az im Prompt
stehen, und ein Entwurf mit einem gesperrten Az die Faktenwarnung auslöst.
Anschließend `wiki distill` auf 157/26 und prüfen, dass die
Argumentationsmuster Fundstellen im Vollformat tragen und die Provenienz
`case_assessment` heißt.

## Nicht in diesem Schritt

- Qwen-Vorschläge für Gutachten aus Vermerken.
- Qwen-Aussageprüfung (`verify_claim`) beim Accept.
- Serverseitige Durchsetzung der Session-Autorenschaft.
- Rückwirkendes Anlegen von Gutachten für alte Akten.
- Änderung des Budgets von Brief plus Strategie.
- Nummerierte Revisionen.
- Web-UI über die Anzeige der Accept-Warnungen hinaus.

## Hinweis für die Umsetzung

Im Repo liegen uncommittete Änderungen anderer Sessions in
`app/agent_memory_service.py`, `app/endpoints/agent_memory.py`,
`app/static/js/app.js` und `scripts/rechtmaschine_cli.py`. Vor dem Bau
`claim check` auf die Tooling-Schlüssel und mit den Haltern abstimmen, in
einem eigenen Worktree arbeiten und auf den dann committeten Stand rebasen.

Kurzurteil: Die Grundrichtung passt, die Spec ist aber noch nicht implementierungsreif. Die größten Risiken sind der feldweise Rebase einer Objektliste, indexbasierte Änderungen, die Zitatformat-Kollision beim Wiki und die Vermischung nicht zitierfähiger Fundstellen mit dem Fakten-Grounding.

## 1. Passt die Spec zum tatsächlichen Code?

Nur teilweise. Folgende Annahmen stimmen nicht oder sind unvollständig:

- Es gibt mehr als die drei erwähnten fest verdrahteten Brief-oder-Strategie-Stellen. Neben `_target_content` und `_apply_patch_ops` müssen mindestens `_write_target_content`, `_update_target_content` und `memory_row_to_dict` angepasst werden. Letzteres klassifiziert jedes Nicht-`CaseBrief` derzeit automatisch als Strategie. Siehe [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:202), [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:226), [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:259), [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:444) und [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:1207).

- `_target_spec` ist kein echtes Register, sondern ein Sieben-Tupel, das an mehreren Stellen positionsabhängig entpackt wird. Eine zusätzliche Validierungsfunktion verändert die Tupellänge und damit alle Aufrufer. Ich würde daraus jetzt eine benannte `TargetSpec`-Struktur machen. Sonst wird diese Erweiterung unnötig fehleranfällig.

- Die tatsächliche Proposal-Route verwendet nicht `MemoryUpdateProposalCreateRequest` aus `app/shared.py`. Sie verwendet das lokale `MemoryProposalCreateRequest`, dessen `target_type` nur `str` ist. Eine Erweiterung des `MemoryTargetType` in `shared.py` allein ändert die HTTP-Validierung daher nicht. Siehe [agent_memory.py](/var/opt/docker/rechtmaschine/app/endpoints/agent_memory.py:130) und [shared.py](/var/opt/docker/rechtmaschine/app/shared.py:1560).

- Auch die Response-Modelle aus `shared.py` steuern den GET-Payload nicht. `/memory/cases/{id}` baut eine freie `JSONResponse` über `_combined_payload`. `CaseAssessmentResponse` allein bewirkt dort nichts. Siehe [agent_memory.py](/var/opt/docker/rechtmaschine/app/endpoints/agent_memory.py:1829).

- Die Accept-Funktion gibt aktuell nur das Proposal-ORM-Objekt zurück. Die Route schiebt es anschließend durch `_proposal_frontend_payload`. Für `assessment_warnings` braucht es daher einen neuen Rückgabevertrag oder eine explizite zusätzliche Payload-Bildung. Siehe [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:861) und [agent_memory.py](/var/opt/docker/rechtmaschine/app/endpoints/agent_memory.py:2128).

- `_proposal_frontend_payload` behandelt jedes Nicht-Strategie-Target als „overview“. Assessment-Proposals würden im Web-UI falsch betitelt und als unhandliches Roh-JSON dargestellt. Auch Accept-Warnungen werden vom Frontend nicht angezeigt. Siehe [agent_memory.py](/var/opt/docker/rechtmaschine/app/endpoints/agent_memory.py:1850) und [app.js](/var/opt/docker/rechtmaschine/app/static/js/app.js:1540).

- „Nur Sessions als Autor“ ist technisch nicht durchgesetzt. Jeder authentifizierte Benutzer kann ein Assessment-Proposal anlegen. `model` ist frei wählbar und damit kein belastbarer Autoritätsnachweis. Wenn dies mehr als eine Arbeitskonvention sein soll, fehlt ein serverseitig gesetztes Herkunftsfeld oder ein gesonderter API-Pfad.

- Fremde verschachtelte Felder werden durch normale Pydantic-Modelle standardmäßig ignoriert. Die manuelle Prüfung in `_validate_*_content` kontrolliert nur die oberste Ebene. `GutachtenEntry`, `Fundstelle` und `PruefungsPunkt` brauchen jeweils `extra="forbid"`.

- Das Create-Dry-Run-Ergebnis wird verworfen. `_apply_patch_ops` validiert zwar das resultierende Dokument, anschließend werden aber die ursprünglichen `ops_list` gespeichert. Ein in den Ops mitgeliefertes `store="verified"` wird daher nicht automatisch als `unchecked` persistiert. Dafür braucht es eine echte Sanitizing-Stufe vor dem Speichern des Proposals. Siehe [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:588).

- Falls `stand` und `datum` als `datetime.date` modelliert werden, liefert das vorhandene `_model_dump()` Python-Datumsobjekte. Diese sind nicht ohne Weiteres JSONB-serialisierbar. Entweder bleiben die Felder streng validierte ISO-Strings oder das Dumping muss im JSON-Modus erfolgen. Siehe [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:51).

- Versionen und Revisionen funktionieren anders als die Spec möglicherweise unterstellt. Die Target-Version beginnt bei 1 und wird bei Accept oder PUT um 1 erhöht. `CaseMemoryRevision` besitzt dagegen überhaupt kein `revision_number`. `_create_revision` übergibt zwar `revision_number`, `change_type`, `summary` und weitere Felder, `_new_model` verwirft sie aber, weil die ORM-Tabelle diese Spalten nicht hat. Revisionen werden also lediglich unnummeriert angehängt. Siehe [models.py](/var/opt/docker/rechtmaschine/app/models.py:676) und [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:369).

- Ein Recheck mit Revision ohne Versionssprung kollidiert deshalb mit keinem Unique Constraint. Er bedeutet aber, dass sich `content_json` trotz unveränderter Optimistic-Lock-Version ändert. Das ist nur sicher, wenn alle Proposal-Ops servergesetzte Store-Felder ignorieren.

- Die Spec-Annahme zum Hygiene-Hook ist falsch. `memory_hygiene_hook.py` erkennt heute keine Upload-Zeilen im Transkript. Er liest ausschließlich `claim-release`-Events aus einer SQLite-Datenbank. Die neue Erkennung ist daher ein neues Subsystem, keine „dritte Erinnerung“ auf bestehender Upload-Erkennung.

- Einen bestehenden „Migrations-Testpfad“ für Case-Memory gibt es nicht. Die vorhandenen Memory-Tests sind überwiegend eigenständige Ad-hoc-Skripte und teilweise reine Source-Text-Prüfungen.

## 2. Funktioniert der Rebase mit Gutachten-Objekten?

Technisch meistens ja, semantisch nein.

`_normalize_rebase_value` serialisiert Dicts ohne `name` oder `label` mit `json.dumps(sort_keys=True)`. Verschachtelte Listen und Dicts sind damit verarbeitbar, solange sie JSON-kompatibel sind. Die Reihenfolge verschachtelter Listen bleibt allerdings relevant.

Für Gutachten ist diese Gleichheitsdefinition falsch:

- Dasselbe Gutachten mit `store="unchecked"` und `store="verified"` gilt als verschieden.
- Dasselbe `id` mit aktualisiertem `stand` gilt als völlig anderes Objekt.
- Ein zweites `append` mit derselben `id` wird nicht als Duplikat erkannt, wenn irgendein Detail abweicht. Es bleibt bis zur späteren Duplicate-ID-Validierung erhalten und scheitert dann beim Accept.
- `_changed_fields` sieht jede Änderung eines Store-Feldes als Änderung des gesamten Feldes `gutachten`.

Benötigt wird eine target- und pfadabhängige Rebase-Strategie:

- Identität eines Gutachtens ist ausschließlich seine normalisierte `id`.
- Inhaltsvergleich erfolgt rekursiv und ohne servergesetzte Felder wie `store`.
- Ein `append` mit bereits vorhandener ID ist kein normales Duplikat. Es ist entweder ein Update-Konflikt, ein id-basiertes `set` oder muss sichtbar superseded werden.
- Nach dem Filtern müssen alle behaltenen Ops gemeinsam gegen den neuen Inhalt angewandt werden. Der aktuelle Rebase prüft jede Op isoliert. Mehrere einzeln gültige Index-Ops können zusammen trotzdem ungültig sein.

Auch [memory_triage.py](/home/jay/.codex/skills/rechtmaschine/scripts/memory_triage.py:174) versteht Gutachten-Dicts nicht. `_op_values` extrahiert aus Dicts nur `name`. Ohne die explizite Assessment-Sonderregel würde ein solches Proposal regelmäßig als `FACT` automatisch akzeptiert.

## 3. Wo gehört der Store-Check in den Accept?

Exakt nach:

```text
new_content = _apply_patch_ops(...)
```

und vor:

```text
_create_revision(...)
```

also derzeit zwischen Zeile 902 und 904 in [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:901).

Die Reihenfolge sollte sein:

1. Ops anwenden und strukturell validieren.
2. Tatsächlich geänderte Gutachten bestimmen, wobei `store` beim Vergleich ignoriert wird.
3. Mitgelieferte Store-Felder verwerfen.
4. Fundstellen prüfen und Store-Felder in `new_content` setzen.
5. Das serverangereicherte `new_content` erneut validieren.
6. Genau dieses Objekt in Revision, `content_json` und `search_text` schreiben.
7. Version erhöhen.
8. Pending-Geschwister rebasen.
9. Committen.
10. Erst nach erfolgreichem Commit `spawn_for_text` auslösen.

Der Spawn sollte nicht vor dem Commit erfolgen. Sonst kann ein später gescheiterter Accept bereits einen irreversiblen Hintergrund-Ingest gestartet haben.

Die Nebenwirkung auf Geschwister ist erheblich. `_changed_fields` meldet nur das Top-Level-Feld `gutachten`. `_rebase_pending_proposals` verwirft anschließend jedes pending `set` auf diesem Feld, auch wenn es ein völlig anderes Gutachten betrifft. Store-Felder sind nicht die eigentliche Ursache, sie verschärfen das Problem nur. Das aktuelle feldweise Konfliktmodell passt nicht zu mehreren unabhängigen Objekten in einer Liste.

Für Assessment braucht der Rebase deshalb Konfliktmengen auf Gutachten-ID-Ebene. Ein Recheck sollte pending Proposals nicht mit `curated_fields={"gutachten"}` rebasen. Er ändert nur Servermetadaten und lässt die Version absichtlich unverändert.

## 4. Ist `set /gutachten/<index>` robust?

Nein. Die stabile `id` wird im Schema eingeführt und anschließend beim Schreiben nicht genutzt. Das ist der falsche Schnitt.

Beispiel:

```text
Ausgang: [A, B, C]
Proposal 1: remove /gutachten/0
Proposal 2: set /gutachten/1 auf aktualisiertes B
```

Nach Proposal 1 bezeichnet Index 1 plötzlich C. Der aktuelle feldweise Rebase verwirft Proposal 2 wahrscheinlich komplett. Ohne diesen groben Schutz würde es C überschreiben.

Ich empfehle von Anfang an:

```text
append /gutachten/-
set /gutachten/by-id/gueb-statt-duldung
remove /gutachten/by-id/gueb-statt-duldung
```

Der Aufwand ist überschaubar:

- Target-spezifische Pfadauflösung in `_apply_patch_ops`
- Eindeutige Suche nach `id`
- 400 bei fehlender oder mehrfacher ID
- Prüfung, dass `value.id` zur Pfad-ID passt
- entsprechende Rebase- und Testspezifikation

Da das Target noch nicht produktiv existiert, gibt es keinen Grund, jetzt technische Index-Schulden einzubauen. Zusätzlich sollte `id` als kanonischer Kleinbuchstaben-Slug validiert werden.

## 5. Prompt-Injektion und Budgets

Aktuell gibt es keinen Aufrufer mit `include_strategy=False`. Generation und Query übergeben ausdrücklich `True`, die Workflow-Aufrufer verwenden den Default. Es gibt auch keinen Aufrufer, der `max_chars` überschreibt. Die Prämisse der Frage ist also derzeit nur eine API-Vertragsfrage, kein realer Call-Site-Fall.

Das Assessment sollte unabhängig von `include_strategy` sein:

```text
Brief
Strategie, falls eingeschaltet
Assessment
Doktrin
Muster-Wiki
Rechtsprechungs-Pack
```

Ohne Strategie steht es direkt nach dem Brief.

Wichtig ist die Budgetreihenfolge. Erst Brief und Strategie zusammen auf die bisherigen 5.000 Zeichen begrenzen. Danach das Assessment separat auf 4.000 Zeichen rendern. Erst anschließend beide Blöcke zusammenführen und pseudonymisieren. Wenn das Assessment vor der vorhandenen `max_chars`-Kürzung angehängt wird, frisst es entgegen der Spec das alte Budget.

Weitere Lücken:

- „Rechtsfrage, Ergebnis und Sperrliste bleiben immer“ ist bei beliebig langen Feldern nicht mit einem harten 4.000-Zeichen-Budget vereinbar. Es fehlen Feldlängen und ein maximales Gesamtvolumen je Gutachten.
- `assessment_ids` darf nur IDs enthalten, die nach Kürzung tatsächlich im Prompt stehen.
- Bei fehlgeschlagener Pseudonymisierung müssen `assessment_used=False` und `assessment_ids=[]` gelten. Das vorhandene `case_memory_used` wird schon vor der Pseudonymisierung gesetzt und kann heute trotz leerem Ergebnis wahr bleiben.
- „Fail-closed wie heute“ ist zu stark formuliert. Hat die Akte keine bekannten Anonymisierungsentitäten, gibt `pseudonymize_case_text_for_cloud` den Text unverändert zurück. Nur technische Fehler führen zu leerem Text. Siehe [agent_memory_service.py](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:1067).

Besonders kritisch: Nicht verifizierte Az sollen in der Sperrliste des `case_memory_text` stehen. Der nachgelagerte Faktenchecker behandelt aber jedes dort vorkommende Az als Beleg. Damit würde ein Entwurf mit einer verbotenen Fundstelle fälschlich keine Faktenwarnung erhalten. Siehe [generation.py](/var/opt/docker/rechtmaschine/app/endpoints/generation.py:230) und [citation_verifier.py](/var/opt/docker/rechtmaschine/app/citation_verifier.py:182).

Dafür braucht es einen zweiten, zitierfähigen Grounding-Text ohne gesperrte Az oder eine strukturierte Ausschlussliste für den Faktenchecker.

## 6. Wie sollte die Wiki-Whitelist funktionieren?

Zuerst muss die Formatkollision behoben werden. `parse_decision_citations` erkennt nur:

```text
Gericht, Urteil oder Beschluss vom TT.MM.JJJJ – Az
```

Die Spec verlangt für den Wiki-Eintrag dagegen:

```text
Gericht, Datum, Az
```

Damit findet der Parser genau null Treffer. Siehe [draft_citation_ingest.py](/var/opt/docker/rechtmaschine/app/draft_citation_ingest.py:32).

Empfohlener Ablauf:

1. Aus aktiven Gutachten eine Menge aller `verified`-Az bilden.
2. Dafür einen gemeinsamen öffentlichen Az-Normalisierer verwenden. Nicht zwei private Helfer mit leicht unterschiedlicher Semantik koppeln.
3. Das Modell zwingend das bereits parserfähige Vollformat ausgeben lassen.
4. Nach der Pydantic-Extraktion alle Textfelder eines Eintrags prüfen, insbesondere `summary` und sämtliche Pattern-Listen.
5. Für jede erkannte Fundstelle den normalisierten Az-Kern mit der Whitelist vergleichen.
6. Bei Fremd-Az die gesamte Fundstellenphrase entfernen und anschließend Klammern und Satzzeichen bereinigen.
7. Zusätzlich jedes Az-artige Token prüfen, das der strenge Zitatparser nicht erfasst hat. Sonst kann ein halluziniertes nacktes Az die Whitelist umgehen.
8. Entfernte Fundstellen als `{entry_title, az, citation}` sammeln.
9. Erst danach `_entry_violations` und Persistierung ausführen.

Der vorhandene Parser liefert keine Match-Spans. Für sauberes Entfernen sollte seine interne Iteration optional Raw-Text und Start-End-Positionen zurückgeben. Bloß das Az zu ersetzen hinterlässt kaputte Sätze.

Es gibt außerdem eine Kollision mit `_forbidden_tokens`. Wenn Assessment-Inhalte dort vollständig einbezogen werden, werden auch erlaubte Fundstellen-Daten und Az zu verbotenen Tokens. Die bestehende Ausnahme erkennt wiederum nur ein bestimmtes Vollzitatformat. Whitelist und PII-Gate müssen deshalb denselben Parser und denselben kanonischen Normalisierer benutzen.

Mit dem Job-Result-Format kollidiert `stripped_citations` nicht. `MemoryReflectionJob.result_payload` ist freies JSONB und der Status-Endpunkt gibt es unverändert unter `result` zurück. Ich würde `stripped_citations` zusätzlich in `warnings` spiegeln, damit bestehende Clients es sichtbar machen.

Die erzeugte `PatternWikiSource` bezeichnet heute immer `source_type="case_brief"` und nennt „Brief+Strategie“. Bei Assessment als Primärquelle ist diese Provenienz falsch und muss mit angepasst werden. Siehe [pattern_wiki.py](/var/opt/docker/rechtmaschine/app/endpoints/pattern_wiki.py:296).

## 7. Reicht der Migrationsmechanismus?

Mechanisch ja, als vollständige Migrationsspezifikation noch nicht.

`apply_schema_migrations` führt zuerst `Base.metadata.create_all()` und danach die noch nicht registrierten SQL-Migrationen unter einem Advisory Lock aus. Auf einem frischen System werden die Tabellen daher bereits durch das ORM angelegt, bevor das `CREATE TABLE IF NOT EXISTS` der Migration läuft. Siehe [main.py](/var/opt/docker/rechtmaschine/app/main.py:1130).

Folgen:

- Defaults müssen auch im ORM korrekt definiert sein. Ein Default nur im Migration-`CREATE` wird auf frischen Datenbanken möglicherweise nie angewandt.
- Ich würde `content_json` als `NOT NULL` mit einem echten serverseitigen JSONB-Default definieren.
- Benötigt werden der Unique-Index auf `(owner_id, case_id)`, Indizes auf `owner_id`, `case_id`, `updated_at` und die entsprechenden Source-Indizes.
- `case_assessment_sources.case_assessment_id` braucht den Cascade-Fremdschlüssel.
- Ein GIN-Index auf `content_json` ist nicht nötig, solange Rechecks Assessment-Zeilen nach Owner laden und im Python-Code prüfen.
- Falls der tägliche Lauf per JSONB-Ausdruck nach nicht verifizierten Fundstellen sucht, sollte genau dieser Ausdruck oder ein separates Statusfeld indexiert werden.

Ein bloßer Source-Test der Migrationsliste reicht nicht. Es braucht mindestens einen echten PostgreSQL-Smoke-Test für eine leere DB und einen Upgrade-Test mit bereits angewandter `2026-04-28_case_memory_mvp`-Migration.

## 8. Was fehlt noch?

Meine wichtigsten zusätzlichen Empfehlungen:

- `store_lookup` lädt bei jedem Aufruf alle aktiven Rechtsprechungseinträge und normalisiert sie in Python. Bei `Gutachten × Fundstellen × Akten` wird der tägliche Recheck unnötig teuer. Pro Accept oder Recheck den Store einmal laden und als normalisierte Map verwenden. Siehe [verify_source.py](/var/opt/docker/rechtmaschine/app/verify_source.py:53).

- `verified` bedeutet nach dieser Spec nur „Az und Datum passen zu einem Store-Eintrag“. Es bestätigt weder Gericht noch tragende Aussage. Der Prompt-Titel „geprüft“ überverkauft das Ergebnis. Mindestens die Semantik muss ausdrücklich „Store- und Datumsabgleich“ heißen.

- Ein einmal als `verified` markierter Eintrag wird vom geplanten Timer nie erneut geprüft. Wird die Store-Entscheidung deaktiviert oder korrigiert, bleibt sie dauerhaft zitierfähig. Entweder täglich alle aktiven Gutachten rechecken oder `store_checked_at` und eine TTL aufnehmen.

- Bei mehreren Store-Treffern mit demselben Az muss definiert werden, ob ein einziger passender Datumswert für `verified` genügt. Das sollte meines Erachtens gelten. Der passende Store-Entry sollte zusätzlich protokolliert werden.

- Ein DB-Fehler im Store-Check kann die laufende SQLAlchemy-Transaktion unbrauchbar machen. Ein pauschales `except` garantiert dann keinen erfolgreichen Accept. Verifikation sollte entweder über eine getrennte Read-Session oder über einen klar begrenzten Savepoint laufen.

- Recheck muss definieren, ob `changed` Fundstellen oder Gutachten zählt, ob `updated_at` gesetzt wird und ob ein SSE-Refresh ausgelöst wird.

- Assessment-Inhalte haben keine Größenlimits. Das gefährdet DB-Größe, CLI-Payload, die ungekürzte Wiki-Destillation und den Qwen-Kontext. Notwendig sind Limits pro String, Fundstellenzahl, Prüfungspunkte und Gesamt-JSON.

- Das Wiki liest laut Spec Gutachten „in voller Länge“. Zusammen mit Brief, Strategie und `MEMORY_EXTRACTION_NUM_CTX=32768` kann das den Kontext sprengen. „Ungekürzt“ braucht trotzdem eine validierte maximale Speichergröße.

- `_get_or_create_target` hat beim erstmaligen parallelen Zugriff ein mögliches Unique-Index-Rennen. Das dritte Target erhöht die Zahl solcher Lazy-Creations. Ein Integrationstest mit zwei parallelen ersten Zugriffen wäre sinnvoll.

- Die bestehenden Tests, die `models` und `shared` stubben, müssen um Assessment-Klassen ergänzt werden. Sonst scheitert bereits der Import. Besonders [test_memory_rebase_changed_fields.py](/var/opt/docker/rechtmaschine/tests/test_memory_rebase_changed_fields.py:42) ist betroffen.

- `test_memory_combined_payload.py` muss das dritte Target ausdrücklich prüfen. Die Spec nennt diesen Test nicht.

- Es fehlen Tests für paralleles Remove plus Set, Store-only-Recheck bei pending Proposal, zwei Gutachten mit verschiedenen IDs, mehrfaches Store-Az, zurückgezogenes Store-Urteil, unparsebare Wiki-Az, übergroße Pflichtblöcke und die Trennung zwischen Prompt-Sperrliste und Fakten-Grounding.

Mein bevorzugter Schnitt wäre ein kleines Assessment-Domainmodul für Validierung, Store-Abgleich, ID-Auflösung, Rendering und Recheck. Die generische Memory-Schicht sollte nur Persistierung, Revision, Version und Proposal-Lebenszyklus behalten. Das Gutachten ist fachlich zu speziell für das heutige generische, feldweise Stringlisten-Modell.

Kein Code wurde verändert. Den operativen Abgleich zu Session-Autorenschaft, Triage und Ordnungsblocker habe ich zusätzlich am [rechtmaschine-memory-Skill](/home/jay/.codex/skills/rechtmaschine-memory/SKILL.md:274) ausgerichtet.

# Bugfix-Sweep 2026-07-07 — Plan

Konsolidierte Fixes aus fünf Subsystem-Audits + Commit-Review-Workflow.
Ausführung subagent-getrieben, sequentiell, Commits direkt auf master.

## Global Constraints

- Alle UI-/Fehlertexte auf Deutsch. Keine Semikolons in deutschen Texten.
- Schema-Änderungen: ORM in `app/models.py` ändern UND neuen benannten, idempotenten
  Eintrag an `MIGRATIONS` in `app/main.py` anhängen (`IF NOT EXISTS`/`IF EXISTS`).
  Kein Alembic.
- Es gibt keine pytest-Suite. Verifikation pro Task: `python3 -c "import ast; ast.parse(open(F).read())"`
  für jede geänderte Python-Datei, plus wo machbar ein fokussiertes Ad-hoc-Testskript
  unter `tests/` (Header-Kommentar: was es prüft, wie man es aufruft). KEINE Container
  neu starten, KEINE laufende Produktions-DB mutieren — Live-E2E macht der Controller
  an Checkpoints. `app/__pycache__` ist nicht beschreibbar: py_compile vermeiden, ast.parse nutzen.
- Commit pro Task auf master, deutsche Commit-Message im Repo-Stil, letzte Zeile:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Nur die eigenen Dateien stagen (git add <files>), nie `git add -A`.
- Code-Stil: bestehende Muster der jeweiligen Datei übernehmen (print-Logging mit
  Präfixen wie `[MEMORY WARN]`, HTTPException mit deutschen detail-Texten, etc.).
- Broadcast nach Mutationen: `broadcast_documents_snapshot`/`broadcast_sources_snapshot`
  aus `app/shared.py` (bestehendes Muster beibehalten).

## Task 1: Privacy-Gate OCR vs. Anonymisierung

**Problem (Audit):** `app/endpoints/ocr.py:316-326` — der manuelle OCR-Endpoint
(`run_document_ocr`) ersetzt `document.anonymization_metadata` durch ein frisches Dict
mit nur OCR-Statistiken. Dadurch verschwindet `anonymized_text_path`, aber
`document.is_anonymized` bleibt `True`. `get_document_for_upload` in
`app/shared.py:534-548` fällt dann still auf den rohen Text unter
`extracted_text_path` zurück — unanonymisierter Mandantentext geht an Cloud-LLMs.

**Fix:**
1. In `run_document_ocr`: OCR-Statistiken in das BESTEHENDE
   `anonymization_metadata`-Dict mergen (Kopie erstellen, Keys ergänzen), nicht ersetzen.
2. Da der Text sich durch OCR geändert hat: `is_anonymized = False` setzen,
   `anonymized_text_path`-Eintrag aus dem Metadata-Dict entfernen und die alte
   anonymisierte Textdatei löschen (Pfad vorher merken). Danach die automatische
   Re-Anonymisierung anstoßen, exakt wie es die Auto-Pipeline nach Klassifikation tut
   (Aufrufmuster in `app/endpoints/classification.py` suchen: dort wird nach OCR für
   Auto-Anon-Kategorien die Anonymisierung gescheduled — dasselbe Muster verwenden,
   nur für Kategorien, die Auto-Anonymisierung haben).
3. Hart absichern in `app/shared.py` `get_document_for_upload`: Wenn
   `is_anonymized` True ist, aber die anonymisierte Textdatei fehlt/nicht lesbar ist,
   NICHT auf Rohtext zurückfallen, sondern eine Exception mit deutschem Fehlertext
   werfen ("Anonymisierte Fassung fehlt — Dokument bitte neu anonymisieren"). Alle
   Aufrufer müssen den Fehler sauber als Job-/HTTP-Fehler durchreichen (prüfen, nicht
   verschlucken).

**Dateien:** app/endpoints/ocr.py, app/shared.py, ggf. app/endpoints/classification.py (nur lesen für Muster).

## Task 2: Anonymisierung — Null-Ersetzungs-Gate

**Problem (Audit):** `app/endpoints/anonymization.py:1387-1390, 1332-1336, 1944-1966` —
liefert der Flair-Service eine degenerierte 200-Antwort (leeres/fehlendes
`anonymized_text`, keine Entities), wird der unveränderte Originaltext als anonymisiert
gespeichert, `is_anonymized=True`, `processing_status='completed'`, Confidence-Default
0.95. Es gibt keinerlei Mindest-Redaktions-Check.

**Fix (Entscheidung Jay: hart failen):**
1. Nach Abschluss der Pipeline in `anonymize_document_record` (bzw. der Stelle, die
   `is_anonymized=True` setzt): Wenn (a) der anonymisierte Text leer ist ODER (b) der
   anonymisierte Text byte-identisch mit dem Eingabetext ist ODER (c) alle
   Entity-/Ersetzungslisten leer sind → Status `anon_failed` mit deutschem Fehlertext
   ("Anonymisierung hat keine Ersetzungen vorgenommen — Ergebnis verworfen, bitte
   prüfen und erneut anonymisieren"), `is_anonymized` bleibt False, keine anonymisierte
   Datei als gültig markieren.
2. Confidence-Default bei unparsebarer Service-Antwort: 0.0 statt 0.95.
3. Der bestehende Fehlerpfad (`anon_failed` + Broadcast) existiert schon für
   Exceptions — denselben Pfad für dieses Gate nutzen.

**Dateien:** app/endpoints/anonymization.py.

## Task 3: documents.py — /reset-Scoping und Datei-Aufräumung

**Problem (Audit):**
- `app/endpoints/documents.py:607` — `/reset` löscht `ResearchSource`-Zeilen nur nach
  `owner_id`, ohne `case_id`-Filter. Reset eines Falls löscht Research-Quellen ALLER
  Fälle des Users (alle anderen Deletes im Endpoint sind auf `active_case_id` gescoped).
- `delete_document` (`documents.py:430-471`) und `/reset` löschen nur
  `extracted_text_path` (über `delete_document_text`), nie die anonymisierte Datei
  (Pfad in `anonymization_metadata['anonymized_text_path']`) und nie
  Übersetzungsdateien (`document_translations`-Zeilen kaskadieren in der DB, aber die
  Dateien unter `translated_text/` bleiben liegen).
- `/documents/from-url` (`documents.py:513-519`): schlägt `shutil.move` UND das
  Fallback-`shutil.copy` fehl, bleibt der Temp-Download in `DOWNLOADS_DIR` liegen.

**Fix:**
1. `/reset`: `case_id`-Filter beim ResearchSource-Delete ergänzen (identisch zu den
   übrigen Deletes im selben Endpoint).
2. Hilfsfunktion (in documents.py oder shared.py, je nach bestehendem Muster), die zu
   einem Document alle zugehörigen Dateien einsammelt und löscht: extracted, anonymized
   (aus Metadata), translations (Pfade vor dem Row-Delete aus `document_translations`
   lesen). In `delete_document` und `/reset` verwenden. Fehlende Dateien still
   überspringen (`missing_ok`-Semantik). Sicherheitsgurt beibehalten: nur Pfade
   löschen, die unter den bekannten Basisverzeichnissen liegen (bestehende
   `startswith("/app/")`-Prüfung bzw. `_safe_text_path`-Muster).
3. `from-url`: Temp-Datei in `finally` mit `missing_ok` aufräumen.

**Dateien:** app/endpoints/documents.py, ggf. app/shared.py.

## Task 4: Upload-Eindeutigkeit + Größenlimit

**Problem (Audit):** `app/endpoints/classification.py:745-748, 878-881` — gespeicherter
Name ist `%Y%m%d_%H%M%S_{safe_name}` im geteilten `UPLOADS_DIR`. Zwei Uploads mit
gleichem Dateinamen in derselben Sekunde (auch verschiedene User) überschreiben sich
gegenseitig, beide DB-Zeilen zeigen auf dieselbe Datei. Außerdem existiert kein
Upload-Größenlimit (`await file.read()` puffert beliebig viel in RAM).

**Fix:**
1. An beiden Upload-Stellen eine kurze Zufallskomponente in den gespeicherten
   Dateinamen aufnehmen: `%Y%m%d_%H%M%S_<uuid4-hex[:8]>_{safe_name}`. Prüfen (grep),
   welche Stellen den gespeicherten Stem parsen (Segmentierung nutzt den Stem für
   Segment-Dateinamen — das funktioniert mit beliebigem Stem weiter, verifizieren).
2. Größenlimit: `MAX_UPLOAD_BYTES` aus Env (Default 100 MB). Beim Einlesen prüfen und
   bei Überschreitung HTTP 413 mit deutschem Fehlertext. Chunked lesen (z. B. 1-MB-Chunks
   mit laufender Summe) statt `file.read()` komplett, damit das Limit VOR dem
   Vollpuffern greift.

**Dateien:** app/endpoints/classification.py.

## Task 5: Generation — Fehlerpfade (Refusal, leer, truncated)

**Problem (Audit):** `app/endpoints/generation.py:810-825` (Job-Pfad), `2161-2273` +
`2699-2710` (Streaming-Pfad), `3147` (Gemini):
- `_generate_with_claude_stream` signalisiert Refusals als `{"type":"error"}`-Chunk und
  returned. Der Job-Executor `_execute_generation_request` kennt nur
  text/thinking/usage-Chunks → error wird verschluckt, leerer Draft wird persistiert,
  Job "completed". Browser-Pfad speichert nach error-Chunk ebenfalls den leeren Draft
  und emittet `done` mit draft_id.
- `stop_reason == "max_tokens"` wird nur geprintet — abgeschnittener Draft ohne Warnung.
- `_generate_with_gemini` gibt `response.text or ""` zurück — Safety-Block/leere
  Candidates werden zum leeren "completed"-Draft.

**Fix:**
1. Job-Pfad: error-Chunk erkennen → Exception mit der Fehlermeldung werfen → Job läuft
   in den bestehenden failed-Pfad (`error_message` gesetzt). Keinen Draft persistieren.
2. Browser-Pfad: nach error-Chunk keinen Draft speichern, `error`-Event an den Client
   streamen (Muster für Event-Format im selben Generator übernehmen), kein `done` mit
   draft_id.
3. Leer-Gate an beiden Pfaden: Wenn `generated_text.strip()` leer ist → Fehler statt
   Persist ("Generierung lieferte keinen Text — nicht gespeichert").
4. `max_tokens`: Draft trotzdem speichern, aber `truncated: true` + Hinweis in die
   Draft-Metadata schreiben und im Streaming-Pfad ein Warn-Event senden.
5. Gemini: leere Response/geblockte Candidates → Exception mit deutschem Fehlertext
   (Block-Reason aus der Response mitgeben, wenn vorhanden).

**Dateien:** app/endpoints/generation.py.

## Task 6: send-to-jlawyer — Draft-Matching scopen

**Problem (Audit):** `app/endpoints/generation.py:3694-3718` — nach dem Senden wird der
Draft für die Memory-Reflection per `GeneratedDraft.generated_text == body.generated_text`
über ALLE User und Fälle gesucht (neuester zuerst). Editierter Text → kein Match →
Reflection fällt still aus. Identischer Text in zwei Fällen/Usern → Reflection läuft in
den fremden Fall.

**Fix:**
1. Query auf `GeneratedDraft.user_id == current_user.id` einschränken.
2. Wenn der Request-Body eine draft_id enthält oder sinnvoll erweitert werden kann:
   bevorzugt per id matchen (Frontend-Aufrufstelle in app/static/js/ prüfen — wenn das
   Frontend die draft_id kennt, mitschicken und serverseitig bevorzugen, Textmatch nur
   als Fallback innerhalb des Users behalten).
3. Kein Match → Reflection überspringen mit `[MEMORY]`-Log, kein Fehler für den User
   (Senden nach j-lawyer selbst war ja erfolgreich).

**Dateien:** app/endpoints/generation.py, ggf. app/static/js/drafts.js oder app.js (Aufrufstelle).

## Task 7: Files-API-Aufräumung (Anthropic/OpenAI)

**Problem (Audit):** `app/endpoints/generation.py:1712-1727, 1808-1820` und
`app/endpoints/query.py:611-617` — jede Claude-Generation lädt PDFs per
`client.beta.files.upload`, jede GPT-Generation/Query per `client.files.create` hoch.
Nirgends ein Delete (grep bestätigt: nur Gemini-Pfade löschen). In query.py ist
Cleanup-Code auskommentiert. Storage wächst unbegrenzt bis zur Quota (dann fallen
Claude-Generationen aus).

**Fix:**
1. Nach Abschluss der Generation/Query (auch im Fehlerfall — `finally`): alle in diesem
   Lauf hochgeladenen Anthropic-File-IDs per `client.beta.files.delete(file_id, betas=[...])`
   und OpenAI-File-IDs per `client.files.delete(file_id)` löschen. Delete-Fehler nur
   loggen (`[CLEANUP WARN]`), nie den Erfolg des Laufs gefährden.
2. Die auskommentierte Cleanup-Stelle in query.py reaktivieren/ersetzen nach demselben
   Muster.
3. Hochgeladene IDs pro Lauf in einer Liste sammeln (an den Upload-Stellen), damit das
   `finally` sie kennt.

**Dateien:** app/endpoints/generation.py, app/endpoints/query.py.

## Task 8: Job-Worker — Dedup per job_id, attempt-Cap, Heartbeat-Crash

**Probleme (Audit):**
- `app/job_worker.py:255` — `heartbeat.stop()` joint einen nie gestarteten Thread,
  wenn der Job vor `heartbeat.start()` fehlschlägt (z. B. "Owner user not found",
  `model_validate`-Fehler) → RuntimeError ersetzt den Originalfehler, mark-failed wird
  übersprungen, `run_worker_loop` crasht den Container.
- SIGTERM-Requeue re-runt nicht-idempotente Jobs: `_execute_generation_request`
  committet den Draft VOR dem completed-Commit des Workers (generation.py:664-666),
  `_persist_research_result` analog → Duplikate bei Requeue.
- `attempt_count` wird beim Claim inkrementiert (job_worker.py:137), aber nirgends
  geprüft → Endlos-Requeue möglich.

**Fix (Entscheidung Jay: Requeue behalten, mit Dedup-Check):**
1. Migration + ORM: nullable Spalte `job_id` (UUID, Index) auf `generated_drafts` und
   `research_runs`. `_persist_generated_draft` und `_persist_research_result` stempeln
   die job_id (Parameter durchreichen — die Executor-Funktionen kennen den Job).
2. Worker: nach dem Claim, vor der Ausführung: existiert bereits eine Ergebnis-Zeile
   mit dieser job_id (Draft für generation_jobs, research_run für research_jobs) →
   Job direkt auf completed setzen mit Verweis auf das vorhandene Ergebnis
   (result_payload minimal befüllen: draft_id/research_run_id + Hinweis
   "Ergebnis aus unterbrochenem Lauf übernommen"), Ausführung überspringen.
   Query-Jobs haben kein persistiertes Artefakt → dort ist Re-Run akzeptabel (nur
   API-Kosten, keine Duplikat-Zeile) — unverändert lassen.
3. `attempt_count`-Cap: beim Claim `attempt_count > 3` → Job failed mit
   "Mehrfach unterbrochen/fehlgeschlagen — nicht erneut versucht".
4. `JobHeartbeat.stop()` robust machen: nur joinen, wenn der Thread gestartet wurde
   (`self._thread.is_alive()` oder started-Flag). Zusätzlich in `_run_claimed_job` den
   Fehlerpfad so ordnen, dass mark-failed auch bei Heartbeat-Problemen erreicht wird
   (stop() in eigenes try/except).
5. `run_worker_loop`: try/except um claim/execute mit kurzem Backoff (5s) und
   `[WORKER ERROR]`-Log statt Container-Crash bei transienten DB-Fehlern.

**Dateien:** app/job_worker.py, app/models.py, app/main.py (MIGRATIONS),
app/endpoints/generation.py, app/endpoints/research_sources.py (persist-Funktionen).

## Task 9: Provider-Timeouts + Migrations-Advisory-Lock

**Probleme (Audit):**
- `app/shared.py:1492-1496` — `get_gemini_client()` ohne Timeout (google-genai-Default:
  keiner). Ein hängender Gemini/xAI-Call blockiert den seriellen Worker für immer, der
  Heartbeat-Thread stempelt weiter → Reconcile greift nie.
- `app/main.py:1042-1070` — `apply_schema_migrations()` läuft in app UND job-worker
  gleichzeitig beim Start ohne Lock → Race (PK-Verletzung auf schema_migrations,
  pg_type-Race bei CREATE TABLE), maskiert durch restart-Loop.

**Fix:**
1. `get_gemini_client`: `http_options` mit Timeout aus Env (`GEMINI_HTTP_TIMEOUT_SECONDS`,
   Default 600) setzen. xai/OpenAI/Anthropic-Factories prüfen: wo kein Timeout gesetzt
   ist, expliziten Timeout aus Env-Default ergänzen (Anthropic hat 3600s-Default — so
   lassen).
2. Worker: `asyncio.wait_for(execute_fn(...), timeout=JOB_EXECUTION_TIMEOUT_SECONDS)`
   (Env, Default 3600) → Timeout markiert den Job failed ("Zeitlimit überschritten")
   und der Worker lebt weiter. (Bewusst zusätzlich zu den Client-Timeouts — greift nur,
   wenn die Coroutine awaitet, das tun die to_thread-basierten Provider.)
3. `apply_schema_migrations`: gesamten Lauf in `pg_advisory_lock(<konstante id>)` /
   `pg_advisory_unlock` klammern (Session-Level, gleiche Connection). Konstante id im
   Code definieren (z. B. hash von 'rechtmaschine_migrations' als int64-Literal).

**Dateien:** app/shared.py, app/job_worker.py, app/main.py.

## Task 10: Events — Listener-Reconnect, SSE-Session, Einfachzustellung

**Probleme (Audit):**
- `app/events.py:91-115` — PostgresListener verbindet einmal; jede Exception beendet
  den Thread endgültig. SSE wirkt verbunden (keep-alives), liefert aber nie wieder
  Events bis zum App-Restart.
- `app/endpoints/documents.py:114-154` — `documents_stream` hält die
  `Depends(get_db)`-Session (und damit eine Pool-Connection, idle in transaction) für
  die gesamte Lebensdauer der SSE-Verbindung. Pool 10+20 → ~30 Verbindungen blockieren
  alles.
- `app/shared.py:1518-1544` — `_emit_event` published lokal in den Hub UND schickt
  NOTIFY; der eigene Listener published dasselbe Event erneut → jede Mutation kommt
  doppelt an. Im Worker returned `_emit_event` VOR dem NOTIFY, wenn kein Hub da ist →
  Worker-Broadcasts wären stille No-ops (latent).

**Fix:**
1. `PostgresListener._run`: äußere Reconnect-Schleife mit Backoff (1s→30s, capped),
   bei jedem Connect alle Channels erneut LISTEN, nach erfolgreichem Reconnect ein
   Resync-Event (`{"type": "resync"}`) in den Hub publishen, damit Clients neu fetchen.
   Fehler loggen mit `[LISTENER]`-Präfix. Thread darf nur bei explizitem `stop()` enden.
2. `documents_stream`: initiale Snapshots mit einer kurzlebigen Session bauen
   (`SessionLocal()` mit try/finally close VOR der Event-Schleife), keine
   Depends-Session über die Verbindungsdauer halten.
3. `_emit_event`: lokalen `hub.publish` entfernen — Zustellung ausschließlich über
   NOTIFY→Listener→Hub (ein Pfad, auch für den eigenen Prozess). Den frühen
   `if not hub: return` so ändern, dass NOTIFY immer gesendet wird.
4. Frontend `app/static/js/app.js`: auf `resync`-Event mit demselben Refetch reagieren
   wie auf snapshot-Events (Aufrufstelle ~1921-1941).

**Dateien:** app/events.py, app/endpoints/documents.py, app/shared.py, app/static/js/app.js.

## Task 11: SSE — Per-User-Scoping + One-Time-Ticket

**Probleme (Audit):**
- Jeder SSE-Client bekommt die Events ALLER User (Dateinamen = Mandantennamen,
  document_ids, case_ids). Hub hat kein Per-User-Routing (`app/events.py:23-67`).
- JWT steht als `?token=...` in der EventSource-URL (`app/static/js/app.js:1998`) und
  landet in Caddy-Logs.

**Fix (Entscheidung Jay: Ticket bauen):**
1. Hub: `subscribe(user_id)` — Queues tragen die user_id. `publish(payload)` stellt zu:
   an Subscriber mit `payload['owner_id'] == user_id` und an alle, wenn das Payload
   keine owner_id trägt (System-Events wie resync). Die `broadcast_*_snapshot`-Helper
   in shared.py bekommen einen owner_id-Parameter (str) und legen ihn ins Payload; alle
   Aufrufstellen (grep `broadcast_documents_snapshot|broadcast_sources_snapshot`)
   reichen die owner_id des betroffenen Dokuments/Users durch. owner_id vor dem Senden
   an den Client aus dem Payload entfernen oder drinlassen (eigene id, unkritisch) —
   drinlassen ist einfacher.
2. Ticket-Endpoint: `POST /documents/stream-ticket` (auth via normalem Bearer) →
   `{"ticket": "<uuid4>", "expires_in": 60}`. In-Memory-Store (Dict mit Lock:
   ticket → (user_id, expiry)), Einmalverwendung (pop beim Connect), Ablauf 60s,
   abgelaufene beim Zugriff lazy wegräumen. `GET /documents/stream?ticket=...`
   akzeptiert NUR noch Tickets (den bisherigen `?token=`-Query-Support entfernen,
   Header-Auth kann als Fallback für curl-Nutzung bleiben, falls vorhanden).
3. Frontend: vor dem EventSource-Connect Ticket holen, mit `?ticket=` verbinden, bei
   Reconnect neues Ticket holen (EventSource-Reconnect-Logik anpassen: onerror →
   selbst neu verbinden mit frischem Ticket statt Browser-Auto-Reconnect mit alter URL,
   z. B. EventSource schließen und mit Backoff neu aufbauen).
4. Hinweis: Der Ticket-Store ist prozesslokal — es gibt nur einen App-Container, das
   ist ausreichend. Kommentar im Code, dass Multi-Prozess ein Redis/DB-Store bräuchte.

**Dateien:** app/events.py, app/shared.py, app/endpoints/documents.py, app/static/js/app.js.

## Task 12: Case-Memory — Härtung

**Probleme (Audit):**
- `app/endpoints/agent_memory.py:1681-1686` — `jlr.save_seen(...)` läuft VOR
  `_create_proposals_from_extraction`. Fehler/SIGTERM dazwischen → Dokumente gelten als
  gesehen, Fakten für immer verloren (widerspricht Kommentar Z. 1590 und dem
  SIGTERM-Design).
- `app/agent_memory_service.py:764-827, 428-472` — accept/put ohne Row-Lock und ohne
  Version-Guard im UPDATE → zwei konkurrierende Accepts (UI + Auto-Apply im Worker)
  verlieren still das erste Update.
- `PUT /memory/cases/{case_id}` (`endpoints/agent_memory.py:1890-1918`) — `overview`/
  `strategy` defaulten auf `""` und werden unconditional geschrieben → Client, der nur
  eines schickt, wischt das andere Feld.
- Rebase: Ganze-Liste-`set`-Ops (Konsolidierung) überleben mit gebumpter Version und
  überschreiben beim Accept zwischenzeitlich ergänzte Fakten. Aktuell werden nur
  String-Scalars über `curated_fields` geschützt.
- `enqueue_memory_reflection` documents-merge (`agent_memory_service.py:1132-1154`)
  ohne Lock/Status-Recheck → merged document_ids können auf einen bereits geclaimten
  Job geschrieben werden und werden nie verarbeitet.
- `_apply_patch_ops` IndexError → 500 auch beim CREATE von Proposals
  (`endpoints/agent_memory.py:1958` fängt nur ValueError; accept wurde schon gefixt).
- `_flush_chunk` (`endpoints/agent_memory.py:1543-1563`) hängt source_refs VOR dem
  Extraktionsaufruf an — gescheiterte Chunks werden trotzdem als Provenienz zitiert.

**Fix:**
1. `save_seen` NACH erfolgreichem `_create_proposals_from_extraction` verschieben.
2. `_get_target`/`_get_or_create_target`-Lesepfade, die in accept/put münden: Ziel mit
   `with_for_update()` laden (nur in den Schreibpfaden, nicht im GET-Endpoint).
3. PUT-Endpoint: leere/fehlende `overview` bzw. `strategy` → bestehenden Wert behalten
   statt leer überschreiben. (Semantik: nur nicht-leere Strings überschreiben den
   Scalar. Wer wirklich leeren will, kann das Feld via brief_content/strategy_content
   explizit setzen — im Docstring vermerken.)
4. Rebase generalisieren: `curated_fields`/changed-fields = Felder, deren Wert sich im
   auslösenden Update geändert hat (beliebiger Typ, Vergleich der normalisierten
   Werte). set-Ops auf geänderte Felder werden verworfen (Scalar wie Liste). Der
   Accept-Pfad übergibt die von der akzeptierten Proposal berührten Felder als
   changed-fields. Der bestehende `_apply_patch_ops`-Probelauf bleibt.
5. documents-merge: Job-Zeile mit `with_for_update()` laden und `status == 'queued'`
   nach dem Lock re-checken — wenn nicht mehr queued, neuen Job anlegen statt mergen.
6. Proposal-CREATE-Endpoint: `except (ValueError, IndexError, KeyError, TypeError)` → 400
   (gleicher Wortlaut wie beim Accept-Fix).
7. `_flush_chunk`: source_refs erst NACH erfolgreicher Extraktion des Chunks anhängen.

**Dateien:** app/agent_memory_service.py, app/endpoints/agent_memory.py.

## Task 13: Research — Fehlertransparenz

**Problem (Audit):** `app/endpoints/research_sources.py:936-1015` — wirft die gewählte
Engine (z. B. grok-4.3) eine Exception, wird sie nur geprintet; wenn asyl.net etwas
lieferte, wird der Run als normaler Erfolg persistiert (`provider: "asyl.net"`), ohne
Warnung. Abgelaufener XAI-Key → jede Grok-Recherche "gelingt" mit nur asyl.net-Links.
Außerdem verschluckt `_persist_research_result` (Z. 361-364) eigene DB-Fehler und gibt
ein Ergebnis ohne research_run_id zurück.

**Fix:**
1. Single-Engine-Pfad: Engine-Exception fangen, aber in den Run-Metadaten sichtbar
   machen: `metadata['engine_error'] = {engine, message}` und ein deutsches
   Warning-Feld, das die UI/CLI anzeigen kann ("Suchmaschine grok-4.3 fehlgeschlagen —
   Ergebnis enthält nur asyl.net-Quellen"). Wenn die gewählte Engine fehlschlägt UND
   asyl.net leer ist → Job failed mit der Engine-Fehlermeldung (statt leerem Erfolg).
2. `_persist_research_result`: DB-Fehler nicht verschlucken — raise, damit der Job in
   den failed-Pfad läuft (der Aufrufer behandelt das bereits).
3. Prüfen, wo das Frontend/die CLI Run-Metadaten anzeigen, und das Warning-Feld dort
   durchreichen, sofern trivial (sonst nur API-seitig bereitstellen und im
   Task-Report vermerken).

**Dateien:** app/endpoints/research_sources.py, ggf. app/static/js/app.js.

## Task 14: Scoping-Rest — Legacy-Drafts, RAG-Owner-Filter

**Probleme (Audit):**
- `app/endpoints/drafts.py:33, 73-83` — Drafts mit `user_id IS NULL` (vor Multi-User)
  sind für alle Accounts sichtbar; `get_draft` überspringt bei NULL die Checks.
- `app/endpoints/rag.py:148-152` + `rag/api/main.py:190-219, 454-489` — upsert stempelt
  owner_id in die Chunk-Metadata, retrieve filtert nie danach → sobald privates
  Material ingestiert wird, cross-tenant lesbar.

**Fix (Entscheidung Jay: NULL-Drafts dem Admin zuweisen):**
1. Migration: `UPDATE generated_drafts SET user_id = (SELECT id FROM users WHERE email =
  '<ADMIN>') WHERE user_id IS NULL` — die Admin-Mail zur Laufzeit aus der bestehenden
  Bootstrap-Admin-Logik in app/main.py ableiten (dort existiert der Admin-User-Insert —
  dieselbe Quelle/Env nutzen, kein Hardcoding). Da MIGRATIONS statisches SQL ist:
  als idempotentes SQL mit Subselect auf die Bootstrap-Admin-UUID formulieren
  (die Bootstrap-UUID ist im Code konstant — nachsehen in main.py:231-240).
2. drafts.py: NULL-Sonderfälle entfernen — alle Queries strikt `user_id == current_user.id`
  (+ bestehende case-Checks).
3. RAG: `_build_retrieve_payload` in app/endpoints/rag.py reicht `owner_id` des
  aufrufenden Users mit; `rag/api/main.py` `_filter_sql` ergänzt
  `(metadata->>'owner_id' IS NULL OR metadata->>'owner_id' = :owner_id)` — öffentlicher
  Korpus bleibt für alle sichtbar, private Chunks nur für den Eigentümer. Hinweis im
  Report: rag-Stack läuft auf debian, Änderung dort wird beim nächsten Pull/Restart
  wirksam (nicht Teil dieses Tasks).

**Dateien:** app/main.py (MIGRATIONS), app/endpoints/drafts.py, app/endpoints/rag.py,
rag/api/main.py.

## Task 15: Ops-Hygiene — Stuck-Status-Reconciliation, Gemini-URI-Staleness, Segment-Cleanup

**Probleme (Audit):**
- Auto-OCR/Anon/Segmentierung laufen als fire-and-forget `loop.create_task` — App-Restart
  strandet Dokumente dauerhaft in `anonymizing`/`ocr_processing`/`anon_pending`/
  `ocr_pending` (classification.py:200/220/334/837/941, document_segmentation.py:657).
  Zudem werden Task-Referenzen nicht gehalten (GC-Risiko).
- `app/shared.py:606-620` — `ensure_document_on_gemini` reused den Gemini-Upload, wenn
  URI auflösbar und MIME gleich — Re-OCR mit gleichem MIME liefert bis 48h alten Inhalt.
- `POST /documents/{id}/segment?force=true` (`documents.py:344-347`) löscht nur
  `DocumentSegment`-Zeilen — alte Kind-Documents + PDFs bleiben und akkumulieren.

**Fix:**
1. Startup-Reconciliation (app/main.py startup event, nach Migrationen): Dokumente in
   transienten Status (`anonymizing`, `ocr_processing`, `anon_pending`, `ocr_pending`,
   analoge Segmentierungs-Status — Status-Werte per grep sammeln), deren `timestamp`/
   Update älter als 1h ist, auf den passenden failed-Status setzen (`anon_failed`/
   `ocr_failed`) mit Log. Broadcast danach einmal.
2. Task-Referenzen: die `create_task`-Aufrufe in ein kleines Helper wrappen, das die
   Task in ein Modul-Set legt und bei done discarded (Standard-Pattern gegen GC).
3. `ensure_document_on_gemini`: Datei-mtime+size in die Cache-Metadata aufnehmen und
   beim Reuse vergleichen — abweichend → neu hochladen.
4. `force=true`-Resegmentierung: vorher existierende Kind-Documents dieses Parents
   (erkennbar über das bestehende Parent/Segment-Verknüpfungsfeld — nachsehen, wie
   Kinder referenziert werden) inkl. Dateien löschen (Task-3-Helper wiederverwenden).

**Dateien:** app/main.py, app/endpoints/classification.py, app/shared.py,
app/endpoints/documents.py, app/endpoints/document_segmentation.py (Pfad prüfen).

## Checkpoints (Controller, nicht Implementer)

- Nach Task 2, Task 9 und Task 15: `docker compose restart app job-worker`, Logs prüfen,
  gezielte Live-E2E (Anonymisierungs-Gate mit Testdokument, SSE-Stream, Memory-Flows).
- Finaler Whole-Branch-Review über den gesamten Sweep, danach Deploy-Restart + Sync-Hinweis
  für desktop/debian (rag/-Änderung).

## Follow-ups (Stand nach Abschluss, 2026-07-07)

Aus den Task-Reviews und dem finalen Whole-Branch-Review übrig geblieben,
bewusst nicht mehr Teil des Sweeps:

1. **Hub-Per-User-Filter-Test committen** — die Routing-Logik in
   `app/events.py` (`BroadcastHub.publish`, owner-Filter) ist live verifiziert,
   aber nur ad-hoc getestet. Pure-Logic-Test analog `tests/test_sse_ticket_store.py`.
2. **Adopted-Job-Payload vervollständigen** — nach SIGTERM-Requeue-Adoption
   enthält `result_payload` nur `draft_id` + Hinweis, kein `generated_text`
   (`app/job_worker.py`, Adopt-Block). Nur CLI-Polling betroffen. Ein-Zeilen-Fix.
3. **Anon-Gate im Preview-Endpoint** — `anonymize_uploaded_file` in
   `app/endpoints/anonymization.py` (DB-loser Vorschau-Endpoint) kann bei
   degenerierter Flair-Antwort unredigierten Text als "anonymisiert" anzeigen.
   Persistiert nichts, aber dasselbe `_anonymization_gate_failed`-Gate einbauen.
4. **Worker-seitige Auto-Anonymisierung als Job** — ein per CLI angestoßener
   OCR-Job löst im job-worker `schedule_auto_anonymization` als Loop-Task aus
   (`app/endpoints/ocr.py`). Ein Worker-SIGTERM strandet das Dokument in
   `anonymizing` bis zum nächsten App-Restart. Sauber: `AnonymizeJob` einreihen
   statt fire-and-forget-Task (Finding N2 des Final-Reviews).
5. **`ANTHROPIC_FILES_API_BETAS`-Konstante konsequent nutzen** — zwei
   Inline-Literale in `app/endpoints/generation.py` neben der Konstante.
   Kosmetisch, beim nächsten Anfassen der Datei mitnehmen.
6. **app.js Ticket-Fetch-Fehlerpfad** — der catch beim Ticket-Holen zählt
   nicht auf den Failure-Counter, daher kein Disable-Fallback wie im
   onerror-Pfad. Nur Retry-Hygiene, kein Datenpfad.
7. **Gemini-Upload-Stat-Altbestand** — Dokumente mit `gemini_file_uri` von vor
   diesem Deploy haben noch keinen mtime/size-Cache-Eintrag und nutzen einmalig
   die alte MIME-only-Reuse-Prüfung. Selbstheilend (48h-URI-Expiry), keine Aktion.
8. **Ad-hoc-Tests importieren Logik nicht** — mehrere `tests/test_*`-Skripte
   re-implementieren die geprüfte Logik statt sie zu importieren (bewusste
   Entscheidung wegen App-Import-Abhängigkeiten). Bei Einführung einer echten
   Test-Infrastruktur konsolidieren.

Deploy-Status: Alle 15 Tasks + Final-Review-Fixes sind auf dem Server live.
Die rag/-Änderung (Owner-Filter, `rag/api/main.py`) wird auf debian erst mit
Pull + RAG-Stack-Restart wirksam.

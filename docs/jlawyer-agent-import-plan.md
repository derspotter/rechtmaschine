# j-lawyer Agent Import Plan

## Ziel
Rechtmaschine soll Dokumente aus einer j-lawyer-Akte nicht nur manuell exportieren, sondern eine Akte lesen, alle Dokument-Metadaten laden, dem Agenten eine Auswahlentscheidung erlauben und die ausgewählten Dokumente anschließend in die bestehenden Dokument-Boxen importieren.

Der sichere Standard ist dabei:
1. erst nur Metadaten lesen
2. dann Agent oder Nutzer Auswahl treffen lassen
3. erst danach Inhalte ziehen
4. dann in Rechtmaschine als normale `Document`-Einträge speichern

Das gleiche Muster kann später für Nextcloud wiederverwendet werden.

## Bereits vorhandene Bausteine

### Rechtmaschine
- Dokument-Kategorien existieren bereits in `app/shared.py:405`
  - `Anhörung`
  - `Bescheid`
  - `Rechtsprechung`
  - `Vorinstanz`
  - `Akte`
  - `Sonstige gespeicherte Quellen`
- Die UI-Boxen dafür existieren bereits in `app/templates/index.html:219`
- Dokumente werden bereits als `Document` gespeichert in `app/models.py:35`
- URL-basierter Dokumentimport existiert schon als Referenz in `app/endpoints/documents.py:188`
- Die Auswahlstruktur für Query/Recherche/Entwurf existiert bereits in `app/shared.py:561`

### j-lawyer
Aus `j-lawyer-anleger/swagger.json` stehen die relevanten Lesepfade bereits fest:
- `GET /j-lawyer-io/rest/v1/cases/list`
- `GET /j-lawyer-io/rest/v1/cases/{id}`
- `GET /j-lawyer-io/rest/v1/cases/{id}/documents`
- `GET /j-lawyer-io/rest/v1/cases/document/{id}`
- `GET /j-lawyer-io/rest/v1/cases/document/{id}/content`

Wichtig:
- `/content` liefert JSON mit Base64-Inhalt, nicht rohe Dateibytes.
- Das reicht aus, um PDFs oder ODTs kontrolliert herunterzuladen und lokal abzulegen.

## Gewünschter Ziel-Flow
1. Nutzer verknüpft einen Rechtmaschine-Fall mit einer j-lawyer-Akte.
2. Rechtmaschine lädt die Dokumentliste der j-lawyer-Akte.
3. Der Agent sieht nur Metadaten und schlägt vor:
   - welche Dokumente relevant sind
   - in welche Box sie gehören
   - welche Rolle sie haben, z. B. Hauptbescheid / weitere Bescheide / Vorinstanz
4. Nutzer bestätigt oder korrigiert.
5. Rechtmaschine lädt nur die bestätigten Dokumentinhalte nach.
6. Die Dateien werden lokal gespeichert und als normale `Document`-Einträge angelegt.
7. Danach funktionieren Query, Recherche und Entwurf wie bisher, weil die Dokumente lokal im normalen Datenmodell liegen.

## Architektur

### 1. Connector-Ebene
Neue Backend-Datei:
- `app/endpoints/jlawyer_import.py`

Verantwortung:
- Basic-Auth gegen j-lawyer
- Aktenliste lesen
- Dokumentliste lesen
- Einzeldokumentinhalt ziehen
- keine Agentik hier
- keine Kategorisierung hier

### 2. Proposal-Ebene
Der Agent bekommt nur strukturierte Dokument-Metadaten und erzeugt eine Import-Empfehlung.

Input pro Dokument:
- `jlawyer_document_id`
- `name`
- `creationDate`
- `changeDate`
- `size`
- `tags`
- optional `folderId`
- optional kurzer Snippet, aber erst später

Output pro Dokument:
- `import: true|false`
- `category: Anhörung|Bescheid|Vorinstanz|Rechtsprechung|Akte|Sonstige gespeicherte Quellen`
- `role: primary|secondary|null`
- `confidence`
- `reason`

### 3. Import-Ebene
Bestätigte Dokumente werden heruntergeladen, im Upload-Speicher abgelegt und als normale `Document`-Rows erzeugt.

## Neue Backend-Endpunkte

### A. Akten suchen/listen
`GET /integrations/jlawyer/cases`

Query:
- `q` optional
- `limit` optional

Response:
- Liste von j-lawyer-Akten mit mindestens:
  - `id`
  - `name`
  - `file_number`
  - `subject_field`
  - `reason`
  - `archived`

Zweck:
- Fall mit j-lawyer-Akte verknüpfen

### B. Akte verknüpfen
`POST /integrations/jlawyer/link-case`

Request:
- `rechtmaschine_case_id`
- `jlawyer_case_id`
- optional Snapshot-Metadaten der Akte

Persistenz:
- entweder in `Case.state`
- oder besser in neuer Tabelle `external_case_links`

Empfehlung:
Neue Tabelle `external_case_links`:
- `id`
- `owner_id`
- `case_id`
- `provider` = `jlawyer`
- `external_case_id`
- `external_label`
- `metadata`
- `created_at`
- `updated_at`

### C. Dokumentliste einer verknüpften Akte laden
`GET /integrations/jlawyer/cases/{jlawyer_case_id}/documents`

Response:
- `case`
- `documents`
- je Dokument:
  - `external_document_id`
  - `name`
  - `creation_date`
  - `change_date`
  - `size`
  - `tags`
  - `version`
  - `folder_id`

Das ist rein read-only.

### D. Agent-Vorschlag für Import und Box-Zuordnung
`POST /integrations/jlawyer/propose-import`

Request:
- `case_id`
- `jlawyer_case_id`
- `documents: [...]`
- optional `strategy_hint`, z. B. `asylklage`

Response:
- `proposals: [...]`
- je Dokument:
  - `external_document_id`
  - `import`
  - `category`
  - `role`
  - `confidence`
  - `reason`

Regeln:
- hier noch nichts herunterladen
- nur Vorschläge
- Bescheid und Vorinstanz dürfen `primary` bekommen
- andere Kategorien nur `secondary` oder `null`

### E. Bestätigte Dokumente importieren
`POST /integrations/jlawyer/import-documents`

Request:
- `case_id`
- `jlawyer_case_id`
- `items: [...]`
- je Item:
  - `external_document_id`
  - `category`
  - `role`
  - optional `title_override`

Server-Ablauf:
1. `GET /v1/cases/document/{id}` oder vorhandene Metadaten verwenden
2. `GET /v1/cases/document/{id}/content`
3. Base64 dekodieren
4. Dateiendung bestimmen
5. Datei in Upload-Speicher schreiben
6. `Document`-Row anlegen mit:
   - `filename`
   - `category`
   - `confidence=1.0`
   - `explanation="Importiert aus j-lawyer: ..."`
   - `file_path`
   - `owner_id`
   - `case_id`
7. `broadcast_documents_snapshot(...)`

Response:
- importierte Dokumente als normale Rechtmaschine-Dokumente
- plus Mapping von `external_document_id -> document_id`

## Persistenzmodell

### Minimalvariante
Nur `Case.state` erweitern:
- `integrations.jlawyer.case_id`
- `integrations.jlawyer.case_label`

Nachteil:
- schlechter suchbar
- weniger auditierbar

### Bessere Variante
Neue Tabellen:

#### `external_case_links`
- Zuordnung Rechtmaschine-Fall -> externe Akte

#### `external_document_imports`
- Historie externer Dokumentimporte
- Felder:
  - `id`
  - `owner_id`
  - `case_id`
  - `provider`
  - `external_case_id`
  - `external_document_id`
  - `document_id`
  - `external_name`
  - `category_assigned`
  - `role_assigned`
  - `imported_at`
  - `metadata`

Nutzen:
- Duplikatvermeidung
- Audit-Trail
- spätere Re-Syncs

## Agent-Entscheidungslogik

### Erste Version: nur Metadaten
Heuristik + LLM auf Dateinamen und Tags.

Beispiele:
- `Bescheid`, `BAMF`, `Ablehnung`, `Widerruf` -> `Bescheid`
- `Urteil`, `Beschluss`, `VG`, `OVG` -> `Vorinstanz` oder `Rechtsprechung`
- `Anhörung`, `Niederschrift`, `Befragung` -> `Anhörung`
- `Gerichtsakte`, `Beiakte`, `Ausländerakte` -> `Akte`

### Zweite Version: Snippet-assisted
Für unsichere Dokumente zusätzlich ersten Textausschnitt ziehen:
- nur erste 1-2 Seiten oder einige KB
- dann bessere Einordnung

### Sicherheitsregel
Der Agent entscheidet nie final unsichtbar.
Standard:
- Vorschlag machen
- Nutzer bestätigen
- dann importieren

## Frontend-Plan

### 1. Neue Integrationssektion am Fall
Neue kleine Box oder Modal:
- `j-lawyer Akte verknüpfen`
- Suche nach Akten
- Auswahl einer Akte
- Anzeige der verknüpften Akte

Keine neue Hauptseite nötig.

### 2. Dokumentliste laden
Button in derselben UI:
- `Dokumente aus j-lawyer prüfen`

Dann Modal/Drawer mit Tabelle:
- Name
- Datum
- Größe
- Tags
- Agent-Vorschlag Kategorie
- Agent-Vorschlag Rolle
- `Importieren` Checkbox
- Kategorie Dropdown
- Rolle Dropdown bei Bescheid/Vorinstanz

### 3. Import bestätigen
Button:
- `Ausgewählte Dokumente importieren`

Danach:
- Import-Request an Backend
- `loadDocuments()` neu ausführen
- die Dokumente erscheinen direkt in den bestehenden Boxen

### 4. Bestehende Boxen weiterverwenden
Kein neues Box-System.
Importziel bleibt:
- `Anhörung`
- `Bescheid`
- `Vorinstanz`
- `Akte`
- `Rechtsprechung`
- `Sonstige gespeicherte Quellen`

Das ist wichtig, weil Query, Recherche und Entwurf bereits auf diesen Kategorien aufbauen.

## Konkrete Box-Zuordnung

### Bescheid
- genau wie heute
- primärer Bescheid kann in `selected_documents.bescheid.primary`
- weitere Bescheide in `selected_documents.bescheid.others`

### Vorinstanz
- primäres Urteil/Beschluss in `selected_documents.vorinstanz.primary`
- weitere Gerichtsentscheidungen in `selected_documents.vorinstanz.others`

### Akte
- Gerichtsakte
- Beiakte
- Ausländerakte
- Anlagenbände

### Rechtsprechung
- externe Entscheidungen, die nicht direkt zur eigenen Akte gehören

### Sonstiges
- alles, was zwar nützlich ist, aber nicht in die Kernboxen gehört

## Empfohlene Reihenfolge der Umsetzung

### Phase 1
- j-lawyer-Connector read-only
- Akten suchen
- Akte verknüpfen
- Dokumentliste laden

### Phase 2
- Proposal-Endpoint
- Agent-Vorschläge nur aus Metadaten
- Frontend-Review-Modal

### Phase 3
- bestätigte Dokumente importieren
- Dateien lokal speichern
- in bestehende Boxen einsortieren

### Phase 4
- Snippet-assisted Klassifikation
- Duplikatvermeidung mit `external_document_imports`
- Re-Sync einer Akte

### Phase 5
- gleicher Connector-Standard für Nextcloud/WebDAV

## Warum das der richtige Schnitt ist
- j-lawyer bleibt Source of Truth für Rohakten
- Rechtmaschine bleibt Arbeitsoberfläche für Auswahl, OCR, Anonymisierung, Recherche und Entwurf
- der Agent bekommt Autonomie, aber erst auf Metadatenebene
- Import bleibt auditierbar und kontrollierbar
- die bestehende UI und das bestehende `Document`-Modell werden weitergenutzt statt umgangen

## Wichtigster technische Punkt
Der Agent sollte nicht sofort alle Inhalte laden.

Besser:
- erst `GET /cases/{id}/documents`
- dann auswählen lassen
- erst danach `GET /cases/document/{id}/content`

Das minimiert:
- unnötige Downloads
- Tokenkosten
- UI-Müll
- falsche Imports

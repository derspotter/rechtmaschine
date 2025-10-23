# Anforderungen & Datenerfassung

## 1. Dokumenteingang
- Quelle: IMAP-Mailbox (Postfix-Account), akzeptiert nur Absender = Scanner.
- Formate: PDF (BAMF-Bescheide, gescannte Formulare, Vollmachten), ggf. mehrere Anhänge pro Mail.
- Erwartetes Volumen: _noch offen (Anzahl Mails/Tag, Dateigrößen)_
- Priorität: Verarbeitung erfolgt FIFO, zeitnahe Anlage der Akten.

## 2. Zielstruktur für JSON-Ausgabe des LLM
- **Akte**
  - `akten_typ` (z. B. Asylverfahren, Familiennachzug, Sozialrecht)
  - `akten_titel` / Kurzbeschreibung
  - `aktenzeichen_extern` (falls vorhanden, z. B. BAMF-Aktennummer)
  - `eingangsdatum` (Mail-Zeitstempel oder Datum im Bescheid)
  - `beschreibung` (Kurztext aus PDF)
- **Mandant / Beteiligte**
  - `rolle` (Mandant, Vertreter, Behörde, Gericht)
  - `name_vollstaendig`
  - `geburtsdatum`
  - `anschrift` (Straße, PLZ, Ort, Land)
  - `kontakt` (Telefon, E-Mail, optional)
  - `identifikatoren` (z. B. AZ, Kundennummer, Staatsangehörigkeit)
- **Dokumentdaten**
  - `dokument_typ` (Bescheid, Vollmacht, Formular)
  - `dokument_datum`
  - `fristen` (Fristtyp, Datum, Beschreibung)
  - `seitenumfang`
  - `bemerkungen`
- **Meta**
  - `confidence_scores` (LLM-Selbstbewertung)
  - `verarbeitungs_hinweise` (z. B. „manuelle Prüfung empfohlen“)

_Noch zu klären_: Pflichtfelder vs. optional, mehrsprachige Namen, mehrere Mandant:innen pro Akte, Umgang mit fehlenden Daten.

## 3. j-lawyer.org Integrationsziele
- Neue Akte erzeugen inkl. Aktenkategorie, Beschreibung.
- Beteiligte/Mandanten mit Rollen anlegen; Duplikatsprüfung (nach Name/Akte?).
- Dokumente als PDF hochladen, inkl. Metadaten (Kategorie, Frist).
- Perspektivisch: Dokumente bestehenden Akten zuordnen (Matching-Regeln definieren).

Offen:
- Welche Aktenarten/Rollen existieren im j-lawyer-System?
- Authentifizierungsmethode (API Key, Basic Auth, OAuth).
- Benötigt die API bestimmte Pflichtfelder (z. B. Kostenstellen, Zuständigkeiten)?

## 4. Nicht-funktionale Anforderungen
- **Sicherheit**: TLS/IMAPS, sichere Secrets-Verwaltung, PII-Logging minimieren.
- **Fehlerhandling**: Wiederholversuche bei IMAP/LLM/API-Fehlern, Alerts bei Dauerfehlern.
- **Nachvollziehbarkeit**: Audit-Log pro Mail (Eingang, LLM-Response, API-Antwort).
- **Performance**: Antwortzeit LLM vs. akzeptable Latenz; Offline-Queue bei Ausfällen.
- **Kostenkontrolle**: Token-Limits, Accounting pro Vorgang.

## 5. Offene Punkte / Fragen an Stakeholder
1. Gibt es Referenz-Akten in j-lawyer mit gewünschter Struktur?
2. Sollen Fristen automatisch überwacht werden (z. B. Reminder)?
3. Welche Sprachen müssen OCR/LLM unterstützen (Deutsch, Englisch, andere)?
4. Müssen Vollmachten signiert werden (Prüflogik)? Digitale Signatur?
5. Wie erfolgt manuelle Freigabe bei unsicherer Extraktion?
6. Wie lange sollen Roh-PDFs und JSON aufbewahrt werden?

> Bitte ergänzen/kommentieren, damit wir die JSON-Spezifikation finalisieren können.

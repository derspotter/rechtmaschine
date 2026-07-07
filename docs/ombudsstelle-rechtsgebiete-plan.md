# Ombudsstelle-Erweiterung: neue Rechtsgebiete (Sozialrecht, Mietrecht)

Stand: 2026-07-05 · Ansatz C (Hybrid in Ausbaustufen) mit Jays Sign-off.
Muster: wie docs/research-pipeline-upgrade-plan.md — pro Stufe TDD,
Feature-Flags, Rollout nur nach explizitem Go; §-Rollout-Log am Ende.

## 1. Zielbild

Die Plattform soll perspektivisch die Ombudsstelle einer Partei tragen
(Vorfeldorganisation): echten Rechts-Mehrwert für Mitglieder, nicht nur
Kanzlei-Werkzeug. Auswahlmatrix für Gebiete: Massenbedarf ×
Automatisierbarkeit × ohne Anwalt machbar × Passung zur bestehenden
Pipeline.

Gebiete in Prioritätsreihenfolge:
1. **Sozialrecht** (SGB II/III, V, IX, XI, Wohngeld/BAföG) — der Kern.
   Widerspruch als Massenprodukt: kostenlos, kein Anwaltszwang,
   Monatsfrist, hohe Abhilfequoten.
2. **Mietrecht** — Nebenkosten-Check, Mieterhöhungs-Prüfung, Mängel/
   Minderung; stark standardisierbar (Mieterverein-Terrain).
3. **Inkasso-/Verbraucherabwehr** — billig zu bauen, sofort fühlbar.
4. **Arbeitsrecht nur als Fristen-Triage** (3-Wochen-Frist-Alarm +
   Weiterleitung); keine Vertretung (Gewerkschafts-Terrain).

Nicht-Ziele: Familienrecht; ArbR-Vertretung; RDG-§7-Vereinsstruktur in
diesen Stufen; Vertretungsübernahme durch die Plattform.

## 2. Leitentscheidungen (Brainstorming 2026-07-05)

- **Selbsthilfe-Modell**: Die Plattform generiert Entwürfe, die das
  Mitglied im eigenen Namen einreicht (Ich-Form des Mitglieds, kein
  Anwaltston). Kein RDG-§7-Rahmen nötig. Eskalation komplexer Fälle an
  die Kanzlei, wo es anwaltlich wird.
- **Bedienmodell gestuft**: Operator-Modell zuerst (Mitglieder schicken
  Bescheide ein, geschulte Leute fahren die Pipeline und schicken den
  Entwurf zurück); Self-Service später pro Flow, wenn die Qualität
  operatorseitig bewiesen ist.
- **Architektur C**: kein Big-Bang-Domänenumbau (A), keine Insel-Flows
  (B); dünne Rechtsgebiets-Registry, die jede Stufe nur so weit
  ausbaut, wie der aktuelle Flow es erzwingt.

## 3. Ist-Zustand (was schon domänen-neutral ist)

Neutral: Segmentierung, Anonymisierung, OCR, Case-Memory,
Draft-Mechanik (Dokumente → Memory → Prompt → Modell → Citation-Verify),
Mail-Intake (Gemma-first), j-lawyer-Anbindung, Kalender/WV.

Asyl-gebunden (die vier Schichten, die pro Gebiet Gegenstücke brauchen):
1. Facetten-Schema (herkunftsland/schutzgruende/verfahrensart).
2. Vokabular (rag_vocabulary.json, aus dem asyl.net-Korpus gebaut).
3. Rechtsprechungs-Store + Ingest (asyl.net; `country` NOT NULL) +
   Doktrin-Wiki (aufentha.lt) + COI-Research.
4. Prompt-Templates (Klagebegründung gegen BAMF-Bescheid, AZB § 78
   AsylG) und Normtexte (`legal_texts/laws/`: nur AsylG, AsylbLG,
   AufenthG, GG).

## 4. Stufe 0 — Rechtsgebiet-Trennung (klein, sofort)

- Migration: `cases.rechtsgebiet` (String, NULL erlaubt), Werte:
  `asyl | aufenthalt | sozial | miete | inkasso | arbeit | sonstiges`.
  NULL verhält sich wie bisher (Migrationsrecht) — Bestandskompatibilität.
- Backfill-CLI: Bestandsfälle per kleinem flachem Qwen-Call
  klassifizieren (Muster backfill_case_facets.py; --dry-run zuerst).
- Intake: Rechtsgebiet bei Fallanlage setzen (Gemma-Klassifikation
  liefert das Signal bereits; Mapping in der Intake-Route).
- Gating: Facetten-Hook (facet_extraction) und Jurisprudenz-Pack
  (maybe_render_jurisprudence_context) laufen nur für
  rechtsgebiet ∈ {NULL, asyl, aufenthalt}. Beendet das nächtliche
  Anklopfen bei Nicht-Asyl-Fällen (008/26).
- Dünne Registry `app/rechtsgebiete.py`: pro Gebiet {key, label,
  facet_schema?, prompt_pack?, laws?, juris_sources?} — nur Felder, die
  eine Stufe wirklich braucht. Asyl/Aufenthalt = Eintrag Nr. 1 mit dem
  Bestand (Facetten aus facets.py, Prompts aus generation.py,
  asyl.net-Quelle).
- Tests: pure Gating-Logik + Registry-Lookup; Migrations-Smoke.

## 5. Stufe 1 — Sozialrecht-Widerspruch (erster Flow, Operator)

- **Normtexte**: SGB I, II, III, X und SGG-Auszüge (§§ 83–86a) über den
  vorhandenen gesetze-im-internet-Downloader nach `legal_texts/laws/`.
- **Facetten SozR** (flacher Qwen-Call, Registry-Schema): traeger
  (jobcenter/krankenkasse/rentenversicherung/sozialamt/familienkasse),
  leistung (buergergeld/krankengeld/rente/…), bescheidart (ablehnung/
  aufhebung/erstattung/sanktion/einstellung), bescheiddatum,
  zugestellt_am. Gleiche Bauart wie facet_extraction.py (klein, flach,
  fill-only, advisory).
- **Fristmodul** (pur, hart getestet): Zugangsvermutung + Monatsfrist
  (§ 84 SGG, § 37 SGB X — genaue Tageszahl BEI IMPLEMENTIERUNG am
  Normtext verifizieren, Stand nach PostModG!) → Kalender-WV.
  SICHERHEITSREGEL: immer konservativste Frist anzeigen; die Plattform
  trifft nie selbst die Aussage "verfristet" — Operator entscheidet.
- **Draft-Type `widerspruch_sozialrecht`**: Aufbau Zulässigkeit (Frist,
  Form) → Begründetheit entlang typischer Bescheidfehler (Anhörung § 24
  SGB X, Begründung § 35 SGB X, Berechnungs-/Sachverhaltsfehler) →
  Antrag (Abhilfe; ggf. aufschiebende Wirkung/Aussetzung § 86a SGG).
  Ich-Form des Mitglieds. Citation-Verify gegen die SGB-Normtexte
  (deterministisch, ohne COI-Pfad).
- **Operator-Workflow**: Mail-Intake → RM-Fall (rechtsgebiet=sozial,
  eigener RM-User "ombudsstelle" zur Datentrennung von Mandantendaten)
  → Operator generiert/prüft → Rückversand als PDF/ODT per Mail.
- Kein Rechtsprechungs-Pack in dieser Stufe.
- Golden Case: 008/26-artiger Jobcenter-Bescheid als Smoke-Fall.

## 6. Stufe 2 — Mietrecht-Flows + Inkasso

- **Nebenkosten-Check**: Positionen per Qwen extrahieren, dann
  deterministisch prüfen (Abrechnungsfrist § 556 Abs. 3 BGB,
  BetrKV-Umlagefähigkeitskatalog, rechnerische Plausibilität) →
  Prüfbericht + Einwendungsschreiben (Einwendungsfrist beachten).
- **Mieterhöhungs-Prüfung**: Kappungsgrenze, Begründungsmittel
  (Mietspiegel), Zustimmungsfrist — rechnerisch prüfbar; Musterantwort.
- Normtexte: BGB Mietrecht (§§ 535–580a), BetrKV, HeizkostenV.
- Zweiter Registry-Eintrag — Nagelprobe für die Registry-Abstraktion:
  was A vorab generalisiert hätte, wird hier am zweiten realen Gebiet
  gehärtet.
- **Inkasso-Antwort** als Mini-Flow (Forderungs-/Gebühren-Check +
  Musterbrief) — fällt weitgehend als Abfallprodukt ab.

## 7. Stufe 3 — Qualitätsschichten pro Gebiet (erst bei Volumen)

- Store-Migration: `rechtsprechung_entries.rechtsgebiet`, `country`
  nullable; alle Bestandseinträge rechtsgebiet=asyl.
- Zweiter Ingest-Adapter: BSG/LSG (rechtsprechung-im-internet bzw.
  sozialgerichtsbarkeit.de), später BGH VIII ZR für MietR.
- Matching/Scoring: Match-Achsen pro Gebiet aus der Registry (SozR:
  traeger/leistung statt herkunftsland/normen); juris_facets
  entsprechend parametrisieren.
- Doktrin-Wiki pro Gebiet nach dem KAG-/aufentha.lt-Muster.
- Enrichment (profil/reliance) nur, wenn das Gebiet Profil-Achsen hat,
  die Entscheidungen tragen (SozR: ja — z.B. Gesundheit, Bedarfsgemeinschaft).

## 8. Stufe 4 — Self-Service pro bewiesenem Flow

- Kandidat Nr. 1: Nebenkosten-Check (am stärksten standardisiert).
- Schmales Ein-Zweck-Frontend (Formular + Upload), Mitglieder-Tenancy
  (Mitglied sieht nur den eigenen Vorgang), Löschkonzept, Haftungs-/
  Selbsthilfe-Hinweise, Rate-Limits/Missbrauchsschutz.
- Startbedingung: Operator-Qualität des Flows über N reale Vorgänge
  belegt (Schwelle bei Stufenbeginn festlegen).

## 9. Risiken & offene Punkte

- Fristlogik ist sicherheitskritisch → konservativ rechnen,
  Operator-Gate, nie automatische Verfristungs-Aussage.
- Datentrennung: Mitgliedervorgänge unter eigenem RM-User, nicht im
  Mandantenbestand.
- Haftungsrahmen/Freigaberegeln für Selbsthilfe-Entwürfe definiert Jay
  (anwaltliche Policy, nicht Code).
- Trägerstruktur, Name, Mitgliedsmodell: politisch, außerhalb dieses
  Plans.
- Alle konkreten Normzitate in diesem Plan (Fristen, §§) sind bei
  Implementierung am aktuellen Normtext zu verifizieren, nicht aus dem
  Plan zu übernehmen.

## 10. Rollout-Log

- 2026-07-05 — Stufe 0 implementiert (Branch ombudsstelle-stufe0, TDD):
  - Registry app/rechtsgebiete.py (7 Gebiete, Aliase, uses_asyl_layers;
    NULL = Legacy = Migrationsrecht).
  - cases.rechtsgebiet (VARCHAR(20), nullable, Index) + Migration
    2026-07-05_case_rechtsgebiet; POST /cases nimmt rechtsgebiet an,
    PUT /cases/{id}/rechtsgebiet für Intake/Operator (422 bei unbekannt,
    null löscht).
  - Gating: Facetten-Hook, Facetten-Backfill und Jurisprudenz-Pack
    (Render + beide Endpoints) laufen nur noch für NULL/asyl/aufenthalt.
  - backfill_rechtsgebiet.py: Qwen-Klassifikation der Bestandsfälle,
    fill-only, --dry-run.
- 2026-07-06 — Stufe 0 ROLLOUT (Jay: "go" + Korrekturauftrag):
  - Merge 6e3d348, Migration verifiziert, volle Suite grün (10 Dateien).
  - Klassifikations-Backfill in 3 Iterationen gehärtet (Dry-Run-Gate):
    (1) Behörden-Signal in den Prompt (Gegenseite ist das stärkste
    Indiz; Stadt/Kreis ≠ automatisch sonstiges) — fixte 215/25, 044/26,
    088/24; (2) compose_case_material kappt pro Dokument (3k), damit ein
    Riesendokument (Belarus-Report, 008/26) die Bescheide nicht aus dem
    Fenster drückt; (3) null-Semantik: "sonstiges" nur bei positiv
    erkanntem Nicht-Gebiet, bei unerkennbarem Streitgegenstand (nur
    Fallname, Akte in j-lawyer) antwortet das Modell null und der Fall
    bleibt NULL (= sicheres Legacy-Verhalten) statt falsch "sonstiges".
  - Live: 41 klassifiziert, 14 bewusst NULL. Verteilung: 32 asyl,
    8 aufenthalt (inkl. manuell 114/26, 111/26, 215/25), 2 sozial,
    1 miete, 1 sonstiges (Henssen/Warner Bros), 11 NULL.
  - REVISION: 008/26 Lisouskaya ist aufenthalt, nicht sozial — mit
    Breiten-Material erkannte Qwen den Kern (Anhörung Ausländerbehörde,
    Passbeschaffung Belarus, Unzumutbarkeit); der Jobcenter-Bescheid ist
    ein Nebenstrang. Die frühere sozial-Empfehlung beruhte auf dem
    verengten Jobcenter-Probe-Material.
  - Akzeptanz: Facetten-Backfill überspringt Nicht-Migrationsfälle
    (3 Fälle von "ohne Ergebnis" nach "übersprungen" gewandert).
- 2026-07-06 — Multi-Label (Jays Rückfrage: "Lisouskaya ist beides"):
  - cases.rechtsgebiete (JSONB-Liste) als Quelle der Wahrheit bei
    Mehrfach-Zuordnung; rechtsgebiet bleibt Primärgebiet (= erstes
    Element). Migration 2026-07-06_case_rechtsgebiete.
  - uses_asyl_layers nimmt Key ODER Liste: Migrationsfall, wenn
    irgendein Gebiet Migrationsrecht ist; leere Liste/NULL = Legacy.
    Alle drei Gates lesen die Liste vor dem Einzelfeld.
  - PUT /cases/{id}/rechtsgebiet und POST /cases akzeptieren String
    oder Liste; null löscht beide Felder.
  - 008/26 Lisouskaya = ["aufenthalt", "sozial"] — Asyl-Schichten
    bleiben an, der Sozialrechts-Strang ist ab Stufe 1 adressierbar.
- 2026-07-06 — j-lawyer-Sync (Jay: "yes lets get that from j-lawyer"):
  - Befund: j-lawyer subjectField ist tot (im List-Endpoint absent,
    einzeln meist leer); das real gepflegte Feld ist reason ("wegen ..."),
    inkl. Mehrfach-Kodierung ("Asyls/Aufenthalts", 187×).
  - gebiete_from_reason in rechtsgebiete.py: Keyword-Regeln per Sign-off
    (asyl|aufenthalt inkl. Einbürgerung/Duldung/Ausländerrecht|
    sozialleistung→sozial|miet→miete|forderung/kaufvertrag→inkasso,
    Mietvertrag-Lookbehind|straftat/owi→sonstiges); unerkannter Freitext
    → leere Liste = kein Sync. Dry-Run fand die Lücke "Ausländerrecht"
    → Pattern ergänzt (TDD).
  - sync_rechtsgebiet_jlawyer.py: Az-Match über extract_file_number,
    ADDITIVER Merge (nie entfernen, manuell Gesetztes bleibt, Primär
    nur wenn leer). Live: 14 aktualisiert, 33 unverändert, 8 ohne
    j-lawyer-Match (Testfälle/Namen ohne Az). Nur noch 3 Fälle NULL,
    9 Multi-Label (u.a. 067/26 sozial+aufenthalt, 089/26
    asyl+aufenthalt via "Ausbildungsduldung").
  - j-lawyer bestätigt 008/26-Framing: "./. Jobcenter Düsseldorf",
    reason "Sozialleistungen" — die Multi-Label-Entscheidung war richtig.
  - OFFEN: Intake-Anbindung (gemma-intake setzt Rechtsgebiet per
    PUT /cases/{id}/rechtsgebiet bei Fallanlage).
  - Rollout-Schritte (nach Go): merge → Container-Neustart (Migration)
    → backfill_rechtsgebiet.py --dry-run → Liste prüfen → live →
    008/26 als Akzeptanz (rechtsgebiet=sozial, kein Facetten-Versuch im
    nächsten Nachtlauf) → Intake-Anbindung (PUT) in gemma-intake.

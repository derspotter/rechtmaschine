# case_assessment Härtung nach Codex-Review — Design

Stand: 07.09.2026. Basis: `docs/superpowers/specs/2026-09-03-case-assessment-memory-design.md`
(Fassung 2) und der Codex-Review der Branch
`docs/superpowers/specs/2026-09-07-case-assessment-branch-review-codex.md`.
Vorgabe von Jay: die Befunde beheben, aber nicht überbauen. Jede Maßnahme unten
ist die kleinste, die den Befund schließt.

## 1. Proposal-Mechanik (Befunde 1, 6, Minor "notizen")

**Erstellung.** `create_memory_update_proposal` lehnt ein Proposal mit
`expected_version != aktuelle Target-Version` mit 400 ab (wie `put`). Der
Dry-Run gegen den aktuellen Inhalt bleibt.

**Rebase: alles oder nichts.** `rebase_assessment_ops(ops, new_content,
conflict_ids)` gibt entweder die vollständige Op-Liste zurück oder eine leere
Liste (= Proposal wird superseded). Ein Proposal ist konfliktbehaftet, wenn
mindestens eine Op

- eine Gutachten-id aus `conflict_ids` betrifft (remove/set/append),
- ein `append` mit einer id ist, die im neuen Inhalt schon existiert,
- `/notizen` setzt und `notizen` in `conflict_ids` steht (Accept hat notizen
  geändert; der Aufrufer trägt `"notizen"` in `curated_fields` ein, wenn sich
  der Wert geändert hat),
- oder die Gesamtliste nicht mehr auf den neuen Inhalt anwendbar ist
  (`apply_assessment_ops` wirft `ValueError`).

Es werden nie einzelne Ops verworfen. Damit kann ein `remove` nicht ohne sein
`append` überleben.

## 2. Server-Felder und Größen (Befunde 7, 8, Minor `store_checked_at`)

- **Abgleich für jede berührte id.** Der Accept gleicht alle Gutachten ab,
  deren id von einer Op adressiert wurde (`append`/`set`), zusätzlich zu
  `changed_gutachten_ids`. Ein inhaltsgleiches `set` wird damit neu
  verifiziert statt still auf `unchecked` zu fallen. Neue Hilfsfunktion
  `touched_gutachten_ids(ops) -> set` in `assessment_memory.py`.
- **Größe ohne Server-Felder messen.** `validate_assessment_content` misst
  `MAX_GUTACHTEN_BYTES` und `MAX_CONTENT_BYTES` auf einer Kopie ohne
  `store`, `store_entry_id`, `store_checked_at`. Die Anreicherung kann ein
  gültiges Gutachten dann nicht mehr über die Grenze schieben.
- **Validierung außerhalb des Störungs-Guards.** In
  `accept_memory_update_proposal` bleibt nur das Laden der Store-Map im
  `try/except`. `reconcile_store` läuft danach ohne Guard; ein
  `ValueError` daraus wird zum 400 wie jede andere Validierung.
- **Recheck schreibt immer `store_checked_at`** für jede geprüfte
  Fundstelle. Version und Revision werden nur erhöht, wenn sich `store` oder
  `store_entry_id` geändert hat (`count_store_changes` bleibt maßgeblich).

## 3. Eine Zitat-Identität (Befunde 2, 3, 4, 5, Minor Duplikate)

Neues Modul `app/citation_identity.py`, ohne DB, ohne Imports aus
`endpoints`:

```python
def canonical_az(az: str) -> str            # = verify_source.az_for_compare, einziger Einstieg
def find_citations(text: str) -> list[Citation]
# Citation: kind in {"decision", "az"}, start, end, raw, az, canonical,
#           date (nur decision), az_start, az_end (Spanne des Az im raw)
```

`find_citations` erkennt in dieser Reihenfolge:

1. **Normspannen**, die für die Az-Suche gesperrt werden: `§ ...`-Zitate mit
   Abs./S./Nr., EU-Normen (`Art. 14 Abs. 2 RL 2008/115/EG`, `VO (EU)
   Nr. 604/2013`, `RL 2011/95`). Kein Treffer darin.
2. **Vollzitate** über `draft_citation_ingest.DECISION_RE`, erweitert um Az
   mit Punkt (`10 ZB 22.1187`), ein- bis zweibuchstabiges Gerichtspräfix
   (`M 10 K 21.3767`), EuGH (`C-151/22`), EGMR (`Nr. 12345/19`).
3. **Nackte Az** außerhalb von Normspannen und Vollzitaten: dieselben
   Az-Formen, zusätzlich kompakt (`18E491/12`).

Verbraucher:

- **`strip_foreign_citations(text, whitelist)`** nutzt `find_citations`,
  entfernt Vollzitate als Spanne und nackte Az als Token, Vergleich nur über
  `canonical`. Keine Präfix/Kern-Äquivalenz mehr. Die Destillation wendet
  es auf **jedes** persistierte Stringfeld an: `title`, `summary`, `tags`,
  die vier Listen und die Stringwerte in `fingerprint`.
- **`_entry_violations`** nimmt nur noch `date` und die Az-Spanne eines
  Vollzitats aus; ein Name zwischen Gericht und "Beschluss vom" bleibt
  prüfbar. Vergleich casefold.
- **`_forbidden_tokens`** liest zusätzlich den Gutachten-Inhalt
  (`render_assessment_for_wiki`-Quelle), ohne das Feld `stand` und ohne die
  `datum`-Werte der Fundstellen. `_EU_NORM_TOKEN_RE` entfällt zugunsten der
  Normspannen.
- **`citation_verifier`**: der Blocklist-Check läuft über `find_citations`
  und `canonical_az`. `verify_facts_with_sources(text, memory, sources,
  blocked_az=...)` bekommt die Blocklist; die Route `/workflow/verify-facts`
  sammelt sie über `collect` aus `get_case_memory_prompt_context` (auch bei
  `pseudonymize_for_cloud=False`) und entfernt die Zeilen "Nicht
  zitierfähig" aus dem Korpus, bevor Fakten gestützt werden. Der Check läuft
  auch bei leerem Restkorpus.
- **`verified_az_whitelist`** liefert `canonical_az`.
- **Grounding** (`assessment_blocked_az`) enthält nur `canonical_az`-Werte.
- **Validierung**: zwei Fundstellen mit gleichem `canonical_az` in einem
  Gutachten sind ein 400.

`_BARE_AZ_RE`, `_DECISION_CITATION_RE` in `pattern_wiki.py` und `_FACT_AZ_RE`
in `citation_verifier.py` werden durch das Modul ersetzt.

## 4. Skills-Seite (Befunde 9, 10, Minor Erinnerung)

- **Recheck-Route** (`POST /memory/cases/{id}/assessment/recheck`): ohne
  aktives Gutachten sofort `{"skipped": "keine Gutachten"}` ohne Store-Laden.
  Rate-Limit `60/hour` → `300/hour`. Der nächtliche Recheck bleibt je Akte;
  `recheck_assessments` zählt Fehlschläge und `main` gibt am Ende
  `recheck: ok=N skipped=N failed=N` aus, Exit 1 bei `failed>0`.
- **Eigentümer unbekannt = überspringen.** `is_foreign_case` gibt
  `"unbekannt"` zurück, wenn `case_owner` scheitert. Triage-Zielauswahl und
  Stop-Hook behandeln `"unbekannt"` wie fremd (kein Lesen, eine Zeile im
  Log).
- **Zeitbudget in die Subprozesse.** `_rm` nimmt `timeout=min(CLI_TIMEOUT,
  verbleibendes Budget)`; der Hook reicht seine Deadline durch.
- **Erinnerung nur für neue Proposals.** Die Vermerk-Erinnerung zählt nur
  case_assessment-Proposals mit `created_at` nach dem Hook-Baseline-Zeitpunkt
  (der Hook schreibt beim Anlegen seines State-Files zusätzlich `started_at`
  als ISO-Zeitstempel; Proposals mit `created_at` davor zählen nicht).

## 5. Kleine Befunde, mitgenommen

- Registry-Tests stellen Stubs per Fixture wieder her
  (`monkeypatch.setitem(sys.modules, ...)`), Reihenfolge-unabhängig.
- Accept-Warnungen im UI bleiben stehen: `loadCaseMemory()` löscht den
  Warnungstext nicht mehr (eigenes Element).
- `render_assessment_for_wiki` gibt `(text, rendered_ids)` zurück; die
  Whitelist wird nur aus tatsächlich gerenderten Gutachten gebildet.

## Bewusst nicht

- Staffelweise Kürzung je Gutachten im Cloud-Block (Minor, Budget 4000 reicht
  in der Praxis).
- Nachträgliche Entwertung aktiver Wiki-Einträge nach einem Recheck.
- Index-Umbenennung in der Migration, Spaltenauswahl im Store-Scan,
  Gutachten-ids in `PatternWikiSource`.
- Batch-Recheck-Endpunkt.

## Abnahme

Tests je Punkt (Rebase remove+append, inhaltsgleiches `set`, Größe mit
Server-Feldern, Blocklist über `verify-facts`, Normspannen und bayerische
Az, Wiki-Felder title/tags/fingerprint, Recheck-Skip, unbekannter
Eigentümer). Live auf 157/26: Proposal mit `remove`+`append` derselben id
neben einem zweiten Proposal, inhaltsgleiches `set`, `verify-facts` mit einer
gesperrten Fundstelle im Text, `wiki distill`.

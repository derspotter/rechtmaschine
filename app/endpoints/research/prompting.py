"""Reusable research prompt fragments for standardized retrieval behavior."""

RESEARCH_BASE_CONTEXT = (
    "Du arbeitest nur auf Basis der Nutzeranfrage und der übergebenen Dokumente/Fallkontextdateien. "
    "Analyse zuerst den Fallkontext vollständig und extrahiere daraus die konkrete rechtliche Kernfrage, "
    "die Begründungslage, den Verfahrensstand und alle entscheidungstragenden Tatsachen. "
    "Erzeuge anschließend Suchanfragen nur aus diesen Fakten und Rechtsfragen."
)

RESEARCH_QUERY_BLOCK = """Recherchestrategie (verbindlich):
1. Leite aus dem Fallkontext 5–10 konkrete Suchformulierungen für Primärentscheidungen ab.
2. Jede Suchanfrage muss mindestens zwei sachlich-natürliche Elemente der Falllage enthalten
   (z. B. Zielgruppe, Schutzgrund, Verfahrenssituation, Region/Lebenslage, Rechtsstreitpunkt).
3. Suche zuerst direkt nach vergleichbaren Entscheidungen (Urteil/Beschluss + Gericht + Aktenzeichen + Datum),
   erst danach nach ergänzenden Kontextquellen.
4. Prüfe vor allem Entscheidungen der letzten Jahre; ältere Leitentscheidungen nur bei klarer Relevanz.
5. Wenn es nur ein Dokument gibt: Suche muss aus der Dokumentlage abgeleitet sein, kein generisches Stichwort-Set.
"""

RESEARCH_PRIORITY_BLOCK = """Priorität:
1. Primärentscheidungen mit eindeutigem Entscheidungskern (Urteil/Beschluss) vor allen anderen Quellen.
2. Offizielle, zitierfähige Quellen bevorzugen (Gerichtsportale, amtliche Datenbanken, veröffentlichte Aktenzeichen).
3. Richterliche Ebenen nach Relevanz gewichten (OVG/VG/BVerwG/BVerfG/EGMR/EuGH), nicht pauschal nach einem Gerichtstyp.
4. NRW-Entscheidungen sind dann relevant, wenn der Sachverhalt die Vergleichbarkeit plausibel stützt.
5. Dokument-zu-Entscheidungs-Fit vor allgemeinem Trefferwert priorisieren.
6. Keine Entscheidungen ohne erkennbare Sachverhaltsbeziehung auswerten.

Vermeide:
- Presse-/Newsartikel, Kanzlei-/Marketingtexte, Blogbeiträge, Kommentare.
- Reine Normen-/Gesetzestexte ohne konkrete fallbezogene Rechtsprechungsentscheidung.
- Quellen ohne URL-Entscheidungssignatur (Gericht/Datum/Aktenzeichen)."""

RESEARCH_OUTPUT_CONSTRAINTS = """Ausgabe:
- Nenne primär 3–8 hochrelevante Entscheidungen, die direkt der Falllage zugeordnet sind.
- Gib direkte, stabile Links zur Entscheidungsstelle an (Original-URL, nicht nur Indexseite).
- Gib nur Treffer zurück, die tatsächlich inhaltlich mit der Falllage übereinstimmen."""


def build_research_priority_prompt(additional_context: str = "") -> str:
    """Build a reusable prompt block for legal research engine instructions."""
    lines = [RESEARCH_BASE_CONTEXT.strip(), RESEARCH_QUERY_BLOCK.strip()]
    if additional_context:
        lines.append(additional_context.strip())
    lines.append(RESEARCH_PRIORITY_BLOCK.strip())
    lines.append(RESEARCH_OUTPUT_CONSTRAINTS.strip())
    return "\n\n".join(lines)

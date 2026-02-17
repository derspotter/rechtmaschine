"""Reusable research prompt fragments for standardized retrieval behavior."""

RESEARCH_BASE_CONTEXT = (
    "Lese das zugehörige Dokument oder den bereitgestellten Kontext zuerst vollständig. "
    "Bestimme daraus die Kernfrage, die Entscheidungsgründe und die konkrete Fallkonstellation. "
    "Leite daraus eine gerichtsspezifische Recherche nach ähnlichen Entscheidungen her, "
    "mit Fokus auf deutsche Rechtsprechung und belastbare Primärentscheidungen. "
    "Nutze Suchbegriffe, die Vergleichsfälle abbilden (z. B. Herkunftsregion, "
    "Behinderungsgründe, Wehrdienst-/Kriegsdienstverweigerung, "
    "Dublin-/Überstellungsfragen, § 60 Abs. 5/7-AufenthG, "
    "medizinische bzw. humanitäre Abschiebungshindernisse, "
    "Vulnerabilität (z. B. Minderjährige, Alleinerziehende, Traumatisierung), "
    "Art. 3 EMRK-Risikolagen, familiäre Verfolgungsrisiken oder andere relevante Schutzgründe). "
    "Stelle Relevanz über bloße Schlüsselworttreffer und richte die Suche auf vergleichbare Fälle aus."
)

RESEARCH_PRIORITY_BLOCK = """Priorität:
1. Priorisiere aktuelle, höchstrichterliche Primärentscheidungen als Kernbestandteil (BVerfG, BVerwG, BGH, EuGH, EGMR, OVG).
2. Ergänze anschließend belastbare landesrechtliche Entscheidungen (insbesondere NRW), wenn sie sachlich relevant sind.
3. Leite die Suchanfragen direkt aus dem Einzelfall ab (Dokumentanalyse, nicht aus vorgegebenen Gerichtsnamen).
4. Suche auf Quellen mit klarer Entscheidungssignatur (Beschluss, Urteil, Aktenzeichen, Datum).
5. Bevorzuge offizielle Primärquellen (Gerichtsdatenbanken, offizielle Gerichtsportale, juristische Veröffentlichungsdienste).
6. Sortiere Treffer strikt nach Relevanz und Aktualität (neueste zuerst; bei gleicher Relevanz nach Entscheidungsniveau).
7. Fördere gerichtliche Diversität:
   - Bei gleicher Relevanz zuerst Treffer aus unterschiedlichen Gerichtsebenen bewerten.
   - Priorisiere nicht allein nach einem Gerichtstyp, wenn passende, vergleichbare Entscheidungen anderer Ebenen vorhanden sind.
   - Mehrere Treffer derselben Gerichtsbarkeit sind erlaubt, solange sie die Relevanz klar erhöhen.
8. Gib für jede zitierte Entscheidung eine direkte URL zum Originaldokument an.
9. Lege ein Mindestgewicht auf vergleichbare Instanzen mit anderen Gerichten als BVerwG.

Vermeide:
- Pressetexte, Kommentierungen, News- und Blogbeiträge.
- Inhalte ohne nachvollziehbaren Entscheidungskern.
- Treffer ohne eindeutige Primärquelle."""


def build_research_priority_prompt(additional_context: str = "") -> str:
    """Build a reusable prompt block for legal research engine instructions."""
    lines = [RESEARCH_BASE_CONTEXT.strip()]
    if additional_context:
        lines.append(additional_context.strip())
    lines.append(RESEARCH_PRIORITY_BLOCK.strip())
    return "\n\n".join(lines)

"""Reusable research prompt fragments for standardized retrieval behavior."""

RESEARCH_BASE_CONTEXT = (
    "Recherchiere webbasiert im deutschen Migrations- und Ausländerrecht. "
    "Arbeite quellenbasiert und belege Aussagen mit konkreten, prüfbaren Primärquellen."
)

RESEARCH_PRIORITY_BLOCK = """Priorität:
1. Priorisiere aktuelle, höchstrichterliche Primärentscheidungen als Kernbestandteil (BVerfG, BVerwG, BGH, EuGH, EGMR, OVG).
2. Ergänze anschließend belastbare landesrechtliche Entscheidungen (insbesondere NRW), wenn sie sachlich relevant sind.
3. Berücksichtige nur Quellen mit klaren Entscheidungssignalen (Beschluss, Urteil, Aktenzeichen, Datum).
4. Ziehe ausschließlich offizielle Primärquellen vor (Gerichtsdatenbanken, offizielle Gerichtsportale, juristische Veröffentlichungsdienste).
5. Sortiere die Treffer intern strikt nach Aktualität und Relevanz:
   - Neueste Entscheidungen vor älteren, bei gleicher Relevanz nach Entscheidungsniveau.
6. Nutze für jede zitierte Entscheidung eine direkte URL zum Originaldokument.

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

"""Reusable research prompt fragments for standardized retrieval behavior."""

RESEARCH_BASE_CONTEXT = (
    "Recherchiere webbasiert im deutschen Migrations- und Ausländerrecht. "
    "Arbeite quellenbasiert und belege Aussagen mit konkreten, prüfbaren Primärquellen."
)

RESEARCH_PRIORITY_BLOCK = """Priorität:
1. Neueste und rechtlich relevante Entscheidungen höherer Instanzen (BVerfG, BVerwG, BGH, OVG, EuGH, EGMR), idealerweise mit Datum und Aktenzeichen.
2. Relevante Entscheidungen aus NRW (OVG NRW, VG Düsseldorf, VG Münster, VG Köln, VG Aachen, VG Arnsberg, LSG NRW sowie vergleichbare Landesgerichte).
3. Offizielle Primärquellen (Gerichte, Behörden, EUAA, UNHCR, Gesetzes- und Verwaltungstexte).
4. Konkrete, zitierfähige Fundstellen mit direkter URL.
5. Bei abweichender Rechtsprechung explizit differenzieren, warum Entscheidungen auseinandergehen.

Vermeide:
- Pressemitteilungen, Blogs und reine Nachrichtenberichte als zentrale Quelle.
- Zusammenfassungen ohne direkten Link zum Originaldokument."""


def build_research_priority_prompt(additional_context: str = "") -> str:
    """Build a reusable prompt block for legal research engine instructions."""
    lines = [RESEARCH_BASE_CONTEXT.strip()]
    if additional_context:
        lines.append(additional_context.strip())
    lines.append(RESEARCH_PRIORITY_BLOCK.strip())
    return "\n\n".join(lines)

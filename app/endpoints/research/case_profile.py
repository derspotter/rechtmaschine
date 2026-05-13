"""Case profile extraction and prompt helpers for research workflows."""

import asyncio
import json
import os
from typing import Dict, List, Optional

from shared import (
    Document,
    ResearchCaseProfile,
    get_openai_client,
    load_document_text,
    resolve_openai_model,
)


CASE_PROFILE_MODEL = (
    os.getenv("RESEARCH_CASE_PROFILE_MODEL", "gpt-5.5").strip() or "gpt-5.5"
)
CASE_PROFILE_MAX_OUTPUT_TOKENS = int(
    os.getenv("RESEARCH_CASE_PROFILE_MAX_OUTPUT_TOKENS", "3200")
)
CASE_PROFILE_MAX_TOTAL_CHARS = int(
    os.getenv("RESEARCH_CASE_PROFILE_MAX_TOTAL_CHARS", "180000")
)
CASE_PROFILE_MAX_CHARS_PER_DOC = int(
    os.getenv("RESEARCH_CASE_PROFILE_MAX_CHARS_PER_DOC", "60000")
)


def _extract_openai_text(response: object) -> str:
    text = getattr(response, "output_text", "") or ""
    if text:
        return text

    output = getattr(response, "output", None) or []
    parts: List[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type in ("output_text", "text"):
                parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def _extract_json_object(raw: str) -> Optional[Dict[str, object]]:
    content = (raw or "").strip()
    if not content:
        return None

    content = content.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(content[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def _build_document_sections(documents: List[Document]) -> List[str]:
    sections: List[str] = []
    used_chars = 0

    for idx, document in enumerate(documents, start=1):
        text = load_document_text(document) or ""
        if not text and document.explanation:
            text = document.explanation
        if not text:
            continue

        remaining = CASE_PROFILE_MAX_TOTAL_CHARS - used_chars
        if remaining <= 0:
            break

        take = min(len(text), CASE_PROFILE_MAX_CHARS_PER_DOC, remaining)
        snippet = text[:take]
        used_chars += len(snippet)
        truncated_note = ""
        if len(text) > take:
            truncated_note = "\n(Hinweis: Dokumenttext wurde aus technischen Gründen gekürzt.)"

        sections.append(
            f"Dokument {idx}: {document.filename}\n"
            f"Kategorie: {document.category}\n"
            "---\n"
            f"{snippet}\n"
            f"---{truncated_note}"
        )

    return sections


def build_case_profile_extraction_prompt(
    documents: List[Document],
    user_query: str = "",
) -> Optional[str]:
    sections = _build_document_sections(documents)
    if not sections:
        return None

    query_block = (user_query or "").strip() or "(Keine zusätzliche Nutzerfrage)"
    docs_blob = "\n\n".join(sections)

    return f"""Du bist ein Extraktionsmodell für deutsche migrations- und asylrechtliche Fallstruktur.

Aufgabe:
Analysiere ausschließlich die beigefügten Dokumente und extrahiere daraus:
1. case_fingerprint
2. search_plan
3. ranking_profile

Ziel:
Die Ausgabe wird anschließend verwendet, um
- gezielte Rechtsprechungsrecherche durchzuführen
- Suchanfragen für mehrere Provider abzuleiten
- gefundene Entscheidungen im Meta-Ranking nach ihrer Fallpassung zu bewerten

Regeln:
- Arbeite nur mit Informationen aus den beigefügten Dokumenten.
- Erfinde nichts.
- Wenn etwas unklar oder nicht belegt ist, verwende "" oder [] oder einen niedrigen confidence-Wert.
- Priorisiere individuelle Tatsachen, Verfahrensart, tragende Streitfragen und konkrete Gefährdungs- oder Zurechnungsmechanismen.
- Vermeide generische Leerformeln.
- Wenn du eine allgemeine Leitidee wie Tatsachennähe oder Verfahrensnähe verwendest, formuliere sie fallspezifisch konkret.
- Alle inhaltlichen Listen müssen konkret auf den vorliegenden Fall zugeschnitten sein.
- Für zentrale Aussagen liefere evidence-Einträge.
- Suche soll primär auf vergleichbare Entscheidungen zielen, nicht auf allgemeine Länderinformationen.
- preferred_source_types darf nur Werte aus dem erlaubten Katalog enthalten.
- preferred_outcomes darf nur Werte aus dem erlaubten Katalog enthalten.

Erlaubter Katalog für preferred_source_types:
- court_decision
- higher_court_decision
- constitutional_court_decision
- supranational_court_decision
- administrative_decision
- official_country_report
- curated_legal_database_entry

Erlaubter Katalog für preferred_outcomes:
- claimant_positive
- negative_if_authoritative
- mixed
- landmark_regardless_of_outcome

Bedeutung von negative_if_authoritative:
Negative Entscheidungen sollen nur dann priorisiert werden, wenn sie von einem fallrelevanten gewichtigen Gericht stammen oder die spätere Argumentation realistisch prägen.

Wichtige Anweisung für ranking_profile:
- Nenne keine bloß abstrakten Kategorien wie "Tatsachennähe".
- Formuliere stattdessen konkret, zu welchen Tatsachen, Mechanismen, Verfahrenslagen oder Kontexten eine Quelle passen oder nicht passen soll.

Zusätzliche Nutzerfrage:
{query_block}

Dokumente:
{docs_blob}

Ausgabe ausschließlich als JSON in genau diesem Schema:

{{
  "case_fingerprint": {{
    "procedure_type": "",
    "countries_relevant": {{
      "origin_country": "",
      "other_relevant_countries": []
    }},
    "document_types_seen": [],
    "core_legal_questions": [],
    "core_fact_patterns": [],
    "relevant_actors": [],
    "relationship_patterns": [],
    "risk_mechanisms": [],
    "decision_match_requirements": [],
    "decision_mismatch_filters": [],
    "evidence": [
      {{
        "field": "",
        "value": "",
        "document": "",
        "page_hint": "",
        "quote": ""
      }}
    ],
    "confidence": 0.0
  }},
  "search_plan": {{
    "search_objective": "",
    "search_queries": [],
    "must_cover": [],
    "avoid": [],
    "preferred_source_types": [],
    "preferred_outcomes": [],
    "preferred_recency_years": 0
  }},
  "ranking_profile": {{
    "primary_match_dimensions": [],
    "downgrade_if": [],
    "prefer_if": []
  }}
}}

Zusätzliche Qualitätsregeln:
- search_queries sollen konkrete, entscheidungsorientierte Web-Suchanfragen sein.
- must_cover soll enthalten, welche Elemente in einer guten Vergleichsentscheidung auftauchen sollten.
- avoid soll enthalten, welche thematisch nahen, aber unpassenden Entscheidungen vermieden werden sollen.
- decision_match_requirements und ranking_profile.primary_match_dimensions dürfen sich überschneiden, sollen aber nicht wortgleich dupliziert werden.
- evidence nur für tatsächlich wichtige Punkte, nicht für jede Kleinigkeit.
- confidence als Zahl zwischen 0 und 1.

Gib nur JSON zurück."""


def render_case_profile_for_search(case_profile: Optional[ResearchCaseProfile]) -> str:
    if not case_profile:
        return ""

    search_plan = case_profile.search_plan
    fingerprint = case_profile.case_fingerprint
    lines: List[str] = []

    if fingerprint.countries_relevant.origin_country:
        lines.append(f"- Herkunftsland: {fingerprint.countries_relevant.origin_country}")
    if fingerprint.core_fact_patterns:
        lines.append("- Zentrale Tatsachen:")
        lines.extend(f"  - {item}" for item in fingerprint.core_fact_patterns[:5])
    if search_plan.search_queries:
        lines.append("- Konkrete Suchanfragen:")
        lines.extend(f"  - {item}" for item in search_plan.search_queries[:6])
    if search_plan.must_cover:
        lines.append("- Treffer sollten enthalten:")
        lines.extend(f"  - {item}" for item in search_plan.must_cover[:5])
    if search_plan.avoid:
        lines.append("- Vermeide:")
        lines.extend(f"  - {item}" for item in search_plan.avoid[:5])
    if search_plan.preferred_source_types:
        lines.append(
            "- Bevorzugte Quellentypen: "
            + ", ".join(search_plan.preferred_source_types)
        )
    if search_plan.preferred_outcomes:
        lines.append(
            "- Bevorzugte Ausgangslagen: "
            + ", ".join(search_plan.preferred_outcomes)
        )
    if search_plan.preferred_recency_years:
        lines.append(
            f"- Bevorzugte Aktualität: letzte {search_plan.preferred_recency_years} Jahre"
        )

    if not lines:
        return ""

    return "Extrahierter Fall- und Suchplan:\n" + "\n".join(lines)


async def extract_case_profile(
    documents: List[Document],
    user_query: str = "",
) -> Optional[ResearchCaseProfile]:
    prompt = build_case_profile_extraction_prompt(documents, user_query=user_query)
    if not prompt:
        return None

    try:
        client = get_openai_client()
        response = await asyncio.to_thread(
            client.responses.create,
            model=resolve_openai_model(CASE_PROFILE_MODEL),
            input=prompt,
            reasoning={"effort": "medium"},
            text={"verbosity": "low"},
            max_output_tokens=CASE_PROFILE_MAX_OUTPUT_TOKENS,
        )
        raw_text = _extract_openai_text(response)
        parsed = _extract_json_object(raw_text)
        if not parsed:
            raise ValueError("Case profile response could not be parsed as JSON object")
        return ResearchCaseProfile.model_validate(parsed)
    except Exception as exc:
        print(f"[RESEARCH] Case profile extraction failed: {exc}")
        return None

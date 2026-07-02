"""Page-grounded structured research output (Pillar 1 of the research upgrade).

Pure module: no xai_sdk, no DB, no app imports — host-testable. grok.py wires
these types into the SDK call; the verifier (Pillar 3) consumes the same fields.

Design (docs/research-pipeline-upgrade-plan.md §2): every court decision the
model returns must be grounded in the page it actually opened — Az, Datum and
Ergebnis verbatim from the page, plus a verbatim Zitat. Sources that cannot be
opened and quoted are omitted by prompt contract; `missing_grounding_fields`
makes the gaps machine-checkable.
"""
import inspect
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

QuelleTyp = Literal["entscheidung", "coi", "sonstiges"]
Ebene = Literal["BVerfG", "EuGH", "EGMR", "BVerwG", "OVG", "VG", "sonstige"]
Ergebnis = Literal["stattgegeben", "abgelehnt", "teilweise", "unklar"]
Lager = Literal["stuetzt", "gegen", "neutral"]

#: Fields a court decision must carry to count as page-grounded.
DECISION_GROUNDING_FIELDS = (
    "gericht",
    "datum",
    "aktenzeichen",
    "ergebnis",
    "zitat",
    "lager",
)


class StructuredSource(BaseModel):
    """Single research source; enriched fields are page-grounded for decisions."""

    title: str = Field(description="Title of the source (e.g., court name and decision)")
    url: str = Field(description="Full URL to the source")
    description: str = Field(default="", description="Brief description of the source's relevance")

    quelle_typ: QuelleTyp = Field(
        default="entscheidung",
        description="entscheidung = Gerichtsentscheidung, coi = Erkenntnismittel/Länderinfo, sonstiges",
    )
    gericht: Optional[str] = Field(default=None, description="Gericht, wörtlich von der geöffneten Seite")
    datum: Optional[str] = Field(default=None, description="Entscheidungsdatum (ISO), von der Seite")
    aktenzeichen: Optional[str] = Field(
        default=None, description="Aktenzeichen, WÖRTLICH von der geöffneten Seite — nie aus dem Gedächtnis"
    )
    ebene: Optional[Ebene] = Field(default=None, description="Instanz")
    ergebnis: Optional[Ergebnis] = Field(default=None, description="Tenor-Ergebnis laut Seite")
    profil: Optional[str] = Field(
        default=None, description="Eine Zeile Antragsteller-Profil (Alter/Geschlecht/Gesundheit/Netzwerk)"
    )
    zitat: Optional[str] = Field(
        default=None, description="WÖRTLICHE Passage von der Seite, die die Relevanz belegt"
    )
    fit: Optional[str] = Field(default=None, description="Warum die Entscheidung zur Falllage passt")
    lager: Optional[Lager] = Field(
        default=None, description="stuetzt/gegen/neutral bezogen auf das Klageziel"
    )


class GrokResearchOutput(BaseModel):
    """Structured output for Grok research results."""

    summary: str = Field(description="Detailed summary of research findings in German")
    sources: List[StructuredSource] = Field(
        description="List of relevant sources found during research"
    )


def missing_grounding_fields(source: StructuredSource) -> List[str]:
    """Names of grounding fields a decision source is missing (empty when complete).

    Non-decision sources (coi/sonstiges) have no decision-grounding duties.
    """
    if source.quelle_typ != "entscheidung":
        return []
    return [f for f in DECISION_GROUNDING_FIELDS if not getattr(source, f)]


def normalize_citation_url(url: str) -> str:
    """Canonical form for URL identity checks between grok sources and
    ``response.citations``: ignore scheme, ``www.``, fragments and trailing
    slashes; keep path and query (they distinguish decisions)."""
    from urllib.parse import urlsplit

    parts = urlsplit((url or "").strip())
    host = (parts.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    # default ports are cosmetic: https://a.de:443/x ≡ https://a.de/x
    scheme = (parts.scheme or "").lower()
    if (scheme in ("https", "") and host.endswith(":443")) or (scheme == "http" and host.endswith(":80")):
        host = host.rsplit(":", 1)[0]
    path = (parts.path or "").rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}"


def gate_sources_by_citations(sources, retrieved_urls):
    """Split sources into (kept, dropped) by whether grok actually retrieved
    their URL via web_search. No citations means the model answered WITHOUT
    searching — then nothing is page-grounded and everything is dropped
    (fail closed; review finding #1). Known limitation (finding #8): matching
    is redirect-unaware and path-case-sensitive, so mismatches over-drop —
    conservative in the right direction for a legal pipeline."""
    if not retrieved_urls:
        return [], list(sources)
    retrieved = {normalize_citation_url(u) for u in retrieved_urls}
    kept, dropped = [], []
    for source in sources:
        (kept if normalize_citation_url(source.url) in retrieved else dropped).append(source)
    return kept, dropped


async def run_structured_research_rounds(
    *,
    create_chat,
    make_user_message,
    base_message: str,
    max_rounds: int,
    max_duration_sec: float,
    clock=None,
):
    """SDK-free multi-round research loop (grok.py provides the real client).

    ``create_chat(round_index)`` returns an object with ``append(message)`` and
    ``parse(model_cls) -> (response, parsed)``; ``response.citations`` carries
    the URLs the model actually retrieved via web_search. Sources are gated by
    those citations, deduped by normalized URL across rounds, and later rounds
    are told what is already on the list. Mirrors the legacy loop's stopping
    rules: mode timeout window, and early stop when a follow-up round yields
    fewer than 2 new sources.
    """
    from time import perf_counter

    now = clock or perf_counter
    started_at = now()

    summary = ""
    kept_all = []
    dropped_all = []
    errors = []
    seen = set()
    seen_display = []  # original URLs, for the follow-up-round instruction
    rounds_run = 0

    for round_index in range(max_rounds):
        extra = ""
        if round_index > 0 and seen_display:
            known = "\n".join(f"- {u}" for u in seen_display[:12])
            extra = (
                "\n\nZusatzrunde:\n"
                "Finde weitere relevante, aktuelle Entscheidungen, die nicht in dieser Liste enthalten sind:\n"
                f"{known}\n"
                "Liefere neue Primärquellen mit klarem Entscheidungskern."
            )

        try:
            chat = create_chat(round_index)
            chat.append(make_user_message(base_message + extra))
            result = chat.parse(GrokResearchOutput)
            if inspect.isawaitable(result):
                result = await result
            response, parsed = result
        except Exception as exc:  # noqa: BLE001 — a failed round must not
            # destroy the results already gathered in earlier rounds.
            rounds_run += 1
            errors.append(f"round {round_index + 1}: {type(exc).__name__}: {exc}")
            break  # no point hammering the API after a failure
        rounds_run += 1

        if parsed.summary and len(parsed.summary) > len(summary):
            summary = parsed.summary

        citations = list(getattr(response, "citations", None) or [])
        kept, dropped = gate_sources_by_citations(parsed.sources, citations)
        dropped_all.extend(dropped)

        new_count = 0
        for source in kept:
            key = normalize_citation_url(source.url)
            if not key or key in seen:
                continue
            seen.add(key)
            seen_display.append(source.url)
            kept_all.append(source)
            new_count += 1

        if now() - started_at >= max_duration_sec:
            break
        if round_index >= 1 and new_count < 2:
            break

    if errors and not kept_all and not summary:
        # Review finding #2: a run where no round ever produced anything must
        # FAIL, not return a success-shaped empty result that reads like an
        # honest "no relevant case law found".
        raise RuntimeError("; ".join(errors))

    return {
        "summary": summary,
        "sources": kept_all,
        "dropped": dropped_all,
        "errors": errors,
        "rounds": rounds_run,
    }


def parse_structured_content(content: str, model_cls):
    """Validate SDK response content into ``model_cls``.

    With ``response_format=<model>`` the server enforces the schema, so the
    content is normally clean JSON; code fences are stripped defensively.
    """
    import re

    text = (content or "").strip()
    try:
        return model_cls.model_validate_json(text)
    except Exception:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            raise
        return model_cls.model_validate_json(match.group(1))


def salvage_or_parse(content: str, model_cls):
    """(parsed, salvaged): parse structured content, salvaging the raw model
    text as the summary when the JSON is invalid/truncated — a paid round's
    analysis must never be silently discarded (review finding #6)."""
    try:
        return parse_structured_content(content, model_cls), False
    except Exception:
        return model_cls(summary=(content or "").strip(), sources=[]), True


class LegacyStructuredSource(BaseModel):
    """Faithful 3-field schema of the pre-v2 path (extras ignored, no enums) —
    used ONLY by the RESEARCH_STRUCTURED_V2=false rollback branch (finding #7)."""

    title: str = ""
    url: str = ""
    description: str = ""


class LegacyGrokResearchOutput(BaseModel):
    summary: str = ""
    sources: List[LegacyStructuredSource] = []

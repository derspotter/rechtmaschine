"""Eine Zitat-Identität für alle Verbraucher (Wiki-Stripping, Wiki-Sperrprüfung,
Fakten-Check, Gutachten-Whitelist). Import-leicht: keine DB, keine endpoints.

`canonical_az` ist die frühere `jurisprudence_ingest._az_for_compare`
(verbatim übernommen, dort und in `verify_source` nur noch delegiert)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# --- Normalisierung (verbatim aus jurisprudence_ingest, Stand 07.09.2026) ----
_AZ_COURT_PREFIX = re.compile(
    r"^(?:VG|OVG|VGH|BayVGH|BVerwG|BVerfG|BSG|LSG|SG|BGH|OLG|LG|AG|BFH|FG|EuGH|EGMR)\b[.\s]*",
    re.IGNORECASE,
)


def canonical_az(az: Optional[str]) -> str:
    """Normalize an Az, dropping known-benign format differences (court
    prefix/suffix, [nickname], appended party names, unicode dashes, journal
    citations, internal whitespace) so only substantive discrepancies remain."""
    az = (az or "").translate(str.maketrans({"‑": "-", "–": "-", "—": "-"}))
    az = re.split(r"\s+-\s+", az)[0]
    az = re.sub(r"\s*\[[^\]]*\]", "", az)
    az = re.sub(r"\s*\([^)]*\)", "", az)
    az = _AZ_COURT_PREFIX.sub("", az.strip())
    az = re.sub(r"[\s.]*\b(OVG|VG|VGH)$", "", az.strip())
    return re.sub(r"\s+", "", az).casefold()


# --- Formen -------------------------------------------------------------------
# Az-Formen: EuGH, EGMR (nur mit "Nr."), deutsch mit optionalem ein- bis
# zweibuchstabigem Gerichtspräfix (bayerische VG), kompakt ohne Leerzeichen.
_AZ_FORMS = (
    r"C-\d{1,4}/\d{2}",
    r"Nr\.\s?\d{3,6}/\d{2}",
    r"(?:\b(?!Az\b)[A-Z][A-Za-z]?\s)?\d{1,3}\s[A-Za-z]{1,3}\s\d+[./]\d+(?:\.[A-Z]{1,2})?",
    r"\d{1,3}[A-Z]{1,3}\d{1,5}/\d{2}",
)
AZ_RE = re.compile(r"(?<![\w/.-])(?:" + "|".join(_AZ_FORMS) + r")(?![\w/])")

# EU-Normen, in denen "2 RL 2008/115" wie ein Az aussieht. Optionaler Vorspann
# "Art. 14 Abs. 2" wird mitgesperrt.
NORM_RE = re.compile(
    r"(?:Art\.?\s?\d+[a-z]?(?:\s?(?:Abs\.|Satz|S\.|Nr\.|Buchst\.|lit\.)\s?\w+\.?)*\s?)?"
    r"(?:RL|VO|Richtlinie|Verordnung)\s?(?:\((?:EG|EU|EWG)\)\s?)?(?:Nr\.\s?)?"
    r"\d{2,4}/\d{1,4}(?:/(?:EG|EU|EWG))?"
)

DECISION_RE = re.compile(
    r"(?P<court>BVerwG|BVerfG|EuGH|EGMR|BGH|BSG|BAG|"
    r"(?:OVG|VGH|VG|LSG|SG|LG|AG)\s+[A-ZÄÖÜ][\wäöüß.-]*(?:\s+[A-ZÄÖÜ][\wäöüß.-]*)?)"
    r"[^();\n]{0,80}?"
    r"(?P<kind>Urteil|Beschluss|Gerichtsbescheid|Urt\.|Beschl\.)\s+(?:vom|v\.)\s+"
    r"(?P<date>\d{1,2}\.\d{1,2}\.\d{4})\s*[–—-]\s*"
    r"(?P<az>" + "|".join(_AZ_FORMS) + r")"
)


@dataclass
class Citation:
    kind: str  # "decision" | "az"
    start: int
    end: int
    raw: str
    az: str
    canonical: str
    date: Optional[str] = None
    az_start: int = 0
    az_end: int = 0


def _overlaps(start: int, end: int, spans: List[tuple]) -> bool:
    return any(s < end and start < e for s, e in spans)


def find_citations(text: str) -> List[Citation]:
    """Alle Zitate im Text, nach Position. Normspannen (EU-Recht) sind für die
    Az-Suche gesperrt, Vollzitate haben Vorrang vor nackten Az."""
    text = text or ""
    blocked = [m.span() for m in NORM_RE.finditer(text)]
    hits: List[Citation] = []
    for m in DECISION_RE.finditer(text):
        if _overlaps(*m.span(), blocked):
            continue
        hits.append(Citation(
            kind="decision", start=m.start(), end=m.end(), raw=m.group(0),
            az=m.group("az"), canonical=canonical_az(m.group("az")),
            date=m.group("date"), az_start=m.start("az"), az_end=m.end("az"),
        ))
    taken = blocked + [(h.start, h.end) for h in hits]
    for m in AZ_RE.finditer(text):
        if _overlaps(*m.span(), taken):
            continue
        hits.append(Citation(
            kind="az", start=m.start(), end=m.end(), raw=m.group(0),
            az=m.group(0), canonical=canonical_az(m.group(0)),
            az_start=m.start(), az_end=m.end(),
        ))
    hits.sort(key=lambda h: h.start)
    return hits

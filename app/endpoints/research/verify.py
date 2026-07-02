"""Research citation verifier — deterministic tier (Pillar 3).

Proves grok's per-decision claims against the page the retriever actually
fetched, before anything may carry a "verifiziert" badge:

  - aktenzeichen: claimed Az appears in the page text (normalization-tolerant)
  - zitat:        claimed verbatim quote actually stands on the page (fuzzy;
                  looser threshold for OCR'd scans)
  - ergebnis:     claimed outcome is consistent with the Tenor (rule-based;
                  the Qwen semantic layer is a separate, advisory tier)

Design rules (docs/research-pipeline-upgrade-plan.md §4/§5a):
  - unfetchable pages yield "unverified", never "refuted"
  - failed quote matches on OCR'd scans read "nicht bestätigt (Scan)"
  - coi/sonstiges sources only need reachability
Pure module: stdlib only — host-testable.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Optional

#: Minimum quote length that can prove anything.
MIN_ZITAT_CHARS = 25
#: Fuzzy thresholds for the verbatim-quote check.
ZITAT_THRESHOLD = 0.90
ZITAT_THRESHOLD_OCR = 0.80

_WS_RE = re.compile(r"\s+")

#: Tenor phrases signalling denial vs grant (checked in normalized text).
_DENIAL_MARKERS = (
    "wird abgelehnt",
    "werden abgelehnt",
    "wird abgewiesen",
    "werden abgewiesen",
    "wird zurueckgewiesen",
    "wird zurückgewiesen",
    "hat keinen erfolg",
    "bleibt ohne erfolg",
)
_GRANT_MARKERS = (
    "wird stattgegeben",
    "wird verpflichtet",
    "werden verpflichtet",
    "wird zuerkannt",
    "wird festgestellt, dass ein abschiebungsverbot",
    "aufschiebende wirkung wird angeordnet",
    "wird aufgehoben",
)


@dataclass
class VerifyResult:
    """Outcome of verifying one research source against its fetched page."""

    verifiziert: bool
    checks: Dict[str, Optional[bool]] = field(default_factory=dict)
    notes: str = ""


def _norm_az(value: str) -> str:
    """Aktenzeichen canonical form: lowercase, no whitespace."""
    return _WS_RE.sub("", (value or "").lower())


def _norm_text(value: str) -> str:
    return _WS_RE.sub(" ", (value or "").lower()).strip()


def check_aktenzeichen(aktenzeichen: Optional[str], page_text: str) -> bool:
    """Does the claimed Az appear in the page text (whitespace-insensitive)?"""
    az = _norm_az(aktenzeichen or "")
    if not az:
        return False
    return az in _norm_az(page_text)


def check_zitat(zitat: Optional[str], page_text: str, ocr_applied: bool = False) -> bool:
    """Does the claimed verbatim quote actually stand on the page?

    Fuzzy containment: best matching window in the page must reach the
    threshold (looser for OCR'd scans). Very short quotes prove nothing.
    """
    quote = _norm_text(zitat or "")
    if len(quote) < MIN_ZITAT_CHARS:
        return False
    text = _norm_text(page_text)
    if quote in text:
        return True

    threshold = ZITAT_THRESHOLD_OCR if ocr_applied else ZITAT_THRESHOLD
    window = len(quote)
    step = max(1, window // 4)
    best = 0.0
    for start in range(0, max(1, len(text) - window + 1), step):
        ratio = SequenceMatcher(None, quote, text[start:start + window]).ratio()
        if ratio > best:
            best = ratio
            if best >= threshold:
                return True
    return best >= threshold


def check_ergebnis(ergebnis: Optional[str], page_text: str) -> Optional[bool]:
    """Is the claimed outcome consistent with the Tenor? Rule-based:
    returns None when no claim is made (unklar/missing) or the Tenor gives
    no clear signal — absence of evidence is not refutation."""
    claim = (ergebnis or "").strip().lower()
    if claim in ("", "unklar", "teilweise"):
        return None
    text = _norm_text(page_text)
    denial = any(m in text for m in _DENIAL_MARKERS)
    grant = any(m in text for m in _GRANT_MARKERS)
    if denial == grant:  # neither or both → no clear signal
        return None
    page_outcome = "abgelehnt" if denial else "stattgegeben"
    return claim == page_outcome


def verify_source(grounding: Dict, fetch_result) -> VerifyResult:
    """Combined deterministic gate for one source.

    ``grounding`` is the source's grounding dict (StructuredSource fields);
    ``fetch_result`` a retrieval.FetchResult for the source URL.
    """
    quelle_typ = (grounding or {}).get("quelle_typ", "entscheidung")

    if fetch_result.status != "ok":
        return VerifyResult(
            verifiziert=False,
            checks={"fetch": False},
            notes=f"Seite nicht verifizierbar: {fetch_result.status}",
        )

    if quelle_typ != "entscheidung":
        # COI/sonstige sources carry no decision claims — reachability suffices.
        return VerifyResult(verifiziert=True, checks={"fetch": True}, notes="")

    ocr_applied = bool(getattr(fetch_result, "ocr_applied", False))
    text = fetch_result.text or ""

    checks: Dict[str, Optional[bool]] = {"fetch": True}
    notes = []

    checks["aktenzeichen"] = check_aktenzeichen(grounding.get("aktenzeichen"), text)
    if not checks["aktenzeichen"]:
        notes.append("aktenzeichen nicht auf der Seite gefunden")

    checks["zitat"] = check_zitat(grounding.get("zitat"), text, ocr_applied=ocr_applied)
    if not checks["zitat"]:
        notes.append(
            "zitat nicht bestätigt (Scan)" if ocr_applied else "zitat nicht auf der Seite gefunden"
        )

    checks["ergebnis"] = check_ergebnis(grounding.get("ergebnis"), text)
    if checks["ergebnis"] is False:
        notes.append("ergebnis widerspricht dem Tenor")

    verifiziert = bool(checks["aktenzeichen"]) and bool(checks["zitat"]) and checks["ergebnis"] is not False
    return VerifyResult(verifiziert=verifiziert, checks=checks, notes="; ".join(notes))

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


#: Az shapes in prose: "20 K 2991/24", "2 BvR 460/25", "1 C 10.22", "III ZR 160/94".
_AZ_TEXT_RE = re.compile(
    r"\b(?:\d{1,3}|[IVX]{1,4})\s?[A-Za-z]{1,4}\s?\d{1,5}[./]\d{2,4}(?:\.[A-Z]{1,4})?\b"
)
#: Az tokens in URL paths: nrwe "20_K_2991_24", anwalt24 "iii-zr-160_94".
_AZ_URLTOKEN_RE = re.compile(
    r"(?<![a-zA-Z0-9])(?:\d{1,3}|[ivxIVX]{1,4})[-_][a-zA-Z]{1,4}[-_]\d{1,5}[-_.]\d{2,4}(?![0-9])"
)


def extract_aktenzeichen_hint(source: Dict) -> str:
    """Best-effort Aktenzeichen for a web source: explicit ``case_number``
    first, else the first Az-shaped pattern in title/description, else a
    token from the (decoded) URL path rewritten to canonical form
    (``20_K_2991_24`` → ``20 K 2991/24``). Empty string when nothing looks
    like an Az — such sources honestly stay unverified-by-skip."""
    az = str(source.get("case_number") or "").strip()
    if az:
        return az
    text = " ".join(str(source.get(k) or "") for k in ("title", "description", "summary"))
    text = _WS_RE.sub(" ", text)
    m = _AZ_TEXT_RE.search(text)
    if m:
        return m.group(0)
    from urllib.parse import unquote_plus

    url = unquote_plus(" ".join(str(source.get(k) or "") for k in ("url", "pdf_url")))
    m = _AZ_URLTOKEN_RE.search(url)
    if m:
        parts = re.split(r"[-_]", m.group(0))
        if len(parts) >= 3:
            return " ".join(parts[:-1]) + "/" + parts[-1]
    return ""


async def verify_meta_sources(sources, fetch_fn, limit: int = 8):
    """Reduced deterministic verification for meta/web sources WITHOUT a
    structured grounding block (Gemini/ChatGPT arms of the meta engine).

    Decision-like sources whose Aktenzeichen is claimed (``case_number``) or
    recognizable in title/description/URL get two checks: the page is
    fetchable, and the Az actually appears on it. The result lands as
    ``grounding = {verifiziert, verify_notes, verify_level: "aktenzeichen"}``.
    Deliberately reduced: without datum/ergebnis/zitat these sources can never
    pass the store-writeback gate (grounding_to_extraction_fields) — the badge
    only says "this link really is that decision", not "quote verified".
    """
    import asyncio

    def _candidate_az(source) -> str:
        if isinstance(source.get("grounding"), dict):
            return ""  # structured (grok) sources use the full check
        az = extract_aktenzeichen_hint(source)
        if not az:
            return ""
        if source.get("court") or source.get("evidence_type") == "decision_like":
            return az
        return ""

    candidates = [(s, az) for s in sources if (az := _candidate_az(s))][:limit]

    async def _one(source, az):
        grounding = {
            "quelle_typ": "entscheidung",
            "gericht": str(source.get("court") or "").strip(),
            "aktenzeichen": az,
            "verify_level": "aktenzeichen",
        }
        try:
            url = source.get("url") or source.get("pdf_url") or ""
            fetch_result = await fetch_fn(url)
            if fetch_result.status != "ok":
                grounding["verifiziert"] = False
                grounding["verify_notes"] = f"Seite nicht verifizierbar: {fetch_result.status}"
            elif check_aktenzeichen(az, fetch_result.text or ""):
                grounding["verifiziert"] = True
                grounding["verify_notes"] = "nur Aktenzeichen geprüft (Meta-Quelle ohne Zitat)"
            else:
                grounding["verifiziert"] = False
                grounding["verify_notes"] = "aktenzeichen nicht auf der Seite gefunden"
        except Exception as exc:  # noqa: BLE001 — verification must not kill research
            grounding["verifiziert"] = False
            grounding["verify_notes"] = f"Verifikation fehlgeschlagen: {exc}"
        source["grounding"] = grounding
        return bool(grounding["verifiziert"])

    outcomes = await asyncio.gather(*(_one(s, az) for s, az in candidates))
    return {
        "verified": sum(1 for v in outcomes if v),
        "unverified": sum(1 for v in outcomes if not v),
        "skipped": len(sources) - len(candidates),
    }


async def verify_ranked_sources(sources, fetch_fn, limit: int = 8, verify_fn=None):
    """Verify ranked source dicts in place; returns counts.

    Only sources carrying a ``grounding`` block (structured grok path) are
    checked — at most ``limit`` of them, concurrently. Each checked source
    gains ``grounding["verifiziert"]`` and ``grounding["verify_notes"]``.
    A fetch exception yields unverified-with-note, never a crash.

    ``verify_fn`` (async ``(grounding, fetch_result) -> VerifyResult``) is
    the backend seam: grok.py injects the Qwen verifier here; without it the
    deterministic ``verify_source`` runs. This module stays pure.
    """
    import asyncio

    if verify_fn is None:
        async def verify_fn(grounding, fetch_result):
            return verify_source(grounding, fetch_result)

    candidates = [s for s in sources if isinstance(s.get("grounding"), dict)][:limit]
    skipped = len(sources) - len(candidates)

    async def _one(source):
        grounding = source["grounding"]
        try:
            fetch_result = await fetch_fn(source.get("url") or "")
            result = await verify_fn(grounding, fetch_result)
        except Exception as exc:  # noqa: BLE001 — verification must not kill research
            result = VerifyResult(
                verifiziert=False, checks={"fetch": False},
                notes=f"Verifikation fehlgeschlagen: {exc}",
            )
        grounding["verifiziert"] = result.verifiziert
        grounding["verify_notes"] = result.notes
        return result.verifiziert

    outcomes = await asyncio.gather(*(_one(s) for s in candidates))
    return {
        "verified": sum(1 for v in outcomes if v),
        "unverified": sum(1 for v in outcomes if not v),
        "skipped": skipped,
    }

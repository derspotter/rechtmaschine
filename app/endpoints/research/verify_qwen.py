"""Research-Verifikation durch lokales Qwen (Spec 2026-07-13).

Ersetzt für Grok-Quellen mit strukturiertem Grounding den regelbasierten
Ergebnis-Gate aus verify.py: Qwen liest den gefetchten Seitenausschnitt und
beurteilt alle vier Behauptungen (Aktenzeichen, Zitat, Ergebnis, lager). Die
deterministischen String-Befunde gehen als Fakten-Anker mit in den Prompt.

Fail-Richtung strikt closed: nicht erreichbar, Timeout, unparsebare Antwort
oder "unklar" → verifiziert=False → kein Store-Write-back. Ein
Desktop-Ausfall kostet Recall, nie Korrektheit.

Pur bis auf die injizierte ``qwen_call``-Seam (async prompt→dict); die
impure Anbindung (Service-Wake + call_qwen_json) baut grok.py.
"""
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from .verify import VerifyResult, check_aktenzeichen, check_zitat

#: Zeichenbudget für den Seitenausschnitt (Kopf mit Tenor + Zitat-Region).
VERIFY_PAGE_CHAR_LIMIT = 12_000


def select_page_excerpt(page_text: str, zitat: Optional[str], limit: int = VERIFY_PAGE_CHAR_LIMIT) -> str:
    """Kopf der Seite (Rubrum + Tenor stehen vorn) plus ein Fenster um den
    Zitat-Fundort, wenn das Zitat jenseits des Kopfes steht."""
    text = page_text or ""
    if len(text) <= limit:
        return text

    head_chars = int(limit * 0.7)
    window_chars = limit - head_chars
    head = text[:head_chars]

    probe = (zitat or "").strip()[:60]
    pos = text.find(probe) if probe else -1
    if pos < 0 and probe:
        # Fallback: whitespace-tolerant über die ersten Wörter des Zitats.
        words = probe.split()
        if len(words) >= 3:
            pos = text.find(" ".join(words[:3]))
    if pos > head_chars:
        start = max(head_chars, pos - window_chars // 4)
        region = text[start:start + window_chars]
        return f"{head}\n…[gekürzt]…\n{region}"
    # Zitat im Kopf oder nicht auffindbar: Rest des Budgets ans Seitenende
    # (Tenor steht bei manchen Portalen unten).
    return f"{head}\n…[gekürzt]…\n{text[-window_chars:]}"


def build_verify_prompt(grounding: Dict, page_text: str, det_checks: Dict[str, Any]) -> str:
    """Deutscher Verifikations-Prompt: Behauptung, String-Befunde, Ausschnitt."""
    excerpt = select_page_excerpt(page_text, grounding.get("zitat"))
    det_az = "gefunden" if det_checks.get("aktenzeichen") else "NICHT gefunden"
    det_zitat = "gefunden" if det_checks.get("zitat") else "NICHT gefunden"
    return f"""Du prüfst eine Rechtsprechungs-Behauptung gegen den Text der tatsächlich abgerufenen Fundstelle.

BEHAUPTUNG (aus einer KI-Recherche, kann falsch sein):
- Gericht: {grounding.get('gericht') or '—'}
- Datum: {grounding.get('datum') or '—'}
- Aktenzeichen: {grounding.get('aktenzeichen') or '—'}
- Ergebnis: {grounding.get('ergebnis') or '—'} (abgelehnt = Klage/Antrag ohne Erfolg, stattgegeben = Erfolg für die klagende Seite)
- Lager: {grounding.get('lager') or '—'} (stuetzt = hilft der Klägerseite in Asylsachen, gegen = hilft ihr nicht)
- Wörtliches Zitat: "{grounding.get('zitat') or '—'}"

FAKTEN-ANKER (deterministische String-Suche im Seitentext, verlässlich):
- Aktenzeichen per String-Suche: {det_az}
- Zitat per Fuzzy-String-Suche: {det_zitat}

SEITENTEXT (Ausschnitt):
{excerpt}

Prüfe jede Behauptung NUR gegen diesen Seitentext. Vorsicht bei Aufhebungen:
"wird aufgehoben" kann je nach Instanz Erfolg ODER Misserfolg für die
Klägerseite bedeuten — entscheidend ist, wer am Ende was bekommt.
Antworte ausschließlich mit JSON:
{{
  "aktenzeichen_bestaetigt": true|false,
  "zitat_bestaetigt": true|false,
  "ergebnis_bestaetigt": true|false|"unklar",
  "lager_plausibel": true|false|"unklar",
  "verifiziert": true|false,
  "confidence": 0.0-1.0,
  "begruendung": "ein bis zwei deutsche Sätze"
}}"""


@dataclass
class QwenVerdict:
    verifiziert: bool
    checks: Dict[str, Optional[bool]]
    confidence: float
    begruendung: str
    lager_plausibel: Optional[bool]


def _tri(value: Any) -> Optional[bool]:
    """true/false durchreichen, alles andere (unklar, fehlend, Müll) → None."""
    return value if isinstance(value, bool) else None


def parse_verdict(parsed: Any) -> QwenVerdict:
    """Strukturiertes Qwen-Urteil → QwenVerdict; unbrauchbar → ValueError.

    Das Gesamt-verifiziert wird konservativ NEU berechnet: alle drei
    Kern-Checks müssen positiv bestätigt sein, egal was Qwen ins
    verifiziert-Feld schreibt. lager gated nicht."""
    if not isinstance(parsed, dict) or "verifiziert" not in parsed:
        raise ValueError(f"unbrauchbares Qwen-Verdict: {parsed!r}")
    checks = {
        "aktenzeichen": _tri(parsed.get("aktenzeichen_bestaetigt")),
        "zitat": _tri(parsed.get("zitat_bestaetigt")),
        "ergebnis": _tri(parsed.get("ergebnis_bestaetigt")),
        "lager": _tri(parsed.get("lager_plausibel")),
    }
    verifiziert = (
        checks["aktenzeichen"] is True
        and checks["zitat"] is True
        and checks["ergebnis"] is True
        and parsed.get("verifiziert") is True
    )
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return QwenVerdict(
        verifiziert=verifiziert,
        checks=checks,
        confidence=confidence,
        begruendung=str(parsed.get("begruendung") or "").strip(),
        lager_plausibel=checks["lager"],
    )


async def verify_source_qwen(
    grounding: Dict,
    fetch_result: Any,
    qwen_call: Callable[[str], Awaitable[Dict[str, Any]]],
) -> VerifyResult:
    """Qwen-Gate für eine Quelle; Semantik-kompatibel zu verify.verify_source.

    COI/sonstige Quellen brauchen wie bisher nur Erreichbarkeit (kein
    LLM-Call). Widerspricht Qwen dem lager-Tag, wird es entfernt, damit
    keine falsche Grok-Einschätzung in den Store gelangt."""
    quelle_typ = (grounding or {}).get("quelle_typ", "entscheidung")

    if fetch_result.status != "ok":
        return VerifyResult(
            verifiziert=False,
            checks={"fetch": False},
            notes=f"Seite nicht verifizierbar: {fetch_result.status}",
        )
    if quelle_typ != "entscheidung":
        return VerifyResult(verifiziert=True, checks={"fetch": True}, notes="")

    text = fetch_result.text or ""
    det_checks = {
        "aktenzeichen": check_aktenzeichen(grounding.get("aktenzeichen"), text),
        "zitat": check_zitat(
            grounding.get("zitat"), text,
            ocr_applied=bool(getattr(fetch_result, "ocr_applied", False)),
        ),
    }

    try:
        verdict = parse_verdict(await qwen_call(build_verify_prompt(grounding, text, det_checks)))
    except Exception as exc:  # noqa: BLE001 — fail-closed, Recherche läuft weiter
        return VerifyResult(
            verifiziert=False,
            checks={"fetch": True},
            notes=f"Qwen-Verifikation fehlgeschlagen: {exc}",
        )

    notes = [verdict.begruendung] if verdict.begruendung else []
    if grounding.get("lager") and verdict.lager_plausibel is False:
        notes.append(f"lager-Tag '{grounding['lager']}' laut Qwen nicht plausibel — entfernt")
        grounding.pop("lager", None)

    checks: Dict[str, Optional[bool]] = {"fetch": True}
    checks.update(verdict.checks)
    return VerifyResult(
        verifiziert=verdict.verifiziert,
        checks=checks,
        notes="; ".join(n for n in notes if n),
    )

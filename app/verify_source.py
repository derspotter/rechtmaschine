"""Verify a case-law citation against the firm's Rechtsprechung store.

Closes the gap that draft citations were never checked against what the
Kanzlei already holds: the store carries asyl.net ingests, research-verified
sources and (since 2026-07-21) the wiki media decisions
(source_type="kanzlei_wiki", often unpublished own-practice rulings).

Per claim (Gericht/Datum/Az + optional Aussage/Zitat):
 1. Store lookup by normalized Aktenzeichen (whitespace/hyphen tolerant).
 2. Full text refetch from the entry's source (kanzlei_wiki: media PDF,
    asylnet: fileadmin PDF via M-number with detail-page fallback,
    otherwise: source_url as PDF, HTML tag-strip fallback).
 3. Deterministic anchors: Az string check, Zitat fuzzy check
    (endpoints.research.verify).
 4. Local Qwen judgment modeled on endpoints/research/verify_qwen.py —
    strictly fail-closed: Qwen unreachable, unparsable or "unklar"
    => verifiziert False. A desktop outage costs recall, never correctness.

Run inside the app container:
    docker exec rechtmaschine-app python /app/verify_source.py \
        --az "3 L 4061/25.F.A" --gericht "VG Frankfurt am Main" \
        --datum 10.10.2025 --aussage "Keine Zustellungsfiktion nach § 10 II AsylG bei bestelltem Bevollmächtigten"
    docker exec rechtmaschine-app python /app/verify_source.py --json /tmp/claims.json
    ... --no-qwen        # store lookup + deterministic checks only

Exit codes: 0 = all claims verified, 1 = at least one not verified,
2 = at least one claim not found in the store (verify externally: doktrin
wiki raw -> free full-text portals -> Codex web search).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Optional

import httpx

from database import SessionLocal
from endpoints.research.verify import check_aktenzeichen, check_zitat
from endpoints.research.verify_qwen import select_page_excerpt
from jurisprudence_ingest import download_pdf_text, _az_for_compare
from models import RechtsprechungEntry

ASYLNET_PDF = "https://www.asyl.net/fileadmin/user_upload/{num}.pdf"
_TAG_RE = re.compile(r"<[^>]+>")


def store_lookup(db, az: str) -> list[RechtsprechungEntry]:
    """All active entries whose normalized Az equals the claimed one."""
    want = _az_for_compare(az)
    if not want:
        return []
    hits = []
    for e in db.query(RechtsprechungEntry).filter(
        RechtsprechungEntry.is_active.is_(True),
        RechtsprechungEntry.aktenzeichen.isnot(None),
    ).all():
        if _az_for_compare(e.aktenzeichen) == want:
            hits.append(e)
    return hits


def fetch_fulltext(entry: RechtsprechungEntry) -> tuple[str, str]:
    """(text, source_used). Raises on total failure."""
    candidates: list[str] = []
    if entry.source_type == "asylnet" and entry.source_ref:
        num = re.sub(r"\D", "", entry.source_ref)
        if num:
            candidates.append(ASYLNET_PDF.format(num=num))
    if entry.source_url:
        candidates.append(entry.source_url)
    last_exc: Optional[Exception] = None
    for url in candidates:
        try:
            text = download_pdf_text(url)
            if len(text) >= 200:
                return text, url
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        # HTML fallback (detail pages, portals)
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0 (Rechtmaschine/1.0)"})
            resp.raise_for_status()
            if "pdf" not in (resp.headers.get("content-type") or ""):
                text = _TAG_RE.sub(" ", resp.text)
                text = re.sub(r"\s+", " ", text)
                # asylnet detail page: follow the first fileadmin PDF link
                m = re.search(r'href="(/fileadmin/[^"]+\.pdf)"', resp.text)
                if m:
                    try:
                        return download_pdf_text("https://www.asyl.net" + m.group(1)), url
                    except Exception:  # noqa: BLE001
                        pass
                if len(text) >= 400:
                    return text, url
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"kein Volltext beziehbar ({last_exc})")


def build_prompt(claim: dict[str, Any], page_text: str, det: dict[str, Any]) -> str:
    excerpt = select_page_excerpt(page_text, claim.get("zitat") or claim.get("aussage"))
    det_az = "gefunden" if det.get("aktenzeichen") else "NICHT gefunden"
    det_zitat = ("gefunden" if det.get("zitat") else "NICHT gefunden") if claim.get("zitat") else "—"
    return f"""Du prüfst eine Rechtsprechungs-Behauptung aus einem Schriftsatz-Entwurf gegen den Volltext der Entscheidung aus dem Kanzlei-Bestand.

BEHAUPTUNG (kann falsch sein):
- Gericht: {claim.get('gericht') or '—'}
- Datum: {claim.get('datum') or '—'}
- Aktenzeichen: {claim.get('az') or '—'}
- Behauptete Aussage der Entscheidung: {claim.get('aussage') or '—'}
- Wörtliches Zitat: "{claim.get('zitat') or '—'}"

FAKTEN-ANKER (deterministische String-Suche im Volltext, verlässlich):
- Aktenzeichen per String-Suche: {det_az}
- Zitat per Fuzzy-String-Suche: {det_zitat}

VOLLTEXT (Ausschnitt):
{excerpt}

Prüfe NUR gegen diesen Volltext. Die behauptete Aussage muss sich aus den
tragenden Gründen ergeben, nicht bloß aus wiedergegebenem Parteivortrag oder
zitierter fremder Rechtsprechung. Mit „—" markierte Punkte werden NICHT
behauptet: gib für sie "unklar" an und lasse sie das Gesamturteil
"verifiziert" nicht beeinflussen — "verifiziert" bewertet ausschließlich
die tatsächlich behaupteten Punkte. Antworte ausschließlich mit JSON:
{{
  "gericht_bestaetigt": true|false|"unklar",
  "datum_bestaetigt": true|false|"unklar",
  "aktenzeichen_bestaetigt": true|false,
  "aussage_gestuetzt": true|false|"unklar",
  "zitat_bestaetigt": true|false|"unklar",
  "verifiziert": true|false,
  "confidence": 0.0-1.0,
  "begruendung": "ein bis zwei deutsche Sätze"
}}"""


def _tri(v: Any) -> Optional[bool]:
    return v if isinstance(v, bool) else None


def judge_verdict(claim: dict[str, Any], parsed: Any) -> dict[str, Any]:
    """Conservative recomputation, mirroring verify_qwen.parse_verdict:
    core checks must be positively true; claims not made are not gated."""
    if not isinstance(parsed, dict) or "verifiziert" not in parsed:
        raise ValueError(f"unbrauchbares Qwen-Verdict: {parsed!r}")
    checks = {
        "gericht": _tri(parsed.get("gericht_bestaetigt")),
        "datum": _tri(parsed.get("datum_bestaetigt")),
        "aktenzeichen": _tri(parsed.get("aktenzeichen_bestaetigt")),
        "aussage": _tri(parsed.get("aussage_gestuetzt")),
        "zitat": _tri(parsed.get("zitat_bestaetigt")),
    }
    required = ["aktenzeichen"]
    for field in ("gericht", "datum"):
        if claim.get(field if field != "datum" else "datum"):
            required.append(field)
    if claim.get("aussage"):
        required.append("aussage")
    if claim.get("zitat"):
        required.append("zitat")
    verified = all(checks[r] is True for r in required) and parsed.get("verifiziert") is True
    return {
        "verifiziert": verified,
        "checks": checks,
        "required": required,
        "confidence": parsed.get("confidence"),
        "begruendung": parsed.get("begruendung"),
    }


async def qwen_judge(claim: dict[str, Any], page_text: str, det: dict[str, Any]) -> dict[str, Any]:
    import shared
    from citation_qwen import call_qwen_json
    try:
        await shared.ensure_anonymization_service_ready()
        service_url = os.environ["ANONYMIZATION_SERVICE_URL"]
        parsed = await call_qwen_json(service_url, build_prompt(claim, page_text, det),
                                      num_predict=800)
        return judge_verdict(claim, parsed)
    except Exception as exc:  # noqa: BLE001
        return {"verifiziert": False, "checks": {}, "required": [],
                "confidence": 0.0,
                "begruendung": f"Qwen-Verifikation nicht verfügbar (fail-closed): {exc}"}


async def verify_claim(db, claim: dict[str, Any], use_qwen: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"claim": claim}
    hits = store_lookup(db, claim.get("az") or "")
    if not hits:
        result["status"] = "NOT_IN_STORE"
        return result
    entry = hits[0]
    result["entry"] = {
        "id": str(entry.id), "source_type": entry.source_type,
        "court": entry.court,
        "decision_date": str(entry.decision_date or ""),
        "aktenzeichen": entry.aktenzeichen, "source_url": entry.source_url,
    }
    if len(hits) > 1:
        result["note"] = f"{len(hits)} Store-Treffer, erster verwendet"
    try:
        text, used = fetch_fulltext(entry)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "NO_FULLTEXT"
        result["error"] = str(exc)
        return result
    result["fulltext_source"] = used
    det = {
        "aktenzeichen": check_aktenzeichen(claim.get("az"), text),
        "zitat": check_zitat(claim.get("zitat"), text) if claim.get("zitat") else None,
    }
    result["deterministic"] = det
    if not use_qwen:
        result["status"] = "FOUND" if det["aktenzeichen"] else "AZ_MISMATCH"
        return result
    verdict = await qwen_judge(claim, text, det)
    result["qwen"] = verdict
    result["status"] = "VERIFIED" if verdict["verifiziert"] else "NOT_VERIFIED"
    return result


async def main_async(args) -> int:
    claims: list[dict[str, Any]]
    if args.json:
        with open(args.json) as fh:
            claims = json.load(fh)
        if isinstance(claims, dict):
            claims = [claims]
    else:
        if not args.az:
            print("--az oder --json erforderlich", file=sys.stderr)
            return 2
        claims = [{"az": args.az, "gericht": args.gericht, "datum": args.datum,
                   "aussage": args.aussage, "zitat": args.zitat}]

    db = SessionLocal()
    try:
        results = []
        for claim in claims:
            results.append(await verify_claim(db, claim, use_qwen=not args.no_qwen))
    finally:
        db.close()

    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    statuses = [r["status"] for r in results]
    if any(s == "NOT_IN_STORE" for s in statuses):
        return 2
    ok = {"VERIFIED"} if not args.no_qwen else {"FOUND"}
    return 0 if all(s in ok for s in statuses) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--az", help="Aktenzeichen der behaupteten Entscheidung")
    parser.add_argument("--gericht")
    parser.add_argument("--datum", help="TT.MM.JJJJ oder JJJJ-MM-TT")
    parser.add_argument("--aussage", help="behauptete tragende Aussage")
    parser.add_argument("--zitat", help="behauptetes wörtliches Zitat")
    parser.add_argument("--json", help="Datei mit Claim-Liste (Batch)")
    parser.add_argument("--no-qwen", action="store_true",
                        help="nur Store-Lookup + deterministische Checks")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

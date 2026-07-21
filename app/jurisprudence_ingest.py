"""
Jurisprudence ingest — asyl.net vertical slice.

Fetches recent decisions from the asyl.net Entscheidungsdatenbank, tags them
with the existing Gemini extraction, chunks them, and upserts the chunks into
the debian `jurisprudence` collection for hybrid retrieval. Court-published
decisions are pre-redacted, so there is NO anonymization step (public-source
pipeline).

This slice proves the standing-corpus + hybrid-retrieval path. It does not yet
persist RechtsprechungEntry rows or run incremental refresh — those are the
next increments (see docs/jurisprudence-corpus-build-plan.md).

Run inside the app container:
    docker exec rechtmaschine-app python jurisprudence_ingest.py \
        --query "Afghanistan Abschiebungsverbot" --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Any, Optional

import httpx

import fitz  # PyMuPDF
from google.genai import types
from playwright.async_api import async_playwright

from shared import get_gemini_client
from database import SessionLocal
from models import RechtsprechungEntry
from rag_vocabulary import (
    load_vocabulary, normalize_themen, normalize_country, normalize_normen,
    tag_line, facet_metadata,
)
from endpoints.rechtsprechung_playbook import (
    ArgumentPattern,
    CitationRef,
    RechtsprechungExtraction,
    _normalize_tags,
    _parse_date,
)

# Ranking weight by instance: top/binding courts rank above lower courts on
# equal relevance (used later by freshness/instance-aware retrieval).
_INSTANCE_WEIGHT = {
    "bverfg": 3, "bverwg": 3, "eugh": 3, "egmr": 3,
    "ovg": 2, "vgh": 2,
    "vg": 1, "sg": 1, "ag": 1, "lg": 1,
}


def _instance_weight(court_level: Optional[str], court: Optional[str]) -> int:
    blob = f"{court_level or ''} {court or ''}".lower()
    for key, weight in _INSTANCE_WEIGHT.items():
        if key in blob:
            return weight
    return 0


def persist_entry(db, tags: RechtsprechungExtraction, *, source_type: str,
                  source_url: str, source_ref: Optional[str], content_sha256: str,
                  schlagworte: Optional[list] = None, normen: Optional[list] = None,
                  leitsatz: Optional[str] = None,
                  model_label: str = "asylnet-metadata") -> RechtsprechungEntry:
    """Create a global RechtsprechungEntry from an extraction + source metadata.

    Mirrors create_playbook_entry's field mapping (no document_id, since these
    are external decisions). Caller has already confirmed it is not a duplicate.
    asyl.net's curated Schlagwörter/Normen/Leitsatz (if scraped) are stored in
    dedicated fields and the Schlagwörter also enrich the tag list."""
    decision_date = _parse_date(tags.decision_date)
    tag_list = _normalize_tags(tags.tags)
    country = (tags.country or "").strip() or "Unbekannt"
    if country.lower() not in tag_list:
        tag_list.append(country.lower())
    for sw in (schlagworte or []):
        if sw.lower() not in tag_list:
            tag_list.append(sw.lower())

    entry = RechtsprechungEntry(id=uuid.uuid4())
    entry.country = country
    entry.tags = tag_list
    entry.court = tags.court
    entry.court_level = tags.court_level
    entry.decision_date = decision_date
    entry.aktenzeichen = tags.aktenzeichen
    entry.outcome = tags.outcome or "unknown"
    entry.key_facts = tags.key_facts or []
    entry.key_holdings = tags.key_holdings or []
    entry.argument_patterns = [a.model_dump() for a in (tags.argument_patterns or [])]
    entry.citations = [c.model_dump() for c in (tags.citations or [])]
    entry.summary = tags.summary
    entry.extracted_at = datetime.utcnow()
    entry.model = model_label
    entry.confidence = tags.confidence
    entry.warnings = tags.warnings or []
    entry.is_active = True
    entry.source_type = source_type
    entry.source_url = source_url
    entry.source_ref = source_ref
    entry.content_sha256 = content_sha256
    entry.instance_weight = _instance_weight(tags.court_level, tags.court)
    entry.schlagworte = schlagworte or []
    entry.normen = normen or []
    entry.leitsatz = leitsatz
    entry.created_at = datetime.utcnow()
    entry.updated_at = datetime.utcnow()
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry

ASYLNET_DB_URL = "https://www.asyl.net/recht/entscheidungsdatenbank"
ASYLNET_BASE = "https://www.asyl.net"
_FOOTER_DATE = re.compile(r"(\d{2}\.\d{2}\.\d{4})")
_FOOTER_M = re.compile(r"asyl\.net:\s*(M\d+)", re.IGNORECASE)

import html as _html
_LABEL_RE = {
    "schlagworte": re.compile(r">\s*Schlagw(?:ö|oe)rter\s*:?\s*<.*?>(.*?)</", re.IGNORECASE | re.DOTALL),
    "normen": re.compile(r">\s*Normen\s*:?\s*<.*?>(.*?)</", re.IGNORECASE | re.DOTALL),
}
_HEADNOTE_RE = re.compile(r"rsdb_single_headnote[\"' ]*>(.*?)</div>", re.DOTALL)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def parse_detail_metadata(page_html: str) -> dict[str, Any]:
    """Harvest asyl.net's curated editorial metadata from a decision detail page:
    controlled Schlagwörter, cited Normen, and the editorial Leitsatz."""
    out: dict[str, Any] = {"schlagworte": [], "normen": [], "leitsatz": None}
    for key, rx in _LABEL_RE.items():
        m = rx.search(page_html)
        if m:
            val = _clean(m.group(1))
            out[key] = [p.strip() for p in val.split(",") if p.strip()]
    hn = _HEADNOTE_RE.search(page_html)
    if hn:
        leit = _clean(hn.group(1))
        out["leitsatz"] = leit[:4000] or None
    return out


# asyl.net writes several footer shapes (all observed in production):
#   "VG Berlin, vom 09.10.2025 - 1 K 6/24 A - asyl.net: M33834"
#   "VG Schleswig-Holstein, Beschluss vom 22.10.2025 - 11 B 165/25 - ..."
#   "VG Hamburg Urteil vom 06.06.2025 - 4 A 139/25 - ..."          (no comma)
#   "BMI, Erlass/Behördliche Mitteilung vom 10.04.2025 - - ..."    (no Az)
# The decision type is optional, the Az may be empty (Erlasse), and EuGH Az
# contain hyphens (C-91/20) — hence the non-greedy .*? for az.
_DECISION_TYPES = (
    r"Urteil und Beschluss|Urteil|Beschluss|Gerichtsbescheid|Vergleich|"
    r"Erlass/Beh(?:ö|oe)rdliche Mitteilung|Erlass|Beh(?:ö|oe)rdliche Mitteilung|Stellungnahme"
)
_ERLASS_TYPES = {"erlass", "erlass/behördliche mitteilung", "behördliche mitteilung", "stellungnahme"}
_FOOTER_PARSE = re.compile(
    r"^(?P<court>.+?),?\s*(?:(?P<typ>" + _DECISION_TYPES + r")\s+)?vom\s+"
    r"(?P<date>\d{2}\.\d{2}\.\d{4})\s*-\s*(?P<az>.*?)\s*-\s*asyl\.net:",
    re.IGNORECASE,
)


def canonical_detail_url(url: str, m_number: Optional[str]) -> str:
    """asyl.net list hrefs are occasionally broken (M34106 shipped as /rsdb/m-1,
    a 404 — the entry was unverifiable and had to be deactivated). The M-number
    is authoritative: rebuild the detail URL whenever the two disagree."""
    if m_number and m_number.lower() not in (url or "").lower():
        return f"{ASYLNET_BASE}/rsdb/{m_number.lower()}"
    return url


def extraction_from_asylnet(rec: dict[str, Any], vocab) -> RechtsprechungExtraction:
    """Build a RechtsprechungExtraction from asyl.net curated metadata only (no LLM).
    Court/date/Aktenzeichen come from the structured listing footer; the
    Herkunftsland is the country Schlagwort; tags are the curated Schlagwoerter.
    The AI-only fields (outcome/key_holdings/summary/argument_patterns) stay
    empty. Court decisions without an Az get a warning (Erlasse legitimately
    have none); an unparseable footer falls back to the listing's court heading."""
    footer = rec.get("footer") or ""
    court = aktenzeichen = None
    typ = ""
    date_str = rec.get("footer_date")
    warnings: list[str] = []
    m = _FOOTER_PARSE.search(footer)
    if m:
        court = m.group("court").strip().rstrip(",")
        typ = (m.group("typ") or "").strip().lower()
        date_str = m.group("date")
        # EuGH/EGMR footers append party names after the Az ("C-147/24 [Safi]
        # - V. gegen Niederlande"); Az never contain " - " themselves.
        az = re.split(r"\s+-\s+", m.group("az"))[0]
        # asyl.net sometimes appends a journal citation, e.g. "(Asylmagazin ...)"
        az = re.sub(r"\s*\([^)]*\)\s*$", "", az).strip()
        aktenzeichen = az or None
    if not court:
        court = (rec.get("court_heading") or "").strip() or None
    if not aktenzeichen and typ not in _ERLASS_TYPES:
        warnings.append(
            f"asyl.net-Footer ohne Aktenzeichen ({footer[:80]!r}) — Fundstelle vor Zitierung prüfen"
        )
    court_level = court.split()[0] if court else None
    country = None
    for sw in (rec.get("schlagworte") or []):
        c = normalize_country(vocab, sw)
        if c:
            country = c
            break
    return RechtsprechungExtraction(
        country=country or "Unbekannt",
        tags=rec.get("schlagworte") or [],
        court=court,
        court_level=court_level,
        decision_date=date_str,
        aktenzeichen=aktenzeichen,
        warnings=warnings,
    )


def _norm_az(az: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (az or "")).strip().lower()


_AZ_COURT_PREFIX = re.compile(
    r"^(?:VG|OVG|VGH|BayVGH|BVerwG|BVerfG|BSG|LSG|SG|BGH|OLG|LG|AG|BFH|FG|EuGH|EGMR)\b[.\s]*",
    re.IGNORECASE,
)


def _az_for_compare(az: Optional[str]) -> str:
    """Normalize an Az for footer-vs-LLM comparison, dropping the KNOWN-benign
    format differences (court designator prefix/suffix, [nickname], appended
    party names, unicode dashes, journal citations, internal whitespace) so
    only substantive discrepancies — wrong digits, wrong chamber, wrong
    register letter — surface as warnings."""
    az = (az or "").translate(str.maketrans({"‑": "-", "–": "-", "—": "-"}))
    az = re.split(r"\s+-\s+", az)[0]                 # party names after " - "
    az = re.sub(r"\s*\[[^\]]*\]", "", az)            # EuGH case nicknames
    az = re.sub(r"\s*\([^)]*\)", "", az)             # journal cites, "(Wx)" style
    az = _AZ_COURT_PREFIX.sub("", az.strip())
    az = re.sub(r"[\s.]*\b(OVG|VG|VGH)$", "", az.strip())   # SH-style trailing court token
    return re.sub(r"\s+", "", az).casefold()         # internal whitespace is noise


def merge_footer_and_llm(
    footer: RechtsprechungExtraction,
    llm: Optional[RechtsprechungExtraction],
) -> RechtsprechungExtraction:
    """Belt and braces: the curated footer citation wins for court/date/Az
    (digit-exact, auditable), the LLM full-text extraction supplies the
    semantic fields (outcome/key_facts/key_holdings/summary/patterns) regex
    cannot produce. Where both sides carry a value they cross-check each
    other — disagreement becomes a warning, never a silent override; LLM
    values fill footer gaps only with a 'prüfen' marker."""
    if llm is None:
        return footer

    warnings = list(footer.warnings or []) + list(llm.warnings or [])
    court, date_str, az = footer.court, footer.decision_date, footer.aktenzeichen

    if az and llm.aktenzeichen and _az_for_compare(az) != _az_for_compare(llm.aktenzeichen):
        warnings.append(
            f"Az-Abweichung: Footer '{az}' vs. Volltext '{llm.aktenzeichen}' — Fundstelle prüfen"
        )
    elif not az and llm.aktenzeichen:
        az = llm.aktenzeichen
        warnings = [w for w in warnings if "ohne Aktenzeichen" not in w]
        warnings.append("Aktenzeichen aus Volltext (LLM) ergänzt — vor Zitierung prüfen")

    footer_d, llm_d = _parse_date(date_str), _parse_date(llm.decision_date)
    if footer_d and llm_d and footer_d != llm_d:
        warnings.append(
            f"Datums-Abweichung: Footer {footer_d} vs. Volltext {llm_d} — prüfen"
        )
    elif not footer_d and llm_d:
        date_str = llm.decision_date

    if not court:
        court = llm.court

    country = footer.country if (footer.country or "Unbekannt") != "Unbekannt" else (llm.country or "Unbekannt")
    tags = list(footer.tags or [])
    for t in llm.tags or []:
        if t not in tags:
            tags.append(t)

    return RechtsprechungExtraction(
        country=country,
        tags=tags,
        court=court,
        court_level=court.split()[0] if court else None,
        decision_date=date_str,
        aktenzeichen=az,
        outcome=llm.outcome,
        key_facts=llm.key_facts or [],
        key_holdings=llm.key_holdings or [],
        argument_patterns=llm.argument_patterns or [],
        citations=llm.citations or [],
        summary=llm.summary,
        confidence=llm.confidence,
        warnings=warnings,
    )


async def _collect_page_items(page, out: list[dict[str, Any]], limit: int) -> None:
    for item in await page.query_selector_all("div.rsdb_listitem"):
        link = await item.query_selector("div.rsdb_listitem_court a")
        if not link:
            continue
        href = await link.get_attribute("href")
        if not href:
            continue
        detail_url = href if href.startswith("http") else f"{ASYLNET_BASE}{href}"
        court_heading = re.sub(r"\s+", " ", (await link.text_content() or "")).strip()
        footer_el = await item.query_selector(".rsdb_listitem_footer")
        footer = (await footer_el.text_content() if footer_el else "") or ""
        footer = re.sub(r"\s+", " ", footer).strip()
        date_m = _FOOTER_DATE.search(footer)
        m_num = _FOOTER_M.search(footer)
        detail_url = canonical_detail_url(detail_url, m_num.group(1) if m_num else None)
        out.append({
            "url": detail_url,
            "court_heading": court_heading,
            "footer": footer,
            "footer_date": date_m.group(1) if date_m else None,
            "m_number": m_num.group(1) if m_num else None,
        })
        if len(out) >= limit:
            return


async def fetch_asylnet(query: str, limit: int, datefrom: Optional[str],
                        dateto: Optional[str] = None, max_pages: int = 6) -> list[dict[str, Any]]:
    """Search the asyl.net Entscheidungsdatenbank via its POST form and return
    decisions (detail URL, PDF URL, footer metadata), newest first, paginating
    across result pages up to `limit`/`max_pages`.

    The DB moved from a GET query API to a stateful POST form (old GET URL 404s);
    result markup (div.rsdb_listitem) is unchanged. The search caps at 500 hits,
    so callers backfilling large ranges should window with datefrom/dateto."""
    out: list[dict[str, Any]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(ASYLNET_DB_URL, wait_until="networkidle", timeout=40000)
            try:
                btn = await page.query_selector(
                    "button#CybotCookiebotDialogBodyLevelButtonAccept, button:has-text('Akzeptieren')"
                )
                if btn:
                    await btn.click()
                    await page.wait_for_timeout(600)
            except Exception:
                pass

            if query:
                await page.fill("input[name='fulltext']", query)
            if datefrom:
                await page.fill("input[name='datefrom']", datefrom)
            if dateto:
                await page.fill("input[name='dateto']", dateto)
            try:
                await page.select_option("select[name='limit']", "100")  # 100 results/page
            except Exception:
                pass
            async with page.expect_navigation(wait_until="networkidle", timeout=40000):
                await page.click("button[name='newsearch']")
            await page.wait_for_timeout(1000)

            await _collect_page_items(page, out, limit)
            # Follow pagination (tx_ksrsdb_pi1[currentPage]=N) newest-first.
            current = 1
            while len(out) < limit and current < max_pages:
                current += 1
                next_link = await page.query_selector(
                    f"a[href*='currentPage%5D={current}'], a[href*='currentPage]={current}']"
                )
                if not next_link:
                    break
                href = await next_link.get_attribute("href")
                if not href:
                    break
                await page.goto(href if href.startswith("http") else f"{ASYLNET_BASE}{href}",
                                wait_until="networkidle", timeout=40000)
                await page.wait_for_timeout(800)
                before = len(out)
                await _collect_page_items(page, out, limit)
                if len(out) == before:
                    break

            # Resolve PDF + harvest curated metadata on each detail page.
            for rec in out:
                try:
                    dp = await browser.new_page()
                    await dp.goto(rec["url"], wait_until="domcontentloaded", timeout=30000)
                    pdf = await dp.query_selector("a[href$='.pdf'], a[href*='.pdf']")
                    if pdf:
                        ph = await pdf.get_attribute("href")
                        if ph:
                            rec["pdf_url"] = ph if ph.startswith("http") else f"{ASYLNET_BASE}{ph}"
                    rec.update(parse_detail_metadata(await dp.content()))
                    await dp.close()
                except Exception as exc:
                    print(f"  (detail lookup failed for {rec['url']}: {exc})")
        finally:
            await browser.close()
    return out

_EXTRACT_PROMPT = (
    "Analysiere dieses deutsche Gerichtsurteil oder diesen Beschluss (Rechtsprechung) "
    "und extrahiere die folgenden Informationen. Antworte ausschließlich im JSON-Format "
    "gemäß Schema.\n\n"
    "Felder:\n"
    "- country: Herkunftsland des Antragstellers (z. B. Iran, Afghanistan)\n"
    "- tags: Stichwörter wie frau/mann, politische Verfolgung, Religion/Konversion, "
    "Wehrdienst, Nachfluchtgründe, Hazara, LGBTQ, etc.\n"
    "- court / court_level (VG/OVG/BVerwG/BVerfG/EGMR/EuGH)\n"
    "- decision_date (YYYY-MM-DD), aktenzeichen\n"
    "  WICHTIG: court/decision_date/aktenzeichen NUR aus dem Rubrum bzw. Kopf "
    "DIESER Entscheidung übernehmen — NIEMALS aus 'Vgl.'-Blöcken oder sonst "
    "zitierter Fremdrechtsprechung (die gehört in citations). Fehlt das Rubrum "
    "(Teilauszug), diese Felder null lassen und eine warning setzen.\n"
    "- outcome: grant|partial|deny|remand (sonst unknown)\n"
    "- key_facts, key_holdings: je 3-7 knappe Stichpunkte\n"
    "- argument_patterns: Objekte mit use_when/rebuttal/notes\n"
    "- citations: zitierte Entscheidungen (court/date/az)\n"
    "- summary: 2-4 Sätze\n- confidence: 0-1\n- warnings: optional\n"
)


# Postgres (JSONB and text columns) rejects NUL/C0 control chars, which the
# PDF text layer of some scans carries and Gemini then echoes into its output
# (same failure mode the memory subsystem hit; killed the first --backfill-llm
# run after 56 entries). Strip at the chokepoints: model input AND output.
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _strip_ctrl_deep(value: Any) -> Any:
    if isinstance(value, str):
        return _CTRL_CHARS_RE.sub("", value)
    if isinstance(value, list):
        return [_strip_ctrl_deep(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_ctrl_deep(v) for k, v in value.items()}
    return value


# Default backend is the LOCAL Qwen on the desktop 3090 (Jay, 2026-07-06:
# no standing cloud spend for this). "gemini" stays available as explicit
# fallback via JURIS_EXTRACT_BACKEND for emergencies.
JURIS_EXTRACT_BACKEND = (os.getenv("JURIS_EXTRACT_BACKEND", "qwen") or "qwen").strip().lower()
JURIS_EXTRACT_MODEL = (
    os.getenv("JURIS_EXTRACT_MODEL", os.getenv("LLAMA_SERVER_MODEL", "qwen3.6-27b-udq5xl-vision")).strip()
    or "qwen3.6-27b-udq5xl-vision"
)

# Qwen gets the schema spelled out (no server-side response_schema like Gemini).
_QWEN_JSON_SPEC = """{
  "country": "Herkunftsland des Antragstellers oder null",
  "tags": ["stichwort", "..."],
  "court": "z.B. VG Berlin oder null",
  "court_level": "VG|OVG|VGH|BVerwG|BVerfG|LSG|EGMR|EuGH oder null",
  "decision_date": "YYYY-MM-DD oder null",
  "aktenzeichen": "z.B. 1 K 6/24 A oder null",
  "outcome": "grant|partial|deny|remand|unknown",
  "key_facts": ["3-7 knappe Stichpunkte"],
  "key_holdings": ["3-7 tragende Erwägungen"],
  "argument_patterns": [{"use_when": "...", "rebuttal": "...", "notes": "..."}],
  "citations": [{"court": "...", "date": "YYYY-MM-DD", "az": "..."}],
  "summary": "2-4 Sätze",
  "confidence": 0.8,
  "warnings": []
}"""


def _extraction_from_payload(data: Any) -> Optional[RechtsprechungExtraction]:
    """Leniently validate a model-emitted dict: malformed nested items are
    dropped, not fatal (local-model JSON is best-effort)."""
    if not isinstance(data, dict):
        return None
    data = _strip_ctrl_deep(data)

    def _s(v: Any) -> Optional[str]:
        v = str(v).strip() if v is not None else ""
        return v or None

    def _slist(v: Any) -> list:
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x or "").strip()]

    patterns = []
    for a in data.get("argument_patterns") or []:
        if isinstance(a, dict):
            patterns.append(ArgumentPattern(
                use_when=_s(a.get("use_when")), rebuttal=_s(a.get("rebuttal")), notes=_s(a.get("notes"))
            ))
    citations = []
    for c in data.get("citations") or []:
        if isinstance(c, dict):
            citations.append(CitationRef(
                court=_s(c.get("court")), date=_s(c.get("date")),
                az=_s(c.get("az") or c.get("aktenzeichen")),
            ))
    try:
        confidence = float(data.get("confidence")) if data.get("confidence") is not None else None
    except (TypeError, ValueError):
        confidence = None
    return RechtsprechungExtraction(
        country=_s(data.get("country")) or "Unbekannt",
        tags=_slist(data.get("tags")),
        court=_s(data.get("court")),
        court_level=_s(data.get("court_level")),
        decision_date=_s(data.get("decision_date")),
        aktenzeichen=_s(data.get("aktenzeichen")),
        outcome=_s(data.get("outcome")),
        key_facts=_slist(data.get("key_facts")),
        key_holdings=_slist(data.get("key_holdings")),
        argument_patterns=patterns,
        citations=citations,
        summary=_s(data.get("summary")),
        confidence=confidence,
        warnings=_slist(data.get("warnings")),
    )


async def _extract_tags_qwen(text: str) -> Optional[RechtsprechungExtraction]:
    from citation_qwen import call_qwen_json
    from shared import ensure_anonymization_service_ready

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        raise RuntimeError("ANONYMIZATION_SERVICE_URL not set (Qwen service manager)")
    await ensure_anonymization_service_ready()
    prompt = (
        f"{_EXTRACT_PROMPT}\nAntworte NUR mit diesem JSON-Objekt:\n{_QWEN_JSON_SPEC}"
        f"\n\nENTSCHEIDUNG:\n{_strip_ctrl_deep(text)[:30000]}"
    )
    parsed = await call_qwen_json(
        service_url, prompt, model=JURIS_EXTRACT_MODEL, num_predict=2500, temperature=0.0,
    )
    return _extraction_from_payload(parsed)


def _extract_tags_gemini(text: str) -> Optional[RechtsprechungExtraction]:
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[_EXTRACT_PROMPT, _strip_ctrl_deep(text)[:30000]],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RechtsprechungExtraction,
        ),
    )
    if response.parsed is None:
        return None
    return RechtsprechungExtraction(**_strip_ctrl_deep(response.parsed.model_dump()))


async def extract_tags(text: str) -> Optional[RechtsprechungExtraction]:
    if JURIS_EXTRACT_BACKEND == "gemini":
        return _extract_tags_gemini(text)
    return await _extract_tags_qwen(text)


def extract_model_label() -> str:
    return "gemini-3.5-flash" if JURIS_EXTRACT_BACKEND == "gemini" else JURIS_EXTRACT_MODEL


def download_pdf_text(pdf_url: str, timeout: float = 30.0) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Rechtmaschine/1.0)"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        resp = client.get(pdf_url)
        resp.raise_for_status()
        data = resp.content
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(data)
        tmp.flush()
        doc = fitz.open(tmp.name)
        try:
            return "\n\n".join((page.get_text() or "").strip() for page in doc)
        finally:
            doc.close()


def chunk_text(text: str, target: int = 1800, hard: int = 2400) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paras:
        while len(para) > hard:
            cut = para.rfind(" ", 0, hard)
            cut = cut if cut > hard // 2 else hard
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        candidate = f"{cur}\n\n{para}" if cur else para
        if len(candidate) > target and cur:
            chunks.append(cur)
            cur = para
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    if len(chunks) >= 2 and len(chunks[-1]) < 200:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()
    return chunks


def upsert(chunks: list[dict[str, Any]], collection: str) -> int:
    base = os.getenv("RAG_SERVICE_URL", "").strip().rstrip("/")
    key = os.getenv("RAG_API_KEY") or os.getenv("RAG_SERVICE_API_KEY")
    headers = {"X-API-Key": key} if key else {}
    total = 0
    with httpx.Client(timeout=300.0) as client:
        for start in range(0, len(chunks), 16):
            resp = client.post(
                f"{base}/v1/rag/chunks/upsert",
                json={"collection": collection, "chunks": chunks[start : start + 16]},
                headers=headers,
            )
            resp.raise_for_status()
            total += int(resp.json().get("upserted", 0))
    return total


async def main_async(args) -> int:
    db = SessionLocal()
    try:
        fetched = await fetch_asylnet(args.query, args.limit * 3, args.datefrom, args.dateto)
        # Pre-download dedup by asyl.net M-number: skip decisions already stored
        # before spending a PDF download + Gemini call (metadata-first).
        candidates = []
        dup_ref = 0
        for r in fetched:
            m = r.get("m_number")
            if m and db.query(RechtsprechungEntry.id).filter(
                RechtsprechungEntry.source_ref == m
            ).first():
                dup_ref += 1
                continue
            candidates.append(r)
        results = [r for r in candidates if r.get("pdf_url")][: args.limit]
        print(f"asyl.net: {len(fetched)} results ({dup_ref} already stored), "
              f"{len(results)} new with PDFs for '{args.query}' (since {args.datefrom or 'any'})\n")

        ingested = dup_content = short = failed = chunk_total = 0
        for r in results:
            url = r["url"]
            try:
                text = download_pdf_text(r["pdf_url"])
                if len(text) < 400:
                    print(f"  SHORT {url} — only {len(text)} chars")
                    short += 1
                    continue
                full_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                sha16 = full_sha[:16]
                # Post-download dedup by content (catches the same decision under
                # a different M-number / cross-source republication).
                if db.query(RechtsprechungEntry.id).filter(
                    RechtsprechungEntry.content_sha256 == full_sha
                ).first():
                    print(f"  DUP   {url} (content)")
                    dup_content += 1
                    continue

                _vocab = load_vocabulary()
                tags = extraction_from_asylnet(r, _vocab)
                # Full-text LLM pass: semantic fields + cross-check of the
                # footer citation. Advisory — on failure the entry still lands
                # with curated metadata only (as before).
                model_label = "asylnet-metadata"
                if not args.no_llm:
                    try:
                        llm_tags = await extract_tags(text)
                        tags = merge_footer_and_llm(tags, llm_tags)
                        if llm_tags is not None:
                            model_label = f"asylnet+{extract_model_label()}"
                    except Exception as exc:
                        print(f"  (LLM extraction failed for {url}: {exc})")
                _themen = normalize_themen(_vocab, (r.get("schlagworte") or []) + (tags.tags or []))
                _country = normalize_country(_vocab, tags.country)
                _normen = normalize_normen(_vocab, r.get("normen") or [])
                header_bits = ["Rechtsprechung", tags.court or "", tags.court_level or "",
                               tags.decision_date or "", tags.country or "",
                               tag_line(_themen, _country, _normen)]
                context_header = " | ".join(b for b in header_bits if b)

                if args.dry_run:
                    ingested += 1
                    print(f"  OK*   {tags.court} {tags.aktenzeichen} ({tags.country}, "
                          f"{tags.decision_date}) — {len(text)}c, {len(chunk_text(text))} chunks [dry-run]")
                    continue

                entry = persist_entry(
                    db, tags, source_type="asylnet", source_url=url,
                    source_ref=r.get("m_number"), content_sha256=full_sha,
                    schlagworte=r.get("schlagworte"), normen=r.get("normen"),
                    leitsatz=r.get("leitsatz"), model_label=model_label,
                )
                metadata = {
                    "source_system": "asylnet",
                    "rechtsprechung_entry_id": str(entry.id),
                    "country": tags.country,
                    "court": tags.court,
                    "court_level": tags.court_level,
                    "outcome": tags.outcome,
                    "decision_date": tags.decision_date,
                    "aktenzeichen": tags.aktenzeichen,
                    "issue_tags": tags.tags or [],
                    **facet_metadata(_themen, _country, _normen),
                    "instance_weight": entry.instance_weight,
                    "language": "de",
                }
                provenance = [f"asylnet:{url}", f"entry:{entry.id}", f"sha256:{sha16}"]
                payload = [
                    {
                        "chunk_id": f"juris-{sha16}-{idx:03d}",
                        "text": chunk,
                        "context_header": context_header,
                        "metadata": {**metadata, "chunk_index": idx},
                        "provenance": provenance,
                    }
                    for idx, chunk in enumerate(chunk_text(text))
                ]
                upserted = upsert(payload, args.collection)
                chunk_total += upserted
                ingested += 1
                print(f"  OK    {tags.court} {tags.aktenzeichen} ({tags.country}, "
                      f"{tags.decision_date}, {tags.outcome}, w{entry.instance_weight}) "
                      f"— {len(text)}c -> {upserted} chunks")
                for w in tags.warnings or []:
                    print(f"  WARN  {url}: {w}")
            except Exception as exc:
                failed += 1
                print(f"  FAIL  {url} — {exc}")

        verb = "would ingest" if args.dry_run else "ingested"
        print(f"\n{verb} {ingested}, dup-ref {dup_ref}, dup-content {dup_content}, "
              f"short {short}, failed {failed}"
              + ("" if args.dry_run else f"; {chunk_total} chunks into '{args.collection}'."))
        return 0 if failed == 0 else 1
    finally:
        db.close()


def backfill_metadata(limit: Optional[int] = None) -> int:
    """Harvest asyl.net Schlagwörter/Normen/Leitsatz onto existing entries
    (detail-page scrape only — no PDF, no LLM, no re-embed)."""
    db = SessionLocal()
    try:
        q = db.query(RechtsprechungEntry).filter(
            RechtsprechungEntry.source_type == "asylnet",
            RechtsprechungEntry.source_url.isnot(None),
        )
        rows = q.limit(limit).all() if limit else q.all()
        print(f"Backfilling curated metadata for {len(rows)} entries…")
        updated = empty = failed = 0
        with httpx.Client(timeout=30.0, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (Rechtmaschine/1.0)"}) as client:
            for i, e in enumerate(rows, 1):
                try:
                    meta = parse_detail_metadata(client.get(e.source_url).text)
                except Exception as exc:
                    failed += 1
                    print(f"  ERR {e.source_ref} {exc}")
                    continue
                if not meta["schlagworte"] and not meta["normen"] and not meta["leitsatz"]:
                    empty += 1
                    continue
                e.schlagworte = meta["schlagworte"]
                e.normen = meta["normen"]
                e.leitsatz = meta["leitsatz"]
                tags = list(e.tags or [])
                for sw in meta["schlagworte"]:
                    if sw.lower() not in tags:
                        tags.append(sw.lower())
                e.tags = tags
                e.updated_at = datetime.utcnow()
                updated += 1
                if i % 50 == 0:
                    db.commit()
                    print(f"  …{i}/{len(rows)} (updated {updated})")
        db.commit()
        print(f"\nBackfill done: updated {updated}, no-metadata {empty}, failed {failed}")
        return 0
    finally:
        db.close()


def _rag_chunk_texts(collection: str = "jurisprudence") -> dict[str, str]:
    """Scroll the RAG store once and reassemble full text per entry id."""
    base = os.getenv("RAG_SERVICE_URL", "").strip().rstrip("/")
    key = os.getenv("RAG_API_KEY") or os.getenv("RAG_SERVICE_API_KEY")
    headers = {"X-API-Key": key} if key else {}
    parts: dict[str, list] = {}
    cursor = None
    with httpx.Client(timeout=120.0) as client:
        while True:
            resp = client.post(
                f"{base}/v1/rag/chunks/scroll",
                json={"collection": collection, "cursor": cursor, "limit": 256},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            for ch in data["chunks"]:
                meta = ch.get("metadata") or {}
                eid = meta.get("rechtsprechung_entry_id")
                if eid:
                    parts.setdefault(str(eid), []).append(
                        (meta.get("chunk_index", 0), ch.get("text") or "")
                    )
            cursor = data.get("next_cursor")
            if cursor is None:
                break
    return {
        eid: "\n\n".join(t for _, t in sorted(chunks))
        for eid, chunks in parts.items()
    }


def backfill_llm(limit: Optional[int] = None) -> int:
    """One-off: run the Gemini full-text extraction over entries ingested
    metadata-only (model=asylnet-metadata — 637 entries as of 2026-07-05:
    outcome unknown, no key_holdings/summary, so they could never rank as
    STÜTZEND/GEGEN UNS and gave the reliance judge only the Leitsatz).

    Text comes from the already-chunked RAG store; the stored footer fields
    stay authoritative via merge_footer_and_llm (disagreement -> warning).
    enriched_at is reset so the nightly Qwen pass re-judges reliance with
    real tragende Erwägungen."""
    texts = _rag_chunk_texts()
    print(f"RAG store: text for {len(texts)} entries")
    db = SessionLocal()
    try:
        q = (db.query(RechtsprechungEntry)
             .filter(RechtsprechungEntry.model == "asylnet-metadata",
                     RechtsprechungEntry.is_active == True)  # noqa: E712
             .order_by(RechtsprechungEntry.decision_date.desc().nullslast()))
        rows = q.limit(limit).all() if limit else q.all()
        print(f"LLM backfill for {len(rows)} metadata-only entries…")
        done = no_text = failed = mismatches = 0
        for i, e in enumerate(rows, 1):
            text = texts.get(str(e.id))
            if not text or len(text) < 400:
                no_text += 1
                continue
            try:
                llm = asyncio.run(extract_tags(text))
            except Exception as exc:
                failed += 1
                print(f"  ERR {e.source_ref} {e.aktenzeichen}: {exc}")
                continue
            footer = RechtsprechungExtraction(
                country=e.country or "Unbekannt",
                tags=list(e.tags or []),
                court=e.court,
                court_level=e.court_level,
                decision_date=e.decision_date.isoformat() if e.decision_date else None,
                aktenzeichen=e.aktenzeichen,
            )
            merged = merge_footer_and_llm(footer, llm)
            e.outcome = merged.outcome or "unknown"
            e.key_facts = merged.key_facts or []
            e.key_holdings = merged.key_holdings or []
            e.argument_patterns = [a.model_dump() for a in (merged.argument_patterns or [])]
            e.citations = [c.model_dump() for c in (merged.citations or [])]
            e.summary = merged.summary
            e.confidence = merged.confidence
            e.court = merged.court or e.court
            e.aktenzeichen = merged.aktenzeichen or e.aktenzeichen
            e.decision_date = e.decision_date or _parse_date(merged.decision_date)
            e.tags = merged.tags or e.tags
            if merged.warnings:
                e.warnings = list(e.warnings or []) + merged.warnings
                mismatches += 1
                for w in merged.warnings:
                    print(f"  WARN {e.source_ref} {e.aktenzeichen}: {w}")
            e.model = f"asylnet+{extract_model_label()}"
            e.enriched_at = None
            e.enrichment_model = None
            e.updated_at = datetime.utcnow()
            db.add(e)
            db.commit()
            done += 1
            if i % 25 == 0:
                print(f"  …{i}/{len(rows)} (done {done}, warn {mismatches})")
        print(f"\nLLM backfill done: {done} updated, {no_text} without RAG text, "
              f"{failed} failed, {mismatches} with warnings")
        return 0
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Jurisprudence ingest (asyl.net)")
    parser.add_argument("--query", default="", help="Fulltext seed for asyl.net (empty = all)")
    parser.add_argument("--datefrom", default=None, help="Only decisions from this date (DD.MM.YYYY)")
    parser.add_argument("--dateto", default=None, help="Only decisions up to this date (DD.MM.YYYY)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--collection", default="jurisprudence")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the Gemini full-text extraction (curated metadata only)")
    parser.add_argument("--backfill-metadata", action="store_true",
                        help="Harvest Schlagwörter/Normen/Leitsatz onto existing entries and exit.")
    parser.add_argument("--backfill-llm", action="store_true",
                        help="Run the Gemini full-text extraction over metadata-only entries and exit.")
    args = parser.parse_args()
    if args.backfill_metadata:
        return backfill_metadata(args.limit if args.limit and args.limit > 5 else None)
    if args.backfill_llm:
        return backfill_llm(args.limit if args.limit and args.limit > 5 else None)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

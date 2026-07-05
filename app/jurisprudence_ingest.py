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
                  leitsatz: Optional[str] = None) -> RechtsprechungEntry:
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
    entry.model = "asylnet-metadata"
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
    "- outcome: grant|partial|deny|remand (sonst unknown)\n"
    "- key_facts, key_holdings: je 3-7 knappe Stichpunkte\n"
    "- argument_patterns: Objekte mit use_when/rebuttal/notes\n"
    "- citations: zitierte Entscheidungen (court/date/az)\n"
    "- summary: 2-4 Sätze\n- confidence: 0-1\n- warnings: optional\n"
)


def extract_tags(text: str) -> RechtsprechungExtraction:
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[_EXTRACT_PROMPT, text[:30000]],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RechtsprechungExtraction,
        ),
    )
    return response.parsed


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
                    leitsatz=r.get("leitsatz"),
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Jurisprudence ingest (asyl.net)")
    parser.add_argument("--query", default="", help="Fulltext seed for asyl.net (empty = all)")
    parser.add_argument("--datefrom", default=None, help="Only decisions from this date (DD.MM.YYYY)")
    parser.add_argument("--dateto", default=None, help="Only decisions up to this date (DD.MM.YYYY)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--collection", default="jurisprudence")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backfill-metadata", action="store_true",
                        help="Harvest Schlagwörter/Normen/Leitsatz onto existing entries and exit.")
    args = parser.parse_args()
    if args.backfill_metadata:
        return backfill_metadata(args.limit if args.limit and args.limit > 5 else None)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

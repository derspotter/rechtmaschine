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
from typing import Any, Optional

import httpx

import fitz  # PyMuPDF
from google.genai import types
from playwright.async_api import async_playwright

from shared import get_gemini_client
from endpoints.rechtsprechung_playbook import RechtsprechungExtraction

ASYLNET_DB_URL = "https://www.asyl.net/recht/entscheidungsdatenbank"
ASYLNET_BASE = "https://www.asyl.net"
_FOOTER_DATE = re.compile(r"(\d{2}\.\d{2}\.\d{4})")
_FOOTER_M = re.compile(r"asyl\.net:\s*(M\d+)", re.IGNORECASE)


async def fetch_asylnet(query: str, limit: int, datefrom: Optional[str]) -> list[dict[str, Any]]:
    """Search the asyl.net Entscheidungsdatenbank via its POST form and return
    decisions with their detail URL, PDF URL, and footer metadata.

    The DB moved from a GET query API to a stateful POST form (the old GET URL
    now 404s); the result-item markup (div.rsdb_listitem) is unchanged."""
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
            async with page.expect_navigation(wait_until="networkidle", timeout=40000):
                await page.click("button[name='newsearch']")
            await page.wait_for_timeout(1000)

            items = await page.query_selector_all("div.rsdb_listitem")
            for item in items:
                link = await item.query_selector("div.rsdb_listitem_court a")
                if not link:
                    continue
                href = await link.get_attribute("href")
                if not href:
                    continue
                detail_url = href if href.startswith("http") else f"{ASYLNET_BASE}{href}"
                footer_el = await item.query_selector(".rsdb_listitem_footer")
                footer = (await footer_el.text_content() if footer_el else "") or ""
                footer = re.sub(r"\s+", " ", footer).strip()
                date_m = _FOOTER_DATE.search(footer)
                m_num = _FOOTER_M.search(footer)
                out.append({
                    "url": detail_url,
                    "footer": footer,
                    "footer_date": date_m.group(1) if date_m else None,
                    "m_number": m_num.group(1) if m_num else None,
                })
                if len(out) >= limit:
                    break

            # Resolve the PDF on each detail page.
            for rec in out:
                try:
                    dp = await browser.new_page()
                    await dp.goto(rec["url"], wait_until="domcontentloaded", timeout=30000)
                    pdf = await dp.query_selector("a[href$='.pdf'], a[href*='.pdf']")
                    if pdf:
                        ph = await pdf.get_attribute("href")
                        if ph:
                            rec["pdf_url"] = ph if ph.startswith("http") else f"{ASYLNET_BASE}{ph}"
                    await dp.close()
                except Exception as exc:
                    print(f"  (detail PDF lookup failed for {rec['url']}: {exc})")
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
    fetched = await fetch_asylnet(args.query, args.limit * 2, args.datefrom)
    results = [r for r in fetched if r.get("pdf_url")][: args.limit]
    print(f"asyl.net: {len(fetched)} results, {len(results)} with PDFs for "
          f"'{args.query}' (since {args.datefrom or 'any'})\n")

    ingested = failed = chunk_total = 0
    for r in results:
        url = r["url"]
        try:
            text = download_pdf_text(r["pdf_url"])
            if len(text) < 400:
                print(f"  SKIP  {url} — only {len(text)} chars")
                continue
            tags = extract_tags(text)
            sha16 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            statutes = [c.az for c in (tags.citations or []) if c and c.az]
            header_bits = ["Rechtsprechung", tags.court or "", tags.court_level or "",
                           tags.decision_date or "", tags.country or ""]
            context_header = " | ".join(b for b in header_bits if b)
            metadata = {
                "source_system": "asylnet",
                "country": tags.country,
                "court": tags.court,
                "court_level": tags.court_level,
                "outcome": tags.outcome,
                "decision_date": tags.decision_date,
                "aktenzeichen": tags.aktenzeichen,
                "issue_tags": tags.tags or [],
                "language": "de",
            }
            provenance = [f"asylnet:{url}", f"sha256:{sha16}"]
            if r.get("pdf_url"):
                provenance.append(f"pdf:{r['pdf_url']}")

            if args.dry_run:
                ingested += 1
                print(f"  OK*   {tags.court} {tags.aktenzeichen} ({tags.country}, "
                      f"{tags.decision_date}) — {len(text)}c, {len(chunk_text(text))} chunks [dry-run]")
                continue

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
                  f"{tags.decision_date}, {tags.outcome}) — {len(text)}c -> {upserted} chunks")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {url} — {exc}")

    print(f"\nIngested {ingested}, failed {failed}; {chunk_total} chunks into '{args.collection}'.")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Jurisprudence ingest (asyl.net slice)")
    parser.add_argument("--query", default="", help="Fulltext seed for asyl.net (empty = all)")
    parser.add_argument("--datefrom", default=None, help="Only decisions from this date (DD.MM.YYYY)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--collection", default="jurisprudence")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

"""Sync wiki.aufentha.lt (the firm's public DokuWiki) into the Doktrin layer.

Crawls the wiki index, fetches raw page markup, cleans it via
dokuwiki_markup, chunks per section, upserts into the RAG collection
"doktrin" (pgvector on debian, server-side embedding) and keeps one
DoktrinPage bookkeeping row per page (sha dedup, chunk ids, derived facets).

Runs inside the app container:
    docker exec rechtmaschine-app python /app/doktrin_sync.py [--dry-run ...]

Nightly via systemd user timer doktrin-sync.timer (scripts/doktrin_sync.sh).
Idempotent: unchanged pages (same content sha) are skipped, changed pages
delete their old chunk ids before upserting the new ones.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import httpx

from database import SessionLocal
from dokuwiki_markup import (
    chunk_section,
    clean_markup,
    context_header,
    extract_normen,
    split_sections,
)
from models import DoktrinPage
from rag_vocabulary import facet_metadata, load_vocabulary, normalize_country

DEFAULT_BASE_URL = "https://wiki.aufentha.lt"
DEFAULT_INDEX_ID = "start"
DOKTRIN_COLLECTION = os.getenv("DOKTRIN_COLLECTION", "doktrin")
DOKTRIN_MIN_PAGE_CHARS = int((os.getenv("DOKTRIN_MIN_PAGE_CHARS", "200") or "200").strip())
SKIP_NAMESPACES = {
    ns.strip()
    for ns in (os.getenv("DOKTRIN_SKIP_NAMESPACES", "wiki,playground") or "").split(",")
    if ns.strip()
}
USER_AGENT = "rechtmaschine-doktrin-sync/0.1"


# --- fetch helpers, copied from rag/export_dokuwiki.py (app container only
# --- mounts ./app, so the rag/ module is not importable here). fetch_page is
# --- extended to capture the Last-Modified header for staleness tracking.


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.hrefs.append(value)


@dataclass(frozen=True)
class WikiPage:
    page_id: str
    title: str
    url: str
    export_url: str
    text: str
    last_modified: Optional[datetime] = None


def _fetch_text(url: str, timeout: float) -> tuple[str, Optional[datetime]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/plain,text/html;q=0.9,*/*;q=0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        last_modified: Optional[datetime] = None
        header = response.headers.get("Last-Modified")
        if header:
            try:
                last_modified = parsedate_to_datetime(header).replace(tzinfo=None)
            except (TypeError, ValueError):
                last_modified = None
        return response.read().decode(charset, errors="replace"), last_modified


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _page_url(base_url: str, page_id: str) -> str:
    return f"{_normalize_base_url(base_url)}/{urllib.parse.quote(page_id, safe=':/._-')}"


def _raw_url(base_url: str, page_id: str) -> str:
    return f"{_normalize_base_url(base_url)}/_export/raw/{urllib.parse.quote(page_id, safe=':/._-')}"


def _safe_rel_id(raw: str) -> str:
    return raw.strip().strip("/")


def _is_page_id(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("_", "lib/", "feed.php", "doku.php", "validator.")):
        return False
    if value.startswith(("http://", "https://", "mailto:", "#")):
        return False
    if value.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico")):
        return False
    if ".." in value:
        return False
    return True


def _href_to_page_id(base_url: str, href: str) -> Optional[str]:
    parsed = urllib.parse.urlparse(urllib.parse.urljoin(base_url + "/", href))
    base = urllib.parse.urlparse(base_url)
    if parsed.netloc and parsed.netloc != base.netloc:
        return None
    path = urllib.parse.unquote(parsed.path or "").lstrip("/")
    if path.startswith("_export/raw/"):
        path = path.removeprefix("_export/raw/")
    elif path.startswith("_export/xhtml/"):
        path = path.removeprefix("_export/xhtml/")
    page_id = _safe_rel_id(path)
    if not page_id:
        query = urllib.parse.parse_qs(parsed.query)
        page_id = _safe_rel_id((query.get("id") or [""])[0])
    if not _is_page_id(page_id):
        return None
    if urllib.parse.parse_qs(parsed.query).get("do") not in (None, [], ["index"]):
        return None
    return page_id


def discover_page_ids(base_url: str, index_id: str, timeout: float) -> list[str]:
    index_url = f"{_page_url(base_url, index_id)}?do=index"
    html_text, _ = _fetch_text(index_url, timeout=timeout)
    parser = LinkParser()
    parser.feed(html_text)
    page_ids = {
        page_id
        for href in parser.hrefs
        for page_id in [_href_to_page_id(base_url, href)]
        if page_id
    }
    page_ids.add(index_id)
    return sorted(page_ids)


def _extract_title(page_id: str, text: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^\s*=+\s*(.*?)\s*=+\s*$", line)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return page_id


def fetch_page(base_url: str, page_id: str, timeout: float) -> WikiPage:
    export_url = _raw_url(base_url, page_id)
    text, last_modified = _fetch_text(export_url, timeout=timeout)
    text = text.strip()
    return WikiPage(
        page_id=page_id,
        title=_extract_title(page_id, text),
        url=_page_url(base_url, page_id),
        export_url=export_url,
        text=text,
        last_modified=last_modified,
    )


# --- RAG store I/O (upsert copied from jurisprudence_ingest.upsert; that
# --- module is not imported because it pulls fitz/genai at import time).


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


def delete_chunks(chunk_ids: list[str], collection: str) -> int:
    if not chunk_ids:
        return 0
    base = os.getenv("RAG_SERVICE_URL", "").strip().rstrip("/")
    key = os.getenv("RAG_API_KEY") or os.getenv("RAG_SERVICE_API_KEY")
    headers = {"X-API-Key": key} if key else {}
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{base}/v1/rag/chunks/delete",
            json={"collection": collection, "chunk_ids": chunk_ids},
            headers=headers,
        )
        resp.raise_for_status()
        return int(resp.json().get("deleted", 0))


# --- page processing


def _namespace(page_id: str) -> str:
    return page_id.split(":")[0] if ":" in page_id else ""


def _page_sha(page_id: str, raw_text: str) -> str:
    # page_id is part of the hash so identical texts on different pages never
    # collide into the same chunk-id family.
    return hashlib.sha256(f"{page_id}\n{raw_text}".encode("utf-8", errors="ignore")).hexdigest()


def build_chunk_payloads(page: WikiPage, clean_text: str, sha16: str, vocab) -> list[dict[str, Any]]:
    country = normalize_country(vocab, page.title) or normalize_country(
        vocab, page.page_id.split(":")[-1].replace("_", " ")
    )
    normen = extract_normen(f"{page.title}\n{clean_text[:4000]}", vocab=vocab)
    payloads: list[dict[str, Any]] = []
    index = 0
    for section in split_sections(clean_text, page.title):
        header = context_header(page.title, section.heading_path, page.url)
        for chunk in chunk_section(section.text):
            metadata: dict[str, Any] = {
                "source_system": "dokuwiki",
                "source_kind": "public_legal_wiki",
                "source_public": True,
                "anonymized": True,
                "page_id": page.page_id,
                "page_title": page.title,
                "namespace": _namespace(page.page_id),
                "url": page.url,
                "heading_path": section.heading_path,
                "chunk_index": index,
                "language": "de",
            }
            metadata.update(facet_metadata([], country, normen))
            payloads.append(
                {
                    "chunk_id": f"doku-{sha16}-{index:03d}",
                    "text": chunk,
                    "context_header": header,
                    "metadata": metadata,
                    "provenance": [f"dokuwiki:{page.url}", f"sha256:{sha16}"],
                }
            )
            index += 1
    return payloads


def _touch(row: DoktrinPage, now: datetime) -> None:
    row.last_synced_at = now
    row.updated_at = now


def sync(args: argparse.Namespace) -> int:
    base_url = _normalize_base_url(args.base_url)
    vocab = load_vocabulary()
    now = datetime.utcnow()

    explicit_pages = [p.strip() for p in (args.pages or "").split(",") if p.strip()]
    full_discovery = False
    if explicit_pages:
        page_ids = explicit_pages
    else:
        page_ids = [
            pid
            for pid in discover_page_ids(base_url, args.index_id, timeout=args.timeout)
            if _namespace(pid) not in SKIP_NAMESPACES
        ]
        full_discovery = not args.limit
        if args.limit:
            page_ids = page_ids[: args.limit]

    db = SessionLocal()
    stats = {
        "discovered": len(page_ids),
        "unchanged": 0,
        "new": 0,
        "updated": 0,
        "thin": 0,
        "gone": 0,
        "failed": 0,
        "chunks_upserted": 0,
    }
    ratio_flags: list[str] = []

    try:
        for idx, page_id in enumerate(page_ids, 1):
            try:
                page = fetch_page(base_url, page_id, timeout=args.timeout)
                if page.text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
                    # Nonexistent pages come back as the HTML shell with status
                    # 200 — never ingest that as wiki markup.
                    stats["failed"] += 1
                    print(f"{idx}/{len(page_ids)} fail {page_id}: HTML instead of raw markup", file=sys.stderr)
                    continue
                sha = _page_sha(page_id, page.text)
                row = db.query(DoktrinPage).filter(DoktrinPage.page_id == page_id).first()

                if row is not None and row.content_sha256 == sha and row.status != "gone":
                    stats["unchanged"] += 1
                    if not args.dry_run:
                        _touch(row, now)
                        db.commit()
                    continue

                clean = clean_markup(page.text)
                is_new = row is None
                if row is None:
                    row = DoktrinPage(page_id=page_id)

                if len(clean) < DOKTRIN_MIN_PAGE_CHARS:
                    stats["thin"] += 1
                    if not args.dry_run:
                        if row.chunk_ids:
                            delete_chunks(list(row.chunk_ids), DOKTRIN_COLLECTION)
                        _update_row(row, page, sha, clean, [], status="thin", now=now)
                        db.add(row)
                        db.commit()
                    continue

                payloads = build_chunk_payloads(page, clean, sha[:16], vocab)
                if page.text and len(clean) < 0.4 * len(page.text):
                    ratio_flags.append(f"{page_id} ({len(page.text)}->{len(clean)} chars)")

                stats["new" if is_new else "updated"] += 1
                if args.dry_run:
                    print(
                        f"{idx}/{len(page_ids)} DRY {'new' if is_new else 'upd'} {page_id} "
                        f"chunks={len(payloads)} chars={len(page.text)}->{len(clean)}"
                    )
                    continue

                if row.chunk_ids:
                    delete_chunks(list(row.chunk_ids), DOKTRIN_COLLECTION)
                upserted = upsert(payloads, DOKTRIN_COLLECTION)
                stats["chunks_upserted"] += upserted
                _update_row(
                    row, page, sha, clean, [p["chunk_id"] for p in payloads],
                    status="active", now=now,
                )
                row.country = (payloads[0]["metadata"].get("applicant_origin") if payloads else None)
                row.normen = list(payloads[0]["metadata"].get("citations") or []) if payloads else []
                db.add(row)
                db.commit()
                print(f"{idx}/{len(page_ids)} ok {page_id} chunks={upserted}")
            except (urllib.error.URLError, httpx.HTTPError, TimeoutError, OSError, UnicodeError) as exc:
                stats["failed"] += 1
                db.rollback()
                print(f"{idx}/{len(page_ids)} fail {page_id}: {exc}", file=sys.stderr)
            if args.delay:
                time.sleep(args.delay)

        if full_discovery and not args.dry_run:
            known = {pid for pid in page_ids}
            gone_rows = (
                db.query(DoktrinPage)
                .filter(DoktrinPage.status != "gone", ~DoktrinPage.page_id.in_(known))
                .all()
            )
            for row in gone_rows:
                if row.chunk_ids:
                    try:
                        delete_chunks(list(row.chunk_ids), DOKTRIN_COLLECTION)
                    except httpx.HTTPError as exc:
                        stats["failed"] += 1
                        print(f"fail delete gone {row.page_id}: {exc}", file=sys.stderr)
                        continue
                row.status = "gone"
                row.chunk_ids = []
                row.chunk_count = 0
                _touch(row, now)
                db.add(row)
                stats["gone"] += 1
            db.commit()
    finally:
        db.close()

    print()
    print(" ".join(f"{k}={v}" for k, v in stats.items()))
    if ratio_flags:
        print(f"RATIO-CHECK (clean <40% of raw, eyeball these): {', '.join(ratio_flags[:20])}")
    return 1 if stats["failed"] else 0


def _update_row(
    row: DoktrinPage,
    page: WikiPage,
    sha: str,
    clean: str,
    chunk_ids: list[str],
    *,
    status: str,
    now: datetime,
) -> None:
    row.namespace = _namespace(page.page_id)
    row.title = page.title
    row.url = page.url
    row.content_sha256 = sha
    row.clean_text = clean
    row.raw_chars = len(page.text)
    row.clean_chars = len(clean)
    row.chunk_ids = chunk_ids
    row.chunk_count = len(chunk_ids)
    row.status = status
    row.wiki_last_modified = page.last_modified
    row.last_changed_at = now
    _touch(row, now)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync wiki.aufentha.lt into the Doktrin layer.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--index-id", default=DEFAULT_INDEX_ID)
    parser.add_argument("--limit", type=int, default=0, help="0 = all pages")
    parser.add_argument("--pages", default="", help="comma-separated page ids instead of discovery")
    parser.add_argument("--full", action="store_true", help="re-clean and re-upsert unchanged pages too")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.full:
        # --full forces re-processing by ignoring the sha shortcut: clear the
        # stored hashes up front so every page takes the changed-page path.
        db = SessionLocal()
        try:
            db.query(DoktrinPage).update({DoktrinPage.content_sha256: None})
            db.commit()
        finally:
            db.close()
    return sync(args)


if __name__ == "__main__":
    raise SystemExit(main())

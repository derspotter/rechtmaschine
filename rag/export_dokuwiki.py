#!/usr/bin/env python
"""
Export a public DokuWiki into Rechtmaschine RAG's ingested JSONL shape.

The wiki is treated as public legal knowledge, not case memory. This script
fetches raw wiki markup, keeps page provenance, and can optionally mirror the
raw page text for audit/debugging.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASE_URL = "https://wiki.aufentha.lt"
DEFAULT_EXPORT_ROOT = Path(
    os.getenv("RECHTMASCHINE_RAG_EXPORT_ROOT", "~/rechtmaschine-rag-export")
).expanduser()
DEFAULT_DOKUWIKI_ROOT = DEFAULT_EXPORT_ROOT / "dokuwiki"


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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


def _fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "rechtmaschine-dokuwiki-export/0.1",
            "Accept": "text/plain,text/html;q=0.9,*/*;q=0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


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


def _href_to_page_id(base_url: str, href: str) -> str | None:
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
    html = _fetch_text(index_url, timeout=timeout)
    parser = LinkParser()
    parser.feed(html)
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
    text = _fetch_text(export_url, timeout=timeout).strip()
    title = _extract_title(page_id, text)
    return WikiPage(
        page_id=page_id,
        title=title,
        url=_page_url(base_url, page_id),
        export_url=export_url,
        text=text,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _source_rel_path(page_id: str) -> str:
    safe = page_id.replace(":", "/").strip("/")
    return f"dokuwiki/{safe}.txt"


def _record(page: WikiPage, run_id: str, base_url: str) -> dict[str, Any]:
    metadata = {
        "source_system": "dokuwiki",
        "source_kind": "public_legal_wiki",
        "source_public": True,
        "anonymized": True,
        "wiki_base_url": base_url,
        "page_id": page.page_id,
        "page_title": page.title,
        "url": page.url,
        "export_url": page.export_url,
        "ingestion_run_id": run_id,
    }
    return {
        "source_rel_path": _source_rel_path(page.page_id),
        "source_abs_path": page.export_url,
        "ok": True,
        "used_ocr": False,
        "extracted_chars": len(page.text),
        "sha256": _sha256_text(page.text),
        "error": None,
        "metadata": metadata,
        "text": page.text,
    }


def _write_raw(page: WikiPage, raw_dir: Path) -> None:
    out_path = raw_dir / _source_rel_path(page.page_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page.text + "\n", encoding="utf-8")


def _write_manifest(
    *,
    manifest_dir: Path,
    run_id: str,
    base_url: str,
    out_path: Path,
    raw_dir: Path,
    checksums_path: Path,
    discovered_pages: int,
    ok: int,
    failed: int,
    skipped_empty: int,
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"dokuwiki_{run_id}.json"
    payload = {
        "run_id": run_id,
        "source_system": "dokuwiki",
        "source_kind": "public_legal_wiki",
        "base_url": base_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records_jsonl": str(out_path),
        "raw_dir": str(raw_dir),
        "checksums": str(checksums_path),
        "discovered_pages": discovered_pages,
        "ok": ok,
        "failed": failed,
        "skipped_empty": skipped_empty,
        "production_boundary": "source export only; Debian performs production chunking, embedding, storage, reranking, and retrieval",
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export public DokuWiki pages into RAG ingested JSONL.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--index-id", default="start")
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_DOKUWIKI_ROOT,
        help="export bundle root; defaults outside the repo",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--checksums-dir", type=Path, default=None)
    parser.add_argument("--write-raw", action=argparse.BooleanOptionalAction, default=True, help="mirror raw wiki pages to disk")
    parser.add_argument("--limit", type=int, default=0, help="0 = all pages")
    parser.add_argument("--delay", type=float, default=0.05, help="delay between page fetches")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--include-empty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_url = _normalize_base_url(args.base_url)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_root: Path = args.export_root.expanduser().resolve()
    out_dir: Path = args.out_dir.expanduser().resolve() if args.out_dir else export_root / "ingested"
    raw_dir: Path = args.raw_dir.expanduser().resolve() if args.raw_dir else export_root / "raw"
    manifest_dir: Path = (
        args.manifest_dir.expanduser().resolve() if args.manifest_dir else export_root / "manifests"
    )
    checksums_dir: Path = (
        args.checksums_dir.expanduser().resolve() if args.checksums_dir else export_root / "checksums"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dokuwiki_{run_id}.jsonl"
    checksums_path = checksums_dir / f"dokuwiki_{run_id}.sha256"

    page_ids = discover_page_ids(base_url, args.index_id, timeout=args.timeout)
    if args.limit:
        page_ids = page_ids[: args.limit]

    print("=" * 80)
    print("DOKUWIKI EXPORT")
    print("=" * 80)
    print(f"Base URL: {base_url}")
    print(f"Export:   {export_root}")
    print(f"Pages:    {len(page_ids)}")
    print(f"Output:   {out_path}")
    print()

    ok = 0
    failed = 0
    skipped_empty = 0

    with out_path.open("w", encoding="utf-8") as out_f, checksums_path.open("w", encoding="utf-8") as sum_f:
        for idx, page_id in enumerate(page_ids, 1):
            try:
                page = fetch_page(base_url, page_id, timeout=args.timeout)
                if not page.text and not args.include_empty:
                    skipped_empty += 1
                    continue
                if args.write_raw:
                    _write_raw(page, raw_dir)
                rec = _record(page, run_id, base_url)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sum_f.write(f"{rec['sha256']}  {rec['source_rel_path']}\n")
                ok += 1
                print(f"{idx}/{len(page_ids)} ok {page_id} chars={len(page.text)}")
            except (urllib.error.URLError, TimeoutError, OSError, UnicodeError) as exc:
                failed += 1
                error_rec = {
                    "source_rel_path": _source_rel_path(page_id),
                    "source_abs_path": _raw_url(base_url, page_id),
                    "ok": False,
                    "used_ocr": False,
                    "extracted_chars": 0,
                    "sha256": _sha256_text(""),
                    "error": str(exc),
                    "metadata": {
                        "source_system": "dokuwiki",
                        "page_id": page_id,
                        "url": _page_url(base_url, page_id),
                        "export_url": _raw_url(base_url, page_id),
                        "ingestion_run_id": run_id,
                    },
                    "text": "",
                }
                out_f.write(json.dumps(error_rec, ensure_ascii=False) + "\n")
                print(f"{idx}/{len(page_ids)} fail {page_id}: {exc}", file=sys.stderr)
            if args.delay:
                time.sleep(args.delay)

    manifest_path = _write_manifest(
        manifest_dir=manifest_dir,
        run_id=run_id,
        base_url=base_url,
        out_path=out_path,
        raw_dir=raw_dir,
        checksums_path=checksums_path,
        discovered_pages=len(page_ids),
        ok=ok,
        failed=failed,
        skipped_empty=skipped_empty,
    )

    print()
    print(f"OK:            {ok}")
    print(f"FAILED:        {failed}")
    print(f"SKIPPED EMPTY: {skipped_empty}")
    print(f"Output:        {out_path}")
    print(f"Manifest:      {manifest_path}")
    print(f"Checksums:     {checksums_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validated source retrieval (Pillar 2 of the research upgrade).

Replaces "blindly render anything to PDF and call it success": every fetch is
classified before anyone treats it as a source. A CAPTCHA/JS wall, paywall,
soft-404 or non-decision page must never masquerade as a retrieved decision
(docs/research-pipeline-upgrade-plan.md §3).

Pure-ish module: httpx + stdlib (+ lazy fitz/pdftotext for PDF text). No SDK,
no DB — host-testable with httpx.MockTransport and real page fixtures under
tests/fixtures/retrieval/.

Lessons encoded from the 2026-07-01 fixture capture:
  - voris serves full decisions to a browser UA (an earlier 403 was UA-based),
    but the page carries Login/Kaufen chrome → keyword paywall detection alone
    would misclassify it. Decision signals must dominate.
  - openJur's wall is a JS slider puzzle with NO 'captcha' keyword anywhere;
    the reliable signal is a tiny page without any decision content.
"""
import html as html_lib
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import httpx

#: Browser-like UA — voris (and others) gate on UA, not auth.
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) rechtmaschine-research/1.0"

DEFAULT_TIMEOUT = 25.0

#: German court file numbers: "17 L 3613/25", "7 B 19/26", "1 C 10.21" style …
_AZ_RE = re.compile(r"\b\d{1,4}\s+[A-Z]{1,4}\s+\d{1,6}[/.]\d{2,6}\b")
#: … and Bavarian style: "RN 11 K 25.33928".
_AZ_BAYERN_RE = re.compile(r"\b[A-Z]{1,3}\s+\d{1,3}\s+[A-Z]{1,3}\s+\d{2}\.\d{3,6}\b")

_DECISION_WORDS = (
    "Urteil",
    "Beschluss",
    "Tenor",
    "Tatbestand",
    "Entscheidungsgründe",
    "Gründe",
    "Leitsatz",
    "Randnummer",
)

_TAG_BLOCKS_RE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)


@dataclass
class FetchResult:
    """Outcome of retrieving one source URL."""

    status: str  # ok | blocked | notfound | not_decision | error
    resolved_url: str
    text: str = ""
    title: str = ""
    is_pdf: bool = False
    http_status: Optional[int] = None
    notes: str = ""


def extract_visible_text(html: str) -> str:
    """Visible text of an HTML page: script/style stripped, tags removed,
    entities unescaped, whitespace collapsed. Stdlib only."""
    without_blocks = _TAG_BLOCKS_RE.sub(" ", html or "")
    without_tags = _TAG_RE.sub(" ", without_blocks)
    return " ".join(html_lib.unescape(without_tags).split())


def _has_aktenzeichen(text: str) -> bool:
    return bool(_AZ_RE.search(text) or _AZ_BAYERN_RE.search(text))


def _decision_word_count(text: str) -> int:
    return sum(1 for w in _DECISION_WORDS if w in text)


def classify_page(http_status: int, visible_text: str) -> str:
    """Classify a fetched page. Decision signals dominate chrome keywords:
    a full decision with Login/Kaufen buttons is still ``ok``; a tiny page
    without any decision content is a wall regardless of HTTP 200."""
    if http_status in (404, 410):
        return "notfound"
    if http_status in (401, 403, 407, 429, 451):
        return "blocked"
    if http_status >= 400:
        return "error"

    text = visible_text or ""
    if _has_aktenzeichen(text) and _decision_word_count(text) >= 1 and len(text) >= 1200:
        return "ok"
    if len(text) < 1500:
        # CAPTCHA/JS walls, consent screens, empty shells: no decision content
        # and next to no visible text (openJur slider wall: ~600 chars).
        return "blocked"
    return "not_decision"


def extract_pdf_text(data: bytes) -> str:
    """Text of a PDF via PyMuPDF, falling back to pdftotext."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(data)
            tmp.flush()
            proc = subprocess.run(
                ["pdftotext", "-layout", tmp.name, "-"],
                capture_output=True,
                timeout=60,
            )
            return proc.stdout.decode("utf-8", errors="ignore")


def _looks_like_pdf(content_type: str, body: bytes) -> bool:
    if "application/pdf" in (content_type or "").lower():
        return True
    return body[:5] == b"%PDF-"


async def fetch_source(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> FetchResult:
    """Fetch one source URL and classify what actually came back."""
    own_client = None
    if client is None:
        own_client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": BROWSER_UA},
        )
        client = own_client
    try:
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            return FetchResult(status="error", resolved_url=url, notes=str(exc))

        resolved = str(response.url)
        content_type = response.headers.get("content-type", "")

        if _looks_like_pdf(content_type, response.content[:5]):
            try:
                text = extract_pdf_text(response.content)
            except Exception as exc:
                return FetchResult(
                    status="error", resolved_url=resolved, is_pdf=True,
                    http_status=response.status_code, notes=f"pdf extraction failed: {exc}",
                )
            status = classify_page(response.status_code, text)
            return FetchResult(
                status=status,
                resolved_url=resolved,
                text=text if status == "ok" else "",
                is_pdf=True,
                http_status=response.status_code,
            )

        raw_html = response.text
        visible = extract_visible_text(raw_html)
        status = classify_page(response.status_code, visible)
        title_match = _TITLE_RE.search(raw_html)
        title = html_lib.unescape(title_match.group(1)).strip() if title_match else ""
        return FetchResult(
            status=status,
            resolved_url=resolved,
            text=visible if status == "ok" else "",
            title=title,
            http_status=response.status_code,
        )
    finally:
        if own_client is not None:
            await own_client.aclose()

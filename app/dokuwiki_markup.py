"""DokuWiki markup -> readable plaintext for the Doktrin layer.

Pure text processing, no DB/network: cleaning wiki.aufentha.lt raw exports
(`/_export/raw/{page_id}`), splitting them into heading-scoped sections, and
chunking sections for RAG upsert. Consumed by doktrin_sync.py.

The cleaner is deliberately lossy-but-safe: unknown plugin syntax loses its
markers, never its inner text. External link URLs are KEPT — they frequently
point at the decisions and norms the Doktrin layer should steer generation
toward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rag_vocabulary import Vocabulary, load_vocabulary, normalize_normen


# --- clean_markup -----------------------------------------------------------

_HEADING_RE = re.compile(r"^\s*(={2,6})\s*(.+?)\s*=+\s*$")
# Inner text kept verbatim, tags dropped.
_VERBATIM_TAG_RE = re.compile(
    r"<(code|file|nowiki|html)\b[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL
)
_NOTE_RE = re.compile(r"<note\b[^>]*>(.*?)</note>", re.IGNORECASE | re.DOTALL)
_MEDIA_RE = re.compile(r"\{\{([^}]*)\}\}")
_LINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]*))?\]\]")
_FOOTNOTE_RE = re.compile(r"\(\((.+?)\)\)", re.DOTALL)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_UNDERLINE_RE = re.compile(r"__(.+?)__", re.DOTALL)
_MONOSPACE_RE = re.compile(r"''(.+?)''", re.DOTALL)
# Italic markers, but never the "//" inside http:// or https://.
_ITALIC_RE = re.compile(r"(?<!:)//(.+?)(?<!:)//", re.DOTALL)
_LEFTOVER_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]*>")
_LIST_RE = re.compile(r"^(\s{2,})([*-])\s+")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _link_replacement(match: "re.Match[str]") -> str:
    target = (match.group(1) or "").strip()
    label = (match.group(2) or "").strip()
    external = target.startswith(("http://", "https://"))
    if external:
        return f"{label} ({target})" if label else target
    if label:
        return label
    # Internal link without label: last path segment, underscores as spaces.
    tail = target.split(":")[-1].split("#")[0]
    return tail.replace("_", " ").strip() or target


def _media_replacement(match: "re.Match[str]") -> str:
    inner = (match.group(1) or "").strip()
    if "|" in inner:
        caption = inner.split("|", 1)[1].strip()
        return caption
    return ""


def _table_row(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or stripped[0] not in "^|":
        return None
    cells = [c.strip() for c in re.split(r"[\^|]", stripped) if c.strip()]
    # Drop colspan/alignment leftovers like ":::".
    cells = [c for c in cells if c != ":::"]
    return " | ".join(cells) if cells else ""


def clean_markup(text: str) -> str:
    """DokuWiki raw markup -> plaintext with markdown-style headings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = _VERBATIM_TAG_RE.sub(lambda m: m.group(2), text)
    text = _NOTE_RE.sub(lambda m: f"Hinweis: {m.group(1).strip()}", text)
    text = _MEDIA_RE.sub(_media_replacement, text)
    text = _LINK_RE.sub(_link_replacement, text)
    text = _FOOTNOTE_RE.sub(lambda m: f" (Fn: {m.group(1).strip()})", text)
    text = _BOLD_RE.sub(lambda m: m.group(1), text)
    text = _UNDERLINE_RE.sub(lambda m: m.group(1), text)
    text = _MONOSPACE_RE.sub(lambda m: m.group(1), text)
    text = _ITALIC_RE.sub(lambda m: m.group(1), text)
    # Unknown/plugin tags: strip the tag, keep whatever it wrapped.
    text = _LEFTOVER_TAG_RE.sub("", text)

    lines: list[str] = []
    for line in text.split("\n"):
        heading = _HEADING_RE.match(line)
        if heading:
            # DokuWiki: ====== is H1 (6 equals) down to == as H5.
            level = max(1, min(6, 7 - len(heading.group(1))))
            lines.append(f"{'#' * level} {heading.group(2).strip()}")
            continue
        row = _table_row(line)
        if row is not None:
            if row:
                lines.append(row)
            continue
        line = _LIST_RE.sub("- ", line)
        lines.append(line.rstrip())

    cleaned = "\n".join(lines)
    cleaned = _BLANK_RUN_RE.sub("\n\n", cleaned)
    return cleaned.strip()


# --- split_sections ---------------------------------------------------------


@dataclass
class Section:
    heading_path: str
    text: str


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def split_sections(clean_text: str, page_title: str) -> list[Section]:
    """Split cleaned text at headings; each section carries the full heading
    path (e.g. "Afghanistan > Abschiebungsverbote > § 60 Abs. 5 AufenthG").
    Content before the first heading becomes a lead section under the title."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    buffer: list[str] = []

    def _path() -> str:
        titles = [t for _, t in stack]
        if titles and titles[0].casefold() == (page_title or "").casefold():
            return " > ".join(titles)
        return " > ".join([page_title, *titles]) if page_title else " > ".join(titles)

    def _flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            sections.append(Section(heading_path=_path() or page_title, text=body))

    for line in clean_text.split("\n"):
        heading = _MD_HEADING_RE.match(line)
        if heading:
            _flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
            continue
        buffer.append(line)
    _flush()
    return sections


# --- chunk_section ----------------------------------------------------------


def chunk_section(text: str, target: int = 1800, hard: int = 2400) -> list[str]:
    """Paragraph-greedy chunker, copied from jurisprudence_ingest.chunk_text
    (same algorithm as rag/ingest_runner.py) so doktrin chunks match the
    granularity of the other collections."""
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


# --- context_header ---------------------------------------------------------


def context_header(page_title: str, heading_path: str, url: str) -> str:
    """Chunk context header, following rag/chunk_v0.py's dokuwiki branch."""
    parts = ["Aufenthaltswiki", (page_title or "").strip()]
    path = (heading_path or "").strip()
    if path and path.casefold() != (page_title or "").casefold():
        parts.append(path)
    lines = [f"[{' | '.join(p for p in parts if p)}]", "[Source: public legal wiki]"]
    if url:
        lines.append(f"[URL: {url}]")
    return "\n".join(lines)


# --- extract_normen ---------------------------------------------------------

# The vocabulary's canonical normen are Gesetz-first ("AufenthG § 60 Abs. 5",
# "EMRK Art. 3"); wiki prose writes them citation-style ("§ 60 Abs. 5
# AufenthG", "Art. 3 EMRK"), so both orders are matched and reordered.
_PARA_NORM_RE = re.compile(
    r"(?P<body>§\s*\d+[a-z]?"
    r"(?:\s+Abs\.\s*\d+)?"
    r"(?:\s+(?:Satz|S\.)\s*\d+)?"
    r"(?:\s+Nr\.\s*\d+)?)"
    r"\s+(?P<gesetz>AufenthG|AsylG|AsylbLG|StAG|GG|BGB|VwGO|VwVfG|BeurkG|"
    r"FamFG|AufenthV|BeschV|FreizügG/EU|SGB\s+[IVXLC]+)"
)
_ART_NORM_RE = re.compile(
    r"(?P<body>Art\.\s*\d+[a-z]?"
    r"(?:\s+Abs\.\s*\d+)?"
    r"(?:\s+lit\.\s*[a-z])?)"
    r"\s+(?P<gesetz>EMRK|GG|GRCh?|GR-Charta|AEUV|QRL|Dublin-III-VO|"
    r"RL\s*\d{4}/\d+(?:/EU|/EG)?|VO\s*\(EU\)\s*(?:Nr\.\s*)?\d+/\d+)"
)
# Gesetz-first order as written in headings/vocab ("AufenthG § 60 Abs. 5").
_GESETZ_FIRST_RE = re.compile(
    r"(?P<gesetz>AufenthG|AsylG|AsylbLG|StAG|GG|BGB|VwGO|VwVfG|BeurkG|FamFG|"
    r"AufenthV|BeschV|FreizügG/EU|EMRK|GRCh?|GR-Charta|AEUV|SGB\s+[IVXLC]+)"
    r"\s+(?P<body>(?:§|Art\.)\s*\d+[a-z]?"
    r"(?:\s+Abs\.\s*\d+)?"
    r"(?:\s+(?:Satz|S\.)\s*\d+)?"
    r"(?:\s+Nr\.\s*\d+)?"
    r"(?:\s+lit\.\s*[a-z])?)"
)

_VOCAB: Optional[Vocabulary] = None


def _vocab() -> Vocabulary:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = load_vocabulary()
    return _VOCAB


def extract_normen(text: str, vocab: Optional[Vocabulary] = None) -> list[str]:
    """Deterministic norm extraction, normalized into the shared vocabulary so
    doktrin pages and case facets (schutzgruende) live in the same canonical
    space. Unmappable citations are dropped, mirroring facets.py."""
    raw: list[str] = []
    for pattern in (_PARA_NORM_RE, _ART_NORM_RE, _GESETZ_FIRST_RE):
        for match in pattern.finditer(text):
            gesetz = re.sub(r"\s+", " ", match.group("gesetz")).strip()
            body = re.sub(r"\s+", " ", match.group("body")).strip()
            cite = f"{gesetz} {body}"
            if cite not in raw:
                raw.append(cite)
    return normalize_normen(vocab or _vocab(), raw)

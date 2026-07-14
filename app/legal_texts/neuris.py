"""NeuRIS-Quelle für Normtexte (Rechtsinformationen des Bundes, Testphase).

Löst Gesetze über die Such-API auf (exakter abbreviation-Match), lädt die
HTML-Fassung der aktuellen Expression und konvertiert sie in dasselbe
Markdown-Format, das der bestehende Extractor liest (``### § N Titel``).

Warum HTML statt XML: die API liefert Akoma-Ntoso-basiertes, maschinell
erzeugtes HTML mit stabilen Klassen (einzelvorschrift/akn-*) — ein kleiner
stdlib-Parser reicht, keine neue Abhängigkeit im Container-Image.

Pur bis auf fetch_current_law (httpx); alle Konverter sind host-testbar
gegen echte API-Fixtures (tests/fixtures/legal_texts/).
"""
import html as html_lib
import os
import re
from typing import Any, Dict, Optional, Tuple

NEURIS_BASE_URL = os.getenv(
    "NEURIS_BASE_URL", "https://testphase.rechtsinformationen.bund.de"
).rstrip("/")

#: Suchbegriffe je Gesetz (die Suche rankt nach Volltext, der exakte
#: abbreviation-Match filtert die Rausch-Treffer weg).
SEARCH_TERMS = {
    "AsylG": "Asylgesetz",
    "AufenthG": "Aufenthaltsgesetz",
    "GG": "Grundgesetz",
    "AsylbLG": "Asylbewerberleistungsgesetz",
}

_ELI_DATE_RE = re.compile(r"/(\d{4}-\d{2}-\d{2})/")
_ART_SPLIT_RE = re.compile(r'<div id="art-[^"]*"')
_HEADING_RE = re.compile(
    r'<h\d[^>]*class="einzelvorschrift"[^>]*>.*?'
    r'class="akn-num"[^>]*>(?P<num>[^<]*)</span>\s*'
    r'(?:<span[^>]*class="akn-heading"[^>]*>(?P<heading>[^<]*)</span>)?',
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_END_RE = re.compile(r"</(?:p|div|section|h\d|li|td|tr)>|<br\s*/?>", re.I)
_WS_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def pick_law_from_search(data: Any, abbreviation: str) -> Optional[Dict[str, Any]]:
    """Suchtreffer mit exakt passender Abkürzung, sonst None (kein Raten:
    lieber GitHub-Fallback als das falsche Gesetz)."""
    members = (data or {}).get("member") or []
    for member in members:
        item = member.get("item") or {}
        if item.get("abbreviation") == abbreviation and item.get("legislationIdentifier"):
            return item
    return None


def version_date_from_eli(eli: str) -> str:
    match = _ELI_DATE_RE.search(eli or "")
    return match.group(1) if match else ""


def _strip_to_text(fragment: str) -> str:
    """Markup raus, Blockgrenzen werden Zeilenumbrüche, Entities aufgelöst."""
    text = _BLOCK_END_RE.sub("\n", fragment)
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text).replace(" ", " ")
    lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANKS_RE.sub("\n\n", "\n".join(line for line in lines if line))


def html_law_to_markdown(html: str, title: str, abbreviation: str, eli: str) -> str:
    """Volles Gesetzes-HTML der API → Extractor-kompatibles Markdown.

    Nur ``art-*``-Blöcke werden übernommen (Präambel/Inhaltsübersicht und
    Abschnittsgerüst tragen keinen Normtext)."""
    version = version_date_from_eli(eli)
    out = [
        "---",
        f"Title: {title}",
        f"jurabk: {abbreviation}",
        f"quelle: NeuRIS {eli}",
        f"stand: {version}",
        "---",
        "",
        f"# {title} ({abbreviation})",
        "",
    ]
    blocks = _ART_SPLIT_RE.split(html)[1:]  # alles vor dem ersten Artikel ist Gerüst
    for block in blocks:
        match = _HEADING_RE.search(block)
        if not match:
            continue
        num = html_lib.unescape(match.group("num") or "").replace(" ", " ").strip()
        heading = html_lib.unescape(match.group("heading") or "").replace(" ", " ").strip()
        if not num:
            continue
        body = _strip_to_text(block[match.end():])
        out.append(f"### {num} {heading}".rstrip())
        out.append("")
        if body:
            out.append(body)
            out.append("")
    return "\n".join(out)


async def fetch_current_law(abbreviation: str, timeout: float = 30.0) -> Tuple[str, str]:
    """(markdown, version_date) der aktuellen Fassung aus NeuRIS.

    Wirft bei jedem Problem (nicht im Datensatz, kein HTML-Encoding,
    HTTP-Fehler) — der Aufrufer entscheidet über den Fallback."""
    import httpx

    term = SEARCH_TERMS.get(abbreviation, abbreviation)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(
            f"{NEURIS_BASE_URL}/v1/legislation",
            params={"searchTerm": term, "size": 20},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        item = pick_law_from_search(response.json(), abbreviation)
        if not item:
            raise LookupError(f"{abbreviation} nicht im NeuRIS-Datensatz (Testphase)")

        eli = item["legislationIdentifier"]
        response = await client.get(
            f"{NEURIS_BASE_URL}/v1/legislation/{eli}",
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        document = response.json()
        html_url = next(
            (
                e.get("contentUrl")
                for e in document.get("encoding") or []
                if e.get("encodingFormat") == "text/html" and e.get("contentUrl")
            ),
            None,
        )
        if not html_url:
            raise LookupError(f"{abbreviation}: kein text/html-Encoding in NeuRIS")

        response = await client.get(f"{NEURIS_BASE_URL}{html_url}")
        response.raise_for_status()
        markdown = html_law_to_markdown(
            response.text,
            title=document.get("name") or abbreviation,
            abbreviation=abbreviation,
            eli=eli,
        )
        return markdown, version_date_from_eli(eli)

"""GII-Quelle für Normtexte (gesetze-im-internet.de, BMJ — amtlich, aktuell).

Zweite Stufe der Downloader-Kette (NeuRIS → GII → GitHub): die
NeuRIS-Testphase führt GG und AsylbLG noch nicht, das GitHub-Repo
bundestag/gesetze ist tot (Stand ≤2021). GII liefert je Gesetz ein
xml.zip (gii-norm-DTD) mit konsolidiertem Stand.

Parser: stdlib xml.etree — <norm>-Elemente mit <enbez>§ N</enbez> bzw.
<enbez>Art N</enbez> werden zu ``### § N Titel``-Abschnitten im selben
Markdown-Format, das der Extractor und das looks_valid_law_markdown-Gate
erwarten. Pur bis auf fetch_current_law_gii (httpx).
"""

from __future__ import annotations

import io
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Optional, Tuple

GII_BASE_URL = os.getenv("GII_BASE_URL", "https://www.gesetze-im-internet.de").rstrip("/")

#: GII-Slugs (URL-Pfade) je Gesetz — alle vier, damit GII auch als
#: Fallback für AsylG/AufenthG taugt, falls NeuRIS ausfällt.
GII_SLUGS = {
    "AsylG": "asylvfg_1992",
    "AufenthG": "aufenthg_2004",
    "GG": "gg",
    "AsylbLG": "asylblg",
}

_ENBEZ_PROVISION_RE = re.compile(r"^(?:§+\s|Art\s)")
_NBSP = " "

#: Nach diesen Tags endet ein Textblock (Absatz, Listenpunkt, Zeile).
_BLOCK_TAGS = {"P", "DD", "LA", "BR", "PRE", "TITLE", "SUBTITLE"}


def _flatten(elem: ET.Element) -> str:
    """Content-Baum (P/DL/DT/DD/LA/BR …) zu Fließtext mit Zeilenumbrüchen
    an Block-Grenzen; DT-Gliederungsnummern stehen inline vor ihrem DD."""
    parts: list[str] = []

    def walk(e: ET.Element) -> None:
        tag = (e.tag or "").upper()
        if e.text:
            parts.append(e.text)
        if tag == "DT":
            parts.append(" ")
        for child in e:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if tag in _BLOCK_TAGS:
            parts.append("\n")

    walk(elem)
    text = "".join(parts).replace(_NBSP, " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    out = "\n".join(line for line in lines if line)
    return out.strip()


def _element_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).replace(_NBSP, " ").strip()


def _builddate_iso(root: ET.Element) -> str:
    raw = (root.get("builddate") or "").strip()
    if not re.match(r"^\d{8}", raw):
        raise ValueError(f"GII-XML ohne plausibles builddate: {raw!r}")
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def law_xml_to_markdown(xml_text: str, abbreviation: str) -> Tuple[str, str]:
    """(markdown, version) aus einem GII-Norm-XML; wirft bei allem, was
    kein gii-norm-Dokument ist (Downloader fällt dann auf GitHub zurück)."""
    # XXE-/Billion-Laughs-Guard ohne defusedxml (nicht im Container-Image):
    # beide Angriffe brauchen ENTITY-Deklarationen; das GII-Format nutzt
    # keine (nur eine externe DTD-Referenz, die expat ohnehin nicht lädt).
    if re.search(r"<!ENTITY", xml_text, re.IGNORECASE):
        raise ValueError("XML mit ENTITY-Deklaration abgelehnt")
    root = ET.fromstring(xml_text)
    if root.tag != "dokumente":
        raise ValueError(f"kein gii-norm-XML (root={root.tag!r})")
    version = _builddate_iso(root)

    norms = root.findall("norm")
    if not norms:
        raise ValueError("gii-norm-XML ohne <norm>-Elemente")

    stammnorm = norms[0]
    title = _element_text(stammnorm.find("metadaten/langue")) or abbreviation
    geaendert = ""
    for standangabe in stammnorm.findall("metadaten/standangabe"):
        if _element_text(standangabe.find("standtyp")) == "Stand":
            geaendert = _element_text(standangabe.find("standkommentar"))

    out = [
        "---",
        f"Title: {title}",
        f"jurabk: {abbreviation}",
        f"quelle: GII {GII_BASE_URL}/{GII_SLUGS.get(abbreviation, abbreviation.lower())}/xml.zip",
        f"stand: {version}",
    ]
    if geaendert:
        out.append(f"geaendert: {geaendert}")
    out += ["---", "", f"# {title} ({abbreviation})", ""]

    for norm in norms:
        enbez = _element_text(norm.find("metadaten/enbez"))
        if not _ENBEZ_PROVISION_RE.match(enbez):
            continue
        titel = _element_text(norm.find("metadaten/titel"))
        heading = f"### {enbez} {titel}".rstrip()
        content = norm.find("textdaten/text/Content")
        body = _flatten(content) if content is not None else ""
        out.append(heading)
        out.append("")
        if body:
            out.append(body)
            out.append("")

    return "\n".join(out), version


async def fetch_current_law_gii(law: str) -> Tuple[str, str]:
    """(markdown, version) live von gesetze-im-internet.de; wirft bei
    jedem Problem (HTTP, Zip, Parse) — Caller entscheidet über Fallback."""
    import httpx

    slug = GII_SLUGS.get(law)
    if not slug:
        raise ValueError(f"kein GII-Slug für {law}")
    url = f"{GII_BASE_URL}/{slug}/xml.zip"
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_names:
            raise ValueError(f"{url}: kein XML im Zip")
        xml_text = zf.read(xml_names[0]).decode("utf-8")
    return law_xml_to_markdown(xml_text, law)

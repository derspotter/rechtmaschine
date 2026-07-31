"""Post-processing for j-lawyer template renders (send-to-j-lawyer pipeline).

Learnings aus der jlawyer-cli/rubrum-cli Drafting-Pipeline (Skills `api` +
`drafting`), hier für die Rechtmaschine-Endpoints nutzbar gemacht:

1. Der Server-Render setzt einen mehrzeiligen Platzhalterwert als EINEN
   Absatz mit line-breaks — keine echten Absätze, kein sauberer Blocksatz.
   Außerdem 500t der Render bei Werten über ~20 kB. Deshalb wird nur der
   literale Marker "INGO_TEXT" gerendert und der Body danach lokal als echte
   Absätze eingesetzt (Leerzeile = Absatztrenner, einfacher Umbruch wird zu
   Fließtext verschmolzen — Body-Vertrag der CLI-Pipeline).
2. Der Body-Stil wird auf Blocksatz (fo:text-align=justify) gesetzt.
3. Danach läuft die Kanzlei-Formatierung (`rubrum_lib.format_file`):
   Anträge fett+eingerückt, Aufzählungen eingerückt, Rollen rechtsbündig,
   Leerzeilen-Grammatik, Platzhalter-Leerzeichen-Artefakte usw.
4. `rubrum_lib.check_odt` verifiziert und meldet Verstöße (bricht nicht ab).

Die Patch-Helfer sind aus ~/.codex/skills/api/scripts/jlawyer_cli.py
übernommen (Stand 2026-07-30) — Regel-Fixes dort auch hier nachziehen.
"""

import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

# stdlib ET ist hier vertretbar: geparst werden ausschließlich ODTs, die unser
# eigener j-lawyer-Server (VPN, Basic-Auth) gerade selbst gerendert hat — kein
# Fremd-XML. Bei einer künftigen Image-Aktualisierung auf defusedxml umstellen.
from xml.etree import ElementTree as ET

import rubrum_lib

# Der Wert, der beim Server-Render als Platzhalterwert gesendet wird - der
# lokale Patcher ersetzt genau diesen Marker-Absatz durch die echten Absätze.
INGO_TEXT_MARKER = "INGO_TEXT"


def _normalize_ingo_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    return normalized.strip()


def _paragraphs_from_ingo_text(text: str) -> List[str]:
    normalized = _normalize_ingo_text(text)
    if not normalized:
        return []
    paragraphs = re.split(r"\n\s*\n+", normalized)
    cleaned: List[str] = []
    for paragraph in paragraphs:
        line_joined = re.sub(r"[ \t]*\n[ \t]*", " ", paragraph.strip())
        line_joined = re.sub(r"[ \t]{2,}", " ", line_joined)
        cleaned.append(line_joined)
    return cleaned


def _paragraph_text_with_breaks(node: ET.Element) -> str:
    text_ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    parts: List[str] = []

    def walk(element: ET.Element) -> None:
        if element.text:
            parts.append(element.text)
        for child in list(element):
            if child.tag == text_ns + "line-break":
                parts.append("\n")
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(node)
    return "".join(parts)


def _justify_paragraph_styles(xml_bytes: bytes, style_names: set) -> bytes:
    if not style_names:
        return xml_bytes

    style_ns = "urn:oasis:names:tc:opendocument:xmlns:style:1.0"
    fo_ns = "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
    q_style_style = "{" + style_ns + "}style"
    q_style_family = "{" + style_ns + "}family"
    q_style_name = "{" + style_ns + "}name"
    q_paragraph_properties = "{" + style_ns + "}paragraph-properties"
    q_fo_text_align = "{" + fo_ns + "}text-align"
    q_justify_single_word = "{" + style_ns + "}justify-single-word"

    root = ET.fromstring(xml_bytes)
    changed = False
    for style in root.iter(q_style_style):
        if style.attrib.get(q_style_family) != "paragraph":
            continue
        if style.attrib.get(q_style_name) not in style_names:
            continue

        properties = style.find(q_paragraph_properties)
        if properties is None:
            properties = ET.Element(q_paragraph_properties)
            style.insert(0, properties)
        if properties.attrib.get(q_fo_text_align) != "justify":
            properties.set(q_fo_text_align, "justify")
            changed = True
        if properties.attrib.get(q_justify_single_word) != "false":
            properties.set(q_justify_single_word, "false")
            changed = True

    if not changed:
        return xml_bytes
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _patch_ingo_text_paragraphs(source_odt: Path, target_odt: Path, ingo_text: str) -> bool:
    office_ns = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    text_ns = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    q_text_p = "{" + text_ns + "}p"
    q_text_span = "{" + text_ns + "}span"
    q_style_name = "{" + text_ns + "}style-name"

    expected_text = _normalize_ingo_text(ingo_text)
    replacement_paragraphs = _paragraphs_from_ingo_text(ingo_text)
    if not expected_text or not replacement_paragraphs:
        raise ValueError("Body-Text ist nach Normalisierung leer - Patch verweigert.")
    placeholder_markers = {
        "INGO_TEXT",
        "{{INGO_TEXT}}",
        "{{<text:span text:style-name=\"T15\">INGO_TEXT</text:span>}}",
        "{{<text:span text:style-name=&quot;T15&quot;>INGO_TEXT</text:span>}}",
    }

    with zipfile.ZipFile(source_odt, "r") as archive:
        content_xml = archive.read("content.xml")
        infolist = archive.infolist()
        original_data = {info.filename: archive.read(info.filename) for info in infolist}

    root = ET.fromstring(content_xml)
    office_text = root.find(f".//{{{office_ns}}}text")
    if office_text is None:
        raise ValueError("ODT content.xml enthält kein office:text.")

    matched = False
    para_style = None
    parents = [office_text, *list(office_text.iter())]
    for parent in parents:
        children = list(parent)
        for index, child in enumerate(children):
            if child.tag != q_text_p:
                continue
            normalized_paragraph_text = _normalize_ingo_text(_paragraph_text_with_breaks(child))
            if normalized_paragraph_text != expected_text and normalized_paragraph_text not in placeholder_markers:
                continue

            matched = True
            para_style = child.attrib.get(q_style_name)
            span_styles = [
                span.attrib.get(q_style_name)
                for span in child.findall(f"./{q_text_span}")
                if span.attrib.get(q_style_name)
            ]
            span_style = span_styles[0] if len(set(span_styles)) == 1 and span_styles else None
            tail_text = child.tail

            new_nodes: List[ET.Element] = []
            for paragraph_text in replacement_paragraphs:
                new_p = ET.Element(q_text_p)
                if para_style:
                    new_p.set(q_style_name, para_style)
                if span_style:
                    new_span = ET.SubElement(new_p, q_text_span)
                    new_span.set(q_style_name, span_style)
                    new_span.text = paragraph_text
                else:
                    new_p.text = paragraph_text
                new_nodes.append(new_p)

            child.tail = None
            parent.remove(child)
            for offset, new_node in enumerate(new_nodes):
                parent.insert(index + offset, new_node)
            if new_nodes:
                new_nodes[-1].tail = tail_text
            break
        if matched:
            break

    if not matched:
        return False

    body_style_names = {style for style in (para_style,) if style}
    updated_content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    updated_content = _justify_paragraph_styles(updated_content, body_style_names)
    updated_styles = (
        _justify_paragraph_styles(original_data["styles.xml"], body_style_names)
        if "styles.xml" in original_data
        else None
    )
    target_odt.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target_odt, "w") as archive:
        for info in infolist:
            if info.filename == "content.xml":
                data = updated_content
            elif info.filename == "styles.xml" and updated_styles is not None:
                data = updated_styles
            else:
                data = original_data[info.filename]
            archive.writestr(info, data)
    return True


def postprocess_rendered_odt(
    odt_bytes: bytes,
    body_text: str,
    behoerde: bool = False,
) -> Tuple[Optional[bytes], List[tuple]]:
    """Body-Absätze einsetzen, Blocksatz + Kanzlei-Formatierung anwenden, prüfen.

    Returns (finalized_bytes, check_fails). finalized_bytes is None when the
    marker/body paragraph could not be located in the rendered ODT.
    """
    with tempfile.TemporaryDirectory(prefix="rm-jlawyer-odt-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        rendered = tmp_path / "rendered.odt"
        final = tmp_path / "final.odt"
        rendered.write_bytes(odt_bytes)

        if not _patch_ingo_text_paragraphs(rendered, final, body_text):
            return None, []

        rubrum_lib.format_file(str(final), behoerde=behoerde)
        check_fails = rubrum_lib.check_odt(str(final), behoerde=behoerde)
        return final.read_bytes(), check_fails

"""Direct j-lawyer Akte reading for case-memory reflection.

The Akte stays in j-lawyer: nothing here creates Document rows or stores
document files in Rechtmaschine. Documents are fetched into temp files,
distilled into memory proposals, and discarded. A small per-case state
file under ``jlawyer_seen/`` remembers which j-lawyer document ids have
already been read so repeated runs only process new material.
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import httpx

JLAWYER_BASE_URL = (os.environ.get("JLAWYER_BASE_URL") or "").strip().rstrip("/")
JLAWYER_USERNAME = os.environ.get("JLAWYER_USERNAME")
JLAWYER_PASSWORD = os.environ.get("JLAWYER_PASSWORD")

JLAWYER_SEEN_DIR = Path(__file__).resolve().parent / "jlawyer_seen"

# Leading file number in Rechtmaschine case names, e.g. "089/26 Balulov".
_FILE_NUMBER_RE = re.compile(r"\b\d{1,4}/\d{2}\b")

# Transport/boilerplate artifacts that carry no case substance.
_JUNK_NAME_RE = re.compile(
    r"deckblatt|sendebericht|sendeprotokoll|empfangsbe|pr(ue|ü)fprotokoll|(?<![a-z])eeb(?![a-z])|"
    r"(?<![a-z])vhn(?![a-z])|xjustiz|\.p7s$|leitdokument|merkblatt|briefvorlage|"
    r"akteneinsichtkanzlei|nachrichtentext",
    re.IGNORECASE,
)

_READABLE_EXT_RE = re.compile(r"\.(pdf|txt|eml|html?|odt|jpe?g|png|webp|bea)$", re.IGNORECASE)
_IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp)$", re.IGNORECASE)

# Numbering token of BAMF-Akte exports (e.g. "_1322_080_"); the same token
# appears in filenames already imported into Rechtmaschine.
_AKTE_TOKEN_RE = re.compile(r"_(\d{3,5}_\d{3})_")


def is_configured() -> bool:
    return bool(JLAWYER_BASE_URL and JLAWYER_USERNAME and JLAWYER_PASSWORD)


def _api_base() -> str:
    if JLAWYER_BASE_URL.endswith("/rest"):
        return JLAWYER_BASE_URL
    return f"{JLAWYER_BASE_URL}/rest"


def _auth() -> tuple:
    return (JLAWYER_USERNAME, JLAWYER_PASSWORD)


def extract_file_number(case_name: str) -> Optional[str]:
    match = _FILE_NUMBER_RE.search(case_name or "")
    return match.group(0) if match else None


def _normalize_file_number(value: str) -> str:
    return re.sub(r"\s+", "", value or "").casefold()


async def resolve_case_id(file_number: str) -> Optional[str]:
    """Find the active j-lawyer case whose fileNumber matches."""
    wanted = _normalize_file_number(file_number)
    if not wanted:
        return None
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{_api_base()}/v1/cases/list/active", auth=_auth())
        response.raise_for_status()
        cases = response.json() or []
    matches = [
        case for case in cases
        if _normalize_file_number(str(case.get("fileNumber") or "")) == wanted
    ]
    if len(matches) != 1:
        return None
    return str(matches[0].get("id") or "") or None


async def list_documents(jl_case_id: str) -> list:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{_api_base()}/v1/cases/{jl_case_id}/documents", auth=_auth()
        )
        response.raise_for_status()
        return response.json() or []


async def fetch_document_content(jl_document_id: str) -> bytes:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(
            f"{_api_base()}/v1/cases/document/{jl_document_id}/content", auth=_auth()
        )
        response.raise_for_status()
        payload = response.json() or {}
    encoded = payload.get("base64content") or ""
    return base64.b64decode(encoded) if encoded else b""


def is_junk_name(name: str) -> bool:
    return bool(_JUNK_NAME_RE.search(name or ""))


def is_readable_name(name: str) -> bool:
    return bool(_READABLE_EXT_RE.search(name or ""))


def is_image_name(name: str) -> bool:
    return bool(_IMAGE_EXT_RE.search(name or ""))


def doc_stem(name: str) -> str:
    """Filename without extension, for twin detection (draft.odt vs export.pdf)."""
    return os.path.splitext((name or "").strip())[0].casefold()


def akte_token(name: str) -> Optional[str]:
    match = _AKTE_TOKEN_RE.search(name or "")
    return match.group(1) if match else None


def _strip_html(value: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def extract_mail_text(content: bytes, name: str) -> str:
    """Plain text from .eml (headers + body) or .html attachments."""
    if name.lower().endswith(".eml"):
        import email
        from email import policy

        msg = email.message_from_bytes(content, policy=policy.default)
        header = (
            f"Von: {msg.get('From', '?')}\nAn: {msg.get('To', '?')}\n"
            f"Datum: {msg.get('Date', '?')}\nBetreff: {msg.get('Subject', '?')}\n"
        )
        body = msg.get_body(preferencelist=("plain", "html"))
        text = body.get_content() if body else ""
        if not (text or "").strip():
            # Oddly structured multipart mails where get_body() finds nothing.
            parts = []
            for part in msg.walk():
                if part.get_content_maintype() == "text" and not part.get_filename():
                    try:
                        parts.append(part.get_content())
                    except Exception:
                        continue
            text = "\n".join(parts)
        if "<" in text[:300] and ">" in text[:300]:
            text = _strip_html(text)
        attachments = [f for part in msg.walk() if (f := part.get_filename())]
        if attachments:
            text = f"{text}\n\nAnlagen: {', '.join(attachments)}"
        return f"{header}\n{text}".strip()
    return _strip_html(content.decode("utf-8", errors="replace"))


def extract_bea_text(content: bytes, name: str) -> str:
    """Envelope text from a j-lawyer .bea export (decrypted beA message container).

    Deliberately envelope-only: direction, date, sender/recipient, Aktenzeichen,
    subject and attachment names — the send-event facts. The attached Schriftsatz
    PDFs usually exist as separate Akte documents (and are base64 in here), so
    decoding them would mostly duplicate content the reflect pass already reads;
    the twin detection works by filename stem and could not catch that.
    Returns "" on unparseable XML (the doc is then skipped as unreadable)."""
    import xml.etree.ElementTree as ET

    # beA content is sent by external parties. Legitimate j-lawyer exports never
    # carry a DTD, so refuse any input with entity/doctype declarations outright —
    # this blocks XXE and billion-laughs without needing defusedxml.
    if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
        return ""
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return ""

    def txt(tag: str) -> str:
        return (root.findtext(tag) or "").strip()

    received = (txt("received") or txt("created"))[:16].replace("T", " ")
    recipients = ", ".join(
        n for r in root.findall("recipient") if (n := (r.findtext("name") or "").strip())
    )
    attachments = ", ".join(
        fn
        for a in root.findall("attachment")
        if (fn := (a.get("fileName") or "").strip())
        and not re.search(r"vhn\.xml|xjustiz|nachrichtentext", fn, re.IGNORECASE)
    )
    lines = [f"beA-Nachricht vom {received}" if received else "beA-Nachricht"]
    for label, value in (
        ("Absender", txt("senderName")),
        ("Empfänger", recipients),
        ("Betreff", txt("subject")),
        ("Aktenzeichen Kanzlei", txt("reference")),
        ("Aktenzeichen Gericht/Behörde", txt("referenceJustice")),
        ("Übersandte Anlagen", attachments),
    ):
        if value:
            lines.append(f"{label}: {value}")
    body = (txt("body") or "").strip()
    if body:
        lines.append(f"Nachrichtentext: {body[:1500]}")
    return "\n".join(lines)


def extract_odt_text(content: bytes) -> str:
    """Plain text from an OpenDocument file (content.xml inside the zip)."""
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        xml = archive.read("content.xml").decode("utf-8", errors="replace")
    # Paragraph/heading ends and explicit breaks become newlines, tab and
    # space elements become spaces, before all remaining tags are stripped.
    xml = re.sub(r"</text:(p|h)>", "\n", xml)
    xml = re.sub(r"<text:line-break\s*/>", "\n", xml)
    xml = re.sub(r"<text:(tab|s)(\s[^>]*)?/>", " ", xml)
    return _strip_html(xml)


def _seen_path(case_id: Any) -> Path:
    return JLAWYER_SEEN_DIR / f"{case_id}.json"


def load_seen(case_id: Any) -> set:
    try:
        return set(json.loads(_seen_path(case_id).read_text()))
    except Exception:
        return set()


def save_seen(case_id: Any, seen: set) -> None:
    try:
        JLAWYER_SEEN_DIR.mkdir(parents=True, exist_ok=True)
        _seen_path(case_id).write_text(json.dumps(sorted(seen)))
    except Exception as exc:
        print(f"[MEMORY WARN] Failed to persist j-lawyer seen state: {exc}")

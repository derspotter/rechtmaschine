"""
Download German federal laws — NeuRIS-first, GitHub-Fallback.

Historie: bis 2026-07-14 kam alles vom GitHub-Repo bundestag/gesetze. Das
Repo ist faktisch tot (AsylG-Datei zuletzt 2018 committet, Inhalt Stand
2021) — die GEAS-Reform (in Kraft 12.06.2026) fehlte komplett. Primärquelle
ist jetzt die NeuRIS-API (Rechtsinformationen des Bundes, amtlich
konsolidiert); GitHub bleibt Fallback für Gesetze, die die Testphase noch
nicht führt (GG, AsylbLG — Stand 2026-07-14).

Kernregel: eine unplausible Konvertierung überschreibt NIE eine vorhandene
Datei (looks_valid_law_markdown-Gate, fail-closed).
"""

import asyncio
import re
from pathlib import Path
from typing import Optional

import httpx


# GitHub raw content base URL (Fallback-Quelle)
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/bundestag/gesetze/master"

# Law mappings: abbreviation -> (directory_path, filename)
LAWS = {
    "AsylG": ("a/asylvfg_1992", "index.md"),
    "AufenthG": ("a/aufenthg_2004", "index.md"),
    "GG": ("g/gg", "index.md"),
    "AsylbLG": ("a/asylblg", "index.md"),
}

# Local storage directory
LEGAL_TEXTS_DIR = Path(__file__).parent.parent / "legal_texts" / "laws"

#: Diese Normen MÜSSEN als Überschrift vorkommen, sonst ist die Datei kaputt
#: oder das falsche Gesetz. GG-Anker im GitHub-Format ("Art 16a").
VALIDATION_ANCHORS = {
    "AsylG": ("§ 3", "§ 30", "§ 77"),
    "AufenthG": ("§ 25", "§ 60", "§ 104"),
    "GG": ("Art 16a",),
    "AsylbLG": ("§ 2", "§ 4"),
}

#: Mindestzahl an §-/Art-Überschriften je Gesetz (grobe Untergrenze).
MIN_PROVISIONS = {"AsylG": 60, "AufenthG": 80, "GG": 100, "AsylbLG": 10}

_PROVISION_HEADING_RE = re.compile(r"^#+\s*(?:§|Art\.?)\s*\S+", re.M)


def get_law_path(law: str) -> Path:
    """Get the local file path for a law."""
    LEGAL_TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    return LEGAL_TEXTS_DIR / f"{law.lower()}.md"


def looks_valid_law_markdown(law: str, markdown: str) -> bool:
    """Plausibilitäts-Gate vor jedem Schreiben: genug Normen-Überschriften
    und alle Anker-Normen vorhanden."""
    if not (markdown or "").strip():
        return False
    headings = _PROVISION_HEADING_RE.findall(markdown)
    if len(headings) < MIN_PROVISIONS.get(law, 10):
        return False
    for anchor in VALIDATION_ANCHORS.get(law, ()):
        if not re.search(rf"^#+\s*{re.escape(anchor)}(?:\s|$)", markdown, re.M):
            return False
    return True


async def _neuris_fetch(law: str):
    """(markdown, version_date) aus NeuRIS; wirft bei jedem Problem."""
    from .neuris import fetch_current_law

    return await fetch_current_law(law)


async def _github_fetch(law: str) -> str:
    directory, filename = LAWS[law]
    url = f"{GITHUB_RAW_BASE}/{directory}/{filename}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def download_law(
    law: str,
    force: bool = False,
    neuris_fetch=None,
    github_fetch=None,
) -> bool:
    """Download a specific law: NeuRIS zuerst, GitHub als Fallback.

    Args:
        law: Law abbreviation (AsylG, AufenthG, GG, AsylbLG)
        force: Force re-download even if file exists
        neuris_fetch/github_fetch: injizierbare Seams für Tests

    Returns:
        True if a valid file is in place afterwards, False otherwise
    """
    if law not in LAWS:
        print(f"[DOWNLOADER] Unknown law: {law}")
        return False

    local_path = get_law_path(law)
    if local_path.exists() and not force:
        print(f"[DOWNLOADER] {law} already exists at {local_path}")
        return True

    neuris_fetch = neuris_fetch or _neuris_fetch
    github_fetch = github_fetch or _github_fetch

    try:
        markdown, version = await neuris_fetch(law)
        if looks_valid_law_markdown(law, markdown):
            local_path.write_text(markdown, encoding="utf-8")
            print(f"[DOWNLOADER] {law}: NeuRIS-Fassung {version} gespeichert ({len(markdown)} chars)")
            return True
        print(f"[DOWNLOADER] {law}: NeuRIS-Konvertierung unplausibel — Fallback GitHub")
    except Exception as exc:
        print(f"[DOWNLOADER] {law}: NeuRIS nicht nutzbar ({exc}) — Fallback GitHub")

    try:
        markdown = await github_fetch(law)
        if not looks_valid_law_markdown(law, markdown):
            print(f"[DOWNLOADER] {law}: GitHub-Inhalt unplausibel — Datei bleibt unverändert")
            return False
        local_path.write_text(markdown, encoding="utf-8")
        print(
            f"[DOWNLOADER] {law}: GitHub-Fallback gespeichert ({len(markdown)} chars) — "
            f"ACHTUNG: bundestag/gesetze hinkt Gesetzesänderungen ggf. Jahre hinterher"
        )
        return True
    except Exception as exc:
        print(f"[DOWNLOADER] Failed to download {law}: {exc}")
        return False


async def download_all_laws(force: bool = False) -> dict[str, bool]:
    """
    Download all configured laws (NeuRIS-first, GitHub-Fallback).

    Args:
        force: Force re-download even if files exist

    Returns:
        Dict mapping law name to success status
    """
    print("[DOWNLOADER] Starting download of all laws...")

    tasks = [download_law(law, force=force) for law in LAWS.keys()]
    results = await asyncio.gather(*tasks)

    status = dict(zip(LAWS.keys(), results))

    success_count = sum(1 for success in results if success)
    print(f"[DOWNLOADER] Downloaded {success_count}/{len(LAWS)} laws successfully")

    return status


async def update_law(law: str) -> bool:
    """
    Update a specific law to the latest version.

    Args:
        law: Law abbreviation (AsylG, AufenthG, GG, AsylbLG)

    Returns:
        True if updated successfully, False otherwise
    """
    return await download_law(law, force=True)


def stored_version(law: str) -> Optional[str]:
    """Fassungsdatum aus dem Front-Matter der lokalen Datei ("stand: ...");
    None bei GitHub-Bestand ohne Versionsstempel."""
    path = get_law_path(law)
    if not path.exists():
        return None
    head = path.read_text(encoding="utf-8")[:600]
    match = re.search(r"^stand:\s*(\d{4}-\d{2}-\d{2})", head, re.M)
    return match.group(1) if match else None


# Synchronous wrappers for convenience
def download_all_laws_sync(force: bool = False) -> dict[str, bool]:
    """Synchronous version of download_all_laws"""
    return asyncio.run(download_all_laws(force=force))


def download_law_sync(law: str, force: bool = False) -> bool:
    """Synchronous version of download_law"""
    return asyncio.run(download_law(law, force=force))

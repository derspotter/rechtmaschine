"""
Download German federal laws from GitHub bundestag/gesetze repository.
"""

import asyncio
from pathlib import Path
from typing import Optional
import httpx


# GitHub raw content base URL
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


def get_law_path(law: str) -> Path:
    """Get the local file path for a law."""
    LEGAL_TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    return LEGAL_TEXTS_DIR / f"{law.lower()}.md"


async def download_law(law: str, force: bool = False) -> bool:
    """
    Download a specific law from GitHub.

    Args:
        law: Law abbreviation (AsylG, AufenthG, GG, AsylbLG)
        force: Force re-download even if file exists

    Returns:
        True if downloaded successfully, False otherwise
    """
    if law not in LAWS:
        print(f"[DOWNLOADER] Unknown law: {law}")
        return False

    local_path = get_law_path(law)

    if local_path.exists() and not force:
        print(f"[DOWNLOADER] {law} already exists at {local_path}")
        return True

    directory, filename = LAWS[law]
    url = f"{GITHUB_RAW_BASE}/{directory}/{filename}"

    print(f"[DOWNLOADER] Downloading {law} from {url}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Write to file
            local_path.write_text(response.text, encoding="utf-8")
            print(f"[DOWNLOADER] Successfully downloaded {law} ({len(response.text)} chars)")
            return True

    except Exception as e:
        print(f"[DOWNLOADER] Failed to download {law}: {e}")
        return False


async def download_all_laws(force: bool = False) -> dict[str, bool]:
    """
    Download all configured laws from GitHub.

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


# Synchronous wrappers for convenience
def download_all_laws_sync(force: bool = False) -> dict[str, bool]:
    """Synchronous version of download_all_laws"""
    return asyncio.run(download_all_laws(force=force))


def download_law_sync(law: str, force: bool = False) -> bool:
    """Synchronous version of download_law"""
    return asyncio.run(download_law(law, force=force))

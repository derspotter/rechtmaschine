"""Hybrid-Downloader: Kette NeuRIS → GII → GitHub, Validierungs-Gate.

Kernregel: eine kaputte oder unvollständige Konvertierung überschreibt NIE
eine vorhandene gute Datei — lieber alter Stand plus Warnung als Datenmüll
im Prompt-Pfad.

Run: .venv/bin/python -m pytest tests/test_legal_texts_downloader.py -q
"""
import asyncio

import pytest

import legal_texts.downloader as dl
from legal_texts.downloader import download_law, looks_valid_law_markdown


def _valid_md(law="AsylG", extra=""):
    anchors = {
        "AsylG": ["§ 3", "§ 30", "§ 77"],
        "AufenthG": ["§ 25", "§ 60", "§ 104"],
    }[law]
    parts = [f"---\njurabk: {law}\nstand: 2026-07-10\n---\n"]
    needed = max(dl.MIN_PROVISIONS[law], 3)
    nums = [a.split()[-1] for a in anchors]
    nums += [str(n) for n in range(200, 200 + needed)]
    for n in nums:
        parts.append(f"### § {n} Titel\n\nText zu § {n}.\n")
    return "\n".join(parts) + extra


def _run(coro):
    return asyncio.run(coro)


async def _explode(law):
    raise AssertionError(f"Quelle darf für {law} nicht angefragt werden")


async def _unavailable(law):
    raise LookupError("nicht im Datensatz")


# --- Validierungs-Gate ----------------------------------------------------------


def test_valid_markdown_passes_gate():
    assert looks_valid_law_markdown("AsylG", _valid_md()) is True


def test_gate_rejects_missing_anchor_and_thin_files():
    md = _valid_md().replace("### § 30 Titel", "### § 999 Titel")
    assert looks_valid_law_markdown("AsylG", md) is False
    assert looks_valid_law_markdown("AsylG", "### § 3 Titel\n\nnur einer") is False
    assert looks_valid_law_markdown("AsylG", "") is False


def test_gate_accepts_current_github_format(tmp_path):
    # Der GitHub-Bestand (### § N Titel) muss das Gate ebenfalls passieren,
    # sonst wäre der Fallback wertlos.
    real = (dl.LEGAL_TEXTS_DIR / "asylg.md")
    if not real.exists():
        pytest.skip("kein lokaler Bestand")
    assert looks_valid_law_markdown("AsylG", real.read_text(encoding="utf-8")) is True


# --- Quellen-Auswahl ------------------------------------------------------------


@pytest.fixture()
def law_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "LEGAL_TEXTS_DIR", tmp_path)
    return tmp_path


def test_neuris_success_asks_no_other_source(law_dir):
    async def neuris(law):
        return _valid_md(law), "2026-07-10"

    ok = _run(download_law(
        "AsylG", force=True,
        neuris_fetch=neuris, gii_fetch=_explode, github_fetch=_explode,
    ))
    assert ok is True
    content = (law_dir / "asylg.md").read_text(encoding="utf-8")
    assert "stand: 2026-07-10" in content


def test_neuris_failure_falls_back_to_gii(law_dir):
    async def gii(law):
        return _valid_md(law).replace("stand: 2026-07-10", "stand: 2026-06-11"), "2026-06-11"

    ok = _run(download_law(
        "AsylG", force=True,
        neuris_fetch=_unavailable, gii_fetch=gii, github_fetch=_explode,
    ))
    assert ok is True
    assert "stand: 2026-06-11" in (law_dir / "asylg.md").read_text(encoding="utf-8")


def test_gii_failure_falls_back_to_github(law_dir):
    async def github(law):
        return _valid_md(law)

    ok = _run(download_law(
        "AsylG", force=True,
        neuris_fetch=_unavailable, gii_fetch=_unavailable, github_fetch=github,
    ))
    assert ok is True
    assert (law_dir / "asylg.md").exists()


def test_invalid_conversions_never_overwrite_good_file(law_dir):
    good = _valid_md()
    (law_dir / "asylg.md").write_text(good, encoding="utf-8")

    async def broken(law):
        return "### § 1 kaputt\n", "2026-07-10"  # Gate muss das abfangen

    async def github_down(law):
        raise RuntimeError("github down")

    ok = _run(download_law(
        "AsylG", force=True,
        neuris_fetch=broken, gii_fetch=broken, github_fetch=github_down,
    ))
    assert ok is False
    assert (law_dir / "asylg.md").read_text(encoding="utf-8") == good


def test_existing_file_skipped_without_force(law_dir):
    (law_dir / "asylg.md").write_text(_valid_md(), encoding="utf-8")

    ok = _run(download_law(
        "AsylG",
        neuris_fetch=_explode, gii_fetch=_explode, github_fetch=_explode,
    ))
    assert ok is True

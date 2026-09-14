"""Einzelabruf eines beck-online-Dokuments mit einer MITGEBRACHTEN Sitzung.

STATUS: Testwerkzeug, auf ausdrueckliche Anweisung von Jay (07.09.2026) gebaut,
nachdem die Bedenken benannt waren. Bewusst NICHT an die Ingest-Pipeline
angeschlossen — beck-online steht in `_LICENSED_DOMAINS` von
draft_citation_ingest, die automatische Beschaffung fasst es also nie an.

Kein Login im Code, kein Passwort, keine Zugangsdaten.
--------------------------------------------------------
Der Server ist headless, es gibt hier keinen Browser zum Anmelden. Deshalb
bringt der Anwalt die Sitzung mit: er meldet sich in SEINEM Browser an und
exportiert die Cookies fuer beck-online.beck.de als JSON. Dieses Skript laedt
sie als Playwright-`storage_state` und ruft damit genau ein Dokument ab. Der
Abruf bleibt damit die lizenzierte Nutzung des Anwalts, automatisiert ist nur
die Ablage.

Warum die Bremsen unten kein Zierrat sind:
  * Die Nutzungsbedingungen von beck-online untersagen automatisierten Abruf.
    Sanktioniert wird nicht das Skript, sondern der KANZLEIZUGANG.
  * §§ 87a ff. UrhG schuetzen die Datenbank gegen Entnahme wesentlicher Teile.
    Der Entscheidungstext selbst ist nach § 5 UrhG gemeinfrei, becks
    Aufbereitung und die Sammlung sind es nicht.
  * Erkannt wird nicht der einzelne Abruf, sondern das MUSTER: gleiche
    Intervalle, keine Lesezeiten, Dokumente in Serie, Zugriffe nachts.
    Genau dagegen richten sich Mindestabstand, Stundenkontingent und die
    Beschraenkung auf ein Dokument je Aufruf.

Sitzung bereitstellen (einmalig, am eigenen Rechner):
    Im angemeldeten Browser die Cookies fuer beck-online.beck.de exportieren
    (DevTools oder eine Cookie-Export-Erweiterung), als JSON speichern und in
    den Container legen — ~/.codex/secrets ist dort NICHT gemountet:
        docker cp ~/.codex/secrets/beck-session.json \\
            rechtmaschine-app:/app/downloaded_sources/beck/.session.json
    Das Volume ist kein Repo-Pfad, das Secret landet also nie im Git.

    Erwartet wird entweder ein Playwright-storage_state
    ({"cookies": [...], "origins": [...]}) oder eine blanke Cookie-Liste
    ([{"name": ..., "value": ..., "domain": ..., "path": ...}, ...]).

Nutzung:
    docker exec rechtmaschine-app python /app/beck_fetch.py \\
        --allow-automated-access "https://beck-online.beck.de/Dokument?vpath=..."

Ergebnis ist ein PDF aus der Chromium-Druckausgabe, also MIT Textebene. Der
Ingest laeuft danach wie bei jeder anderen Quelle ueber den Dateipfad:
    docker exec rechtmaschine-app python /app/cited_ingest.py <pfad>.pdf \\
        --az "<Az>" --court "<Gericht>" --date "TT.MM.JJJJ"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

# ACHTUNG: ~/.codex/secrets ist NICHT in den Container gemountet (geprueft
# 07.09.2026). Der Standardpfad liegt deshalb im gemounteten Volume
# downloaded_sources, nicht im Repo — dort darf kein Secret landen.
OUT_DIR = Path("/app/downloaded_sources/beck")
SESSION_FILE = OUT_DIR / ".session.json"
RATE_FILE = OUT_DIR / ".last-fetch"
MIN_INTERVAL_S = 60      # Mindestabstand zwischen zwei Abrufen
MAX_PER_HOUR = 10        # harte Obergrenze — hoehere Frequenz ist das Muster


def load_session(session_file: Path) -> dict:
    """Mitgebrachte Sitzung als Playwright-storage_state.

    Akzeptiert beide ueblichen Exportformen. Cookie-Werte werden nie
    ausgegeben, nur ihre Anzahl und Gueltigkeit.
    """
    if not session_file.is_file():
        raise SystemExit(
            f"Keine Sitzung unter {session_file}.\n"
            "Im angemeldeten Browser die Cookies fuer beck-online.beck.de "
            "exportieren, als JSON speichern und in den Container legen:\n"
            "  docker cp ~/.codex/secrets/beck-session.json "
            f"rechtmaschine-app:{session_file}"
        )
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Sitzungsdatei ist kein gueltiges JSON: {exc}")

    if isinstance(data, list):
        data = {"cookies": data, "origins": []}
    cookies = data.get("cookies") or []
    if not cookies:
        raise SystemExit("Sitzungsdatei enthaelt keine Cookies.")

    now = time.time()
    beck = [c for c in cookies if "beck" in str(c.get("domain", "")).lower()]
    if not beck:
        raise SystemExit("Sitzungsdatei enthaelt keine beck-online-Cookies.")
    expired = [c for c in beck
               if isinstance(c.get("expires"), (int, float))
               and 0 < c["expires"] < now]
    print(f"[beck] Sitzung geladen: {len(beck)} Cookies fuer beck-online"
          + (f", davon {len(expired)} abgelaufen" if expired else ""))
    if expired and len(expired) == len(beck):
        raise SystemExit("Alle beck-Cookies sind abgelaufen — bitte neu "
                         "exportieren.")
    # Playwright verlangt je Cookie entweder url oder domain+path.
    for cookie in cookies:
        cookie.setdefault("path", "/")
    data["cookies"] = cookies
    data.setdefault("origins", [])
    return data


def check_rate_limit() -> None:
    """Mindestabstand und Stundenkontingent."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamps: list[float] = []
    if RATE_FILE.is_file():
        for raw in RATE_FILE.read_text(encoding="utf-8").split():
            try:
                stamps.append(float(raw))
            except ValueError:
                continue
    recent = [t for t in stamps if now - t < 3600]
    if recent and now - max(recent) < MIN_INTERVAL_S:
        wait = int(MIN_INTERVAL_S - (now - max(recent)))
        raise SystemExit(f"Zu schnell: noch {wait}s warten "
                         f"(Mindestabstand {MIN_INTERVAL_S}s).")
    if len(recent) >= MAX_PER_HOUR:
        raise SystemExit(
            f"Stundenkontingent erschoepft ({MAX_PER_HOUR}). Das ist Absicht: "
            "eine hoehere Frequenz ist genau das Muster, an dem "
            "automatisierter Abruf erkannt wird."
        )
    recent.append(now)
    RATE_FILE.write_text(" ".join(f"{t:.0f}" for t in recent), encoding="utf-8")


def safe_name(url: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("/")[-1])[:80] or "beck"
    return f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}.pdf"


async def looks_logged_out(page) -> bool:
    """Sperr- statt Dokumentseite? Dann ist die mitgebrachte Sitzung tot."""
    for selector in ("input[type=password]", "form[action*=login i]"):
        if await page.query_selector(selector):
            return True
    return False


async def fetch(url: str, out_dir: Path, session_file: Path) -> Path:
    from playwright.async_api import async_playwright

    storage_state = load_session(session_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / safe_name(url)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            storage_state=storage_state,
            user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"),
            locale="de-DE",
            viewport={"width": 1440, "height": 900},
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            # Lesezeit statt Maschinentakt.
            await asyncio.sleep(random.uniform(1.5, 3.5))
            if await looks_logged_out(page):
                raise SystemExit(
                    "Die mitgebrachte Sitzung wird nicht akzeptiert (Login- "
                    "statt Dokumentseite). Cookies neu exportieren. Nichts "
                    "gespeichert."
                )
            # Chromium-Druckausgabe: echtes PDF MIT Textebene, damit
            # cited_ingest ohne OCR auskommt.
            await page.pdf(path=str(target), format="A4",
                           margin={"top": "15mm", "bottom": "15mm",
                                   "left": "15mm", "right": "15mm"})
        finally:
            await context.close()
            await browser.close()
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ein einzelnes beck-online-Dokument mit mitgebrachter "
                    "Sitzung abrufen (Testwerkzeug, nicht Teil der Pipeline).")
    parser.add_argument("url", help="Dokument-URL, KEINE Trefferliste")
    parser.add_argument("-o", "--out-dir", default=str(OUT_DIR))
    parser.add_argument("--session", default=str(SESSION_FILE),
                        help="JSON mit der mitgebrachten Sitzung "
                             f"(Standard: {SESSION_FILE})")
    parser.add_argument(
        "--allow-automated-access", action="store_true",
        help="Pflicht. Bestaetigt bei JEDEM Aufruf, dass der automatisierte "
             "Zugriff auf ein lizenzpflichtiges Portal bewusst erfolgt.")
    args = parser.parse_args()

    if not args.allow_automated_access:
        print("Abgebrochen: --allow-automated-access fehlt.\n"
              "beck-online untersagt automatisierten Abruf, sanktioniert wird "
              "der Kanzleizugang. Das Flag macht die Entscheidung bei jedem "
              "Aufruf sichtbar.", file=sys.stderr)
        return 2
    if "beck-online.beck.de" not in args.url:
        print("Abgebrochen: nur beck-online-URLs.", file=sys.stderr)
        return 2
    if re.search(r"(trefferliste|search|suche|results)", args.url, re.I):
        print("Abgebrochen: Trefferlisten sind ausgenommen. Dieses Werkzeug "
              "holt genau EIN Dokument, keine Sammlung.", file=sys.stderr)
        return 2

    check_rate_limit()
    path = asyncio.run(fetch(args.url, Path(args.out_dir), Path(args.session)))
    size = path.stat().st_size
    if size < 5000:
        print(f"WARNUNG: nur {size} Bytes — vermutlich Sperr- statt "
              "Dokumentseite. Vor dem Ingest pruefen.")
    print(f"gespeichert: {path} ({size} Bytes)")
    print("Ingest:  docker exec rechtmaschine-app python /app/cited_ingest.py "
          f"{path} --az '<Az>' --court '<Gericht>' --date 'TT.MM.JJJJ'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

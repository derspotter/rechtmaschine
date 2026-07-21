"""Auto-ingest court decisions cited in generated drafts (post-generation).

Runs alongside the citation verification (Jay, 21.07.2026): right after
``run_citation_checks`` the generation endpoints spawn this script as a
detached subprocess with the draft text. It deterministically parses
decision citations ("BVerwG, Urteil vom 27.04.2010 – 10 C 5.09",
"EuGH, Urteil vom 21.09.2023 – C-151/22"), resolves fulltext URLs where a
deterministic scheme exists (bverwg.de, EUR-Lex/CELEX) and feeds them into
the shared cited_ingest pipeline (source_type="cited", store-wide dedup by
content sha + Aktenzeichen). Citations without a deterministic resolver are
reported as MANUAL for the verify-source escalation chain — never guessed.

The subprocess is fire-and-forget: generation latency is unaffected, and a
crash here can never fail a draft. Output goes to
/tmp/draft_citation_ingest.log inside the app/job-worker container.

Manual runs:
    docker exec rechtmaschine-app python /app/draft_citation_ingest.py \
        --text-file /tmp/entwurf.txt [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DECISION_RE = re.compile(
    r"(?P<court>BVerwG|BVerfG|EuGH|EGMR|BGH|BSG|BAG|"
    r"(?:OVG|VGH|VG|LSG|SG|LG|AG)\s+[A-ZÄÖÜ][\wäöüß.-]*(?:\s+[A-ZÄÖÜ][\wäöüß.-]*)?)"
    r",\s*(?P<kind>Urteil|Beschluss|Gerichtsbescheid)\s+vom\s+"
    r"(?P<date>\d{1,2}\.\d{1,2}\.\d{4})\s*[–—-]\s*"
    r"(?P<az>C-\d+/\d+|\d{1,3}\s+[A-Z]{1,3}\s+\d+[./]\d+(?:\.[A-Z]{1,2})?)"
)


def parse_decision_citations(text: str) -> list[dict]:
    """Deterministic parse of decision citations; deduped by (court, az)."""
    seen: set[tuple[str, str]] = set()
    citations: list[dict] = []
    for match in DECISION_RE.finditer(text):
        court = re.sub(r"\s+", " ", match.group("court")).strip()
        az = re.sub(r"\s+", " ", match.group("az")).strip()
        key = (court.casefold(), az.casefold())
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "court": court,
            "kind": match.group("kind"),
            "date": match.group("date"),
            "az": az,
        })
    return citations


_QWEN_EXTRACT_PROMPT = """/no_think
Extrahiere ALLE zitierten Gerichtsentscheidungen aus dem folgenden Schriftsatz-Auszug.
Gib genau ein JSON-Objekt zurück:
{"citations": [{"gericht": string, "art": "Urteil"|"Beschluss"|"Gerichtsbescheid",
"datum": "TT.MM.JJJJ", "aktenzeichen": string}]}
Regeln: Nur Entscheidungen, die im Text tatsächlich mit Aktenzeichen zitiert werden.
"gericht" ist die Kurzbezeichnung wie im Text (z.B. "BVerwG", "EuGH", "OVG NRW",
"VG Düsseldorf"). "aktenzeichen" OHNE Gerichtsnamen und ohne Zusätze wie "juris",
"Rn." oder Fundstellenbände. Datum immer als TT.MM.JJJJ. Keine Entscheidung erfinden.
Wenn keine zitiert werden: {"citations": []}

TEXT:
"""

_MONTHS = {"januar": "01", "februar": "02", "märz": "03", "april": "04", "mai": "05",
           "juni": "06", "juli": "07", "august": "08", "september": "09",
           "oktober": "10", "november": "11", "dezember": "12"}


def _grounded(citation: dict, text: str) -> bool:
    """Deterministic anchors against the draft text — kills hallucinated
    citations (LLM extraction is never trusted on its own, same pattern as
    wiki_media_ingest's az_ok and the party_agent's ungrounded_facts)."""
    from jurisprudence_ingest import _norm_az

    az = (citation.get("az") or "").strip()
    if not az or _norm_az(az) not in _norm_az(text):
        return False
    date = (citation.get("date") or "").strip()
    if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", date):
        return False
    day, month, year = date.split(".")
    date_variants = [
        date,
        f"{int(day)}.{int(month)}.{year}",
        f"{int(day)}. {next((k for k, v in _MONTHS.items() if v == month), '???').capitalize()} {year}",
    ]
    if not any(variant in text for variant in date_variants):
        return False
    court = (citation.get("court") or "").strip()
    return bool(court) and court.split()[0] in text


async def extract_citations_qwen(text: str) -> list[dict] | None:
    """LLM extraction via the local Qwen service; None = service unavailable.

    Every returned citation is deterministically grounded in the text."""
    import os

    from citation_qwen import call_qwen_json
    from shared import ensure_anonymization_service_ready

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        return None
    try:
        await ensure_anonymization_service_ready()
    except Exception:
        return None

    citations: list[dict] = []
    seen: set[tuple[str, str]] = set()
    # Chunk long drafts the simple way — citations never span chunk borders
    # meaningfully because both anchors (Az + Datum) sit in one sentence.
    step, overlap = 9000, 600
    for start in range(0, max(len(text), 1), step - overlap):
        chunk = text[start:start + step]
        if not chunk.strip():
            continue
        try:
            data = await call_qwen_json(service_url, _QWEN_EXTRACT_PROMPT + chunk,
                                        num_predict=1200)
        except Exception:
            return None
        for raw in (data or {}).get("citations") or []:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("art") or "Urteil").strip()
            citation = {
                "court": re.sub(r"\s+", " ", str(raw.get("gericht") or "")).strip(),
                "kind": kind if kind in ("Urteil", "Beschluss", "Gerichtsbescheid") else "Urteil",
                "date": str(raw.get("datum") or "").strip(),
                "az": re.sub(r"\s+", " ", str(raw.get("aktenzeichen") or "")).strip(),
            }
            key = (citation["court"].casefold(), citation["az"].casefold())
            if key in seen:
                continue
            if not _grounded(citation, text):
                # Nicht blockierend (Jay, 21.07.2026): nur Beobachtungslog —
                # der Az-Kreuzcheck in ingest_one bleibt der harte Gate.
                print(f"  HINWEIS {citation['court']} {citation['az']} — Anker nicht "
                      "deterministisch im Entwurfstext gefunden")
            seen.add(key)
            citations.append(citation)
    return citations


async def collect_citations(text: str) -> list[dict]:
    """Qwen extraction with deterministic grounding, union'd with the regex
    baseline (regex adds recall when Qwen misses, Qwen adds the format
    variants the regex cannot know). Qwen down -> regex only, reported."""
    regex_citations = parse_decision_citations(text)
    qwen_citations = await extract_citations_qwen(text)
    if qwen_citations is None:
        print("Qwen-Extraktion nicht verfügbar — nur Regex-Baseline.")
        return regex_citations
    merged = {(c["court"].casefold(), c["az"].casefold()): c for c in regex_citations}
    for citation in qwen_citations:
        merged.setdefault((citation["court"].casefold(), citation["az"].casefold()), citation)
    return list(merged.values())


def resolve_fulltext_url(citation: dict) -> str | None:
    """Deterministic fulltext URL, or None -> manual escalation."""
    court = citation["court"]
    day, month, year = citation["date"].split(".")
    if court == "BVerwG":
        if int(year) < 2002:
            return None  # bverwg.de fulltexts start ~2002 — older is manual
        kind_code = {"Urteil": "U", "Beschluss": "B", "Gerichtsbescheid": "G"}[citation["kind"]]
        az_compact = citation["az"].replace(" ", "")
        return (f"https://www.bverwg.de/entscheidungen/pdf/"
                f"{day.zfill(2)}{month.zfill(2)}{year[2:]}{kind_code}{az_compact}.0.pdf")
    if court == "EuGH":
        m = re.fullmatch(r"C-(\d+)/(\d+)", citation["az"])
        if not m:
            return None
        number, yy = int(m.group(1)), int(m.group(2))
        year4 = 2000 + yy if yy < 50 else 1900 + yy
        doc_code = "CJ" if citation["kind"] == "Urteil" else "CO"
        return (f"https://eur-lex.europa.eu/legal-content/DE/TXT/PDF/"
                f"?uri=CELEX:6{year4}{doc_code}{number:04d}")
    return None


async def ingest_from_text(text: str, *, dry_run: bool = False) -> int:
    from cited_ingest import find_active_by_az, ingest_one
    from database import SessionLocal
    from rag_vocabulary import load_vocabulary

    citations = await collect_citations(text)
    if not citations:
        print("Keine Entscheidungszitate im Entwurf gefunden.")
        return 0

    db = SessionLocal()
    try:
        vocab = load_vocabulary()
        manual: list[dict] = []
        for citation in citations:
            label = f"{citation['court']}, {citation['kind']} vom {citation['date']} – {citation['az']}"
            if find_active_by_az(db, citation["az"]) is not None:
                print(f"  DUP    {label} (bereits im Store)")
                continue
            url = resolve_fulltext_url(citation)
            if url is None:
                manual.append(citation)
                print(f"  MANUAL {label} — kein deterministischer Volltext-Resolver")
                continue
            status, detail = await ingest_one(
                db, vocab, url, az=citation["az"], court=citation["court"],
                date=citation["date"], dry_run=dry_run,
            )
            print(f"  {status:<10} {label} -> {detail}")
        if manual:
            print(f"{len(manual)} Zitat(e) manuell auflösen (verify-source-Eskalationskette).")
        return 0
    finally:
        db.close()


def spawn_for_text(generated_text: str) -> None:
    """Fire-and-forget hook for the generation endpoints. Never raises."""
    try:
        if not (generated_text or "").strip():
            return
        # Cheap gate only — the real extraction (Qwen + grounding) runs in
        # the subprocess. The loose indicator keeps recall: the strict regex
        # would suppress spawning for formats only Qwen recognizes.
        if not re.search(r"(?:Urteil|Beschluss|Gerichtsbescheid)\s+vom\s+\d", generated_text):
            return
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", dir="/tmp",
            prefix="draft-citations-", delete=False,
        ) as handle:
            handle.write(generated_text)
            text_path = handle.name
        log = open("/tmp/draft_citation_ingest.log", "a", encoding="utf-8")
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--text-file", text_path,
             "--unlink-text-file"],
            stdout=log, stderr=log, start_new_session=True,
        )
        print("[CITED INGEST] Hintergrund-Ingest für Entwurfszitate gestartet")
    except Exception as exc:  # noqa: BLE001 - the draft must never fail on this
        print(f"[CITED INGEST WARN] Hook übersprungen: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text-file", required=True,
                        help="UTF-8 file with the draft text")
    parser.add_argument("--unlink-text-file", action="store_true",
                        help="delete the text file afterwards (subprocess mode)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = Path(args.text_file)
    text = path.read_text(encoding="utf-8")
    try:
        return asyncio.run(ingest_from_text(text, dry_run=args.dry_run))
    finally:
        if args.unlink_text_file:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

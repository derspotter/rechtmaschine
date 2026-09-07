"""Chunk-Reparatur fuer RechtsprechungEntries ohne Chunks in der Collection.

Aendert den Eintrag NICHT und ruft Qwen NICHT erneut auf: Text neu beschaffen,
mit den gespeicherten Metadaten chunken, upserten. Damit kann kein
Doppeleintrag und keine abweichende Verschlagwortung entstehen.
"""
import hashlib
import re
import sys
from pathlib import Path

import httpx

from database import SessionLocal
from jurisprudence_ingest import chunk_text, download_pdf_text, upsert
from models import RechtsprechungEntry
from rag_vocabulary import (facet_metadata, load_vocabulary, normalize_country,
                            normalize_normen, normalize_themen, tag_line)

COLLECTION = "jurisprudence"


# Domains, die die Ingest-Pipeline bewusst nicht anfasst (Paywall/Captcha).
# Die Reparatur haelt sich an dieselbe Politik.
# Nur die lizenzpflichtigen Hosts (AGB/§§ 87a ff. UrhG). Technisch tote
# Portale duerfen hier durchaus versucht werden - anders als bei einem blind
# geladenen Suchtreffer ist die Quelle hier bekannt und der Eintrag existiert
# schon. `voris.` (freies Niedersachsen-System) ist bewusst NICHT dabei.
BLOCKED_HOSTS = ("juris.de", "research.wolterskluwer-online.de",
                 "beck-online.beck.de")


def html_document_text(url: str) -> str:
    """Fliesstext einer HTML-Entscheidungsseite (Quellen ohne PDF-Zwilling)."""
    from lxml import html as lxml_html

    resp = httpx.get(url, timeout=30.0, follow_redirects=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    tree = lxml_html.fromstring(resp.text)
    for bad in tree.xpath("//script | //style | //nav | //header | //footer"):
        bad.getparent().remove(bad)
    main = tree.xpath("//main") or tree.xpath("//*[@id='content']") or [tree]
    text = main[0].text_content()
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def resolve_pdf_url(url: str) -> str:
    """asyl.net-Detailseiten verlinken die Entscheidung als PDF."""
    if url.lower().endswith(".pdf"):
        return url
    # NRWE liefert PDFs ueber einen PHP-Endpunkt OHNE .pdf-Endung
    # (Content-Type application/pdf, verifiziert 07.09.2026).
    if "nrwe.justiz.nrw.de/pdfdownload/" in url:
        return url
    # NRWE-HTML-Seite -> ihr PDF-Zwilling ueber denselben Endpunkt.
    m = re.match(r"https://nrwe\.justiz\.nrw\.de/(.+\.html)$", url)
    if m:
        return ("https://nrwe.justiz.nrw.de/pdfdownload/downloadEntscheidung.php"
                f"?entscheidung=/nrwe/{m.group(1)}")
    resp = httpx.get(url, timeout=30.0, follow_redirects=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    hits = re.findall(r'href="([^"]+\.pdf[^"]*)"', resp.text, re.I)
    if not hits:
        raise RuntimeError("kein PDF-Link auf der Detailseite")
    href = hits[0]
    return href if href.startswith("http") else "https://www.asyl.net" + href


def repair(entry, vocab, dry_run: bool) -> tuple[str, str]:
    label = f"{entry.court} {entry.aktenzeichen}"
    src = (entry.source_url or "").strip()
    if not src:
        return "SKIP", f"{label} — keine source_url"
    try:
        if src.startswith("/") and Path(src).is_file():
            # Aeltere cited-Laeufe haben den lokalen Downloadpfad statt der URL
            # gespeichert - die Datei liegt noch im Container.
            import fitz

            doc = fitz.open(src)
            try:
                text = "\n\n".join((page.get_text() or "").strip() for page in doc)
            finally:
                doc.close()
        elif any(host in src for host in BLOCKED_HOSTS):
            return "SKIP", f"{label} — Quelle auf der Blockliste der Pipeline: {src}"
        else:
            try:
                text = download_pdf_text(resolve_pdf_url(src))
            except Exception:
                text = html_document_text(src)
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"{label} — Textbeschaffung: {exc}"
    if len(text) < 400:
        return "SHORT", f"{label} — nur {len(text)} Zeichen"

    full_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sha16 = (entry.content_sha256 or full_sha)[:16]
    same = "sha=gleich" if entry.content_sha256 == full_sha else "sha=ABWEICHEND"

    tags_list = list(entry.tags or [])
    _themen = normalize_themen(vocab, tags_list)
    _country = normalize_country(vocab, entry.country)
    _normen = normalize_normen(vocab, [])
    header_bits = ["Rechtsprechung", entry.court or "", entry.court_level or "",
                   str(entry.decision_date or ""), entry.country or "",
                   tag_line(_themen, _country, _normen)]
    context_header = " | ".join(b for b in header_bits if b)
    metadata = {
        "source_system": entry.source_type,
        "rechtsprechung_entry_id": str(entry.id),
        "country": entry.country,
        "court": entry.court,
        "court_level": entry.court_level,
        "outcome": entry.outcome,
        "decision_date": str(entry.decision_date or ""),
        "aktenzeichen": entry.aktenzeichen,
        "issue_tags": tags_list,
        **facet_metadata(_themen, _country, _normen),
        "instance_weight": entry.instance_weight,
        "language": "de",
    }
    provenance = [f"{entry.source_type}:{entry.source_url}",
                  f"entry:{entry.id}", f"sha256:{sha16}"]
    payload = [
        {"chunk_id": f"juris-{sha16}-{idx:03d}", "text": chunk,
         "context_header": context_header,
         "metadata": {**metadata, "chunk_index": idx}, "provenance": provenance}
        for idx, chunk in enumerate(chunk_text(text))
    ]
    if dry_run:
        return "DRY", f"{label} — {len(payload)} Chunks ({same})"
    try:
        written = upsert(payload, COLLECTION)
    except Exception as exc:  # noqa: BLE001 - ein Eintrag darf den Lauf nicht kippen
        return "FAIL", f"{label} — Upsert: {type(exc).__name__}: {exc}"
    return "OK", f"{label} — {written} Chunks ({same})"


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    ids = [a for a in sys.argv[1:] if not a.startswith("--")]
    db = SessionLocal()
    counts: dict[str, int] = {}
    try:
        vocab = load_vocabulary()
        for entry_id in ids:
            entry = db.query(RechtsprechungEntry).filter(
                RechtsprechungEntry.id == entry_id).first()
            if entry is None:
                status, detail = "FAIL", f"{entry_id} — Eintrag nicht gefunden"
            else:
                status, detail = repair(entry, vocab, dry_run)
            counts[status] = counts.get(status, 0) + 1
            print(f"  {status:<6} {detail}", flush=True)
    finally:
        db.close()
    print("Ergebnis: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0 if not counts.get("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())

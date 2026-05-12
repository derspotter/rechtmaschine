#!/usr/bin/env python
"""
Export an ingestion manifest from a filter report JSONL.

Safety:
- Never writes to the corpus directory.
- Manifest contains only INCLUDE records by default (REVIEW excluded).

Outputs:
- JSONL manifest with normalized metadata per source file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "data" / "filter_reports"
DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parent / "data" / "manifests"


SIGNAL_CODES = {
    "KLAGE_SIGNAL",
    "AZB_SIGNAL",
    "BZA_SIGNAL",
    "NE_SIGNAL",
    "EB_SIGNAL",
}


@dataclass(frozen=True)
class ManifestItem:
    source_rel_path: str
    source_abs_path: str
    extension: str
    case_year: Optional[str]
    case_folder: Optional[str]
    filename: str
    signal_codes: list[str]
    date_prefix: Optional[str]
    court_token: Optional[str]
    doc_token: Optional[str]
    letterhead_confidence: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_rel_path": self.source_rel_path,
            "source_abs_path": self.source_abs_path,
            "extension": self.extension,
            "case_year": self.case_year,
            "case_folder": self.case_folder,
            "filename": self.filename,
            "signal_codes": self.signal_codes,
            "date_prefix": self.date_prefix,
            "court_token": self.court_token,
            "doc_token": self.doc_token,
            "letterhead_confidence": self.letterhead_confidence,
        }


def _latest_report(reports_dir: Path) -> Path:
    reports = sorted(reports_dir.glob("filter_report_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        raise FileNotFoundError(f"No filter_report_*.jsonl found in {reports_dir}")
    return reports[0]


def _parse_filename_tokens(filename: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Try to parse `YYMMDD_court_doctype...` from filename.
    Returns (date_prefix, court_token, doc_token).
    """
    lower = filename.lower()
    if len(lower) >= 7 and lower[:6].isdigit() and lower[6] == "_":
        date_prefix = lower[:6]
        rest = lower[7:]
        parts = rest.split("_")
        if len(parts) >= 2:
            court_token = parts[0]
            doc_token = parts[1]
            return date_prefix, court_token, doc_token
        return date_prefix, None, None
    return None, None, None


def _extract_case_parts(rel_path: str) -> tuple[Optional[str], Optional[str], str]:
    p = Path(rel_path)
    parts = p.parts
    if len(parts) >= 2:
        return parts[0], parts[1], p.name
    return None, None, p.name


def _signal_codes_from_reasons(reason_codes: list[str]) -> list[str]:
    signals = [code for code in reason_codes if code in SIGNAL_CODES]
    # deterministic ordering
    return sorted(set(signals))


def load_manifest_items(
    report_path: Path,
    corpus_dir: Path,
    include_decision: str = "INCLUDE",
    prefer_pdf: bool = True,
) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("decision") != include_decision:
                continue

            rel_path = str(rec.get("path", ""))
            if not rel_path:
                continue

            case_year, case_folder, filename = _extract_case_parts(rel_path)
            date_prefix, court_token, doc_token = _parse_filename_tokens(filename)
            reason_codes = rec.get("reason_codes") or []
            if not isinstance(reason_codes, list):
                reason_codes = []

            items.append(
                ManifestItem(
                    source_rel_path=rel_path,
                    source_abs_path=str((corpus_dir / rel_path).resolve()),
                    extension=str(rec.get("extension") or ""),
                    case_year=case_year,
                    case_folder=case_folder,
                    filename=filename,
                    signal_codes=_signal_codes_from_reasons([str(x) for x in reason_codes]),
                    date_prefix=date_prefix,
                    court_token=court_token,
                    doc_token=doc_token,
                    letterhead_confidence=rec.get("letterhead_confidence"),
                )
            )

    if not prefer_pdf:
        return items

    # Drop ODT items when a sibling PDF exists (either present in manifest or on disk).
    rel_paths = {item.source_rel_path for item in items}
    filtered: list[ManifestItem] = []
    dropped = 0
    for item in items:
        if item.extension != "odt":
            filtered.append(item)
            continue
        sibling_pdf_rel = str(Path(item.source_rel_path).with_suffix(".pdf"))
        if sibling_pdf_rel in rel_paths or (corpus_dir / sibling_pdf_rel).exists():
            dropped += 1
            continue
        filtered.append(item)

    return filtered


def write_manifest(items: list[ManifestItem], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export INCLUDE-only ingestion manifest from filter report JSONL.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"Filter report JSONL. Default: latest in {DEFAULT_REPORTS_DIR}",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory containing filter reports (default: rag/data/filter_reports).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("/home/jayjag/Kanzlei/kanzlei"),
        help="Corpus directory used to compute absolute paths.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output manifest JSONL path. Default: rag/data/manifests/manifest_<timestamp>.jsonl",
    )
    parser.add_argument(
        "--prefer-pdf",
        action="store_true",
        help="Prefer PDFs over ODTs when both exist (drops ODT duplicates).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report_path = args.report or _latest_report(args.reports_dir)
    corpus_dir: Path = args.corpus_dir

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.out or (DEFAULT_MANIFESTS_DIR / f"manifest_{ts}.jsonl")

    items = load_manifest_items(
        report_path=report_path,
        corpus_dir=corpus_dir,
        prefer_pdf=bool(args.prefer_pdf),
    )
    write_manifest(items, output_path)

    print(f"Report:   {report_path}")
    print(f"Corpus:   {corpus_dir}")
    print(f"Items:    {len(items)}")
    print(f"Manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

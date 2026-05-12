#!/usr/bin/env python
"""
Dry-run corpus filter for RAG ingestion.

Outputs:
- JSONL with per-file decision and reason codes
- CSV with flattened report rows
- JSON summary with decision/reason counts

Decision buckets:
- INCLUDE: high-confidence candidates for ingestion
- EXCLUDE: hard excludes or clear non-target docs
- REVIEW: ambiguous files requiring manual validation
"""

from __future__ import annotations

import argparse
import builtins
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - environment dependent
    fitz = None


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "filter_reports"
CANDIDATE_CORPUS_DIRS = [
    Path("/home/jayjag/Kanzlei/kanzlei"),
    Path("/run/media/jayjag/My Book1/RAG/kanzlei/"),
    Path("/home/jayjag/Nextcloud/Kanzlei/kanzlei"),
]


def resolve_default_corpus_dir() -> Path:
    for candidate in CANDIDATE_CORPUS_DIRS:
        if candidate.exists():
            return candidate
    return CANDIDATE_CORPUS_DIRS[0]


DEFAULT_CORPUS_DIR = resolve_default_corpus_dir()


STRONG_LETTERHEAD_PATTERNS: dict[str, re.Pattern[str]] = {
    "LETTERHEAD_KEIENBORG": re.compile(r"\bkeienborg\b", re.IGNORECASE),
    "LETTERHEAD_ADDRESS": re.compile(
        r"friedrich[-\s]?ebert[-\s]str", re.IGNORECASE
    ),
}

SUPPORTING_LETTERHEAD_PATTERNS: dict[str, re.Pattern[str]] = {
    "LETTERHEAD_ZIP_CITY": re.compile(r"\b40210\s+d[üu]sseldorf\b", re.IGNORECASE),
    "LETTERHEAD_MARCEL": re.compile(r"\bmarcel\s+keienborg\b", re.IGNORECASE),
    "LETTERHEAD_SCHOTTE": re.compile(r"\bchristian\s+schotte\b", re.IGNORECASE),
    "LETTERHEAD_KANZLEI": re.compile(r"\bkanzlei\s+keienborg\b", re.IGNORECASE),
}

HARD_EXCLUDE_PATH_PATTERNS: dict[str, re.Pattern[str]] = {
    "EXCLUDE_PATH_BUCHHALTUNG": re.compile(r"\bbuchhalt", re.IGNORECASE),
    "EXCLUDE_PATH_00AT": re.compile(r"(^|[/\\])00\s*at([/\\]|$)", re.IGNORECASE),
    "EXCLUDE_PATH_RECHNUNG": re.compile(r"\brechn", re.IGNORECASE),
}

HARD_EXCLUDE_FILENAME_PATTERNS: dict[str, re.Pattern[str]] = {
    "EXCLUDE_FILE_VOLLMACHT": re.compile(r"vollmacht", re.IGNORECASE),
    "EXCLUDE_FILE_PKH": re.compile(r"\bpkh\b", re.IGNORECASE),
    "EXCLUDE_FILE_PKA": re.compile(r"\bpka\b", re.IGNORECASE),
    "EXCLUDE_FILE_MITTELLOS": re.compile(r"mittellos", re.IGNORECASE),
}

LIKELY_EXTERNAL_FILENAME_PATTERNS: dict[str, re.Pattern[str]] = {
    "EXTERNAL_FILE_BESCHEID": re.compile(r"bescheid", re.IGNORECASE),
    "EXTERNAL_FILE_URTEIL": re.compile(r"urteil", re.IGNORECASE),
    "EXTERNAL_FILE_BESCHLUSS": re.compile(r"beschluss", re.IGNORECASE),
    "EXTERNAL_FILE_NIEDERSCHRIFT": re.compile(r"niederschrift", re.IGNORECASE),
    "EXTERNAL_FILE_ANHOERUNG": re.compile(r"anh[öo]rung|anhoerung", re.IGNORECASE),
}

LEGAL_FILENAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"klage", re.IGNORECASE),
    re.compile(r"schriftsatz", re.IGNORECASE),
    re.compile(r"beschwerde", re.IGNORECASE),
    re.compile(r"antrag", re.IGNORECASE),
    re.compile(r"begr[üu]ndung|begruendung", re.IGNORECASE),
    re.compile(r"stellungnahme", re.IGNORECASE),
    re.compile(r"\b(vg|ovg|bverwg)\b", re.IGNORECASE),
]

TARGET_DOC_SIGNAL_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "KLAGE_SIGNAL": [
        re.compile(r"\bklage\b", re.IGNORECASE),
        re.compile(r"klageschrift", re.IGNORECASE),
        re.compile(r"_klage(?:_|\.|$)", re.IGNORECASE),
    ],
    "AZB_SIGNAL": [
        re.compile(r"(?:^|[_\-\s])azb(?:[_\-\.\s]|$)", re.IGNORECASE),
    ],
    "BZA_SIGNAL": [
        re.compile(r"(?:^|[_\-\s])bza(?:[_\-\.\s]|$)", re.IGNORECASE),
    ],
    "NE_SIGNAL": [
        re.compile(r"(?:^|[_\-\s])ne(?:[_\-\.\s]|$)", re.IGNORECASE),
    ],
    "EB_SIGNAL": [
        re.compile(r"(?:^|[_\-\s])eb(?:[_\-\.\s]|$)", re.IGNORECASE),
    ],
}

TARGET_DOC_EXCLUDE_PATTERNS: dict[str, re.Pattern[str]] = {
    "TARGET_EXCLUDE_ANLAGE": re.compile(r"(^|[_\-\s])anlage", re.IGNORECASE),
    "TARGET_EXCLUDE_DOCSCAN": re.compile(r"^doc\d+", re.IGNORECASE),
    "TARGET_EXCLUDE_BAMF": re.compile(r"\bbamf\b", re.IGNORECASE),
    "TARGET_EXCLUDE_BESCHEID": re.compile(r"bescheid", re.IGNORECASE),
    "TARGET_EXCLUDE_NIEDERSCHRIFT": re.compile(
        r"niederschrift", re.IGNORECASE
    ),
    "TARGET_EXCLUDE_ATTEST": re.compile(r"attest", re.IGNORECASE),
}

DATE_PREFIX_PATTERN = re.compile(r"^\d{6}_")
CASE_FOLDER_PATTERN = re.compile(r"^\d{3}\b")
YEAR_FOLDER_VALUES = {"23", "24", "25"}


@dataclass
class LetterheadResult:
    confidence: float = 0.0
    strong_matches: list[str] = field(default_factory=list)
    supporting_matches: list[str] = field(default_factory=list)
    text_chars: int = 0
    looks_scanned: bool = False
    error: Optional[str] = None


@dataclass
class FilterRecord:
    path: str
    extension: str
    decision: str
    score: int
    reason_codes: list[str]
    letterhead_confidence: Optional[float] = None
    letterhead_strong_matches: list[str] = field(default_factory=list)
    letterhead_supporting_matches: list[str] = field(default_factory=list)
    text_chars: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "extension": self.extension,
            "decision": self.decision,
            "score": self.score,
            "reason_codes": self.reason_codes,
            "letterhead_confidence": self.letterhead_confidence,
            "letterhead_strong_matches": self.letterhead_strong_matches,
            "letterhead_supporting_matches": self.letterhead_supporting_matches,
            "text_chars": self.text_chars,
            "error": self.error,
        }


def is_directory_writable(path: Path) -> bool:
    return os.access(path, os.W_OK)


def ensure_output_dir_outside_corpus(corpus_dir: Path, output_dir: Path) -> None:
    corpus_resolved = corpus_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == corpus_resolved or corpus_resolved in output_resolved.parents:
        raise ValueError(
            f"Output directory must be outside corpus. corpus={corpus_resolved}, output={output_resolved}"
        )


def snapshot_file_stats(files: list[Path]) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in files:
        stat = path.stat()
        snapshot[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def detect_source_changes(
    files: list[Path], before_snapshot: dict[str, tuple[int, int]]
) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    missing: list[str] = []
    for path in files:
        key = str(path)
        expected = before_snapshot.get(key)
        if expected is None:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            missing.append(key)
            continue
        current = (stat.st_size, stat.st_mtime_ns)
        if current != expected:
            changed.append(key)
    return changed, missing


def _mode_has_write(mode: str) -> bool:
    return any(token in mode for token in ("w", "a", "x", "+"))


def _path_in_corpus(path_like: object, corpus_root: Path) -> bool:
    if not isinstance(path_like, (str, bytes, os.PathLike)):
        return False
    try:
        candidate = Path(path_like).expanduser().resolve(strict=False)
    except Exception:
        return False
    return candidate == corpus_root or corpus_root in candidate.parents


def _deny_write(path_like: object, operation: str, corpus_root: Path) -> None:
    raise PermissionError(
        f"Blocked {operation} on corpus path: {path_like} (guarded root: {corpus_root})"
    )


@contextmanager
def guard_corpus_writes(corpus_dir: Path):
    """
    Runtime safeguard: block write-like operations targeting corpus paths.
    """
    corpus_root = corpus_dir.resolve()

    original_open = builtins.open
    original_path_open = Path.open
    original_write_text = Path.write_text
    original_write_bytes = Path.write_bytes
    original_unlink = Path.unlink
    original_rename = Path.rename
    original_replace = Path.replace
    original_mkdir = Path.mkdir
    original_rmdir = Path.rmdir
    original_chmod = Path.chmod
    original_touch = Path.touch
    original_os_remove = os.remove
    original_os_unlink = os.unlink
    original_os_rename = os.rename
    original_os_replace = os.replace
    original_os_mkdir = os.mkdir
    original_os_rmdir = os.rmdir

    def guarded_open(file, mode="r", *args, **kwargs):
        if _mode_has_write(mode) and _path_in_corpus(file, corpus_root):
            _deny_write(file, f"open(mode={mode!r})", corpus_root)
        return original_open(file, mode, *args, **kwargs)

    def guarded_path_open(self, mode="r", *args, **kwargs):
        if _mode_has_write(mode) and _path_in_corpus(self, corpus_root):
            _deny_write(self, f"Path.open(mode={mode!r})", corpus_root)
        return original_path_open(self, mode, *args, **kwargs)

    def guarded_write_text(self, *args, **kwargs):
        if _path_in_corpus(self, corpus_root):
            _deny_write(self, "Path.write_text", corpus_root)
        return original_write_text(self, *args, **kwargs)

    def guarded_write_bytes(self, *args, **kwargs):
        if _path_in_corpus(self, corpus_root):
            _deny_write(self, "Path.write_bytes", corpus_root)
        return original_write_bytes(self, *args, **kwargs)

    def guarded_unlink(self, *args, **kwargs):
        if _path_in_corpus(self, corpus_root):
            _deny_write(self, "Path.unlink", corpus_root)
        return original_unlink(self, *args, **kwargs)

    def guarded_rename(self, target, *args, **kwargs):
        if _path_in_corpus(self, corpus_root) or _path_in_corpus(target, corpus_root):
            _deny_write(f"{self} -> {target}", "Path.rename", corpus_root)
        return original_rename(self, target, *args, **kwargs)

    def guarded_replace(self, target, *args, **kwargs):
        if _path_in_corpus(self, corpus_root) or _path_in_corpus(target, corpus_root):
            _deny_write(f"{self} -> {target}", "Path.replace", corpus_root)
        return original_replace(self, target, *args, **kwargs)

    def guarded_mkdir(self, *args, **kwargs):
        if _path_in_corpus(self, corpus_root):
            _deny_write(self, "Path.mkdir", corpus_root)
        return original_mkdir(self, *args, **kwargs)

    def guarded_rmdir(self, *args, **kwargs):
        if _path_in_corpus(self, corpus_root):
            _deny_write(self, "Path.rmdir", corpus_root)
        return original_rmdir(self, *args, **kwargs)

    def guarded_chmod(self, *args, **kwargs):
        if _path_in_corpus(self, corpus_root):
            _deny_write(self, "Path.chmod", corpus_root)
        return original_chmod(self, *args, **kwargs)

    def guarded_touch(self, *args, **kwargs):
        if _path_in_corpus(self, corpus_root):
            _deny_write(self, "Path.touch", corpus_root)
        return original_touch(self, *args, **kwargs)

    def guarded_os_remove(path, *args, **kwargs):
        if _path_in_corpus(path, corpus_root):
            _deny_write(path, "os.remove", corpus_root)
        return original_os_remove(path, *args, **kwargs)

    def guarded_os_unlink(path, *args, **kwargs):
        if _path_in_corpus(path, corpus_root):
            _deny_write(path, "os.unlink", corpus_root)
        return original_os_unlink(path, *args, **kwargs)

    def guarded_os_rename(src, dst, *args, **kwargs):
        if _path_in_corpus(src, corpus_root) or _path_in_corpus(dst, corpus_root):
            _deny_write(f"{src} -> {dst}", "os.rename", corpus_root)
        return original_os_rename(src, dst, *args, **kwargs)

    def guarded_os_replace(src, dst, *args, **kwargs):
        if _path_in_corpus(src, corpus_root) or _path_in_corpus(dst, corpus_root):
            _deny_write(f"{src} -> {dst}", "os.replace", corpus_root)
        return original_os_replace(src, dst, *args, **kwargs)

    def guarded_os_mkdir(path, *args, **kwargs):
        if _path_in_corpus(path, corpus_root):
            _deny_write(path, "os.mkdir", corpus_root)
        return original_os_mkdir(path, *args, **kwargs)

    def guarded_os_rmdir(path, *args, **kwargs):
        if _path_in_corpus(path, corpus_root):
            _deny_write(path, "os.rmdir", corpus_root)
        return original_os_rmdir(path, *args, **kwargs)

    builtins.open = guarded_open
    Path.open = guarded_path_open
    Path.write_text = guarded_write_text
    Path.write_bytes = guarded_write_bytes
    Path.unlink = guarded_unlink
    Path.rename = guarded_rename
    Path.replace = guarded_replace
    Path.mkdir = guarded_mkdir
    Path.rmdir = guarded_rmdir
    Path.chmod = guarded_chmod
    Path.touch = guarded_touch
    os.remove = guarded_os_remove
    os.unlink = guarded_os_unlink
    os.rename = guarded_os_rename
    os.replace = guarded_os_replace
    os.mkdir = guarded_os_mkdir
    os.rmdir = guarded_os_rmdir

    try:
        yield
    finally:
        builtins.open = original_open
        Path.open = original_path_open
        Path.write_text = original_write_text
        Path.write_bytes = original_write_bytes
        Path.unlink = original_unlink
        Path.rename = original_rename
        Path.replace = original_replace
        Path.mkdir = original_mkdir
        Path.rmdir = original_rmdir
        Path.chmod = original_chmod
        Path.touch = original_touch
        os.remove = original_os_remove
        os.unlink = original_os_unlink
        os.rename = original_os_rename
        os.replace = original_os_replace
        os.mkdir = original_os_mkdir
        os.rmdir = original_os_rmdir


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_letterhead(pdf_path: Path, check_pages: int = 1) -> LetterheadResult:
    if fitz is None:
        return LetterheadResult(error="PYMUPDF_NOT_INSTALLED")

    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return LetterheadResult(error="EMPTY_PDF")

        text_parts: list[str] = []
        for page_idx in range(min(check_pages, len(doc))):
            text_parts.append(doc[page_idx].get_text() or "")
        doc.close()

        raw_text = "\n".join(text_parts)
        text = normalize_text(raw_text)
        text_chars = len(text)

        strong_matches = [
            code
            for code, pattern in STRONG_LETTERHEAD_PATTERNS.items()
            if pattern.search(text)
        ]
        supporting_matches = [
            code
            for code, pattern in SUPPORTING_LETTERHEAD_PATTERNS.items()
            if pattern.search(text)
        ]

        # Conservative confidence: bias toward precision.
        confidence = 0.0
        if "LETTERHEAD_KEIENBORG" in strong_matches:
            confidence = 0.98
        elif "LETTERHEAD_ADDRESS" in strong_matches and len(supporting_matches) >= 1:
            confidence = 0.85
        elif len(supporting_matches) >= 3:
            confidence = 0.72
        elif len(supporting_matches) == 2:
            confidence = 0.55
        elif len(supporting_matches) == 1:
            confidence = 0.35

        looks_scanned = text_chars < 100
        return LetterheadResult(
            confidence=confidence,
            strong_matches=strong_matches,
            supporting_matches=supporting_matches,
            text_chars=text_chars,
            looks_scanned=looks_scanned,
        )
    except Exception as exc:
        return LetterheadResult(error=f"PDF_PARSE_ERROR:{exc}")


def has_date_prefix(filename: str) -> bool:
    return bool(DATE_PREFIX_PATTERN.search(filename))


def looks_legal_filename(filename: str) -> bool:
    return any(pattern.search(filename) for pattern in LEGAL_FILENAME_PATTERNS)


def get_target_doc_signals(filename: str) -> list[str]:
    signals: list[str] = []
    for signal_code, patterns in TARGET_DOC_SIGNAL_PATTERNS.items():
        if any(pattern.search(filename) for pattern in patterns):
            signals.append(signal_code)
    return signals


def match_reason_codes(
    text: str, patterns: dict[str, re.Pattern[str]]
) -> list[str]:
    return [code for code, pattern in patterns.items() if pattern.search(text)]


def is_in_case_structure(relative_path: Path) -> bool:
    parts = relative_path.parts
    if len(parts) < 2:
        return False
    year_part = parts[0]
    case_part = parts[1]
    return year_part in YEAR_FOLDER_VALUES and bool(CASE_FOLDER_PATTERN.match(case_part))


def decide_file(
    corpus_dir: Path,
    file_path: Path,
    check_pages: int,
) -> FilterRecord:
    rel_path = file_path.relative_to(corpus_dir)
    rel_str = str(rel_path)
    rel_lower = rel_str.lower()
    filename_lower = file_path.name.lower()
    extension = file_path.suffix.lower().lstrip(".")

    reasons: list[str] = []
    score = 0

    path_excludes = match_reason_codes(rel_lower, HARD_EXCLUDE_PATH_PATTERNS)
    file_excludes = match_reason_codes(filename_lower, HARD_EXCLUDE_FILENAME_PATTERNS)
    if path_excludes or file_excludes:
        reasons.extend(path_excludes + file_excludes)
        return FilterRecord(
            path=rel_str,
            extension=extension,
            decision="EXCLUDE",
            score=0,
            reason_codes=reasons,
        )

    in_case = is_in_case_structure(rel_path)
    if in_case:
        score += 30
        reasons.append("IN_CASE_FOLDER")
    else:
        reasons.append("OUTSIDE_CASE_FOLDER")
        reasons.append("TARGET_OUTSIDE_CASE")
        return FilterRecord(
            path=rel_str,
            extension=extension,
            decision="EXCLUDE",
            score=max(score, 0),
            reason_codes=reasons,
        )

    if has_date_prefix(filename_lower):
        score += 20
        reasons.append("DATE_PREFIX_FILENAME")

    if looks_legal_filename(filename_lower):
        score += 20
        reasons.append("LEGAL_FILENAME_SIGNAL")

    target_signals = get_target_doc_signals(filename_lower)
    if target_signals:
        score += 25
        reasons.extend(target_signals)
    else:
        reasons.append("TARGET_MISSING_SIGNAL")
        return FilterRecord(
            path=rel_str,
            extension=extension,
            decision="EXCLUDE",
            score=max(score, 0),
            reason_codes=reasons,
        )

    target_excludes = match_reason_codes(filename_lower, TARGET_DOC_EXCLUDE_PATTERNS)
    if target_excludes:
        reasons.extend(target_excludes)
        return FilterRecord(
            path=rel_str,
            extension=extension,
            decision="EXCLUDE",
            score=max(score, 0),
            reason_codes=reasons,
        )

    external_file_signals = match_reason_codes(
        filename_lower, LIKELY_EXTERNAL_FILENAME_PATTERNS
    )
    if external_file_signals:
        reasons.extend(external_file_signals)
        score -= 15

    if extension == "pdf":
        lh = detect_letterhead(file_path, check_pages=check_pages)
        if lh.error:
            reasons.append("PDF_READ_ERROR")
            return FilterRecord(
                path=rel_str,
                extension=extension,
                decision="REVIEW",
                score=max(score, 0),
                reason_codes=reasons,
                error=lh.error,
            )

        if lh.confidence >= 0.70:
            reasons.append("LETTERHEAD_CONFIDENT")
            score += 50
        elif lh.confidence >= 0.40:
            reasons.append("LETTERHEAD_WEAK")
            score += 20
        else:
            reasons.append("NO_LETTERHEAD_SIGNAL")

        if lh.looks_scanned and lh.confidence < 0.70:
            reasons.append("NEEDS_OCR_HEADER_CHECK")
            return FilterRecord(
                path=rel_str,
                extension=extension,
                decision="REVIEW",
                score=max(score, 0),
                reason_codes=reasons,
                letterhead_confidence=lh.confidence,
                letterhead_strong_matches=lh.strong_matches,
                letterhead_supporting_matches=lh.supporting_matches,
                text_chars=lh.text_chars,
            )

        if lh.confidence >= 0.70 and score >= 70:
            decision = "INCLUDE"
        elif lh.confidence < 0.40 and external_file_signals:
            decision = "EXCLUDE"
            reasons.append("LIKELY_EXTERNAL")
        elif score <= 25:
            decision = "EXCLUDE"
            reasons.append("LOW_SCORE")
        else:
            decision = "EXCLUDE"
            reasons.append("TARGET_STRICT_EXCLUDE")

        return FilterRecord(
            path=rel_str,
            extension=extension,
            decision=decision,
            score=max(score, 0),
            reason_codes=reasons,
            letterhead_confidence=lh.confidence,
            letterhead_strong_matches=lh.strong_matches,
            letterhead_supporting_matches=lh.supporting_matches,
            text_chars=lh.text_chars,
        )

    # ODT/DOCX decision (no text-level letterhead check here)
    if extension in {"odt", "docx"}:
        if in_case and not external_file_signals and score >= 50:
            return FilterRecord(
                path=rel_str,
                extension=extension,
                decision="INCLUDE",
                score=score,
                reason_codes=reasons,
            )
        if not in_case and external_file_signals:
            reasons.append("LIKELY_EXTERNAL")
            return FilterRecord(
                path=rel_str,
                extension=extension,
                decision="EXCLUDE",
                score=max(score, 0),
                reason_codes=reasons,
            )
        reasons.append("TARGET_STRICT_EXCLUDE")
        return FilterRecord(
            path=rel_str,
            extension=extension,
            decision="EXCLUDE",
            score=max(score, 0),
            reason_codes=reasons,
        )

    return FilterRecord(
        path=rel_str,
        extension=extension,
        decision="EXCLUDE",
        score=0,
        reason_codes=["UNSUPPORTED_EXTENSION"],
    )


def gather_files(
    corpus_dir: Path,
    extensions: set[str],
    max_files: Optional[int],
    sample_size: Optional[int],
    seed: int,
) -> list[Path]:
    files = [
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in extensions
    ]
    files.sort()

    if max_files is not None and max_files > 0:
        files = files[:max_files]

    if sample_size is not None and sample_size > 0 and sample_size < len(files):
        rng = random.Random(seed)
        files = sorted(rng.sample(files, sample_size))

    return files


def write_reports(
    records: list[FilterRecord], output_dir: Path, run_id: str
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"filter_report_{run_id}.jsonl"
    csv_path = output_dir / f"filter_report_{run_id}.csv"
    summary_path = output_dir / f"filter_summary_{run_id}.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "extension",
                "decision",
                "score",
                "reason_codes",
                "letterhead_confidence",
                "letterhead_strong_matches",
                "letterhead_supporting_matches",
                "text_chars",
                "error",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": record.path,
                    "extension": record.extension,
                    "decision": record.decision,
                    "score": record.score,
                    "reason_codes": ";".join(record.reason_codes),
                    "letterhead_confidence": (
                        "" if record.letterhead_confidence is None else record.letterhead_confidence
                    ),
                    "letterhead_strong_matches": ";".join(
                        record.letterhead_strong_matches
                    ),
                    "letterhead_supporting_matches": ";".join(
                        record.letterhead_supporting_matches
                    ),
                    "text_chars": "" if record.text_chars is None else record.text_chars,
                    "error": record.error or "",
                }
            )

    decisions = Counter(record.decision for record in records)
    by_extension: dict[str, Counter] = defaultdict(Counter)
    reason_counts: Counter = Counter()
    for record in records:
        by_extension[record.extension][record.decision] += 1
        reason_counts.update(record.reason_codes)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(records),
        "decision_counts": dict(decisions),
        "decision_by_extension": {
            ext: dict(counter) for ext, counter in sorted(by_extension.items())
        },
        "top_reason_codes": reason_counts.most_common(30),
        "outputs": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "summary": str(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return jsonl_path, csv_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run filter for RAG corpus with INCLUDE/EXCLUDE/REVIEW output."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help=f"Corpus directory (default: {DEFAULT_CORPUS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default="pdf,odt,docx",
        help="Comma-separated extensions to scan (default: pdf,odt,docx)",
    )
    parser.add_argument(
        "--check-pages",
        type=int,
        default=1,
        help="Number of first pages used for letterhead detection in PDFs (default: 1)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional hard cap on files processed (for quick dry-runs).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional random sample size across scanned files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --sample-size (default: 42).",
    )
    parser.add_argument(
        "--print-review",
        type=int,
        default=20,
        help="How many REVIEW samples to print (default: 20).",
    )
    parser.add_argument(
        "--allow-writable-corpus",
        action="store_true",
        help="Allow running even if corpus directory is writable (unsafe; default is strict read-only).",
    )
    parser.add_argument(
        "--skip-unchanged-check",
        action="store_true",
        help="Skip before/after source snapshot verification.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    corpus_dir: Path = args.corpus_dir
    if not corpus_dir.exists():
        print(f"[ERROR] Corpus directory not found: {corpus_dir}")
        return 2
    if not corpus_dir.is_dir():
        print(f"[ERROR] Corpus path is not a directory: {corpus_dir}")
        return 2

    try:
        ensure_output_dir_outside_corpus(corpus_dir, args.output_dir)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    if is_directory_writable(corpus_dir) and not args.allow_writable_corpus:
        print(
            "[ERROR] Corpus directory is writable. Refusing to run in strict mode.\n"
            "        Mount/set corpus read-only, or use --allow-writable-corpus to override."
        )
        return 2

    extensions = {
        ext.strip().lower().lstrip(".")
        for ext in args.extensions.split(",")
        if ext.strip()
    }
    if not extensions:
        print("[ERROR] No extensions configured.")
        return 2

    with guard_corpus_writes(corpus_dir):
        files = gather_files(
            corpus_dir=corpus_dir,
            extensions=extensions,
            max_files=args.max_files,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        if not files:
            print("[WARN] No matching files found.")
            return 0

        before_snapshot: dict[str, tuple[int, int]] = {}
        if not args.skip_unchanged_check:
            before_snapshot = snapshot_file_stats(files)

        print("=" * 80)
        print("RAG FILTER DRY-RUN")
        print("=" * 80)
        print(f"Corpus: {corpus_dir}")
        print("Profile: litigation (klage + azb + bza + ne + eb)")
        print(f"Files matched: {len(files)}")
        print(f"Extensions: {', '.join(sorted(extensions))}")
        if "pdf" in extensions and fitz is None:
            print(
                "[WARN] PyMuPDF (fitz) not installed. PDF letterhead checks will be REVIEW."
            )
        print()

        records: list[FilterRecord] = []
        for index, file_path in enumerate(files, 1):
            if index % 50 == 0:
                pct = index / len(files) * 100
                print(f"Progress: {index}/{len(files)} ({pct:.1f}%)", end="\r")
            records.append(
                decide_file(
                    corpus_dir=corpus_dir,
                    file_path=file_path,
                    check_pages=args.check_pages,
                )
            )
        if len(files) >= 50:
            print()

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        jsonl_path, csv_path, summary_path = write_reports(records, args.output_dir, run_id)

        if not args.skip_unchanged_check:
            changed, missing = detect_source_changes(files, before_snapshot)
            if changed or missing:
                print()
                print("[WARN] Source files changed during run.")
                if changed:
                    print(f"  Changed: {len(changed)}")
                if missing:
                    print(f"  Missing: {len(missing)}")
                print(
                    "  This tool never writes to corpus files; changes likely came from external processes."
                )

    decision_counts = Counter(record.decision for record in records)
    print()
    print("Decision counts:")
    print(f"  INCLUDE: {decision_counts.get('INCLUDE', 0)}")
    print(f"  EXCLUDE: {decision_counts.get('EXCLUDE', 0)}")
    print(f"  REVIEW:  {decision_counts.get('REVIEW', 0)}")
    print()
    print("Reports:")
    print(f"  JSONL:   {jsonl_path}")
    print(f"  CSV:     {csv_path}")
    print(f"  SUMMARY: {summary_path}")

    review_limit = max(0, args.print_review)
    if review_limit:
        review_records = [record for record in records if record.decision == "REVIEW"]
        if review_records:
            print()
            print(f"Sample REVIEW files (up to {review_limit}):")
            for record in review_records[:review_limit]:
                print(
                    f"  - {record.path} | score={record.score} | reasons={','.join(record.reason_codes)}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

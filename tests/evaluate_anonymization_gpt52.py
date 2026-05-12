#!/usr/bin/env python3
"""
Evaluate anonymized text with GPT-5.2 to detect missed plaintiff/family names.

Usage:
  python tests/evaluate_anonymization_gpt52.py [paths...]

Defaults to a small set of OCR text files in test_files/ocred/.

Notes:
- Requires OPENAI_API_KEY in the environment (or --api-key).
- Uses the Responses API (https://api.openai.com/v1/responses).
- Evaluates anonymized output from anon/anonymization_service_flair.py by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [
    REPO_ROOT / "test_files" / "ocred" / "ocr_Anhoerung.md",
    REPO_ROOT / "test_files" / "ocred" / "ocr_2025-02-10_Bescheid.md",
]

PROMPT = """
Du bist Qualitaetspruefer fuer Anonymisierung in deutschen Asylverfahren.
Du bekommst anonymisierten Text. Finde verbliebene echte PERSONENNAMEN
(Vorname, Nachname oder beides), die sich sehr wahrscheinlich auf den
Klaeger/Antragsteller oder seine Familienangehoerigen beziehen.

WICHTIG:
- Ignoriere Platzhalter in eckigen Klammern wie [KLAEGER], [KLAEGERIN],
  [FAMILIENANGEHOERIGER], [KIND], [PERSON_1], [ADRESSE], [GEBURTSDATUM].
- Ignoriere Behoerden, Gerichte, Anwaelte, Richter, Dolmetscher, Sachbearbeiter.
- Nutze Kontextwoerter: Klaeger, Antragsteller, geb., geboren, Sohn/Tochter,
  Ehefrau/Ehemann, Familie, wohnhaft, Adresse.

Gib NUR JSON zurueck (kein Markdown):
{
  "findings": [
    {
      "name": "...",
      "role_guess": "plaintiff|family|unknown",
      "evidence": "kurzer Textausschnitt",
      "confidence": 0.0
    }
  ]
}
""".strip()


@dataclass
class Finding:
    name: str
    role_guess: str
    evidence: str
    confidence: float
    chunk_index: int
    raw_response_text: str


@dataclass
class ChunkResult:
    chunk_index: int
    chunk_chars: int
    findings: List[Finding]
    error: Optional[str] = None


@dataclass
class FileReport:
    path: str
    anonymized_chars: int
    model: str
    chunk_chars: int
    total_chunks: int
    results: List[ChunkResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate anonymized text with GPT-5.2 to detect missed names."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(p) for p in DEFAULT_PATHS],
        help="Paths to OCR text files or PDFs (default: built-in sample list).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Model name for OpenAI Responses API (default: gpt-5.2).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=3000,
        help="Max characters per chunk (default: 3000).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Max number of chunks per file (0 = no limit).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout seconds per request (default: 120).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count on HTTP errors (default: 2).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env).",
    )
    parser.add_argument(
        "--input-is-anonymized",
        action="store_true",
        help="Skip anonymization and treat input as already anonymized.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "tests" / "anonymization_eval_results"),
        help="Directory to write JSON reports.",
    )
    return parser.parse_args()


def load_text_from_path(path: Path, max_pages: int = 0) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Failed to import pymupdf (fitz): {exc}") from exc

        with fitz.open(path) as doc:
            total_pages = len(doc)
            pages_to_read = total_pages if max_pages == 0 else min(total_pages, max_pages)

            text_parts = []
            for i in range(pages_to_read):
                page = doc[i]
                text = page.get_text()
                if text:
                    text_parts.append(f"--- Page {i + 1} ---\n{text}")

        return "\n\n".join(text_parts)

    return path.read_text(encoding="utf-8")


def chunk_text(text: str, max_chars: int, max_chunks: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        if max_chunks and len(chunks) >= max_chunks:
            break
        end = min(len(text), start + max_chars)
        split = text.rfind("\n\n", start, end)
        if split == -1 or split <= start + max_chars * 0.5:
            split = end
        else:
            split = split + 2
        chunks.append(text[start:split])
        start = split
    return chunks


def extract_output_text(response_json: Dict[str, Any]) -> str:
    if "output_text" in response_json and isinstance(response_json["output_text"], str):
        return response_json["output_text"].strip()

    texts: List[str] = []
    for item in response_json.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type in ("output_text", "text"):
                    text = part.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            elif isinstance(part, str):
                texts.append(part)
    return "\n".join(texts).strip()


def call_responses_api(
    api_key: str,
    model: str,
    prompt: str,
    timeout: float,
    retries: int,
) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ],
            }
        ],
        "temperature": 0.0,
        "max_output_tokens": 800,
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < retries:
            time.sleep(2 ** attempt)

    raise RuntimeError(last_error or "Unknown error calling Responses API")


def anonymize_text(text: str) -> str:
    sys.path.append(str(REPO_ROOT / "anon"))
    try:
        from anonymization_service_flair import anonymize_with_flair
    except Exception as exc:
        raise RuntimeError(f"Failed to import anonymization_service_flair: {exc}") from exc

    anonymized_text, _, _, _, _ = anonymize_with_flair(text)
    return anonymized_text


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set (or use --api-key)")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[FileReport] = []

    for raw_path in args.paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            continue

        print(f"\n=== {path} ===")
        text = load_text_from_path(path)
        if not text.strip():
            print("[WARN] Empty text after loading; skipping.")
            continue

        if args.input_is_anonymized:
            anon_text = text
        else:
            print("[INFO] Running anonymize_with_flair...")
            anon_text = anonymize_text(text)

        chunks = chunk_text(anon_text, args.chunk_chars, args.max_chunks)
        results: List[ChunkResult] = []

        for idx, chunk in enumerate(chunks):
            print(f"[INFO] Evaluating chunk {idx + 1}/{len(chunks)} ({len(chunk)} chars)")
            chunk_prompt = PROMPT + "\n\n--- TEXT ---\n" + chunk
            try:
                response_json = call_responses_api(
                    api_key=api_key,
                    model=args.model,
                    prompt=chunk_prompt,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                response_text = extract_output_text(response_json)
                parsed = json.loads(response_text) if response_text else {"findings": []}
                findings: List[Finding] = []
                for item in parsed.get("findings", []) or []:
                    findings.append(
                        Finding(
                            name=str(item.get("name", "")),
                            role_guess=str(item.get("role_guess", "unknown")),
                            evidence=str(item.get("evidence", "")),
                            confidence=float(item.get("confidence", 0.0)),
                            chunk_index=idx,
                            raw_response_text=response_text,
                        )
                    )
                results.append(
                    ChunkResult(
                        chunk_index=idx,
                        chunk_chars=len(chunk),
                        findings=findings,
                        error=None,
                    )
                )
            except Exception as exc:
                results.append(
                    ChunkResult(
                        chunk_index=idx,
                        chunk_chars=len(chunk),
                        findings=[],
                        error=str(exc),
                    )
                )

        report = FileReport(
            path=str(path),
            anonymized_chars=len(anon_text),
            model=args.model,
            chunk_chars=args.chunk_chars,
            total_chunks=len(chunks),
            results=results,
        )
        reports.append(report)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"eval_{path.stem}_{timestamp}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote report: {report_path}")

    summary_path = out_dir / "latest_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in reports], f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Wrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

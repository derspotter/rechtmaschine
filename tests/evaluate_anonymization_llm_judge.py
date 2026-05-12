#!/usr/bin/env python3
"""
LLM-based leak check for anonymized German legal documents.

This complements the deterministic heuristic leak check in
tests/test_anonymization_flair.py. The judge is guided by candidate PII
extracted from the original text, but it is also allowed to report additional
residual leaks it finds in the anonymized text.

Usage:
  python tests/evaluate_anonymization_llm_judge.py [paths...]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [
    REPO_ROOT / "test_files" / "ocred" / "ocr_Anhoerung.md",
    REPO_ROOT / "test_files" / "ocred" / "ocr_2025-02-10_Bescheid.md",
]

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_SERVICE_URL = "http://127.0.0.1:8004"

JUDGE_PROMPT = """
Du bist Qualitaetspruefer fuer Anonymisierung in deutschen Asyl- und
verwaltungsrechtlichen Dokumenten.

Du bekommst:
1. Eine Liste moeglicher privater Daten aus dem Originaltext.
2. Einen Ausschnitt des anonymisierten Textes.

Deine Aufgabe:
- Finde private Daten, die im anonymisierten Text NOCH SICHTBAR sind.
- Bevorzuge echte Leaks von:
  - PERSONENNAMEN
  - GEBURTSDATEN
  - ADRESSEN / ORTE mit Privatbezug
  - AKTEN-/AZR-/ID-Nummern mit Personenbezug
- Nutze die Kandidatenlisten als Hilfe, aber du darfst auch weitere Leaks melden.

WICHTIG:
- Melde NUR Daten, die im anonymisierten Text tatsaechlich noch sichtbar sind.
- Zitiere bei jedem Fund eine kurze exakte Evidenz aus dem anonymisierten Text.
- Ignoriere Platzhalter wie [KLAEGER], [PERSON_1], [ADRESSE], [ORT], [GEBURTSDATUM].
- Ignoriere Behoerden, Gerichte, Gesetzesstellen, Aktenzeichen ohne Personenbezug,
  allgemeine Laender-/Staedtenamen ohne Privatbezug und rein oeffentliche
  Behordenanschriften, sofern sie offensichtlich keine private Person identifizieren.
- Wenn ein Kandidat nur teilweise/leicht OCR-beschaedigt sichtbar ist, melde ihn nur,
  wenn er fuer einen Menschen noch klar als dieselbe private Information erkennbar ist.

Gib NUR JSON zurueck:
{
  "findings": [
    {
      "category": "name|birth_date|address|id|other",
      "value": "sichtbarer Leak",
      "evidence": "kurzer exakter Ausschnitt",
      "reason": "warum das noch ein Leak ist",
      "confidence": 0.0
    }
  ],
  "summary": {
    "has_leaks": true,
    "severity": "none|low|medium|high"
  }
}
""".strip()


@dataclass
class Finding:
    category: str
    value: str
    evidence: str
    reason: str
    confidence: float
    chunk_index: int


@dataclass
class ChunkResult:
    chunk_index: int
    chunk_chars: int
    findings: List[Finding]
    error: Optional[str] = None
    raw_response_text: str = ""


@dataclass
class FileReport:
    path: str
    document_type: str
    anonymizer: str
    model: str
    original_chars: int
    anonymized_chars: int
    chunk_chars: int
    total_chunks: int
    candidate_counts: Dict[str, int]
    results: List[ChunkResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM judge for residual anonymization leaks."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(p) for p in DEFAULT_PATHS],
        help="Paths to OCR text files or PDFs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model for judging (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=8000,
        help="Max characters per judge chunk (default: 8000, 0 = no chunking).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Max chunks per file (0 = no limit).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds per judge request.",
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
        help="Treat input files as already anonymized text/PDF-extracted text.",
    )
    parser.add_argument(
        "--anonymizer",
        choices=["live-manager", "local-flair", "hybrid", "none"],
        default="live-manager",
        help="How to produce anonymized text when input is not already anonymized.",
    )
    parser.add_argument(
        "--service-url",
        default=DEFAULT_SERVICE_URL,
        help=f"Service manager URL (default: {DEFAULT_SERVICE_URL}).",
    )
    parser.add_argument(
        "--document-type",
        default=None,
        help="Override document type for all files.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "tests" / "anonymization_eval_results"),
        help="Directory to write JSON reports.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent judge requests per document (default: 4).",
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
            parts = []
            for i in range(pages_to_read):
                text = doc[i].get_text()
                if text:
                    parts.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(parts)

    return path.read_text(encoding="utf-8")


def chunk_text(text: str, max_chars: int, max_chunks: int) -> List[str]:
    if max_chars <= 0:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        if max_chunks and len(chunks) >= max_chunks:
            break
        end = min(len(text), start + max_chars)
        split = text.rfind("\n\n", start, end)
        if split == -1 or split <= start + int(max_chars * 0.5):
            split = end
        else:
            split += 2
        chunks.append(text[start:split])
        start = split
    return chunks


def extract_output_text(response_json: Dict[str, Any]) -> str:
    if "output_text" in response_json and isinstance(response_json["output_text"], str):
        return response_json["output_text"].strip()

    texts: List[str] = []
    for item in response_json.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
            elif isinstance(part, str):
                texts.append(part)
    return "\n".join(texts).strip()


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"findings": [], "summary": {"has_leaks": False, "severity": "none"}}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


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
    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "max_output_tokens": 1200,
    }

    if model.startswith("gpt-5"):
        payload["reasoning"] = {"effort": "low"}
    else:
        payload["temperature"] = 0.0

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

    raise RuntimeError(last_error or "Unknown Responses API error")


def detect_document_type(path: Path, text: str, override: Optional[str]) -> str:
    if override:
        return override
    lowered_name = path.name.casefold()
    if "anhörung" in lowered_name or "anhoerung" in lowered_name:
        return "Anhörung"
    if "bescheid" in lowered_name:
        return "Bescheid"
    lowered_text = text[:5000].casefold()
    if "anhörung" in lowered_text or "anhoerung" in lowered_text:
        return "Anhörung"
    if "bescheid" in lowered_text:
        return "Bescheid"
    return "Sonstige gespeicherte Quellen"


def load_expected_helpers():
    spec = importlib.util.spec_from_file_location(
        "test_anonymization_flair",
        str(REPO_ROOT / "tests" / "test_anonymization_flair.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def anonymize_text_local_flair(text: str) -> str:
    sys.path.append(str(REPO_ROOT / "anon"))
    try:
        from anonymization_service_flair import anonymize_with_flair
    except Exception as exc:
        raise RuntimeError(f"Failed to import anonymization_service_flair: {exc}") from exc
    anonymized_text, _, _, _, _ = anonymize_with_flair(text)
    return anonymized_text


def anonymize_text_live_manager(text: str, document_type: str, service_url: str) -> str:
    with httpx.Client(timeout=600.0) as client:
        response = client.post(
            f"{service_url.rstrip('/')}/anonymize",
            json={"text": text, "document_type": document_type},
        )
        response.raise_for_status()
        payload = response.json()
    anonymized_text = payload.get("anonymized_text")
    if not isinstance(anonymized_text, str) or not anonymized_text:
        raise RuntimeError("Live manager returned no anonymized_text")
    return anonymized_text


async def anonymize_text_hybrid(text: str, document_type: str, service_url: str) -> str:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "app"))
    os.environ.setdefault("ANONYMIZATION_SERVICE_URL", service_url)
    from app.endpoints.anonymization import anonymize_document_text

    result = await anonymize_document_text(text, document_type, "flair_presidio")
    if result is None or not result.anonymized_text:
        raise RuntimeError("Hybrid anonymizer returned no anonymized_text")
    return result.anonymized_text


def build_prompt(candidates: Dict[str, List[str]], chunk: str) -> str:
    candidate_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    return (
        f"{JUDGE_PROMPT}\n\n"
        f"--- KANDIDATEN AUS ORIGINALTEXT ---\n{candidate_json}\n\n"
        f"--- ANONYMISIERTER TEXT ---\n{chunk}"
    )


def total_findings(results: List[ChunkResult]) -> int:
    return sum(len(item.findings) for item in results)


async def judge_chunk(
    *,
    idx: int,
    chunk: str,
    candidates: Dict[str, List[str]],
    api_key: str,
    model: str,
    timeout: float,
    retries: int,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, ChunkResult]:
    prompt = build_prompt(candidates, chunk)
    async with semaphore:
        try:
            response_json = await asyncio.to_thread(
                call_responses_api,
                api_key,
                model,
                prompt,
                timeout,
                retries,
            )
            response_text = extract_output_text(response_json)
            parsed = extract_json_block(response_text)
            findings: List[Finding] = []
            for item in parsed.get("findings", []) or []:
                findings.append(
                    Finding(
                        category=str(item.get("category", "other")),
                        value=str(item.get("value", "")),
                        evidence=str(item.get("evidence", "")),
                        reason=str(item.get("reason", "")),
                        confidence=float(item.get("confidence", 0.0)),
                        chunk_index=idx,
                    )
                )
            result = ChunkResult(
                chunk_index=idx,
                chunk_chars=len(chunk),
                findings=findings,
                error=None,
                raw_response_text=response_text,
            )
        except Exception as exc:
            result = ChunkResult(
                chunk_index=idx,
                chunk_chars=len(chunk),
                findings=[],
                error=str(exc),
                raw_response_text="",
            )
    return idx, result


async def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set (or use --api-key)")
        return 1

    expected_mod = load_expected_helpers()

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

        document_type = detect_document_type(path, text, args.document_type)
        candidates = expected_mod.extract_expected(
            text, include_phones=False, include_addresses=True
        )
        candidates = {
            "names": candidates.get("names", []),
            "birth_dates": candidates.get("dobs", []),
            "addresses": candidates.get("addresses", []),
            "ids": candidates.get("ids", []),
        }

        if args.input_is_anonymized or args.anonymizer == "none":
            anonymized_text = text
            anonymizer = "none"
        elif args.anonymizer == "local-flair":
            print("[INFO] Running local Flair anonymizer...")
            anonymized_text = anonymize_text_local_flair(text)
            anonymizer = "local-flair"
        elif args.anonymizer == "hybrid":
            print("[INFO] Running app hybrid anonymizer...")
            anonymized_text = await anonymize_text_hybrid(
                text, document_type, args.service_url
            )
            anonymizer = "hybrid"
        else:
            print("[INFO] Running live manager anonymizer...")
            anonymized_text = anonymize_text_live_manager(
                text, document_type, args.service_url
            )
            anonymizer = "live-manager"

        chunks = chunk_text(anonymized_text, args.chunk_chars, args.max_chunks)
        results: List[Optional[ChunkResult]] = [None] * len(chunks)
        semaphore = asyncio.Semaphore(max(1, args.concurrency))
        tasks = []
        for idx, chunk in enumerate(chunks):
            print(f"[INFO] Queueing chunk {idx + 1}/{len(chunks)} ({len(chunk)} chars)")
            tasks.append(
                judge_chunk(
                    idx=idx,
                    chunk=chunk,
                    candidates=candidates,
                    api_key=api_key,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                    semaphore=semaphore,
                )
            )

        for idx, result in await asyncio.gather(*tasks):
            results[idx] = result

        final_results = [result for result in results if result is not None]

        report = FileReport(
            path=str(path),
            document_type=document_type,
            anonymizer=anonymizer,
            model=args.model,
            original_chars=len(text),
            anonymized_chars=len(anonymized_text),
            chunk_chars=args.chunk_chars,
            total_chunks=len(chunks),
            candidate_counts={k: len(v) for k, v in candidates.items()},
            results=final_results,
        )
        reports.append(report)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"llm_judge_{path.stem}_{timestamp}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(
            f"[INFO] Wrote report: {report_path} "
            f"(findings={total_findings(final_results)})"
        )

    summary_path = out_dir / "latest_llm_judge_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in reports], f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Wrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

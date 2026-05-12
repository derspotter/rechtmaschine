#!/usr/bin/env python3
"""
Benchmark Gemma 3 anonymization (standard vs optimized decoding).

Measures:
- time to completion
- GPU VRAM usage (nvidia-smi sampling)
- entity counts + placeholder counts

Outputs JSON report + optional anonymized text outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anon.anonymization_service import (
    apply_regex_replacements as production_apply_regex_replacements,
)


DEFAULT_INPUTS = [
    Path("/home/jayjag/Downloads/onwachukwu_bescheid.pdf"),
    REPO_ROOT / "test_files" / "ocred" / "ocr_Anhoerung.md",
    REPO_ROOT / "test_files" / "ocred" / "ocr_2025-02-10_Bescheid.md",
]


PROMPT_TEMPLATE = """<|im_start|>system
Extract ALL person names and PII from this legal document. Return JSON only. /no_think

Find:
- names: ALL person names mentioned (clients, applicants, family members, anyone referred to by name including after titles like Genossin/Genosse/Frau/Herr/Mandant/Klient)
- birth_dates: Birth dates (DD.MM.YYYY)
- birth_places: Birthplace cities (foreign)
- streets: Residence streets (NOT BAMF offices)
- postal_codes: Residence postal codes (NOT 90343/90461/53115)
- cities: Residence cities (NOT Nürnberg/Bonn BAMF offices)
- azr_numbers: AZR numbers
- case_numbers: Case numbers

JSON:
{{"names":[], "birth_dates":[], "birth_places":[], "streets":[], "postal_codes":[], "cities":[], "azr_numbers":[], "case_numbers":[]}}
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant
"""


BAMF_OFFICE_DATA = {
    "postal_codes": {"90343", "90461", "53115"},
    "cities": {"nürnberg", "bonn"},
    "streets": {"frankenstraße", "frankenstrabe", "reuterstraße", "reuterstrabe"},
}

HONORIFICS = {
    "herr",
    "herrn",
    "frau",
    "genosse",
    "genossin",
    "mandant",
    "mandantin",
    "klient",
    "klientin",
}


def filter_bamf_addresses(entities: Dict[str, Any]) -> Dict[str, Any]:
    filtered: Dict[str, Any] = {}
    for key, values in entities.items():
        if not isinstance(values, list):
            filtered[key] = values
            continue
        if key == "postal_codes":
            filtered[key] = [v for v in values if v not in BAMF_OFFICE_DATA["postal_codes"]]
        elif key == "cities":
            filtered[key] = [v for v in values if v.lower() not in BAMF_OFFICE_DATA["cities"]]
        elif key == "streets":
            filtered[key] = [
                v for v in values
                if not any(bamf in v.lower() for bamf in BAMF_OFFICE_DATA["streets"])
            ]
        else:
            filtered[key] = values
    return filtered


def _escape_fuzzy(term: str) -> str:
    pattern = re.escape(term)
    pattern = pattern.replace(r"\ ", r"\s*")
    pattern = pattern.replace(r"\-", r"\s*-\s*")
    pattern = pattern.replace(",", r"\s*,\s*")
    return pattern


def _person_term_variants(term: str) -> List[str]:
    clean = re.sub(r"\s+", " ", term).strip()
    if not clean:
        return []

    variants = [clean]

    if "," in clean:
        left, right = [p.strip() for p in clean.split(",", 1)]
        if left and right:
            left_tokens = [t for t in left.split(" ") if t]
            while left_tokens and left_tokens[0].lower() in HONORIFICS:
                left_tokens = left_tokens[1:]
            left_core = " ".join(left_tokens).strip()
            if left_core:
                variants.append(f"{right} {left_core}")
    else:
        tokens = [t for t in clean.split(" ") if t]
        tokens_wo_titles = tokens[:]
        while tokens_wo_titles and tokens_wo_titles[0].lower() in HONORIFICS:
            tokens_wo_titles = tokens_wo_titles[1:]

        if len(tokens_wo_titles) >= 2:
            last = tokens_wo_titles[-1]
            first = " ".join(tokens_wo_titles[:-1])
            variants.append(f"{last}, {first}")

    seen = set()
    out: List[str] = []
    for v in variants:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def safe_replace(text: str, terms: List[str], placeholder: str) -> str:
    sorted_terms = sorted(
        [t for t in terms if isinstance(t, str)],
        key=lambda t: len(t.strip()),
        reverse=True,
    )

    for raw_term in sorted_terms:
        term = raw_term.strip()
        if len(term) < 2:
            continue

        variants = [term]
        if placeholder == "[PERSON]":
            variants = _person_term_variants(term)

        for variant in variants:
            if len(variant) < 2:
                continue
            pattern = _escape_fuzzy(variant)
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    return text


def apply_regex_replacements(text: str, entities: Dict[str, Any]) -> str:
    anon = text
    anon = safe_replace(anon, entities.get("names", []), "[PERSON]")
    anon = safe_replace(anon, entities.get("birth_dates", []), "[GEBURTSDATUM]")
    anon = safe_replace(anon, entities.get("birth_places", []), "[GEBURTSORT]")
    anon = safe_replace(anon, entities.get("streets", []), "[ADRESSE]")
    anon = safe_replace(anon, entities.get("postal_codes", []), "[PLZ]")
    anon = safe_replace(anon, entities.get("cities", []), "[ORT]")
    anon = safe_replace(anon, entities.get("azr_numbers", []), "[AZR-NUMMER]")
    anon = safe_replace(anon, entities.get("case_numbers", []), "[AKTENZEICHEN]")
    anon = re.sub(r"ZUE\\s+\\w+", "[UNTERKUNFT]", anon)
    return anon


def load_text(path: Path, max_pages: int = 50) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"PyMuPDF not available: {exc}") from exc
        with fitz.open(path) as doc:
            pages_to_read = min(len(doc), max_pages)
            parts: List[str] = []
            for i in range(pages_to_read):
                text = doc[i].get_text()
                if text:
                    parts.append(text)
            return "".join(parts)
    return path.read_text(encoding="utf-8")


def resolve_ollama_url(cli_url: Optional[str]) -> str:
    if cli_url:
        return cli_url
    env_url = os.getenv("OLLAMA_URL")
    if env_url:
        return env_url
    env_host = os.getenv("OLLAMA_HOST")
    if env_host:
        base = env_host.rstrip("/")
        if base.endswith("/api/generate"):
            return base
        return f"{base}/api/generate"
    return "http://localhost:11434/api/generate"


def sample_gpu(stop_event: threading.Event, interval: float, samples: List[int]) -> None:
    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            if output:
                first = output.splitlines()[0].strip()
                if first:
                    samples.append(int(first))
        except Exception:
            pass
        time.sleep(interval)


def placeholder_counts(text: str) -> Dict[str, int]:
    return {
        "PERSON": len(re.findall(r"\[PERSON(?:_\d+)?\]", text)),
        "GEBURTSDATUM": text.count("[GEBURTSDATUM]"),
        "GEBURTSORT": text.count("[GEBURTSORT]"),
        "ADRESSE": text.count("[ADRESSE]"),
        "PLZ": text.count("[PLZ]"),
        "ORT": text.count("[ORT]"),
        "AZR": text.count("[AZR-NUMMER]"),
        "AKTENZEICHEN": text.count("[AKTENZEICHEN]"),
    }


@dataclass
class RunResult:
    config_name: str
    model: str
    options: Dict[str, Any]
    duration_sec: float
    gpu_mem_max_mb: Optional[int]
    input_chars: int
    output_chars: int
    entities: Dict[str, int]
    raw_entities: Dict[str, Any]
    placeholders: Dict[str, int]
    error: Optional[str] = None
    output_path: Optional[str] = None


def run_config(
    *,
    config_name: str,
    model: str,
    options: Dict[str, Any],
    text: str,
    document_type: str,
    ollama_url: str,
    timeout: float,
    gpu_sample_interval: float,
    output_dir: Optional[Path],
    output_stem: str,
) -> RunResult:
    samples: List[int] = []
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=sample_gpu,
        args=(stop_event, gpu_sample_interval, samples),
        daemon=True,
    )

    prompt = PROMPT_TEMPLATE.format(text=text)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "raw": True,
        "options": options,
    }

    start = time.monotonic()
    sampler.start()
    error = None
    anonymized_text = ""
    entities: Dict[str, Any] = {}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()
        raw_response = result.get("response", "")
        entities = json.loads(raw_response)
        entities = filter_bamf_addresses(entities)
        anonymized_text = production_apply_regex_replacements(text, entities)
    except Exception as exc:
        error = str(exc)
    finally:
        stop_event.set()
        sampler.join(timeout=2.0)

    duration = time.monotonic() - start
    gpu_max = max(samples) if samples else None

    entity_counts = {
        "names": len(entities.get("names", [])) if entities else 0,
        "birth_dates": len(entities.get("birth_dates", [])) if entities else 0,
        "birth_places": len(entities.get("birth_places", [])) if entities else 0,
        "streets": len(entities.get("streets", [])) if entities else 0,
        "postal_codes": len(entities.get("postal_codes", [])) if entities else 0,
        "cities": len(entities.get("cities", [])) if entities else 0,
        "azr_numbers": len(entities.get("azr_numbers", [])) if entities else 0,
        "case_numbers": len(entities.get("case_numbers", [])) if entities else 0,
    }

    output_path = None
    if output_dir and anonymized_text:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{output_stem}_{config_name}.txt")
        Path(output_path).write_text(anonymized_text, encoding="utf-8")

    return RunResult(
        config_name=config_name,
        model=model,
        options=options,
        duration_sec=round(duration, 3),
        gpu_mem_max_mb=gpu_max,
        input_chars=len(text),
        output_chars=len(anonymized_text),
        entities=entity_counts,
        raw_entities=entities if entities else {},
        placeholders=placeholder_counts(anonymized_text),
        error=error,
        output_path=output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gemma 3 anonymization.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(p) for p in DEFAULT_INPUTS if p.exists()],
        help="Paths to PDFs or text files.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
        help="Ollama model name (default: gemma3:12b).",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama generate endpoint (default: env OLLAMA_URL/OLLAMA_HOST or localhost:11434).",
    )
    parser.add_argument(
        "--document-type",
        default="Bescheid",
        help="Document type label (default: Bescheid).",
    )
    parser.add_argument(
        "--config",
        choices=("standard", "optimized", "both"),
        default="optimized",
        help="Decoding config to run (default: optimized).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout seconds per request (default: 300).",
    )
    parser.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=0.5,
        help="Seconds between nvidia-smi samples (default: 0.5).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "tests" / "anonymization_bench_results"),
        help="Output directory for reports and anonymized text.",
    )
    parser.add_argument(
        "--no-write-outputs",
        action="store_true",
        help="Do not write anonymized text outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.paths]
    if not paths:
        raise SystemExit("No input paths provided.")

    ollama_url = resolve_ollama_url(args.ollama_url)
    out_dir = None if args.no_write_outputs else Path(args.out_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    all_configs = {
        "standard": {
            "name": "standard",
            # Current production-ish settings in anon/anonymization_service.py
            "options": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "num_ctx": 32768,
            },
        },
        "optimized": {
            "name": "optimized",
            # Suggested extraction-friendly settings
            "options": {
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 64,
                "min_p": 0.0,
                "num_ctx": 32768,
            },
        },
    }
    configs = (
        list(all_configs.values())
        if args.config == "both"
        else [all_configs[args.config]]
    )

    report = {
        "timestamp": timestamp,
        "model": args.model,
        "ollama_url": ollama_url,
        "document_type": args.document_type,
        "config": args.config,
        "results": [],
    }

    for path in paths:
        if not path.exists():
            report["results"].append(
                {
                    "path": str(path),
                    "error": "file_not_found",
                }
            )
            continue

        text = load_text(path)
        file_block = {
            "path": str(path),
            "input_chars": len(text),
            "runs": [],
        }
        output_stem = f"{path.stem}_{timestamp}"
        for cfg in configs:
            result = run_config(
                config_name=cfg["name"],
                model=args.model,
                options=cfg["options"],
                text=text,
                document_type=args.document_type,
                ollama_url=ollama_url,
                timeout=args.timeout,
                gpu_sample_interval=args.gpu_sample_interval,
                output_dir=out_dir,
                output_stem=output_stem,
            )
            file_block["runs"].append(asdict(result))
        report["results"].append(file_block)

    out_dir_report = Path(args.out_dir)
    out_dir_report.mkdir(parents=True, exist_ok=True)
    report_path = out_dir_report / f"gemma_benchmark_{timestamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(report_path))


if __name__ == "__main__":
    main()

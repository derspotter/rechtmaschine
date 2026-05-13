#!/usr/bin/env python3
"""Benchmark deterministic citation checks against a cheap LLM page judge.

Run from the host against the app container, for example:

    docker exec -i rechtmaschine-app python - --limit 12 --sample-size 24 < scripts/benchmark_citation_verifier_llm.py

The script sends cited page text plus the claim sentence to the configured OpenAI
model. It is intentionally a benchmark/calibration tool, not production logic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
APP_DIR = REPO_ROOT / "app"
if APP_DIR.exists():
    sys.path.insert(0, str(APP_DIR))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file(APP_DIR / ".env")

from sqlalchemy import desc  # noqa: E402

from citation_verifier import (  # noqa: E402
    AMBIGUOUS,
    FOUND_ON_DIFFERENT_PAGE,
    NO_PAGE_TEXT_AVAILABLE,
    NOT_FOUND,
    VERIFIED_ON_CITED_PAGE,
    _load_document_texts,
    _page_texts_from_entry,
    verify_page_citations,
)
from database import SessionLocal  # noqa: E402
from citation_qwen import judge_citation_page_with_qwen  # noqa: E402
from models import Case, Document, GeneratedDraft  # noqa: E402
from shared import get_native_openai_client  # noqa: E402


DEFAULT_STATUSES = {
    NOT_FOUND,
    AMBIGUOUS,
    FOUND_ON_DIFFERENT_PAGE,
    VERIFIED_ON_CITED_PAGE,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12, help="Recent drafts to inspect.")
    parser.add_argument("--sample-size", type=int, default=24, help="Max citation checks to send to the LLM.")
    parser.add_argument("--case-id", default="", help="Only inspect drafts for this case UUID.")
    parser.add_argument("--case-query", default="", help="Only inspect drafts for cases whose name contains this text.")
    parser.add_argument("--draft-id", default="", help="Only inspect this draft UUID.")
    parser.add_argument("--model-used", default="", help="Only inspect drafts with this stored model_used value.")
    parser.add_argument(
        "--model",
        default=os.getenv("CITATION_BENCHMARK_LLM_MODEL", "gpt-5.4-mini"),
        help="OpenAI judge model. Default: CITATION_BENCHMARK_LLM_MODEL or gpt-5.4-mini.",
    )
    parser.add_argument(
        "--judge",
        choices=["openai", "qwen"],
        default=os.getenv("CITATION_BENCHMARK_JUDGE", "openai"),
        help="Page judge backend. Default: openai.",
    )
    parser.add_argument(
        "--statuses",
        default=",".join(sorted(DEFAULT_STATUSES)),
        help="Comma-separated deterministic statuses to sample.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic sample seed.")
    parser.add_argument("--page-char-limit", type=int, default=12000, help="Max chars of page text per LLM call.")
    parser.add_argument("--show-samples", action="store_true", help="Print sanitized per-sample details.")
    parser.add_argument("--show-reasons", action="store_true", help="Include LLM reason text in sample output.")
    parser.add_argument("--dry-run", action="store_true", help="Build the benchmark sample without calling the LLM.")
    return parser.parse_args()


def _entry_for_document(doc: Document, item: Dict[str, Any], category: str) -> Dict[str, Any]:
    entry = {
        "id": str(doc.id),
        "filename": doc.filename,
        "category": category,
        "role": item.get("role") or "",
        "file_path": doc.file_path,
        "extracted_text_path": doc.extracted_text_path,
        "anonymization_metadata": doc.anonymization_metadata,
        "is_anonymized": doc.is_anonymized,
    }
    entry["page_texts"] = _page_texts_from_entry(entry)
    return entry


def _selected_documents_for_draft(session, draft: GeneratedDraft, cache: Dict[Any, Dict[str, Any]]) -> tuple[Dict[str, List[Dict[str, Any]]], int]:
    selected = {
        "anhoerung": [],
        "bescheid": [],
        "vorinstanz": [],
        "rechtsprechung": [],
        "saved_sources": [],
        "sonstiges": [],
        "internal_notes": [],
        "akte": [],
    }
    unresolved = 0
    for item in (draft.metadata_ or {}).get("used_documents") or []:
        filename = (item.get("filename") or "").strip()
        category = (item.get("category") or "sonstiges").strip()
        if not filename:
            continue
        cache_key = (str(draft.case_id), filename)
        entry = cache.get(cache_key)
        if entry is None:
            query = session.query(Document).filter(Document.filename == filename)
            if draft.case_id:
                query = query.filter(Document.case_id == draft.case_id)
            doc = query.order_by(desc(Document.created_at)).first()
            if not doc:
                unresolved += 1
                continue
            entry = _entry_for_document(doc, item, category)
            cache[cache_key] = deepcopy(entry)
        entry = deepcopy(entry)
        entry["category"] = category
        entry["role"] = item.get("role") or entry.get("role") or ""
        selected.setdefault(category, []).append(entry)
    return selected, unresolved


def _collect_checks(
    limit: int,
    statuses: set[str],
    model_used: str = "",
    case_id: str = "",
    case_query: str = "",
    draft_id: str = "",
) -> tuple[List[Dict[str, Any]], Counter, int]:
    session = SessionLocal()
    entry_cache: Dict[Any, Dict[str, Any]] = {}
    checks: List[Dict[str, Any]] = []
    totals: Counter = Counter()
    unresolved_total = 0
    try:
        query = session.query(GeneratedDraft)
        if draft_id:
            query = query.filter(GeneratedDraft.id == draft_id)
        if case_id:
            query = query.filter(GeneratedDraft.case_id == case_id)
        if case_query:
            matching_case_ids = [
                case.id
                for case in session.query(Case)
                .filter(Case.name.ilike(f"%{case_query}%"))
                .all()
            ]
            query = query.filter(GeneratedDraft.case_id.in_(matching_case_ids))
        if model_used:
            query = query.filter(GeneratedDraft.model_used == model_used)
        drafts = query.order_by(desc(GeneratedDraft.created_at)).limit(limit).all()
        for draft in drafts:
            selected, unresolved = _selected_documents_for_draft(session, draft, entry_cache)
            unresolved_total += unresolved
            deterministic = verify_page_citations(draft.generated_text or "", selected)
            totals.update(deterministic.get("summary") or {})

            documents = _load_document_texts(selected)
            by_label = {document.label: document for document in documents}
            for check in deterministic.get("checks") or []:
                status = str(check.get("status") or "")
                if status not in statuses:
                    continue
                samples = _samples_for_check(check, by_label)
                for sample in samples:
                    sample["draft_id"] = str(draft.id)
                    sample["deterministic_status"] = status
                    sample["citation"] = check.get("citation")
                    sample["claim"] = check.get("claim")
                    sample["sentence"] = check.get("sentence")
                    checks.append(sample)
        return checks, totals, unresolved_total
    finally:
        session.close()


def _samples_for_check(check: Dict[str, Any], by_label: Dict[str, Any]) -> List[Dict[str, Any]]:
    cited_pages = [int(page) for page in check.get("cited_pages") or [] if str(page).isdigit()]
    if not cited_pages:
        return []

    document_payloads: List[Dict[str, Any]] = []
    if isinstance(check.get("document"), dict):
        document_payloads.append(check["document"])
    for candidate in check.get("candidate_documents") or []:
        if isinstance(candidate, dict):
            document_payloads.append(candidate)

    samples = []
    seen = set()
    for payload in document_payloads:
        label = payload.get("label")
        if not label or label in seen:
            continue
        seen.add(label)
        document = by_label.get(label)
        if not document or not document.pages:
            continue
        page_text = "\n\n".join(document.pages.get(page, "") for page in cited_pages).strip()
        if not page_text:
            continue
        samples.append({
            "document_label": label,
            "page_numbers": cited_pages,
            "page_text": page_text,
        })
    return samples


def _judge_with_llm(client, model: str, sample: Dict[str, Any], page_char_limit: int) -> Dict[str, Any]:
    page_text = sample["page_text"][:page_char_limit]
    prompt = f"""You are a strict citation verification judge.

Task: Decide whether the CLAIM is supported by the PAGE TEXT.

Rules:
- Use only PAGE TEXT.
- Answer yes if the same factual content is present, even if wording differs.
- Answer no if the claim is absent or contradicted.
- Answer unclear if the page text is too noisy or insufficient.
- Return JSON only with keys: verdict, confidence, reason.

CLAIM:
{sample.get("claim") or ""}

FULL SENTENCE:
{sample.get("sentence") or ""}

PAGE TEXT:
{page_text}
"""
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 350,
    }
    if model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": "low"}
        kwargs["text"] = {"verbosity": "low"}
    response = client.responses.create(**kwargs)
    output_text = (getattr(response, "output_text", "") or "").strip()
    try:
        parsed = json.loads(output_text)
    except Exception:
        parsed = {"verdict": "unclear", "confidence": 0.0, "reason": output_text[:240]}
    verdict = str(parsed.get("verdict") or "unclear").strip().lower()
    if verdict not in {"yes", "no", "unclear"}:
        verdict = "unclear"
    return {
        "verdict": verdict,
        "confidence": _safe_float(parsed.get("confidence")),
        "reason": str(parsed.get("reason") or "")[:300],
    }


async def _judge_with_qwen(sample: Dict[str, Any], page_char_limit: int) -> Dict[str, Any]:
    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        return {
            "verdict": "unclear",
            "confidence": 0.0,
            "reason": "ANONYMIZATION_SERVICE_URL not configured",
        }
    qwen_sample = dict(sample)
    qwen_sample["page_text"] = str(qwen_sample.get("page_text") or "")[:page_char_limit]
    return await judge_citation_page_with_qwen(service_url, qwen_sample)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _expected_yes(status: str) -> Optional[bool]:
    if status == VERIFIED_ON_CITED_PAGE:
        return True
    if status in {NOT_FOUND, FOUND_ON_DIFFERENT_PAGE}:
        return False
    return None


def main() -> int:
    args = _parse_args()
    statuses = {status.strip() for status in args.statuses.split(",") if status.strip()}
    samples, deterministic_totals, unresolved = _collect_checks(
        args.limit,
        statuses,
        args.model_used,
        args.case_id,
        args.case_query,
        args.draft_id,
    )
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.sample_size]

    print(f"deterministic_totals={dict(deterministic_totals)}")
    print(f"unresolved_used_documents={unresolved}")
    print(
        f"llm_samples={len(samples)} judge={args.judge} model={args.model} "
        f"model_used_filter={args.model_used or 'none'} case_id={args.case_id or 'none'} "
        f"case_query={args.case_query or 'none'} draft_id={args.draft_id or 'none'}"
    )
    if not samples:
        return 0
    if args.dry_run:
        print("dry_run=true")
        print(f"sample_statuses={dict(Counter(sample['deterministic_status'] for sample in samples))}")
        return 0

    client = get_native_openai_client() if args.judge == "openai" else None
    verdicts: Counter = Counter()
    agreements: Counter = Counter()
    by_status: Dict[str, Counter] = {}

    for idx, sample in enumerate(samples, start=1):
        if args.judge == "qwen":
            llm = asyncio.run(_judge_with_qwen(sample, args.page_char_limit))
        else:
            llm = _judge_with_llm(client, args.model, sample, args.page_char_limit)
        status = sample["deterministic_status"]
        verdicts[llm["verdict"]] += 1
        by_status.setdefault(status, Counter())[llm["verdict"]] += 1
        expected = _expected_yes(status)
        if expected is not None:
            agrees = (llm["verdict"] == "yes") == expected if llm["verdict"] != "unclear" else False
            agreements["agree" if agrees else "disagree_or_unclear"] += 1
        else:
            agreements["not_scored"] += 1

        if args.show_samples:
            llm_payload = dict(llm)
            if not args.show_reasons:
                llm_payload.pop("reason", None)
            print(
                json.dumps(
                    {
                        "i": idx,
                        "draft_id": sample["draft_id"][:8],
                        "status": status,
                        "citation": sample.get("citation"),
                        "pages": sample.get("page_numbers"),
                        "llm": llm_payload,
                    },
                    ensure_ascii=False,
                )
            )

    print(f"llm_verdicts={dict(verdicts)}")
    print(f"agreement={dict(agreements)}")
    print(f"by_status={ {key: dict(value) for key, value in by_status.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

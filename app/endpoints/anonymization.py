import json
import os
import tempfile
import hashlib
from datetime import datetime
from typing import Any, Optional
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from shared import (
    AnonymizationResult,
    DocumentCategory,
    broadcast_documents_snapshot,
    ensure_service_manager_ready,
    limiter,
    load_document_text,
    store_document_text,
    ANONYMIZED_TEXT_DIR,
)
from auth import get_current_active_user
from database import get_db
from models import Document, User
from .ocr import extract_pdf_text, perform_ocr_on_file, check_pdf_needs_ocr
from anon.anonymization_service import (
    filter_bamf_addresses,
    filter_non_person_group_labels,
    filter_non_person_organization_labels,
    augment_names_from_role_markers,
    augment_names_from_person_fields,
    apply_regex_replacements,
)

router = APIRouter()

GEMMA_MODEL = os.getenv("OLLAMA_MODEL_GEMMA", os.getenv("OLLAMA_MODEL", "gemma3:12b"))
QWEN_MODEL = os.getenv("OLLAMA_MODEL_QWEN", "qwen3:8b")
ANONYMIZATION_ENGINE_DEFAULT = os.getenv(
    "ANONYMIZATION_ENGINE_DEFAULT", "qwen_flair"
).strip().lower()
SUPPORTED_ANONYMIZATION_ENGINES = {"gemma", "qwen_flair"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Invalid int env {name}={raw!r}, using default={default}")
        return default
    if value <= 0:
        print(f"[WARN] Non-positive int env {name}={raw!r}, using default={default}")
        return default
    return value


def _optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_positive_int_env(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Invalid int env {name}={raw!r}, ignoring")
        return None
    if value <= 0:
        return None
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"[WARN] Invalid float env {name}={raw!r}, using default={default}")
        return default
    return value


NAMES_EXTRACTION_PROMPT_PREFIX = """Extract PERSON names from this German legal document.
Return valid JSON only with exactly:
{"names":[]}

Rules:
- names: natural persons only (applicants, family members, officials, signers)
- include exact surface forms from text, including surname-only and "SURNAME, Given" forms
- include abbreviated and hyphen forms (e.g. "S. Quast", "A-Nabi")
- if text contains "Es erscheint Herr/Frau X, Y geb.", include BOTH person tokens X and Y
- include every person from family lines (Vater, Mutter, Ehemann, Ehefrau, Sohn, Tochter, Geschwister)
- include names near role/signature markers (Anhörender Entscheider, Sachbearbeiter, Unterzeichner, Im Auftrag, gez., Unterschrift)
- include standalone person-like signature lines before page footers ("Seite X von Y")
- include OCR-noisy single-word signature names in signer blocks (if person-like)
- exclude organizations, courts, legal citations, ethnicities, religions, and nationalities
- deduplicate exact duplicates

Document:
"""

ADDRESSES_EXTRACTION_PROMPT_PREFIX = """Extract only private residence addresses from this German legal document.
Return valid JSON only with exactly:
{"streets":[], "postal_codes":[], "cities":[]}

Rules:
- streets: private residence street + house number only
- postal_codes: 5-digit private residence ZIP codes only
- cities: private residence cities tied to applicant/family address context
- prioritize address contexts like wohnhaft/Anschrift/Wohnort/Adresse
- do NOT extract BAMF office addresses (e.g. Nürnberg/Bonn office blocks; ZIP 90343/90461/53115)
- do NOT extract court/authority addresses or generic location mentions
- deduplicate exact duplicates

Document:
"""

BIRTH_IDS_EXTRACTION_PROMPT_PREFIX = """Extract birth details and personal document IDs from this German legal document.
Return valid JSON only with exactly:
{"birth_dates":[], "birth_places":[], "azr_numbers":[], "aufenthaltsgestattung_ids":[], "case_numbers":[]}

Rules:
- birth_dates: full date strings for person birth data (e.g. DD.MM.YYYY, "geb. am ...")
- if only a birth year is given, include the full surrounding birth phrase (e.g. "1992 geboren")
- birth_places: city/place directly tied to birth context
- azr_numbers: AZR numbers (labeled or clearly AZR context)
- aufenthaltsgestattung_ids: IDs explicitly labeled as Aufenthaltsgestattung
- case_numbers: personal/document IDs (e.g. Dolmetscher-Nr, D4S..., numeric id blocks)
- do NOT include court citations/references (ECLI, BVerwG/BVerfG/VG/OVG Az., §/Art. citations)
- deduplicate exact duplicates

Document:
"""

EXTRACTION_ENTITY_KEYS = [
    "names",
    "birth_dates",
    "birth_places",
    "streets",
    "postal_codes",
    "cities",
    "azr_numbers",
    "aufenthaltsgestattung_ids",
    "case_numbers",
]

EXTRACTION_FIELD_SCHEMA = {
    "names": {"type": "array", "items": {"type": "string"}},
    "birth_dates": {"type": "array", "items": {"type": "string"}},
    "birth_places": {"type": "array", "items": {"type": "string"}},
    "streets": {"type": "array", "items": {"type": "string"}},
    "postal_codes": {"type": "array", "items": {"type": "string"}},
    "cities": {"type": "array", "items": {"type": "string"}},
    "azr_numbers": {"type": "array", "items": {"type": "string"}},
    "aufenthaltsgestattung_ids": {"type": "array", "items": {"type": "string"}},
    "case_numbers": {"type": "array", "items": {"type": "string"}},
}

EXTRACTION_STAGE_SPECS = [
    {
        "name": "names",
        "keys": ["names"],
        "prompt_prefix": NAMES_EXTRACTION_PROMPT_PREFIX,
    },
    {
        "name": "addresses",
        "keys": ["streets", "postal_codes", "cities"],
        "prompt_prefix": ADDRESSES_EXTRACTION_PROMPT_PREFIX,
    },
    {
        "name": "birth_ids",
        "keys": [
            "birth_dates",
            "birth_places",
            "azr_numbers",
            "aufenthaltsgestattung_ids",
            "case_numbers",
        ],
        "prompt_prefix": BIRTH_IDS_EXTRACTION_PROMPT_PREFIX,
    },
]


def resolve_anonymization_engine(requested_engine: Optional[str]) -> str:
    engine = (requested_engine or ANONYMIZATION_ENGINE_DEFAULT).strip().lower()
    if engine in SUPPORTED_ANONYMIZATION_ENGINES:
        return engine
    print(
        f"[WARN] Unknown anonymization engine '{engine}', "
        f"falling back to default '{ANONYMIZATION_ENGINE_DEFAULT}'"
    )
    if ANONYMIZATION_ENGINE_DEFAULT in SUPPORTED_ANONYMIZATION_ENGINES:
        return ANONYMIZATION_ENGINE_DEFAULT
    return "gemma"


def _dedupe_entity_lists(entities: dict) -> dict:
    deduped = {}
    for key, values in entities.items():
        if not isinstance(values, list):
            deduped[key] = values
            continue
        seen = set()
        out = []
        for value in values:
            if not isinstance(value, str):
                continue
            clean = value.strip()
            if not clean:
                continue
            token = clean.casefold()
            if token in seen:
                continue
            seen.add(token)
            out.append(clean)
        deduped[key] = out
    return deduped


def _normalize_extraction_entities(payload: Any) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {key: [] for key in EXTRACTION_ENTITY_KEYS}
    if not isinstance(payload, dict):
        return normalized

    for key in EXTRACTION_ENTITY_KEYS:
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        out: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            clean = value.strip()
            if not clean:
                continue
            out.append(clean)
        normalized[key] = out
    return normalized


def _merge_extraction_entities(base: dict[str, list[str]], incoming: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {key: list(base.get(key, [])) for key in EXTRACTION_ENTITY_KEYS}
    for key in EXTRACTION_ENTITY_KEYS:
        merged[key].extend(incoming.get(key, []))
    return _dedupe_entity_lists(merged)


def _build_extraction_format_schema(keys: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {key: EXTRACTION_FIELD_SCHEMA[key] for key in keys},
        "required": keys,
        "additionalProperties": False,
    }


def _split_text_for_extraction(
    text: str, chunk_pages: int, fallback_chunk_chars: int
) -> list[str]:
    if chunk_pages <= 0:
        return [text]

    clean_text = text or ""
    if not clean_text.strip():
        return [clean_text]

    pages: list[str] = []
    if "\f" in clean_text:
        pages = [p.strip() for p in clean_text.split("\f") if p and p.strip()]

    if pages:
        chunks: list[str] = []
        for i in range(0, len(pages), chunk_pages):
            chunk = "\n\n\f\n\n".join(pages[i : i + chunk_pages]).strip()
            if chunk:
                chunks.append(chunk)
        if chunks:
            return chunks

    if fallback_chunk_chars <= 0 or len(clean_text) <= fallback_chunk_chars:
        return [clean_text]

    chunks: list[str] = []
    start = 0
    length = len(clean_text)
    while start < length:
        end = min(length, start + fallback_chunk_chars)
        if end < length:
            split_at = clean_text.rfind("\n\n", start, end)
            if split_at > start + 1024:
                end = split_at
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks or [clean_text]


def _merge_flair_names(entities: dict, flair_names: list[str]) -> dict:
    names = entities.get("names", [])
    if not isinstance(names, list):
        names = []

    seen = {n.strip().casefold() for n in names if isinstance(n, str) and n.strip()}
    added = []
    for raw in flair_names:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if len(candidate) < 2:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(candidate)
        added.append(candidate)

    if added:
        print(f"[INFO] Added Flair name hints: {added}")
    entities["names"] = names
    return entities


def _filter_name_artifacts(entities: dict) -> dict:
    names = entities.get("names")
    if not isinstance(names, list):
        return entities

    filtered: list[str] = []
    for raw in names:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if not candidate:
            continue

        lowered = candidate.casefold()
        if lowered.startswith("nr") and any(ch.isdigit() for ch in candidate):
            continue
        if not any(ch.isalpha() for ch in candidate):
            continue
        if sum(ch.isdigit() for ch in candidate) >= 3:
            continue

        filtered.append(candidate)

    entities["names"] = filtered
    return entities


def _stage_temperature(stage_name: str, is_gemma3: bool, default_temperature: float) -> float:
    if is_gemma3:
        if stage_name == "names":
            return _float_env("OLLAMA_NAMES_TEMP_GEMMA3", 0.5)
        if stage_name == "addresses":
            return _float_env("OLLAMA_ADDRESSES_TEMP_GEMMA3", 0.2)
        if stage_name == "birth_ids":
            return _float_env("OLLAMA_BIRTH_IDS_TEMP_GEMMA3", 0.2)
    return default_temperature


def _stage_passes(stage_name: str, is_gemma3: bool) -> int:
    if is_gemma3 and stage_name == "names":
        return max(1, _int_env("OLLAMA_NAMES_PASSES_GEMMA3", 2))
    return 1


async def _fetch_flair_name_hints(
    service_url: str, text: str, document_type: str
) -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            status_resp = await client.get(f"{service_url}/status")
            if status_resp.status_code == 200:
                status_payload = status_resp.json()
                anon_backend = (
                    status_payload.get("services", {}).get("anon_backend") or ""
                ).strip().lower()
                if anon_backend != "flair":
                    print(
                        f"[WARN] qwen_flair engine requested but service manager backend "
                        f"is '{anon_backend or 'unknown'}' (expected 'flair')."
                    )

            flair_resp = await client.post(
                f"{service_url}/anonymize",
                json={"text": text, "document_type": document_type},
            )
            flair_resp.raise_for_status()
            flair_data = flair_resp.json()
    except Exception as exc:
        print(f"[WARN] Flair name hint fetch failed: {exc}")
        return []

    plaintiff_names = flair_data.get("plaintiff_names") or []
    family_members = flair_data.get("family_members") or []
    if not isinstance(plaintiff_names, list):
        plaintiff_names = []
    if not isinstance(family_members, list):
        family_members = []
    return [*plaintiff_names, *family_members]


async def anonymize_document_text(
    text: str,
    document_type: str,
    engine: str,
    extract_chunk_pages: Optional[int] = None,
    extract_num_ctx: Optional[int] = None,
) -> Optional[AnonymizationResult]:
    """Extract entities via desktop LLM, then apply regex anonymization locally."""
    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        print("[WARNING] ANONYMIZATION_SERVICE_URL not configured")
        return None

    await ensure_service_manager_ready()

    model = GEMMA_MODEL
    default_temperature = 0.3
    if engine == "qwen_flair":
        model = QWEN_MODEL
        default_temperature = 0.0

    is_gemma3 = (model or "").strip().lower().startswith("gemma3")
    num_ctx = _int_env("OLLAMA_NUM_CTX_DEFAULT", 32768)
    if is_gemma3:
        num_ctx = _int_env("OLLAMA_NUM_CTX_GEMMA3", 32768)
        print(
            f"[INFO] Gemma3 detected (model={model}); using format='json' "
            "instead of JSON schema for extraction stability"
        )

    if extract_num_ctx is not None and extract_num_ctx > 0:
        num_ctx = extract_num_ctx

    env_chunk_pages = _optional_positive_int_env("OLLAMA_EXTRACT_CHUNK_PAGES")
    env_chunk_chars = _optional_positive_int_env("OLLAMA_EXTRACT_CHUNK_CHARS")
    active_chunk_pages = extract_chunk_pages or env_chunk_pages or 0
    fallback_chunk_chars = env_chunk_chars or 18000

    stage_plans: list[dict[str, Any]] = []
    for stage_spec in EXTRACTION_STAGE_SPECS:
        stage_name = stage_spec["name"]
        stage_keys = list(stage_spec["keys"])
        stage_format: str | dict[str, Any] = (
            "json" if is_gemma3 else _build_extraction_format_schema(stage_keys)
        )
        stage_temperature = _stage_temperature(stage_name, is_gemma3, default_temperature)
        stage_pass_count = _stage_passes(stage_name, is_gemma3)
        stage_plans.append(
            {
                "name": stage_name,
                "keys": stage_keys,
                "prompt_prefix": stage_spec["prompt_prefix"],
                "format": stage_format,
                "temperature": stage_temperature,
                "passes": stage_pass_count,
            }
        )

    def _build_payload(
        prompt_text: str, stage_format: str | dict[str, Any], stage_temperature: float
    ) -> dict[str, Any]:
        return {
            "model": model,
            "prompt": prompt_text,
            "stream": False,
            "format": stage_format,
            "options": {
                "temperature": stage_temperature,
                "num_predict": 4096,
                "num_ctx": num_ctx,
                "repeat_penalty": 1.0,
            },
        }

    primary_stage = stage_plans[0]
    extraction_inference_params: dict[str, Any] = {
        "model": model,
        "format": (
            primary_stage["format"]
            if isinstance(primary_stage["format"], str)
            else "json_schema"
        ),
        "temperature": primary_stage["temperature"],
        "num_ctx": num_ctx,
        "top_k": None,
        "top_p": None,
        "min_p": None,
        "repeat_penalty": 1.0,
        "extract_chunk_pages": active_chunk_pages or None,
        "extract_chunk_chars": fallback_chunk_chars if active_chunk_pages else None,
        "staged_extraction": True,
        "stages": [
            {
                "name": stage["name"],
                "keys": stage["keys"],
                "format": stage["format"] if isinstance(stage["format"], str) else "json_schema",
                "temperature": stage["temperature"],
                "passes": stage["passes"],
            }
            for stage in stage_plans
        ],
    }

    try:
        chunks = [text]
        if active_chunk_pages > 0:
            chunks = _split_text_for_extraction(text, active_chunk_pages, fallback_chunk_chars)

        chunk_mode = len(chunks) > 1
        if chunk_mode:
            print(
                f"[INFO] Chunked staged extraction enabled: chunks={len(chunks)} "
                f"chunk_pages={active_chunk_pages} chunk_chars={fallback_chunk_chars}"
            )
        print(
            f"[INFO] Staged extraction request "
            f"url={service_url}/extract-entities model={model} "
            f"payload_chars={len(text)} document_type={document_type} "
            f"engine={engine} num_ctx={num_ctx} "
            f"stages={[(s['name'], s['passes']) for s in stage_plans]}"
        )

        extraction_prompt_tokens_sum = 0
        extraction_completion_tokens_sum = 0
        extraction_total_duration_ns_sum = 0
        extraction_prompt_tokens = None
        extraction_completion_tokens = None
        extraction_total_duration_ns = None
        merged_entities = {key: [] for key in EXTRACTION_ENTITY_KEYS}

        async with httpx.AsyncClient(timeout=300.0) as client:
            for stage in stage_plans:
                stage_name = stage["name"]
                stage_keys = stage["keys"]
                stage_prompt_prefix = stage["prompt_prefix"]
                stage_format = stage["format"]
                stage_temperature = stage["temperature"]
                stage_passes = stage["passes"]

                for pass_idx in range(1, stage_passes + 1):
                    for chunk_idx, chunk_text in enumerate(chunks, start=1):
                        prompt = stage_prompt_prefix + chunk_text
                        payload = _build_payload(prompt, stage_format, stage_temperature)
                        if chunk_mode or stage_passes > 1:
                            print(
                                f"[INFO] Extraction stage={stage_name} "
                                f"pass={pass_idx}/{stage_passes} "
                                f"chunk={chunk_idx}/{len(chunks)} "
                                f"payload_chars={len(chunk_text)} temp={stage_temperature}"
                            )
                        response = await client.post(
                            f"{service_url}/extract-entities",
                            json=payload,
                        )
                        response.raise_for_status()
                        data = response.json()

                        raw_response = data.get("response", "{}")
                        parsed_payload = json.loads(raw_response)
                        parsed_entities = _normalize_extraction_entities(parsed_payload)
                        stage_entities = {
                            key: parsed_entities.get(key, []) for key in stage_keys
                        }
                        merged_entities = _merge_extraction_entities(
                            merged_entities, stage_entities
                        )

                        prompt_tokens = _optional_int(data.get("prompt_eval_count")) or 0
                        completion_tokens = _optional_int(data.get("eval_count")) or 0
                        total_duration_ns = _optional_int(data.get("total_duration")) or 0
                        extraction_prompt_tokens_sum += prompt_tokens
                        extraction_completion_tokens_sum += completion_tokens
                        extraction_total_duration_ns_sum += total_duration_ns

        extraction_prompt_tokens = extraction_prompt_tokens_sum
        extraction_completion_tokens = extraction_completion_tokens_sum
        extraction_total_duration_ns = extraction_total_duration_ns_sum
        extraction_inference_params["chunk_count"] = len(chunks)

        print(
            "[INFO] Entity extraction usage "
            f"prompt_tokens={extraction_prompt_tokens} "
            f"completion_tokens={extraction_completion_tokens} "
            f"total_duration_ns={extraction_total_duration_ns} "
            f"inference_params={extraction_inference_params}"
        )
        entities = _dedupe_entity_lists(merged_entities)

        entity_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        print(f"[INFO] Raw extraction: {entity_count} entities")

        entities = filter_bamf_addresses(entities)
        entities = filter_non_person_group_labels(entities, text)
        entities = augment_names_from_role_markers(entities, text)
        entities = augment_names_from_person_fields(entities, text)
        entities = filter_non_person_organization_labels(entities)
        entities = _filter_name_artifacts(entities)
        if engine == "qwen_flair":
            flair_names = await _fetch_flair_name_hints(service_url, text, document_type)
            entities = _merge_flair_names(entities, flair_names)
            entities = filter_non_person_group_labels(entities, text)
            entities = filter_non_person_organization_labels(entities)
            entities = _filter_name_artifacts(entities)
        entities = _dedupe_entity_lists(entities)

        filtered_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        print(f"[INFO] After BAMF filter: {filtered_count} entities")
        for key, values in entities.items():
            if values:
                print(f"  {key}: {values}")

        anonymized_text = apply_regex_replacements(text, entities)

        all_addresses = entities.get("streets", []) + entities.get("cities", [])

        print("[SUCCESS] Anonymization completed (local regex)")

        return AnonymizationResult(
            anonymized_text=anonymized_text,
            plaintiff_names=entities.get("names", []),
            birth_dates=entities.get("birth_dates", []),
            addresses=all_addresses,
            confidence=0.95,
            original_text=text,
            processed_characters=len(text),
            extraction_prompt_tokens=extraction_prompt_tokens,
            extraction_completion_tokens=extraction_completion_tokens,
            extraction_total_duration_ns=extraction_total_duration_ns,
            extraction_inference_params=extraction_inference_params,
        )

    except HTTPException:
        raise
    except httpx.TimeoutException:
        print("[ERROR] Entity extraction timeout (>300s)")
        raise HTTPException(
            status_code=504,
            detail="Entity extraction timeout (>300s). Please retry.",
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        message = None
        try:
            err = exc.response.json()
            if isinstance(err, dict):
                message = err.get("detail") or err.get("message")
            else:
                message = str(err)
        except Exception:
            message = exc.response.text or str(exc)

        detail_message = message or f"HTTP {status} from service manager"
        print(f"[ERROR] Entity extraction HTTP error: {status} – {detail_message}")
        raise HTTPException(status_code=status, detail=detail_message)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse LLM entity JSON: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse model response as JSON.",
        )
    except Exception as exc:
        print(f"[ERROR] Anonymization error: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Anonymization error: {exc}",
        )


@router.post("/documents/{document_id}/anonymize")
@limiter.limit("100/hour")
async def anonymize_document_endpoint(
    request: Request,
    document_id: str,
    force: bool = Query(False),
    engine: Optional[str] = Query(None),
    extract_chunk_pages: Optional[int] = Query(None, ge=1, le=50),
    extract_num_ctx: Optional[int] = Query(None, ge=1024, le=131072),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Anonymize a classified document (Anhörung, Bescheid, or Sonstige gespeicherte Quellen)."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = (
        db.query(Document)
        .filter(
            Document.id == doc_uuid,
            Document.owner_id == current_user.id,
            Document.case_id == current_user.active_case_id,
        )
        .first()
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # All document types can be anonymized (names and addresses are extracted)
    resolved_engine = resolve_anonymization_engine(engine)

    if force and document.is_anonymized:
        print(f"[INFO] Force re-anonymization requested for {document.filename}")

    if document.is_anonymized and document.anonymization_metadata and not force:
        # Only path-based anonymized text is supported
        anonymized_text = ""
        path = document.anonymization_metadata.get("anonymized_text_path")
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    anonymized_text = f.read()
            except Exception as e:
                print(f"[ERROR] Failed to read anonymized text from file: {e}")
        else:
            print(f"[WARN] Missing anonymized_text_path for {document.filename}; reprocessing.")
            anonymized_text = ""

        if anonymized_text:
            processed_chars = document.anonymization_metadata.get(
                "processed_characters"
            )
            remaining_chars = document.anonymization_metadata.get(
                "remaining_characters"
            )
            if processed_chars is not None and remaining_chars is not None:
                input_characters = processed_chars + remaining_chars
            else:
                input_characters = len(anonymized_text)

            return {
                "status": "success",
                "anonymized_text": anonymized_text,
                "plaintiff_names": document.anonymization_metadata.get(
                    "plaintiff_names", []
                ),
                "addresses": document.anonymization_metadata.get("addresses", []),
                "confidence": document.anonymization_metadata.get("confidence", 0.0),
                "input_characters": input_characters,
                "processed_characters": processed_chars,
                "remaining_characters": remaining_chars,
                "extraction_prompt_tokens": document.anonymization_metadata.get(
                    "extraction_prompt_tokens"
                ),
                "extraction_completion_tokens": document.anonymization_metadata.get(
                    "extraction_completion_tokens"
                ),
                "extraction_total_duration_ns": document.anonymization_metadata.get(
                    "extraction_total_duration_ns"
                ),
                "extraction_inference_params": document.anonymization_metadata.get(
                    "extraction_inference_params"
                ),
                "cached": True,
                "engine": document.anonymization_metadata.get("engine", resolved_engine),
            }

    pdf_path = document.file_path
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    extracted_text = None
    ocr_used = False

    cached_text = load_document_text(document)
    if document.ocr_applied and cached_text:
        extracted_text = cached_text
        ocr_used = True
        print(
            f"[INFO] Using cached OCR text for {document.filename}: {len(extracted_text)} characters"
        )

    if not extracted_text:
        # Check if we need OCR using shared logic (consistent with classification)
        # We respect the DB flag if it's True, otherwise we verify with the shared check.
        should_use_ocr = document.needs_ocr or check_pdf_needs_ocr(pdf_path)

        if should_use_ocr:
            print(
                f"[INFO] Document needs OCR (flag={document.needs_ocr}). Skipping direct extraction."
            )
            extracted_text = None
        else:
            try:
                extracted_text = extract_pdf_text(
                    pdf_path, max_pages=50, include_page_headers=False
                )
                # Final sanity check: even if check passed, maybe extraction failed or yielded garbage
                if extracted_text and len(extracted_text.strip()) >= 500:
                    print(
                        f"[INFO] Direct text extraction successful: {len(extracted_text)} characters"
                    )
                else:
                    print(
                        f"[INFO] Direct extraction insufficient ({len(extracted_text) if extracted_text else 0} chars), trying OCR..."
                    )
                    extracted_text = None
            except Exception as exc:
                print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
                extracted_text = None

    if not extracted_text:
        extracted_text = await perform_ocr_on_file(pdf_path)
        if extracted_text:
            ocr_used = True
            print(
                f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Could not extract text from PDF. OCR service unavailable. Please ensure home PC OCR service is running.",
            )

    if not extracted_text or len(extracted_text.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail="Insufficient text extracted from PDF. The document may be empty or corrupted.",
        )

    document_type = document.category
    if document_type == "Sonstiges":
        document_type = DocumentCategory.SONSTIGES.value

    text_for_anonymization = extracted_text
    try:
        text_hash = hashlib.sha256(text_for_anonymization.encode("utf-8")).hexdigest()
        text_len = len(text_for_anonymization)
        line_count = text_for_anonymization.count("\n") + 1 if text_for_anonymization else 0
        word_count = len(text_for_anonymization.split())
        null_count = text_for_anonymization.count("\x00")
        non_ascii = sum(1 for ch in text_for_anonymization if ord(ch) > 127)
        print(
            "[INFO] Anonymization payload stats "
            f"(doc={document.filename}, type={document_type}, force={force}): "
            f"chars={text_len}, words={word_count}, lines={line_count}, "
            f"non_ascii={non_ascii}, nulls={null_count}, sha256={text_hash}"
        )
    except Exception as exc:
        print(f"[WARN] Failed to compute anonymization payload stats: {exc}")
    print(
        f"[INFO] Sending {len(text_for_anonymization)} characters to anonymization service"
    )

    result = await anonymize_document_text(
        text_for_anonymization,
        document_type,
        resolved_engine,
        extract_chunk_pages=extract_chunk_pages,
        extract_num_ctx=extract_num_ctx,
    )
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Anonymization service unavailable. Please ensure it is running.",
        )

    anonymized_full_text = result.anonymized_text
    processed_chars = result.processed_characters
    remaining_chars = 0

    # Save anonymized text to file
    anonymized_filename = f"{document.id}.txt"
    anonymized_path = ANONYMIZED_TEXT_DIR / anonymized_filename
    try:
        with open(anonymized_path, "w", encoding="utf-8") as f:
            f.write(anonymized_full_text)
    except Exception as e:
        print(f"[ERROR] Failed to write anonymized text to file: {e}")
        # Fallback? Or raise? For now, we proceed, but metadata might be incomplete if we rely on path.
        # Actually, if write fails, we should probably fail the request or fallback to DB storage.
        # Let's fallback to DB storage implicitly if path is missing, but here we want to enforce file.
        raise HTTPException(
            status_code=500, detail=f"Failed to save anonymized text: {e}"
        )

    store_document_text(document, extracted_text)
    document.is_anonymized = True
    document.ocr_applied = ocr_used
    document.anonymization_metadata = {
        "plaintiff_names": result.plaintiff_names,
        "birth_dates": result.birth_dates,
        "addresses": result.addresses,
        "confidence": result.confidence,
        "anonymized_at": datetime.utcnow().isoformat(),
        "anonymized_text_path": str(anonymized_path),
        "anonymized_excerpt": result.anonymized_text,
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "input_characters": len(extracted_text),
        "ocr_used": ocr_used,
        "engine": resolved_engine,
        "extraction_prompt_tokens": result.extraction_prompt_tokens,
        "extraction_completion_tokens": result.extraction_completion_tokens,
        "extraction_total_duration_ns": result.extraction_total_duration_ns,
        "extraction_inference_params": result.extraction_inference_params,
    }
    document.processing_status = "completed"
    db.commit()

    broadcast_documents_snapshot(db, "anonymize", {"document_id": document_id})

    return {
        "status": "success",
        "anonymized_text": anonymized_full_text,
        "plaintiff_names": result.plaintiff_names,
        "birth_dates": result.birth_dates,
        "addresses": result.addresses,
        "confidence": result.confidence,
        "input_characters": len(extracted_text),
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "extraction_prompt_tokens": result.extraction_prompt_tokens,
        "extraction_completion_tokens": result.extraction_completion_tokens,
        "extraction_total_duration_ns": result.extraction_total_duration_ns,
        "extraction_inference_params": result.extraction_inference_params,
        "ocr_used": ocr_used,
        "cached": False,
        "engine": resolved_engine,
    }


@router.post("/anonymize-file")
@limiter.limit("100/hour")
async def anonymize_uploaded_file(
    request: Request,
    document_type: str = Form(...),
    file: UploadFile = File(...),
    engine: Optional[str] = Query(None),
    extract_chunk_pages: Optional[int] = Query(None, ge=1, le=50),
    extract_num_ctx: Optional[int] = Query(None, ge=1024, le=131072),
    current_user: User = Depends(get_current_active_user),
):
    """Anonymize an uploaded PDF without storing it in the database."""
    sanitized_type = document_type.strip() or "Sonstiges"
    resolved_engine = resolve_anonymization_engine(engine)

    filename = (file.filename or "upload.pdf").strip()
    _, ext = os.path.splitext(filename.lower())
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmpf.write(chunk)
        tmp_path = tmpf.name

    try:
        extracted_text = None
        ocr_used = False

        # Check if we need OCR using shared logic
        should_use_ocr = check_pdf_needs_ocr(tmp_path)

        if not should_use_ocr:
            try:
                extracted_text = extract_pdf_text(
                    tmp_path, max_pages=50, include_page_headers=False
                )
                # Final sanity check: even if check passed, maybe extraction failed or yielded garbage
                if extracted_text and len(extracted_text.strip()) >= 500:
                    print(
                        f"[INFO] Direct text extraction successful: {len(extracted_text)} characters"
                    )
                else:
                    print(
                        f"[INFO] Direct extraction insufficient ({len(extracted_text) if extracted_text else 0} chars), trying OCR..."
                    )
                    extracted_text = None
            except Exception as exc:
                print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
                extracted_text = None

        if not extracted_text:
            extracted_text = await perform_ocr_on_file(tmp_path)
            if extracted_text:
                ocr_used = True
                print(
                    f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Could not extract text from PDF. OCR service unavailable. Please ensure home PC OCR service is running.",
                )

        if not extracted_text or len(extracted_text.strip()) < 100:
            raise HTTPException(
                status_code=422,
                detail="Insufficient text extracted from PDF. The document may be empty or corrupted.",
            )

        text_for_anonymization = extracted_text
        print(
            f"[INFO] Sending uploaded PDF ({len(text_for_anonymization)} chars) to anonymization service"
        )

        result = await anonymize_document_text(
            text_for_anonymization,
            sanitized_type,
            resolved_engine,
            extract_chunk_pages=extract_chunk_pages,
            extract_num_ctx=extract_num_ctx,
        )
        if result is None:
            raise HTTPException(
                status_code=503,
                detail="Anonymization service unavailable. Please ensure it is running.",
            )

        anonymized_full_text = result.anonymized_text
        processed_chars = result.processed_characters
        remaining_chars = 0

        return {
            "status": "success",
            "filename": filename,
            "anonymized_text": anonymized_full_text,
            "plaintiff_names": result.plaintiff_names,
            "birth_dates": result.birth_dates,
            "addresses": result.addresses,
            "confidence": result.confidence,
            "input_characters": len(extracted_text),
            "processed_characters": processed_chars,
            "remaining_characters": remaining_chars,
            "extraction_prompt_tokens": result.extraction_prompt_tokens,
            "extraction_completion_tokens": result.extraction_completion_tokens,
            "extraction_total_duration_ns": result.extraction_total_duration_ns,
            "extraction_inference_params": result.extraction_inference_params,
            "ocr_used": ocr_used,
            "engine": resolved_engine,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


__all__ = ["router"]

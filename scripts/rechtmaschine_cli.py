#!/usr/bin/env python3
"""Thin official CLI wrapper for the Rechtmaschine HTTP API."""

from __future__ import annotations

import argparse
import getpass
import contextlib
import json
import mimetypes
import os
import re
import socket
import time
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, parse, request


DEFAULT_BASE_URL = os.getenv("RECHTMASCHINE_BASE_URL", "https://rechtmaschine.de").rstrip("/")
DEFAULT_TOKEN_PATH = Path(os.getenv("RECHTMASCHINE_TOKEN_PATH", "~/.config/rechtmaschine-cli/token")).expanduser()
DEFAULT_TIMEOUT = float(os.getenv("RECHTMASCHINE_TIMEOUT", "30"))
DEFAULT_RETRIES = int(os.getenv("RECHTMASCHINE_RETRIES", "2"))
DEFAULT_POLL_INTERVAL = float(os.getenv("RECHTMASCHINE_POLL_INTERVAL", "5"))
DEFAULT_FORCE_IPV4 = os.getenv("RECHTMASCHINE_FORCE_IPV4", "1").strip().lower() not in {"0", "false", "no"}
FINAL_JOB_STATES = {"completed", "failed", "cancelled"}
CALLER_CWD = Path(os.getenv("RECHTMASCHINE_CALLER_CWD", os.getcwd())).expanduser().resolve()


class ApiError(RuntimeError):
    """Raised when the API returns a non-2xx response."""


def _resolve_cli_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (CALLER_CWD / path).resolve()


@contextlib.contextmanager
def _prefer_ipv4(enabled: bool):
    if not enabled:
        yield
        return
    original = socket.getaddrinfo

    def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return original(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_only
    try:
        yield
    finally:
        socket.getaddrinfo = original


def _load_token(token_path: Path) -> str:
    env_token = (os.getenv("RECHTMASCHINE_TOKEN") or "").strip()
    if env_token:
        return env_token
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()
    raise ApiError(
        "No token available. Use 'rechtmaschine_cli.py login ...' or set RECHTMASCHINE_TOKEN."
    )


def _save_token(token_path: Path, token: str) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token.strip() + "\n", encoding="utf-8")
    os.chmod(token_path, 0o600)


def _request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    token: Optional[str] = None,
    json_body: Optional[Dict[str, Any]] = None,
    form_body: Optional[Dict[str, str]] = None,
    query: Optional[Dict[str, Any]] = None,
) -> Any:
    url = f"{base_url}{path}"
    if query:
        filtered = {key: value for key, value in query.items() if value is not None}
        if filtered:
            url = f"{url}?{parse.urlencode(filtered, doseq=True)}"

    headers = {
        "Accept": "application/json",
    }
    data: Optional[bytes] = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if json_body is not None:
        data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif form_body is not None:
        data = parse.urlencode(form_body).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    req = request.Request(url, method=method.upper(), headers=headers, data=data)
    last_exc: Optional[Exception] = None
    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            with _prefer_ipv4(DEFAULT_FORCE_IPV4):
                with request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                    raw = resp.read().decode("utf-8")
                    if not raw:
                        return {}
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" in content_type or raw[:1] in {"{", "["}:
                        return json.loads(raw)
                    return {"raw": raw}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ApiError(f"{exc.code} {exc.reason}: {detail}") from exc
        except error.URLError as exc:
            last_exc = exc
            if attempt >= DEFAULT_RETRIES:
                raise ApiError(f"Connection failed: {exc.reason}") from exc
            time.sleep(min(1.5 * (attempt + 1), 5.0))
        except TimeoutError as exc:
            last_exc = exc
            if attempt >= DEFAULT_RETRIES:
                raise ApiError(f"Connection timed out after {DEFAULT_TIMEOUT:.0f}s") from exc
            time.sleep(min(1.5 * (attempt + 1), 5.0))
    raise ApiError(f"Request failed: {last_exc}")


def _request_multipart(
    method: str,
    base_url: str,
    path: str,
    *,
    token: Optional[str] = None,
    files: Dict[str, Path],
    fields: Optional[Dict[str, str]] = None,
) -> Any:
    boundary = "----rechtmaschine-cli-boundary"
    body = bytearray()

    def add_text(name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    def add_file(name: str, path_obj: Path) -> None:
        mime_type = mimetypes.guess_type(str(path_obj))[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{path_obj.name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        body.extend(path_obj.read_bytes())
        body.extend(b"\r\n")

    for key, value in (fields or {}).items():
        add_text(key, value)
    for key, path_obj in files.items():
        add_file(key, path_obj)
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = request.Request(
        f"{base_url}{path}",
        method=method.upper(),
        headers=headers,
        data=bytes(body),
    )
    last_exc: Optional[Exception] = None
    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            with _prefer_ipv4(DEFAULT_FORCE_IPV4):
                with request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                    raw = resp.read().decode("utf-8")
                    if not raw:
                        return {}
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return {"raw": raw}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ApiError(f"{exc.code} {exc.reason}: {detail}") from exc
        except error.URLError as exc:
            last_exc = exc
            if attempt >= DEFAULT_RETRIES:
                raise ApiError(f"Connection failed: {exc.reason}") from exc
            time.sleep(min(1.5 * (attempt + 1), 5.0))
        except TimeoutError as exc:
            last_exc = exc
            if attempt >= DEFAULT_RETRIES:
                raise ApiError(f"Upload timed out after {DEFAULT_TIMEOUT:.0f}s") from exc
            time.sleep(min(1.5 * (attempt + 1), 5.0))
    raise ApiError(f"Upload failed: {last_exc}")


def _read_json_payload(path: str) -> Dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _cases_payload(base_url: str, token: str) -> Dict[str, Any]:
    data = _request_json("GET", base_url, "/cases", token=token)
    if not isinstance(data, dict):
        raise ApiError(f"Unexpected /cases response shape: {type(data).__name__}")
    return data


def _cases_list(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    cases = payload.get("cases") or []
    if not isinstance(cases, list):
        raise ApiError("Unexpected /cases payload: 'cases' is not a list.")
    return [item for item in cases if isinstance(item, dict)]


def _find_case_by_exact_name(payload: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for case in _cases_list(payload):
        if str(case.get("name") or "") == name:
            return case
    return None


def _active_case_id(payload: Dict[str, Any]) -> str:
    case_id = str(payload.get("active_case_id") or "").strip()
    if not case_id:
        raise ApiError("No active Rechtmaschine case is set. Activate a case first or pass --case-id.")
    return case_id


def _resolve_case_id(base_url: str, token: str, case_id: Optional[str]) -> str:
    if case_id:
        return case_id
    return _active_case_id(_cases_payload(base_url, token))


def _documents_payload(base_url: str, token: str, case_id: str) -> Dict[str, Any]:
    data = _request_json("GET", base_url, "/documents", token=token, query={"case_id": case_id})
    if not isinstance(data, dict):
        raise ApiError(f"Unexpected /documents response shape: {type(data).__name__}")
    return data


def _flatten_documents(documents_payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    flattened: list[Dict[str, Any]] = []
    for category, docs in documents_payload.items():
        if not isinstance(docs, list):
            continue
        for item in docs:
            if isinstance(item, dict):
                normalized = dict(item)
                normalized.setdefault("category", category)
                flattened.append(normalized)
    return flattened


def _extract_uploaded_document(response: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(response, dict):
        return None
    candidate = response.get("document")
    if isinstance(candidate, dict):
        return candidate
    if all(key in response for key in ("id", "filename", "category")):
        return response
    return None


def _wait_for_job(
    base_url: str,
    token: str,
    path: str,
    *,
    timeout_seconds: float,
    poll_interval: float,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: Optional[Dict[str, Any]] = None
    while True:
        payload = _request_json("GET", base_url, path, token=token)
        if not isinstance(payload, dict):
            raise ApiError(f"Unexpected job status payload: {type(payload).__name__}")
        last_payload = payload
        status = str(payload.get("status") or "").strip().lower()
        if status in FINAL_JOB_STATES:
            return payload
        if time.monotonic() >= deadline:
            raise ApiError(
                f"Job did not reach a final state within {timeout_seconds:.0f}s. "
                f"Last known status: {status or 'unknown'}"
            )
        time.sleep(poll_interval)


def _read_text_arg(inline: Optional[str], path: Optional[str]) -> Optional[str]:
    """Return text from an inline value or a file path ('-' reads stdin)."""
    if inline is not None:
        return inline
    if path:
        if path == "-":
            return sys.stdin.read()
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return None


def _extract_generated_text(payload: Any) -> str:
    """Pull the generated draft body out of a draft / job-result payload."""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    for key in ("generated_text", "text", "content", "body"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    inner = payload.get("result")
    if isinstance(inner, (dict, str)):
        return _extract_generated_text(inner)
    return ""


def _fetch_draft_text(base_url: str, token: str, draft_id: str, case_id: Optional[str]) -> str:
    """Fetch a stored draft's generated text via GET /drafts/{id}."""
    query = {"case_id": case_id} if case_id else None
    data = _request_json("GET", base_url, f"/drafts/{draft_id}", token=token, query=query)
    text = _extract_generated_text(data)
    if not text.strip():
        raise ApiError(f"Draft {draft_id} has no generated_text to iterate on.")
    return text


def _fetch_generation_result_text(base_url: str, token: str, job_id: str) -> str:
    """Fetch a completed generation job's text via GET /generate/jobs/{id}/result."""
    data = _request_json("GET", base_url, f"/generate/jobs/{job_id}/result", token=token)
    return _extract_generated_text(data)


_TIMESTAMPED_UPLOAD_RE = re.compile(r"^\d{8}_\d{6}_(.+)$")


def _normalize_uploaded_filename(filename: str) -> str:
    name = (filename or "").strip()
    match = _TIMESTAMPED_UPLOAD_RE.match(name)
    if match:
        return match.group(1)
    return name


def _canonical_upload_match_name(filename: str) -> str:
    """Match backend-sanitized upload names without hiding meaningful words."""
    name = _normalize_uploaded_filename(filename)
    name = unicodedata.normalize("NFKC", name).casefold()
    name = re.sub(r"[^\w.]+", " ", name, flags=re.UNICODE)
    return re.sub(r"\s+", " ", name).strip()


def _find_existing_document(
    documents_payload: Dict[str, Any],
    *,
    category: str,
    local_name: str,
) -> Optional[Dict[str, Any]]:
    docs = documents_payload.get(category) or []
    if not isinstance(docs, list):
        return None
    canonical_local_name = _canonical_upload_match_name(local_name)
    for item in docs:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "")
        if (
            filename == local_name
            or _normalize_uploaded_filename(filename) == local_name
            or _canonical_upload_match_name(filename) == canonical_local_name
        ):
            return item
    return None


def _safe_pdf_name_from_title(title: str) -> str:
    safe_name = "".join(c for c in (title or "") if c.isalnum() or c in (" ", "-", "_")).strip()
    if not safe_name:
        safe_name = "research_source"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    return safe_name


def _research_result_payload(base_url: str, token: str, job_id: str) -> Dict[str, Any]:
    data = _request_json("GET", base_url, f"/research/jobs/{job_id}/result", token=token)
    if not isinstance(data, dict):
        raise ApiError(f"Unexpected research result payload: {type(data).__name__}")
    return data


def _probe_pdf_url(url: str) -> Dict[str, Any]:
    req = request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Codex Rechtmaschine CLI)",
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
        },
    )
    with _prefer_ipv4(DEFAULT_FORCE_IPV4):
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
            prefix = resp.read(64)
            content_type = str(resp.headers.get("Content-Type") or "")
            return {
                "content_type": content_type,
                "prefix": prefix,
                "is_pdf": ("pdf" in content_type.lower()) or prefix.startswith(b"%PDF-"),
            }


def cmd_login(args: argparse.Namespace) -> int:
    password = (os.getenv("RECHTMASCHINE_PASSWORD") or "").strip()
    if not password:
        password = getpass.getpass("Rechtmaschine password: ")
    payload = {
        "username": args.username,
        "password": password,
    }
    data = _request_json("POST", args.base_url, "/token", form_body=payload)
    token = str(data.get("access_token") or "").strip()
    if not token:
        raise ApiError("Login succeeded but no access_token was returned.")
    _save_token(args.token_path, token)
    _print({"token_type": data.get("token_type", "bearer"), "token_saved_to": str(args.token_path)})
    return 0


def cmd_whoami(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, "/users/me", token=token))
    return 0


def cmd_cases_list(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, "/cases", token=token))
    return 0


def cmd_cases_create(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    if not args.no_reuse_existing:
        payload = _cases_payload(args.base_url, token)
        existing = _find_case_by_exact_name(payload, args.name)
        if existing is not None:
            case_id = str(existing.get("id") or "").strip()
            if args.activate and case_id:
                _request_json("POST", args.base_url, f"/cases/{case_id}/activate", token=token)
            _print({"reused_existing": True, "case": existing})
            return 0
    payload: Dict[str, Any] = {"name": args.name}
    if args.state_file:
        payload["state"] = _read_json_payload(args.state_file)
    data = _request_json("POST", args.base_url, "/cases", token=token, json_body=payload)
    if args.activate:
        case_id = str((data.get("case") or {}).get("id") or "")
        if case_id:
            _request_json("POST", args.base_url, f"/cases/{case_id}/activate", token=token)
    _print(data)
    return 0


def cmd_cases_activate(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("POST", args.base_url, f"/cases/{args.case_id}/activate", token=token))
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(
        _request_json(
            "GET",
            args.base_url,
            "/workflow/inventory",
            token=token,
            query={"case_id": args.case_id, "draft_limit": args.draft_limit},
        )
    )
    return 0


def _resolve_case_id_by_name(base_url: str, token: str, needle: str) -> str:
    """Resolve a case reference like '044/26' against /cases by name prefix."""
    payload = _cases_payload(base_url, token)
    matches = [
        case
        for case in _cases_list(payload)
        if str(case.get("name", "")).strip().startswith(needle.strip())
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"Case reference '{needle}' matched {len(matches)} cases - use --case-id."
        )
    return str(matches[0]["id"])


def cmd_draft_context(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    case_id = args.case_id
    if not case_id and args.case:
        case_id = _resolve_case_id_by_name(args.base_url, token, args.case)
    payload = {"query": args.query, "case_id": case_id, "rag_limit": args.rag_limit}
    data = _request_json(
        "POST", args.base_url, "/workflow/draft-context", token=token, json_body=payload
    )
    sections = [
        ("KANZLEI-PRÄZEDENZ (RAG)", data.get("rag_block", "")),
        ("GESETZESTEXTE", data.get("statute_block", "")),
        ("FALLGEDÄCHTNIS", data.get("case_memory_block", "")),
        ("STILREGELN", data.get("style_rules", "")),
    ]
    lines = []
    for title, block in sections:
        lines.append(f"## {title}\n")
        lines.append(block.strip() + "\n" if block.strip() else "_(leer)_\n")
    text = "\n".join(lines)
    if args.out:
        out_path = _resolve_cli_path(args.out)
        out_path.write_text(text, encoding="utf-8")
        print(f"Kontext geschrieben: {out_path}")
    else:
        print(text)
    empty = [title for title, block in sections[:3] if not block.strip()]
    if empty:
        print(
            f"[Hinweis] Leere Blöcke: {', '.join(empty)} (RAG-Service/Memory prüfen?)",
            file=sys.stderr,
        )
    return 0


def cmd_verify_facts(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    text_path = _resolve_cli_path(args.text_file)
    text = text_path.read_text(encoding="utf-8")
    sources = []
    for path in args.source_file or []:
        sources.append(_resolve_cli_path(path).read_text(encoding="utf-8"))
    payload = {"text": text, "case_id": args.case_id, "sources": sources}
    data = _request_json(
        "POST", args.base_url, "/workflow/verify-facts", token=token, json_body=payload
    )
    checks = data.get("fact_checks", [])
    corpus_empty = bool(data.get("corpus_empty"))
    if corpus_empty:
        print(
            "[WARNUNG] Kein Prüf-Korpus (weder Fallgedächtnis noch --source-file) - "
            "'Keine Beanstandungen' ist dann bedeutungslos. --case-id oder --source-file "
            "mitgeben.",
            file=sys.stderr,
        )
    if not checks:
        print("Keine Beanstandungen.")
        return 2 if corpus_empty else 0
    for c in checks:
        print(
            f"[{c.get('severity','?').upper():4}] {c.get('type','?'):12} "
            f"{c.get('value','')!r}: {c.get('reason','')}"
        )
    if any(c.get("severity") == "high" for c in checks):
        return 1
    return 2 if corpus_empty else 0


def cmd_documents_list(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = _documents_payload(args.base_url, token, args.case_id)
    if args.flat:
        docs = _flatten_documents(payload)
        if args.category:
            docs = [doc for doc in docs if str(doc.get("category") or "") == args.category]
        _print(docs)
        return 0
    if args.category:
        _print({args.category: payload.get(args.category) or []})
        return 0
    _print(payload)
    return 0


def cmd_documents_upload_direct(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    path_obj = _resolve_cli_path(args.file)
    if not path_obj.is_file():
        raise ApiError(f"File not found: {path_obj}")
    target_case_id = args.case_id
    if target_case_id:
        # The backend accepts an explicit case_id form field, so no need to
        # mutate the user's active case (which drives the live UI selection).
        active_case_id = target_case_id
    else:
        cases_payload = _cases_payload(args.base_url, token)
        active_case_id = _active_case_id(cases_payload)
    if not args.force_upload:
        existing = _find_existing_document(
            _documents_payload(args.base_url, token, active_case_id),
            category=args.category,
            local_name=path_obj.name,
        )
        if existing is not None:
            _print(
                {
                    "skipped_existing": True,
                    "case_id": active_case_id,
                    "category": args.category,
                    "file": str(path_obj),
                    "existing_document": existing,
                }
            )
            return 0
    response = _request_multipart(
        "POST",
        args.base_url,
        "/upload-direct",
        token=token,
        files={"file": path_obj},
        fields=(
            {"category": args.category, "case_id": target_case_id}
            if target_case_id
            else {"category": args.category}
        ),
    )
    uploaded = _extract_uploaded_document(response)
    documents_payload = _documents_payload(args.base_url, token, active_case_id)
    verified = _find_existing_document(
        documents_payload,
        category=args.category,
        local_name=path_obj.name,
    )
    result: Dict[str, Any] = {
        "case_id": active_case_id,
        "category": args.category,
        "file": str(path_obj),
        "response": response,
    }
    if uploaded is not None:
        result["uploaded_document"] = uploaded
    if verified is not None:
        result["verified_document"] = verified
    elif args.require_verify:
        raise ApiError(
            "Upload request returned, but the document could not be verified in the target case/category. "
            "Inspect 'documents list' before retrying."
        )
    _print(result)
    return 0


def cmd_documents_ocr(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("POST", args.base_url, f"/documents/{args.document_id}/ocr", token=token))
    return 0


def _summarize_anonymization_response(document_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    anonymized_text = data.get("anonymized_text") or ""
    summary: Dict[str, Any] = {
        "document_id": document_id,
        "status": data.get("status"),
        "cached": data.get("cached"),
        "engine": data.get("engine"),
        "confidence": data.get("confidence"),
        "ocr_used": data.get("ocr_used"),
        "input_characters": data.get("input_characters"),
        "processed_characters": data.get("processed_characters"),
        "remaining_characters": data.get("remaining_characters"),
        "anonymized_text_length": len(anonymized_text),
        "plaintiff_names_count": len(data.get("plaintiff_names") or []),
        "birth_dates_count": len(data.get("birth_dates") or []),
        "addresses_count": len(data.get("addresses") or []),
    }
    for key in (
        "extraction_prompt_tokens",
        "extraction_completion_tokens",
        "extraction_total_duration_ns",
        "extraction_inference_params",
    ):
        if data.get(key) is not None:
            summary[key] = data.get(key)
    return summary


def cmd_documents_anonymize(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    query = {
        "force": args.force,
        "engine": args.engine,
        "extract_chunk_pages": args.extract_chunk_pages,
        "extract_num_ctx": args.extract_num_ctx,
    }
    result = _request_json(
        "POST",
        args.base_url,
        f"/documents/{args.document_id}/anonymize",
        token=token,
        query=query,
    )
    if not isinstance(result, dict):
        raise ApiError(f"Unexpected anonymization response shape: {type(result).__name__}")

    summary = _summarize_anonymization_response(args.document_id, result)

    if args.output_file:
        output_path = Path(args.output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.chmod(output_path, 0o600)
        summary["output_file"] = str(output_path)

    if args.text_output_file:
        text_output_path = Path(args.text_output_file).expanduser()
        text_output_path.parent.mkdir(parents=True, exist_ok=True)
        text_output_path.write_text(str(result.get("anonymized_text") or ""), encoding="utf-8")
        os.chmod(text_output_path, 0o600)
        summary["text_output_file"] = str(text_output_path)

    if args.include_text:
        _print(result)
    else:
        _print(summary)
    return 0


def cmd_generate_job_submit(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = _read_json_payload(args.payload_file)
    created = _request_json("POST", args.base_url, "/generate/jobs", token=token, json_body=payload)
    if args.wait:
        job_id = str((created or {}).get("id") or "").strip()
        if not job_id:
            raise ApiError("Generation job submission returned no job id.")
        # Emit the id up front so a --wait timeout never loses the running job.
        print(f"[generate-job] job {job_id} submitted; waiting...", file=sys.stderr)
        _print(
            _wait_for_job(
                args.base_url,
                token,
                f"/generate/jobs/{job_id}",
                timeout_seconds=args.wait_timeout,
                poll_interval=args.poll_interval,
            )
        )
        return 0
    _print(created)
    return 0


def cmd_generate_job_iterate(args: argparse.Namespace) -> int:
    """Interaktive Verbesserung ('Text überarbeiten'): refine an existing draft.

    Mirrors the frontend amelioration flow: re-submit the original generation
    payload with ``user_prompt`` set to the change request and ``chat_history``
    carrying the prior turn(s) ([{user: original prompt}, {assistant: prior draft}]).
    The backend detects this ("Amelioration detected") and edits with the prior
    draft in context while keeping the same documents attached.
    """
    token = _load_token(args.token_path)
    payload = dict(_read_json_payload(args.payload_file))

    instruction = _read_text_arg(args.instruction, args.instruction_file)
    if not instruction or not instruction.strip():
        raise ApiError("An amelioration instruction is required (--instruction or --instruction-file).")
    instruction = instruction.strip()

    case_id = args.case_id or payload.get("case_id")

    if args.chat_history_file:
        history = _read_json_payload(args.chat_history_file)
        if not isinstance(history, list):
            raise ApiError("--chat-history-file must contain a JSON array of {role, content} messages.")
    else:
        original_prompt = str(payload.get("user_prompt") or "").strip()
        if not original_prompt:
            raise ApiError(
                "Base payload has no 'user_prompt' to seed chat history. "
                "Pass --chat-history-file to continue an existing history instead."
            )
        if args.prior_draft_file:
            prior = _read_text_arg(None, args.prior_draft_file) or ""
        elif args.from_draft_id:
            prior = _fetch_draft_text(args.base_url, token, args.from_draft_id, case_id)
        else:
            raise ApiError(
                "Provide the prior draft to refine via --prior-draft-file or --from-draft-id "
                "(or pass --chat-history-file to continue a multi-round history)."
            )
        if not prior.strip():
            raise ApiError("Prior draft text is empty; nothing to iterate on.")
        history = [
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": prior},
        ]

    payload["user_prompt"] = instruction
    payload["chat_history"] = history
    if case_id:
        payload["case_id"] = case_id

    created = _request_json("POST", args.base_url, "/generate/jobs", token=token, json_body=payload)
    job_id = str((created or {}).get("id") or "").strip()
    if not args.wait:
        _print(created)
        return 0
    if not job_id:
        raise ApiError("Generation job submission returned no job id.")
    print(f"[iterate] generation job {job_id} submitted; waiting...", file=sys.stderr)
    final = _wait_for_job(
        args.base_url,
        token,
        f"/generate/jobs/{job_id}",
        timeout_seconds=args.wait_timeout,
        poll_interval=args.poll_interval,
    )
    if args.save_history_file:
        try:
            new_text = _fetch_generation_result_text(args.base_url, token, job_id)
            if not new_text.strip():
                raise ApiError("completed job returned no generated_text")
            grown = list(history) + [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": new_text},
            ]
            with open(args.save_history_file, "w", encoding="utf-8") as handle:
                json.dump(grown, handle, ensure_ascii=False, indent=2)
            print(f"[iterate] updated chat history written to {args.save_history_file}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - non-fatal convenience step
            print(f"[iterate] warning: could not save updated history: {exc}", file=sys.stderr)
    _print(final)
    return 0


def cmd_generate_job_status(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/generate/jobs/{args.job_id}", token=token))
    return 0


def cmd_generate_job_result(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/generate/jobs/{args.job_id}/result", token=token))
    return 0


def cmd_query_job_submit(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = _read_json_payload(args.payload_file)
    created = _request_json("POST", args.base_url, "/query-documents/jobs", token=token, json_body=payload)
    if args.wait:
        job_id = str((created or {}).get("id") or "").strip()
        if not job_id:
            raise ApiError("Query job submission returned no job id.")
        _print(
            _wait_for_job(
                args.base_url,
                token,
                f"/query-documents/jobs/{job_id}",
                timeout_seconds=args.wait_timeout,
                poll_interval=args.poll_interval,
            )
        )
        return 0
    _print(created)
    return 0


def cmd_query_job_status(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/query-documents/jobs/{args.job_id}", token=token))
    return 0


def cmd_ocr_job_submit(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    created = _request_json(
        "POST",
        args.base_url,
        "/documents/ocr-jobs",
        token=token,
        json_body={"document_id": args.document_id},
    )
    if args.wait:
        job_id = str((created or {}).get("id") or "").strip()
        if not job_id:
            raise ApiError("OCR job submission returned no job id.")
        _print(
            _wait_for_job(
                args.base_url,
                token,
                f"/documents/ocr-jobs/{job_id}",
                timeout_seconds=args.wait_timeout,
                poll_interval=args.poll_interval,
            )
        )
        return 0
    _print(created)
    return 0


def cmd_anonymize_job_submit(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    body = {"document_id": args.document_id, "force": bool(args.force)}
    if args.engine:
        body["engine"] = args.engine
    if getattr(args, "known_entities_file", None):
        body["known_entities"] = _read_json_payload(args.known_entities_file)
    created = _request_json(
        "POST",
        args.base_url,
        "/documents/anonymize-jobs",
        token=token,
        json_body=body,
    )
    if args.wait:
        job_id = str((created or {}).get("id") or "").strip()
        if not job_id:
            raise ApiError("Anonymize job submission returned no job id.")
        _print(
            _wait_for_job(
                args.base_url,
                token,
                f"/documents/anonymize-jobs/{job_id}",
                timeout_seconds=args.wait_timeout,
                poll_interval=args.poll_interval,
            )
        )
        return 0
    _print(created)
    return 0


def cmd_anonymize_job_status(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/documents/anonymize-jobs/{args.job_id}", token=token))
    return 0


def cmd_ocr_job_status(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/documents/ocr-jobs/{args.job_id}", token=token))
    return 0


def cmd_query_job_result(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/query-documents/jobs/{args.job_id}/result", token=token))
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = _read_json_payload(args.payload_file)
    _print(_request_json("POST", args.base_url, "/research", token=token, json_body=payload))
    return 0


def cmd_research_job_submit(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = _read_json_payload(args.payload_file)
    created = _request_json("POST", args.base_url, "/research/jobs", token=token, json_body=payload)
    if args.wait:
        job_id = str((created or {}).get("id") or "").strip()
        if not job_id:
            raise ApiError("Research job submission returned no job id.")
        _print(
            _wait_for_job(
                args.base_url,
                token,
                f"/research/jobs/{job_id}",
                timeout_seconds=args.wait_timeout,
                poll_interval=args.poll_interval,
            )
        )
        return 0
    _print(created)
    return 0


def cmd_research_job_status(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/research/jobs/{args.job_id}", token=token))
    return 0


def cmd_research_job_result(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/research/jobs/{args.job_id}/result", token=token))
    return 0


def cmd_research_job_ingest_pdfs(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    target_case_id = args.case_id
    if target_case_id:
        _request_json("POST", args.base_url, f"/cases/{target_case_id}/activate", token=token)
    else:
        target_case_id = _active_case_id(_cases_payload(args.base_url, token))

    result_payload = _research_result_payload(args.base_url, token, args.job_id)
    sources = result_payload.get("sources") or []
    if not isinstance(sources, list):
        raise ApiError("Unexpected research result payload: 'sources' is not a list.")

    existing_docs = _documents_payload(args.base_url, token, target_case_id)
    ingested: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []
    errors: list[Dict[str, Any]] = []

    for index, source in enumerate(sources, start=1):
        if args.limit is not None and len(ingested) >= args.limit:
            break
        if not isinstance(source, dict):
            continue

        title = str(source.get("title") or f"research_source_{index}").strip()
        pdf_url = str(source.get("pdf_url") or "").strip()
        source_url = str(source.get("url") or "").strip()

        if not pdf_url and args.require_pdf_url:
            skipped.append({
                "title": title,
                "reason": "missing_pdf_url",
                "url": source_url or None,
            })
            continue

        ingest_url = pdf_url or source_url
        if not ingest_url:
            skipped.append({
                "title": title,
                "reason": "missing_url",
            })
            continue

        expected_name = _safe_pdf_name_from_title(title)
        existing = _find_existing_document(
            existing_docs,
            category=args.category,
            local_name=expected_name,
        )
        if existing is not None and not args.force_upload:
            skipped.append({
                "title": title,
                "reason": "existing_document",
                "existing_document": existing,
            })
            continue

        try:
            probe = _probe_pdf_url(ingest_url)
            if not probe["is_pdf"]:
                errors.append({
                    "title": title,
                    "url": ingest_url,
                    "error": (
                        "remote_url_not_pdf"
                        f" (content_type={probe['content_type']!r}, prefix={probe['prefix'][:20]!r})"
                    ),
                })
                continue
            response = _request_json(
                "POST",
                args.base_url,
                "/documents/from-url",
                token=token,
                json_body={
                    "case_id": target_case_id,
                    "title": title,
                    "url": ingest_url,
                    "category": args.category,
                    "auto_download": True,
                },
            )
            ingested.append({
                "title": title,
                "url": ingest_url,
                "response": response,
            })
            existing_docs = _documents_payload(args.base_url, token, target_case_id)
        except Exception as exc:
            errors.append({
                "title": title,
                "url": ingest_url,
                "error": str(exc),
            })

    _print({
        "case_id": target_case_id,
        "job_id": args.job_id,
        "category": args.category,
        "ingested": ingested,
        "skipped": skipped,
        "errors": errors,
    })
    return 0


def cmd_jlawyer_templates(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, "/jlawyer/templates", token=token, query={"folder": args.folder}))
    return 0


def cmd_jlawyer_resolve_case(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(
        _request_json(
            "GET",
            args.base_url,
            "/workflow/jlawyer/resolve-case",
            token=token,
            query={"reference": args.reference},
        )
    )
    return 0


def cmd_jlawyer_send_draft(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = {
        "draft_id": args.draft_id,
        "case_reference": args.case_reference,
        "template_name": args.template_name,
        "file_name": args.file_name,
        "template_folder": args.template_folder,
    }
    _print(_request_json("POST", args.base_url, "/workflow/jlawyer/send-draft", token=token, json_body=payload))
    return 0


def cmd_api_tokens_list(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, "/api-tokens", token=token))
    return 0


def cmd_api_tokens_create(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = {
        "name": args.name,
        "expires_in_days": args.expires_in_days,
    }
    _print(_request_json("POST", args.base_url, "/api-tokens", token=token, json_body=payload))
    return 0


def cmd_api_tokens_revoke(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("DELETE", args.base_url, f"/api-tokens/{args.token_id}", token=token))
    return 0


def cmd_memory_get(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    case_id = _resolve_case_id(args.base_url, token, args.case_id)
    _print(_request_json("GET", args.base_url, f"/memory/cases/{case_id}", token=token))
    return 0


def cmd_memory_put(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    case_id = _resolve_case_id(args.base_url, token, args.case_id)
    if args.payload_file:
        payload = _read_json_payload(args.payload_file)
    else:
        payload = {
            "overview": args.overview or "",
            "strategy": args.strategy or "",
        }
    _print(_request_json("PUT", args.base_url, f"/memory/cases/{case_id}", token=token, json_body=payload))
    return 0


def cmd_memory_reflect(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    case_id = _resolve_case_id(args.base_url, token, args.case_id)
    payload: Dict[str, Any] = {"case_id": case_id, "trigger": args.trigger}
    if args.document_id:
        payload["document_ids"] = args.document_id
    job = _request_json(
        "POST",
        args.base_url,
        f"/memory/cases/{case_id}/reflect",
        token=token,
        json_body=payload,
    )
    if not args.wait:
        _print(job)
        return 0

    job_id = job.get("job_id")
    deadline = time.monotonic() + args.wait_timeout
    while True:
        status = _request_json(
            "GET",
            args.base_url,
            f"/memory/cases/{case_id}/reflect/{job_id}",
            token=token,
        )
        if status.get("status") in FINAL_JOB_STATES:
            _print(status)
            return 0 if status.get("status") == "completed" else 1
        if time.monotonic() > deadline:
            _print(status)
            print(f"Error: reflection job {job_id} still {status.get('status')} after {args.wait_timeout}s", file=sys.stderr)
            return 1
        time.sleep(DEFAULT_POLL_INTERVAL)


def cmd_memory_proposals_list(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    case_id = _resolve_case_id(args.base_url, token, args.case_id)
    payload = _request_json(
        "GET",
        args.base_url,
        f"/memory/cases/{case_id}/proposals",
        token=token,
        query={"status": args.status},
    )
    if getattr(args, "full", False):
        _print(payload)
        return 0
    proposals = payload.get("proposals", payload) if isinstance(payload, dict) else payload
    compact = []
    for proposal in proposals or []:
        ops = proposal.get("ops") or []
        op_summaries = []
        for op in ops:
            if not isinstance(op, dict):
                continue
            value = op.get("value")
            if isinstance(value, dict):
                value = value.get("name") or json.dumps(value, ensure_ascii=False)
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            if len(text) > 120:
                text = text[:117] + "..."
            op_summaries.append(f"{op.get('op')} {op.get('path')}: {text}")
        compact.append({
            "id": proposal.get("id"),
            "status": proposal.get("status"),
            "target_type": proposal.get("target_type"),
            "expected_version": proposal.get("expected_version"),
            "model": proposal.get("model"),
            "created_at": proposal.get("created_at"),
            "reject_reason": (proposal.get("metadata") or {}).get("reject_reason"),
            "ops": op_summaries,
        })
    _print({"count": len(compact), "proposals": compact})
    return 0


def cmd_memory_proposals_create(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    case_id = _resolve_case_id(args.base_url, token, args.case_id)
    payload = _read_json_payload(args.payload_file)
    _print(
        _request_json(
            "POST",
            args.base_url,
            f"/memory/cases/{case_id}/proposals",
            token=token,
            json_body=payload,
        )
    )
    return 0


def cmd_memory_proposals_accept(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("POST", args.base_url, f"/memory/proposals/{args.proposal_id}/accept", token=token))
    return 0


def cmd_memory_proposals_reject(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    body = {"reason": args.reason} if getattr(args, "reason", None) else None
    _print(
        _request_json(
            "POST",
            args.base_url,
            f"/memory/proposals/{args.proposal_id}/reject",
            token=token,
            json_body=body,
        )
    )
    return 0


def cmd_doktrin_status(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, "/doktrin/status", token=token))
    return 0


def cmd_doktrin_pages(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(
        _request_json(
            "GET", args.base_url, "/doktrin/pages",
            token=token,
            query={"status": args.status, "country": args.country, "q": args.q},
        )
    )
    return 0


def cmd_doktrin_show(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(
        _request_json(
            "GET", args.base_url, f"/doktrin/pages/{args.page_id}",
            token=token,
            query={"include_text": "true"} if args.include_text else None,
        )
    )
    return 0


def cmd_wiki_list(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(
        _request_json(
            "GET", args.base_url, "/wiki/entries",
            token=token, query={"status": args.status},
        )
    )
    return 0


def cmd_wiki_show(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/wiki/entries/{args.entry_id}", token=token))
    return 0


def cmd_wiki_create(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = _read_json_payload(args.payload_file)
    _print(_request_json("POST", args.base_url, "/wiki/entries", token=token, json_body=payload))
    return 0


def cmd_wiki_edit(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = _read_json_payload(args.payload_file)
    _print(_request_json("PUT", args.base_url, f"/wiki/entries/{args.entry_id}", token=token, json_body=payload))
    return 0


def cmd_wiki_accept(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("POST", args.base_url, f"/wiki/entries/{args.entry_id}/accept", token=token))
    return 0


def cmd_wiki_reject(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("POST", args.base_url, f"/wiki/entries/{args.entry_id}/reject", token=token))
    return 0


def cmd_wiki_delete(args: argparse.Namespace) -> int:
    if not args.yes:
        print("Refusing to delete without --yes.", file=sys.stderr)
        return 1
    token = _load_token(args.token_path)
    _print(_request_json("DELETE", args.base_url, f"/wiki/entries/{args.entry_id}", token=token))
    return 0


def cmd_wiki_distill(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    case_id = _resolve_case_id(args.base_url, token, args.case_id)
    job = _request_json("POST", args.base_url, f"/wiki/cases/{case_id}/distill", token=token)
    if not args.wait:
        _print(job)
        return 0
    job_id = job.get("job_id")
    deadline = time.monotonic() + args.wait_timeout
    while True:
        status = _request_json(
            "GET", args.base_url, f"/memory/cases/{case_id}/reflect/{job_id}", token=token,
        )
        if status.get("status") in FINAL_JOB_STATES:
            _print(status)
            return 0 if status.get("status") == "completed" else 1
        if time.monotonic() > deadline:
            _print(status)
            print(f"Error: distill job {job_id} still {status.get('status')} after {args.wait_timeout}s", file=sys.stderr)
            return 1
        time.sleep(DEFAULT_POLL_INTERVAL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thin CLI wrapper for the Rechtmaschine HTTP API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Rechtmaschine base URL")
    parser.add_argument("--token-path", type=Path, default=DEFAULT_TOKEN_PATH, help="Path used to store/read the bearer token")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login = subparsers.add_parser("login", help="Log in and store the JWT locally")
    login.add_argument("username")
    login.set_defaults(func=cmd_login)

    whoami = subparsers.add_parser("whoami", help="Show the current authenticated user")
    whoami.set_defaults(func=cmd_whoami)

    cases = subparsers.add_parser("cases", help="Case operations")
    cases_sub = cases.add_subparsers(dest="cases_command", required=True)
    cases_list = cases_sub.add_parser("list", help="List available cases")
    cases_list.set_defaults(func=cmd_cases_list)
    cases_create = cases_sub.add_parser("create", help="Create a case")
    cases_create.add_argument("name")
    cases_create.add_argument("--state-file", help="JSON file path or - for stdin")
    cases_create.add_argument("--activate", action="store_true", help="Activate the case after creation")
    cases_create.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="Do not reuse an existing case with the exact same name before creating.",
    )
    cases_create.set_defaults(func=cmd_cases_create)
    cases_activate = cases_sub.add_parser("activate", help="Activate a case")
    cases_activate.add_argument("case_id")
    cases_activate.set_defaults(func=cmd_cases_activate)

    inventory = subparsers.add_parser("inventory", help="Fetch case-scoped inventory")
    inventory.add_argument("--case-id", required=True)
    inventory.add_argument("--draft-limit", type=int, default=50)
    inventory.set_defaults(func=cmd_inventory)

    draft_ctx = subparsers.add_parser(
        "draft-context",
        help="Drafting-Kontext wie /generate (RAG, Gesetze, Fallgedächtnis, Stilregeln)",
    )
    draft_ctx.add_argument("--case-id", help="Rechtmaschine case UUID")
    draft_ctx.add_argument("--case", help="Case-Referenz, z.B. '044/26' (Namens-Präfix)")
    draft_ctx.add_argument("--query", required=True, help="Drafting-Auftrag/Thema")
    draft_ctx.add_argument("--rag-limit", type=int, default=None)
    draft_ctx.add_argument("--out", help="Markdown in Datei schreiben statt stdout")
    draft_ctx.set_defaults(func=cmd_draft_context)

    verify_facts_p = subparsers.add_parser(
        "verify-facts", help="Deterministische Fakten-Prüfung eines Entwurfs"
    )
    verify_facts_p.add_argument("--text-file", required=True)
    verify_facts_p.add_argument("--case-id")
    verify_facts_p.add_argument("--source-file", action="append", default=[])
    verify_facts_p.set_defaults(func=cmd_verify_facts)

    documents = subparsers.add_parser("documents", help="Document operations")
    documents_sub = documents.add_subparsers(dest="documents_command", required=True)
    documents_list = documents_sub.add_parser("list", help="List documents for a case")
    documents_list.add_argument("--case-id", required=True)
    documents_list.add_argument("--category", default=None, help="Return only one category")
    documents_list.add_argument("--flat", action="store_true", help="Flatten all categories into a single list")
    documents_list.set_defaults(func=cmd_documents_list)
    documents_upload = documents_sub.add_parser("upload-direct", help="Upload a file directly into the active case")
    documents_upload.add_argument("file")
    documents_upload.add_argument("category")
    documents_upload.add_argument("--case-id", help="Upload into this case (no active-case switch).")
    documents_upload.add_argument(
        "--no-require-verify",
        dest="require_verify",
        action="store_false",
        help="Do not fail if the upload cannot be verified via a follow-up documents list call.",
    )
    documents_upload.set_defaults(require_verify=True)
    documents_upload.add_argument(
        "--force-upload",
        action="store_true",
        help="Upload even if a same-named document already exists in the target category.",
    )
    documents_upload.set_defaults(func=cmd_documents_upload_direct)
    documents_ocr = documents_sub.add_parser("ocr", help="Run OCR for a document in the active case")
    documents_ocr.add_argument("document_id")
    documents_ocr.set_defaults(func=cmd_documents_ocr)
    documents_anonymize = documents_sub.add_parser(
        "anonymize",
        help="Run anonymization for a document in the active case",
    )
    documents_anonymize.add_argument("document_id")
    documents_anonymize.add_argument("--force", action="store_true", help="Re-anonymize even if cached output exists")
    documents_anonymize.add_argument("--engine", help="Anonymization engine override")
    documents_anonymize.add_argument("--extract-chunk-pages", type=int, help="LLM extraction chunk page count")
    documents_anonymize.add_argument("--extract-num-ctx", type=int, help="LLM extraction context size")
    documents_anonymize.add_argument(
        "--output-file",
        help="Write the full JSON response, including anonymized text, to this local file",
    )
    documents_anonymize.add_argument(
        "--text-output-file",
        help="Write only anonymized text to this local file",
    )
    documents_anonymize.add_argument(
        "--include-text",
        action="store_true",
        help="Print the full response, including anonymized text, instead of a safe summary",
    )
    documents_anonymize.set_defaults(func=cmd_documents_anonymize)

    generate_job = subparsers.add_parser("generate-job", help="Generation job operations")
    generate_sub = generate_job.add_subparsers(dest="generate_command", required=True)
    generate_submit = generate_sub.add_parser("submit", help="Submit a generation job from JSON payload")
    generate_submit.add_argument("--payload-file", required=True, help="JSON file path or - for stdin")
    generate_submit.add_argument("--wait", action="store_true", help="Poll until the job reaches a final state")
    generate_submit.add_argument("--wait-timeout", type=float, default=300.0, help="Maximum seconds to wait")
    generate_submit.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="Polling interval in seconds")
    generate_submit.set_defaults(func=cmd_generate_job_submit)
    generate_status = generate_sub.add_parser("status", help="Inspect a generation job")
    generate_status.add_argument("job_id")
    generate_status.set_defaults(func=cmd_generate_job_status)
    generate_result = generate_sub.add_parser("result", help="Fetch generation job result")
    generate_result.add_argument("job_id")
    generate_result.set_defaults(func=cmd_generate_job_result)
    generate_iterate = generate_sub.add_parser(
        "iterate",
        help="Interaktive Verbesserung ('Text überarbeiten'): refine an existing draft via chat_history",
    )
    generate_iterate.add_argument(
        "--payload-file",
        required=True,
        help="The ORIGINAL generation payload JSON (document_type, model, verbosity, "
        "selected_documents, case_id, user_prompt). JSON file path or - for stdin",
    )
    generate_iterate.add_argument("--instruction", help="The revision instruction (the 'Text überarbeiten' prompt)")
    generate_iterate.add_argument(
        "--instruction-file", help="Read the revision instruction from a file (or - for stdin)"
    )
    generate_iterate.add_argument(
        "--prior-draft-file", help="File with the prior generated draft text to refine (seeds chat_history)"
    )
    generate_iterate.add_argument(
        "--from-draft-id", help="Fetch the prior draft text from GET /drafts/{id} (seeds chat_history)"
    )
    generate_iterate.add_argument(
        "--chat-history-file",
        help="Continue an existing multi-round history (JSON array of {role,content}); "
        "overrides --prior-draft-file/--from-draft-id",
    )
    generate_iterate.add_argument(
        "--save-history-file",
        help="After a successful --wait run, write the grown chat history here for the next round",
    )
    generate_iterate.add_argument("--case-id", help="Case id for draft resolution / case scoping")
    generate_iterate.add_argument("--wait", action="store_true", help="Poll until the job reaches a final state")
    generate_iterate.add_argument(
        "--wait-timeout",
        type=float,
        default=900.0,
        help="Maximum seconds to wait (default 900; amelioration with xhigh reasoning can take >10 min)",
    )
    generate_iterate.add_argument(
        "--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="Polling interval in seconds"
    )
    generate_iterate.set_defaults(func=cmd_generate_job_iterate)

    anon_job = subparsers.add_parser("anonymize-job", help="Background anonymization job operations (reload-sicher im job-worker)")
    anon_sub = anon_job.add_subparsers(dest="anon_command", required=True)
    anon_submit = anon_sub.add_parser("submit", help="Submit a background anonymization job for a document")
    anon_submit.add_argument("--document-id", required=True, help="Document UUID")
    anon_submit.add_argument("--force", action="store_true", help="Re-anonymize even if already anonymized")
    anon_submit.add_argument("--engine", help="Anonymization engine override")
    anon_submit.add_argument("--known-entities-file", help="JSON mit bekannten Entitaeten (names, birth_dates, streets, ...): ueberspringt die LLM-Extraktion")
    anon_submit.add_argument("--wait", action="store_true", help="Poll until the job reaches a final state")
    anon_submit.add_argument("--wait-timeout", type=float, default=3600.0, help="Maximum seconds to wait")
    anon_submit.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="Polling interval in seconds")
    anon_submit.set_defaults(func=cmd_anonymize_job_submit)
    anon_status = anon_sub.add_parser("status", help="Inspect an anonymization job")
    anon_status.add_argument("job_id")
    anon_status.set_defaults(func=cmd_anonymize_job_status)

    ocr_job = subparsers.add_parser("ocr-job", help="Background OCR job operations (reload-sicher im job-worker)")
    ocr_sub = ocr_job.add_subparsers(dest="ocr_command", required=True)
    ocr_submit = ocr_sub.add_parser("submit", help="Submit a background OCR job for a document")
    ocr_submit.add_argument("--document-id", required=True, help="Document UUID")
    ocr_submit.add_argument("--wait", action="store_true", help="Poll until the job reaches a final state")
    ocr_submit.add_argument("--wait-timeout", type=float, default=3600.0, help="Maximum seconds to wait")
    ocr_submit.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="Polling interval in seconds")
    ocr_submit.set_defaults(func=cmd_ocr_job_submit)
    ocr_status = ocr_sub.add_parser("status", help="Inspect an OCR job")
    ocr_status.add_argument("job_id")
    ocr_status.set_defaults(func=cmd_ocr_job_status)

    query_job = subparsers.add_parser("query-job", help="Query job operations")
    query_sub = query_job.add_subparsers(dest="query_command", required=True)
    query_submit = query_sub.add_parser("submit", help="Submit a query job from JSON payload")
    query_submit.add_argument("--payload-file", required=True, help="JSON file path or - for stdin")
    query_submit.add_argument("--wait", action="store_true", help="Poll until the job reaches a final state")
    query_submit.add_argument("--wait-timeout", type=float, default=300.0, help="Maximum seconds to wait")
    query_submit.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="Polling interval in seconds")
    query_submit.set_defaults(func=cmd_query_job_submit)
    query_status = query_sub.add_parser("status", help="Inspect a query job")
    query_status.add_argument("job_id")
    query_status.set_defaults(func=cmd_query_job_status)
    query_result = query_sub.add_parser("result", help="Fetch query job result")
    query_result.add_argument("job_id")
    query_result.set_defaults(func=cmd_query_job_result)

    research = subparsers.add_parser("research", help="Run research from JSON payload")
    research.add_argument("--payload-file", required=True, help="JSON file path or - for stdin")
    research.set_defaults(func=cmd_research)

    research_job = subparsers.add_parser("research-job", help="Research job operations")
    research_sub = research_job.add_subparsers(dest="research_command", required=True)
    research_submit = research_sub.add_parser("submit", help="Submit a research job from JSON payload")
    research_submit.add_argument("--payload-file", required=True, help="JSON file path or - for stdin")
    research_submit.add_argument("--wait", action="store_true", help="Poll until the job reaches a final state")
    research_submit.add_argument("--wait-timeout", type=float, default=300.0, help="Maximum seconds to wait")
    research_submit.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="Polling interval in seconds")
    research_submit.set_defaults(func=cmd_research_job_submit)
    research_status = research_sub.add_parser("status", help="Inspect a research job")
    research_status.add_argument("job_id")
    research_status.set_defaults(func=cmd_research_job_status)
    research_result = research_sub.add_parser("result", help="Fetch research job result")
    research_result.add_argument("job_id")
    research_result.set_defaults(func=cmd_research_job_result)
    research_ingest = research_sub.add_parser(
        "ingest-pdfs",
        help="Download direct PDF sources from a completed research job into a case category",
    )
    research_ingest.add_argument("job_id")
    research_ingest.add_argument("--case-id", help="Target case UUID; defaults to the active case")
    research_ingest.add_argument("--category", default="Rechtsprechung", help="Target document category")
    research_ingest.add_argument("--limit", type=int, default=None, help="Maximum number of sources to ingest")
    research_ingest.add_argument(
        "--allow-url-fallback",
        dest="require_pdf_url",
        action="store_false",
        help="Allow ingest from source URL when pdf_url is missing.",
    )
    research_ingest.add_argument(
        "--force-upload",
        action="store_true",
        help="Ingest even if a same-titled PDF already exists in the target category.",
    )
    research_ingest.set_defaults(require_pdf_url=True)
    research_ingest.set_defaults(func=cmd_research_job_ingest_pdfs)

    jlawyer = subparsers.add_parser("jlawyer", help="j-lawyer operations")
    jlawyer_sub = jlawyer.add_subparsers(dest="jlawyer_command", required=True)
    jlawyer_templates = jlawyer_sub.add_parser("templates", help="List available j-lawyer templates")
    jlawyer_templates.add_argument("--folder", default=None)
    jlawyer_templates.set_defaults(func=cmd_jlawyer_templates)
    jlawyer_resolve = jlawyer_sub.add_parser("resolve-case", help="Resolve j-lawyer file number/case reference")
    jlawyer_resolve.add_argument("reference")
    jlawyer_resolve.set_defaults(func=cmd_jlawyer_resolve_case)
    jlawyer_send = jlawyer_sub.add_parser("send-draft", help="Send an existing draft to j-lawyer")
    jlawyer_send.add_argument("--draft-id", required=True)
    jlawyer_send.add_argument("--case-reference", required=True)
    jlawyer_send.add_argument("--template-name", required=True)
    jlawyer_send.add_argument("--file-name", required=True)
    jlawyer_send.add_argument("--template-folder", default=None)
    jlawyer_send.set_defaults(func=cmd_jlawyer_send_draft)

    api_tokens = subparsers.add_parser("api-tokens", help="Personal API token operations")
    api_sub = api_tokens.add_subparsers(dest="api_tokens_command", required=True)
    api_list = api_sub.add_parser("list", help="List API tokens")
    api_list.set_defaults(func=cmd_api_tokens_list)
    api_create = api_sub.add_parser("create", help="Create an API token")
    api_create.add_argument("name")
    api_create.add_argument("--expires-in-days", type=int, default=None)
    api_create.set_defaults(func=cmd_api_tokens_create)
    api_revoke = api_sub.add_parser("revoke", help="Revoke an API token")
    api_revoke.add_argument("token_id")
    api_revoke.set_defaults(func=cmd_api_tokens_revoke)

    memory = subparsers.add_parser("memory", help="Case memory operations")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    memory_get = memory_sub.add_parser("get", help="Fetch case brief and strategy memory")
    memory_get.add_argument("--case-id", help="Case UUID; defaults to the active case")
    memory_get.set_defaults(func=cmd_memory_get)
    memory_put = memory_sub.add_parser("put", help="Manually replace overview/strategy memory text")
    memory_put.add_argument("--case-id", help="Case UUID; defaults to the active case")
    memory_put.add_argument("--overview", default="", help="Curated case brief text")
    memory_put.add_argument("--strategy", default="", help="Curated case strategy text")
    memory_put.add_argument("--payload-file", help="JSON file or - with overview/strategy fields")
    memory_put.set_defaults(func=cmd_memory_put)

    memory_reflect = memory_sub.add_parser(
        "reflect",
        help="Queue a memory-reflection job (build memory from documents, the j-lawyer Akte, or consolidate)",
    )
    memory_reflect.add_argument("--case-id", help="Case UUID; defaults to the active case")
    memory_reflect.add_argument(
        "--trigger",
        choices=["documents", "jlawyer", "consolidate"],
        default="documents",
        help="documents: extract from --document-id; jlawyer: read new docs from the linked j-lawyer Akte; consolidate: merge duplicates in grown memory",
    )
    memory_reflect.add_argument(
        "--document-id",
        action="append",
        help="Document UUID for trigger=documents (repeatable)",
    )
    memory_reflect.add_argument("--wait", action="store_true", help="Poll until the job finishes and print its result")
    memory_reflect.add_argument("--wait-timeout", type=int, default=1800, help="Max seconds to wait with --wait")
    memory_reflect.set_defaults(func=cmd_memory_reflect)

    memory_proposals = memory_sub.add_parser("proposals", help="Reviewable memory update proposals")
    memory_proposals_sub = memory_proposals.add_subparsers(dest="memory_proposals_command", required=True)
    memory_proposals_list = memory_proposals_sub.add_parser("list", help="List memory proposals for a case")
    memory_proposals_list.add_argument("--case-id", help="Case UUID; defaults to the active case")
    memory_proposals_list.add_argument("--status", choices=["pending", "accepted", "rejected", "superseded"], default=None)
    memory_proposals_list.add_argument("--full", action="store_true", help="Print full raw proposals instead of the compact summary")
    memory_proposals_list.set_defaults(func=cmd_memory_proposals_list)
    memory_proposals_create = memory_proposals_sub.add_parser("create", help="Create a memory proposal from JSON")
    memory_proposals_create.add_argument("--case-id", help="Case UUID; defaults to the active case")
    memory_proposals_create.add_argument("--payload-file", required=True, help="JSON file or - for stdin")
    memory_proposals_create.set_defaults(func=cmd_memory_proposals_create)
    memory_proposals_accept = memory_proposals_sub.add_parser("accept", help="Accept a memory proposal")
    memory_proposals_accept.add_argument("proposal_id")
    memory_proposals_accept.set_defaults(func=cmd_memory_proposals_accept)
    memory_proposals_reject = memory_proposals_sub.add_parser("reject", help="Reject a memory proposal")
    memory_proposals_reject.add_argument("proposal_id")
    memory_proposals_reject.add_argument("--reason", help="Optional reject reason (stored in proposal metadata)")
    memory_proposals_reject.set_defaults(func=cmd_memory_proposals_reject)

    wiki = subparsers.add_parser("wiki", help="Muster-Wiki (pattern_wiki) operations")
    wiki_sub = wiki.add_subparsers(dest="wiki_command", required=True)

    wiki_list = wiki_sub.add_parser("list", help="List wiki entries")
    wiki_list.add_argument("--status", choices=["pending", "active", "rejected"], default=None)
    wiki_list.set_defaults(func=cmd_wiki_list)

    wiki_show = wiki_sub.add_parser("show", help="Show one wiki entry")
    wiki_show.add_argument("entry_id")
    wiki_show.set_defaults(func=cmd_wiki_show)

    wiki_create = wiki_sub.add_parser("create", help="Create a curated wiki entry (lands pending)")
    wiki_create.add_argument("--payload-file", required=True, help="JSON file or - for stdin")
    wiki_create.set_defaults(func=cmd_wiki_create)

    wiki_edit = wiki_sub.add_parser("edit", help="Edit a wiki entry from JSON")
    wiki_edit.add_argument("entry_id")
    wiki_edit.add_argument("--payload-file", required=True, help="JSON file or - for stdin")
    wiki_edit.set_defaults(func=cmd_wiki_edit)

    wiki_accept = wiki_sub.add_parser("accept", help="Accept (activate) a wiki entry")
    wiki_accept.add_argument("entry_id")
    wiki_accept.set_defaults(func=cmd_wiki_accept)

    wiki_reject = wiki_sub.add_parser("reject", help="Reject a wiki entry")
    wiki_reject.add_argument("entry_id")
    wiki_reject.set_defaults(func=cmd_wiki_reject)

    wiki_delete = wiki_sub.add_parser("delete", help="Delete a wiki entry")
    wiki_delete.add_argument("entry_id")
    wiki_delete.add_argument("--yes", action="store_true", help="Confirm deletion")
    wiki_delete.set_defaults(func=cmd_wiki_delete)

    wiki_distill = wiki_sub.add_parser("distill", help="Queue pattern distillation from a case")
    wiki_distill.add_argument("--case-id", help="Case UUID; defaults to the active case")
    wiki_distill.add_argument("--wait", action="store_true", help="Poll until the job finishes")
    wiki_distill.add_argument("--wait-timeout", type=int, default=1800, help="Max seconds to wait with --wait")
    wiki_distill.set_defaults(func=cmd_wiki_distill)

    doktrin = subparsers.add_parser(
        "doktrin",
        help="Doktrin layer (wiki.aufentha.lt mirror). Sync runs via "
        "'docker exec rechtmaschine-app python /app/doktrin_sync.py', not HTTP.",
    )
    doktrin_sub = doktrin.add_subparsers(dest="doktrin_command", required=True)

    doktrin_status = doktrin_sub.add_parser("status", help="Sync state and chunk counts")
    doktrin_status.set_defaults(func=cmd_doktrin_status)

    doktrin_pages = doktrin_sub.add_parser("pages", help="List mirrored wiki pages")
    doktrin_pages.add_argument("--status", choices=["active", "thin", "gone"], default=None)
    doktrin_pages.add_argument("--country", default=None)
    doktrin_pages.add_argument("--q", default=None, help="Substring match on page_id/title")
    doktrin_pages.set_defaults(func=cmd_doktrin_pages)

    doktrin_show = doktrin_sub.add_parser("show", help="Show one mirrored page")
    doktrin_show.add_argument("page_id")
    doktrin_show.add_argument("--include-text", action="store_true", help="Include the cleaned page text")
    doktrin_show.set_defaults(func=cmd_doktrin_show)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except ApiError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

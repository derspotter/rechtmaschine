#!/usr/bin/env python3
"""Thin official CLI wrapper for the Rechtmaschine HTTP API."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, parse, request


DEFAULT_BASE_URL = os.getenv("RECHTMASCHINE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TOKEN_PATH = Path(os.getenv("RECHTMASCHINE_TOKEN_PATH", "~/.config/rechtmaschine-cli/token")).expanduser()


class ApiError(RuntimeError):
    """Raised when the API returns a non-2xx response."""


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
    try:
        with request.urlopen(req) as resp:
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
        raise ApiError(f"Connection failed: {exc.reason}") from exc


def _read_json_payload(path: str) -> Dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


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


def cmd_generate_job_submit(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    payload = _read_json_payload(args.payload_file)
    _print(_request_json("POST", args.base_url, "/generate/jobs", token=token, json_body=payload))
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
    _print(_request_json("POST", args.base_url, "/query-documents/jobs", token=token, json_body=payload))
    return 0


def cmd_query_job_status(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/query-documents/jobs/{args.job_id}", token=token))
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
    _print(_request_json("POST", args.base_url, "/research/jobs", token=token, json_body=payload))
    return 0


def cmd_research_job_status(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/research/jobs/{args.job_id}", token=token))
    return 0


def cmd_research_job_result(args: argparse.Namespace) -> int:
    token = _load_token(args.token_path)
    _print(_request_json("GET", args.base_url, f"/research/jobs/{args.job_id}/result", token=token))
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

    inventory = subparsers.add_parser("inventory", help="Fetch case-scoped inventory")
    inventory.add_argument("--case-id", required=True)
    inventory.add_argument("--draft-limit", type=int, default=50)
    inventory.set_defaults(func=cmd_inventory)

    generate_job = subparsers.add_parser("generate-job", help="Generation job operations")
    generate_sub = generate_job.add_subparsers(dest="generate_command", required=True)
    generate_submit = generate_sub.add_parser("submit", help="Submit a generation job from JSON payload")
    generate_submit.add_argument("--payload-file", required=True, help="JSON file path or - for stdin")
    generate_submit.set_defaults(func=cmd_generate_job_submit)
    generate_status = generate_sub.add_parser("status", help="Inspect a generation job")
    generate_status.add_argument("job_id")
    generate_status.set_defaults(func=cmd_generate_job_status)
    generate_result = generate_sub.add_parser("result", help="Fetch generation job result")
    generate_result.add_argument("job_id")
    generate_result.set_defaults(func=cmd_generate_job_result)

    query_job = subparsers.add_parser("query-job", help="Query job operations")
    query_sub = query_job.add_subparsers(dest="query_command", required=True)
    query_submit = query_sub.add_parser("submit", help="Submit a query job from JSON payload")
    query_submit.add_argument("--payload-file", required=True, help="JSON file path or - for stdin")
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
    research_submit.set_defaults(func=cmd_research_job_submit)
    research_status = research_sub.add_parser("status", help="Inspect a research job")
    research_status.add_argument("job_id")
    research_status.set_defaults(func=cmd_research_job_status)
    research_result = research_sub.add_parser("result", help="Fetch research job result")
    research_result.add_argument("job_id")
    research_result.set_defaults(func=cmd_research_job_result)

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

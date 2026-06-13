# Design: Muster-Wiki management CLI + create endpoint

Date: 2026-06-13
Status: approved by Jay

## Purpose

Give the terminal a way to manage the Rechtmaschine Muster-Wiki (the anonymized
cross-case `pattern_wiki_entries` store): list, show, create, edit, accept,
reject, delete entries, and trigger distillation from a case. Today a curated
entry had to be inserted directly into the database because no API/CLI path
exists; this closes that gap.

## Background (verified)

- `app/endpoints/pattern_wiki.py` exists, router prefix `/wiki`, model
  `PatternWikiEntry` (`app/models.py`): `id, owner_id, scope(private|firm),
  status(pending|active|rejected), fingerprint(JSONB), tags(JSONB), title,
  summary, argument_patterns, risk_patterns, evidence_patterns,
  recommended_next_steps, confidence, model, reviewed_by, last_used_at,
  created_at, updated_at`. Matching/injection is by `fingerprint` + `tags`
  (no embedding column). Only `active` entries are injected into case contexts.
- Existing routes: `GET /wiki/entries` (status filter), `POST /wiki/cases/{id}/distill`,
  `POST /wiki/entries/{id}/accept`, `POST /wiki/entries/{id}/reject`,
  `PUT /wiki/entries/{id}`, `DELETE /wiki/entries/{id}`. There is **no** create
  or get-single endpoint.
- The CLI `scripts/rechtmaschine_cli.py` is a thin HTTP-API client: helpers
  `_request_json(method, base_url, path, token=, json_body=, query=)`, `_print`,
  `_load_token`, `_resolve_case_id`, `_read_json_payload`; commands are
  `cmd_<group>_<action>` registered via subparsers with `set_defaults(func=...)`.
  The `memory` group (get/put/reflect/proposals list/create/accept/reject) is
  the closest template; `cmd_memory_reflect` shows the `--wait` job-polling
  pattern to reuse for `distill`.

## Backend changes (`app/endpoints/pattern_wiki.py`)

1. `POST /wiki/entries` — create a curated entry.
   - Request model `PatternWikiCreateRequest`: `title: str` (required, non-empty);
     optional `summary: str`, `tags: list[str]`, `fingerprint: dict`,
     `argument_patterns / risk_patterns / evidence_patterns /
     recommended_next_steps: list[str]`, `scope: Literal["private","firm"]="firm"`,
     `confidence: float|None`, `model: str|None`.
   - Server forces `status="pending"`, sets `owner_id=current_user.id`,
     `created_at/updated_at=now`. Reject empty/whitespace title with HTTP 422.
   - Returns the created entry via `entry.to_dict()`, HTTP 201.
2. `GET /wiki/entries/{entry_id}` — return one entry (`to_dict()`), HTTP 404 if
   not found or not visible to the user (mirror the scoping the list endpoint
   already applies).

No change to existing routes.

## CLI changes (`scripts/rechtmaschine_cli.py`)

New top-level `wiki` subcommand group (mirrors `memory`):

| Command | Method/path |
|---|---|
| `wiki list [--status pending\|active\|rejected]` | GET /wiki/entries (query `status`) |
| `wiki show ENTRY_ID` | GET /wiki/entries/{id} |
| `wiki create --payload-file F` | POST /wiki/entries (body = JSON file, `-` = stdin) |
| `wiki edit ENTRY_ID --payload-file F` | PUT /wiki/entries/{id} |
| `wiki accept ENTRY_ID` | POST /wiki/entries/{id}/accept |
| `wiki reject ENTRY_ID` | POST /wiki/entries/{id}/reject |
| `wiki delete ENTRY_ID --yes` | DELETE /wiki/entries/{id} (refuse without `--yes`) |
| `wiki distill --case-id ID [--wait] [--wait-timeout N]` | POST /wiki/cases/{id}/distill, reusing the reflect-job poll loop |

- Output: JSON via `_print`, consistent with the rest of the CLI.
- `create`/`edit` payloads read with the existing `_read_json_payload`.
- `delete` prints an error and exits non-zero unless `--yes` is given.
- `distill` resolves the case id with `_resolve_case_id` and, with `--wait`,
  polls the reflection job exactly like `cmd_memory_reflect`.

## Skill docs

- `~/.codex/skills/rechtmaschine/SKILL.md`: add a "Muster-Wiki" section listing
  the `wiki` commands and the create-payload shape.
- `~/.codex/skills/rechtmaschine-memory/SKILL.md`: replace the stale
  "Pattern wiki ... is not live yet; do not put reusable cross-case doctrine"
  note with the live workflow (curate via `wiki create`, or distill a case;
  entries land `pending` → review → `active`).

## Testing

The app's existing tests (`tests/test_generate_flow.py` …) use `TestClient(app)`
with `app.dependency_overrides[get_db]` but a no-op `DummyDB` and faked services —
there is **no conftest, no real-DB fixture, and no auth fixture**. Since the wiki
endpoints actually persist to the DB, the plan adds a small reusable fixture
(`tests/test_pattern_wiki_api.py`, self-contained — no shared conftest required):

- Override `get_db` with a real **in-memory SQLite** session
  (`create_engine("sqlite://")`, `Base.metadata.create_all` so
  `pattern_wiki_entries` exists), and override `get_current_active_user` with a
  fake user (fixed `owner_id`). Both via `app.dependency_overrides`.
- Tests: `POST /wiki/entries` → 201, persisted row has `status="pending"`,
  `owner_id` = the fake user, echoed fields match; empty/whitespace title → 422;
  `GET /wiki/entries/{id}` → 200 with the entry, random UUID → 404.

- CLI: live smoke against the running container (token at the configured
  `--token-path`) — `wiki list`, `wiki create` a throwaway entry, `wiki show`,
  `wiki edit`, `wiki reject`, `wiki delete --yes`, confirm gone. Existing entries
  (`fac0d2a5…` pending, the rejected Balulov one) serve as read fixtures. The
  throwaway entry is always cleaned up (`delete --yes`) at the end.

## Out of scope

- No changes to distillation logic, fingerprint/matching, or injection.
- No bulk operations; one entry id per accept/reject/edit/delete/show call.
- No new auth/scoping model beyond what the list endpoint already enforces.

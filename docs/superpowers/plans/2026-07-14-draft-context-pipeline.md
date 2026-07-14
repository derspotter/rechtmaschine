# Draft-Context Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give direct in-terminal drafting the same context enrichment `/generate` has: two app endpoints (`/workflow/draft-context`, `/workflow/verify-facts`), two `rechtmaschine-cli` subcommands, and the `gerichtsrubrum` skill renamed to `drafting` with a mandatory pipeline section.

**Architecture:** New pure-assembly module `app/draft_context.py` (unit-tested with stub functions, no DB), thin endpoints in `app/endpoints/workflow.py` that inject the exact same builder functions `/generate` uses, CLI wrappers via the existing `_request_json` helper. Skill rename happens in the canonical `~/.codex/skills` repo with all cross-references updated.

**Tech Stack:** FastAPI (existing app), pytest (existing `tests/`), argparse CLI (`scripts/rechtmaschine_cli.py`).

**Spec:** `docs/superpowers/specs/2026-07-14-draft-context-pipeline-design.md`

## Global Constraints

- Every context block degrades to `""` on failure — the endpoints must never 500 because RAG/memory/statute lookup is down (same semantics as `/generate`).
- `style_rules` come from the EXISTING constant `NEUTRAL_LEGAL_TONE_RULES` in `app/endpoints/generation.py:128` — import it, never copy the string.
- App code is volume-mounted with uvicorn reload: NO container restart, NEVER `docker restart rechtmaschine-job-worker`.
- Tests in this repo are pure-function tests (no TestClient, no DB fixtures) — follow that pattern.
- Skills: `~/.codex/skills` is canonical (a git repo); `~/.claude/skills/<name>` are per-skill symlinks into it. Stage only explicit paths, never `git add -A`.
- The plan's smoke tests run against the hosted app via the CLI wrapper `/home/jay/.codex/skills/rechtmaschine/scripts/rechtmaschine-cli` (auth already configured). `RAG_RETRIEVAL_ENABLED=true` is set in `app/.env` (verified 14.07.).

---

### Task 1: `app/draft_context.py` — assembly + verify helpers

**Files:**
- Create: `/var/opt/docker/rechtmaschine/app/draft_context.py`
- Test: `/var/opt/docker/rechtmaschine/tests/test_draft_context.py`

**Interfaces:**
- Produces: `assemble_draft_context(query: str, *, rag_block_fn, statute_block_fn, memory_block_fn, style_rules: str) -> dict` returning keys `rag_block`, `statute_block`, `case_memory_block`, `style_rules` (all `str`).
- Produces: `verify_facts_with_sources(text: str, memory_text: str = "", sources: tuple = ()) -> dict` returning the unchanged `citation_verifier.verify_facts` dict (`{"fact_checks": [...], "fact_summary": {...}}`).

- [ ] **Step 1: Write the failing tests**

```python
# /var/opt/docker/rechtmaschine/tests/test_draft_context.py
"""Draft-context assembly: /generate-parity blocks for direct drafting.
Pure functions only — builders are injected, no DB/network."""
from draft_context import assemble_draft_context, verify_facts_with_sources


def test_assemble_returns_all_four_keys():
    out = assemble_draft_context(
        "gewöhnlicher Aufenthalt",
        rag_block_fn=lambda: "RAGBLOCK\n",
        statute_block_fn=lambda: "STATUTES\n",
        memory_block_fn=lambda: "MEMORY\n",
        style_rules="RULES",
    )
    assert out == {
        "rag_block": "RAGBLOCK\n",
        "statute_block": "STATUTES\n",
        "case_memory_block": "MEMORY\n",
        "style_rules": "RULES",
    }


def test_assemble_degrades_failing_block_to_empty_string():
    def boom():
        raise RuntimeError("rag store down")
    out = assemble_draft_context(
        "q", rag_block_fn=boom, statute_block_fn=lambda: None,
        memory_block_fn=lambda: "", style_rules="RULES",
    )
    assert out["rag_block"] == ""
    assert out["statute_block"] == ""      # None normalizes to ""
    assert out["case_memory_block"] == ""
    assert out["style_rules"] == "RULES"   # style rules are static, never degraded


def test_verify_facts_flags_unsourced_date_and_az():
    result = verify_facts_with_sources(
        "Mit Bescheid vom 03.02.2026 (Az. 12 K 345/26) wurde abgelehnt.",
        memory_text="", sources=(),
    )
    high = {c["type"] for c in result["fact_checks"] if c["severity"] == "high"}
    assert "date" in high
    assert "aktenzeichen" in high


def test_verify_facts_passes_sourced_facts():
    result = verify_facts_with_sources(
        "Mit Bescheid vom 03.02.2026 wurde abgelehnt.",
        memory_text="",
        sources=("Der Bescheid datiert vom 03.02.2026.",),
    )
    assert [c for c in result["fact_checks"] if c["severity"] == "high"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /var/opt/docker/rechtmaschine && python -m pytest tests/test_draft_context.py -v`
(use the same interpreter the existing tests run with — check `tests/conftest.py` for the `sys.path` bootstrap; if the host python lacks app deps, run inside the app container: `docker compose exec app python -m pytest tests/test_draft_context.py -v`)
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'draft_context'`

- [ ] **Step 3: Write the implementation**

```python
# /var/opt/docker/rechtmaschine/app/draft_context.py
"""Draft-context assembly for /workflow/draft-context and /workflow/verify-facts.

Gives direct (terminal-side) drafting the same context enrichment /generate
injects. The endpoint layer passes in the exact builder callables /generate
uses; this module only handles composition and degradation, so it stays
unit-testable without DB or network."""
from __future__ import annotations

from typing import Callable, Iterable


def _safe_block(fn: Callable[[], str], label: str) -> str:
    """Run one block builder; any failure degrades to '' (same semantics as
    the inline try/excepts in /generate — context must never break drafting)."""
    try:
        return fn() or ""
    except Exception as exc:  # noqa: BLE001 — degradation is the contract
        print(f"[WARN] draft-context block '{label}' failed: {exc}")
        return ""


def assemble_draft_context(
    query: str,
    *,
    rag_block_fn: Callable[[], str],
    statute_block_fn: Callable[[], str],
    memory_block_fn: Callable[[], str],
    style_rules: str,
) -> dict:
    """Compose the four /generate-parity context blocks for `query`."""
    return {
        "rag_block": _safe_block(rag_block_fn, "rag"),
        "statute_block": _safe_block(statute_block_fn, "statutes"),
        "case_memory_block": _safe_block(memory_block_fn, "case_memory"),
        "style_rules": style_rules,
    }


def verify_facts_with_sources(
    text: str, memory_text: str = "", sources: Iterable[str] = ()
) -> dict:
    """verify_facts for terminal drafts: plain source strings instead of the
    selected_documents structure. _fact_corpus concatenates memory + document
    pages, so folding the sources into memory_text is equivalent."""
    from citation_verifier import verify_facts

    corpus = "\n".join([memory_text or "", *[s or "" for s in sources]])
    return verify_facts(text, {}, memory_text=corpus)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same command as Step 2. Expected: 4 passed. Also run `python -m pytest tests/test_citation_verifier.py -q` (must stay green — we call into it).

- [ ] **Step 5: Commit**

```bash
cd /var/opt/docker/rechtmaschine
git add app/draft_context.py tests/test_draft_context.py
git commit -m "feat: draft-context assembly module (/generate parity for direct drafting)"
```

---

### Task 2: `rag_limit` passthrough in `_rag_block_for_generation`

**Files:**
- Modify: `/var/opt/docker/rechtmaschine/app/endpoints/generation.py:75-96` (function `_rag_block_for_generation`)

**Interfaces:**
- Produces: `_rag_block_for_generation(db, current_user, target_case_id, user_prompt: str, collect=None, limit: int | None = None) -> str` — existing callers (line ~561) keep working unchanged (new kwarg has a default).
- Consumes: `rag_context.build_rag_block(query, owner_id, case_name=None, limit=None, collect=None)` — already accepts `limit`.

- [ ] **Step 1: Add the kwarg**

In `app/endpoints/generation.py`, change the signature and the `build_rag_block` call:

```python
def _rag_block_for_generation(db, current_user, target_case_id, user_prompt: str, collect=None, limit=None) -> str:
```

and

```python
        return build_rag_block(
            user_prompt, str(current_user.id), case_name=case_name, collect=collect, limit=limit
        )
```

(only these two lines change — the body in between stays as-is).

- [ ] **Step 2: Verify nothing broke**

Run: `cd /var/opt/docker/rechtmaschine && python -c "import ast; ast.parse(open('app/endpoints/generation.py').read())" && grep -rn "_rag_block_for_generation(" app/ | grep -v "def _rag"`
Expected: syntax OK; the one existing call site (generation.py ~561) passes positional/keyword args unaffected by the new default.

- [ ] **Step 3: Commit**

```bash
git add app/endpoints/generation.py
git commit -m "feat: optional limit passthrough for generation RAG block"
```

---

### Task 3: endpoints `POST /workflow/draft-context` + `POST /workflow/verify-facts`

**Files:**
- Modify: `/var/opt/docker/rechtmaschine/app/endpoints/workflow.py` (append after the existing endpoints; extend the `from .generation import (...)` block)

**Interfaces:**
- Consumes: `assemble_draft_context`, `verify_facts_with_sources` (Task 1), `_rag_block_for_generation(..., limit=)` (Task 2), `NEUTRAL_LEGAL_TONE_RULES` (existing constant `app/endpoints/generation.py:128`), `legal_context.build_statute_block(text, ...) -> str`, `agent_memory_service.get_case_memory_prompt_context(db, current_user, case_id, include_strategy=True, max_chars=5000, collect=None) -> str`.
- Produces (HTTP): `POST /workflow/draft-context` body `{"query": str, "case_id": str|null, "rag_limit": int|null}` → `{"rag_block","statute_block","case_memory_block","style_rules","grounding"}`; `POST /workflow/verify-facts` body `{"text": str, "case_id": str|null, "sources": [str]}` → verify_facts dict.

- [ ] **Step 1: Add imports and request models**

In `app/endpoints/workflow.py`, extend the existing `.generation` import block with `NEUTRAL_LEGAL_TONE_RULES` and `_rag_block_for_generation`, and add near the other imports:

```python
from typing import List

from pydantic import BaseModel

from draft_context import assemble_draft_context, verify_facts_with_sources

try:
    from legal_context import build_statute_block
except ImportError:
    build_statute_block = None

try:
    from agent_memory_service import get_case_memory_prompt_context
except ImportError:
    get_case_memory_prompt_context = None


class DraftContextRequest(BaseModel):
    query: str
    case_id: Optional[str] = None
    rag_limit: Optional[int] = None


class VerifyFactsRequest(BaseModel):
    text: str
    case_id: Optional[str] = None
    sources: List[str] = []
```

- [ ] **Step 2: Add the two endpoints (append at end of file)**

```python
@router.post("/draft-context")
@limiter.limit("240/hour")
async def workflow_draft_context(
    request: Request,
    body: DraftContextRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """The /generate context blocks (RAG precedent, statutes, case memory,
    style rules) as JSON, for terminal-side drafting. Blocks degrade to ''."""
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="query must be non-empty")
    grounding: dict = {}

    def _memory_block() -> str:
        if not (body.case_id and get_case_memory_prompt_context):
            return ""
        text = get_case_memory_prompt_context(
            db, current_user, body.case_id, collect=grounding
        )
        return f"KOMPAKTES FALLGEDÄCHTNIS:\n{text}\n\n" if text else ""

    payload = assemble_draft_context(
        body.query,
        rag_block_fn=lambda: _rag_block_for_generation(
            db, current_user, body.case_id, body.query,
            collect=grounding, limit=body.rag_limit,
        ),
        statute_block_fn=lambda: (
            build_statute_block(body.query) if build_statute_block else ""
        ),
        memory_block_fn=_memory_block,
        style_rules=NEUTRAL_LEGAL_TONE_RULES,
    )
    payload["grounding"] = grounding
    return payload


@router.post("/verify-facts")
@limiter.limit("240/hour")
async def workflow_verify_facts(
    request: Request,
    body: VerifyFactsRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Deterministic date/Aktenzeichen/amount check of a draft against case
    memory + provided source texts (citation_verifier.verify_facts)."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    memory_text = ""
    if body.case_id and get_case_memory_prompt_context:
        try:
            memory_text = get_case_memory_prompt_context(db, current_user, body.case_id)
        except Exception as exc:  # noqa: BLE001 — memory absence must not block the check
            print(f"[WARN] verify-facts memory load failed: {exc}")
    return verify_facts_with_sources(body.text, memory_text, body.sources)
```

- [ ] **Step 3: Syntax check + live reload check**

Run: `cd /var/opt/docker/rechtmaschine && python -c "import ast; ast.parse(open('app/endpoints/workflow.py').read())" && sleep 3 && docker compose logs app --since 1m 2>&1 | tail -5`
Expected: no syntax error; uvicorn reload line, no traceback. (If `Optional` is unimported in workflow.py it already is — line 2.)

- [ ] **Step 4: Smoke via curl through the CLI's auth**

Run (token via CLI login already stored; simplest smoke through the CLI once Task 4 lands — here just confirm the route exists):
`curl -s -o /dev/null -w '%{http_code}\n' -X POST https://rechtmaschine.de/workflow/draft-context -H 'Content-Type: application/json' -d '{}'`
Expected: `401` (auth required) — NOT `404`/`405`.

- [ ] **Step 5: Commit**

```bash
git add app/endpoints/workflow.py
git commit -m "feat: /workflow/draft-context + /workflow/verify-facts endpoints"
```

---

### Task 4: CLI subcommands `draft-context` and `verify-facts`

**Files:**
- Modify: `/var/opt/docker/rechtmaschine/scripts/rechtmaschine_cli.py` (subparsers block starts line ~1342; add handlers next to the other `cmd_*` functions; wire into the command dispatch)

**Interfaces:**
- Consumes: `_request_json(method, path, ...)` (line ~77, existing auth/retry helper), the two endpoints from Task 3.
- Produces (shell): `rechtmaschine-cli draft-context [--case-id UUID | --case "044/26 …"] --query "…" [--rag-limit N] [--out ctx.md]` and `rechtmaschine-cli verify-facts --text-file draft.txt [--case-id UUID] [--source-file f.txt]...` (exit 1 on high-severity findings).

- [ ] **Step 1: Add the handlers**

```python
def _resolve_case_id_by_name(args, needle: str) -> str:
    """Resolve a case reference like '044/26' against /cases by name prefix."""
    cases = _request_json("GET", "/cases", args)
    matches = [c for c in cases if str(c.get("name", "")).strip().startswith(needle.strip())]
    if len(matches) != 1:
        raise SystemExit(
            f"Case reference '{needle}' matched {len(matches)} cases - use --case-id."
        )
    return matches[0]["id"]


def cmd_draft_context(args) -> int:
    case_id = args.case_id
    if not case_id and args.case:
        case_id = _resolve_case_id_by_name(args, args.case)
    payload = {"query": args.query, "case_id": case_id, "rag_limit": args.rag_limit}
    data = _request_json("POST", "/workflow/draft-context", args, json_body=payload)
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
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Kontext geschrieben: {args.out}")
    else:
        print(text)
    empty = [t for t, b in sections[:3] if not b.strip()]
    if empty:
        print(f"[Hinweis] Leere Blöcke: {', '.join(empty)} (RAG-Service/Memory prüfen?)",
              file=sys.stderr)
    return 0


def cmd_verify_facts(args) -> int:
    with open(args.text_file, encoding="utf-8") as fh:
        text = fh.read()
    sources = []
    for path in args.source_file or []:
        with open(path, encoding="utf-8") as fh:
            sources.append(fh.read())
    payload = {"text": text, "case_id": args.case_id, "sources": sources}
    data = _request_json("POST", "/workflow/verify-facts", args, json_body=payload)
    checks = data.get("fact_checks", [])
    if not checks:
        print("Keine Beanstandungen.")
        return 0
    for c in checks:
        print(f"[{c.get('severity','?').upper():4}] {c.get('type','?'):12} "
              f"{c.get('value','')!r}: {c.get('reason','')}")
    return 1 if any(c.get("severity") == "high" for c in checks) else 0
```

(Adapt the exact `_request_json` call signature to the file's convention — check one existing handler like the `inventory` command and mirror how it passes auth/base-url args and JSON bodies.)

- [ ] **Step 2: Register the subparsers**

In the subparser block (after `inventory`, ~line 1373):

```python
    draft_ctx = subparsers.add_parser(
        "draft-context",
        help="Drafting-Kontext wie /generate (RAG, Gesetze, Fallgedächtnis, Stilregeln)",
    )
    draft_ctx.add_argument("--case-id", help="Rechtmaschine case UUID")
    draft_ctx.add_argument("--case", help="Case-Referenz, z.B. '044/26' (Namens-Präfix)")
    draft_ctx.add_argument("--query", required=True, help="Drafting-Auftrag/Thema")
    draft_ctx.add_argument("--rag-limit", type=int, default=None)
    draft_ctx.add_argument("--out", help="Markdown in Datei schreiben statt stdout")

    verify_facts_p = subparsers.add_parser(
        "verify-facts", help="Deterministische Fakten-Prüfung eines Entwurfs"
    )
    verify_facts_p.add_argument("--text-file", required=True)
    verify_facts_p.add_argument("--case-id")
    verify_facts_p.add_argument("--source-file", action="append", default=[])
```

and wire both into the command dispatch the same way the neighboring commands are dispatched.

- [ ] **Step 3: Smoke test both commands**

Run:
```bash
CLI=/home/jay/.codex/skills/rechtmaschine/scripts/rechtmaschine-cli
$CLI draft-context --query "gewöhnlicher Aufenthalt § 10 StAG Auslandsstudium" | head -30
printf 'Mit Bescheid vom 09.09.2099 (Az. 99 K 999/99) ...\n' > /tmp/claude-1000/-home-jay/*/scratchpad/vf_smoke.txt 2>/dev/null || printf 'Mit Bescheid vom 09.09.2099 (Az. 99 K 999/99) ...\n' > /tmp/vf_smoke.txt
$CLI verify-facts --text-file /tmp/vf_smoke.txt; echo "exit=$?"
```
Expected: draft-context prints non-empty RAG + Stilregeln sections; verify-facts prints two `[HIGH]` lines and `exit=1`.

- [ ] **Step 4: Commit**

```bash
git add scripts/rechtmaschine_cli.py
git commit -m "feat: draft-context + verify-facts CLI subcommands"
```

---

### Task 5: skill rename `gerichtsrubrum` → `drafting` + pipeline section

**Files:**
- Rename: `/home/jay/.codex/skills/gerichtsrubrum/` → `/home/jay/.codex/skills/drafting/` (`git mv`)
- Modify: `/home/jay/.codex/skills/drafting/SKILL.md` (frontmatter + new section + self-references)
- Modify: `/home/jay/.codex/skills/api/SKILL.md` (2 references, lines ~106-107)
- Modify: `/home/jay/.codex/skills/api/scripts/jlawyer_cli.py:2328` (hard path to `rubrum_cli.py`)
- Modify: `/home/jay/.claude/projects/-home-jay/memory/formatvorbild-empirisch-ableiten.md` (mentions "gerichtsrubrum-SKILL.md")
- Replace symlink: `/home/jay/.claude/skills/gerichtsrubrum` → new `/home/jay/.claude/skills/drafting`

**Interfaces:**
- Produces: skill `drafting` whose description keeps ALL old triggers (Rubrum, mehrere Kläger/Antragsteller, rechtsbündig, Formatierung prüfen) plus new ones (Schriftsatz/Stellungnahme/Klagebegründung entwerfen, draft a filing). Scripts stay at `~/.codex/skills/drafting/scripts/` unchanged.

- [ ] **Step 1: Rename and rewire**

```bash
cd /home/jay/.codex/skills
git mv gerichtsrubrum drafting
sed -i 's|skills/gerichtsrubrum|skills/drafting|g; s|the gerichtsrubrum skill|the drafting skill|g; s|`gerichtsrubrum` skill|`drafting` skill|g' api/SKILL.md api/scripts/jlawyer_cli.py
grep -rn "gerichtsrubrum" api/ drafting/ && echo "STILL REFERENCED - fix manually" || echo "clean"
rm /home/jay/.claude/skills/gerichtsrubrum
ln -s /home/jay/.codex/skills/drafting /home/jay/.claude/skills/drafting
sed -i 's|gerichtsrubrum-SKILL|drafting-SKILL (früher gerichtsrubrum)|' /home/jay/.claude/projects/-home-jay/memory/formatvorbild-empirisch-ableiten.md
```

- [ ] **Step 2: Update frontmatter**

In `drafting/SKILL.md`: `name: gerichtsrubrum` → `name: drafting`; extend the description (KEEP the whole existing text) by prepending: "Use for drafting any Schriftsatz, Stellungnahme, Klagebegründung, or Behördenschreiben — enforces the mandatory context pipeline (draft-context → formulate → verify-facts → rubrum-cli check) — and specifically when building or verifying court filings: …(existing text)…". Update any `gerichtsrubrum` self-references in the body to `drafting`.

- [ ] **Step 3: Add the pipeline section** (right after the intro, before the Rubrum specifics)

```markdown
## Kontext-Pipeline (PFLICHT bei jedem Schriftsatz/jeder Stellungnahme)

Direktes Drafting ohne diese Pipeline erzeugt Entwürfe ohne Kanzlei-Präzedenz,
ohne Gesetzestexte und ohne Faktenprüfung — genau das, was /generate
automatisch mitbringt. Reihenfolge:

1. **VOR dem Formulieren — Kontext ziehen:**
   `rechtmaschine-cli draft-context --case "NNN/YY" --query "<Thema/Auftrag>" --out ctx.md`
   (Wrapper: `/home/jay/.codex/skills/rechtmaschine/scripts/rechtmaschine-cli`).
   Alle vier Blöcke einweben: RAG-Chunks sind ANONYMISIERTE Argumentations-
   muster aus FREMDEN Akten — Muster und Formulierungen übernehmen, NIEMALS
   Fakten/Platzhalter ([PERSON], [ORT]). Fallgedächtnis = Fakten DIESER Akte.
   Stilregeln gelten wortwörtlich (keine Semikolons, Mandanten-E-Mails nie
   als Anlage zitieren, Parteivortrag im Indikativ).
2. **NACH dem Formulieren — Fakten prüfen:**
   `rechtmaschine-cli verify-facts --text-file entwurf.txt --case-id <uuid>
   [--source-file bescheid.txt]` — Exit 1 = high-severity (Datum/Az ohne
   Beleg): erst auflösen, dann weiter.
3. **Formatierung:** `rubrum-cli check` wie gehabt (Pflicht vor Upload/beA).
4. Upload/beA-Entwurf nach den bestehenden Regeln dieses Skills.

Ausnahme: Für triviale Ein-Absatz-Anschreiben ohne rechtliche Würdigung
genügt Schritt 3.
```

- [ ] **Step 4: Verify + commit**

```bash
ls -la /home/jay/.claude/skills/drafting && /home/jay/.codex/skills/drafting/scripts/rubrum-cli --help >/dev/null 2>&1 || ls /home/jay/.codex/skills/drafting/scripts/
grep -rn "gerichtsrubrum" /home/jay/.codex/skills --include="*.md" --include="*.py" | grep -v "früher gerichtsrubrum" | grep -v ".git"
# expected: no hits (except historical mentions explicitly marked)
cd /home/jay/.codex/skills
git add -A drafting api/SKILL.md api/scripts/jlawyer_cli.py
git status --porcelain --untracked-files=no   # confirm only intended paths staged
git commit -m "rename gerichtsrubrum -> drafting; mandatory draft-context/verify-facts pipeline"
```

(Note: `git add -A drafting` is path-scoped — it stages the rename pair inside `drafting/` only, which is required for `git mv` bookkeeping; it is NOT a repo-wide `add -A`.)

---

### Task 6: End-to-end acceptance on 044/26

**Files:** none (verification only)

- [ ] **Step 1: draft-context against the real case**

```bash
CLI=/home/jay/.codex/skills/rechtmaschine/scripts/rechtmaschine-cli
$CLI draft-context --case "044/26" --query "Stellungnahme gewöhnlicher Aufenthalt § 10 StAG trotz Auslandsstudium, § 51 Abs. 7 AufenthG" --out /tmp/ctx_04426.md
head -50 /tmp/ctx_04426.md
```
Expected: RAG block non-empty (store is live, `RAG_RETRIEVAL_ENABLED=true`), Gesetzestexte block contains § 10 StAG text, Fallgedächtnis block present if the RM case exists (else `_(leer)_` + stderr hint), Stilregeln always present.

- [ ] **Step 2: verify-facts catches a planted error**

```bash
printf 'Der Antrag wurde am 31.02.2019 gestellt (Az. 044/26). Die Gebühr beträgt 255,00 EUR.\n' > /tmp/vf_e2e.txt
$CLI verify-facts --text-file /tmp/vf_e2e.txt --case-id <UUID-from-step-1-or-cases-list>; echo "exit=$?"
```
Expected: `[HIGH]` for the fake date (and any unsourced Az), `exit=1`.

- [ ] **Step 3: app tests green**

Run: `cd /var/opt/docker/rechtmaschine && python -m pytest tests/test_draft_context.py tests/test_citation_verifier.py -q`
Expected: all pass.

- [ ] **Step 4: report**

Summarize to Jay: endpoints live, CLI commands working, skill renamed, pipeline mandatory. No commit.

"""Ad-hoc test for Task 12 (Case-Memory Härtung), sub-fix 4.

Verifies the pure building blocks of the generalized rebase:

  * ``_changed_fields`` detects a changed value for ANY field type — scalar and
    list — using normalized comparison, and treats whitespace-only differences
    as unchanged. These fields are the ones whose pending whole-list/scalar
    ``set`` ops must be dropped on the next accept/put.
  * ``_normalize_field_value`` normalizes scalars and lists consistently.
  * ``_apply_patch_ops`` is a real dry-run: a valid append/whole-list set is
    accepted, a stale list index raises — the rebase relies on that raise to
    drop ops that can no longer apply instead of 500-ing on a later accept.

Pure logic only -- no DB, no containers, no network. The heavy deps
(sqlalchemy / models / shared) are stubbed so the module imports on a bare
host. Run from the repo root:

    python3 tests/test_memory_rebase_changed_fields.py
"""

import os
import sys
import types
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

# --- stub the heavy imports the module pulls at load time ------------------
_sqlalchemy = types.ModuleType("sqlalchemy")
_sqlalchemy.desc = lambda *a, **k: None
sys.modules.setdefault("sqlalchemy", _sqlalchemy)
_orm = types.ModuleType("sqlalchemy.orm")


class _Session:  # pragma: no cover - only needed as a type reference
    pass


_orm.Session = _Session
sys.modules.setdefault("sqlalchemy.orm", _orm)
_models = types.ModuleType("models")
# _target_spec only needs these names to resolve to *some* object; the pure
# ops logic reads its field sets from the module constants, not the ORM models.
for _name in (
    "CaseBrief", "CaseBriefSource", "CaseStrategy", "CaseStrategySource",
    "CaseMemoryRevision", "MemoryUpdateProposal", "MemoryReflectionJob",
):
    setattr(_models, _name, type(_name, (), {}))
sys.modules.setdefault("models", _models)

_BRIEF_LIST = [
    "beteiligte", "verfahrensstand", "sachverhalt", "antraege_ziele",
    "streitige_punkte", "beweismittel", "risiken", "offene_fragen",
]
_BRIEF_SCALAR = ["notizen"]
_STRAT_LIST = [
    "argumentationslinien", "rechtliche_ansatzpunkte", "beweisstrategie",
    "prozessuale_schritte", "vergleich_oder_taktik",
    "risiken_und_gegenargumente", "offene_fragen",
]
_STRAT_SCALAR = ["kernstrategie", "notizen"]


def _make_content_class(list_fields, scalar_fields):
    class _Content:
        def __init__(self, **kw):
            self._d = {f: list(kw.get(f, []) or []) for f in list_fields}
            self._d.update({f: str(kw.get(f, "") or "") for f in scalar_fields})

        def model_dump(self):
            return dict(self._d)

    return _Content


_shared = types.ModuleType("shared")
_shared.CaseBriefContent = _make_content_class(_BRIEF_LIST, _BRIEF_SCALAR)
_shared.CaseStrategyContent = _make_content_class(_STRAT_LIST, _STRAT_SCALAR)
_shared.MemoryPatchOperation = object
_shared.MemorySourceRef = object
_shared.MemoryTargetType = str
sys.modules.setdefault("shared", _shared)

import agent_memory_service as m  # noqa: E402

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


print("changed-fields (scalar + list, normalized):")
# scalar change
check(
    "scalar notizen change detected",
    "notizen" in m._changed_fields({"notizen": "alt"}, {"notizen": "neu"}),
)
# scalar whitespace-only difference is NOT a change
check(
    "scalar whitespace-normalized equal is unchanged",
    "notizen" not in m._changed_fields({"notizen": "Text"}, {"notizen": "  Text  "}),
)
# list change (element added)
check(
    "list sachverhalt change detected",
    "sachverhalt"
    in m._changed_fields({"sachverhalt": ["a"]}, {"sachverhalt": ["a", "b"]}),
)
# identical list is unchanged
check(
    "identical list is unchanged",
    "sachverhalt"
    not in m._changed_fields({"sachverhalt": ["a", "b"]}, {"sachverhalt": ["a", "b"]}),
)
# list of dicts compared by normalized label
check(
    "list-of-dicts unchanged when labels equal",
    "beteiligte"
    not in m._changed_fields(
        {"beteiligte": [{"name": "Ali"}]}, {"beteiligte": [{"name": "ali"}]}
    ),
)
# field present only on one side counts as changed
check(
    "field added on new side is changed",
    "risiken" in m._changed_fields({}, {"risiken": ["x"]}),
)

print("normalize field value:")
check("list -> list of norms", m._normalize_field_value(["A ", " b"]) == ["a", "b"])
check("scalar -> norm string", m._normalize_field_value("  Hallo  ") == "hallo")

print("apply_patch_ops dry-run (rebase relies on the raise to drop stale ops):")
base = m.default_case_brief_json()
base["sachverhalt"] = ["fakt eins"]

# valid append applies
out = m._apply_patch_ops(
    m.BRIEF_TARGET, base, [{"op": "append", "path": "/sachverhalt", "value": "fakt zwei"}]
)
check("valid append applied", out["sachverhalt"] == ["fakt eins", "fakt zwei"])

# whole-list set applies (consolidation op)
out2 = m._apply_patch_ops(
    m.BRIEF_TARGET, base, [{"op": "set", "path": "/sachverhalt", "value": ["nur eins"]}]
)
check("whole-list set applied", out2["sachverhalt"] == ["nur eins"])

# stale list index raises -> rebase would drop this op instead of 500-ing
raised = False
try:
    m._apply_patch_ops(
        m.BRIEF_TARGET, base, [{"op": "remove", "path": "/sachverhalt/9"}]
    )
except Exception:
    raised = True
check("stale remove index raises", raised)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nAll checks passed.")

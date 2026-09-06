"""The target registry is a named struct and knows three targets.

    .venv/bin/python -m pytest tests/test_memory_target_registry.py -q
"""

import sys
import types
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

# --- stub the heavy imports (same pattern as test_memory_rebase_changed_fields)
_sqlalchemy = types.ModuleType("sqlalchemy")
_sqlalchemy.desc = lambda *a, **k: None
sys.modules.setdefault("sqlalchemy", _sqlalchemy)
_orm = types.ModuleType("sqlalchemy.orm")


class _Session:
    pass


_orm.Session = _Session
sys.modules.setdefault("sqlalchemy.orm", _orm)
_models = types.ModuleType("models")
for _name in (
    "CaseBrief",
    "CaseStrategy",
    "CaseBriefSource",
    "CaseStrategySource",
    "CaseAssessment",
    "CaseAssessmentSource",
    "CaseMemoryRevision",
    "MemoryUpdateProposal",
    "Document",
    "Case",
    "User",
):
    setattr(_models, _name, type(_name, (), {}))
sys.modules.setdefault("models", _models)


def _make_content_class(fields):
    class _Content:
        def __init__(self, **kw):
            self._d = dict(kw)

        def model_dump(self):
            return dict(self._d)

    return _Content


_shared = types.ModuleType("shared")
_shared.CaseBriefContent = _make_content_class([])
_shared.CaseStrategyContent = _make_content_class([])
_shared.MemoryPatchOperation = object
_shared.MemorySourceRef = object
_shared.MemoryTargetType = str
sys.modules.setdefault("shared", _shared)

import agent_memory_service as ams  # noqa: E402


def test_registry_knows_three_targets():
    for target in ("case_brief", "case_strategy", "case_assessment"):
        spec = ams._target_spec(target)
        assert spec.model is not None
        assert callable(spec.validate)
        assert callable(spec.renderer)


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        ams._target_spec("case_nonsense")


def test_spec_is_attribute_addressable_not_positional():
    spec = ams._target_spec("case_brief")
    assert hasattr(spec, "list_fields")
    assert "beteiligte" in spec.list_fields


def test_assessment_spec_delegates_to_domain_module():
    from assessment_memory import apply_assessment_ops, validate_assessment_content

    spec = ams._target_spec("case_assessment")
    assert spec.validate is validate_assessment_content
    assert spec.apply_ops is apply_assessment_ops

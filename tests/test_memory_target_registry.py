"""The target registry is a named struct and knows three targets.

    .venv/bin/python -m pytest tests/test_memory_target_registry.py -q
"""

import sys
import types
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _make_content_class(fields):
    class _Content:
        def __init__(self, **kw):
            self._d = dict(kw)

        def model_dump(self):
            return dict(self._d)

    return _Content


@pytest.fixture(scope="module", autouse=True)
def _stub_heavy_imports():
    """Stub the heavy imports (same pattern as test_memory_rebase_changed_fields).

    Installed with monkeypatch.setitem (a module-scoped pytest.MonkeyPatch,
    since the built-in ``monkeypatch`` fixture is function-scoped) so they
    are undone again once this module's tests are done, instead of leaking
    into every later test file's sys.modules (see tests/conftest.py
    docstring: that bug hid four broken files for 8 months).
    """
    mp = pytest.MonkeyPatch()

    _sqlalchemy = types.ModuleType("sqlalchemy")
    _sqlalchemy.desc = lambda *a, **k: None
    mp.setitem(sys.modules, "sqlalchemy", _sqlalchemy)

    _orm = types.ModuleType("sqlalchemy.orm")

    class _Session:
        pass

    _orm.Session = _Session
    mp.setitem(sys.modules, "sqlalchemy.orm", _orm)

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
    mp.setitem(sys.modules, "models", _models)

    _shared = types.ModuleType("shared")
    _shared.CaseBriefContent = _make_content_class([])
    _shared.CaseStrategyContent = _make_content_class([])
    _shared.MemoryPatchOperation = object
    _shared.MemorySourceRef = object
    _shared.MemoryTargetType = str
    mp.setitem(sys.modules, "shared", _shared)

    # Force a fresh import of the module under test against the stubs above,
    # regardless of what an earlier test file already left behind.
    mp.delitem(sys.modules, "agent_memory_service", raising=False)

    yield

    # Drop our stub-backed import too, so a later test file that imports
    # agent_memory_service for real doesn't see this stubbed copy.
    mp.delitem(sys.modules, "agent_memory_service", raising=False)
    mp.undo()


def test_registry_knows_three_targets():
    import agent_memory_service as ams

    for target in ("case_brief", "case_strategy", "case_assessment"):
        spec = ams._target_spec(target)
        assert spec.model is not None
        assert callable(spec.validate)
        assert callable(spec.renderer)


def test_unknown_target_raises():
    import agent_memory_service as ams

    with pytest.raises(ValueError):
        ams._target_spec("case_nonsense")


def test_spec_is_attribute_addressable_not_positional():
    import agent_memory_service as ams

    spec = ams._target_spec("case_brief")
    assert hasattr(spec, "list_fields")
    assert "beteiligte" in spec.list_fields


def test_assessment_spec_delegates_to_domain_module():
    import agent_memory_service as ams
    from assessment_memory import apply_assessment_ops, validate_assessment_content

    spec = ams._target_spec("case_assessment")
    assert spec.validate is validate_assessment_content
    assert spec.apply_ops is apply_assessment_ops

"""Prompt-context integration of the case_assessment block.

get_case_memory_prompt_context glues brief, strategy and Gutachten together,
applies two different budgets and gates the provenance on the pseudonymizer.
That wiring had no test (final review, finding 5).

    .venv/bin/python -m pytest tests/test_memory_assessment_prompt_context.py -q
"""

import sys
import types
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

# --- stub the heavy imports (same pattern as test_memory_target_registry) ---
_INSERTED = []


def _stub(name, module):
    if name not in sys.modules:
        sys.modules[name] = module
        _INSERTED.append(name)


_sqlalchemy = types.ModuleType("sqlalchemy")
_sqlalchemy.desc = lambda *a, **k: None
_stub("sqlalchemy", _sqlalchemy)
_orm = types.ModuleType("sqlalchemy.orm")


class _Session:
    pass


_orm.Session = _Session
_stub("sqlalchemy.orm", _orm)
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
_stub("models", _models)


def _make_content_class():
    class _Content:
        def __init__(self, **kw):
            self._d = dict(kw)

        def model_dump(self):
            return dict(self._d)

    return _Content


_shared = types.ModuleType("shared")
_shared.CaseBriefContent = _make_content_class()
_shared.CaseStrategyContent = _make_content_class()
_shared.MemoryPatchOperation = object
_shared.MemorySourceRef = object
_shared.MemoryTargetType = str
_stub("shared", _shared)

import agent_memory_service as ams  # noqa: E402
from assessment_memory import ASSESSMENT_BLOCK_HEADER  # noqa: E402

# conftest rule: files that stub sys.modules entries pop them again, so a
# later test file still imports the real thing.
for _name in _INSERTED:
    sys.modules.pop(_name, None)


# The brief/strategy renderers are stubbed in the fixture: this file is about
# how the assessment block is glued in and budgeted, not about their content
# models (whose validation differs depending on whether the real `shared` is
# already imported by an earlier test file).
BRIEF = {"beteiligte": ["Mandant"]}
STRATEGY = {"kernstrategie": "Duldung erzwingen"}
BRIEF_TEXT = "Fallbrief:\nBeteiligte: Mandant"
STRATEGY_TEXT = "Fallstrategie:\nKernstrategie: Duldung erzwingen"
VERIFIED_AZ = "18 E 491/12"
BLOCKED_AZ = "9 K 77/25"


def _assessment(with_blocked=False):
    fundstellen = [
        {
            "gericht": "OVG NRW",
            "datum": "2012-06-18",
            "az": VERIFIED_AZ,
            "art": "Beschluss",
            "aussage": "Die GÜB ersetzt die Duldung nicht.",
            "richtung": "pro",
            "store": "verified",
        }
    ]
    if with_blocked:
        fundstellen.append(
            {
                "gericht": "VG Musterstadt",
                "datum": "2025-03-04",
                "az": BLOCKED_AZ,
                "aussage": "Gegenlinie.",
                "richtung": "contra",
                "store": "not_in_store",
            }
        )
    return {
        "gutachten": [
            {
                "id": "gueb-statt-duldung",
                "rechtsfrage": "Duldung oder GÜB?",
                "ergebnis": "Duldung.",
                "stand": "2026-09-03",
                "status": "aktiv",
                "pruefung": [
                    {
                        "these": "Kein ungeregelter Aufenthalt",
                        "bewertung": "Trägt.",
                        "fundstellen": [VERIFIED_AZ],
                    }
                ],
                "fundstellen": fundstellen,
                "risiken": ["Behörde belegt eine zeitnahe Abschiebung."],
                "quelle": "Vermerk_2026-09-02.pdf",
            }
        ],
        "notizen": "",
    }


class _Row:
    """Stand-in for a memory ORM row -- _target_content is stubbed out, so
    only the identity of the row matters."""

    def __init__(self, kind):
        self.kind = kind


@pytest.fixture
def ctx(monkeypatch):
    """Wire get_case_memory_prompt_context to in-memory content and silence
    the three optional context renderers."""
    state = {
        "assessment": _assessment(),
        "pseudonymized": None,
        "pseudonymize": lambda text: text,
    }

    monkeypatch.setattr(ams, "get_or_create_case_brief", lambda db, o, c, **k: _Row("brief"))
    monkeypatch.setattr(ams, "get_or_create_case_strategy", lambda db, o, c, **k: _Row("strategy"))
    monkeypatch.setattr(ams, "get_or_create_case_assessment", lambda db, o, c, **k: _Row("assessment"))

    def fake_target_content(target_type, target):
        return {
            "case_brief": BRIEF,
            "case_strategy": STRATEGY,
            "case_assessment": state["assessment"],
        }[target_type]

    monkeypatch.setattr(ams, "_target_content", fake_target_content)
    monkeypatch.setattr(ams, "render_case_brief_compact", lambda content: BRIEF_TEXT)
    monkeypatch.setattr(ams, "render_case_strategy_compact", lambda content: STRATEGY_TEXT)

    def fake_pseudonymize(db, owner_id, case_id, text):
        state["pseudonymized"] = text
        return state["pseudonymize"](text)

    monkeypatch.setattr(ams, "pseudonymize_case_text_for_cloud", fake_pseudonymize)

    for mod_name, attr in (
        ("doktrin_context", "render_doktrin_context"),
        ("endpoints.pattern_wiki", "render_pattern_wiki_context"),
        ("endpoints.jurisprudence", "maybe_render_jurisprudence_context"),
    ):
        stub = types.ModuleType(mod_name)
        setattr(stub, attr, lambda *a, **k: "")
        monkeypatch.setitem(sys.modules, mod_name, stub)

    return state


def _render(**kwargs):
    return ams.get_case_memory_prompt_context(None, "owner-1", "case-1", **kwargs)


def test_assessment_block_follows_the_brief_without_strategy(ctx):
    text = _render(include_strategy=False)
    assert text.startswith("Fallbrief:")
    assert "Fallstrategie" not in text
    header_at = text.index(ASSESSMENT_BLOCK_HEADER)
    assert text[:header_at].rstrip() == BRIEF_TEXT
    assert "RECHTLICHE WÜRDIGUNG DER KANZLEI" in text


def test_memory_truncation_does_not_touch_the_assessment_block(ctx):
    budget = len(BRIEF_TEXT + "\n\n" + STRATEGY_TEXT) - 20
    text = _render(max_chars=budget)
    assert "[Fallgedächtnis gekürzt]" in text
    # the Gutachten has its own budget and survives the memory truncation
    block, _, _ = __import__("assessment_memory").render_assessment_block(_assessment())
    assert block in text
    assert text.index("[Fallgedächtnis gekürzt]") < text.index(ASSESSMENT_BLOCK_HEADER)


def test_pseudonymizer_sees_the_assessment_block(ctx):
    _render()
    assert ctx["pseudonymized"] is not None
    assert ASSESSMENT_BLOCK_HEADER in ctx["pseudonymized"]
    assert BRIEF_TEXT in ctx["pseudonymized"]
    assert STRATEGY_TEXT in ctx["pseudonymized"]


def test_fail_closed_pseudonymizer_ungates_nothing(ctx):
    ctx["assessment"] = _assessment(with_blocked=True)
    ctx["pseudonymize"] = lambda text: ""
    collect = {}
    text = _render(collect=collect)
    assert text == ""
    assert collect["assessment_used"] is False
    assert collect["assessment_ids"] == []
    assert collect["assessment_blocked_az"] == []
    # case_memory_used stays True: brief_used/strategy_used are NOT gated on
    # the pseudonymizer. Pre-existing behaviour outside this branch -- only
    # the assessment share of the flag was gated (Task 6 ruling).
    assert collect["case_memory_used"] is True
    assert collect["case_memory_text"] == ""


def test_blocked_az_is_reported_for_the_fact_check(ctx):
    ctx["assessment"] = _assessment(with_blocked=True)
    collect = {}
    text = _render(collect=collect)
    assert collect["assessment_blocked_az"] == [BLOCKED_AZ]
    assert collect["assessment_ids"] == ["gueb-statt-duldung"]
    assert collect["assessment_used"] is True
    assert f"Nicht zitierfähig (nicht im Bestand): {BLOCKED_AZ}" in text

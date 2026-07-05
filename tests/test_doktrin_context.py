import os
import sys
import types
import uuid
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ["DOKTRIN_INJECT_ENABLED"] = "true"

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.dialects.postgresql import JSONB  # noqa: E402
from sqlalchemy.ext.compiler import compiles  # noqa: E402
from sqlalchemy.orm import declarative_base, sessionmaker  # noqa: E402


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


Base = declarative_base()
_engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=_engine)

sys.modules["database"] = types.SimpleNamespace(
    Base=Base, SessionLocal=SessionLocal, engine=_engine
)
sys.modules.setdefault(
    "shared",
    types.SimpleNamespace(ensure_anonymization_service_ready=lambda *a, **k: None),
)

import doktrin_context  # noqa: E402
from doktrin_context import render_doktrin_context  # noqa: E402
from models import Case, DoktrinPage  # noqa: E402

# doktrin_context has captured the stubs; drop them so later test modules that
# import the REAL shared/database (e.g. via app.endpoints) don't get poisoned.
sys.modules.pop("shared", None)
sys.modules.pop("database", None)

Base.metadata.create_all(_engine, tables=[Case.__table__, DoktrinPage.__table__])

OWNER = uuid.uuid4()
USER = types.SimpleNamespace(id=OWNER)


def _seed(db, case_facets):
    case = Case(owner_id=OWNER, name="001/26 Test", facets_json=case_facets)
    db.add(case)
    pages = [
        DoktrinPage(
            page_id="afghanistan",
            title="Afghanistan",
            url="https://wiki.aufentha.lt/afghanistan",
            country="Afghanistan",
            normen=[],
            clean_text="# Afghanistan\nLage im Land.\n\n## Abschiebungsverbote\nDetails § 60.",
            clean_chars=70,
            status="active",
        ),
        DoktrinPage(
            page_id="60_aufenthg",
            title="§ 60 AufenthG",
            url="https://wiki.aufentha.lt/60_aufenthg",
            country=None,
            normen=["AufenthG § 60 Abs. 5"],
            clean_text="# § 60 AufenthG\nAbschiebungsverbote im Einzelnen." + " Mehr." * 50,
            clean_chars=300,
            status="active",
        ),
        DoktrinPage(
            page_id="dublin",
            title="Dublin-Verfahren",
            url="https://wiki.aufentha.lt/dublin",
            country=None,
            normen=[],
            clean_text="# Dublin\nZustaendigkeitsfragen." + " Mehr." * 20,
            clean_chars=140,
            status="active",
        ),
        DoktrinPage(
            page_id="inaktiv",
            title="Alte Seite",
            country="Afghanistan",
            normen=[],
            clean_text="Veraltet.",
            clean_chars=9,
            status="gone",
        ),
    ]
    db.add_all(pages)
    db.commit()
    return case


def _fresh_case(facets):
    db = SessionLocal()
    db.query(DoktrinPage).delete()
    db.query(Case).delete()
    db.commit()
    case = _seed(db, facets)
    return db, case


FACETS = {
    "herkunftsland": "afghanistan",
    "schutzgruende": ["AufenthG § 60 Abs. 5"],
    "verfahrensart": "dublin",
    "themen": [],
}


def test_deterministic_scoring_picks_country_then_best_scorer(monkeypatch=None):
    db, case = _fresh_case(FACETS)
    doktrin_context.retrieve_chunks = lambda *a, **k: []
    collect = []
    block = render_doktrin_context(db, USER, case.id, "Fallzusammenfassung.", collect=collect)
    assert "### Afghanistan" in block
    assert "(Kanzlei-Wiki, nur intern)" in block
    # Country match (3) ranks above the norm (2) and dublin (2) pages; the
    # second deterministic slot goes to the larger 2-scorer (§ 60 page).
    assert "### § 60 AufenthG" in block
    assert "gone" not in block and "Veraltet" not in block
    modes = [e["mode"] for e in collect]
    assert modes == ["deterministic", "deterministic"]
    db.close()


def test_semantic_fills_slots_and_dedupes_deterministic_picks():
    db, case = _fresh_case(FACETS)

    def fake_retrieve(query, limit=6, use_reranker=True, collection=None, **kw):
        assert collection == "doktrin"
        return [
            {  # duplicate of a deterministic pick -> must be dropped
                "text": "Duplikat.",
                "metadata": {"page_id": "afghanistan", "page_title": "Afghanistan"},
            },
            {
                "text": "Teil zwei.",
                "metadata": {
                    "page_id": "syrien",
                    "page_title": "Syrien",
                    "heading_path": "Syrien > Wehrdienst",
                    "url": "https://wiki.aufentha.lt/syrien",
                    "chunk_index": 1,
                },
            },
            {
                "text": "Teil eins.",
                "metadata": {"page_id": "syrien", "page_title": "Syrien", "chunk_index": 0},
            },
        ]

    doktrin_context.retrieve_chunks = fake_retrieve
    collect = []
    block = render_doktrin_context(db, USER, case.id, "Syrien Wehrdienst Fall.", collect=collect)
    assert "### Syrien — Syrien > Wehrdienst" in block
    # chunk_index ordering: Teil eins before Teil zwei.
    assert block.index("Teil eins.") < block.index("Teil zwei.")
    assert [e["page_id"] for e in collect] == ["afghanistan", "60_aufenthg", "syrien"]
    assert collect[2]["mode"] == "semantic"
    db.close()


def test_budget_truncates_and_skips_when_nearly_exhausted():
    db, case = _fresh_case(FACETS)
    doktrin_context.retrieve_chunks = lambda *a, **k: []
    big = SessionLocal()
    row = big.query(DoktrinPage).filter_by(page_id="60_aufenthg").one()
    row.clean_text = "# § 60 AufenthG\n" + ("Zeile mit Inhalt.\n" * 900)
    big.commit()
    big.close()

    old_budget = doktrin_context.DOKTRIN_INJECT_MAX_CHARS
    old_entry = doktrin_context.DOKTRIN_ENTRY_MAX_CHARS
    try:
        doktrin_context.DOKTRIN_ENTRY_MAX_CHARS = 50000

        # Budget 1200: small country entry fits, the oversized § 60 entry is
        # truncated at a line boundary (remaining >= 800).
        doktrin_context.DOKTRIN_INJECT_MAX_CHARS = 1200
        block = render_doktrin_context(db, USER, case.id, "Fall.")
        assert len(block) <= 1200 + 50
        assert "[Eintrag gekürzt]" in block
        assert block.count("### ") == 2

        # Budget 700: after the first entry less than 800 remains -> the
        # oversized entry is skipped entirely, no broken fragment injected.
        doktrin_context.DOKTRIN_INJECT_MAX_CHARS = 700
        block = render_doktrin_context(db, USER, case.id, "Fall.")
        assert block.count("### ") == 1
        assert "[Eintrag gekürzt]" not in block
    finally:
        doktrin_context.DOKTRIN_INJECT_MAX_CHARS = old_budget
        doktrin_context.DOKTRIN_ENTRY_MAX_CHARS = old_entry
    db.close()


def test_disabled_gate_and_exception_degrade_to_empty():
    db, case = _fresh_case(FACETS)
    doktrin_context.retrieve_chunks = lambda *a, **k: []
    old = doktrin_context.DOKTRIN_INJECT_ENABLED
    try:
        doktrin_context.DOKTRIN_INJECT_ENABLED = False
        assert render_doktrin_context(db, USER, case.id, "Fall.") == ""
    finally:
        doktrin_context.DOKTRIN_INJECT_ENABLED = old

    def boom(*a, **k):
        raise RuntimeError("rag down")

    doktrin_context.retrieve_chunks = boom
    # No facets -> deterministic empty -> semantic raises -> caught -> "".
    no_facet_db, no_facet_case = _fresh_case({})
    assert render_doktrin_context(no_facet_db, USER, no_facet_case.id, "Fall.") == ""
    no_facet_db.close()
    db.close()


def test_last_used_at_stamped_on_injected_pages():
    db, case = _fresh_case(FACETS)
    doktrin_context.retrieve_chunks = lambda *a, **k: []
    block = render_doktrin_context(db, USER, case.id, "Fall.")
    assert block
    check = SessionLocal()
    afg = check.query(DoktrinPage).filter_by(page_id="afghanistan").one()
    dub = check.query(DoktrinPage).filter_by(page_id="dublin").one()
    assert afg.last_used_at is not None
    assert dub.last_used_at is None  # not injected (third scorer, slot taken)
    check.close()
    db.close()

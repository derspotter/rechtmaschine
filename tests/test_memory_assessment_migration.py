"""Schema smoke test for case_assessments.

Needs the live Postgres of the dev stack. Marked slow so the pre-push hook
skips it.

    .venv/bin/python -m pytest tests/test_memory_assessment_migration.py -q
"""

import os
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

pytestmark = pytest.mark.slow


# tests/conftest.py seeds this dummy URL so importing app modules works
# without a database. It points at a closed port, so it is as good as unset
# for this test -- treat it as a sentinel, not as a live database.
DUMMY_DATABASE_URL = "postgresql://test:test@127.0.0.1:1/test"


def _engine():
    url = os.environ.get("DATABASE_URL")
    if not url or url == DUMMY_DATABASE_URL:
        pytest.skip("no live DATABASE_URL (unset or the conftest dummy)")
    from sqlalchemy import create_engine

    return create_engine(url)


def test_migration_key_is_registered():
    main_py = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text()
    assert '"2026-09-03_case_assessments"' in main_py


def test_orm_classes_exist_with_expected_columns():
    from models import CaseAssessment, CaseAssessmentSource

    cols = {c.name for c in CaseAssessment.__table__.columns}
    assert {"id", "owner_id", "case_id", "content_json", "search_text", "version"} <= cols
    assert CaseAssessment.__table__.name == "case_assessments"
    src_cols = {c.name for c in CaseAssessmentSource.__table__.columns}
    assert "case_assessment_id" in src_cols


def test_tables_exist_in_live_database():
    from sqlalchemy import inspect
    from sqlalchemy.exc import OperationalError

    engine = _engine()
    try:
        inspector = inspect(engine)
        inspector.get_table_names()
    except OperationalError as exc:
        pytest.skip(f"no reachable database: {str(exc)[:120]}")
    assert "case_assessments" in inspector.get_table_names()
    assert "case_assessment_sources" in inspector.get_table_names()
    indexes = {i["name"] for i in inspector.get_indexes("case_assessments")}
    assert any("owner_case" in name for name in indexes)

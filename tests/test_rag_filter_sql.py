"""Unit tests for rag/api/main.py's _filter_sql owner clause. Pure Python,
no DB — loads the module directly (deps: fastapi/pydantic, no live DB conn
is made at import time). Requires fastapi/pydantic/psycopg2 to be importable
(e.g. `docker exec rechtmaschine-app python3 tests/test_rag_filter_sql.py`
with the repo mounted, or any env with those three installed).
Run: python tests/test_rag_filter_sql.py"""
import importlib.util
import os
import sys

_MAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "rag", "api", "main.py")
_spec = importlib.util.spec_from_file_location("rag_api_main_under_test", _MAIN_PATH)
rag_main = importlib.util.module_from_spec(_spec)
sys.modules["rag_api_main_under_test"] = rag_main
_spec.loader.exec_module(rag_main)

_filter_sql = rag_main._filter_sql


def test_owner_id_present_scopes_to_owner_or_public():
    clause, params = _filter_sql(None, "user-123")
    assert clause == " AND (metadata ->> 'owner_id' IS NULL OR metadata ->> 'owner_id' = %s)", clause
    assert params == ["user-123"], params


def test_owner_id_none_restricts_to_public_only():
    clause, params = _filter_sql(None, None)
    assert clause == " AND (metadata ->> 'owner_id' IS NULL)", clause
    assert params == [], params


def test_owner_id_empty_string_restricts_to_public_only():
    clause, params = _filter_sql(None, "")
    assert clause == " AND (metadata ->> 'owner_id' IS NULL)", clause
    assert params == [], params


def test_owner_clause_combines_with_metadata_filters_consistently():
    filters = rag_main.RagFilters(statute="AsylG")
    clause, params = _filter_sql(filters, "user-123")
    assert clause == (
        " AND (metadata ->> %s = %s) AND (metadata ->> 'owner_id' IS NULL OR metadata ->> 'owner_id' = %s)"
    ), clause
    assert params == ["statute", "AsylG", "user-123"], params

    clause_no_owner, params_no_owner = _filter_sql(filters, None)
    assert clause_no_owner == (
        " AND (metadata ->> %s = %s) AND (metadata ->> 'owner_id' IS NULL)"
    ), clause_no_owner
    assert params_no_owner == ["statute", "AsylG"], params_no_owner


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")

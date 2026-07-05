"""Shared test bootstrap.

The app modules use intra-app absolute imports (``import shared``) and some
require env at import time (``auth.py`` insists on SECRET_KEY). Individual
test files historically set this up themselves — do it once here so every
file collects, and the whole suite can run as one command:

    .venv/bin/python -m pytest tests/ -q

The integration-leaning tests import the full endpoint tree, so the .venv
needs the app deps bcrypt, python-jose[cryptography] and xai-sdk (installed
2026-07-05). A pre-push hook runs the suite minus "slow" marks — install via
cp scripts/git-hooks/pre-push .git/hooks/.

Test files that stub sys.modules entries ("shared", "database") MUST pop them
after importing their module under test, or they poison every later test
file's imports in the same run (that bug hid four broken files for 8 months).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Lazy URL: create_engine() accepts the app's pool kwargs for postgres and
# never connects unless a test actually uses the DB (none should).
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@127.0.0.1:1/test")
os.environ.setdefault("SECRET_KEY", "test-secret-not-production")

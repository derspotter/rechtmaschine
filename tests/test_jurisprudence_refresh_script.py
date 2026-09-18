"""scripts/jurisprudence_refresh.sh exit semantics (drives OnFailure= mail).

A completed run with per-decision failures (asyl.net timeouts, scans) is NOT
a unit failure — the 180-day window retries them next week and 6 of 12 weekly
runs had such failures. Only a run that never reached its summary line (crash,
Playwright timeout, docker down) fails the unit and triggers the mail.

Run: .venv/bin/python -m pytest tests/test_jurisprudence_refresh_script.py -q
"""
import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jurisprudence_refresh.sh"


def _run(tmp_path, fake_output: str, fake_rc: int):
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' {_sh_quote(fake_output)}\n"
        f"exit {fake_rc}\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "refresh.log"
    env = {
        **os.environ,
        "JURIS_REFRESH_DOCKER": str(fake),
        "JURIS_REFRESH_LOG": str(log),
        "JURIS_REFRESH_CONTAINER": "fake-container",
    }
    proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    return proc.returncode, log.read_text() if log.exists() else ""


def _sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def test_completed_run_with_per_decision_failures_is_success(tmp_path):
    out = (
        "asyl.net: 11 results (5 already stored), 6 new with PDFs\n"
        "  FAIL  https://www.asyl.net/rsdb/m34396 — [Errno 110] Connection timed out\n"
        "ingested 3, dup-ref 5, dup-content 0, short 0, failed 3; 61 chunks into 'jurisprudence'."
    )
    rc, log = _run(tmp_path, out, fake_rc=1)
    assert rc == 0
    assert "ingested 3" in log and "=== refresh done" in log


def test_crashed_run_without_summary_fails_the_unit(tmp_path):
    out = (
        "Traceback (most recent call last):\n"
        "playwright._impl._errors.TimeoutError: Page.click: Timeout 30000ms exceeded."
    )
    rc, log = _run(tmp_path, out, fake_rc=1)
    assert rc != 0
    assert "TimeoutError" in log and "=== refresh FAILED" in log


def test_clean_run_is_success(tmp_path):
    out = "ingested 0, dup-ref 6, dup-content 0, short 0, failed 0; 0 chunks into 'jurisprudence'."
    rc, log = _run(tmp_path, out, fake_rc=0)
    assert rc == 0
    assert "=== refresh done" in log

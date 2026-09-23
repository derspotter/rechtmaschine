"""Regression: die pymupdf-Deprecation-Warnung darf nicht auf stdout landen.

2026-09-23 lieferte ``verify-source --json`` kein parsebares JSON: die erste
Zeile auf stdout war ``warning: The `fitz` API is deprecated ...``. Ursache
war nicht verify_source.py selbst, sondern ein transitiver Import:
verify_source importiert auf oberster Ebene jurisprudence_ingest, und dieses
importierte ``fitz``. Das ``fitz``-Paket ist bei pymupdf nur noch eine
Kompatibilitätsschicht, die beim Import ``message_warning`` ruft, und pymupdf
schreibt seine Meldungen standardmäßig auf stdout (nicht per ``warnings``
auf stderr). Jeder JSON-Konsument des Werkzeugs brach daran.

Der Import muss in einem frischen Prozess laufen: pymupdf gibt die Warnung
nur beim ersten Import aus, im Testprozess wäre sie längst verbraucht.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"
FITZ_WARNING = "The `fitz` API is deprecated"


@pytest.mark.parametrize("module", ["jurisprudence_ingest", "verify_source"])
def test_import_writes_no_fitz_warning_to_stdout(module):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=APP,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert FITZ_WARNING not in proc.stdout

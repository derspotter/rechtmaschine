#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv_hpi}"
PY_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PY_BIN" ]]; then
    echo "Missing Python at $PY_BIN" >&2
    exit 1
fi

# Ensure UltraInfer can find CUDA shared libs inside the venv.
NVIDIA_LIBS="$("$PY_BIN" - <<'PY'
import glob
import os
import sys
base = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")
paths = sorted({p for p in glob.glob(os.path.join(base, "*/lib")) if os.path.isdir(p)})
print(":".join(paths))
PY
)"

if [[ -n "$NVIDIA_LIBS" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PY_BIN" "$ROOT_DIR/ocr_service.py"

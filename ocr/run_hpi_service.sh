#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv_hpi}"
PY_BIN="$VENV_DIR/bin/python"
SERVICE_FILE="${OCR_SERVICE_FILE:-ocr_service.py}"

if [[ ! -x "$PY_BIN" ]]; then
    echo "Missing Python at $PY_BIN" >&2
    exit 1
fi

# Ensure UltraInfer can find CUDA shared libs inside the venv.
NVIDIA_LIBS="$("$PY_BIN" - <<'PY'
import glob
import os
import sys

site = os.path.join(
    sys.prefix,
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)

paths = set()
paths.update(glob.glob(os.path.join(site, "nvidia", "*", "lib")))
paths.update(glob.glob(os.path.join(site, "nvidia", "*", "lib64")))
paths.update(glob.glob(os.path.join(site, "ultra_infer", "libs")))
paths.update(glob.glob(os.path.join(site, "paddle", "libs")))

paths = [p for p in sorted(paths) if os.path.isdir(p)]
print(":".join(paths))
PY
)"

if [[ -n "$NVIDIA_LIBS" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ! -f "$ROOT_DIR/$SERVICE_FILE" ]]; then
    echo "Missing OCR service file at $ROOT_DIR/$SERVICE_FILE" >&2
    exit 1
fi

exec "$PY_BIN" "$ROOT_DIR/$SERVICE_FILE"

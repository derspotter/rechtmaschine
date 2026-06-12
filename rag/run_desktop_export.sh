#!/usr/bin/env bash
# Desktop RAG datasource export: snapshot -> sieve -> manifest -> stage.
# Run on desktop as jayjag. Re-runnable; the live corpus is never written to.
#
#   bash rag/run_desktop_export.sh [--skip-snapshot]

set -euo pipefail

CORPUS_DIR=${RAG_CORPUS_DIR:-/home/jayjag/Kanzlei/kanzlei}
SNAPSHOT_DIR=${RAG_SNAPSHOT_DIR:-/home/jayjag/Kanzlei/kanzlei-rag-snapshot}
EXPORT_ROOT=${RAG_EXPORT_ROOT:-/home/jayjag/rechtmaschine-rag-export}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${RAG_PYTHON:-${REPO_DIR}/.venv/bin/python}

if [[ "${1:-}" != "--skip-snapshot" ]]; then
    echo "[export] snapshotting ${CORPUS_DIR} -> ${SNAPSHOT_DIR}"
    rsync -a --delete "${CORPUS_DIR}/" "${SNAPSHOT_DIR}/"
fi

echo "[export] sieving snapshot (filter_corpus)"
"${PYTHON}" "${REPO_DIR}/rag/filter_corpus.py" \
    --corpus-dir "${SNAPSHOT_DIR}" \
    --allow-writable-corpus

echo "[export] building INCLUDE manifest (prefer PDF over ODT duplicates)"
"${PYTHON}" "${REPO_DIR}/rag/export_manifest.py" \
    --corpus-dir "${SNAPSHOT_DIR}" \
    --prefer-pdf

echo "[export] staging files + checksums into ${EXPORT_ROOT}"
"${PYTHON}" "${REPO_DIR}/rag/export_staged_files.py" \
    --corpus-dir "${SNAPSHOT_DIR}" \
    --export-root "${EXPORT_ROOT}"

echo "[export] done. Pull on debian with:"
echo "  rsync -aH --info=progress2 jayjag@desktop:${EXPORT_ROOT}/ ~/rechtmaschine/rag/data/imports/desktop-export/"

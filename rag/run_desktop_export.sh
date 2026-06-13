#!/usr/bin/env bash
# Desktop RAG datasource export: sieve -> manifest -> stage.
# Run on desktop as jayjag. Re-runnable.
#
# The sieve reads the live corpus directly; filter_corpus.py blocks all
# write operations toward the corpus and verifies it unchanged afterwards.
#
#   bash rag/run_desktop_export.sh

set -euo pipefail

CORPUS_DIR=${RAG_CORPUS_DIR:-/home/jayjag/Kanzlei/kanzlei}
EXPORT_ROOT=${RAG_EXPORT_ROOT:-/home/jayjag/rechtmaschine-rag-export}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${RAG_PYTHON:-${REPO_DIR}/.venv/bin/python}

echo "[export] sieving ${CORPUS_DIR} (read-only, write-guarded)"
"${PYTHON}" "${REPO_DIR}/rag/filter_corpus.py" \
    --corpus-dir "${CORPUS_DIR}" \
    --allow-writable-corpus

echo "[export] building INCLUDE manifest (prefer PDF over ODT duplicates)"
"${PYTHON}" "${REPO_DIR}/rag/export_manifest.py" \
    --corpus-dir "${CORPUS_DIR}" \
    --prefer-pdf

echo "[export] staging files + checksums into ${EXPORT_ROOT}"
"${PYTHON}" "${REPO_DIR}/rag/export_staged_files.py" \
    --corpus-dir "${CORPUS_DIR}" \
    --export-root "${EXPORT_ROOT}" \
    --prune

echo "[export] done. Pull on debian with:"
echo "  rsync -aH --info=progress2 jayjag@desktop:${EXPORT_ROOT}/ ~/rechtmaschine/rag/data/imports/desktop-export/"

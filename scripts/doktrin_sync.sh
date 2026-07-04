#!/usr/bin/env bash
# Nightly Doktrin sync: mirror wiki.aufentha.lt into the doktrin RAG
# collection + doktrin_pages bookkeeping.
#
# Sha-dedup makes unchanged pages free; only new/edited pages are re-cleaned,
# re-chunked and re-embedded (server-side on debian). Network-bound, no GPU —
# scheduled 04:30, after the 03:30/03:50 GPU window.
#
# Install (systemd user timer doktrin-sync.timer, 04:30).
set -euo pipefail

CONTAINER="${DOKTRIN_SYNC_CONTAINER:-rechtmaschine-app}"
DELAY="${DOKTRIN_SYNC_DELAY:-0.1}"
LOG="${DOKTRIN_SYNC_LOG:-/var/opt/docker/rechtmaschine/rag/data/doktrin_sync.log}"

{
    echo "=== doktrin sync $(date -Is) | delay=${DELAY} ==="
    /usr/bin/docker exec "${CONTAINER}" python /app/doktrin_sync.py --delay "${DELAY}"
    echo "=== doktrin sync done $(date -Is) ==="
} >> "${LOG}" 2>&1

#!/usr/bin/env bash
# Nächtlicher Rechtsgebiets-Sync j-lawyer → Rechtmaschine (Stufe 0).
#
# Additiv & idempotent: reason-Feld → Gebietsliste, nie entfernen,
# manuell Gesetztes bleibt. Billig (ein API-Call + DB), kein GPU.
set -euo pipefail

CONTAINER="${RECHTSGEBIET_SYNC_CONTAINER:-rechtmaschine-app}"
LOG="${RECHTSGEBIET_SYNC_LOG:-/var/opt/docker/rechtmaschine/rag/data/rechtsgebiet_sync.log}"

{
    echo "=== rechtsgebiet sync $(date -Is) ==="
    /usr/bin/docker exec "${CONTAINER}" python /app/sync_rechtsgebiet_jlawyer.py
} >> "${LOG}" 2>&1

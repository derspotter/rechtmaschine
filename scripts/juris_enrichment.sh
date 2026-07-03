#!/usr/bin/env bash
# Nightly jurisprudence enrichment (Pillar 4 increment 2).
#
# Backfills applicant profil + per-axis reliance (traegt/erwaehnt/irrelevant)
# on RechtsprechungEntry via the local Qwen worker. Cached once per decision;
# re-runs only touch new entries or entries judged by an older model. The
# service-manager auto-wakes the desktop GPU. Advisory: failures leave
# entries "ungeprueft", scoring degrades gracefully.
#
# Install (systemd user timer jurisprudence-enrichment.timer, 03:50 —
# after the 03:30 case-memory build shares the same GPU window).
set -euo pipefail

LIMIT="${JURIS_ENRICHMENT_LIMIT:-60}"
CONTAINER="${JURIS_ENRICHMENT_CONTAINER:-rechtmaschine-app}"
LOG="${JURIS_ENRICHMENT_LOG:-/var/opt/docker/rechtmaschine/rag/data/juris_enrichment.log}"

{
    echo "=== juris enrichment $(date -Is) | limit=${LIMIT} ==="
    /usr/bin/docker exec "${CONTAINER}" python /app/juris_enrichment.py --limit "${LIMIT}"
    echo "=== enrichment done $(date -Is) ==="
} >> "${LOG}" 2>&1

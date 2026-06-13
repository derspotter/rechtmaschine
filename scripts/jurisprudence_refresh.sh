#!/usr/bin/env bash
# Weekly asyl.net jurisprudence refresh.
#
# Ingests the last N days of decisions (all countries, newest-first) into the
# `jurisprudence` collection + RechtsprechungEntry. The window deliberately
# overlaps previous runs (default 45 days); the M-number/content dedup in
# jurisprudence_ingest skips everything already stored, so only genuinely new
# decisions are added. GPU-light (Gemini tag + TEI embed), so safe any time.
#
# Install (weekly, Mondays 04:00):
#   0 4 * * 1 /var/opt/docker/rechtmaschine/scripts/jurisprudence_refresh.sh
set -euo pipefail

DAYS="${JURIS_REFRESH_DAYS:-45}"
LIMIT="${JURIS_REFRESH_LIMIT:-150}"
CONTAINER="${JURIS_REFRESH_CONTAINER:-rechtmaschine-app}"
LOG="${JURIS_REFRESH_LOG:-/var/opt/docker/rechtmaschine/rag/data/jurisprudence_refresh.log}"

FROM="$(date -d "${DAYS} days ago" +%d.%m.%Y)"
{
    echo "=== jurisprudence refresh $(date -Is) | datefrom=${FROM} limit=${LIMIT} ==="
    /usr/bin/docker exec "${CONTAINER}" python /app/jurisprudence_ingest.py \
        --datefrom "${FROM}" --limit "${LIMIT}"
    echo "=== refresh done $(date -Is) ==="
} >> "${LOG}" 2>&1

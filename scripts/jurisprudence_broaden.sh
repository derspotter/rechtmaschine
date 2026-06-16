#!/usr/bin/env bash
# One-time (re-runnable) asyl.net breadth backfill.
#
# The weekly refresh (jurisprudence_refresh.sh) only pulls the last ~45 days.
# This deepens the corpus by running per-seed fulltext searches (each asyl.net
# search is capped at ~500 hits, so many narrow seeds surface decisions the
# "newest overall" window never reaches). Floor: 2021-01-01 (no older law).
#
# Dedup (M-number + content_sha256 in jurisprudence_ingest) makes this safe to
# re-run and safe against overlapping seeds. Runs host-side, one `docker exec`
# per seed, so an app-container restart only loses the in-flight seed.
#
# Usage:  scripts/jurisprudence_broaden.sh            # full set
#         JURIS_LIMIT=50 scripts/jurisprudence_broaden.sh
set -uo pipefail

DATEFROM="${JURIS_DATEFROM:-01.01.2021}"
LIMIT="${JURIS_LIMIT:-30}"
CONTAINER="${JURIS_CONTAINER:-rechtmaschine-app}"
LOG="${JURIS_LOG:-/var/opt/docker/rechtmaschine/rag/data/jurisprudence_broaden.log}"

# Herkunftsländer (high-volume asylum origins) + recurring legal topics. Each is
# a fulltext seed; breadth comes from the union across seeds (dedup merges).
SEEDS=(
  # countries
  "Syrien" "Afghanistan" "Iran" "Irak" "Türkei" "Somalia" "Eritrea"
  "Russische Föderation" "Nigeria" "Guinea" "Sierra Leone" "Pakistan"
  "Sri Lanka" "Äthiopien" "Sudan" "Libanon" "Georgien" "Nordmazedonien"
  "Albanien" "Kosovo" "Serbien" "Gambia" "Kamerun" "Kongo" "Ägypten"
  "Tunesien" "Algerien" "Marokko" "Bangladesch" "Indien" "Armenien"
  "Aserbaidschan" "Ukraine" "Venezuela" "China" "Jemen" "Bosnien"
  # topics
  "Abschiebungsverbot" "subsidiärer Schutz" "Flüchtlingseigenschaft"
  "Dublin" "Widerruf" "Folgeantrag" "§ 60 Abs. 7" "§ 60 Abs. 5"
  "Wehrdienst" "Konversion" "geschlechtsspezifische Verfolgung" "PTBS"
  "Familienasyl" "Ausbildungsduldung" "interner Schutz" "Eilantrag"
  "Zulassung der Berufung" "offensichtlich unbegründet"
  "Genitalverstümmelung" "psychische Erkrankung" "Reiseunfähigkeit"
)

{
  echo "=== jurisprudence broaden $(date -Is) | datefrom=${DATEFROM} limit=${LIMIT} seeds=${#SEEDS[@]} ==="
  for seed in "${SEEDS[@]}"; do
    echo "--- seed: ${seed} ($(date -Is)) ---"
    /usr/bin/docker exec "${CONTAINER}" python /app/jurisprudence_ingest.py \
        --query "${seed}" --datefrom "${DATEFROM}" --limit "${LIMIT}" \
        2>&1 | grep -E "asyl.net:|OK |chunks into|ERROR|Traceback" || true
  done
  echo "=== broaden done $(date -Is) ==="
} >> "${LOG}" 2>&1

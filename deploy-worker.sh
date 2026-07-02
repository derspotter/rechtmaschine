#!/usr/bin/env bash
# Restart the RM job-worker without losing in-flight jobs.
#
# The worker requeues its running job on SIGTERM (see job_worker.py), so a
# restart is safe at any time — the job just re-runs from scratch. To avoid
# needlessly re-running a long consolidation, we first wait (up to WAIT_SECS)
# for the queue to go idle, then restart regardless.
#
# Usage: ./deploy-worker.sh [--now]   (--now skips the idle wait)
set -euo pipefail

WAIT_SECS="${WAIT_SECS:-300}"
PSQL=(docker exec rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db -tA)

running() {
  "${PSQL[@]}" -c "select count(*) from memory_reflection_jobs where status in ('running','claimed');" 2>/dev/null || echo "?"
}

if [[ "${1:-}" != "--now" ]]; then
  waited=0
  while [[ "$(running)" != "0" && $waited -lt $WAIT_SECS ]]; do
    echo "job-worker busy ($(running) running) - waiting... (${waited}s/${WAIT_SECS}s, --now to skip)"
    sleep 15
    waited=$((waited + 15))
  done
fi

docker restart rechtmaschine-job-worker
echo "restarted. tail:"
sleep 2
docker logs --tail 3 rechtmaschine-job-worker 2>&1 | grep -v pydantic || true

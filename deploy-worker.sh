#!/usr/bin/env bash
# Recreate the RM job-worker without losing in-flight jobs.
#
# The worker requeues its running job on SIGTERM (see job_worker.py), so this
# is safe at any time — the job just re-runs from scratch. To avoid needlessly
# re-running a long consolidation, we first wait (up to WAIT_SECS) for the
# queue to go idle, then proceed regardless.
#
# WICHTIG: hier wird `docker compose up -d`, NICHT `docker restart` benutzt.
# `docker restart` behält die Umgebung des bestehenden Containers und zieht
# .env-Änderungen NIE nach. Genau das ist am 04.08.2026 aufgeflogen: XAI_MODEL
# stand seit dem 29.07. auf grok-4.5, der Worker war aber vom 06.07. und lief
# über Wochen weiter auf dem Fallback grok-4.3 — alle Research-Jobs mit ihm.
# Aufgefallen ist es nur, weil ein Ergebnis metadata["engines"]["grok-4.3"]
# mit model="grok-4.3" auswies, während die App-Container-Umgebung 4.5 zeigte.
#
# Usage: ./deploy-worker.sh [--now]   (--now skips the idle wait)
set -euo pipefail

COMPOSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_SECS="${WAIT_SECS:-300}"
PSQL=(docker exec rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db -tA)

# Alle Job-Tabellen, die dieser Worker abarbeitet — nicht nur die
# Memory-Reflection. Vorher zählte nur memory_reflection_jobs, wodurch ein
# laufender Research- oder Generierungsjob den Idle-Check nicht aufhielt.
running() {
  "${PSQL[@]}" -c "select
      (select count(*) from memory_reflection_jobs where status in ('running','claimed'))
    + (select count(*) from research_jobs           where status in ('running','claimed'))
    + (select count(*) from generation_jobs         where status in ('running','claimed'));" 2>/dev/null || echo "?"
}

if [[ "${1:-}" != "--now" ]]; then
  waited=0
  while [[ "$(running)" != "0" && $waited -lt $WAIT_SECS ]]; do
    echo "job-worker busy ($(running) running) - waiting... (${waited}s/${WAIT_SECS}s, --now to skip)"
    sleep 15
    waited=$((waited + 15))
  done
fi

cd "$COMPOSE_DIR"
# --force-recreate: bei reinen Code-Änderungen (bind-mount ./app) ändert sich
# die Compose-Config nicht und `up -d` wäre ein stiller No-op — der Worker
# liefe mit dem alten, beim Start importierten Modulstand weiter (so geschehen
# 05.08.2026: llm_costs-Änderung kam ohne Recreate nie im Worker an).
docker compose up -d --force-recreate job-worker
echo "recreated. XAI_MODEL im Worker: $(docker exec rechtmaschine-job-worker printenv XAI_MODEL 2>/dev/null || echo '(nicht gesetzt)')"
echo "tail:"
sleep 2
docker logs --tail 3 rechtmaschine-job-worker 2>&1 | grep -v pydantic || true

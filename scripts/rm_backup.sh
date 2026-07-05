#!/usr/bin/env bash
# Nightly Rechtmaschine backup (the critical gap from the 2026-07-04 audit).
#
# The DB is the irreplaceable asset (jurisprudence enrichment, case memory,
# Muster-Wiki, doktrin index, drafts, facets) — uploaded case documents are
# mostly recoverable from j-lawyer, so volumes only get a weekly safety net.
#
#   daily    pg_dump -Fc of the app DB + copy of app/.env (secrets)
#   Sundays  tar of uploads_data + downloaded_sources volumes
#   always   rsync everything to debian (off-host), prune by retention
#
# Install: systemd user timer rm-backup.timer, 05:15 (after doktrin-sync 04:34;
# no GPU involved). Restore recipe: see "Backups" in the rechtmaschine skill.
set -euo pipefail

ROOT="${RM_BACKUP_ROOT:-/var/opt/docker/rechtmaschine/backups}"
REMOTE="${RM_BACKUP_REMOTE:-justus@debian:rm-backups/}"
KEEP_DAILY="${RM_BACKUP_KEEP_DAILY:-14}"      # local daily dumps/env copies
KEEP_WEEKLY="${RM_BACKUP_KEEP_WEEKLY:-4}"     # local weekly volume tars
KEEP_REMOTE_DAYS="${RM_BACKUP_KEEP_REMOTE_DAYS:-45}"
LOG="${RM_BACKUP_LOG:-/var/opt/docker/rechtmaschine/rag/data/rm_backup.log}"
STAMP="$(date +%F)"

run() {
    mkdir -p "${ROOT}"/{db,env,volumes}
    chmod 700 "${ROOT}"

    # -- daily: DB dump (credentials stay inside the container env) + .env
    /usr/bin/docker exec rechtmaschine-postgres sh -c \
        'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
        > "${ROOT}/db/rm_db_${STAMP}.dump.tmp"
    mv "${ROOT}/db/rm_db_${STAMP}.dump.tmp" "${ROOT}/db/rm_db_${STAMP}.dump"
    # a -Fc dump starts with the PGDMP magic — catch silently empty/garbage dumps
    head -c 5 "${ROOT}/db/rm_db_${STAMP}.dump" | grep -q '^PGDMP' \
        || { echo "ERROR: dump lacks PGDMP magic"; exit 1; }
    install -m 600 /var/opt/docker/rechtmaschine/app/.env "${ROOT}/env/env_${STAMP}"

    # -- Sundays: volume tars (uploads ~1.1G, downloaded_sources ~7M)
    if [ "$(date +%u)" = "7" ]; then
        for vol in uploads_data downloaded_sources; do
            /usr/bin/docker run --rm -v "rechtmaschine_${vol}:/src:ro" alpine \
                tar czf - -C /src . > "${ROOT}/volumes/${vol}_${STAMP}.tgz.tmp"
            mv "${ROOT}/volumes/${vol}_${STAMP}.tgz.tmp" \
               "${ROOT}/volumes/${vol}_${STAMP}.tgz"
        done
    fi

    # -- retention (local), then ship off-host, then prune remote
    ls -1t "${ROOT}/db/"rm_db_*.dump 2>/dev/null | tail -n +$((KEEP_DAILY + 1)) | xargs -r rm --
    ls -1t "${ROOT}/env/"env_* 2>/dev/null | tail -n +$((KEEP_DAILY + 1)) | xargs -r rm --
    for vol in uploads_data downloaded_sources; do
        ls -1t "${ROOT}/volumes/${vol}"_*.tgz 2>/dev/null | tail -n +$((KEEP_WEEKLY + 1)) | xargs -r rm --
    done

    ssh "${REMOTE%%:*}" "mkdir -p ${REMOTE#*:} && chmod 700 ${REMOTE#*:}"
    rsync -a --exclude '*.tmp' "${ROOT}/" "${REMOTE}"
    # no --delete above: a compromised/empty local dir must not wipe the remote
    ssh "${REMOTE%%:*}" "find ${REMOTE#*:} -type f -mtime +${KEEP_REMOTE_DAYS} -delete"

    echo "backup ok: $(du -sh "${ROOT}" | cut -f1) local, remote pruned >${KEEP_REMOTE_DAYS}d"
}

{
    echo "=== rm backup $(date -Is) ==="
    run
    echo "=== rm backup done $(date -Is) ==="
} >> "${LOG}" 2>&1

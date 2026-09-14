#!/usr/bin/env bash
# OnFailure= handler for the nightly Rechtmaschine jobs (systemd user template
# job-failure-mail@.service). doktrin-sync failed silently on 13. and 14.09.2026;
# this mails the journal tail so a failed night is seen the next morning.
set -uo pipefail
UNIT=${1:?unit name}
HIMALAYA=/home/jay/.local/bin/himalaya
NOTIFY_TO=${JOB_FAILURE_NOTIFY_TO:-justus.spott@posteo.de}
JOURNAL=$(journalctl --user -u "$UNIT" -n 40 --no-pager 2>&1 | tail -40)
case "$UNIT" in
    doktrin-sync*) LOGFILE=/var/opt/docker/rechtmaschine/rag/data/doktrin_sync.log ;;
    jurisprudence-refresh*) LOGFILE=/var/opt/docker/rechtmaschine/rag/data/jurisprudence_refresh.log ;;
    jurisprudence-enrichment*) LOGFILE=/var/opt/docker/rechtmaschine/rag/data/juris_enrichment.log ;;
    *) LOGFILE="" ;;
esac
LOGTAIL=""
[ -n "$LOGFILE" ] && [ -r "$LOGFILE" ] && LOGTAIL=$(grep -vE "^[0-9]+/[0-9]+ ok " "$LOGFILE" | tail -25)

"$HIMALAYA" message send <<MAIL
From: Justus Spott <spott@keienborg.de>
To: $NOTIFY_TO
Subject: Rechtmaschine: Job $UNIT fehlgeschlagen

Der systemd-Job $UNIT ist mit Fehler beendet worden ($(date -Is)).

Journal (letzte Zeilen):
$JOURNAL

Log-Datei ${LOGFILE:-—} (letzte Zeilen ohne ok-Zeilen):
$LOGTAIL

Naechster Schritt: Skill rechtmaschine, references/juris-enrichment.md ("Betrieb").
MAIL

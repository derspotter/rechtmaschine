#!/usr/bin/env bash
# Hourly probe of the debian RAG store (systemd user timer rag-health-probe.timer).
#
# 12.–14.09.2026: the embedder container was down for 42 h, every retrieve
# failed with embed_failed, and nothing noticed because rag-api reported
# "healthy". This probe mails on the transition healthy -> unhealthy and on
# recovery. A sleeping worker (no ping) is normal and never an alert; the
# last observed health is kept across sleep so a recovery is still reported.
set -uo pipefail

REPO=/var/opt/docker/rechtmaschine
LOG=$REPO/rag/data/rag_health_probe.log
STATE=$REPO/rag/data/rag_health_probe.state
HOST=${RAG_PROBE_HOST:-debian}
URL=${RAG_PROBE_URL:-http://debian:8090/v1/rag/health}
HIMALAYA=/home/jay/.local/bin/himalaya
NOTIFY_TO=${RAG_PROBE_NOTIFY_TO:-justus.spott@posteo.de}

prev=$(cat "$STATE" 2>/dev/null || echo unknown)
detail=""
if ! ping -c1 -W3 "$HOST" >/dev/null 2>&1; then
    state=asleep
else
    body=$(mktemp)
    code=$(curl -sS -m 15 -o "$body" -w '%{http_code}' "$URL" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then
        state=healthy
    else
        state=unhealthy
        detail="HTTP $code $(head -c 600 "$body" | tr '\n' ' ')"
    fi
    rm -f "$body"
fi
echo "$(date -Is) prev=$prev now=$state $detail" >> "$LOG"

mail() {  # $1 subject, stdin body
    "$HIMALAYA" message send <<MAIL || echo "$(date -Is) Mail-Versand fehlgeschlagen" >> "$LOG"
From: Justus Spott <spott@keienborg.de>
To: $NOTIFY_TO
Subject: $1

$(cat)
MAIL
}

case "$state" in
    asleep) exit 0 ;;   # keep previous observed state
    unhealthy)
        if [ "$prev" != "unhealthy" ]; then
            mail "Rechtmaschine: RAG-Store auf debian NICHT gesund" <<EOM
Der stuendliche Probe-Aufruf von $URL ist fehlgeschlagen:

$detail

Folge: Rechtmaschine-Generierungen laufen OHNE Rechtsprechung/Doktrin/Kanzlei-Retrieval,
nightly doktrin-sync und Ingests schlagen fehl.

Pruefen (ssh justus@debian):
  docker ps -a --format '{{.Names}} {{.Status}}' | grep rag
  nvidia-smi   # Driver/library mismatch => Reboot noetig (12.09.2026: apt-Upgrade ohne Reboot)
  cd ~/rechtmaschine && docker compose -f rag/docker-compose.debian.yml --env-file rag/.env.debian up -d
Details: Skill ssh, Abschnitt "debian: Repo aktualisieren + RAG-API neu bauen".
EOM
        fi ;;
    healthy)
        if [ "$prev" = "unhealthy" ]; then
            mail "Rechtmaschine: RAG-Store auf debian wieder gesund" <<EOM
$URL antwortet wieder 200. Nightly doktrin-sync holt fehlende Chunks beim naechsten Lauf nach
(oder sofort: systemctl --user start doktrin-sync.service).
EOM
        fi ;;
esac
echo "$state" > "$STATE"

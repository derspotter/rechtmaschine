#!/usr/bin/env bash
# Täglicher Check: sobald xAI grok-4.5 für die EU freischaltet, XAI_MODEL
# in app/.env aktivieren, App-Stack neu starten und Justus benachrichtigen.
# Cron-Eintrag (User jay): 53 6 * * * .../scripts/grok45-flip-check.sh
# Entfernt sich nicht selbst — nach erfolgtem Flip ist der Lauf ein No-op;
# den Cron-Eintrag danach manuell austragen.
set -euo pipefail

REPO=/var/opt/docker/rechtmaschine
ENV_FILE="$REPO/app/.env"
HIMALAYA=/home/jay/.local/bin/himalaya
NOTIFY_TO=justus.spott@posteo.de

echo "[$(date -Is)] grok45-flip-check"

# Schon umgestellt? Dann nichts tun.
if grep -q '^XAI_MODEL=grok-4.5' "$ENV_FILE"; then
    echo "bereits aktiv, nichts zu tun"
    exit 0
fi

KEY=$(grep -m1 '^XAI_API_KEY=' "$ENV_FILE" | cut -d= -f2-)
RESP=$(/usr/bin/curl -s -m 60 https://api.x.ai/v1/chat/completions \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"model":"grok-4.5","messages":[{"role":"user","content":"Antworte nur mit OK"}],"max_tokens":5}') || {
    echo "curl fehlgeschlagen: $RESP"
    exit 0
}

if ! echo "$RESP" | grep -q '"choices"'; then
    echo "noch gesperrt: $(echo "$RESP" | head -c 200)"
    exit 0
fi

echo "grok-4.5 antwortet — stelle um"
sed -i 's/^# XAI_MODEL=grok-4.5/XAI_MODEL=grok-4.5/' "$ENV_FILE"
grep -q '^XAI_MODEL=grok-4.5' "$ENV_FILE" || { echo "sed-Flip fehlgeschlagen"; exit 1; }

cd "$REPO"
/usr/bin/docker compose restart app job-worker

"$HIMALAYA" message send <<EOF || echo "Mail-Versand fehlgeschlagen"
From: Justus Spott <spott@keienborg.de>
To: $NOTIFY_TO
Subject: Rechtmaschine: Grok 4.5 ist jetzt aktiv

xAI hat grok-4.5 fuer die EU freigeschaltet. Der taegliche Check hat
XAI_MODEL=grok-4.5 in app/.env aktiviert und app + job-worker neu
gestartet. Research laeuft ab jetzt auf Grok 4.5.

Bitte gelegentlich einen Research-Job pruefen und den Cron-Eintrag
(crontab -e, grok45-flip-check) austragen.
EOF

echo "Flip abgeschlossen"

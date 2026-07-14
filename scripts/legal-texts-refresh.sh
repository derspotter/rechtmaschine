#!/usr/bin/env bash
# Wöchentlicher Normtext-Refresh: Kette NeuRIS → gesetze-im-internet.de (GII)
# → GitHub. NeuRIS-Testphase führt GG/AsylbLG noch nicht — die kommen von GII.
# Meldet Fassungsänderungen per Mail an Justus — eine Gesetzesänderung an
# AsylG/AufenthG ist anwaltlich relevante Nachricht, nicht nur Ops.
set -euo pipefail

REPO=/var/opt/docker/rechtmaschine
PY=$REPO/.venv/bin/python
LOG=$REPO/rag/data/legal_texts_refresh.log
HIMALAYA=/home/jay/.local/bin/himalaya
NOTIFY_TO=justus.spott@posteo.de

versions() {
    $PY - <<'EOF'
import sys
sys.path.insert(0, "/var/opt/docker/rechtmaschine/app")
from legal_texts.downloader import LAWS, stored_version
for law in LAWS:
    print(f"{law}={stored_version(law) or 'unbekannt'}")
EOF
}

{
    echo "=== legal-texts refresh $(date -Is) ==="
    BEFORE=$(versions)
    $PY - <<'EOF'
import sys
sys.path.insert(0, "/var/opt/docker/rechtmaschine/app")
from legal_texts.downloader import download_all_laws_sync
status = download_all_laws_sync(force=True)
failed = [law for law, ok in status.items() if not ok]
if failed:
    raise SystemExit(f"Refresh unvollständig: {failed}")
EOF
    AFTER=$(versions)
    echo "vorher:  $BEFORE" | tr '\n' ' '; echo
    echo "nachher: $AFTER" | tr '\n' ' '; echo

    if [ "$BEFORE" != "$AFTER" ]; then
        echo "Fassungsänderung erkannt — Mail an $NOTIFY_TO"
        "$HIMALAYA" message send <<EOF || echo "Mail-Versand fehlgeschlagen"
From: Justus Spott <spott@keienborg.de>
To: $NOTIFY_TO
Subject: Rechtmaschine: Normtext-Fassung geaendert

Der woechentliche Normtext-Refresh hat eine neue Fassung uebernommen.

Vorher:
$BEFORE

Nachher:
$AFTER

Quelle: NeuRIS (rechtsinformationen.bund.de); GG/AsylbLG von
gesetze-im-internet.de. Die Rechtmaschine-Prompts nutzen ab sofort den
neuen Stand.
Bitte pruefen, ob die Aenderung laufende Verfahren beruehrt.
EOF
    else
        echo "keine Fassungsänderung"
    fi
} >> "$LOG" 2>&1

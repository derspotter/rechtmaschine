# systemd setup for service_manager on desktop

This guide is meant to be executed on the desktop host.
If you do not have sudo, use the user-service option below.

## Option A) System service (sudo)

### 1) Install the unit file

Copy the template unit file from the repo:

sudo cp /home/jayjag/rechtmaschine/systemd/service-manager@.service /etc/systemd/system/

Optional: add environment overrides (ANON_BACKEND, OLLAMA_URL, OLLAMA_MODEL, etc.):

sudo mkdir -p /etc/rechtmaschine
sudo tee /etc/rechtmaschine/service-manager.env >/dev/null <<'EOF'
# ANON_BACKEND=flair
# OLLAMA_URL=http://localhost:11435
# OLLAMA_MODEL=qwen3:14b
EOF

Set up the log file:

sudo touch /var/log/service-manager.log
sudo chown jayjag:jayjag /var/log/service-manager.log

### 2) Enable and start

sudo systemctl daemon-reload
sudo systemctl enable --now service-manager@jayjag

### 3) Verify status

systemctl status service-manager@jayjag --no-pager

### 4) Live logs

journalctl -u service-manager@jayjag -f

### 5) Stop/restart

sudo systemctl stop service-manager@jayjag
sudo systemctl restart service-manager@jayjag

## Option B) User service (no sudo)

### 1) Install the unit file

mkdir -p ~/.config/systemd/user ~/.config/rechtmaschine ~/.local/state/rechtmaschine
cp /home/jayjag/rechtmaschine/systemd/user/service-manager.service ~/.config/systemd/user/

Optional: add environment overrides (ANON_BACKEND, OLLAMA_URL, OLLAMA_MODEL, etc.):

cat > ~/.config/rechtmaschine/service-manager.env <<'EOF'
# ANON_BACKEND=flair
# OLLAMA_URL=http://localhost:11435
# OLLAMA_MODEL=qwen3:14b
EOF

Set up the log file:

touch ~/.local/state/rechtmaschine/service-manager.log

### 2) Enable and start

systemctl --user daemon-reload
systemctl --user enable --now service-manager

### 3) Verify status

systemctl --user status service-manager --no-pager

### 4) Live logs

journalctl --user -u service-manager -f

### 5) Stop/restart

systemctl --user stop service-manager
systemctl --user restart service-manager

## Notes

- The template is an instance unit; replace `jayjag` with your username.
- The system service template uses `/home/%i/rechtmaschine`; adjust if your home or repo path differs.
- If you prefer no log file, remove StandardOutput/StandardError and use only journalctl.
- User services stop when you log out unless `loginctl enable-linger $USER` is set (requires sudo).
- OCR is now run on the host via `ocr/run_hpi_service.sh` (HPI + PaddleOCR 3.3.3). The
  `paddlex-ocr-hpi` container is deprecated; stop/disable it if it's still running.

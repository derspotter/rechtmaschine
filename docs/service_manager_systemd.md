# systemd setup for service_manager on desktop

This guide is meant to be executed on the desktop host.

## 1) Create the unit file

Run:

sudo tee /etc/systemd/system/service-manager.service >/dev/null <<'EOF'
[Unit]
Description=Rechtmaschine Service Manager
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jayjag
WorkingDirectory=/home/jayjag/rechtmaschine
ExecStart=/bin/bash -lc "source .venv/bin/activate && python service_manager.py"
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/service-manager.log
StandardError=append:/var/log/service-manager.log

[Install]
WantedBy=multi-user.target
EOF

sudo touch /var/log/service-manager.log
sudo chown jayjag:jayjag /var/log/service-manager.log

## 2) Enable and start

sudo systemctl daemon-reload
sudo systemctl enable service-manager
sudo systemctl start service-manager

## 3) Verify status

systemctl status service-manager --no-pager

## 4) Live logs

journalctl -u service-manager -f

## 5) Stop/restart

sudo systemctl stop service-manager
sudo systemctl restart service-manager

## Notes

- Adjust User/WorkingDirectory if your repo path differs.
- If you prefer no log file, remove StandardOutput/StandardError and use only journalctl.

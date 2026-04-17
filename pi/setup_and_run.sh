#!/bin/bash
# setup_and_run.sh — one-shot installer for the Pi scanner box.
#
#   1. Installs comitup and configures it with
#        ap_name: SETUP_AP_<nnn>      (3-digit machine-id hash)
#        ap_password: poker
#      so the Pi will self-host a Wi-Fi setup AP whenever no known
#      network is in range.
#   2. Installs a systemd service for scan_controller.py that starts
#      after comitup has settled and auto-restarts on any exit.
#
# Re-runnable / idempotent. Run as root:
#   sudo ./setup_and_run.sh

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

RUN_USER="${SUDO_USER:-srw}"
REPO_DIR="/home/${RUN_USER}/CardGameRemote"
PI_DIR="${REPO_DIR}/pi"

if [ ! -f "${PI_DIR}/scan_controller.py" ]; then
    echo "Expected ${PI_DIR}/scan_controller.py — is the repo cloned at ${REPO_DIR}?" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. comitup — captive-portal Wi-Fi setup
# ---------------------------------------------------------------------------

if ! dpkg -s comitup >/dev/null 2>&1; then
    echo "==> Installing comitup…"
    apt-get update
    apt-get install -y comitup
else
    echo "==> comitup already installed"
fi

CONF=/etc/comitup.conf

# Upsert a `key: value` line in $CONF. Replaces an active line if present,
# uncomments+replaces a commented default if present, otherwise appends.
set_conf_key() {
    local key="$1" value="$2" file="$3"
    if grep -qE "^[[:space:]]*${key}:" "$file"; then
        sed -i -E "s|^[[:space:]]*${key}:.*|${key}: ${value}|" "$file"
    elif grep -qE "^[[:space:]]*#[[:space:]]*${key}:" "$file"; then
        sed -i -E "s|^[[:space:]]*#[[:space:]]*${key}:.*|${key}: ${value}|" "$file"
    else
        echo "${key}: ${value}" >> "$file"
    fi
}

echo "==> Configuring ${CONF}"
set_conf_key "ap_name"     "SETUP_AP_<nnn>" "$CONF"
set_conf_key "ap_password" "poker"          "$CONF"

systemctl restart comitup

# ---------------------------------------------------------------------------
# 2. scan_controller systemd service
# ---------------------------------------------------------------------------

SERVICE=/etc/systemd/system/scan_controller.service
echo "==> Writing ${SERVICE}"
cat > "$SERVICE" <<EOF
[Unit]
Description=Card scanner controller (CardGameRemote)
After=network.target comitup.service

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PI_DIR}
ExecStart=/usr/bin/python3 ${PI_DIR}/scan_controller.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable scan_controller.service
systemctl restart scan_controller.service

echo
echo "==> Done."
echo "   Comitup AP:       SETUP_AP_<nnn>   (password: poker)"
echo "   scan_controller:  http://$(hostname).local:8080/"
echo
echo "Check status:"
echo "   sudo systemctl status scan_controller"
echo "   sudo journalctl -u scan_controller -f"

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
# 1.5. Python deps — picamera2/cv2/flask/gpiozero from apt, ultralytics
#      via pip with the CPU-only PyTorch wheel index. Going through the
#      default PyPI wheel for ultralytics drags in the CUDA-pinned torch
#      build (nvidia-cudnn-cu12, nvidia-cublas-cu12, etc.) which won't
#      run on the Pi anyway and chews through ~6 GB of SD card.
# ---------------------------------------------------------------------------

echo "==> Installing apt-provided Python packages"
apt-get install -y \
    python3-opencv \
    python3-picamera2 \
    python3-flask \
    python3-gpiozero \
    python3-pip \
    python3-pil

# flask-sock has no apt package; small enough to pull via pip directly.
echo "==> Installing flask-sock (pip)"
pip3 install --break-system-packages --quiet flask-sock

# Ultralytics + CPU-only torch. Two steps so torch comes from the
# pytorch.org CPU wheel index — otherwise pip resolves the manylinux
# wheel that depends on the nvidia-* CUDA packages and the install
# fills the SD card before the Pi even boots a card scan.
if ! python3 -c "import ultralytics" 2>/dev/null; then
    echo "==> Installing CPU-only PyTorch (pytorch.org wheel index)"
    pip3 install --break-system-packages --quiet \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

    echo "==> Installing ultralytics (PyPI; torch already in place)"
    pip3 install --break-system-packages --quiet ultralytics
else
    echo "==> ultralytics already installed"
fi

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
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -u ${PI_DIR}/scan_controller.py
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

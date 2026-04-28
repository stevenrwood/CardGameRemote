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
# 1.25. Enable both CSI cameras on the CM4 IO Board.
#
# The Pi 4/5 auto-detects a single camera via camera_auto_detect=1, but
# the CM4 IO Board has two CAM connectors and neither is detected
# automatically — you have to explicitly load the IMX708 (Camera Module 3)
# overlay for each. Without these three lines, picamera2 returns an empty
# global_camera_info() and scan_controller stays in degraded mode.
#
# Substitute imx477 (HQ) or imx219 (v2) for whichever sensor you have.
# Idempotent: each line is upserted in /boot/firmware/config.txt under
# the [all] section. A reboot is required when any line changes;
# the script tracks that and prints a reminder at the end.
# ---------------------------------------------------------------------------

# Bookworm/Trixie put it at /boot/firmware; older Pi OS at /boot.
if [ -f /boot/firmware/config.txt ]; then
    BOOT_CFG=/boot/firmware/config.txt
elif [ -f /boot/config.txt ]; then
    BOOT_CFG=/boot/config.txt
else
    echo "WARNING: no boot config.txt found; skipping camera-overlay setup" >&2
    BOOT_CFG=""
fi

CAMERA_CFG_CHANGED=0

# Ensure exactly one line `<full_line>` exists in $BOOT_CFG. If a line
# matching `^<line_prefix>` is already there with a different value,
# replace it. Otherwise append. Returns nonzero exit on any actual change.
ensure_config_line() {
    local file="$1" line_prefix="$2" full_line="$3"
    if grep -qFx "$full_line" "$file"; then
        return 1   # already present, no change
    fi
    if grep -qE "^${line_prefix}" "$file"; then
        sed -i -E "s|^${line_prefix}.*|${full_line}|" "$file"
    else
        # Append, prefixed by a header comment if first edit.
        if ! grep -qF "# CardGameRemote: CM4 dual-camera setup" "$file"; then
            printf "\n# CardGameRemote: CM4 dual-camera setup\n" >> "$file"
        fi
        echo "$full_line" >> "$file"
    fi
    return 0   # changed
}

if [ -n "$BOOT_CFG" ]; then
    echo "==> Ensuring CM4 dual-camera dtoverlay in ${BOOT_CFG}"
    if ensure_config_line "$BOOT_CFG" "camera_auto_detect=" "camera_auto_detect=0"; then
        CAMERA_CFG_CHANGED=1
    fi
    if ensure_config_line "$BOOT_CFG" "dtoverlay=imx708,cam0" "dtoverlay=imx708,cam0"; then
        CAMERA_CFG_CHANGED=1
    fi
    if ensure_config_line "$BOOT_CFG" "dtoverlay=imx708,cam1" "dtoverlay=imx708,cam1"; then
        CAMERA_CFG_CHANGED=1
    fi
    if [ "$CAMERA_CFG_CHANGED" -eq 1 ]; then
        echo "    -> ${BOOT_CFG} updated; reboot required for cameras to enumerate"
    else
        echo "    -> already configured, no change"
    fi
fi

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

if [ "${CAMERA_CFG_CHANGED:-0}" -eq 1 ]; then
    echo
    echo "*** REBOOT REQUIRED ***"
    echo "   ${BOOT_CFG} was updated to enable both CM4 CSI cameras."
    echo "   Reboot before scan_controller will see the cameras:"
    echo "   sudo reboot"
fi

#!/bin/bash
set -e

# Train a scanner-specific YOLO model from the Pi's /train template captures.
# Expects the Pi templates rsynced to a local directory first:
#
#   rsync -av srw@pokerbuddy.local:~/CardGameRemote/pi/card_recognition/reference/slot_templates/ \
#         ~/pi_slot_templates/
#
# Usage:
#   host/train_pi_yolo.sh                       # uses default ~/pi_slot_templates
#   host/train_pi_yolo.sh /path/to/templates
#
# Output: host/models/pi_card_detector.pt  (yolov8n, ~6MB, Pi-friendly)

cd "$(dirname "$0")"

SRC="${1:-$HOME/pi_slot_templates}"
if [[ ! -d "$SRC" ]]; then
    echo "Templates not found: $SRC"
    echo "rsync them from the Pi first (see header comment)."
    exit 1
fi

# Prefer the host virtualenv if present, else system python with a fallback
# to --break-system-packages on PEP 668 Pythons.
if [[ -x ".venv/bin/python3" ]]; then
    PY=".venv/bin/python3"
else
    PY=python3
fi

if ! "$PY" -c "import ultralytics, cv2, numpy" 2>/dev/null; then
    echo "Installing training deps..."
    "$PY" -m pip install --quiet --break-system-packages ultralytics opencv-python numpy
fi

echo
echo "=== Preparing Pi YOLO dataset from $SRC ==="
echo
"$PY" prepare_pi_training.py --src "$SRC" --out pi_yolo_dataset

echo
echo "=== Training yolov8n on Pi crops ==="
echo
BASE_MODEL="${BASE_MODEL:-yolov8n.pt}"
"$PY" -c "
from pathlib import Path
from ultralytics import YOLO
model = YOLO('$BASE_MODEL')
model.train(
    data='pi_yolo_dataset/data.yaml',
    epochs=150,
    imgsz=320,          # slot crops are roughly card-shaped + small
    batch=32,
    patience=25,
    name='pi_card_detector',
    project='pi_yolo_runs',
    device='mps',
    exist_ok=True,
    # The training set is tiny (260 images), so lean on augmentation.
    hsv_h=0.02, hsv_s=0.6, hsv_v=0.5,
    degrees=4.0, translate=0.08, scale=0.25, perspective=0.0002,
    fliplr=0.0, flipud=0.0,  # cards always land in one orientation
    mosaic=0.0, mixup=0.0,   # those hurt more than help on single-card crops
)

save = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
dst = Path('models') / 'pi_card_detector.pt'
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(save.read_bytes())
print()
print('=== Training complete ===')
print(f'Best weights: {save}')
print(f'Copied to:   {dst.resolve()}')
"

echo
echo "Next: copy host/models/pi_card_detector.pt onto the Pi at"
echo "      pi/models/pi_card_detector.pt, install ultralytics on the Pi,"
echo "      and restart scan_controller.py."

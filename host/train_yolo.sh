#!/bin/bash
#
# Prepare training data and train YOLO model for card recognition.
# Run from the host/ directory.
#
# Usage:
#   ./train_yolo.sh              # prepare + train
#   ./train_yolo.sh --prepare    # prepare only (show distribution)
#   ./train_yolo.sh --train      # train only (skip prepare)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_DIR/.venv/bin"

cd "$SCRIPT_DIR"

# Ensure dependencies
"$VENV/pip" install --quiet ultralytics

if [[ "$1" != "--train" ]]; then
    echo ""
    echo "=== Preparing dataset ==="
    echo ""
    "$VENV/python" prepare_training.py
    echo ""
fi

if [[ "$1" == "--prepare" ]]; then
    echo "Done. Run with --train or no args to train."
    exit 0
fi

echo ""
echo "=== Training YOLO model ==="
echo ""

"$VENV/python" -c "
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='yolo_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    device='mps',
    patience=20,
    batch=16,
    name='card_detector',
    project='yolo_runs',
)

# Show results
print()
print('=== Training complete ===')
print(f'Best model: yolo_runs/card_detector/weights/best.pt')
print()

# Validate
model = YOLO('yolo_runs/card_detector/weights/best.pt')
metrics = model.val()
print(f'mAP50: {metrics.box.map50:.3f}')
print(f'mAP50-95: {metrics.box.map:.3f}')
"

echo ""
echo "Done! Model saved to: yolo_runs/card_detector/weights/best.pt"
echo ""

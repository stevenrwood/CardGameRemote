#!/bin/bash
set -e

cd "$(dirname "$0")"

pip3 install --quiet ultralytics opencv-python numpy

if [[ "$1" != "--train" ]]; then
    echo ""
    echo "=== Preparing dataset ==="
    echo ""
    python3 prepare_training.py
    echo ""
fi

if [[ "$1" == "--prepare" ]]; then
    echo "Done. Run with --train or no args to train."
    exit 0
fi

echo ""
echo "=== Training YOLO model ==="
echo ""

# Pick base model. Default is yolov8s (small, ~22MB, 4x the params of current nano).
# To fine-tune from existing trained model: BASE_MODEL=models/card_detector.pt
# To stay with nano: BASE_MODEL=yolov8n.pt
# For bigger: BASE_MODEL=yolov8m.pt (medium, ~52MB)
BASE_MODEL="${BASE_MODEL:-yolov8s.pt}"
echo "Base model: $BASE_MODEL"

python3 -c "
import os
from ultralytics import YOLO

base = os.environ.get('BASE_MODEL', 'yolov8s.pt')
model = YOLO(base)
results = model.train(
    data='yolo_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    patience=20,
    batch=16,
    name='card_detector',
    project='yolo_runs',
    device='mps',
    exist_ok=True,
)

best = 'yolo_runs/card_detector/weights/best.pt'
print()
print('=== Training complete ===')
print(f'Best model: {best}')
print()

model = YOLO(best)
metrics = model.val()
print(f'mAP50: {metrics.box.map50:.3f}')
print(f'mAP50-95: {metrics.box.map:.3f}')
"

echo ""
echo "Done! Model saved to: yolo_runs/card_detector/weights/best.pt"
echo ""

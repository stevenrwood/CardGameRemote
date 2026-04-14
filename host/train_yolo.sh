#!/bin/bash
set -e

cd "$(dirname "$0")"

# Parse args:
#   --train            skip dataset prep, just train
#   --prepare          just prep the dataset, don't train
#   --max-per-class N  cap each class to N images (default: balance to min count)
#   All other args forwarded to prepare_training.py

PREP_ARGS=()
SKIP_PREP=""
PREP_ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train)
            SKIP_PREP=1
            shift
            ;;
        --prepare)
            PREP_ONLY=1
            shift
            ;;
        --max-per-class)
            PREP_ARGS+=("--max-per-class" "$2")
            shift 2
            ;;
        *)
            PREP_ARGS+=("$1")
            shift
            ;;
    esac
done

pip3 install --quiet ultralytics opencv-python numpy

if [[ -z "$SKIP_PREP" ]]; then
    echo ""
    echo "=== Preparing dataset ==="
    echo ""
    python3 prepare_training.py "${PREP_ARGS[@]}"
    echo ""
fi

if [[ -n "$PREP_ONLY" ]]; then
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
from pathlib import Path
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

# Resolve the actual save directory reported by the trainer; newer
# ultralytics prepends 'runs/detect/' to the project path.
save_dir = Path(getattr(model.trainer, 'save_dir', 'yolo_runs/card_detector'))
best = save_dir / 'weights' / 'best.pt'
print()
print('=== Training complete ===')
print(f'Best model: {best}')
print()

model = YOLO(str(best))
metrics = model.val()
print(f'mAP50: {metrics.box.map50:.3f}')
print(f'mAP50-95: {metrics.box.map:.3f}')
"

echo ""
echo "Done! Check training output above for the exact best.pt path."
echo ""

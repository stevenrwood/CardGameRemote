"""
YOLO-based card detector for Pi scanner slot crops.

Loads pi/models/pi_card_detector.pt (a yolov8n trained specifically on
Pi scanner box captures) and runs inference per slot crop. Much higher
baseline accuracy than the template matcher on Pi camera input.
"""

from pathlib import Path

import cv2
import numpy as np


class YoloDetector:
    RANK_RE_MAP = {"A": "A", "K": "K", "Q": "Q", "J": "J"}

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self.names = {}
        if not self.model_path.exists():
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed — run "
                "`pip3 install --break-system-packages ultralytics` on the Pi"
            )
        self.model = YOLO(str(self.model_path))
        # After a prediction the model exposes class names on the result;
        # also available on the loaded model directly.
        self.names = getattr(self.model, "names", {}) or {}

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, crop_bgr: np.ndarray):
        """Return (rank, suit, confidence) or None on no-detection.

        Class names are of the form "Kd", "10s", etc. — we split into rank +
        suit. imgsz=320 matches the training config.
        """
        if self.model is None or crop_bgr is None or crop_bgr.size == 0:
            return None
        results = self.model.predict(crop_bgr, conf=0.25, imgsz=320, verbose=False)
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        # Take the highest-confidence box
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
        if confs.size == 0:
            return None
        idx = int(np.argmax(confs))
        cls_idx = int(clss[idx])
        conf = float(confs[idx])
        name = self.names.get(cls_idx) or results[0].names.get(cls_idx)
        if not name:
            return None
        rank, suit_letter = name[:-1], name[-1].lower()
        suit = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}.get(suit_letter)
        if not suit:
            return None
        return rank, suit, conf

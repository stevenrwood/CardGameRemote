#!/usr/bin/env python3
"""
Overhead Camera Card Recognition Test Harness

Captures video from a ceiling-mounted Logitech Brio 4K camera, monitors
designated landing zones on a poker table for card placement, and uses
Claude's vision API to identify cards.

Usage:
    python overhead_test.py [--camera 1] [--threshold 30.0]

Controls:
    c = enter calibration mode
    r = reset/recapture baselines for empty zones
    q = quit
    s = take a snapshot and save it
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

PLAYER_NAMES = ["Steve", "Bill", "David", "Joe", "Rodney"]
NUM_ZONES = len(PLAYER_NAMES)

DEFAULT_CAMERA_INDEX = 1
DEFAULT_THRESHOLD = 30.0        # mean absolute difference to trigger detection
CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
TRAINING_DIR = Path(__file__).parent / "training_data"
CONFIG_FILE = Path(__file__).parent.parent / "local" / "config.json"

MODEL = "claude-sonnet-4-20250514"

COLOR_WHITE  = (255, 255, 255)
COLOR_GREEN  = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED    = (0, 0, 255)
COLOR_CYAN   = (255, 255, 0)

# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

class Calibration:
    """Holds the felt-circle and landing-zone geometry."""

    def __init__(self):
        self.circle_center: tuple[int, int] | None = None
        self.circle_radius: int | None = None
        self.zones: list[dict] = []          # {"name", "x1", "y1", "x2", "y2"}

    # -- persistence --------------------------------------------------------

    def save(self, path: Path = CALIBRATION_FILE):
        data = {
            "circle_center": list(self.circle_center) if self.circle_center else None,
            "circle_radius": self.circle_radius,
            "zones": self.zones,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[calibration] saved to {path}")

    def load(self, path: Path = CALIBRATION_FILE) -> bool:
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        cc = data.get("circle_center")
        self.circle_center = tuple(cc) if cc else None
        self.circle_radius = data.get("circle_radius")
        self.zones = data.get("zones", [])
        print(f"[calibration] loaded from {path}")
        return True

    @property
    def is_complete(self) -> bool:
        return (
            self.circle_center is not None
            and self.circle_radius is not None
            and len(self.zones) == NUM_ZONES
        )


# ---------------------------------------------------------------------------
# Zone monitor — change detection + Claude API
# ---------------------------------------------------------------------------

class ZoneMonitor:
    """Tracks per-zone baselines and triggers card recognition."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.baselines: dict[str, np.ndarray] = {}       # name -> baseline crop
        self.last_card: dict[str, str] = {}               # name -> "Ace of Hearts"
        self.zone_state: dict[str, str] = {}              # name -> "empty"|"processing"|"recognized"
        self.pending: dict[str, bool] = {}                # name -> True if API call in flight
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key and CONFIG_FILE.exists():
                with open(CONFIG_FILE) as f:
                    cfg = json.load(f)
                    api_key = cfg.get("anthropic_api_key")
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else None
        return self._client

    def capture_baselines(self, frame: np.ndarray, zones: list[dict]):
        """Capture empty-zone baselines from current frame."""
        for zone in zones:
            crop = self._crop_zone(frame, zone)
            self.baselines[zone["name"]] = crop.copy()
            self.zone_state[zone["name"]] = "empty"
            self.last_card[zone["name"]] = ""
            self.pending[zone["name"]] = False
        print("[baselines] captured for all zones")

    def check_zones(self, frame: np.ndarray, zones: list[dict]):
        """Compare each zone against its baseline; trigger recognition if changed."""
        for zone in zones:
            name = zone["name"]
            if name not in self.baselines:
                continue
            if self.pending.get(name, False):
                continue  # API call already in flight

            crop = self._crop_zone(frame, zone)
            diff = cv2.absdiff(crop, self.baselines[name])
            mean_diff = float(np.mean(diff))

            if mean_diff > self.threshold:
                if self.zone_state.get(name) != "processing":
                    self.zone_state[name] = "processing"
                    self.pending[name] = True
                    # Fire recognition in a background thread
                    t = Thread(target=self._recognize, args=(name, crop.copy()), daemon=True)
                    t.start()
            else:
                # Zone returned to baseline (card removed)
                if self.zone_state.get(name) == "recognized":
                    self.zone_state[name] = "empty"
                    self.last_card[name] = ""

    def _crop_zone(self, frame: np.ndarray, zone: dict) -> np.ndarray:
        x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
        return frame[y1:y2, x1:x2]

    def _recognize(self, name: str, crop: np.ndarray):
        """Send crop to Claude vision API, announce result."""
        t0 = time.time()
        try:
            # Show the crop being submitted
            cv2.imshow(f"Zone: {name}", crop)

            # Encode as JPEG for the API
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                print(f"[{name}] failed to encode crop")
                return
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            response = self.client.messages.create(
                model=MODEL,
                max_tokens=64,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Identify the single playing card in this image. "
                                    "Return only the rank and suit, e.g. 'Ace of Hearts'. "
                                    "If no card is clearly visible, say 'No card'."
                                ),
                            },
                        ],
                    }
                ],
            )

            result = response.content[0].text.strip()
            elapsed = time.time() - t0

            self.last_card[name] = result
            self.zone_state[name] = "recognized"
            print(f"[{name}] {result}  ({elapsed:.2f}s)")

            # Save training data
            self._save_training(name, crop, result)

            # Voice announcement
            self._announce(name, result)

        except Exception as exc:
            print(f"[{name}] API error: {exc}")
            self.zone_state[name] = "empty"
        finally:
            self.pending[name] = False

    def _save_training(self, name: str, crop: np.ndarray, result: str):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_result = result.replace(" ", "_").replace("/", "-")
        stem = f"{ts}_{name}_{safe_result}"
        img_path = TRAINING_DIR / f"{stem}.jpg"
        txt_path = TRAINING_DIR / f"{stem}.txt"
        cv2.imwrite(str(img_path), crop)
        txt_path.write_text(result)

    def _announce(self, name: str, result: str):
        if "no card" in result.lower():
            return
        phrase = f"{name} has the {result}"
        subprocess.Popen(["say", phrase], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Calibration UI (interactive clicks)
# ---------------------------------------------------------------------------

class CalibrationUI:
    """Walks the user through defining the felt circle and landing zones."""

    def __init__(self, calibration: Calibration):
        self.cal = calibration
        self.active = False
        self._step = ""          # "circle_center", "circle_edge", "zone_tl", "zone_br"
        self._zone_idx = 0
        self._zone_tl: tuple[int, int] | None = None
        self._instructions = ""

    def start(self):
        self.active = True
        self.cal.circle_center = None
        self.cal.circle_radius = None
        self.cal.zones = []
        self._zone_idx = 0
        self._step = "circle_center"
        self._instructions = "Click the CENTER of the felt circle"
        print(f"[calibration] {self._instructions}")

    def handle_click(self, x: int, y: int):
        if not self.active:
            return

        if self._step == "circle_center":
            self.cal.circle_center = (x, y)
            self._step = "circle_edge"
            self._instructions = "Click the EDGE of the felt circle"
            print(f"[calibration] {self._instructions}")

        elif self._step == "circle_edge":
            cx, cy = self.cal.circle_center
            self.cal.circle_radius = int(np.hypot(x - cx, y - cy))
            self._step = "zone_tl"
            self._zone_idx = 0
            self._instructions = f"Click TOP-LEFT of {PLAYER_NAMES[0]}'s zone"
            print(f"[calibration] {self._instructions}")

        elif self._step == "zone_tl":
            self._zone_tl = (x, y)
            name = PLAYER_NAMES[self._zone_idx]
            self._step = "zone_br"
            self._instructions = f"Click BOTTOM-RIGHT of {name}'s zone"
            print(f"[calibration] {self._instructions}")

        elif self._step == "zone_br":
            name = PLAYER_NAMES[self._zone_idx]
            tl = self._zone_tl
            self.cal.zones.append({
                "name": name,
                "x1": min(tl[0], x),
                "y1": min(tl[1], y),
                "x2": max(tl[0], x),
                "y2": max(tl[1], y),
            })
            print(f"[calibration] zone '{name}' defined")

            self._zone_idx += 1
            if self._zone_idx < NUM_ZONES:
                self._step = "zone_tl"
                self._instructions = f"Click TOP-LEFT of {PLAYER_NAMES[self._zone_idx]}'s zone"
                print(f"[calibration] {self._instructions}")
            else:
                self.cal.save()
                self.active = False
                self._step = ""
                self._instructions = ""
                print("[calibration] complete!")

    def draw_instructions(self, frame: np.ndarray):
        if self.active and self._instructions:
            cv2.putText(
                frame, f"CALIBRATION: {self._instructions}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_CYAN, 3,
            )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_overlay(frame: np.ndarray, cal: Calibration, monitor: ZoneMonitor):
    """Draw circle boundary, zone rectangles, labels, and card results."""
    # Felt circle
    if cal.circle_center and cal.circle_radius:
        cv2.circle(frame, cal.circle_center, cal.circle_radius, COLOR_WHITE, 2)

    # Landing zones
    for zone in cal.zones:
        name = zone["name"]
        x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]

        state = monitor.zone_state.get(name, "empty")
        if state == "recognized":
            color = COLOR_GREEN
        elif state == "processing":
            color = COLOR_YELLOW
        else:
            color = COLOR_WHITE

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Player name label (above the rectangle)
        cv2.putText(frame, name, (x1, y1 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Card result (below the rectangle)
        card = monitor.last_card.get(name, "")
        if card:
            cv2.putText(frame, card, (x1, y2 + 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overhead camera card recognition test harness")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f"Camera index (default {DEFAULT_CAMERA_INDEX})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Change detection threshold (default {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    # -- Ensure training dir exists -----------------------------------------
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # -- Open camera --------------------------------------------------------
    print(f"[camera] opening index {args.camera} ...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open camera index {args.camera}")

    # Request 4K
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] resolution: {actual_w}x{actual_h}")

    # -- Load calibration ---------------------------------------------------
    cal = Calibration()
    cal.load()

    # -- Monitors and UI ----------------------------------------------------
    monitor = ZoneMonitor(threshold=args.threshold)
    cal_ui = CalibrationUI(cal)

    # Mouse callback
    window_name = "Overhead Card Scanner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Translate click from display coordinates to frame coordinates
            # We'll compute this based on the actual resize ratio
            win_w, win_h = 1280, 720
            scale_x = actual_w / win_w
            scale_y = actual_h / win_h
            fx = int(x * scale_x)
            fy = int(y * scale_y)
            if cal_ui.active:
                cal_ui.handle_click(fx, fy)

    cv2.setMouseCallback(window_name, on_mouse)

    # -- Capture initial baselines if calibration already loaded ------------
    baselines_captured = False

    print("\n--- Overhead Card Scanner ---")
    print("  c = calibrate | r = reset baselines | s = snapshot | q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[camera] frame grab failed, retrying ...")
            time.sleep(0.1)
            continue

        # Capture baselines on first good frame after calibration loaded
        if cal.is_complete and not baselines_captured:
            monitor.capture_baselines(frame, cal.zones)
            baselines_captured = True

        # -- Change detection (only when not calibrating) -------------------
        if cal.is_complete and not cal_ui.active and baselines_captured:
            monitor.check_zones(frame, cal.zones)

        # -- Draw overlay ---------------------------------------------------
        display = frame.copy()
        draw_overlay(display, cal, monitor)
        cal_ui.draw_instructions(display)

        # Show FPS
        # (lightweight — no rolling average to keep things simple)
        cv2.putText(display, f"{actual_w}x{actual_h}", (actual_w - 300, actual_h - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        cv2.imshow(window_name, display)

        # -- Key handling ---------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            cal_ui.start()
            baselines_captured = False

        elif key == ord("r"):
            if cal.is_complete:
                ret2, fresh = cap.read()
                if ret2:
                    monitor.capture_baselines(fresh, cal.zones)
                    baselines_captured = True
                    print("[baselines] recaptured")
            else:
                print("[baselines] calibrate first (press 'c')")

        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = Path(__file__).parent / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(snap_path), frame)
            print(f"[snapshot] saved to {snap_path}")

    # -- Cleanup ------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("[done]")


if __name__ == "__main__":
    main()

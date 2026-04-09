#!/usr/bin/env python3
"""
Overhead Camera Card Recognition Test Harness

Captures video from a ceiling-mounted Logitech Brio 4K camera, monitors
designated landing zones on a poker table for card placement, and uses
Claude's vision API to identify cards.

Usage:
    python overhead_test.py [--camera 0] [--threshold 30.0]
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
from threading import Thread, Event

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

PLAYER_NAMES = ["Steve", "Bill", "David", "Joe", "Rodney"]
NUM_ZONES = len(PLAYER_NAMES)

DEFAULT_CAMERA_INDEX = 0          # 0 = Brio on Neo, 1 = built-in camera
DEFAULT_THRESHOLD = 30.0
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

# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

class Calibration:
    """Holds the felt-circle and landing-zone geometry."""

    def __init__(self):
        self.circle_center: tuple[int, int] | None = None
        self.circle_radius: int | None = None
        self.zones: list[dict] = []

    def save(self, path: Path = CALIBRATION_FILE):
        data = {
            "circle_center": list(self.circle_center) if self.circle_center else None,
            "circle_radius": self.circle_radius,
            "zones": self.zones,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Calibration saved to {path}")

    def load(self, path: Path = CALIBRATION_FILE) -> bool:
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        cc = data.get("circle_center")
        self.circle_center = tuple(cc) if cc else None
        self.circle_radius = data.get("circle_radius")
        self.zones = data.get("zones", [])
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
        self.baselines: dict[str, np.ndarray] = {}
        self.last_card: dict[str, str] = {}
        self.zone_state: dict[str, str] = {}
        self.pending: dict[str, bool] = {}
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key and CONFIG_FILE.exists():
                    with open(CONFIG_FILE) as f:
                        cfg = json.load(f)
                        api_key = cfg.get("anthropic_api_key")
                if api_key and api_key != "YOUR_KEY_HERE":
                    self._client = anthropic.Anthropic(api_key=api_key)
                else:
                    print("  WARNING: No valid API key found. Card recognition disabled.")
                    print("  Edit local/config.json to add your Anthropic API key.")
            except ImportError:
                print("  WARNING: anthropic package not installed. Card recognition disabled.")
        return self._client

    def capture_baselines(self, frame: np.ndarray, zones: list[dict]):
        for zone in zones:
            crop = self._crop_zone(frame, zone)
            self.baselines[zone["name"]] = crop.copy()
            self.zone_state[zone["name"]] = "empty"
            self.last_card[zone["name"]] = ""
            self.pending[zone["name"]] = False

    def check_zones(self, frame: np.ndarray, zones: list[dict]):
        for zone in zones:
            name = zone["name"]
            if name not in self.baselines:
                continue
            if self.pending.get(name, False):
                continue

            crop = self._crop_zone(frame, zone)
            diff = cv2.absdiff(crop, self.baselines[name])
            mean_diff = float(np.mean(diff))

            if mean_diff > self.threshold:
                if self.zone_state.get(name) != "processing":
                    self.zone_state[name] = "processing"
                    self.pending[name] = True
                    t = Thread(target=self._recognize, args=(name, crop.copy()), daemon=True)
                    t.start()
            else:
                if self.zone_state.get(name) == "recognized":
                    self.zone_state[name] = "empty"
                    self.last_card[name] = ""

    def _crop_zone(self, frame: np.ndarray, zone: dict) -> np.ndarray:
        x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
        return frame[y1:y2, x1:x2]

    def _recognize(self, name: str, crop: np.ndarray):
        t0 = time.time()
        try:
            # Show the crop being submitted in its own window
            cv2.imshow(f"Zone: {name}", crop)

            if self.client is None:
                print(f"  [{name}] API not available — skipping recognition")
                self.zone_state[name] = "empty"
                return

            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                print(f"  [{name}] failed to encode image")
                return
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            response = self.client.messages.create(
                model=MODEL,
                max_tokens=64,
                messages=[{
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
                }],
            )

            result = response.content[0].text.strip()
            elapsed = time.time() - t0

            self.last_card[name] = result
            self.zone_state[name] = "recognized"
            print(f"  {name}: {result}  ({elapsed:.1f}s)")

            self._save_training(name, crop, result)
            self._announce(name, result)

        except Exception as exc:
            print(f"  [{name}] API error: {exc}")
            self.zone_state[name] = "empty"
        finally:
            self.pending[name] = False

    def _save_training(self, name: str, crop: np.ndarray, result: str):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_result = result.replace(" ", "_").replace("/", "-")
        stem = f"{ts}_{name}_{safe_result}"
        cv2.imwrite(str(TRAINING_DIR / f"{stem}.jpg"), crop)
        (TRAINING_DIR / f"{stem}.txt").write_text(result)

    def _announce(self, name: str, result: str):
        if "no card" in result.lower():
            subprocess.Popen(
                ["say", f"{name}, try repositioning upcard"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["say", f"{name}, {result}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_overlay(frame: np.ndarray, cal: Calibration, monitor: ZoneMonitor,
                 monitoring: bool, cal_step: str = ""):
    if cal.circle_center and cal.circle_radius:
        cv2.circle(frame, cal.circle_center, cal.circle_radius, COLOR_WHITE, 2)

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
        cv2.putText(frame, name, (x1, y1 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        card = monitor.last_card.get(name, "")
        if card:
            cv2.putText(frame, card, (x1, y2 + 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)

    # Status in camera window
    if cal_step:
        cv2.putText(frame, cal_step, (20, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 3)
    elif monitoring:
        cv2.putText(frame, "MONITORING — place cards in zones", (20, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_GREEN, 2)


# ---------------------------------------------------------------------------
# Camera display thread
# ---------------------------------------------------------------------------

class CameraDisplay:
    """Runs camera capture and display in a background thread."""

    def __init__(self, cap, cal, monitor):
        self.cap = cap
        self.cal = cal
        self.monitor = monitor
        self.monitoring = False
        self.cal_step = ""          # calibration instruction for camera overlay
        self.latest_frame = None
        self._stop = Event()
        self._click_callback = None
        self._actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self):
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def set_click_callback(self, cb):
        self._click_callback = cb

    def _run(self):
        window_name = "Overhead Card Scanner"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and self._click_callback:
                scale_x = self._actual_w / 1280
                scale_y = self._actual_h / 720
                fx = int(x * scale_x)
                fy = int(y * scale_y)
                self._click_callback(fx, fy)

        cv2.setMouseCallback(window_name, on_mouse)

        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            self.latest_frame = frame.copy()

            # Run zone monitoring if active
            if self.monitoring and self.cal.is_complete:
                self.monitor.check_zones(frame, self.cal.zones)

            # Draw overlay
            display = frame.copy()
            draw_overlay(display, self.cal, self.monitor, self.monitoring, self.cal_step)
            cv2.imshow(window_name, display)
            cv2.waitKey(30)

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------

def print_menu():
    print("\n╔══════════════════════════════════════════╗")
    print("║       Overhead Card Scanner              ║")
    print("╠══════════════════════════════════════════╣")
    print("║  c = calibrate felt circle + zones       ║")
    print("║  m = start/stop monitoring               ║")
    print("║  r = reset baselines (clear table first) ║")
    print("║  s = save a snapshot                     ║")
    print("║  q = quit                                ║")
    print("╚══════════════════════════════════════════╝")


def do_calibrate(cam_display: CameraDisplay, cal: Calibration):
    """Interactive calibration driven from terminal with clicks in camera window."""
    print("\n  Calibration will walk you through 12 clicks in the camera window:")
    print("    1. Click the CENTER of the black felt circle")
    print(f"    2. Click a point on the EDGE of the felt circle (at Bill's position)")
    for i, name in enumerate(PLAYER_NAMES):
        print(f"    {3 + i*2}. Click TOP-LEFT corner of {name}'s landing zone")
        print(f"    {4 + i*2}. Click BOTTOM-RIGHT corner of {name}'s landing zone")
    print()
    input("  Press Enter to begin calibration...")

    # Reset calibration
    cal.circle_center = None
    cal.circle_radius = None
    cal.zones = []

    click_result = [None]
    click_ready = Event()

    def on_click(x, y):
        click_result[0] = (x, y)
        click_ready.set()

    cam_display.set_click_callback(on_click)

    def wait_for_click(prompt):
        print(f"\n  >>> {prompt}")
        cam_display.cal_step = prompt
        click_ready.clear()
        click_result[0] = None
        while not click_ready.is_set():
            time.sleep(0.05)
        cam_display.cal_step = ""
        return click_result[0]

    # Step 1: Circle center
    x, y = wait_for_click("Click the CENTER of the felt circle")
    cal.circle_center = (x, y)
    print(f"      Center set at ({x}, {y})")

    # Step 2: Circle edge
    x, y = wait_for_click("Click the EDGE of the felt circle (at Bill's position)")
    cx, cy = cal.circle_center
    cal.circle_radius = int(np.hypot(x - cx, y - cy))
    print(f"      Radius: {cal.circle_radius}px")

    # Steps 3-12: Player zones
    for i, name in enumerate(PLAYER_NAMES):
        x1, y1 = wait_for_click(f"Click TOP-LEFT of {name}'s zone")
        print(f"      Top-left at ({x1}, {y1})")

        x2, y2 = wait_for_click(f"Click BOTTOM-RIGHT of {name}'s zone")
        zone = {
            "name": name,
            "x1": min(x1, x2), "y1": min(y1, y2),
            "x2": max(x1, x2), "y2": max(y1, y2),
        }
        cal.zones.append(zone)
        w = zone["x2"] - zone["x1"]
        h = zone["y2"] - zone["y1"]
        print(f"      Zone '{name}' defined — {w}x{h}px")

    cam_display.set_click_callback(None)
    cal.save()
    print("\n  Calibration complete!")


def do_monitor_toggle(cam_display: CameraDisplay, monitor: ZoneMonitor, cal: Calibration):
    """Toggle monitoring on/off."""
    if not cal.is_complete:
        print("\n  Cannot monitor — calibrate first (press 'c')")
        return

    if cam_display.monitoring:
        cam_display.monitoring = False
        print("\n  Monitoring STOPPED")
    else:
        # Capture baselines first
        print("\n  Starting monitoring mode.")
        print("  Make sure all landing zones are EMPTY (no cards on the table).")
        input("  Press Enter when table is clear...")

        frame = cam_display.latest_frame
        if frame is not None:
            monitor.capture_baselines(frame, cal.zones)
            cam_display.monitoring = True
            print("  Baselines captured. Monitoring STARTED.")
            print("  Place cards in landing zones — recognition is automatic.")
            print("  Recognized cards will be announced by voice.")
        else:
            print("  ERROR: No camera frame available")


def do_reset_baselines(cam_display: CameraDisplay, monitor: ZoneMonitor, cal: Calibration):
    """Recapture baselines."""
    if not cal.is_complete:
        print("\n  Cannot reset — calibrate first (press 'c')")
        return

    print("\n  Resetting baselines.")
    print("  Make sure all landing zones are EMPTY.")
    input("  Press Enter when table is clear...")

    frame = cam_display.latest_frame
    if frame is not None:
        monitor.capture_baselines(frame, cal.zones)
        print("  Baselines recaptured.")
    else:
        print("  ERROR: No camera frame available")


def do_snapshot(cam_display: CameraDisplay):
    """Save a snapshot."""
    frame = cam_display.latest_frame
    if frame is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = Path(__file__).parent / f"snapshot_{ts}.jpg"
        cv2.imwrite(str(snap_path), frame)
        print(f"\n  Snapshot saved to {snap_path}")
    else:
        print("\n  ERROR: No camera frame available")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overhead camera card recognition test")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f"Camera index (default {DEFAULT_CAMERA_INDEX})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Change detection threshold (default {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # Open camera
    print(f"  Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"  ERROR: cannot open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera resolution: {actual_w}x{actual_h}")

    # Load calibration
    cal = Calibration()
    if cal.load():
        print(f"  Calibration loaded — {len(cal.zones)} zones defined")
    else:
        print("  No calibration found — press 'c' to calibrate")

    # Start camera display thread
    monitor = ZoneMonitor(threshold=args.threshold)
    cam_display = CameraDisplay(cap, cal, monitor)
    cam_display.start()

    # Give camera window a moment to appear
    time.sleep(0.5)

    # Main terminal loop
    print_menu()

    try:
        while True:
            cmd = input("\n  Enter command: ").strip().lower()

            if cmd == "q":
                print("\n  Shutting down...")
                break
            elif cmd == "c":
                do_calibrate(cam_display, cal)
            elif cmd == "m":
                do_monitor_toggle(cam_display, monitor, cal)
            elif cmd == "r":
                do_reset_baselines(cam_display, monitor, cal)
            elif cmd == "s":
                do_snapshot(cam_display)
            elif cmd == "":
                continue
            else:
                print(f"  Unknown command: '{cmd}'")
                print_menu()

    except (KeyboardInterrupt, EOFError):
        print("\n  Interrupted.")

    cam_display.stop()
    cap.release()
    print("  Done.")


if __name__ == "__main__":
    main()

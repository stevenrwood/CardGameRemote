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
from queue import Queue, Empty

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
                    print("  WARNING: No valid API key. Card recognition disabled.")
                    print("  Edit local/config.json to add your Anthropic API key.")
            except ImportError:
                print("  WARNING: anthropic package not installed.")
        return self._client

    def capture_baselines(self, frame: np.ndarray, zones: list[dict]):
        for zone in zones:
            crop = self._crop_zone(frame, zone)
            if crop is None:
                print(f"  WARNING: zone '{zone['name']}' is out of frame bounds")
                continue
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
            if crop is None:
                continue
            diff = cv2.absdiff(crop, self.baselines[name])
            if diff is None or diff.size == 0:
                continue
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

    def _crop_zone(self, frame: np.ndarray, zone: dict):
        h, w = frame.shape[:2]
        x1 = max(0, min(zone["x1"], w - 1))
        y1 = max(0, min(zone["y1"], h - 1))
        x2 = max(0, min(zone["x2"], w))
        y2 = max(0, min(zone["y2"], h))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _recognize(self, name: str, crop: np.ndarray):
        t0 = time.time()
        try:
            if self.client is None:
                print(f"  [{name}] API not available — skipping")
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
        safe = result.replace(" ", "_").replace("/", "-")
        cv2.imwrite(str(TRAINING_DIR / f"{ts}_{name}_{safe}.jpg"), crop)
        (TRAINING_DIR / f"{ts}_{name}_{safe}.txt").write_text(result)

    def _announce(self, name: str, result: str):
        if "no card" in result.lower():
            subprocess.Popen(["say", f"{name}, try repositioning upcard"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["say", f"{name}, {result}"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_overlay(frame, cal, monitor, monitoring, cal_step=""):
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

    if cal_step:
        cv2.putText(frame, cal_step, (20, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 3)
    elif monitoring:
        cv2.putText(frame, "MONITORING — place cards in zones", (20, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_GREEN, 2)


# ---------------------------------------------------------------------------
# Application state shared between main thread (OpenCV) and input thread
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, cap, cal, monitor):
        self.cap = cap
        self.cal = cal
        self.monitor = monitor
        self.monitoring = False
        self.baselines_captured = False
        self.cal_step = ""
        self.latest_frame = None
        self.quit_flag = False
        self.actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # For calibration clicks: main thread sets callback, input thread waits
        self.click_queue = Queue()

        # Commands from input thread to main thread
        self.command_queue = Queue()


# ---------------------------------------------------------------------------
# Input thread — reads terminal commands
# ---------------------------------------------------------------------------

def input_thread(state: AppState):
    """Runs in background thread, sends commands to main thread via queue."""

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

    if state.cal.is_complete:
        print(f"\n  Calibration loaded — {len(state.cal.zones)} zones defined")
    else:
        print("\n  No calibration found")

    print_menu()

    try:
        while not state.quit_flag:
            cmd = input("\n  Enter command: ").strip().lower()

            if cmd == "q":
                print("\n  Shutting down...")
                state.quit_flag = True
                break

            elif cmd == "c":
                do_calibrate_terminal(state)

            elif cmd == "m":
                do_monitor_toggle_terminal(state)

            elif cmd == "r":
                do_reset_baselines_terminal(state)

            elif cmd == "s":
                state.command_queue.put("snapshot")

            elif cmd == "":
                continue

            else:
                print(f"  Unknown command: '{cmd}'")
                print_menu()

    except (KeyboardInterrupt, EOFError):
        print("\n  Interrupted.")
        state.quit_flag = True


def do_calibrate_terminal(state: AppState):
    """Run calibration from terminal, clicks happen in camera window."""
    print("\n  Calibration will walk you through 12 clicks in the camera window:")
    print("    1. Click the CENTER of the black felt circle")
    print(f"    2. Click a point on the EDGE of the felt circle (at Bill's position)")
    for i, name in enumerate(PLAYER_NAMES):
        print(f"    {3 + i*2}. Click TOP-LEFT corner of {name}'s landing zone")
        print(f"    {4 + i*2}. Click BOTTOM-RIGHT corner of {name}'s landing zone")
    print()
    input("  Press Enter to begin calibration...")

    state.cal.circle_center = None
    state.cal.circle_radius = None
    state.cal.zones = []

    def wait_for_click(prompt):
        print(f"\n  >>> {prompt}")
        state.cal_step = prompt
        # Clear any stale clicks
        while not state.click_queue.empty():
            try:
                state.click_queue.get_nowait()
            except Empty:
                break
        # Wait for click from main thread
        while True:
            try:
                result = state.click_queue.get(timeout=0.1)
                state.cal_step = ""
                return result
            except Empty:
                if state.quit_flag:
                    return None

    # Step 1: Circle center
    pt = wait_for_click("Click the CENTER of the felt circle")
    if pt is None:
        return
    state.cal.circle_center = pt
    print(f"      Center set at ({pt[0]}, {pt[1]})")

    # Step 2: Circle edge
    pt = wait_for_click("Click the EDGE of the felt circle (at Bill's position)")
    if pt is None:
        return
    cx, cy = state.cal.circle_center
    state.cal.circle_radius = int(np.hypot(pt[0] - cx, pt[1] - cy))
    print(f"      Radius: {state.cal.circle_radius}px")

    # Steps 3-12: Player zones
    for i, name in enumerate(PLAYER_NAMES):
        pt1 = wait_for_click(f"Click TOP-LEFT of {name}'s zone")
        if pt1 is None:
            return
        print(f"      Top-left at ({pt1[0]}, {pt1[1]})")

        pt2 = wait_for_click(f"Click BOTTOM-RIGHT of {name}'s zone")
        if pt2 is None:
            return
        zone = {
            "name": name,
            "x1": min(pt1[0], pt2[0]), "y1": min(pt1[1], pt2[1]),
            "x2": max(pt1[0], pt2[0]), "y2": max(pt1[1], pt2[1]),
        }
        state.cal.zones.append(zone)
        w = zone["x2"] - zone["x1"]
        h = zone["y2"] - zone["y1"]
        print(f"      Zone '{name}' defined — {w}x{h}px")

    state.cal.save()
    print("\n  Calibration complete!")


def do_monitor_toggle_terminal(state: AppState):
    if not state.cal.is_complete:
        print("\n  Cannot monitor — calibrate first (press 'c')")
        return

    if state.monitoring:
        state.monitoring = False
        print("\n  Monitoring STOPPED")
    else:
        print("\n  Starting monitoring mode.")
        print("  Make sure all landing zones are EMPTY (no cards on the table).")
        input("  Press Enter when table is clear...")
        state.command_queue.put("capture_baselines")
        time.sleep(0.3)  # let main thread process it
        state.monitoring = True
        print("  Baselines captured. Monitoring STARTED.")
        print("  Place cards in landing zones — recognition is automatic.")
        print("  Recognized cards will be announced by voice.")


def do_reset_baselines_terminal(state: AppState):
    if not state.cal.is_complete:
        print("\n  Cannot reset — calibrate first (press 'c')")
        return
    print("\n  Resetting baselines.")
    print("  Make sure all landing zones are EMPTY.")
    input("  Press Enter when table is clear...")
    state.command_queue.put("capture_baselines")
    time.sleep(0.3)
    print("  Baselines recaptured.")


# ---------------------------------------------------------------------------
# Main — OpenCV runs here (main thread, required by macOS)
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

    # Load calibration
    cal = Calibration()
    cal.load()

    # Create shared state
    monitor = ZoneMonitor(threshold=args.threshold)
    state = AppState(cap, cal, monitor)

    print(f"  Camera resolution: {state.actual_w}x{state.actual_h}")

    # Set up OpenCV window (MUST be on main thread for macOS)
    window_name = "Overhead Card Scanner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get actual window size (may differ from requested 1280x720)
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
                if win_w <= 0 or win_h <= 0:
                    win_w, win_h = 1280, 720
            except Exception:
                win_w, win_h = 1280, 720
            scale_x = state.actual_w / win_w
            scale_y = state.actual_h / win_h
            fx = int(x * scale_x)
            fy = int(y * scale_y)
            state.click_queue.put((fx, fy))

    cv2.setMouseCallback(window_name, on_mouse)

    # Start input thread
    inp_thread = Thread(target=input_thread, args=(state,), daemon=True)
    inp_thread.start()

    # Main loop — camera capture + display (must be main thread on macOS)
    while not state.quit_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        state.latest_frame = frame.copy()

        # Process commands from input thread
        try:
            while True:
                cmd = state.command_queue.get_nowait()
                if cmd == "capture_baselines":
                    monitor.capture_baselines(frame, cal.zones)
                    state.baselines_captured = True
                elif cmd == "snapshot":
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = Path(__file__).parent / f"snapshot_{ts}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    print(f"\n  Snapshot saved to {snap_path}")
        except Empty:
            pass

        # Zone monitoring
        if state.monitoring and cal.is_complete and state.baselines_captured:
            monitor.check_zones(frame, cal.zones)

        # Draw overlay
        display = frame.copy()
        draw_overlay(display, cal, monitor, state.monitoring, state.cal_step)
        cv2.imshow(window_name, display)
        cv2.waitKey(30)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("  Done.")


if __name__ == "__main__":
    main()

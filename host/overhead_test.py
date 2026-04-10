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
from threading import Thread, Event, Lock
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
# Speech queue — serialized voice output, no overlapping
# ---------------------------------------------------------------------------

class SpeechQueue:
    """Speaks phrases one at a time, never overlapping. Drops stale messages."""

    def __init__(self):
        self._queue = Queue()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, phrase):
        """Queue a phrase. If queue is backing up, only keep the latest per player."""
        self._queue.put(phrase)

    def _run(self):
        while True:
            phrase = self._queue.get()
            # Drain any queued phrases — keep only the latest one per player
            latest = {phrase: phrase}
            try:
                while True:
                    p = self._queue.get_nowait()
                    latest[p] = p
            except Empty:
                pass
            # Speak each unique phrase synchronously (blocks until done)
            for p in latest.values():
                subprocess.run(["say", p],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

speech = SpeechQueue()

# ---------------------------------------------------------------------------
# Calibration data — zones are now circles
# ---------------------------------------------------------------------------

class Calibration:
    def __init__(self):
        self.circle_center = None   # (x, y) felt circle center
        self.circle_radius = None   # felt circle radius in pixels
        self.zones = []             # [{"name", "cx", "cy", "r"}, ...]

    def save(self, path=CALIBRATION_FILE):
        data = {
            "circle_center": list(self.circle_center) if self.circle_center else None,
            "circle_radius": self.circle_radius,
            "zones": self.zones,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Calibration saved to {path}")

    def load(self, path=CALIBRATION_FILE):
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        cc = data.get("circle_center")
        self.circle_center = tuple(cc) if cc else None
        self.circle_radius = data.get("circle_radius")
        self.zones = data.get("zones", [])
        # Validate zone format — must have cx/cy/r (circle), not x1/y1/x2/y2 (old rect)
        if self.zones and "cx" not in self.zones[0]:
            print("  Old rectangle calibration found — recalibration needed (press 'c')")
            self.zones = []
            return False
        return True

    @property
    def is_complete(self):
        return (
            self.circle_center is not None
            and self.circle_radius is not None
            and len(self.zones) == NUM_ZONES
        )


# ---------------------------------------------------------------------------
# Zone monitor — change detection + Claude API
# ---------------------------------------------------------------------------

class ZoneMonitor:
    def __init__(self, threshold):
        self.threshold = threshold
        self.baselines = {}       # name -> baseline crop
        self.last_card = {}       # name -> "Ace of Hearts"
        self.zone_state = {}      # name -> "empty"|"processing"|"recognized"
        self.pending = {}         # name -> True if API call in flight
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

    def capture_baselines(self, frame, zones):
        for zone in zones:
            crop = self._crop_zone(frame, zone)
            if crop is None or crop.size == 0:
                print(f"  WARNING: zone '{zone['name']}' is out of frame bounds")
                continue
            self.baselines[zone["name"]] = crop.copy()
            self.zone_state[zone["name"]] = "empty"
            self.last_card[zone["name"]] = ""
            self.pending[zone["name"]] = False

    def check_zones(self, frame, zones):
        for zone in zones:
            name = zone["name"]
            if name not in self.baselines:
                continue
            if self.pending.get(name, False):
                continue
            # Don't re-scan zones that already have a recognized card
            if self.zone_state.get(name) == "recognized":
                # But check if card was removed (zone returned to baseline)
                crop = self._crop_zone(frame, zone)
                if crop is None or crop.size == 0:
                    continue
                baseline = self.baselines[name]
                if crop.shape != baseline.shape:
                    continue
                diff = cv2.absdiff(crop, baseline)
                mean_diff = float(np.mean(diff))
                if mean_diff < self.threshold:
                    # Card removed — reset to empty
                    self.zone_state[name] = "empty"
                    self.last_card[name] = ""
                continue

            crop = self._crop_zone(frame, zone)
            if crop is None or crop.size == 0:
                continue

            baseline = self.baselines[name]
            if crop.shape != baseline.shape:
                continue

            diff = cv2.absdiff(crop, baseline)
            mean_diff = float(np.mean(diff))

            if mean_diff > self.threshold:
                self.zone_state[name] = "processing"
                self.pending[name] = True
                t = Thread(target=self._recognize, args=(name, crop.copy()), daemon=True)
                t.start()

    def _crop_zone(self, frame, zone):
        """Crop a circular zone as a bounding-box rectangle."""
        h, w = frame.shape[:2]
        cx, cy, r = zone["cx"], zone["cy"], zone["r"]
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _recognize(self, name, crop):
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
                max_tokens=20,
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
                                "What playing card is this? Reply with ONLY the rank "
                                "and suit in exactly this format: 'Rank of Suit' "
                                "(e.g. '4 of Clubs', 'King of Hearts'). "
                                "If you cannot identify the card, reply with exactly: 'No card'"
                            ),
                        },
                    ],
                }],
            )

            raw_result = response.content[0].text.strip()
            result = self._parse_card_result(raw_result)
            elapsed = time.time() - t0

            self.last_card[name] = result
            self.zone_state[name] = "recognized"
            print(f"  {name}: {result}  ({elapsed:.1f}s)")

            self._save_training(name, crop, result)

            # Voice announcement (serialized, never overlaps)
            if "no card" in result.lower():
                speech.say(f"{name}, try repositioning upcard")
            else:
                speech.say(f"{name}, {result}")

        except Exception as exc:
            print(f"  [{name}] API error: {exc}")
            self.zone_state[name] = "empty"
        finally:
            self.pending[name] = False

    def _parse_card_result(self, raw):
        """Extract 'Rank of Suit' from API response, handling verbose replies."""
        import re
        # Look for pattern like "4 of Clubs", "King of Hearts", "Ace of Spades"
        match = re.search(
            r'(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)',
            raw, re.IGNORECASE
        )
        if match:
            rank = match.group(1).capitalize()
            suit = match.group(2).capitalize()
            return f"{rank} of {suit}"
        return "No card"

    def _save_training(self, name, crop, result):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Truncate result for filename — keep only first 30 chars
        safe = result.replace(" ", "_").replace("/", "-")[:30]
        cv2.imwrite(str(TRAINING_DIR / f"{ts}_{name}_{safe}.jpg"), crop)
        (TRAINING_DIR / f"{ts}_{name}_{safe}.txt").write_text(result)


# ---------------------------------------------------------------------------
# Drawing helpers — circular zones
# ---------------------------------------------------------------------------

def draw_overlay(frame, cal, monitor, monitoring, cal_step="", preview_circle=None):
    # Felt circle
    if cal.circle_center and cal.circle_radius:
        cv2.circle(frame, cal.circle_center, cal.circle_radius, COLOR_WHITE, 2)

    # Landing zones (circles)
    for zone in cal.zones:
        name = zone["name"]
        cx, cy, r = zone["cx"], zone["cy"], zone["r"]

        state = monitor.zone_state.get(name, "empty")
        if state == "recognized":
            color = COLOR_GREEN
        elif state == "processing":
            color = COLOR_YELLOW
        else:
            color = COLOR_WHITE

        cv2.circle(frame, (cx, cy), r, color, 2)
        cv2.putText(frame, name, (cx - 30, cy - r - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        card = monitor.last_card.get(name, "")
        if card:
            cv2.putText(frame, card, (cx - 60, cy + r + 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)

    # Preview circle during calibration (follows mouse)
    if preview_circle:
        pcx, pcy, pr = preview_circle
        cv2.circle(frame, (pcx, pcy), pr, COLOR_YELLOW, 2)

    # Status text
    if cal_step:
        cv2.putText(frame, cal_step, (20, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 3)
    elif monitoring:
        cv2.putText(frame, "MONITORING — place cards in zones", (20, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_GREEN, 2)


# ---------------------------------------------------------------------------
# Application state
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

        self.click_queue = Queue()
        self.command_queue = Queue()

        # For zone circle preview during calibration
        self.preview_circle = None     # (cx, cy, r) or None
        self.zone_center_pending = None  # (cx, cy) waiting for radius click
        self.mouse_pos = None          # current mouse position in frame coords


# ---------------------------------------------------------------------------
# Input thread
# ---------------------------------------------------------------------------

def input_thread(state):
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


def do_calibrate_terminal(state):
    print("\n  Calibration steps:")
    print("    1. Click the CENTER of the black felt circle")
    print(f"    2. Click a point on the EDGE of the felt circle (at Bill's position)")
    print()
    print("  For each player zone:")
    for i, name in enumerate(PLAYER_NAMES):
        print(f"    {3 + i}. {name}: Click CENTER of zone, then move mouse to")
        print(f"        set zone size and click to lock it in")
    print()
    input("  Press Enter to begin calibration...")

    state.cal.circle_center = None
    state.cal.circle_radius = None
    state.cal.zones = []
    state.preview_circle = None
    state.zone_center_pending = None

    def wait_for_click(prompt):
        print(f"\n  >>> {prompt}")
        state.cal_step = prompt
        while not state.click_queue.empty():
            try:
                state.click_queue.get_nowait()
            except Empty:
                break
        while True:
            try:
                result = state.click_queue.get(timeout=0.1)
                state.cal_step = ""
                return result
            except Empty:
                if state.quit_flag:
                    return None

    # Step 1: Felt circle center
    pt = wait_for_click("Click the CENTER of the felt circle")
    if pt is None:
        return
    state.cal.circle_center = pt
    print(f"      Center set at ({pt[0]}, {pt[1]})")

    # Step 2: Felt circle edge
    pt = wait_for_click("Click the EDGE of the felt circle (at Bill's position)")
    if pt is None:
        return
    cx, cy = state.cal.circle_center
    state.cal.circle_radius = int(np.hypot(pt[0] - cx, pt[1] - cy))
    print(f"      Radius: {state.cal.circle_radius}px")

    # Steps 3-7: Player zones (circle: click center, move mouse to size, click to lock)
    for i, name in enumerate(PLAYER_NAMES):
        # Click center
        pt = wait_for_click(f"Click CENTER of {name}'s zone")
        if pt is None:
            return
        zone_cx, zone_cy = pt
        print(f"      Center at ({zone_cx}, {zone_cy})")

        # Now show preview circle tracking mouse for radius
        state.zone_center_pending = (zone_cx, zone_cy)
        state.cal_step = f"Move mouse to set {name}'s zone size, then click"

        pt2 = wait_for_click(f"Move mouse to set {name}'s zone size, then click")
        if pt2 is None:
            return

        zone_r = int(np.hypot(pt2[0] - zone_cx, pt2[1] - zone_cy))
        state.zone_center_pending = None
        state.preview_circle = None

        state.cal.zones.append({
            "name": name,
            "cx": zone_cx,
            "cy": zone_cy,
            "r": zone_r,
        })
        print(f"      Zone '{name}' defined — radius {zone_r}px")

    state.cal.save()
    print("\n  Calibration complete!")


def do_monitor_toggle_terminal(state):
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
        time.sleep(0.3)
        state.monitoring = True
        print("  Baselines captured. Monitoring STARTED.")
        print("  Place cards in landing zones — recognition is automatic.")
        print("  Recognized cards will be announced by voice.")


def do_reset_baselines_terminal(state):
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
# Main — OpenCV on main thread (required by macOS)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overhead camera card recognition test")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f"Camera index (default {DEFAULT_CAMERA_INDEX})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Change detection threshold (default {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"  ERROR: cannot open camera index {args.camera}")

    # Try to set 4K resolution — Brio supports it but macOS may limit it
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    cal = Calibration()
    cal.load()

    monitor = ZoneMonitor(threshold=args.threshold)
    state = AppState(cap, cal, monitor)

    print(f"  Camera resolution: {state.actual_w}x{state.actual_h}")

    window_name = "Overhead Card Scanner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    def get_frame_coords(x, y):
        """Convert window click coords to frame coords."""
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w <= 0 or win_h <= 0:
                win_w, win_h = 1280, 720
        except Exception:
            win_w, win_h = 1280, 720
        scale_x = state.actual_w / win_w
        scale_y = state.actual_h / win_h
        return int(x * scale_x), int(y * scale_y)

    def on_mouse(event, x, y, flags, param):
        fx, fy = get_frame_coords(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            state.click_queue.put((fx, fy))

        elif event == cv2.EVENT_MOUSEMOVE:
            state.mouse_pos = (fx, fy)
            # Update preview circle if sizing a zone
            if state.zone_center_pending:
                zcx, zcy = state.zone_center_pending
                r = int(np.hypot(fx - zcx, fy - zcy))
                state.preview_circle = (zcx, zcy, r)

    cv2.setMouseCallback(window_name, on_mouse)

    inp_thread = Thread(target=input_thread, args=(state,), daemon=True)
    inp_thread.start()

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
        draw_overlay(display, cal, monitor, state.monitoring, state.cal_step,
                     state.preview_circle)
        cv2.imshow(window_name, display)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()
    print("  Done.")


if __name__ == "__main__":
    main()

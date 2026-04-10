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
import http.server
import io
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
# Log buffer — keeps recent log lines for debug server
# ---------------------------------------------------------------------------

class LogBuffer:
    """Thread-safe ring buffer of log lines."""

    def __init__(self, max_lines=200):
        self._lines = []
        self._max = max_lines
        self._lock = Lock()

    def log(self, msg):
        """Add a line and also print to terminal."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(f"  {msg}")
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > self._max:
                self._lines = self._lines[-self._max:]

    def get_lines(self):
        with self._lock:
            return list(self._lines)

log_buffer = LogBuffer()

# ---------------------------------------------------------------------------
# Debug web server — access logs, snapshots, zone crops remotely
# ---------------------------------------------------------------------------

_debug_state = None  # set in main() before server starts

class DebugHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP handler for remote debugging."""

    def log_message(self, format, *args):
        pass  # suppress default request logging

    def do_GET(self):
        state = _debug_state
        if state is None:
            self._respond(500, "text/plain", "Not initialized")
            return

        if self.path == "/" or self.path == "/debug":
            self._serve_dashboard(state)
        elif self.path == "/log":
            lines = log_buffer.get_lines()
            self._respond(200, "text/plain", "\n".join(lines))
        elif self.path == "/snapshot":
            self._serve_frame(state)
        elif self.path.startswith("/zone/"):
            zone_name = self.path[6:]
            self._serve_zone_crop(state, zone_name)
        elif self.path == "/calibration":
            if CALIBRATION_FILE.exists():
                self._respond(200, "application/json", CALIBRATION_FILE.read_text())
            else:
                self._respond(404, "text/plain", "No calibration")
        elif self.path == "/training":
            self._serve_training_list()
        elif self.path.startswith("/training/"):
            filename = self.path[10:]
            self._serve_training_file(filename)
        else:
            self._respond(404, "text/plain", "Not found")

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.wfile.write(body)

    def _serve_dashboard(self, state):
        zones_html = ""
        for zone in state.cal.zones:
            name = zone["name"]
            card = state.monitor.last_card.get(name, "")
            zstate = state.monitor.zone_state.get(name, "empty")
            color = {"recognized": "#4caf50", "processing": "#ff9800"}.get(zstate, "#888")
            zones_html += (
                f'<div style="display:inline-block;margin:8px;padding:12px;'
                f'border:2px solid {color};border-radius:8px;min-width:120px;text-align:center">'
                f'<b>{name}</b><br>{zstate}<br>'
                f'<span style="color:{color};font-size:1.2em">{card}</span><br>'
                f'<a href="/zone/{name}"><img src="/zone/{name}" width="150"></a>'
                f'</div>'
            )

        html = f"""<!DOCTYPE html>
<html><head><title>Card Scanner Debug</title>
<meta http-equiv="refresh" content="3">
<style>body{{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}}
a{{color:#4fc3f7}}img{{border:1px solid #444;margin:4px}}
pre{{background:#0d1117;padding:12px;border-radius:6px;max-height:400px;overflow:auto;font-size:0.85em}}</style>
</head><body>
<h1>Overhead Card Scanner — Debug</h1>
<p>Resolution: {state.actual_w}x{state.actual_h} |
Monitoring: {'ON' if state.monitoring else 'OFF'} |
Calibrated: {'Yes' if state.cal.is_complete else 'No'}</p>
<h2>Live Snapshot</h2>
<a href="/snapshot"><img src="/snapshot" width="640"></a>
<h2>Zones</h2>
{zones_html}
<h2>Log (last 50 lines)</h2>
<pre>{"<br>".join(log_buffer.get_lines()[-50:])}</pre>
<h2>Links</h2>
<ul>
<li><a href="/log">Full log (text)</a></li>
<li><a href="/calibration">Calibration JSON</a></li>
<li><a href="/training">Training data files</a></li>
</ul>
</body></html>"""
        self._respond(200, "text/html", html)

    def _serve_frame(self, state):
        frame = state.latest_frame
        if frame is None:
            self._respond(503, "text/plain", "No frame available")
            return
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self._respond(200, "image/jpeg", buf.tobytes())
        else:
            self._respond(500, "text/plain", "Encode failed")

    def _serve_zone_crop(self, state, zone_name):
        frame = state.latest_frame
        if frame is None:
            self._respond(503, "text/plain", "No frame")
            return
        zone = next((z for z in state.cal.zones if z["name"] == zone_name), None)
        if zone is None:
            self._respond(404, "text/plain", f"Zone '{zone_name}' not found")
            return
        crop = state.monitor._crop_zone(frame, zone)
        if crop is None:
            self._respond(500, "text/plain", "Crop failed")
            return
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            self._respond(200, "image/jpeg", buf.tobytes())
        else:
            self._respond(500, "text/plain", "Encode failed")

    def _serve_training_list(self):
        if not TRAINING_DIR.exists():
            self._respond(200, "text/html", "<p>No training data yet</p>")
            return
        files = sorted(TRAINING_DIR.iterdir(), reverse=True)
        html = "<html><body style='font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px'>"
        html += "<h1>Training Data</h1>"
        for f in files[:100]:
            if f.suffix == ".jpg":
                txt = f.with_suffix(".txt")
                label = txt.read_text() if txt.exists() else ""
                html += (f'<div style="display:inline-block;margin:8px;text-align:center">'
                         f'<a href="/training/{f.name}"><img src="/training/{f.name}" width="200"></a>'
                         f'<br><small>{f.name}</small><br>{label}</div>')
        html += "</body></html>"
        self._respond(200, "text/html", html)

    def _serve_training_file(self, filename):
        path = TRAINING_DIR / filename
        if not path.exists():
            self._respond(404, "text/plain", "Not found")
            return
        if path.suffix == ".jpg":
            self._respond(200, "image/jpeg", path.read_bytes())
        elif path.suffix == ".txt":
            self._respond(200, "text/plain", path.read_text())
        else:
            self._respond(404, "text/plain", "Unknown file type")


DEBUG_PORT = 8888

def start_debug_server(state):
    global _debug_state
    _debug_state = state
    server = http.server.HTTPServer(("0.0.0.0", DEBUG_PORT), DebugHandler)
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    log_buffer.log(f"Debug server running at http://localhost:{DEBUG_PORT}")


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
                    log_buffer.log("WARNING: No valid API key. Card recognition disabled.")
                    log_buffer.log("Edit local/config.json to add your Anthropic API key.")
            except ImportError:
                log_buffer.log("WARNING: anthropic package not installed.")
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
                log_buffer.log(f"[{name}] API not available — skipping")
                self.zone_state[name] = "empty"
                return

            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                log_buffer.log(f"[{name}] failed to encode image")
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
            log_buffer.log(f"{name}: {result}  ({elapsed:.1f}s)")

            self._save_training(name, crop, result)

            # Voice announcement — only on successful recognition
            if "no card" not in result.lower():
                speech.say(f"{name}, {result}")

        except Exception as exc:
            log_buffer.log(f"[{name}] API error: {exc}")
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

def draw_overlay(frame, cal, monitor, monitoring, cal_step="", preview_circle=None,
                 flash_zone=None, flash_on=False):
    # Felt circle
    if cal.circle_center and cal.circle_radius:
        cv2.circle(frame, cal.circle_center, cal.circle_radius, COLOR_WHITE, 2)

    # Landing zones (circles)
    for zone in cal.zones:
        name = zone["name"]
        cx, cy, r = zone["cx"], zone["cy"], zone["r"]

        # Flashing zone — alternate between red and off
        if flash_zone == name:
            if flash_on:
                cv2.circle(frame, (cx, cy), r, COLOR_RED, 4)
                cv2.putText(frame, name, (cx - 30, cy - r - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
            continue

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

        # For flashing zones during test mode
        self.flash_zone = None         # zone name to flash, or None
        self.flash_start = 0.0         # time when flash started

        # For test recognition mode
        self.test_result = None        # result from test recognition
        self.test_done = Event()


# ---------------------------------------------------------------------------
# Input thread
# ---------------------------------------------------------------------------

def input_thread(state):
    def print_menu():
        print("\n╔══════════════════════════════════════════╗")
        print("║       Overhead Card Scanner              ║")
        print("╠══════════════════════════════════════════╣")
        print("║  c = calibrate felt circle + zones       ║")
        print("║  t = test recognition (zone by zone)     ║")
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
            elif cmd == "t":
                do_test_recognition(state)
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


def do_test_recognition(state):
    """Test recognition zone by zone with interactive feedback."""
    if not state.cal.is_complete:
        print("\n  Cannot test — calibrate first (press 'c')")
        return

    print("\n  Test Recognition Mode")
    print("  This will test card recognition in each player's zone one at a time.")
    print("  Make sure the table is CLEAR of all cards.")
    input("  Press Enter when table is clear...")

    # Capture fresh baselines
    state.command_queue.put("capture_baselines")
    time.sleep(0.5)

    for zone in state.cal.zones:
        name = zone["name"]
        print(f"\n  --- Testing {name}'s zone ---")
        print(f"  Place a face-up card in {name}'s zone.")
        print(f"  (circle will flash while waiting for card)")

        recognized = False
        attempts = 0
        max_attempts = 3

        while not recognized and attempts < max_attempts:
            # Flash the zone circle while waiting
            state.flash_zone = name
            state.flash_start = time.time()

            # Poll for card detection — keep trying for 30 seconds
            card_found = False
            poll_start = time.time()
            while time.time() - poll_start < 30.0:
                state.test_result = None
                state.test_done.clear()
                state.command_queue.put(("test_recognize", name))

                got_result = state.test_done.wait(timeout=2.0)

                if got_result and state.test_result is not None:
                    if state.test_result != "No card":
                        card_found = True
                        break
                    # API said "No card" — keep trying, card might need repositioning
                    log_buffer.log(f"[{name}] API returned 'No card' — retrying...")

                time.sleep(1.0)  # wait between polls

            state.flash_zone = None

            if not card_found:
                attempts += 1
                if attempts < max_attempts:
                    print(f"  No card recognized. Try repositioning... (attempt {attempts + 1}/{max_attempts})")
                    # Flash circle to indicate failure — no voice
                    state.flash_zone = name
                    state.flash_start = time.time()
                    time.sleep(3)
                    state.flash_zone = None
                continue

            result = state.test_result
            print(f"  Recognized: {result}")
            speech.say(f"{name}, {result}")

            # Wait for click (confirm) or 5 seconds (retry)
            print(f"  Click in camera window to confirm, or wait 5s to retry...")

            # Clear stale clicks
            while not state.click_queue.empty():
                try:
                    state.click_queue.get_nowait()
                except Empty:
                    break

            try:
                state.click_queue.get(timeout=5.0)
                print(f"  Confirmed!")
                recognized = True
            except Empty:
                # No click — flash and retry
                attempts += 1
                if attempts < max_attempts:
                    print(f"  No confirmation. Retrying... (attempt {attempts + 1}/{max_attempts})")
                    state.flash_zone = name
                    state.flash_start = time.time()
                    time.sleep(2)
                    state.flash_zone = None

        if not recognized:
            print(f"  Skipping {name}'s zone after {max_attempts} attempts.")

        # Wait for card to be removed before next player
        if recognized:
            print(f"  Remove the card from {name}'s zone.")
            input("  Press Enter when card is removed...")

    print("\n  Test complete!")
    state.command_queue.put("capture_baselines")
    time.sleep(0.3)


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

    # Start debug web server
    start_debug_server(state)

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
                elif isinstance(cmd, tuple) and cmd[0] == "test_recognize":
                    # Test recognition for a specific zone
                    zone_name = cmd[1]
                    zone = next((z for z in cal.zones if z["name"] == zone_name), None)
                    if zone and state.baselines_captured:
                        crop = monitor._crop_zone(frame, zone)
                        if crop is not None and crop.size > 0:
                            baseline = monitor.baselines.get(zone_name)
                            if baseline is not None and crop.shape == baseline.shape:
                                diff = cv2.absdiff(crop, baseline)
                                mean_diff = float(np.mean(diff))
                                if mean_diff > monitor.threshold:
                                    # Card detected — run recognition
                                    def _do_test_recognize(name, c):
                                        monitor._recognize(name, c)
                                        state.test_result = monitor.last_card.get(name, "No card")
                                        state.test_done.set()
                                    t = Thread(target=_do_test_recognize,
                                               args=(zone_name, crop.copy()), daemon=True)
                                    t.start()
                                else:
                                    # No card detected yet — signal done with None
                                    state.test_result = None
                                    state.test_done.set()
                            else:
                                state.test_result = None
                                state.test_done.set()
                        else:
                            state.test_result = None
                            state.test_done.set()
        except Empty:
            pass

        # Zone monitoring
        if state.monitoring and cal.is_complete and state.baselines_captured:
            monitor.check_zones(frame, cal.zones)

        # Compute flash state (blink at 2Hz)
        flash_on = False
        if state.flash_zone:
            elapsed = time.time() - state.flash_start
            flash_on = int(elapsed * 4) % 2 == 0  # 4 toggles/sec = 2Hz blink

        # Draw overlay
        display = frame.copy()
        draw_overlay(display, cal, monitor, state.monitoring, state.cal_step,
                     state.preview_circle, state.flash_zone, flash_on)
        cv2.imshow(window_name, display)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()
    print("  Done.")


if __name__ == "__main__":
    main()

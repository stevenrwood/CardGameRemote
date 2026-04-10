#!/usr/bin/env python3
"""
Overhead Camera Card Recognition Test Harness

Captures still images from a Logitech Brio 4K camera using ffmpeg,
monitors designated landing zones on a poker table for card placement,
and uses Claude's vision API to identify cards.

No live video — captures a JPEG every ~2 seconds. Terminal-only UI.
Debug dashboard at http://localhost:8888.

Usage:
    python overhead_test.py [--camera 0] [--threshold 30.0] [--resolution 1920x1080]
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Event, Lock
from queue import Queue, Empty

import cv2
import http.server
import numpy as np

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

PLAYER_NAMES = ["Steve", "Bill", "David", "Joe", "Rodney"]
NUM_ZONES = len(PLAYER_NAMES)

DEFAULT_CAMERA_INDEX = 0
DEFAULT_THRESHOLD = 30.0
DEFAULT_RESOLUTION = "1920x1080"

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
TRAINING_DIR = Path(__file__).parent / "training_data"
CONFIG_FILE = Path(__file__).parent.parent / "local" / "config.json"
CAPTURE_FILE = Path("/tmp/card_scanner_frame.jpg")

MODEL = "claude-sonnet-4-20250514"

COLOR_WHITE  = (255, 255, 255)
COLOR_GREEN  = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED    = (0, 0, 255)

# ---------------------------------------------------------------------------
# Speech queue — serialized voice output, no overlapping
# ---------------------------------------------------------------------------

class SpeechQueue:
    def __init__(self):
        self._queue = Queue()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, phrase):
        self._queue.put(phrase)

    def _run(self):
        while True:
            phrase = self._queue.get()
            latest = {phrase: phrase}
            try:
                while True:
                    p = self._queue.get_nowait()
                    latest[p] = p
            except Empty:
                pass
            for p in latest.values():
                subprocess.run(["say", p],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

speech = SpeechQueue()

# ---------------------------------------------------------------------------
# Log buffer
# ---------------------------------------------------------------------------

class LogBuffer:
    def __init__(self, max_lines=200):
        self._lines = []
        self._max = max_lines
        self._lock = Lock()

    def log(self, msg):
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
# Image capture via ffmpeg
# ---------------------------------------------------------------------------

class FrameCapture:
    """Captures still JPEG frames from a camera using ffmpeg."""

    def __init__(self, camera_index, resolution):
        self.camera_index = camera_index
        self.resolution = resolution
        w, h = resolution.split("x")
        self.width = int(w)
        self.height = int(h)
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        except FileNotFoundError:
            log_buffer.log("ERROR: ffmpeg not found. Install with: brew install ffmpeg")
            sys.exit(1)

    def capture(self, output_path=None):
        """Capture a single frame. Returns the image as numpy array, or None."""
        path = str(output_path or CAPTURE_FILE)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "avfoundation",
            "-video_size", self.resolution,
            "-framerate", "5",
            "-i", f"{self.camera_index}:none",
            "-frames:v", "1",
            "-q:v", "2",
            path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode != 0:
                err = result.stderr.decode().strip()
                if err:
                    log_buffer.log(f"ffmpeg error: {err}")
                return None
            frame = cv2.imread(path)
            return frame
        except subprocess.TimeoutExpired:
            log_buffer.log("ffmpeg capture timed out")
            return None
        except Exception as e:
            log_buffer.log(f"Capture error: {e}")
            return None


# ---------------------------------------------------------------------------
# Debug web server
# ---------------------------------------------------------------------------

_debug_state = None

class DebugHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        state = _debug_state
        if state is None:
            self._respond(500, "text/plain", "Not initialized")
            return

        if self.path == "/" or self.path == "/debug":
            self._serve_dashboard(state)
        elif self.path == "/live":
            self._serve_live_view(state)
        elif self.path == "/snapshot/cropped":
            self._serve_cropped_frame(state)
        elif self.path == "/log":
            self._respond(200, "text/plain", "\n".join(log_buffer.get_lines()))
        elif self.path == "/snapshot":
            self._serve_frame(state)
        elif self.path.startswith("/zone/"):
            self._serve_zone_crop(state, self.path[6:])
        elif self.path == "/calibration":
            if CALIBRATION_FILE.exists():
                self._respond(200, "application/json", CALIBRATION_FILE.read_text())
            else:
                self._respond(404, "text/plain", "No calibration")
        elif self.path == "/training":
            self._serve_training_list()
        elif self.path.startswith("/training/"):
            self._serve_training_file(self.path[10:])
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

        res = f"{state.capture.width}x{state.capture.height}" if state.capture else "?"
        html = f"""<!DOCTYPE html>
<html><head><title>Card Scanner Debug</title>
<meta http-equiv="refresh" content="3">
<style>body{{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}}
a{{color:#4fc3f7}}img{{border:1px solid #444;margin:4px}}
pre{{background:#0d1117;padding:12px;border-radius:6px;max-height:400px;overflow:auto;font-size:0.85em}}</style>
</head><body>
<h1>Overhead Card Scanner — Debug</h1>
<p>Resolution: {res} |
Monitoring: {'ON' if state.monitoring else 'OFF'} |
Calibrated: {'Yes' if state.cal.is_complete else 'No'}</p>
<h2>Latest Capture</h2>
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

    def _serve_live_view(self, state):
        html = """<!DOCTYPE html>
<html><head><title>Table View</title>
<style>
body { margin:0; background:#000; display:flex; justify-content:center; align-items:center; height:100vh; }
img { max-width:100%; max-height:100vh; }
</style>
<script>
function refreshImage() {
    var newImg = new Image();
    newImg.onload = function() {
        document.getElementById('frame').src = newImg.src;
    };
    // Only replace if load succeeds — keeps last good image on failure
    newImg.src = '/snapshot/cropped?' + Date.now();
}
setInterval(refreshImage, 2000);
</script>
</head><body>
<img id="frame" src="/snapshot/cropped">
</body></html>"""
        self._respond(200, "text/html", html)

    def _serve_cropped_frame(self, state):
        # Serve the cached cropped JPEG if available
        if state.latest_cropped_jpg is not None:
            self._respond(200, "image/jpeg", state.latest_cropped_jpg)
        else:
            self._respond(503, "text/plain", "No frame yet")

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
# Calibration data — zones are circles
# ---------------------------------------------------------------------------

class Calibration:
    def __init__(self):
        self.circle_center = None
        self.circle_radius = None
        self.zones = []

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
# Image helpers
# ---------------------------------------------------------------------------

def crop_to_felt_circle(frame, cal):
    """Mask frame to the felt circle, black outside."""
    if cal.circle_center is None or cal.circle_radius is None:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, cal.circle_center, cal.circle_radius, 255, -1)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

def draw_overlay(frame, cal, monitor, flash_zone=None, flash_on=False):
    """Draw zone circles and labels on frame."""
    if cal.circle_center and cal.circle_radius:
        cv2.circle(frame, cal.circle_center, cal.circle_radius, COLOR_WHITE, 2)

    for zone in cal.zones:
        name = zone["name"]
        cx, cy, r = zone["cx"], zone["cy"], zone["r"]

        if flash_zone == name:
            if flash_on:
                cv2.circle(frame, (cx, cy), r, COLOR_RED, 4)
                cv2.putText(frame, name, (cx - 30, cy - r - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
            continue

        zstate = monitor.zone_state.get(name, "empty")
        if zstate == "recognized":
            color = COLOR_GREEN
        elif zstate == "processing":
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


# ---------------------------------------------------------------------------
# Zone monitor — change detection + Claude API
# ---------------------------------------------------------------------------

class ZoneMonitor:
    def __init__(self, threshold):
        self.threshold = threshold
        self.baselines = {}
        self.last_card = {}
        self.zone_state = {}
        self.pending = {}
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
                log_buffer.log(f"WARNING: zone '{zone['name']}' out of frame bounds")
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
            if self.zone_state.get(name) == "recognized":
                crop = self._crop_zone(frame, zone)
                if crop is None or crop.size == 0:
                    continue
                baseline = self.baselines[name]
                if crop.shape != baseline.shape:
                    continue
                diff = cv2.absdiff(crop, baseline)
                mean_diff = float(np.mean(diff))
                if mean_diff < self.threshold:
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

    def check_single_zone(self, frame, zone):
        """Check a single zone for card presence. Returns crop if card detected, None otherwise."""
        name = zone["name"]
        if name not in self.baselines:
            return None
        crop = self._crop_zone(frame, zone)
        if crop is None or crop.size == 0:
            return None
        baseline = self.baselines[name]
        if crop.shape != baseline.shape:
            return None
        diff = cv2.absdiff(crop, baseline)
        mean_diff = float(np.mean(diff))
        if mean_diff > self.threshold:
            return crop.copy()
        return None

    def recognize_sync(self, name, crop):
        """Run recognition synchronously. Returns result string."""
        self._recognize(name, crop)
        return self.last_card.get(name, "No card")

    def _crop_zone(self, frame, zone):
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

            if "no card" not in result.lower():
                speech.say(f"{name}, {result}")

        except Exception as exc:
            log_buffer.log(f"[{name}] API error: {exc}")
            self.zone_state[name] = "empty"
        finally:
            self.pending[name] = False

    def _parse_card_result(self, raw):
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
        safe = result.replace(" ", "_").replace("/", "-")[:30]
        cv2.imwrite(str(TRAINING_DIR / f"{ts}_{name}_{safe}.jpg"), crop)
        (TRAINING_DIR / f"{ts}_{name}_{safe}.txt").write_text(result)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, capture, cal, monitor):
        self.capture = capture
        self.cal = cal
        self.monitor = monitor
        self.monitoring = False
        self.baselines_captured = False
        self.latest_frame = None
        self.latest_cropped_jpg = None   # pre-rendered JPEG for live view
        self.quit_flag = False
        self.flash_zone = None
        self.flash_start = 0.0


# ---------------------------------------------------------------------------
# Terminal UI commands
# ---------------------------------------------------------------------------

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


def capture_frame(state):
    """Capture a frame and update state, including cached cropped JPEG for live view."""
    frame = state.capture.capture()
    if frame is not None:
        state.latest_frame = frame.copy()
        # Pre-render cropped JPEG for the live view endpoint
        cropped = crop_to_felt_circle(frame, state.cal)
        display = cropped.copy()
        draw_overlay(display, state.cal, state.monitor)
        ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            state.latest_cropped_jpg = buf.tobytes()
    return frame


def do_calibrate(state):
    """Interactive calibration using OpenCV window for clicks."""
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

    # Capture a frame and show it in OpenCV window for clicking
    frame = capture_frame(state)
    if frame is None:
        print("  ERROR: Could not capture frame for calibration")
        return

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    clicks = []
    preview_center = [None]  # mutable for closure
    mouse_pos = [None]

    actual_h, actual_w = frame.shape[:2]

    def get_frame_coords(x, y):
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w <= 0 or win_h <= 0:
                win_w, win_h = 1280, 720
        except Exception:
            win_w, win_h = 1280, 720
        return int(x * actual_w / win_w), int(y * actual_h / win_h)

    def on_mouse(event, x, y, flags, param):
        fx, fy = get_frame_coords(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((fx, fy))
        elif event == cv2.EVENT_MOUSEMOVE:
            mouse_pos[0] = (fx, fy)

    cv2.setMouseCallback(window_name, on_mouse)

    def wait_for_click(prompt):
        print(f"\n  >>> {prompt}")
        clicks.clear()
        while True:
            display = frame.copy()
            # Draw what we have so far
            if state.cal.circle_center and state.cal.circle_radius:
                cv2.circle(display, state.cal.circle_center, state.cal.circle_radius, COLOR_WHITE, 2)
            for z in state.cal.zones:
                cv2.circle(display, (z["cx"], z["cy"]), z["r"], COLOR_GREEN, 2)
                cv2.putText(display, z["name"], (z["cx"] - 30, z["cy"] - z["r"] - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
            # Draw preview circle if sizing a zone
            if preview_center[0] and mouse_pos[0]:
                pcx, pcy = preview_center[0]
                mx, my = mouse_pos[0]
                pr = int(np.hypot(mx - pcx, my - pcy))
                cv2.circle(display, (pcx, pcy), pr, COLOR_YELLOW, 2)
            cv2.putText(display, prompt, (20, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
            cv2.imshow(window_name, display)
            cv2.waitKey(30)
            if clicks:
                return clicks[0]

    # Step 1: Circle center
    pt = wait_for_click("Click the CENTER of the felt circle")
    state.cal.circle_center = pt
    print(f"      Center set at ({pt[0]}, {pt[1]})")

    # Step 2: Circle edge
    pt = wait_for_click("Click the EDGE of the felt circle (at Bill's position)")
    cx, cy = state.cal.circle_center
    state.cal.circle_radius = int(np.hypot(pt[0] - cx, pt[1] - cy))
    print(f"      Radius: {state.cal.circle_radius}px")

    # Steps 3-7: Player zones
    for i, name in enumerate(PLAYER_NAMES):
        pt = wait_for_click(f"Click CENTER of {name}'s zone")
        zone_cx, zone_cy = pt
        print(f"      Center at ({zone_cx}, {zone_cy})")

        preview_center[0] = (zone_cx, zone_cy)
        pt2 = wait_for_click(f"Move mouse to set {name}'s zone size, then click")
        preview_center[0] = None

        zone_r = int(np.hypot(pt2[0] - zone_cx, pt2[1] - zone_cy))
        state.cal.zones.append({"name": name, "cx": zone_cx, "cy": zone_cy, "r": zone_r})
        print(f"      Zone '{name}' defined — radius {zone_r}px")

    cv2.destroyWindow(window_name)
    state.cal.save()
    print("\n  Calibration complete!")


def do_test_recognition(state):
    if not state.cal.is_complete:
        print("\n  Cannot test — calibrate first (press 'c')")
        return

    print("\n  Test Recognition Mode")
    print("  This will test card recognition in each player's zone one at a time.")
    print("  Make sure the table is CLEAR of all cards.")
    input("  Press Enter when table is clear...")

    # Capture baseline
    frame = capture_frame(state)
    if frame is None:
        print("  ERROR: Could not capture frame")
        return
    state.monitor.capture_baselines(frame, state.cal.zones)

    for zone in state.cal.zones:
        name = zone["name"]
        print(f"\n  --- Testing {name}'s zone ---")
        print(f"  Place a face-up card in {name}'s zone.")

        recognized = False
        attempts = 0
        max_attempts = 3

        while not recognized and attempts < max_attempts:
            # Poll for card every 2 seconds for up to 30 seconds
            card_found = False
            poll_start = time.time()

            while time.time() - poll_start < 30.0:
                frame = capture_frame(state)
                if frame is None:
                    time.sleep(2)
                    continue

                crop = state.monitor.check_single_zone(frame, zone)
                if crop is not None:
                    # Card detected — try recognition
                    log_buffer.log(f"[{name}] Card detected, recognizing...")
                    result = state.monitor.recognize_sync(name, crop)
                    if result != "No card":
                        card_found = True
                        break
                    else:
                        log_buffer.log(f"[{name}] API returned 'No card' — retrying...")

                time.sleep(2)

            if not card_found:
                attempts += 1
                if attempts < max_attempts:
                    print(f"  No card recognized. Try repositioning... (attempt {attempts + 1}/{max_attempts})")
                    time.sleep(3)
                continue

            result = state.monitor.last_card.get(name, "No card")
            print(f"  Recognized: {result}")
            speech.say(f"{name}, {result}")

            print(f"  Press Enter to confirm, or 'n' to retry...")
            resp = input("  ").strip().lower()
            if resp == "n":
                attempts += 1
                if attempts < max_attempts:
                    print(f"  Retrying... (attempt {attempts + 1}/{max_attempts})")
                continue
            else:
                print(f"  Confirmed!")
                recognized = True

        if not recognized:
            print(f"  Skipping {name}'s zone after {max_attempts} attempts.")

        if recognized:
            print(f"  Remove the card from {name}'s zone.")
            input("  Press Enter when card is removed...")

    print("\n  Test complete!")


def do_monitor(state):
    if not state.cal.is_complete:
        print("\n  Cannot monitor — calibrate first (press 'c')")
        return

    print("\n  Starting monitoring mode.")
    print("  Make sure all landing zones are EMPTY (no cards on the table).")
    input("  Press Enter when table is clear...")

    frame = capture_frame(state)
    if frame is None:
        print("  ERROR: Could not capture frame")
        return
    state.monitor.capture_baselines(frame, state.cal.zones)
    state.monitoring = True
    log_buffer.log("Monitoring STARTED — capturing every 2 seconds")
    print("  Place cards in landing zones — recognition is automatic.")
    print("  Press Enter to stop monitoring.")

    # Run monitoring in background
    def monitor_loop():
        while state.monitoring and not state.quit_flag:
            frame = capture_frame(state)
            if frame is not None:
                state.monitor.check_zones(frame, state.cal.zones)
            time.sleep(2)

    t = Thread(target=monitor_loop, daemon=True)
    t.start()

    input()  # Wait for Enter to stop
    state.monitoring = False
    log_buffer.log("Monitoring STOPPED")


def do_reset_baselines(state):
    if not state.cal.is_complete:
        print("\n  Cannot reset — calibrate first (press 'c')")
        return
    print("\n  Resetting baselines.")
    print("  Make sure all landing zones are EMPTY.")
    input("  Press Enter when table is clear...")
    frame = capture_frame(state)
    if frame is not None:
        state.monitor.capture_baselines(frame, state.cal.zones)
        print("  Baselines recaptured.")
    else:
        print("  ERROR: Could not capture frame")


def do_snapshot(state):
    frame = capture_frame(state)
    if frame is not None:
        # Crop to felt circle
        cropped = crop_to_felt_circle(frame, state.cal)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = Path(__file__).parent / f"snapshot_{ts}.jpg"
        cv2.imwrite(str(snap_path), cropped)
        print(f"\n  Snapshot saved to {snap_path} (cropped to felt circle)")
    else:
        print("\n  ERROR: Could not capture frame")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overhead camera card recognition test")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f"Camera index (default {DEFAULT_CAMERA_INDEX})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Change detection threshold (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION,
                        help=f"Capture resolution WxH (default {DEFAULT_RESOLUTION})")
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    capture = FrameCapture(args.camera, args.resolution)
    log_buffer.log(f"Camera {args.camera}, resolution {args.resolution}")

    # Test capture
    print(f"  Testing capture...")
    frame = capture.capture()
    if frame is None:
        sys.exit("  ERROR: Could not capture a frame. Check camera index and ffmpeg.")
    print(f"  Capture OK: {frame.shape[1]}x{frame.shape[0]}")

    cal = Calibration()
    cal.load()

    monitor = ZoneMonitor(threshold=args.threshold)
    state = AppState(capture, cal, monitor)
    state.latest_frame = frame

    start_debug_server(state)

    # Background capture loop — keeps live view fresh
    def background_capture():
        while not state.quit_flag:
            capture_frame(state)
            time.sleep(2)

    bg_thread = Thread(target=background_capture, daemon=True)
    bg_thread.start()

    # Open live view in default browser
    time.sleep(1)  # let first background capture complete
    subprocess.Popen(["open", f"http://localhost:{DEBUG_PORT}/live"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if cal.is_complete:
        print(f"  Calibration loaded — {len(cal.zones)} zones defined")
    else:
        print("  No calibration found")

    print_menu()

    try:
        while not state.quit_flag:
            cmd = input("\n  Enter command: ").strip().lower()

            if cmd == "q":
                print("\n  Shutting down...")
                break
            elif cmd == "c":
                do_calibrate(state)
            elif cmd == "t":
                do_test_recognition(state)
            elif cmd == "m":
                do_monitor(state)
            elif cmd == "r":
                do_reset_baselines(state)
            elif cmd == "s":
                do_snapshot(state)
            elif cmd == "":
                continue
            else:
                print(f"  Unknown command: '{cmd}'")
                print_menu()

    except (KeyboardInterrupt, EOFError):
        print("\n  Interrupted.")

    print("  Done.")


if __name__ == "__main__":
    main()

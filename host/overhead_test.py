#!/usr/bin/env python3
"""
Overhead Camera Card Recognition Test Harness

Single-page browser UI at http://localhost:8888
Terminal is only used for startup — all interaction in the browser.

Usage:
    python overhead_test.py [--camera 0] [--threshold 30.0] [--resolution auto]
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
from threading import Thread, Lock
from queue import Queue, Empty

import cv2
import http.server
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAYER_NAMES = ["Steve", "Bill", "David", "Joe", "Rodney"]
NUM_ZONES = len(PLAYER_NAMES)

DEFAULT_CAMERA_INDEX = 0
DEFAULT_THRESHOLD = 30.0
DEFAULT_RESOLUTION = "auto"

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
TRAINING_DIR = Path(__file__).parent / "training_data"
CONFIG_FILE = Path(__file__).parent.parent / "local" / "config.json"
CAPTURE_FILE = Path("/tmp/card_scanner_frame.jpg")

MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Speech queue
# ---------------------------------------------------------------------------

class SpeechQueue:
    def __init__(self):
        self._queue = Queue()
        Thread(target=self._run, daemon=True).start()

    def say(self, phrase):
        self._queue.put(phrase)

    def _run(self):
        while True:
            phrase = self._queue.get()
            latest = {phrase: phrase}
            try:
                while True:
                    latest[self._queue.get_nowait()] = True
            except Empty:
                pass
            for p in latest:
                subprocess.run(["say", p], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)

speech = SpeechQueue()

# ---------------------------------------------------------------------------
# Log buffer
# ---------------------------------------------------------------------------

class LogBuffer:
    def __init__(self, maxlines=200):
        self._lines = []
        self._lock = Lock()

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(f"  {msg}")
        with self._lock:
            self._lines.append(line)
            self._lines = self._lines[-200:]

    def get(self, n=50):
        with self._lock:
            return list(self._lines[-n:])

log = LogBuffer()

# ---------------------------------------------------------------------------
# Frame capture via ffmpeg
# ---------------------------------------------------------------------------

class FrameCapture:
    def __init__(self, camera_index, resolution="auto"):
        self.camera_index = camera_index
        self._check_ffmpeg()
        self.resolution = self._find_best_resolution() if resolution == "auto" else resolution
        w, h = self.resolution.split("x")
        self.width, self.height = int(w), int(h)

    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        except FileNotFoundError:
            print("  ERROR: ffmpeg not found. Install with: brew install ffmpeg")
            sys.exit(1)

    def _find_best_resolution(self):
        log.log("Auto-detecting resolution...")
        for res in ["3840x2160", "2560x1440", "1920x1080", "1280x720"]:
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
                    "-f", "avfoundation", "-video_size", res, "-framerate", "5",
                    "-i", f"{self.camera_index}:none", "-frames:v", "1",
                    "-q:v", "2", str(CAPTURE_FILE)
                ], capture_output=True, timeout=10, stdin=subprocess.DEVNULL)
                frame = cv2.imread(str(CAPTURE_FILE))
                if frame is not None and f"{frame.shape[1]}x{frame.shape[0]}" == res:
                    log.log(f"  {res} — OK")
                    return res
                log.log(f"  {res} — skipped")
            except Exception:
                log.log(f"  {res} — failed")
        return "1920x1080"

    def capture(self):
        try:
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
                "-f", "avfoundation", "-video_size", self.resolution,
                "-framerate", "5", "-i", f"{self.camera_index}:none",
                "-frames:v", "1", "-q:v", "2", str(CAPTURE_FILE)
            ], capture_output=True, timeout=10, stdin=subprocess.DEVNULL)
            return cv2.imread(str(CAPTURE_FILE))
        except Exception as e:
            log.log(f"Capture error: {e}")
            return None

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class Calibration:
    def __init__(self):
        self.circle_center = None
        self.circle_radius = None
        self.zones = []

    def save(self):
        data = {"circle_center": list(self.circle_center) if self.circle_center else None,
                "circle_radius": self.circle_radius, "zones": self.zones}
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)
        log.log("Calibration saved")

    def load(self):
        if not CALIBRATION_FILE.exists():
            return False
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
        cc = data.get("circle_center")
        self.circle_center = tuple(cc) if cc else None
        self.circle_radius = data.get("circle_radius")
        self.zones = data.get("zones", [])
        if self.zones and "cx" not in self.zones[0]:
            self.zones = []
            return False
        return True

    @property
    def ok(self):
        return self.circle_center and self.circle_radius and len(self.zones) == NUM_ZONES

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def crop_circle(frame, cal):
    if not cal.circle_center or not cal.circle_radius:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, cal.circle_center, cal.circle_radius, 255, -1)
    return cv2.bitwise_and(frame, frame, mask=mask)

def draw_overlay(frame, cal, monitor):
    if cal.circle_center and cal.circle_radius:
        cv2.circle(frame, cal.circle_center, cal.circle_radius, (255,255,255), 2)
    for z in cal.zones:
        name, cx, cy, r = z["name"], z["cx"], z["cy"], z["r"]
        zs = monitor.zone_state.get(name, "empty")
        color = {"recognized":(0,255,0), "processing":(0,255,255)}.get(zs, (255,255,255))
        cv2.circle(frame, (cx,cy), r, color, 2)
        cv2.putText(frame, name, (cx-30, cy-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        card = monitor.last_card.get(name, "")
        if card:
            cv2.putText(frame, card, (cx-60, cy+r+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def to_jpeg(frame, q=85):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else None

# ---------------------------------------------------------------------------
# Zone monitor
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
                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key and CONFIG_FILE.exists():
                    key = json.loads(CONFIG_FILE.read_text()).get("anthropic_api_key")
                if key and key != "YOUR_KEY_HERE":
                    self._client = anthropic.Anthropic(api_key=key)
                else:
                    log.log("WARNING: No API key")
            except ImportError:
                log.log("WARNING: anthropic not installed")
        return self._client

    def capture_baselines(self, frame):
        for z in _state.cal.zones:
            crop = self._crop(frame, z)
            if crop is not None and crop.size > 0:
                self.baselines[z["name"]] = crop.copy()
                self.zone_state[z["name"]] = "empty"
                self.last_card[z["name"]] = ""
                self.pending[z["name"]] = False
        log.log("Baselines captured")

    def check_zones(self, frame):
        for z in _state.cal.zones:
            name = z["name"]
            if name not in self.baselines or self.pending.get(name):
                continue
            crop = self._crop(frame, z)
            if crop is None or crop.size == 0:
                continue
            bl = self.baselines[name]
            if crop.shape != bl.shape:
                continue
            diff = float(np.mean(cv2.absdiff(crop, bl)))

            if self.zone_state.get(name) == "recognized":
                if diff < self.threshold:
                    self.zone_state[name] = "empty"
                    self.last_card[name] = ""
                continue

            if diff > self.threshold:
                self.zone_state[name] = "processing"
                self.pending[name] = True
                Thread(target=self._recognize, args=(name, crop.copy()), daemon=True).start()

    def check_single(self, frame, zone):
        name = zone["name"]
        if name not in self.baselines:
            return None
        crop = self._crop(frame, zone)
        if crop is None or crop.size == 0:
            return None
        bl = self.baselines[name]
        if crop.shape != bl.shape:
            return None
        if float(np.mean(cv2.absdiff(crop, bl))) > self.threshold:
            return crop.copy()
        return None

    def _crop(self, frame, z):
        h, w = frame.shape[:2]
        cx, cy, r = z["cx"], z["cy"], z["r"]
        x1, y1 = max(0, cx-r), max(0, cy-r)
        x2, y2 = min(w, cx+r), min(h, cy+r)
        return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None

    def _recognize(self, name, crop):
        t0 = time.time()
        try:
            if not self.client:
                self.zone_state[name] = "empty"
                return
            b64 = base64.b64encode(
                cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            ).decode()
            resp = self.client.messages.create(
                model=MODEL, max_tokens=20,
                messages=[{"role":"user","content":[
                    {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},
                    {"type":"text","text":"What playing card is this? Reply ONLY: 'Rank of Suit' (e.g. '4 of Clubs'). If unclear: 'No card'"},
                ]}])
            raw = resp.content[0].text.strip()
            m = re.search(r'(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)', raw, re.I)
            result = f"{m.group(1).capitalize()} of {m.group(2).capitalize()}" if m else "No card"
            self.last_card[name] = result
            self.zone_state[name] = "recognized"
            log.log(f"{name}: {result}  ({time.time()-t0:.1f}s)")
            self._save(name, crop, result)
            if "no card" not in result.lower():
                speech.say(f"{name}, {result}")
        except Exception as e:
            log.log(f"[{name}] error: {e}")
            self.zone_state[name] = "empty"
        finally:
            self.pending[name] = False

    def _save(self, name, crop, result):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = result.replace(" ","_").replace("/","-")[:30]
        cv2.imwrite(str(TRAINING_DIR / f"{ts}_{name}_{safe}.jpg"), crop)
        (TRAINING_DIR / f"{ts}_{name}_{safe}.txt").write_text(result)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, capture, cal, monitor):
        self.capture = capture
        self.cal = cal
        self.monitor = monitor
        self.monitoring = False
        self.latest_frame = None
        self.latest_jpg = None  # cropped + overlay
        self.quit_flag = False
        self.test_mode = None   # None or {"zone_idx":0, "waiting":"card"|"confirm", "result":""}
        # Deal test mode — dictation via text input
        self.deal_mode = None   # None or {"game":..., "cards":[...], "last_text":"", "last_parsed":""}

_state = None

# ---------------------------------------------------------------------------
# Deal mode — dictation via text input
# ---------------------------------------------------------------------------

def _start_deal_mode(s):
    if s.deal_mode:
        return
    s.deal_mode = {"game": None, "cards": [], "last_text": "", "last_parsed": ""}
    log.log("Deal mode started — use Dictation (Fn Fn) in the text field")


def _stop_deal_mode(s):
    s.deal_mode = None
    log.log("Deal mode stopped")


def _process_deal_text(s, text):
    """Parse dictated text for game names and card calls."""
    from speech_recognition_module import parse_speech, GameCommand, CardCallCommand, UnrecognizedCommand

    if not s.deal_mode:
        return

    # Only process new text (what was added since last parse)
    old = s.deal_mode["last_text"]
    s.deal_mode["last_text"] = text

    # Find new portion
    if text.lower().startswith(old.lower()) and len(text) > len(old):
        new_text = text[len(old):].strip()
    else:
        new_text = text.strip()

    if not new_text:
        return

    s.deal_mode["last_parsed"] = new_text
    log.log(f"[DEAL] Parsing: \"{new_text}\"")

    commands = parse_speech(new_text)
    for cmd in commands:
        if isinstance(cmd, GameCommand):
            s.deal_mode["game"] = cmd.game_name
            log.log(f"[DEAL] Game: {cmd.game_name}")
        elif isinstance(cmd, CardCallCommand):
            card_str = f"{cmd.rank} of {cmd.suit}"
            s.deal_mode["cards"].append({"player": cmd.player, "card": card_str})
            log.log(f"[DEAL] {cmd.player}: {card_str}")
        elif isinstance(cmd, UnrecognizedCommand):
            log.log(f"[DEAL] Unrecognized: \"{cmd.raw_text}\"")


# ---------------------------------------------------------------------------
# Background capture
# ---------------------------------------------------------------------------

def bg_loop():
    while not _state.quit_flag:
        frame = _state.capture.capture()
        if frame is not None:
            _state.latest_frame = frame
            disp = crop_circle(frame, _state.cal).copy()
            draw_overlay(disp, _state.cal, _state.monitor)
            _state.latest_jpg = to_jpeg(disp)
            if _state.monitoring and _state.cal.ok:
                _state.monitor.check_zones(frame)
            # Test mode: check if card appeared in the active zone
            tm = _state.test_mode
            if tm and tm["waiting"] == "card" and _state.cal.ok:
                zone = _state.cal.zones[tm["zone_idx"]]
                crop = _state.monitor.check_single(frame, zone)
                if crop is not None:
                    log.log(f"[{zone['name']}] Card detected, recognizing...")
                    result = _state.monitor.last_card.get(zone["name"], "")
                    if not result or result == "No card":
                        _state.monitor._recognize(zone["name"], crop)
                        result = _state.monitor.last_card.get(zone["name"], "No card")
                    if result and result != "No card":
                        tm["result"] = result
                        tm["waiting"] = "confirm"
                        speech.say(f"{zone['name']}, {result}")
        time.sleep(2)

# ---------------------------------------------------------------------------
# Web server — single page app
# ---------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        s = _state
        if not s: return self._r(500,"text/plain","Not ready")
        p = self.path.split("?")[0]
        routes = {
            "/": self._page, "/app": self._page,
            "/calibrate": self._calibrate_page,
            "/snapshot": lambda s: self._jpeg(s.latest_frame),
            "/snapshot/cropped": lambda s: self._r(200,"image/jpeg",s.latest_jpg) if s.latest_jpg else self._r(503,"text/plain","wait"),
            "/api/state": self._api_state,
            "/api/log": lambda s: self._r(200,"application/json",json.dumps({"lines":log.get(100)})),
            "/log": lambda s: self._r(200,"text/plain","\n".join(log.get(200))),
            "/calibration": lambda s: self._r(200,"application/json",CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else self._r(404,"text/plain","none"),
            "/training": self._training_list,
        }
        if p in routes:
            routes[p](s)
        elif p.startswith("/zone/"):
            self._zone_img(s, p[6:])
        elif p.startswith("/training/"):
            self._training_file(p[10:])
        else:
            self._r(404,"text/plain","Not found")

    def do_POST(self):
        s = _state
        if not s: return self._r(500,"text/plain","Not ready")
        body = self.rfile.read(int(self.headers.get("Content-Length",0))).decode()
        data = json.loads(body) if body else {}
        p = self.path

        if p == "/api/calibrate/save":
            cc = data.get("circle_center")
            s.cal.circle_center = tuple(cc) if cc else None
            s.cal.circle_radius = data.get("circle_radius")
            s.cal.zones = data.get("zones", [])
            s.cal.save()
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/monitor/start":
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
                s.monitoring = True
            self._r(200,"application/json",json.dumps({"monitoring":s.monitoring}))

        elif p == "/api/monitor/stop":
            s.monitoring = False
            self._r(200,"application/json",'{"monitoring":false}')

        elif p == "/api/baselines":
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/start":
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
                s.test_mode = {"zone_idx": 0, "waiting": "card", "result": ""}
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/confirm":
            tm = s.test_mode
            if tm:
                correct = data.get("correct", True)
                if correct:
                    tm["zone_idx"] += 1
                    if tm["zone_idx"] >= len(s.cal.zones):
                        s.test_mode = None
                    else:
                        tm["waiting"] = "card"
                        tm["result"] = ""
                else:
                    # Retry same zone
                    tm["waiting"] = "card"
                    tm["result"] = ""
                    name = s.cal.zones[tm["zone_idx"]]["name"]
                    s.monitor.zone_state[name] = "empty"
                    s.monitor.last_card[name] = ""
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/skip":
            tm = s.test_mode
            if tm:
                tm["zone_idx"] += 1
                if tm["zone_idx"] >= len(s.cal.zones):
                    s.test_mode = None
                else:
                    tm["waiting"] = "card"
                    tm["result"] = ""
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/stop":
            s.test_mode = None
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/start":
            _start_deal_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/stop":
            _stop_deal_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/text":
            text = data.get("text", "")
            _process_deal_text(s, text)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/clear":
            if s.deal_mode:
                s.deal_mode["cards"] = []
                s.deal_mode["game"] = None
                s.deal_mode["last_text"] = ""
                s.deal_mode["last_parsed"] = ""
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/snapshot/save":
            if s.latest_frame is not None:
                cropped = crop_circle(s.latest_frame, s.cal)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = Path(__file__).parent / f"snapshot_{ts}.jpg"
                cv2.imwrite(str(path), cropped)
                log.log(f"Snapshot saved: {path.name}")
            self._r(200,"application/json",'{"ok":true}')

        else:
            self._r(404,"text/plain","Not found")

    def _r(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        if ct == "image/jpeg":
            self.send_header("Cache-Control","no-store,no-cache,max-age=0")
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body,str) else body)

    def _jpeg(self, frame):
        if frame is None: return self._r(503,"text/plain","No frame")
        j = to_jpeg(frame, 80)
        if j: self._r(200,"image/jpeg",j)

    def _api_state(self, s):
        tm = s.test_mode
        test_info = None
        if tm:
            idx = tm["zone_idx"]
            test_info = {
                "zone": s.cal.zones[idx]["name"] if idx < len(s.cal.zones) else None,
                "zone_idx": idx,
                "total": len(s.cal.zones),
                "waiting": tm["waiting"],
                "result": tm["result"],
            }
        self._r(200,"application/json",json.dumps({
            "monitoring": s.monitoring,
            "calibrated": s.cal.ok,
            "resolution": s.capture.resolution,
            "test_mode": test_info,
            "deal_mode": s.deal_mode,
            "zones": {z["name"]: {"state": s.monitor.zone_state.get(z["name"],"empty"),
                                   "card": s.monitor.last_card.get(z["name"],"")}
                      for z in s.cal.zones},
        }))

    def _page(self, s):
        players_js = json.dumps(PLAYER_NAMES)
        self._r(200,"text/html",f"""<!DOCTYPE html>
<html><head><title>Card Scanner</title><meta name="viewport" content="width=device-width">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:12px}}
button{{padding:8px 14px;border:none;border-radius:6px;cursor:pointer;font-size:.9em;margin:3px}}
.btn-blue{{background:#0f3460;color:#fff}} .btn-blue:hover{{background:#1a4a7a}}
.btn-green{{background:#1b5e20;color:#fff}} .btn-green:hover{{background:#2e7d32}}
.btn-red{{background:#b71c1c;color:#fff}} .btn-red:hover{{background:#c62828}}
.btn-orange{{background:#e65100;color:#fff}}
.btn-off{{background:#333;color:#888}}
img{{border:1px solid #333;border-radius:4px}}
#toolbar{{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px;padding:8px;background:#16213e;border-radius:8px}}
#main{{display:flex;gap:12px;flex-wrap:wrap}}
#left{{flex:1;min-width:300px}} #right{{width:320px;flex-shrink:0}}
.zone{{padding:8px;margin:4px 0;border:2px solid #444;border-radius:8px;display:flex;align-items:center;gap:8px}}
.zone img{{width:100px;height:100px;object-fit:cover;border-radius:4px}}
.zone-info{{flex:1}}
.zone-name{{font-weight:bold;font-size:1.1em}}
.zone-card{{font-size:1.2em;margin-top:2px}}
pre{{background:#0d1117;padding:8px;border-radius:6px;font-size:.8em;max-height:200px;overflow:auto;margin-top:8px}}
#test-panel{{background:#0f3460;padding:12px;border-radius:8px;margin:8px 0;display:none}}
#test-panel h3{{margin-bottom:8px}}
#status-bar{{font-size:.85em;color:#888;margin-top:8px}}
</style></head><body>
<div id="toolbar">
  <button class="btn-blue" onclick="location.href='/calibrate'">Calibrate</button>
  <button class="btn-green" id="btn-monitor" onclick="toggleMonitor()">Start Monitor</button>
  <button class="btn-blue" onclick="startTest()">Test Recognition</button>
  <button class="btn-blue" id="btn-deal" onclick="toggleDeal()">Test Dealing</button>
  <button class="btn-blue" onclick="resetBaselines()">Reset Baselines</button>
  <button class="btn-blue" onclick="saveSnapshot()">Snapshot</button>
  <span id="status-bar">Loading...</span>
</div>
<div id="test-panel">
  <h3 id="test-title">Test Mode</h3>
  <p id="test-prompt"></p>
  <div id="test-buttons" style="margin-top:8px">
    <button class="btn-green" onclick="testConfirm(true)">Correct</button>
    <button class="btn-red" onclick="testConfirm(false)">Incorrect</button>
    <button class="btn-orange" onclick="testSkip()">Skip</button>
    <button class="btn-off" onclick="testStop()">Stop Test</button>
  </div>
</div>
<div id="deal-panel" style="display:none;background:#0f3460;padding:12px;border-radius:8px;margin:8px 0">
  <h3>Test Dealing — Dictation</h3>
  <p style="margin:4px 0">Game: <span id="deal-game" style="color:#4fc3f7;font-size:1.1em">—</span></p>
  <p style="margin:4px 0;font-size:.85em;color:#888">Click the text field, press Fn twice to start Dictation, then call cards.</p>
  <input id="deal-input" type="text" placeholder="Dictate here: 'The game is 5 Card Draw. Bill, ace of spades...'"
    style="width:100%;padding:8px;border-radius:6px;border:1px solid #444;background:#1a1a2e;color:#e0e0e0;font-size:1em;margin:6px 0"
    oninput="dealTextChanged()">
  <div id="deal-cards" style="margin:8px 0"></div>
  <button class="btn-orange" onclick="clearDeal()">Clear</button>
  <button class="btn-red" onclick="toggleDeal()">Stop Dealing</button>
</div>
<div id="main">
  <div id="left">
    <img id="tableimg" src="/snapshot/cropped" style="width:100%;cursor:pointer"
         onclick="this.src='/snapshot/cropped?'+Date.now()">
    <div style="margin-top:8px">
      <h3 style="display:inline">Log</h3>
      <button class="btn-off" style="float:right;padding:3px 8px;font-size:.75em" onclick="copyLog()">Copy Log</button>
    </div>
    <pre id="logpre" style="max-height:300px"></pre>
    <p style="margin-top:4px;font-size:.8em">
      <a href="/log" style="color:#4fc3f7">Full log</a> |
      <a href="/training" style="color:#4fc3f7">Training data</a>
    </p>
  </div>
  <div id="right">
    <h3>Zones</h3>
    <div id="zones"></div>
  </div>
</div>
<script>
var players={players_js};
var monitoring=false, testMode=null;

function api(path, data){{
  return fetch(path,{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify(data||{{}})}}).then(function(r){{return r.json()}});
}}

function toggleMonitor(){{
  if(monitoring) api('/api/monitor/stop').then(update);
  else api('/api/monitor/start').then(update);
}}
function startTest(){{
  if(!confirm('Clear all cards from table, then click OK')) return;
  api('/api/test/start').then(update);
}}
function testConfirm(correct){{ api('/api/test/confirm',{{correct:correct}}).then(update); }}
function testSkip(){{ api('/api/test/skip').then(update); }}
function testStop(){{ api('/api/test/stop').then(update); }}
function toggleDeal(){{
  fetch('/api/state').then(function(r){{return r.json()}}).then(function(d){{
    if(d.deal_mode) api('/api/deal/stop').then(update);
    else api('/api/deal/start').then(function(){{
      update();
      setTimeout(function(){{
        var inp=document.getElementById('deal-input');
        if(inp) inp.focus();
      }}, 300);
    }});
  }});
}}
function clearDeal(){{
  api('/api/deal/clear').then(function(){{
    var inp=document.getElementById('deal-input');
    if(inp) inp.value='';
    update();
  }});
}}
var _dealDebounce=null;
function dealTextChanged(){{
  clearTimeout(_dealDebounce);
  _dealDebounce=setTimeout(function(){{
    var inp=document.getElementById('deal-input');
    if(inp) api('/api/deal/text',{{text:inp.value}}).then(update);
  }}, 500);
}}

function resetBaselines(){{
  if(!confirm('Clear all cards, then click OK')) return;
  api('/api/baselines').then(function(){{ log.log && update(); }});
}}
function saveSnapshot(){{ api('/api/snapshot/save'); }}
function copyLog(){{
  window.open('/log','_blank');
}}

function update(){{
  // Table image
  var ti=new Image();
  ti.onload=function(){{document.getElementById('tableimg').src=ti.src}};
  ti.src='/snapshot/cropped?'+Date.now()+Math.random();

  // State
  fetch('/api/state').then(function(r){{return r.json()}}).then(function(d){{
    monitoring=d.monitoring;
    var btn=document.getElementById('btn-monitor');
    btn.textContent=monitoring?'Stop Monitor':'Start Monitor';
    btn.className=monitoring?'btn-red':'btn-green';

    // Status bar
    var sb='Resolution: '+d.resolution+' | '+(d.calibrated?'Calibrated':'NOT calibrated');
    sb+=' | Monitor: '+(d.monitoring?'ON':'OFF');
    document.getElementById('status-bar').textContent=sb;

    // Zones
    var zh='';
    players.forEach(function(name){{
      var z=d.zones[name]||{{}};
      var st=z.state||'empty';
      var card=z.card||'';
      var bc={{'recognized':'#4caf50','processing':'#ff9800'}}[st]||'#444';
      zh+='<div class="zone" style="border-color:'+bc+'">'
        +'<img src="/zone/'+name+'?'+Date.now()+'" onerror="this.style.display=\\'none\\'">'
        +'<div class="zone-info"><div class="zone-name">'+name+'</div>'
        +'<div style="color:#888;font-size:.8em">'+st+'</div>'
        +'<div class="zone-card" style="color:'+bc+'">'+card+'</div></div></div>';
    }});
    document.getElementById('zones').innerHTML=zh;

    // Deal mode panel
    var dp=document.getElementById('deal-panel');
    var dbtn=document.getElementById('btn-deal');
    if(d.deal_mode){{
      dp.style.display='block';
      dbtn.textContent='Stop Dealing';dbtn.className='btn-red';
      document.getElementById('deal-game').textContent=d.deal_mode.game||'—';
      var ch='';
      (d.deal_mode.cards||[]).forEach(function(c){{
        ch+='<span style="display:inline-block;margin:3px;padding:4px 8px;background:#1b5e20;border-radius:4px">'
          +c.player+': '+c.card+'</span>';
      }});
      document.getElementById('deal-cards').innerHTML=ch;
    }} else {{
      dp.style.display='none';
      dbtn.textContent='Test Dealing';dbtn.className='btn-blue';
    }}

    // Test mode panel
    var tp=document.getElementById('test-panel');
    var tb=document.getElementById('test-buttons');
    if(d.test_mode){{
      tp.style.display='block';
      var tm=d.test_mode;
      if(!tm.zone){{
        document.getElementById('test-title').textContent='Test Complete!';
        document.getElementById('test-prompt').textContent='';
        tb.innerHTML='<button class="btn-blue" onclick="testStop()">Done</button>';
      }} else if(tm.waiting=='card'){{
        document.getElementById('test-title').textContent='Testing: '+tm.zone+' ('+
          (tm.zone_idx+1)+'/'+tm.total+')';
        document.getElementById('test-prompt').textContent='Place a face-up card in '+tm.zone+'\\'s zone...';
        tb.innerHTML='<button class="btn-orange" onclick="testSkip()">Skip</button> '+
          '<button class="btn-off" onclick="testStop()">Stop Test</button>';
      }} else if(tm.waiting=='confirm'){{
        document.getElementById('test-title').textContent='Testing: '+tm.zone;
        document.getElementById('test-prompt').textContent='Recognized: '+tm.result;
        tb.innerHTML='<button class="btn-green" onclick="testConfirm(true)">Correct</button> '+
          '<button class="btn-red" onclick="testConfirm(false)">Incorrect</button> '+
          '<button class="btn-orange" onclick="testSkip()">Skip</button> '+
          '<button class="btn-off" onclick="testStop()">Stop Test</button>';
      }}
    }} else {{
      tp.style.display='none';
    }}
  }}).catch(function(){{}});

  // Log
  fetch('/api/log').then(function(r){{return r.json()}}).then(function(d){{
    var pre=document.getElementById('logpre');
    pre.innerHTML=d.lines.slice(-30).join('<br>');
    pre.scrollTop=pre.scrollHeight;
  }}).catch(function(){{}});
}}

setInterval(update, 2000);
update();
</script></body></html>""")

    def _calibrate_page(self, s):
        players_js = json.dumps(PLAYER_NAMES)
        self._r(200,"text/html",f"""<!DOCTYPE html>
<html><head><title>Calibrate</title>
<style>
body{{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px;margin:0}}
canvas{{border:1px solid #444;cursor:crosshair;display:block;margin:10px 0;max-width:100%}}
#status{{font-size:1.2em;color:#4fc3f7;margin:10px 0}}
button{{padding:8px 16px;background:#e94560;color:#fff;border:none;border-radius:6px;cursor:pointer;margin:5px}}
</style></head><body>
<h1>Calibration</h1>
<div id="status">Loading image...</div>
<canvas id="canvas"></canvas>
<button onclick="location.href='/'">Back</button>
<script>
var c=document.getElementById('canvas'),ctx=c.getContext('2d');
var players={players_js};
var steps=[],step=0,cc=null,cr=null,zones=[],pc=null,imgW=0,imgH=0;
steps.push({{p:'Click CENTER of felt circle',t:'cc'}});
steps.push({{p:"Click EDGE of felt circle (at Bill's position)",t:'ce'}});
players.forEach(function(n){{
  steps.push({{p:'Click CENTER of '+n+"'s zone",t:'zc',n:n}});
  steps.push({{p:'Set '+n+"'s zone size, then click",t:'ze',n:n}});
}});
var img=new Image();
img.onload=function(){{
  imgW=img.naturalWidth;imgH=img.naturalHeight;
  var sc=Math.min((window.innerWidth-40)/imgW,1);
  c.width=Math.round(imgW*sc);c.height=Math.round(imgH*sc);c.dataset.s=sc;
  draw();document.getElementById('status').textContent=steps[0].p;
}};
img.src='/snapshot?'+Date.now();
function S(){{return parseFloat(c.dataset.s)||1}}
function draw(){{
  var s=S();ctx.drawImage(img,0,0,c.width,c.height);
  if(cc&&cr){{ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(cc[0]*s,cc[1]*s,cr*s,0,Math.PI*2);ctx.stroke()}}
  zones.forEach(function(z){{ctx.strokeStyle='#0f0';ctx.lineWidth=2;ctx.beginPath();ctx.arc(z.cx*s,z.cy*s,z.r*s,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='#0f0';ctx.font='16px sans-serif';ctx.fillText(z.name,z.cx*s-20,z.cy*s-z.r*s-8)}});
  if(pc){{ctx.strokeStyle='#ff0';ctx.lineWidth=2;ctx.setLineDash([5,5]);ctx.beginPath();ctx.arc(pc[0]*S(),pc[1]*S(),(pc[2]||0)*S(),0,Math.PI*2);ctx.stroke();ctx.setLineDash([])}}
}}
function xy(e){{var r=c.getBoundingClientRect();return[Math.round((e.clientX-r.left)*imgW/r.width),Math.round((e.clientY-r.top)*imgH/r.height)]}}
c.onclick=function(e){{
  var p=xy(e),x=p[0],y=p[1];if(step>=steps.length)return;var st=steps[step];
  if(st.t=='cc'){{cc=[x,y];step++}}
  else if(st.t=='ce'){{cr=Math.round(Math.sqrt(Math.pow(x-cc[0],2)+Math.pow(y-cc[1],2)));step++}}
  else if(st.t=='zc'){{pc=[x,y,0];step++}}
  else if(st.t=='ze'){{var r=Math.round(Math.sqrt(Math.pow(x-pc[0],2)+Math.pow(y-pc[1],2)));zones.push({{name:st.n,cx:pc[0],cy:pc[1],r:r}});pc=null;step++}}
  draw();
  if(step<steps.length)document.getElementById('status').textContent=steps[step].p;
  else{{document.getElementById('status').textContent='Saving...';
    fetch('/api/calibrate/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{circle_center:cc,circle_radius:cr,zones:zones}})}})
    .then(function(){{document.getElementById('status').textContent='Done! Return to main page.'}})}}
}};
c.onmousemove=function(e){{if(!pc)return;var p=xy(e);pc[2]=Math.round(Math.sqrt(Math.pow(p[0]-pc[0],2)+Math.pow(p[1]-pc[1],2)));draw()}};
</script></body></html>""")

    def _zone_img(self, s, name):
        if not s.latest_frame is not None:
            return self._r(503,"text/plain","No frame")
        z = next((z for z in s.cal.zones if z["name"]==name), None)
        if not z: return self._r(404,"text/plain","Not found")
        crop = s.monitor._crop(s.latest_frame, z)
        if crop is None: return self._r(500,"text/plain","Crop failed")
        j = to_jpeg(crop, 90)
        if j: self._r(200,"image/jpeg",j)

    def _training_list(self, s):
        if not TRAINING_DIR.exists():
            return self._r(200,"text/html","<p>No data</p>")
        files = sorted(TRAINING_DIR.iterdir(), reverse=True)
        h = "<html><body style='font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px'><h1>Training Data</h1>"
        for f in files[:100]:
            if f.suffix == ".jpg":
                lbl = f.with_suffix(".txt").read_text() if f.with_suffix(".txt").exists() else ""
                h += f'<div style="display:inline-block;margin:8px;text-align:center"><img src="/training/{f.name}" width="150"><br><small>{f.stem[:30]}</small><br>{lbl}</div>'
        self._r(200,"text/html",h+"</body></html>")

    def _training_file(self, name):
        p = TRAINING_DIR / name
        if not p.exists(): return self._r(404,"text/plain","Not found")
        self._r(200, "image/jpeg" if p.suffix==".jpg" else "text/plain",
                p.read_bytes() if p.suffix==".jpg" else p.read_text())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _state
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    capture = FrameCapture(args.camera, args.resolution)
    log.log(f"Camera {args.camera}, resolution {capture.resolution}")

    print("  Testing capture...")
    frame = capture.capture()
    if frame is None:
        sys.exit("  ERROR: Could not capture. Check camera and ffmpeg.")
    print(f"  OK: {frame.shape[1]}x{frame.shape[0]}")

    cal = Calibration()
    cal.load()

    monitor = ZoneMonitor(threshold=args.threshold)
    _state = AppState(capture, cal, monitor)
    _state.latest_frame = frame

    # Start server
    server = http.server.HTTPServer(("0.0.0.0", 8888), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    log.log("Server at http://localhost:8888")

    # Start background capture
    Thread(target=bg_loop, daemon=True).start()

    # Open browser
    time.sleep(1)
    subprocess.Popen(["open", "http://localhost:8888"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if cal.ok:
        print(f"  Calibration: {len(cal.zones)} zones")
    else:
        print("  No calibration — use browser to calibrate")

    print("  All UI is in the browser. Press Ctrl+C to quit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        _state.quit_flag = True

if __name__ == "__main__":
    main()

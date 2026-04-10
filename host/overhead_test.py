#!/usr/bin/env python3
"""
Overhead Camera Card Recognition Test Harness

Captures still images from a Logitech Brio 4K camera using ffmpeg,
monitors landing zones on a poker table for card placement,
and uses Claude's vision API to identify cards.

All UI is browser-based — no OpenCV windows.
Debug dashboard at http://localhost:8888
Live view at http://localhost:8888/live
Calibration at http://localhost:8888/calibrate

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
# Constants
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
# Speech queue
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
# Frame capture via ffmpeg
# ---------------------------------------------------------------------------

class FrameCapture:
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

    def capture(self):
        path = str(CAPTURE_FILE)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-nostdin",
            "-f", "avfoundation",
            "-video_size", self.resolution,
            "-framerate", "5",
            "-i", f"{self.camera_index}:none",
            "-frames:v", "1",
            "-q:v", "2",
            path
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=10,
                           stdin=subprocess.DEVNULL)
            frame = cv2.imread(path)
            return frame
        except Exception as e:
            log_buffer.log(f"Capture error: {e}")
            return None


# ---------------------------------------------------------------------------
# Calibration
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
        log_buffer.log(f"Calibration saved to {path}")

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
            print("  Old calibration format — recalibrate (press 'c')")
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
    if cal.circle_center is None or cal.circle_radius is None:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, cal.circle_center, cal.circle_radius, 255, -1)
    return cv2.bitwise_and(frame, frame, mask=mask)

def draw_overlay(frame, cal, monitor, flash_zone=None, flash_on=False):
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
        color = {"recognized": COLOR_GREEN, "processing": COLOR_YELLOW}.get(zstate, COLOR_WHITE)

        cv2.circle(frame, (cx, cy), r, color, 2)
        cv2.putText(frame, name, (cx - 30, cy - r - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        card = monitor.last_card.get(name, "")
        if card:
            cv2.putText(frame, card, (cx - 60, cy + r + 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)

def frame_to_jpeg(frame, quality=85):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
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
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key and CONFIG_FILE.exists():
                    with open(CONFIG_FILE) as f:
                        cfg = json.load(f)
                        api_key = cfg.get("anthropic_api_key")
                if api_key and api_key != "YOUR_KEY_HERE":
                    self._client = anthropic.Anthropic(api_key=api_key)
                else:
                    log_buffer.log("WARNING: No valid API key.")
            except ImportError:
                log_buffer.log("WARNING: anthropic package not installed.")
        return self._client

    def capture_baselines(self, frame, zones):
        for zone in zones:
            crop = self._crop_zone(frame, zone)
            if crop is None or crop.size == 0:
                log_buffer.log(f"WARNING: zone '{zone['name']}' out of bounds")
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
                if float(np.mean(diff)) < self.threshold:
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
            if float(np.mean(diff)) > self.threshold:
                self.zone_state[name] = "processing"
                self.pending[name] = True
                Thread(target=self._recognize, args=(name, crop.copy()), daemon=True).start()

    def check_single_zone(self, frame, zone):
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
        if float(np.mean(diff)) > self.threshold:
            return crop.copy()
        return None

    def recognize_sync(self, name, crop):
        self._recognize(name, crop)
        return self.last_card.get(name, "No card")

    def _crop_zone(self, frame, zone):
        h, w = frame.shape[:2]
        cx, cy, r = zone["cx"], zone["cy"], zone["r"]
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _recognize(self, name, crop):
        t0 = time.time()
        try:
            if self.client is None:
                log_buffer.log(f"[{name}] API not available")
                self.zone_state[name] = "empty"
                return

            b64 = base64.b64encode(
                cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            ).decode("utf-8")

            response = self.client.messages.create(
                model=MODEL, max_tokens=20,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text":
                        "What playing card is this? Reply with ONLY the rank and suit "
                        "in exactly this format: 'Rank of Suit' (e.g. '4 of Clubs', "
                        "'King of Hearts'). If you cannot identify the card, reply "
                        "with exactly: 'No card'"},
                ]}],
            )

            raw = response.content[0].text.strip()
            result = self._parse_card_result(raw)
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
            raw, re.IGNORECASE)
        if match:
            return f"{match.group(1).capitalize()} of {match.group(2).capitalize()}"
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
        self.latest_frame = None
        self.latest_cropped_jpg = None
        self.quit_flag = False
        # Calibration click queue — browser POSTs click coords here
        self.cal_click_queue = Queue()


# ---------------------------------------------------------------------------
# Web server — all UI in browser
# ---------------------------------------------------------------------------

_state = None

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        s = _state
        if s is None:
            return self._r(500, "text/plain", "Not ready")

        p = self.path.split("?")[0]  # strip query params

        if p == "/" or p == "/debug":
            self._serve_dashboard(s)
        elif p == "/live":
            self._serve_live(s)
        elif p == "/calibrate":
            self._serve_calibrate_page(s)
        elif p == "/log":
            self._r(200, "text/plain", "\n".join(log_buffer.get_lines()))
        elif p == "/api/log":
            self._r(200, "application/json", json.dumps({"lines": log_buffer.get_lines()[-50:]}))
        elif p == "/snapshot":
            self._serve_frame(s)
        elif p == "/snapshot/cropped":
            self._serve_cropped(s)
        elif p == "/snapshot/raw":
            self._serve_frame(s)
        elif p.startswith("/zone/"):
            self._serve_zone(s, p[6:])
        elif p == "/calibration":
            if CALIBRATION_FILE.exists():
                self._r(200, "application/json", CALIBRATION_FILE.read_text())
            else:
                self._r(404, "text/plain", "No calibration")
        elif p == "/training":
            self._serve_training_list()
        elif p.startswith("/training/"):
            self._serve_training_file(p[10:])
        elif p == "/api/state":
            self._serve_api_state(s)
        else:
            self._r(404, "text/plain", "Not found")

    def do_POST(self):
        s = _state
        if s is None:
            return self._r(500, "text/plain", "Not ready")

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""

        p = self.path
        if p == "/api/calibrate/click":
            data = json.loads(body)
            s.cal_click_queue.put((int(data["x"]), int(data["y"])))
            self._r(200, "application/json", '{"ok":true}')
        else:
            self._r(404, "text/plain", "Not found")

    def _r(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        if ct == "image/jpeg":
            self.send_header("Cache-Control", "no-store, no-cache, max-age=0")
        self.end_headers()
        self.wfile.write(body.encode("utf-8") if isinstance(body, str) else body)

    def _serve_dashboard(self, s):
        zone_names_js = json.dumps([z["name"] for z in s.cal.zones])
        res = f"{s.capture.width}x{s.capture.height}"
        self._r(200, "text/html", f"""<!DOCTYPE html>
<html><head><title>Card Scanner Debug</title>
<style>
body{{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}}
a{{color:#4fc3f7}}img{{border:1px solid #444;margin:4px}}
pre{{background:#0d1117;padding:12px;border-radius:6px;max-height:400px;overflow:auto;font-size:0.85em}}
.zone{{display:inline-block;margin:8px;padding:12px;border:2px solid #888;border-radius:8px;min-width:120px;text-align:center}}
</style></head><body>
<h1>Overhead Card Scanner</h1>
<p>Resolution: {res} |
Monitoring: <span id="mon">...</span> |
Calibrated: {'Yes' if s.cal.is_complete else 'No'}</p>
<p><a href="/live">Live View</a> | <a href="/calibrate">Calibrate</a></p>
<h2>Snapshot</h2>
<img id="snap" src="/snapshot" width="640" style="cursor:pointer" onclick="this.src='/snapshot?'+Date.now()">
<h2>Zones</h2>
<div id="zones"></div>
<h2>Log (last 50 lines)</h2>
<pre id="logpre"></pre>
<p><a href="/log">Full log</a> | <a href="/calibration">Calibration JSON</a> |
<a href="/training">Training data</a></p>
<script>
var zoneNames={zone_names_js};
function update(){{
  // Update snapshot
  var img=new Image();
  img.onload=function(){{document.getElementById('snap').src=img.src}};
  img.src='/snapshot?'+Date.now()+Math.random();
  // Update zone images
  zoneNames.forEach(function(n){{
    var zi=document.getElementById('zimg_'+n);
    if(zi){{var ni=new Image();ni.onload=function(){{zi.src=ni.src}};ni.src='/zone/'+n+'?'+Date.now()}}
  }});
  // Update state
  fetch('/api/state').then(function(r){{return r.json()}}).then(function(d){{
    document.getElementById('mon').textContent=d.monitoring?'ON':'OFF';
    zoneNames.forEach(function(n){{
      var z=d.zones[n]||{{}};
      var el=document.getElementById('zstate_'+n);
      var cl=document.getElementById('zcard_'+n);
      var div=document.getElementById('zdiv_'+n);
      if(el) el.textContent=z.state||'empty';
      if(cl) cl.textContent=z.card||'';
      if(div){{
        var c={{'recognized':'#4caf50','processing':'#ff9800'}}[z.state]||'#888';
        div.style.borderColor=c;
        if(cl) cl.style.color=c;
      }}
    }});
  }}).catch(function(){{}});
  // Update log
  fetch('/api/log').then(function(r){{return r.json()}}).then(function(d){{
    document.getElementById('logpre').innerHTML=d.lines.join('<br>');
  }}).catch(function(){{}});
}}
// Build zone divs
var zh='';
zoneNames.forEach(function(n){{
  zh+='<div class="zone" id="zdiv_'+n+'"><b>'+n+'</b><br>'
    +'<span id="zstate_'+n+'">empty</span><br>'
    +'<span id="zcard_'+n+'" style="font-size:1.2em"></span><br>'
    +'<img id="zimg_'+n+'" src="/zone/'+n+'" width="150"></div>';
}});
document.getElementById('zones').innerHTML=zh;
setInterval(update,3000);
update();
</script></body></html>""")

    def _serve_live(self, s):
        self._r(200, "text/html", """<!DOCTYPE html>
<html><head><title>Table View</title>
<style>
body{margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}
</style>
<script>
function refresh(){
  // Check if server is still running
  fetch('/api/state').then(function(r){
    if(!r.ok) return setTimeout(refresh,2000);
    var img=new Image();
    img.onload=function(){document.getElementById('f').src=img.src;setTimeout(refresh,2000)};
    img.onerror=function(){setTimeout(refresh,2000)};
    img.src='/snapshot/cropped?'+Date.now()+Math.random();
  }).catch(function(){
    // Server gone — close this tab
    document.title='[Closed] Table View';
    document.body.innerHTML='<p style="color:#666;font-size:2em">Scanner stopped</p>';
  });
}
setTimeout(refresh,2000);
</script></head><body><img id="f" src="/snapshot/cropped"></body></html>""")

    def _serve_calibrate_page(self, s):
        players_js = json.dumps(PLAYER_NAMES)
        self._r(200, "text/html", f"""<!DOCTYPE html>
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
<button onclick="location.href='/'">Back to Dashboard</button>
<script>
var canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
var players={players_js};
var steps=[];
var step=0, circleCenter=null, circleRadius=null, zones=[];
var previewCenter=null;
var imgW=0, imgH=0;

steps.push({{prompt:'Click the CENTER of the felt circle',type:'circle_center'}});
steps.push({{prompt:'Click the EDGE of the felt circle (at Bill\\'s position)',type:'circle_edge'}});
for(var i=0;i<players.length;i++){{
  steps.push({{prompt:'Click CENTER of '+players[i]+'\\'s zone',type:'zone_center',name:players[i]}});
  steps.push({{prompt:'Move mouse to set '+players[i]+'\\'s zone size, then click',type:'zone_edge',name:players[i]}});
}}

var img=new Image();
img.onload=function(){{
  imgW=img.naturalWidth; imgH=img.naturalHeight;
  // Scale canvas to fit browser width (max 1200px)
  var maxW=Math.min(window.innerWidth-40, 1200);
  var scale=maxW/imgW;
  canvas.width=Math.round(imgW*scale);
  canvas.height=Math.round(imgH*scale);
  canvas.dataset.scale=scale;
  redraw();
  document.getElementById('status').textContent=steps[0].prompt;
}};
img.src='/snapshot/raw?'+Date.now();

function sc(){{ return parseFloat(canvas.dataset.scale)||1; }}

function redraw(){{
  var s=sc();
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
  // Draw circle
  if(circleCenter && circleRadius){{
    ctx.strokeStyle='#fff';ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(circleCenter[0]*s,circleCenter[1]*s,circleRadius*s,0,Math.PI*2);ctx.stroke();
  }}
  // Draw zones
  for(var i=0;i<zones.length;i++){{
    ctx.strokeStyle='#0f0';ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(zones[i].cx*s,zones[i].cy*s,zones[i].r*s,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='#0f0';ctx.font='16px sans-serif';
    ctx.fillText(zones[i].name,zones[i].cx*s-20,zones[i].cy*s-zones[i].r*s-8);
  }}
  // Draw preview circle
  if(previewCenter){{
    ctx.strokeStyle='#ff0';ctx.lineWidth=2;ctx.setLineDash([5,5]);
    ctx.beginPath();ctx.arc(previewCenter[0]*s,previewCenter[1]*s,(previewCenter[2]||0)*s,0,Math.PI*2);ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

canvas.addEventListener('click',function(e){{
  var rect=canvas.getBoundingClientRect();
  var s=sc();
  // Convert click to image coordinates (not canvas coordinates)
  var x=Math.round((e.clientX-rect.left)*(imgW/rect.width));
  var y=Math.round((e.clientY-rect.top)*(imgH/rect.height));

  if(step>=steps.length) return;
  var s=steps[step];

  if(s.type=='circle_center'){{
    circleCenter=[x,y];
    step++;
  }} else if(s.type=='circle_edge'){{
    var dx=x-circleCenter[0],dy=y-circleCenter[1];
    circleRadius=Math.round(Math.sqrt(dx*dx+dy*dy));
    step++;
  }} else if(s.type=='zone_center'){{
    previewCenter=[x,y,0];
    step++;
  }} else if(s.type=='zone_edge'){{
    var dx=x-previewCenter[0],dy=y-previewCenter[1];
    var r=Math.round(Math.sqrt(dx*dx+dy*dy));
    zones.push({{name:s.name,cx:previewCenter[0],cy:previewCenter[1],r:r}});
    previewCenter=null;
    step++;
  }}

  redraw();

  if(step<steps.length){{
    document.getElementById('status').textContent=steps[step].prompt;
  }} else {{
    document.getElementById('status').textContent='Saving calibration...';
    // POST calibration data
    fetch('/api/calibrate/click',{{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{
        type:'save',
        circle_center:circleCenter,
        circle_radius:circleRadius,
        zones:zones
      }})
    }}).then(function(){{
      document.getElementById('status').textContent='Calibration complete!';
    }});
  }}
}});

canvas.addEventListener('mousemove',function(e){{
  if(!previewCenter) return;
  var rect=canvas.getBoundingClientRect();
  var x=Math.round((e.clientX-rect.left)*(imgW/rect.width));
  var y=Math.round((e.clientY-rect.top)*(imgH/rect.height));
  var dx=x-previewCenter[0],dy=y-previewCenter[1];
  previewCenter[2]=Math.round(Math.sqrt(dx*dx+dy*dy));
  redraw();
}});
</script></body></html>""")

    def _serve_frame(self, s):
        if s.latest_frame is None:
            return self._r(503, "text/plain", "No frame")
        jpg = frame_to_jpeg(s.latest_frame, 80)
        if jpg:
            self._r(200, "image/jpeg", jpg)

    def _serve_cropped(self, s):
        if s.latest_cropped_jpg:
            self._r(200, "image/jpeg", s.latest_cropped_jpg)
        else:
            self._r(503, "text/plain", "No frame")

    def _serve_zone(self, s, name):
        if s.latest_frame is None:
            return self._r(503, "text/plain", "No frame")
        zone = next((z for z in s.cal.zones if z["name"] == name), None)
        if not zone:
            return self._r(404, "text/plain", "Zone not found")
        crop = s.monitor._crop_zone(s.latest_frame, zone)
        if crop is None:
            return self._r(500, "text/plain", "Crop failed")
        jpg = frame_to_jpeg(crop, 90)
        if jpg:
            self._r(200, "image/jpeg", jpg)

    def _serve_training_list(self):
        if not TRAINING_DIR.exists():
            return self._r(200, "text/html", "<p>No training data</p>")
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
        self._r(200, "text/html", html)

    def _serve_training_file(self, filename):
        path = TRAINING_DIR / filename
        if not path.exists():
            return self._r(404, "text/plain", "Not found")
        if path.suffix == ".jpg":
            self._r(200, "image/jpeg", path.read_bytes())
        elif path.suffix == ".txt":
            self._r(200, "text/plain", path.read_text())

    def _serve_api_state(self, s):
        data = {
            "monitoring": s.monitoring,
            "calibrated": s.cal.is_complete,
            "zones": {z["name"]: {"state": s.monitor.zone_state.get(z["name"], "empty"),
                                   "card": s.monitor.last_card.get(z["name"], "")}
                      for z in s.cal.zones},
        }
        self._r(200, "application/json", json.dumps(data))


PORT = 8888

def start_server(state):
    global _state
    _state = state
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    log_buffer.log(f"Server running at http://localhost:{PORT}")


# ---------------------------------------------------------------------------
# Background capture loop
# ---------------------------------------------------------------------------

def update_frame(state):
    """Called only by background capture thread."""
    frame = state.capture.capture()
    if frame is not None:
        state.latest_frame = frame.copy()
        cropped = crop_to_felt_circle(frame, state.cal)
        display = cropped.copy()
        draw_overlay(display, state.cal, state.monitor)
        jpg = frame_to_jpeg(display, 85)
        if jpg:
            state.latest_cropped_jpg = jpg
    return frame


# ---------------------------------------------------------------------------
# Terminal commands
# ---------------------------------------------------------------------------

def print_menu():
    print("\n╔══════════════════════════════════════════╗")
    print("║       Overhead Card Scanner              ║")
    print("╠══════════════════════════════════════════╣")
    print("║  c = calibrate (opens browser)           ║")
    print("║  t = test recognition (zone by zone)     ║")
    print("║  m = start/stop monitoring               ║")
    print("║  r = reset baselines (clear table first) ║")
    print("║  s = save a snapshot                     ║")
    print("║  q = quit                                ║")
    print("╚══════════════════════════════════════════╝")


def do_calibrate(state):
    print("\n  Opening calibration page in browser...")
    print("  Click on the image to define the felt circle and player zones.")
    subprocess.Popen(["open", f"http://localhost:{PORT}/calibrate"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  Waiting for calibration to complete...")

    # Wait for the save POST from the browser
    while True:
        try:
            data = state.cal_click_queue.get(timeout=1.0)
            if isinstance(data, tuple):
                # Old click format — ignore
                continue
        except Empty:
            continue

        # We got the save data from the POST handler
        if isinstance(data, dict) and data.get("type") == "save":
            cc = data.get("circle_center")
            state.cal.circle_center = tuple(cc) if cc else None
            state.cal.circle_radius = data.get("circle_radius")
            state.cal.zones = data.get("zones", [])
            state.cal.save()
            print("  Calibration complete!")
            return


def do_test_recognition(state):
    if not state.cal.is_complete:
        print("\n  Cannot test — calibrate first (press 'c')")
        return

    print("\n  Test Recognition Mode")
    print("  Make sure the table is CLEAR of all cards.")
    input("  Press Enter when table is clear...")

    frame = state.latest_frame
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

        while not recognized and attempts < 3:
            card_found = False
            poll_start = time.time()

            while time.time() - poll_start < 30.0:
                frame = state.latest_frame
                if frame is None:
                    time.sleep(2)
                    continue

                crop = state.monitor.check_single_zone(frame, zone)
                if crop is not None:
                    log_buffer.log(f"[{name}] Card detected, recognizing...")
                    result = state.monitor.recognize_sync(name, crop)
                    if result != "No card":
                        card_found = True
                        break
                    log_buffer.log(f"[{name}] 'No card' — retrying...")
                time.sleep(2)

            if not card_found:
                attempts += 1
                if attempts < 3:
                    print(f"  Not recognized. Try repositioning... (attempt {attempts + 1}/3)")
                    time.sleep(3)
                continue

            result = state.monitor.last_card.get(name, "No card")
            print(f"  Recognized: {result}")
            speech.say(f"{name}, {result}")

            resp = input("  Press Enter to confirm, or 'n' to retry: ").strip().lower()
            if resp == "n":
                attempts += 1
                continue
            recognized = True

        if not recognized:
            print(f"  Skipping {name}'s zone.")

        if recognized:
            print(f"  Remove the card from {name}'s zone.")
            input("  Press Enter when removed...")

    print("\n  Test complete!")


def do_monitor(state):
    if not state.cal.is_complete:
        print("\n  Cannot monitor — calibrate first (press 'c')")
        return

    print("\n  Make sure all landing zones are EMPTY.")
    input("  Press Enter when table is clear...")

    frame = state.latest_frame
    if frame is None:
        print("  ERROR: Could not capture frame")
        return
    state.monitor.capture_baselines(frame, state.cal.zones)
    state.monitoring = True
    log_buffer.log("Monitoring STARTED")
    print("  Monitoring active. Press Enter to stop.")

    def loop():
        while state.monitoring and not state.quit_flag:
            frame = state.latest_frame
            if frame is not None:
                state.monitor.check_zones(frame, state.cal.zones)
            time.sleep(2)

    Thread(target=loop, daemon=True).start()
    input()
    state.monitoring = False
    log_buffer.log("Monitoring STOPPED")


def do_snapshot(state):
    frame = state.latest_frame
    if frame is not None:
        cropped = crop_to_felt_circle(frame, state.cal)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(__file__).parent / f"snapshot_{ts}.jpg"
        cv2.imwrite(str(path), cropped)
        print(f"\n  Saved: {path}")


# ---------------------------------------------------------------------------
# Handle calibration save POST
# ---------------------------------------------------------------------------

# Override the POST handler to also queue full save data
_orig_do_post = Handler.do_POST
def _patched_do_post(self):
    s = _state
    if s is None:
        return self._r(500, "text/plain", "Not ready")

    length = int(self.headers.get("Content-Length", 0))
    body = self.rfile.read(length).decode("utf-8") if length else ""
    p = self.path

    if p == "/api/calibrate/click":
        data = json.loads(body)
        if data.get("type") == "save":
            s.cal_click_queue.put(data)
        self._r(200, "application/json", '{"ok":true}')
    else:
        self._r(404, "text/plain", "Not found")

Handler.do_POST = _patched_do_post


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overhead camera card recognition test")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    capture = FrameCapture(args.camera, args.resolution)
    log_buffer.log(f"Camera {args.camera}, resolution {args.resolution}")

    print(f"  Testing capture...")
    frame = capture.capture()
    if frame is None:
        sys.exit("  ERROR: Could not capture. Check camera and ffmpeg.")
    print(f"  Capture OK: {frame.shape[1]}x{frame.shape[0]}")

    cal = Calibration()
    cal.load()

    monitor = ZoneMonitor(threshold=args.threshold)
    state = AppState(capture, cal, monitor)
    state.latest_frame = frame

    start_server(state)

    # Background capture loop
    def bg_capture():
        while not state.quit_flag:
            update_frame(state)
            time.sleep(2)
    Thread(target=bg_capture, daemon=True).start()

    time.sleep(1)
    subprocess.Popen(["open", f"http://localhost:{PORT}/live"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if cal.is_complete:
        print(f"  Calibration loaded — {len(cal.zones)} zones")
    else:
        print("  No calibration found")

    print_menu()

    try:
        while True:
            cmd = input("\n  Enter command: ").strip().lower()
            if cmd == "q":
                print("\n  Shutting down...")
                state.quit_flag = True
                break
            elif cmd == "c":
                do_calibrate(state)
            elif cmd == "t":
                do_test_recognition(state)
            elif cmd == "m":
                do_monitor(state)
            elif cmd == "r":
                frame = state.latest_frame
                if frame is not None:
                    print("  Clear table, then press Enter...")
                    input()
                    frame = state.latest_frame
                    if frame:
                        state.monitor.capture_baselines(frame, state.cal.zones)
                        print("  Baselines recaptured.")
            elif cmd == "s":
                do_snapshot(state)
            elif cmd == "":
                continue
            else:
                print(f"  Unknown: '{cmd}'")
                print_menu()
    except (KeyboardInterrupt, EOFError):
        print("\n  Interrupted.")

    print("  Done.")


if __name__ == "__main__":
    main()

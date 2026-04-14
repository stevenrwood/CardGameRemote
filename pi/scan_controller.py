#!/usr/bin/env python3
"""
Scanner box controller service (runs on Pi).

Captures from CM4 dual cameras with flash LED pulse, recognizes cards using
card_recognition.detector. No slot cropping yet — that requires calibration
against assembled hardware.

Usage:
    python scan_controller.py [--port 8080] [--cameras 0,1]

Endpoints:
    GET  /ping              — health check
    GET  /capture?camera=N  — flash + capture on camera N, return JSON (default: 0)
    GET  /capture/image?camera=N — same but return JPEG
    GET  /capture/both      — capture from all cameras in one flash, return combined JPEG
    POST /flash_test        — pulse flash LEDs for hardware verification
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

# GPIO — only import on Pi. Fail gracefully when running elsewhere.
try:
    from gpiozero import LED
    _GPIO_OK = True
except Exception as e:
    _GPIO_OK = False
    print(f"[WARN] gpiozero unavailable: {e}", file=sys.stderr)

# Camera — picamera2 is ARM-only, may fail on dev machines
try:
    from picamera2 import Picamera2
    _CAMERA_OK = True
except Exception as e:
    _CAMERA_OK = False
    print(f"[WARN] picamera2 unavailable: {e}", file=sys.stderr)

from card_recognition.detector import CardDetector

# ---- Config ----

FLASH_GPIO = 16  # BCM pin driving the flash MOSFET gate
FLASH_PULSE_MS = 50
REFERENCE_DIR = Path(__file__).parent / "card_recognition" / "reference"
CALIBRATION_FILE = Path(__file__).parent / "slot_calibration.json"
NUM_SLOTS = 7

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scan")


# ---- Hardware wrappers ----

class Flash:
    """Flash LED controller. No-op if GPIO unavailable."""
    def __init__(self, pin: int):
        self.pin = pin
        self.led = None
        if _GPIO_OK:
            try:
                self.led = LED(pin)
                self.led.off()
                log.info(f"Flash LED on GPIO {pin} ready")
            except Exception as e:
                log.warning(f"Flash LED init failed: {e}")

    def on(self):
        if self.led is not None:
            self.led.on()

    def off(self):
        if self.led is not None:
            self.led.off()

    def pulse(self, duration_ms: int = FLASH_PULSE_MS):
        if self.led is None:
            log.debug("Flash pulse skipped (no LED)")
            return
        self.led.on()
        time.sleep(duration_ms / 1000.0)
        self.led.off()


class Camera:
    """Pi camera wrapper. Captures a BGR numpy frame."""
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cam = None
        self._lock = Lock()
        if not _CAMERA_OK:
            raise RuntimeError("picamera2 not available — cannot initialize camera")
        self.cam = Picamera2(camera_num=camera_index)
        cfg = self.cam.create_still_configuration(
            main={"size": (2304, 1296), "format": "RGB888"},
        )
        self.cam.configure(cfg)
        # Runtime-tunable settings (defaults tuned for scanner box LEDs)
        self.exposure_us = 40000    # 40ms
        self.gain = 2.0             # analogue gain
        self._apply_controls()
        self.cam.start()
        time.sleep(0.5)  # warmup + AF settle
        log.info(f"Camera {camera_index} started at {cfg['main']['size']}")

    def _apply_controls(self):
        controls = {
            "AeEnable": False,
            "ExposureTime": int(self.exposure_us),
            "AnalogueGain": float(self.gain),
            "AwbEnable": True,
        }
        try:
            from libcamera import controls as libc
            controls["AfMode"] = libc.AfModeEnum.Continuous
        except Exception:
            pass
        self.cam.set_controls(controls)

    def set_exposure(self, exposure_us: int | None = None, gain: float | None = None):
        if exposure_us is not None:
            self.exposure_us = int(exposure_us)
        if gain is not None:
            self.gain = float(gain)
        self._apply_controls()

    def capture(self) -> np.ndarray:
        with self._lock:
            frame_rgb = self.cam.capture_array()
        # picamera2 returns RGB, detector uses BGR
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


# ---- App state ----

class AppState:
    def __init__(self, camera_indices: list[int]):
        self.flash = Flash(FLASH_GPIO)
        self.cameras: dict[int, Camera] = {}
        for idx in camera_indices:
            self.cameras[idx] = Camera(idx)
        self.detector = CardDetector(str(REFERENCE_DIR))
        self.last_frames: dict[int, np.ndarray] = {}
        self.flash_settle_ms = 50
        self.calibration = self._load_calibration()
        log.info(f"Scan controller ready with {len(self.cameras)} cameras; "
                 f"{len(self.calibration.get('slots', []))}/{NUM_SLOTS} slots calibrated")

    def _load_calibration(self) -> dict:
        if CALIBRATION_FILE.exists():
            try:
                return json.loads(CALIBRATION_FILE.read_text())
            except Exception as e:
                log.warning(f"Failed to load calibration: {e}")
        return {"slots": []}

    def save_calibration(self, data: dict):
        self.calibration = data
        CALIBRATION_FILE.write_text(json.dumps(data, indent=2))
        log.info(f"Calibration saved: {len(data.get('slots', []))} slots")

    def capture_with_flash(self, camera_idx: int) -> np.ndarray:
        """Turn flash on, capture from specified camera while lit, turn flash off."""
        cam = self.cameras[camera_idx]
        self.flash.on()
        try:
            time.sleep(self.flash_settle_ms / 1000.0)
            frame = cam.capture()
        finally:
            self.flash.off()
        self.last_frames[camera_idx] = frame
        return frame

    def capture_both(self) -> dict[int, np.ndarray]:
        """Capture from all cameras during one flash pulse."""
        self.flash.on()
        try:
            time.sleep(self.flash_settle_ms / 1000.0)
            frames = {idx: cam.capture() for idx, cam in self.cameras.items()}
        finally:
            self.flash.off()
        self.last_frames.update(frames)
        return frames

    def recognize(self, frame: np.ndarray) -> dict:
        t0 = time.time()
        result = self.detector.identify(frame)
        ms = (time.time() - t0) * 1000
        if result is None:
            return {"recognized": False, "ms": round(ms)}
        return {
            "recognized": True,
            "rank": result.rank,
            "suit": result.suit,
            "confidence": round(float(result.confidence), 3),
            "ms": round(ms),
        }


_state: AppState | None = None
app = Flask(__name__)


@app.get("/ping")
def ping():
    return jsonify({"ok": True, "camera": _CAMERA_OK, "gpio": _GPIO_OK})


def _pick_camera(default: int = 0) -> int:
    """Read ?camera=N from query string, default to first available camera."""
    assert _state is not None
    try:
        idx = int(request.args.get("camera", default))
    except ValueError:
        idx = default
    if idx not in _state.cameras:
        # Fall back to any available camera
        return next(iter(_state.cameras))
    return idx


@app.get("/capture")
def capture():
    assert _state is not None
    idx = _pick_camera()
    frame = _state.capture_with_flash(idx)
    result = _state.recognize(frame)
    result["size"] = f"{frame.shape[1]}x{frame.shape[0]}"
    result["camera"] = idx
    log.info(f"Capture cam{idx}: {result}")
    return jsonify(result)


@app.get("/capture/image")
def capture_image():
    assert _state is not None
    idx = _pick_camera()
    frame = _state.capture_with_flash(idx)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return "encode failed", 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.get("/capture/both")
def capture_both():
    """Capture from all cameras in a single flash and return both JPEGs side-by-side."""
    assert _state is not None
    frames = _state.capture_both()
    # Stack horizontally for a single combined image
    ordered = [frames[i] for i in sorted(frames.keys())]
    combined = np.hstack(ordered) if len(ordered) > 1 else ordered[0]
    ok, buf = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return "encode failed", 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.post("/camera_settings")
def camera_settings():
    """Update exposure and gain on all cameras at runtime.

    JSON body:
      {"exposure_ms": 40, "gain": 2.0, "flash_settle_ms": 80}

    All fields optional. Returns current settings.
    """
    assert _state is not None
    body = request.get_json(silent=True) or {}
    if "exposure_ms" in body:
        us = int(float(body["exposure_ms"]) * 1000)
        for cam in _state.cameras.values():
            cam.set_exposure(exposure_us=us)
    if "gain" in body:
        g = float(body["gain"])
        for cam in _state.cameras.values():
            cam.set_exposure(gain=g)
    if "flash_settle_ms" in body:
        _state.flash_settle_ms = int(body["flash_settle_ms"])

    any_cam = next(iter(_state.cameras.values()))
    return jsonify({
        "exposure_ms": any_cam.exposure_us / 1000.0,
        "gain": any_cam.gain,
        "flash_settle_ms": _state.flash_settle_ms,
    })


@app.get("/calibration")
def get_calibration():
    assert _state is not None
    return jsonify(_state.calibration)


@app.post("/calibration")
def save_calibration():
    assert _state is not None
    data = request.get_json(silent=True) or {}
    slots = data.get("slots", [])
    _state.save_calibration({"slots": slots})
    return jsonify({"ok": True, "count": len(slots)})


@app.get("/slots/image")
def slots_image():
    """Capture both cameras and draw calibrated slot rectangles for verification."""
    assert _state is not None
    frames = _state.capture_both()
    ordered_idx = sorted(frames.keys())
    ordered_frames = [frames[i].copy() for i in ordered_idx]

    # Draw slot rectangles
    for slot in _state.calibration.get("slots", []):
        cam_idx = slot.get("camera")
        if cam_idx not in ordered_idx:
            continue
        pos = ordered_idx.index(cam_idx)
        frame = ordered_frames[pos]
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label = f"#{slot['slot']}"
        cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    combined = np.hstack(ordered_frames) if len(ordered_frames) > 1 else ordered_frames[0]
    ok, buf = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return "encode failed", 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.get("/calibrate")
def calibrate_page():
    """Serve calibration web UI."""
    return CALIBRATE_HTML


CALIBRATE_HTML = """<!DOCTYPE html>
<html><head><title>Slot Calibration</title>
<style>
body{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:12px;margin:0}
h1{font-size:1.2em;margin:4px 0}
#status{font-size:1em;color:#4fc3f7;margin:8px 0;padding:8px;background:#0f3460;border-radius:6px}
button{padding:8px 14px;background:#0f3460;color:#fff;border:none;border-radius:6px;cursor:pointer;margin:3px;font-size:.95em}
button:hover{background:#1a5a9a}
.btn-green{background:#1b5e20}
.btn-red{background:#b71c1c}
.cam-box{margin:8px 0;border:1px solid #333;border-radius:6px;padding:6px}
.cam-title{font-weight:600;margin-bottom:4px}
canvas{border:1px solid #444;cursor:crosshair;display:block;max-width:100%;height:auto}
#slots-list{font-size:.85em;margin-top:8px}
.slot-row{padding:3px 6px;margin:2px 0;background:#16213e;border-radius:4px;display:flex;justify-content:space-between}
</style></head><body>
<h1>Scanner Slot Calibration</h1>
<div id="status">Loading...</div>
<div>
  <button onclick="refreshCaptures()">Refresh Images</button>
  <button class="btn-green" onclick="saveSlots()">Save Calibration</button>
  <button class="btn-red" onclick="clearSlots()">Clear All</button>
</div>
<p style="font-size:.9em;color:#aaa">
  Currently marking slot <b id="next-slot">1</b> of 7.
  Click top-left then bottom-right corner of the slot window in the appropriate camera view.
</p>
<div class="cam-box">
  <div class="cam-title">Camera 0 (slots 1-4 area)</div>
  <canvas id="cam0"></canvas>
</div>
<div class="cam-box">
  <div class="cam-title">Camera 1 (slots 4-7 area)</div>
  <canvas id="cam1"></canvas>
</div>
<div id="slots-list"></div>

<script>
var slots = [];
var pendingClick = null;  // {camera, x1, y1} after first click

function refreshCaptures() {
  [0, 1].forEach(function(camIdx) {
    var img = new Image();
    img.onload = function() {
      var canvas = document.getElementById('cam' + camIdx);
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.dataset.scale = Math.min(window.innerWidth - 60, img.naturalWidth) / img.naturalWidth;
      var ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      drawSlots(canvas, ctx, camIdx);
    };
    img.src = '/capture/image?camera=' + camIdx + '&t=' + Date.now();
  });
}

function drawSlots(canvas, ctx, camIdx) {
  slots.filter(function(s){return s.camera === camIdx}).forEach(function(s) {
    ctx.strokeStyle = '#4caf50';
    ctx.lineWidth = 3;
    ctx.strokeRect(s.x, s.y, s.w, s.h);
    ctx.fillStyle = '#4caf50';
    ctx.font = 'bold 32px sans-serif';
    ctx.fillText('#' + s.slot, s.x, s.y - 8);
  });
  if (pendingClick && pendingClick.camera === camIdx) {
    ctx.fillStyle = '#ff9800';
    ctx.beginPath();
    ctx.arc(pendingClick.x1, pendingClick.y1, 10, 0, Math.PI * 2);
    ctx.fill();
  }
}

function attachHandlers() {
  [0, 1].forEach(function(camIdx) {
    var canvas = document.getElementById('cam' + camIdx);
    canvas.onclick = function(ev) {
      var rect = canvas.getBoundingClientRect();
      var scale = canvas.width / rect.width;
      var x = Math.round((ev.clientX - rect.left) * scale);
      var y = Math.round((ev.clientY - rect.top) * scale);
      handleClick(camIdx, x, y);
    };
  });
}

function handleClick(camIdx, x, y) {
  if (!pendingClick) {
    pendingClick = {camera: camIdx, x1: x, y1: y};
    updateStatus();
    redraw();
    return;
  }
  if (pendingClick.camera !== camIdx) {
    // first click on one camera, second on different — cancel
    pendingClick = null;
    updateStatus('Second click must be on same camera. Resetting.');
    redraw();
    return;
  }
  // Second click — create slot rectangle
  var x1 = Math.min(pendingClick.x1, x);
  var y1 = Math.min(pendingClick.y1, y);
  var x2 = Math.max(pendingClick.x1, x);
  var y2 = Math.max(pendingClick.y1, y);
  var nextSlotNum = slots.length + 1;
  if (nextSlotNum > 7) {
    pendingClick = null;
    updateStatus('All 7 slots marked. Click "Save Calibration".');
    redraw();
    return;
  }
  slots.push({
    slot: nextSlotNum,
    camera: camIdx,
    x: x1, y: y1, w: x2 - x1, h: y2 - y1
  });
  pendingClick = null;
  updateStatus();
  redraw();
  updateSlotsList();
}

function updateStatus(msg) {
  var s = document.getElementById('status');
  if (msg) { s.textContent = msg; return; }
  var next = slots.length + 1;
  document.getElementById('next-slot').textContent = Math.min(next, 7);
  if (next > 7) {
    s.textContent = '✓ All 7 slots marked. Review and Save Calibration.';
  } else if (pendingClick) {
    s.textContent = 'Click BOTTOM-RIGHT corner of slot #' + next + ' on camera ' + pendingClick.camera;
  } else {
    s.textContent = 'Click TOP-LEFT corner of slot #' + next + ' on the camera that sees it best';
  }
}

function redraw() {
  [0, 1].forEach(function(camIdx) {
    var canvas = document.getElementById('cam' + camIdx);
    var ctx = canvas.getContext('2d');
    var img = new Image();
    img.onload = function() {
      ctx.drawImage(img, 0, 0);
      drawSlots(canvas, ctx, camIdx);
    };
    // Keep using last loaded image by reading current frame dimensions — we just redraw overlays
    // Simpler: reload
    img.src = '/capture/image?camera=' + camIdx;
  });
}

function updateSlotsList() {
  var list = document.getElementById('slots-list');
  list.innerHTML = slots.map(function(s) {
    return '<div class="slot-row"><span>Slot ' + s.slot + ' (cam ' + s.camera + ')</span>' +
      '<span>' + s.x + ',' + s.y + ' ' + s.w + 'x' + s.h + '</span></div>';
  }).join('');
}

function saveSlots() {
  if (slots.length === 0) { alert('No slots marked'); return; }
  fetch('/calibration', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({slots: slots})
  }).then(function(r){return r.json()}).then(function(d) {
    updateStatus('Saved ' + d.count + ' slots');
  });
}

function clearSlots() {
  if (!confirm('Clear all marked slots?')) return;
  slots = [];
  pendingClick = null;
  updateStatus();
  updateSlotsList();
  redraw();
}

// Load existing calibration on startup
fetch('/calibration').then(function(r){return r.json()}).then(function(d) {
  slots = d.slots || [];
  updateSlotsList();
  refreshCaptures();
  updateStatus();
  attachHandlers();
});
</script>
</body></html>"""


@app.post("/flash_test")
def flash_test():
    assert _state is not None
    body = request.get_json(silent=True) or {}
    ms = int(body.get("duration_ms", 100))
    count = int(body.get("count", 1))
    interval_ms = int(body.get("interval_ms", 500))
    for i in range(count):
        _state.flash.pulse(duration_ms=ms)
        if i < count - 1:
            time.sleep(interval_ms / 1000.0)
    return jsonify({"ok": True, "duration_ms": ms, "count": count, "interval_ms": interval_ms})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cameras", type=str, default="0,1",
                        help="Comma-separated camera indices (default: 0,1 for both CM4 CSI ports)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    indices = [int(x.strip()) for x in args.cameras.split(",") if x.strip()]

    global _state
    _state = AppState(camera_indices=indices)

    log.info(f"Listening on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

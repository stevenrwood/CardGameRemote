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
from threading import Lock, Thread, Timer

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

# GPIO — only import on Pi. Fail gracefully when running elsewhere.
try:
    from gpiozero import LED, PWMLED, Buzzer
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
from card_recognition.yolo_detector import YoloDetector

# ---- Config ----

FLASH_GPIO = 16  # BCM pin driving the flash MOSFET gate
FLASH_PULSE_MS = 50
REFERENCE_DIR = Path(__file__).parent / "card_recognition" / "reference"
YOLO_MODEL_PATH = Path(__file__).parent / "models" / "pi_card_detector.pt"
CALIBRATION_FILE = Path(__file__).parent / "slot_calibration.json"
NUM_SLOTS = 5
# Per-slot green status LED (BCM pins). Matches user's Pi 4B pinout diagram.
SLOT_LED_PINS = {1: 17, 2: 27, 3: 22, 4: 23, 5: 24}
BUZZER_GPIO = 26  # reserved for future audio feedback
# Confidence at/above which /test_slots treats a slot as a clean recognition.
SLOT_LED_GOOD_CONF = 0.50

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scan")


# ---- Hardware wrappers ----

class Flash:
    """Flash LED controller with PWM brightness. No-op if GPIO unavailable."""
    BRIGHTNESS_FILE = Path(__file__).parent / "flash_brightness.txt"

    def __init__(self, pin: int):
        self.pin = pin
        self.led = None
        self.held = False
        # Load persisted brightness or default to full.
        try:
            self.brightness = float(self.BRIGHTNESS_FILE.read_text().strip())
        except Exception:
            self.brightness = 1.0
        self.brightness = max(0.0, min(1.0, self.brightness))
        if _GPIO_OK:
            try:
                self.led = PWMLED(pin)
                self.led.value = 0.0
                log.info(f"Flash LED on GPIO {pin} ready (brightness={self.brightness:.2f})")
            except Exception as e:
                log.warning(f"Flash LED init failed: {e}")

    def set_brightness(self, value: float):
        """Set 0.0–1.0 target brightness. Persists across restarts. If the
        LED is currently on (or held), the change takes effect immediately."""
        self.brightness = max(0.0, min(1.0, float(value)))
        try:
            self.BRIGHTNESS_FILE.write_text(f"{self.brightness:.3f}\n")
        except Exception as e:
            log.warning(f"Could not persist brightness: {e}")
        if self.led is not None and (self.held or float(self.led.value) > 0):
            self.led.value = self.brightness

    def on(self):
        if self.led is not None:
            self.led.value = self.brightness

    def off(self):
        if self.held:
            return  # held on — ignore transient off requests
        if self.led is not None:
            self.led.value = 0.0

    def hold(self):
        """Force flash on and suppress .off() calls until release()."""
        self.held = True
        self.on()

    def release(self):
        """Clear held state and turn flash off."""
        self.held = False
        if self.led is not None:
            self.led.value = 0.0

    def blink_off(self, duration_ms: int):
        """While held, drop the LED for duration_ms then bring it back.

        Used as a visual cue to the user that a capture just completed.
        Runs asynchronously so the HTTP response isn't delayed.
        """
        if self.led is None or not self.held:
            return
        def run():
            self.led.value = 0.0
            time.sleep(duration_ms / 1000.0)
            if self.held:
                self.led.value = self.brightness
        Thread(target=run, daemon=True).start()

    def pulse(self, duration_ms: int = FLASH_PULSE_MS):
        if self.led is None:
            log.debug("Flash pulse skipped (no LED)")
            return
        self.led.value = self.brightness
        time.sleep(duration_ms / 1000.0)
        self.led.value = 0.0


class Buzz:
    """Piezo buzzer wrapper. No-op if GPIO unavailable."""
    def __init__(self, pin: int):
        self.pin = pin
        self.buzzer = None
        if _GPIO_OK:
            try:
                self.buzzer = Buzzer(pin)
                log.info(f"Buzzer on GPIO {pin} ready")
            except Exception as e:
                log.warning(f"Buzzer init failed on GPIO {pin}: {e}")

    def beep(self, n: int = 3, on_time: float = 0.12, off_time: float = 0.12):
        """Beep n times in the background — does not block the caller."""
        if self.buzzer is not None:
            self.buzzer.beep(on_time=on_time, off_time=off_time, n=n, background=True)


class SlotLed:
    """Per-slot green status LED.

    States: off (default), on (solid), blink (async, via gpiozero).
    gpiozero.LED.blink() runs the toggle in its own background thread;
    a subsequent .on()/.off() stops it cleanly.
    """
    def __init__(self, slot_num: int, pin: int):
        self.slot_num = slot_num
        self.pin = pin
        self.led = None
        if _GPIO_OK:
            try:
                self.led = LED(pin)
                self.led.off()
                log.info(f"Slot {slot_num} LED on GPIO {pin} ready")
            except Exception as e:
                log.warning(f"Slot {slot_num} LED init failed on GPIO {pin}: {e}")

    def on(self):
        if self.led is not None:
            self.led.on()

    def off(self):
        if self.led is not None:
            self.led.off()

    def blink(self, on_time: float = 0.3, off_time: float = 0.3):
        if self.led is not None:
            self.led.blink(on_time=on_time, off_time=off_time, background=True)


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

    def autofocus(self, timeout_s: float = 0.8) -> bool:
        """Trigger one-shot AF and wait for lock. Returns True if converged.

        Uses AfMode=Auto + AfTrigger=Start, polling AfState until Focused
        (or timeout). Always re-applies the full exposure/gain/AWB control
        set afterwards, because switching AF modes can nudge other pipeline
        state on some libcamera/imx708 versions.
        """
        try:
            from libcamera import controls as libc
        except Exception:
            return False
        converged = False
        try:
            with self._lock:
                self.cam.set_controls({
                    "AfMode": libc.AfModeEnum.Auto,
                    "AfTrigger": libc.AfTriggerEnum.Start,
                })
            deadline = time.time() + timeout_s
            focused = libc.AfStateEnum.Focused
            while time.time() < deadline:
                md = self.cam.capture_metadata()
                if md.get("AfState") == focused:
                    converged = True
                    break
                time.sleep(0.05)
        finally:
            # Restore exposure/gain/AWB + Continuous AF so subsequent frames
            # stay correctly exposed regardless of how AF ended.
            self._apply_controls()
        return converged


# ---- App state ----

class AppState:
    def __init__(self, camera_indices: list[int]):
        self.flash = Flash(FLASH_GPIO)
        self.flash_lock = Lock()
        self.slot_leds: dict[int, SlotLed] = {
            n: SlotLed(n, pin) for n, pin in SLOT_LED_PINS.items()
        }
        self.buzzer = Buzz(BUZZER_GPIO)
        # Timer to clear /test_slots feedback LEDs after a few seconds so
        # they don't sit on forever. Each new scan cancels the prior one.
        self._slot_led_off_timer: Timer | None = None
        self.cameras: dict[int, Camera] = {}
        for idx in camera_indices:
            self.cameras[idx] = Camera(idx)
        self.detector = CardDetector(str(REFERENCE_DIR))
        # Pi-trained YOLO model. Loaded lazily so that if ultralytics isn't
        # installed or the weights aren't deployed yet, the service still
        # boots and falls back to template matching for /slots.
        try:
            self.yolo = YoloDetector(YOLO_MODEL_PATH)
            if self.yolo.available:
                log.info(f"YOLO detector loaded from {YOLO_MODEL_PATH.name}")
            else:
                log.info(f"YOLO weights not found at {YOLO_MODEL_PATH}; using templates")
        except Exception as e:
            log.warning(f"YOLO detector unavailable: {e}")
            self.yolo = None
        self.last_frames: dict[int, np.ndarray] = {}
        self.flash_settle_ms = 50
        self.test_slot_crops = {}  # slot_num -> BGR crop from last /test_slots/data
        self.test_slot_stamp = 0
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

    # Minimum time the flash needs to be on before the sensor reads a
    # consistently-exposed frame. LED drivers and the rail take a couple of
    # hundred ms to fully stabilize; without this the no-AF path produced
    # visibly dimmer/cream-tinted captures.
    FLASH_WARMUP_MS = 300

    def capture_with_flash(self, camera_idx: int, focus: bool = False) -> np.ndarray:
        """Turn flash on, capture from specified camera while lit, turn flash off.

        If focus=True, trigger a one-shot AF lock (under flash) before the capture
        so the scene is illuminated during focus acquisition. AF already keeps
        the flash on ~800ms, so the AF path just needs the small settle.
        When focus is False, we enforce FLASH_WARMUP_MS (~300ms) before the
        capture so the LEDs have time to reach full brightness.
        """
        cam = self.cameras[camera_idx]
        with self.flash_lock:
            self.flash.on()
            try:
                if focus:
                    cam.autofocus(timeout_s=1.5)
                    time.sleep(self.flash_settle_ms / 1000.0)
                elif self.flash.held:
                    # LEDs already at full brightness from a prior hold()
                    pass
                else:
                    warmup = max(self.flash_settle_ms, self.FLASH_WARMUP_MS) / 1000.0
                    time.sleep(warmup)
                frame = cam.capture()
            finally:
                self.flash.off()
        self.last_frames[camera_idx] = frame
        return frame

    def capture_both(self) -> dict[int, np.ndarray]:
        """Capture from all cameras during one flash pulse."""
        with self.flash_lock:
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
    focus = request.args.get("focus") in ("1", "true", "yes")
    frame = _state.capture_with_flash(idx, focus=focus)
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


@app.get("/test_slots")
def test_slots_page():
    """Diagnostic page: captures every calibrated slot and shows the crops next to
    the detector's rank/suit guess and confidence. No game / deal context,
    just a straight capture-and-recognize for all calibrated slots.
    """
    return TEST_SLOTS_HTML


@app.get("/test_slots/data")
def test_slots_data():
    """JSON payload for /test_slots. Holds the flash for 4 seconds so the
    LEDs are at fully steady brightness, captures both cameras, crops
    every calibrated slot, runs recognition, and caches each crop for the
    per-slot image endpoint to serve without another capture.

    Slot LEDs blink while warming up + capturing. Each goes solid green
    once YOLO recognizes it above SLOT_LED_GOOD_CONF, otherwise stays
    blinking so the user can see which slot needs attention.
    """
    assert _state is not None
    # Start all slot LEDs blinking during warmup + capture.
    for led in _state.slot_leds.values():
        led.blink(on_time=0.25, off_time=0.25)
    any_blinking = False
    _state.flash.hold()
    try:
        time.sleep(4.0)
        frames = _state.capture_both()
    finally:
        _state.flash.release()
    results = []
    crops_cache = {}
    for slot in _state.calibration.get("slots", []):
        if slot["slot"] > NUM_SLOTS:
            continue
        cam_idx = slot.get("camera")
        entry = {
            "slot": slot["slot"],
            "camera": cam_idx,
            "recognized": False,
        }
        if cam_idx not in frames:
            entry["error"] = "camera not available"
            results.append(entry)
            continue
        frame = frames[cam_idx]
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            entry["error"] = "crop out of bounds"
            results.append(entry)
            continue
        crops_cache[slot["slot"]] = crop.copy()
        t0 = time.time()
        rank = suit = None
        conf = 0.0
        source = "none"
        if _state.yolo and _state.yolo.available:
            pred = _state.yolo.predict(crop)
            if pred is not None:
                rank, suit, conf = pred
                source = "yolo"
        if rank is None:
            if _state.detector.has_any_slot_templates():
                res = _state.detector.identify_slot(crop, slot_num=slot["slot"])
                fallback_source = "tmpl"
            else:
                res = _state.detector.identify(crop)
                fallback_source = "corner"
            if res is not None:
                rank, suit, conf = res.rank, res.suit, float(res.confidence)
                source = fallback_source
        ms = round((time.time() - t0) * 1000)
        code = f"{rank}{suit[0]}" if rank and suit else "-"
        log.info(
            f"[SCAN/test] slot{slot['slot']} cam{cam_idx} {source} "
            f"{code} conf={conf:.2f} {ms}ms"
        )
        entry["source"] = source
        if rank is not None:
            entry["recognized"] = True
            entry["rank"] = rank
            entry["suit"] = suit
            entry["confidence"] = round(conf, 3)
        # Update this slot's LED based on confidence.
        slot_led = _state.slot_leds.get(slot["slot"])
        good = rank is not None and conf >= SLOT_LED_GOOD_CONF
        if slot_led is not None:
            if good:
                slot_led.on()
            else:
                slot_led.blink(on_time=0.25, off_time=0.25)
        if not good:
            any_blinking = True
        results.append(entry)
    _state.test_slot_crops = crops_cache
    _state.test_slot_stamp = int(time.time())
    # Audible alert if any slot didn't come back at >= SLOT_LED_GOOD_CONF.
    if any_blinking:
        _state.buzzer.beep(n=3)
    # Clear LED feedback after 10s so the box goes quiet between scans.
    # Cancel any pending timer from a prior scan and start a fresh one.
    if _state._slot_led_off_timer is not None:
        _state._slot_led_off_timer.cancel()

    def _clear_slot_leds():
        for led in _state.slot_leds.values():
            led.off()
    _state._slot_led_off_timer = Timer(10.0, _clear_slot_leds)
    _state._slot_led_off_timer.daemon = True
    _state._slot_led_off_timer.start()
    return jsonify({"slots": results, "stamp": _state.test_slot_stamp})


@app.get("/test_slots/image/<int:slot_num>")
def test_slots_image(slot_num: int):
    """Serve the crop cached by the most recent /test_slots/data call.

    Does NOT re-capture; the bulk scan already paid the 4s LED warmup
    once for all slots. If no cache is present (diagnostic page loaded
    without scanning), 404.
    """
    assert _state is not None
    crops = getattr(_state, "test_slot_crops", None) or {}
    crop = crops.get(slot_num)
    if crop is None:
        return f"No cached crop for slot {slot_num}; click Scan All first", 404
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return "encode failed", 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


TEST_SLOTS_HTML = """<!DOCTYPE html>
<html><head><title>Slot Scanner Test</title>
<meta charset="UTF-8">
<style>
body{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:14px;margin:0}
h1{font-size:1.2em;margin:4px 0 10px}
button{padding:10px 16px;background:#0f3460;color:#fff;border:none;border-radius:6px;
  cursor:pointer;font-size:1em;margin-right:8px}
button:hover{background:#1a5a9a}
table{border-collapse:collapse;margin-top:12px;width:100%}
th,td{padding:8px 10px;border:1px solid #222;text-align:left;vertical-align:middle}
th{background:#0f3460}
td.conf{font-family:monospace}
td.good{color:#81c784}
td.med{color:#ffb74d}
td.bad{color:#ef5350}
td img{max-width:80px;max-height:140px;border:1px solid #444;border-radius:4px;background:#000;display:block}
</style></head><body>
<h1>Slot Scanner Test</h1>
<div><button onclick="run()">Scan All Slots</button><span id="status" style="color:#aaa;margin-left:12px">—</span></div>
<table id="tbl">
  <thead><tr><th>Slot</th><th>Camera</th><th>Crop</th><th>Recognized</th><th>Confidence</th></tr></thead>
  <tbody id="rows"></tbody>
</table>
<script>
var SUIT_SYM = {clubs:"\u2663",diamonds:"\u2666",hearts:"\u2665",spades:"\u2660"};

function confClass(c) {
  if (c == null) return "";
  if (c >= 0.70) return "good";
  if (c >= 0.30) return "med";
  return "bad";
}

function run() {
  var status = document.getElementById('status');
  var n = 4;
  status.textContent = 'LEDs on, warming up ' + n + 's…';
  var tick = setInterval(function() {
    n--;
    if (n > 0) status.textContent = 'LEDs on, warming up ' + n + 's…';
    else { clearInterval(tick); status.textContent = 'Capturing…'; }
  }, 1000);
  fetch('/test_slots/data').then(function(r){return r.json()}).then(function(d) {
    clearInterval(tick);
    var tbody = document.getElementById('rows');
    tbody.innerHTML = '';
    (d.slots || []).forEach(function(entry) {
      var tr = document.createElement('tr');
      var slotTd = document.createElement('td'); slotTd.textContent = entry.slot;
      var camTd = document.createElement('td'); camTd.textContent = entry.camera != null ? ('cam ' + entry.camera) : '—';
      var imgTd = document.createElement('td');
      var img = document.createElement('img');
      img.src = '/test_slots/image/' + entry.slot + '?t=' + Date.now();
      img.alt = 'slot ' + entry.slot;
      imgTd.appendChild(img);
      var cardTd = document.createElement('td');
      if (entry.recognized && entry.rank) {
        cardTd.textContent = entry.rank + (SUIT_SYM[entry.suit] || entry.suit || '');
      } else if (entry.error) {
        cardTd.textContent = 'error: ' + entry.error;
      } else {
        cardTd.textContent = '—';
      }
      var confTd = document.createElement('td');
      confTd.className = 'conf ' + confClass(entry.confidence);
      confTd.textContent = (entry.confidence != null)
        ? (Math.round(entry.confidence * 100) + '%') : '—';
      tr.appendChild(slotTd); tr.appendChild(camTd); tr.appendChild(imgTd);
      tr.appendChild(cardTd); tr.appendChild(confTd);
      tbody.appendChild(tr);
    });
    document.getElementById('status').textContent = 'Done';
  }).catch(function(e) {
    document.getElementById('status').textContent = 'Error: ' + e;
  });
}
run();
</script>
</body></html>"""


@app.post("/slots/<int:slot_num>/led")
def slot_led(slot_num: int):
    """Control the per-slot green LED. Body: {"state": "on"|"off"|"blink"}."""
    assert _state is not None
    body = request.get_json(silent=True) or {}
    state = str(body.get("state", "")).lower()
    if state not in ("on", "off", "blink"):
        return jsonify({"ok": False, "error": "state must be on/off/blink"}), 400
    led = _state.slot_leds.get(slot_num)
    if led is None:
        return jsonify({"ok": False, "error": f"no LED for slot {slot_num}"}), 404
    if state == "on":
        led.on()
    elif state == "off":
        led.off()
    else:
        led.blink(on_time=0.25, off_time=0.25)
    log.info(f"[LED] Slot {slot_num} → {state}")
    return jsonify({"ok": True, "slot": slot_num, "state": state})


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


@app.get("/slots")
def slots_state():
    """Capture both cameras, crop each slot, run recognition, return state.

    Query params:
      max_slot=N     — only scan slots 1..N (caller tells us how many
                       down cards the current game actually uses).
      slots=1,2,3    — explicit allow-list of slot numbers.
    Defaults to scanning every calibrated slot.
    """
    assert _state is not None
    allowed = None
    max_slot_raw = request.args.get("max_slot")
    if max_slot_raw:
        try:
            allowed = set(range(1, int(max_slot_raw) + 1))
        except ValueError:
            pass
    slots_raw = request.args.get("slots")
    if slots_raw:
        try:
            allowed = {int(x) for x in slots_raw.split(",") if x.strip()}
        except ValueError:
            pass

    slots_to_scan = [s for s in _state.calibration.get("slots", [])
                     if allowed is None or s["slot"] in allowed]
    if not slots_to_scan:
        return jsonify({"slots": [], "cards": ["-"] * 7})

    frames = _state.capture_both()
    results = []
    for slot in slots_to_scan:
        cam_idx = slot.get("camera")
        if cam_idx not in frames:
            results.append({"slot": slot["slot"], "error": "camera not available"})
            continue
        frame = frames[cam_idx]
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            results.append({"slot": slot["slot"], "error": "crop out of bounds"})
            continue
        t0 = time.time()
        # Recognition priority:
        #   1. YOLO (yolov8n trained on Pi scanner crops) if loaded
        #   2. Real-image slot templates if any trained
        #   3. Synthetic-corner matcher (last-resort fallback)
        rank = suit = None
        conf = 0.0
        source = "none"
        if _state.yolo and _state.yolo.available:
            pred = _state.yolo.predict(crop)
            if pred is not None:
                rank, suit, conf = pred
                source = "yolo"
        if rank is None:
            if _state.detector.has_any_slot_templates():
                res = _state.detector.identify_slot(crop, slot_num=slot["slot"])
                fallback_source = "tmpl"
            else:
                res = _state.detector.identify(crop)
                fallback_source = "corner"
            if res is not None:
                rank, suit, conf = res.rank, res.suit, float(res.confidence)
                source = fallback_source
        ms = round((time.time() - t0) * 1000)
        code = f"{rank}{suit[0]}" if rank and suit else "-"
        log.info(
            f"[SCAN] slot{slot['slot']} cam{cam_idx} {source} "
            f"{code} conf={conf:.2f} {ms}ms"
        )
        entry = {"slot": slot["slot"], "camera": cam_idx, "ms": ms, "source": source}
        if rank is None:
            entry["recognized"] = False
        else:
            entry["recognized"] = True
            entry["rank"] = rank
            entry["suit"] = suit
            entry["confidence"] = round(conf, 3)
        results.append(entry)
    log.info(f"Slots scan: {sum(1 for r in results if r.get('recognized'))}/{len(results)} recognized")

    # Compact slot-1..N summary: "Qc", "10h", "-" if unrecognized / not calibrated
    by_slot = {r["slot"]: r for r in results}
    cards = []
    for n in range(1, NUM_SLOTS + 1):
        r = by_slot.get(n)
        if r and r.get("recognized"):
            cards.append(f"{r['rank']}{r['suit'][0]}")
        else:
            cards.append("-")
    return jsonify({"slots": results, "cards": cards})


@app.get("/slots/<int:slot_num>/image")
def slot_image(slot_num: int):
    """Return JPEG of just the specified slot's cropped region."""
    assert _state is not None
    slot = next((s for s in _state.calibration.get("slots", []) if s["slot"] == slot_num), None)
    if slot is None:
        return f"Slot {slot_num} not calibrated", 404
    cam_idx = slot["camera"]
    if cam_idx not in _state.cameras:
        return "Camera not available", 500
    frame = _state.capture_with_flash(cam_idx)
    x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
    crop = frame[y:y + h, x:x + w]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return "encode failed", 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


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


RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["clubs", "diamonds", "hearts", "spades"]


def _slot_crop(slot_num: int, focus: bool = True):
    """Capture the given slot (via its calibrated camera) and return the BGR crop.

    Returns (crop_bgr, slot_meta) or (None, error_string).
    """
    assert _state is not None
    slot = next((s for s in _state.calibration.get("slots", []) if s["slot"] == slot_num), None)
    if slot is None:
        return None, f"Slot {slot_num} not calibrated"
    cam_idx = slot["camera"]
    if cam_idx not in _state.cameras:
        return None, f"Camera {cam_idx} not available"
    frame = _state.capture_with_flash(cam_idx, focus=focus)
    x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return None, "Crop out of bounds"
    return crop, slot


@app.post("/flash/hold")
def flash_hold():
    """Keep the flash LEDs on continuously. Captures during a hold skip
    the per-shot LED warmup, giving very short capture cycles and
    rock-steady brightness. Useful during a dealing round so every
    down-card capture has identical exposure without extra flicker.
    """
    assert _state is not None
    _state.flash.hold()
    return jsonify({"ok": True, "held": True})


@app.post("/flash/release")
def flash_release():
    assert _state is not None
    _state.flash.release()
    return jsonify({"ok": True, "held": False})


@app.post("/flash/brightness")
def flash_brightness():
    """Set PWM brightness (0.0–1.0). Persists across restarts."""
    assert _state is not None
    body = request.get_json(silent=True) or {}
    v = body.get("value")
    if v is None:
        return jsonify({"ok": False, "error": "missing value"}), 400
    _state.flash.set_brightness(float(v))
    return jsonify({"ok": True, "brightness": _state.flash.brightness})


@app.get("/flash/brightness")
def flash_brightness_get():
    assert _state is not None
    return jsonify({"brightness": _state.flash.brightness})


# Backward-compat aliases for the old /train/flash/* paths
app.add_url_rule("/train/flash/hold", view_func=flash_hold, methods=["POST"])
app.add_url_rule("/train/flash/release", view_func=flash_release, methods=["POST"])


@app.get("/train/status")
def train_status():
    """Return per-slot training status: which (slot, card) pairs are trained."""
    assert _state is not None
    per_slot = _state.detector.list_slot_templates()  # {slot: [(rank, suit), ...]}
    trained = {}
    for slot_num in range(1, NUM_SLOTS + 1):
        have = set(per_slot.get(slot_num, []))
        trained[slot_num] = {
            f"{rank}{suit[0]}": (rank, suit) in have
            for rank in RANKS for suit in SUITS
        }
    total = sum(sum(1 for v in s.values() if v) for s in trained.values())
    return jsonify({"slots": trained, "count": total, "expected": NUM_SLOTS * 52})


@app.post("/train/capture")
def train_capture():
    """Capture a specific slot and save its image as a per-slot template.

    JSON body: {"rank": "A", "suit": "hearts", "slot": 4, "focus": false}

    focus=false (default) skips autofocus entirely, giving a fast
    predictable flash (~100ms). The training UI sends focus=true only
    for the first capture of a new slot, and relies on lens position
    staying put across the slot's 52 captures.
    """
    assert _state is not None
    body = request.get_json(silent=True) or {}
    rank = str(body.get("rank", "")).upper()
    suit = str(body.get("suit", "")).lower()
    slot_num = int(body.get("slot", 4))
    focus = bool(body.get("focus", False))
    if rank not in RANKS or suit not in SUITS:
        return jsonify({"ok": False, "error": "bad rank/suit"}), 400
    if slot_num < 1 or slot_num > NUM_SLOTS:
        return jsonify({"ok": False, "error": f"slot must be 1..{NUM_SLOTS}"}), 400
    crop, meta = _slot_crop(slot_num, focus=focus)
    if crop is None:
        return jsonify({"ok": False, "error": meta}), 400
    _state.detector.save_slot_template(slot_num, rank, suit, crop)
    log.info(f"Trained template slot{slot_num}/{rank}{suit[0]} (focus={focus})")
    # Visual swap cue: dim the held flash for 500ms so the user knows the
    # capture finished. No-op if flash isn't currently held.
    _state.flash.blink_off(500)
    return jsonify({"ok": True, "slot": slot_num, "rank": rank, "suit": suit})


@app.delete("/train/reset/<int:slot_num>")
def train_reset_slot(slot_num: int):
    """Delete all templates for a single slot."""
    assert _state is not None
    if slot_num < 1 or slot_num > NUM_SLOTS:
        return jsonify({"ok": False, "error": f"slot must be 1..{NUM_SLOTS}"}), 400
    d = _state.detector.reference_dir / "slot_templates" / f"slot{slot_num}"
    removed = 0
    if d.exists():
        for f in list(d.iterdir()):
            if f.suffix.lower() in (".png", ".jpg"):
                f.unlink()
                removed += 1
    _state.detector.reload_slot_templates()
    log.info(f"Reset slot {slot_num}: removed {removed} templates")
    return jsonify({"ok": True, "removed": removed})


@app.delete("/train/capture/<int:slot_num>/<card>")
def train_delete(slot_num: int, card: str):
    assert _state is not None
    import re
    m = re.match(r"^(10|[2-9JQKA])([hdcs])$", card, re.IGNORECASE)
    if not m:
        return jsonify({"ok": False, "error": "bad card code"}), 400
    rank = m.group(1).upper()
    suit_map = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades"}
    suit = suit_map[m.group(2).lower()]
    path = _state.detector.slot_template_path(slot_num, rank, suit)
    if path.exists():
        path.unlink()
    _state.detector.reload_slot_templates()
    return jsonify({"ok": True})


@app.get("/train/template/<int:slot_num>/<card>")
def train_template_image(slot_num: int, card: str):
    """Return the stored template image for a (slot, card) pair."""
    assert _state is not None
    import re
    m = re.match(r"^(10|[2-9JQKA])([hdcs])$", card, re.IGNORECASE)
    if not m:
        return "bad card code", 400
    rank = m.group(1).upper()
    suit_map = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades"}
    suit = suit_map[m.group(2).lower()]
    path = _state.detector.slot_template_path(slot_num, rank, suit)
    if not path.exists():
        return "not trained", 404
    return send_file(str(path), mimetype="image/png")


@app.get("/train")
def train_page():
    return TRAIN_HTML


@app.get("/train/validate/<int:slot_num>")
def train_validate(slot_num: int):
    """Render a 4x13 grid of all 52 templates for a slot so the user can
    eyeball whether every capture is complete and correct."""
    if slot_num < 1 or slot_num > NUM_SLOTS:
        return f"slot must be 1..{NUM_SLOTS}", 400
    rows = []
    for suit in SUITS:
        cells = []
        for rank in RANKS:
            suit_letter = suit[0]
            code = f"{rank}{suit_letter}"
            cells.append((rank, suit, code))
        rows.append((suit, cells))
    red_suits = {"hearts", "diamonds"}

    html_rows = []
    for suit, cells in rows:
        row_color = "color:#ef9a9a" if suit in red_suits else "color:#e0e0e0"
        row_html = f'<tr><th style="{row_color};padding:4px 8px">{suit.capitalize()}</th>'
        for rank, _s, code in cells:
            img_src = f"/train/template/{slot_num}/{code}?t={int(time.time())}"
            row_html += (
                f'<td style="padding:4px;text-align:center;vertical-align:top">'
                f'<div style="font-size:.85em;color:#888">{rank}</div>'
                f'<img src="{img_src}" alt="{code}" '
                f'style="max-width:90px;max-height:180px;border:1px solid #333;background:#000;display:block;margin:0 auto"'
                f' onerror="this.style.visibility=\'hidden\';this.parentNode.innerHTML+=\'<div style=&quot;color:#b71c1c;font-size:.8em&quot;>missing</div>\'" />'
                f"</td>"
            )
        row_html += "</tr>"
        html_rows.append(row_html)

    header = '<tr><th></th>' + "".join(f'<th style="color:#aaa;font-weight:normal">{r}</th>' for r in RANKS) + '</tr>'
    body = header + "".join(html_rows)

    page = f"""<!DOCTYPE html>
<html><head><title>Slot {slot_num} — Validate Templates</title>
<style>
body{{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:12px;margin:0}}
h1{{font-size:1.2em;margin:4px 0}}
table{{border-collapse:collapse;margin-top:10px}}
th{{text-align:left}}
</style></head><body>
<h1>Slot {slot_num} — captured templates ({len(_state.detector.list_slot_templates().get(slot_num, []))}/52)</h1>
<div style="color:#aaa;font-size:.9em">4 suits × 13 ranks. Missing captures show "missing".</div>
<table>{body}</table>
</body></html>"""
    return page


TRAIN_HTML = """<!DOCTYPE html>
<html><head><title>Train Card Templates</title>
<style>
body{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:12px;margin:0}
h1{font-size:1.2em;margin:4px 0}
#status{font-size:1em;color:#4fc3f7;margin:8px 0;padding:8px;background:#0f3460;border-radius:6px}
button{padding:12px 20px;background:#0f3460;color:#fff;border:none;border-radius:6px;cursor:pointer;margin:3px;font-size:1em}
button:hover{background:#1a5a9a}
button:disabled{opacity:.4;cursor:not-allowed}
.btn-green{background:#1b5e20}
.btn-red{background:#b71c1c}
.now{padding:22px;background:#16213e;border-radius:10px;margin:10px 0;text-align:center}
.now .label{font-size:.95em;color:#aaa;margin-bottom:6px}
.now .card{font-size:3.5em;font-weight:700;line-height:1}
.now .card.red{color:#ef5350}
.now .card.black{color:#e0e0e0}
.now .slot-line{font-size:1.6em;margin-top:8px;color:#ffd54f}
.now .countdown{font-size:3em;font-weight:700;margin-top:10px;color:#ffb74d}
.now.flashing{background:#1b5e20}
.now.flashing .countdown{color:#e0e0e0}
.next{padding:10px;background:#0d1b2a;border-radius:8px;margin:8px 0;text-align:center;font-size:.95em;color:#aaa}
.next .line{margin:3px 0}
.matrix{margin-top:14px;overflow-x:auto}
table.matrix-table{border-collapse:collapse;font-size:.72em;min-width:100%}
table.matrix-table th,table.matrix-table td{padding:2px 4px;border:1px solid #222;text-align:center}
table.matrix-table th{background:#0f3460;color:#fff}
table.matrix-table th.slot-header{cursor:pointer;user-select:none}
table.matrix-table th.slot-header:hover{background:#b71c1c}
table.matrix-table td.trained{background:#1b5e20;cursor:pointer}
table.matrix-table td.trained:hover{background:#2e7d32}
table.matrix-table td.active{outline:2px solid #4fc3f7}
table.matrix-table td.marked{outline:2px dashed #ff5722;outline-offset:-2px}
table.matrix-table td.red{color:#ef9a9a}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;justify-content:center;z-index:100}
.modal.show{display:flex}
.modal-content{background:#16213e;padding:16px;border-radius:10px;max-width:90%;max-height:90%;text-align:center}
.modal-content img{max-width:300px;max-height:60vh;border:2px solid #444;border-radius:4px;background:#000}
.modal-content .title{font-size:1.2em;margin-bottom:8px}
.modal-content .buttons{margin-top:12px;display:flex;gap:8px;justify-content:center}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:center}
label{font-size:.9em}
input[type=number]{padding:8px;background:#0f3460;color:#fff;border:1px solid #333;border-radius:4px;width:4em}
</style></head><body>
<h1>Train Card Templates</h1>
<div id="status">Loading…</div>
<div class="controls">
  <label>Seconds per slot: <input id="delay" type="number" min="1" max="20" value="5"/></label>
  <button id="start" class="btn-green" onclick="start()">Start</button>
  <button id="pause" onclick="togglePause()" disabled>Pause</button>
  <button id="skip" onclick="skipCurrent()" disabled>Skip</button>
  <button id="stop" class="btn-red" onclick="stopAll()">Stop</button>
  <button id="reset" class="btn-red" onclick="resetAll()">Reset All</button>
</div>
<div class="now" id="now">
  <div class="label">Insert this card:</div>
  <div class="card" id="current-card">—</div>
  <div class="slot-line">into <b id="current-slot">slot —</b></div>
  <div class="countdown" id="countdown">—</div>
</div>
<div class="next">
  <div class="line">Next card in this slot: <b id="next-slot">—</b></div>
  <div class="line">After this slot: <b id="next-card">—</b></div>
</div>
<div class="matrix">
  <table class="matrix-table" id="matrix"></table>
</div>
<div class="modal" id="slot-menu" onclick="closeSlotMenu(event)">
  <div class="modal-content" onclick="event.stopPropagation()">
    <div class="title" id="slot-menu-title">Slot —</div>
    <div class="buttons" style="flex-direction:column;gap:6px">
      <button class="btn-green" onclick="validateSlot()">Validate (open in new tab)</button>
      <button onclick="retrainMarkedInSlot()">Retrain marked cards</button>
      <button class="btn-red" onclick="resetSlot()">Reset slot</button>
      <button onclick="closeSlotMenu()">Cancel</button>
    </div>
  </div>
</div>
<div class="modal" id="modal" onclick="closeModal(event)">
  <div class="modal-content" onclick="event.stopPropagation()">
    <div class="title" id="modal-title">—</div>
    <img id="modal-img" src="" alt="template"/>
    <div class="buttons">
      <button id="mark-btn" class="btn-red" onclick="toggleMarkFromModal()">Mark for retrain</button>
      <button onclick="closeModal()">Close</button>
    </div>
  </div>
</div>

<script>
var RANKS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"];
var SUITS = ["clubs","diamonds","hearts","spades"];
var SUIT_SYM = {clubs:"♣",diamonds:"♦",hearts:"♥",spades:"♠"};
var SLOTS = [1,2,3,4,5];  // matches NUM_SLOTS on server

// ORDER is slot-major: finish all 52 cards in slot 1, auto-pause, then
// slot 2's 52, pause, ... through slot 7. Within a slot the card order is
// 2C,2D,2H,2S,3C,... (matches YOLO training).
var ORDER = [];
SLOTS.forEach(function(slot){
  RANKS.forEach(function(r){
    SUITS.forEach(function(s){
      ORDER.push({rank:r, suit:s, slot:slot});
    });
  });
});

var trainedMap = {};     // "<slot>/<rank><suitLetter>" -> true
var idx = 0;
var running = false;
var paused = false;
var ticker = null;
var remainingMs = 0;
var flashing = false;
var firstStep = true;
var needsFocus = true;   // AF only on first capture of a run / slot
var FIRST_DELAY_S = 30;

function cardCode(r, s) { return r + s[0]; }
function key(step) { return step.slot + "/" + cardCode(step.rank, step.suit); }
function isRed(s) { return s === "hearts" || s === "diamonds"; }
function isTrained(step) { return !!trainedMap[key(step)]; }

function refreshStatus() {
  return fetch('/train/status').then(function(r){return r.json()}).then(function(d) {
    trainedMap = {};
    Object.keys(d.slots).forEach(function(slot) {
      var cards = d.slots[slot];
      Object.keys(cards).forEach(function(cc) {
        if (cards[cc]) trainedMap[slot + "/" + cc] = true;
      });
    });
    document.getElementById('status').textContent =
      d.count + ' / ' + d.expected + ' (slot,card) pairs trained';
    // Before a run has started, point idx at the first untrained step so the
    // "Insert this card" preview matches where Start will actually begin.
    if (!running) {
      var first = ORDER.findIndex(function(step){ return !isTrained(step); });
      idx = (first >= 0) ? first : 0;
    }
    renderMatrix();
    updateDisplay();
  });
}

function renderMatrix() {
  var t = document.getElementById('matrix');
  var html = '<tr><th></th>';
  SLOTS.forEach(function(s){
    html += '<th class="slot-header" data-reset-slot="' + s + '" title="Click to reset slot ' + s + '">Slot ' + s + '</th>';
  });
  html += '</tr>';
  RANKS.forEach(function(r) {
    SUITS.forEach(function(s) {
      html += '<tr><th' + (isRed(s)?' class="red"':'') + '>' + r + SUIT_SYM[s] + '</th>';
      SLOTS.forEach(function(slot) {
        var step = {rank:r, suit:s, slot:slot};
        var isActive = (ORDER[idx] &&
          ORDER[idx].rank===r && ORDER[idx].suit===s && ORDER[idx].slot===slot);
        var cls = [];
        var attrs = '';
        if (isTrained(step)) {
          cls.push('trained');
          attrs = ' data-slot="' + slot + '" data-card="' + cardCode(r, s) + '"';
        }
        if (isActive) cls.push('active');
        if (isRed(s)) cls.push('red');
        if (isMarked(slot, cardCode(r, s))) cls.push('marked');
        html += '<td class="' + cls.join(' ') + '"' + attrs + '>' + (isTrained(step)?'✓':'') + '</td>';
      });
      html += '</tr>';
    });
  });
  t.innerHTML = html;
}

document.getElementById('matrix').addEventListener('click', function(ev) {
  var header = ev.target.closest('th[data-reset-slot]');
  if (header) {
    openSlotMenu(parseInt(header.dataset.resetSlot, 10));
    return;
  }
  var cell = ev.target.closest('td[data-slot]');
  if (!cell) return;
  showTemplate(parseInt(cell.dataset.slot, 10), cell.dataset.card);
});

function openSlotMenu(slot) {
  document.getElementById('slot-menu-title').textContent = 'Slot ' + slot;
  document.getElementById('slot-menu').dataset.slot = slot;
  document.getElementById('slot-menu').classList.add('show');
}
function closeSlotMenu(ev) {
  if (ev && ev.target !== ev.currentTarget) return;
  document.getElementById('slot-menu').classList.remove('show');
}
function validateSlot() {
  var slot = document.getElementById('slot-menu').dataset.slot;
  window.open('/train/validate/' + slot, '_blank');
  closeSlotMenu();
}
function resetSlot() {
  var slot = document.getElementById('slot-menu').dataset.slot;
  closeSlotMenu();
  if (!confirm('Delete all trained templates for slot ' + slot + '?')) return;
  fetch('/train/reset/' + slot, {method:'DELETE'})
    .then(function(r){return r.json()}).then(refreshStatus);
}

var modalSlot = null;
var modalCard = null;
function showTemplate(slot, card) {
  modalSlot = slot;
  modalCard = card;
  document.getElementById('modal-title').textContent =
    'Slot ' + slot + ' — ' + card.toUpperCase();
  document.getElementById('modal-img').src =
    '/train/template/' + slot + '/' + card + '?t=' + Date.now();
  document.getElementById('modal').classList.add('show');
  updateMarkButton();
}
function closeModal(ev) {
  if (ev && ev.target !== ev.currentTarget) return;
  document.getElementById('modal').classList.remove('show');
  modalSlot = null; modalCard = null;
}
// Client-side mark set: "<slot>/<card>" entries flagged by the user
// for a future retrain pass (via the slot menu's "Retrain marked cards").
var marksForRetrain = {};

function markKey(slot, card) { return slot + "/" + card; }
function isMarked(slot, card) { return !!marksForRetrain[markKey(slot, card)]; }
function setMarked(slot, card, on) {
  var k = markKey(slot, card);
  if (on) marksForRetrain[k] = true;
  else delete marksForRetrain[k];
}

function updateMarkButton() {
  var btn = document.getElementById('mark-btn');
  if (!btn || modalSlot == null || !modalCard) return;
  var marked = isMarked(modalSlot, modalCard);
  btn.textContent = marked ? 'Unmark' : 'Mark for retrain';
  btn.classList.toggle('btn-red', !marked);
  btn.classList.toggle('btn-green', marked);
}

function toggleMarkFromModal() {
  if (modalSlot == null || !modalCard) return;
  setMarked(modalSlot, modalCard, !isMarked(modalSlot, modalCard));
  renderMatrix();
  // Close the preview modal so the next click can flow straight into
  // marking another card.
  closeModal();
}

// ---- Slot-scoped retrain runner ----
// Mirrors the main Start/Pause wizard but only walks the cards marked
// for the selected slot, in rank/suit order. Holds the flash on for the
// whole run and releases when finished.

var retrainRunning = false;
var retrainPaused = false;
var retrainTicker = null;
var retrainRemainingMs = 0;
var retrainFlashing = false;
var retrainSlot = null;
var retrainQueue = [];
var retrainIdx = 0;
var retrainFirstStep = true;

function retrainMarkedInSlot() {
  var slot = parseInt(document.getElementById('slot-menu').dataset.slot, 10);
  closeSlotMenu();
  // Build the ordered list of marked cards for this slot.
  var picks = [];
  RANKS.forEach(function(r) {
    SUITS.forEach(function(s) {
      if (isMarked(slot, cardCode(r, s))) {
        picks.push({rank: r, suit: s, slot: slot});
      }
    });
  });
  if (!picks.length) {
    alert('No cards marked for slot ' + slot + '. Open a trained cell and click "Mark for retrain" first.');
    return;
  }
  if (!confirm('Retrain ' + picks.length + ' card(s) in slot ' + slot + '? LEDs will stay on until done.')) return;
  if (running) {
    alert('Main training wizard is running. Stop it first.');
    return;
  }
  retrainSlot = slot;
  retrainQueue = picks;
  retrainIdx = 0;
  retrainRunning = true;
  retrainPaused = false;
  retrainFirstStep = true;
  fetch('/flash/hold', {method:'POST'});
  updateRetrainDisplay();
  beginRetrainCountdown();
}

function stopRetrain() {
  retrainRunning = false;
  retrainPaused = false;
  if (retrainTicker) { clearInterval(retrainTicker); retrainTicker = null; }
  setCountdown('—');
  setFlashing(false);
  fetch('/flash/release', {method:'POST'});
}

function updateRetrainDisplay() {
  var step = retrainQueue[retrainIdx];
  if (!step) return;
  var cur = document.getElementById('current-card');
  cur.textContent = step.rank + ' ' + SUIT_SYM[step.suit];
  cur.className = 'card ' + (isRed(step.suit) ? 'red' : 'black');
  document.getElementById('current-slot').textContent = 'slot ' + step.slot + ' (retrain)';
  var nxt = retrainQueue[retrainIdx + 1];
  var nextSlotEl = document.getElementById('next-slot');
  nextSlotEl.textContent = nxt ? (describeStep(nxt) + ' (slot ' + step.slot + ')')
                               : '— (retrain complete)';
  document.getElementById('next-card').textContent =
    retrainQueue.length + ' marked card(s); ' + (retrainIdx + 1) + ' of ' + retrainQueue.length;
  renderMatrix();
}

function beginRetrainCountdown() {
  if (retrainIdx >= retrainQueue.length) {
    stopRetrain();
    setCountdown('✓ Done');
    // Clear marks for the cards we just retrained.
    retrainQueue.forEach(function(step) {
      setMarked(step.slot, cardCode(step.rank, step.suit), false);
    });
    renderMatrix();
    return;
  }
  var secs;
  if (retrainFirstStep) {
    secs = FIRST_DELAY_S;
    retrainFirstStep = false;
  } else {
    secs = Math.max(1, parseInt(document.getElementById('delay').value, 10) || 5);
  }
  retrainRemainingMs = secs * 1000;
  setCountdown(secs.toString());
  if (retrainTicker) clearInterval(retrainTicker);
  retrainTicker = setInterval(function() {
    if (retrainPaused || retrainFlashing) return;
    retrainRemainingMs -= 100;
    if (retrainRemainingMs <= 0) {
      clearInterval(retrainTicker); retrainTicker = null;
      fireRetrainCapture();
      return;
    }
    setCountdown(Math.ceil(retrainRemainingMs / 1000).toString());
  }, 100);
}

function fireRetrainCapture() {
  var step = retrainQueue[retrainIdx];
  retrainFlashing = true;
  setFlashing(true);
  setCountdown('📸');
  // focus=false — LEDs already held and warm
  fetch('/train/capture', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({rank: step.rank, suit: step.suit, slot: step.slot, focus: false})
  }).then(function(r){return r.json()}).then(function(d) {
    retrainFlashing = false;
    setFlashing(false);
    if (!d.ok) {
      setCountdown('✗');
      alert('Retrain capture failed: ' + d.error);
      stopRetrain();
      return;
    }
    setMarked(step.slot, cardCode(step.rank, step.suit), false);
    refreshStatus().then(function() {
      retrainIdx++;
      if (retrainIdx >= retrainQueue.length) {
        stopRetrain();
        setCountdown('✓ Done');
        renderMatrix();
        return;
      }
      updateRetrainDisplay();
      beginRetrainCountdown();
    });
  }).catch(function(e) {
    retrainFlashing = false;
    setFlashing(false);
    alert('Network error: ' + e);
    stopRetrain();
  });
}

function describeStep(step) {
  return step.rank + SUIT_SYM[step.suit];
}

function updateDisplay() {
  var step = ORDER[idx];
  if (!step) return;
  var cur = document.getElementById('current-card');
  cur.textContent = step.rank + ' ' + SUIT_SYM[step.suit];
  cur.className = 'card ' + (isRed(step.suit)?'red':'black');
  document.getElementById('current-slot').textContent = 'slot ' + step.slot;
  // next card (in same slot) — slot-major ordering
  var nxt = ORDER[idx+1];
  var nextCardInSlotEl = document.getElementById('next-slot');
  if (nxt && nxt.slot === step.slot) {
    nextCardInSlotEl.textContent = describeStep(nxt) + ' (slot ' + nxt.slot + ')';
  } else {
    nextCardInSlotEl.textContent = '— (last card in slot ' + step.slot + ')';
  }
  // next slot = first step whose slot differs from current
  var nextSlot = null;
  for (var i = idx+1; i < ORDER.length; i++) {
    if (ORDER[i].slot !== step.slot) { nextSlot = ORDER[i]; break; }
  }
  var ncEl = document.getElementById('next-card');
  ncEl.textContent = nextSlot ? ('slot ' + nextSlot.slot + ' starting with ' + describeStep(nextSlot))
                              : '— (training complete)';
  renderMatrix();
}

function setCountdown(text) { document.getElementById('countdown').textContent = text; }
function setFlashing(on) {
  flashing = on;
  document.getElementById('now').className = 'now' + (on?' flashing':'');
}

function start() {
  if (running) return;
  running = true;
  paused = false;
  document.getElementById('start').disabled = true;
  document.getElementById('pause').disabled = false;
  document.getElementById('skip').disabled = false;
  // jump to first untrained step
  var first = ORDER.findIndex(function(step){ return !isTrained(step); });
  if (first >= 0) idx = first;
  firstStep = true;
  needsFocus = true;
  // Keep flash on continuously for the whole session — consistent
  // brightness and no per-capture LED warmup.
  fetch('/train/flash/hold', {method:'POST'});
  updateDisplay();
  beginCountdown();
}

function stop() {
  running = false;
  paused = false;
  if (ticker) { clearInterval(ticker); ticker = null; }
  document.getElementById('start').disabled = false;
  document.getElementById('pause').disabled = true;
  document.getElementById('pause').textContent = 'Pause';
  document.getElementById('skip').disabled = true;
  setCountdown('—');
  setFlashing(false);
  fetch('/train/flash/release', {method:'POST'});
}

function togglePause() {
  if (!running) return;
  paused = !paused;
  document.getElementById('pause').textContent = paused ? 'Resume' : 'Pause';
  if (paused) {
    fetch('/train/flash/release', {method:'POST'});
  } else {
    fetch('/train/flash/hold', {method:'POST'});
    // If we were paused at a slot boundary (no active ticker), kick off the
    // next slot's countdown on resume.
    if (!ticker) beginCountdown();
  }
}

function skipCurrent() {
  if (!running) return;
  if (ticker) { clearInterval(ticker); ticker = null; }
  advance();
}

function beginCountdown() {
  if (idx >= ORDER.length) { stop(); setCountdown('✓ Done'); return; }
  var secs;
  if (firstStep) {
    secs = FIRST_DELAY_S;
    firstStep = false;
  } else {
    secs = Math.max(1, parseInt(document.getElementById('delay').value, 10) || 5);
  }
  remainingMs = secs * 1000;
  setCountdown(secs.toString());
  if (ticker) clearInterval(ticker);
  ticker = setInterval(function() {
    if (paused || flashing) return;
    remainingMs -= 100;
    if (remainingMs <= 0) {
      clearInterval(ticker); ticker = null;
      fireCapture();
      return;
    }
    setCountdown(Math.ceil(remainingMs / 1000).toString());
  }, 100);
}

function fireCapture() {
  var step = ORDER[idx];
  setFlashing(true);
  setCountdown('📸');
  var useFocus = needsFocus;
  fetch('/train/capture', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({rank: step.rank, suit: step.suit, slot: step.slot, focus: useFocus})
  }).then(function(r){return r.json()}).then(function(d) {
    setFlashing(false);
    if (!d.ok) {
      setCountdown('✗');
      alert('Capture failed: ' + d.error);
      stop();
      return;
    }
    needsFocus = false;   // subsequent captures in this slot skip AF
    refreshStatus().then(function() { advance(); });
  }).catch(function(e) {
    setFlashing(false);
    alert('Network error: ' + e);
    stop();
  });
}

function advance() {
  var prev = ORDER[idx];
  idx++;
  if (idx >= ORDER.length) { stop(); setCountdown('✓ Done'); updateDisplay(); return; }
  var cur = ORDER[idx];
  updateDisplay();
  // Auto-pause at slot boundaries (every 52 captures)
  if (prev && cur && prev.slot !== cur.slot) {
    paused = true;
    needsFocus = true;    // re-AF for the new slot's first capture
    document.getElementById('pause').textContent = 'Resume';
    setCountdown('⏸ slot ' + prev.slot + ' done — press Resume for slot ' + cur.slot);
    fetch('/train/flash/release', {method:'POST'});
    return;
  }
  beginCountdown();
}

function stopAll() {
  // Halt whichever flow is running (main wizard or slot-scoped retrain),
  // release the flash, and send the UI back to its pre-Start resting state.
  if (running) stop();
  if (retrainRunning) stopRetrain();
  // Reset the preview pointer to the first untrained step — matches what
  // refreshStatus does when nothing is running.
  var first = ORDER.findIndex(function(step){ return !isTrained(step); });
  idx = (first >= 0) ? first : 0;
  firstStep = true;
  needsFocus = true;
  retrainFirstStep = true;
  retrainQueue = [];
  retrainIdx = 0;
  paused = false;
  retrainPaused = false;
  setCountdown('—');
  setFlashing(false);
  updateDisplay();
  renderMatrix();
}

function resetAll() {
  if (!confirm('Delete ALL trained templates across all slots?')) return;
  var deletes = [];
  Object.keys(trainedMap).forEach(function(k) {
    var parts = k.split('/');
    deletes.push(fetch('/train/capture/' + parts[0] + '/' + parts[1], {method:'DELETE'}));
  });
  Promise.all(deletes).then(refreshStatus);
}

refreshStatus();
</script>
</body></html>"""


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
  <button onclick="undoLastSlot()" title="Undo last rectangle (⌥Z)">Undo</button>
  <button class="btn-red" onclick="clearSlots()">Clear All</button>
</div>
<p style="font-size:.9em;color:#aaa">
  Currently marking slot <b id="next-slot">1</b> of 5.
  Press on the top-left corner and drag to the bottom-right of the slot window, then release.
</p>
<div class="cam-box">
  <div class="cam-title">Camera 1 (slots 1-3 area)</div>
  <canvas id="cam1"></canvas>
</div>
<div class="cam-box">
  <div class="cam-title">Camera 0 (slots 3-5 area)</div>
  <canvas id="cam0"></canvas>
</div>
<div id="slots-list"></div>

<script>
var slots = [];
var cachedImages = {};     // camIdx -> HTMLImageElement (last loaded)
var drag = null;           // {camera, x1, y1, x2, y2} while pointer is down

function refreshCaptures() {
  // Hold the LEDs on for a full 4 seconds so they're at steady brightness,
  // then grab a fresh image from each camera, then release.
  fetch('/flash/hold', {method:'POST'}).then(function() {
    setTimeout(function() {
      var remaining = 2;
      [0, 1].forEach(function(camIdx) {
        var img = new Image();
        img.onload = function() {
          cachedImages[camIdx] = img;
          var canvas = document.getElementById('cam' + camIdx);
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          canvas.dataset.scale = Math.min(window.innerWidth - 60, img.naturalWidth) / img.naturalWidth;
          redrawCam(camIdx);
          remaining--;
          if (remaining === 0) {
            fetch('/flash/release', {method:'POST'});
          }
        };
        img.onerror = function() {
          remaining--;
          if (remaining === 0) {
            fetch('/flash/release', {method:'POST'});
          }
        };
        img.src = '/capture/image?camera=' + camIdx + '&focus=1&t=' + Date.now();
      });
    }, 4000);
  });
}

function drawSlots(ctx, camIdx) {
  slots.filter(function(s){return s.camera === camIdx}).forEach(function(s) {
    ctx.strokeStyle = '#4caf50';
    ctx.lineWidth = 3;
    ctx.strokeRect(s.x, s.y, s.w, s.h);
    ctx.fillStyle = '#4caf50';
    ctx.font = 'bold 32px sans-serif';
    ctx.fillText('#' + s.slot, s.x, s.y - 8);
  });
  if (drag && drag.camera === camIdx) {
    var x = Math.min(drag.x1, drag.x2);
    var y = Math.min(drag.y1, drag.y2);
    var w = Math.abs(drag.x2 - drag.x1);
    var h = Math.abs(drag.y2 - drag.y1);
    ctx.strokeStyle = '#ff9800';
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}

function redrawCam(camIdx) {
  var canvas = document.getElementById('cam' + camIdx);
  var ctx = canvas.getContext('2d');
  var img = cachedImages[camIdx];
  if (img) ctx.drawImage(img, 0, 0);
  drawSlots(ctx, camIdx);
}

function canvasCoords(canvas, ev) {
  var rect = canvas.getBoundingClientRect();
  var scale = canvas.width / rect.width;
  return {
    x: Math.round((ev.clientX - rect.left) * scale),
    y: Math.round((ev.clientY - rect.top) * scale)
  };
}

function attachHandlers() {
  [0, 1].forEach(function(camIdx) {
    var canvas = document.getElementById('cam' + camIdx);
    canvas.style.touchAction = 'none';  // prevent scroll/zoom on drag
    canvas.addEventListener('pointerdown', function(ev) {
      if (slots.length >= NUM_SLOTS) return;
      var p = canvasCoords(canvas, ev);
      drag = {camera: camIdx, x1: p.x, y1: p.y, x2: p.x, y2: p.y};
      canvas.setPointerCapture(ev.pointerId);
      ev.preventDefault();
      updateStatus();
      redrawCam(camIdx);
    });
    canvas.addEventListener('pointermove', function(ev) {
      if (!drag || drag.camera !== camIdx) return;
      var p = canvasCoords(canvas, ev);
      drag.x2 = p.x;
      drag.y2 = p.y;
      redrawCam(camIdx);
    });
    function finish(ev) {
      if (!drag || drag.camera !== camIdx) return;
      var p = canvasCoords(canvas, ev);
      drag.x2 = p.x;
      drag.y2 = p.y;
      var x1 = Math.min(drag.x1, drag.x2);
      var y1 = Math.min(drag.y1, drag.y2);
      var x2 = Math.max(drag.x1, drag.x2);
      var y2 = Math.max(drag.y1, drag.y2);
      var w = x2 - x1, h = y2 - y1;
      drag = null;
      if (w < 5 || h < 5) {
        // accidental tap — ignore
        redrawCam(camIdx);
        updateStatus();
        return;
      }
      slots.push({
        slot: slots.length + 1,
        camera: camIdx,
        x: x1, y: y1, w: w, h: h
      });
      redrawCam(camIdx);
      updateStatus();
      updateSlotsList();
    }
    canvas.addEventListener('pointerup', finish);
    canvas.addEventListener('pointercancel', function() {
      drag = null;
      redrawCam(camIdx);
      updateStatus();
    });
  });
}

var NUM_SLOTS = 5;  // matches NUM_SLOTS on server

function updateStatus(msg) {
  var s = document.getElementById('status');
  if (msg) { s.textContent = msg; return; }
  var next = slots.length + 1;
  document.getElementById('next-slot').textContent = Math.min(next, NUM_SLOTS);
  if (next > NUM_SLOTS) {
    s.textContent = '✓ All ' + NUM_SLOTS + ' slots marked. Review and Save Calibration.';
  } else if (drag) {
    s.textContent = 'Drag to bottom-right of slot #' + next + ' and release…';
  } else {
    s.textContent = 'Press on TOP-LEFT of slot #' + next + ' and drag to BOTTOM-RIGHT';
  }
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
  drag = null;
  updateStatus();
  updateSlotsList();
  [0, 1].forEach(redrawCam);
}

function undoLastSlot() {
  if (drag) {
    // Cancel an in-progress drag before touching committed slots.
    var camIdx = drag.camera;
    drag = null;
    updateStatus();
    redrawCam(camIdx);
    return;
  }
  if (!slots.length) return;
  var last = slots.pop();
  // Renumber the remaining slots is unnecessary — they were added
  // in order and numbered sequentially, so popping the last one
  // keeps numbers contiguous.
  updateStatus();
  updateSlotsList();
  [0, 1].forEach(redrawCam);
}

// Option+Z (Alt+Z) anywhere on the page undoes the last rectangle.
window.addEventListener('keydown', function(ev) {
  if (ev.altKey && (ev.key === 'z' || ev.key === 'Z' ||
                    ev.code === 'KeyZ' || ev.code === 'Ω')) {
    ev.preventDefault();
    undoLastSlot();
  }
});

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

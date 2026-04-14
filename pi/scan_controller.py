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
        # Lock exposure for consistent flash-lit captures; keep AWB enabled
        # so colors look right. AfMode=Continuous keeps cards in focus.
        controls = {
            "AeEnable": False,       # disable auto-exposure
            "ExposureTime": 20000,   # 20ms — long enough for LED flash to fully illuminate
            "AnalogueGain": 1.0,     # minimum gain to reduce noise
            "AwbEnable": True,       # auto white balance
        }
        # Camera v3 has autofocus; v2 and HQ don't. Ignore control errors.
        try:
            from libcamera import controls as libc
            controls["AfMode"] = libc.AfModeEnum.Continuous
        except Exception:
            pass
        self.cam.set_controls(controls)
        self.cam.start()
        time.sleep(0.5)  # warmup + AF settle
        log.info(f"Camera {camera_index} started at {cfg['main']['size']}")

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
        log.info(f"Scan controller ready with {len(self.cameras)} cameras")

    def capture_with_flash(self, camera_idx: int) -> np.ndarray:
        """Turn flash on, capture from specified camera while lit, turn flash off."""
        cam = self.cameras[camera_idx]
        self.flash.on()
        try:
            time.sleep(0.05)
            frame = cam.capture()
        finally:
            self.flash.off()
        self.last_frames[camera_idx] = frame
        return frame

    def capture_both(self) -> dict[int, np.ndarray]:
        """Capture from all cameras during one flash pulse."""
        self.flash.on()
        try:
            time.sleep(0.05)
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

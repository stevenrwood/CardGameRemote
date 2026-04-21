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
import random
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

from game_engine import GameEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAYER_NAMES = ["Steve", "Bill", "David", "Joe", "Rodney"]
NUM_ZONES = len(PLAYER_NAMES)

DEFAULT_CAMERA_INDEX = 1  # Fallback if we can't find the Brio by name
DEFAULT_CAMERA_NAME = "BRIO"  # avfoundation device name substring to prefer
DEFAULT_THRESHOLD = 30.0
DEFAULT_RESOLUTION = "auto"

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
TRAINING_DIR = Path(__file__).parent / "training_data"
CONFIG_FILE = Path(__file__).parent.parent / "local" / "config.json"
CAPTURE_FILE = Path("/tmp/card_scanner_frame.jpg")
YOLO_MODEL_PATH = Path(__file__).parent / "models" / "card_detector.pt"

CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Speech queue
# ---------------------------------------------------------------------------

PREFERRED_VOICE_BASE = os.environ.get("SPEECH_VOICE", "Tessa")


def _resolve_best_voice(base):
    """Pick the highest-quality installed variant of `base` from `say -v ?`.

    Quality order: Premium > Enhanced > base. Matches the macOS naming
    convention of "<Name> (Enhanced)" / "<Name> (Premium)" siblings to
    the plain voice.
    """
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True,
                             timeout=5, text=True)
    except Exception:
        return base
    lines = (out.stdout or "").splitlines()
    low = base.lower()
    tiers = {"premium": None, "enhanced": None, "base": None}
    for line in lines:
        # Voice name is the start of the line, columns are padded with spaces.
        m = re.match(r"^\s*(.+?)\s{2,}", line)
        if not m:
            continue
        name = m.group(1).strip()
        if low not in name.lower():
            continue
        if "(premium)" in name.lower():
            tiers["premium"] = name
        elif "(enhanced)" in name.lower():
            tiers["enhanced"] = name
        elif name.lower() == low:
            tiers["base"] = name
    return tiers["premium"] or tiers["enhanced"] or tiers["base"] or base


class SpeechQueue:
    def __init__(self):
        self._queue = Queue()
        self.voice = _resolve_best_voice(PREFERRED_VOICE_BASE)
        # `log` isn't constructed until after SpeechQueue, so use print here.
        print(f"[INFO] Speech voice: {self.voice}", file=sys.stderr)
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
                subprocess.run(["say", "-v", self.voice, p],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)

speech = SpeechQueue()

# ---------------------------------------------------------------------------
# Log buffer
# ---------------------------------------------------------------------------

# Logs live in ~/Library/Logs/cardgame-host/ rather than ~/Downloads so they
# stay outside macOS Transparency Consent and Control (TCC) protected
# directories. This lets SSH/scp sessions read them without granting sshd
# Full Disk Access.
LOG_DIR = Path.home() / "Library" / "Logs" / "cardgame-host"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "log.txt"
LOG_ARCHIVE_DIR = LOG_DIR

class LogBuffer:
    def __init__(self, maxlines=500):
        self._lines = []
        self._lock = Lock()
        # Path of the current hand-by-hand archive (the poker_* file),
        # set when start_night() runs. Each log.log() also appends here
        # so the archive is a full copy of the live log for this night,
        # available for later inspection (e.g. cleanup_training_data.py).
        self._archive_path = None
        # Overwrite log file on startup
        LOG_FILE.write_text("")

    def log(self, msg):
        # ms precision so we can time sub-second pipeline stages.
        now = datetime.now()
        ts = now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"
        line = f"[{ts}] {msg}"
        print(line)
        with self._lock:
            self._lines.append(line)
            self._lines = self._lines[-500:]
        # Append to both the live log and (if a poker night is active) the
        # per-night archive file, so historical analysis tools can replay
        # what happened after the fact.
        try:
            with open(LOG_FILE, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
        if self._archive_path is not None:
            try:
                with open(self._archive_path, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def get(self, n=50):
        with self._lock:
            return list(self._lines[-n:])

    def clear(self):
        """Wipe the in-memory buffer and the backing log file."""
        with self._lock:
            self._lines = []
        try:
            LOG_FILE.write_text("")
        except Exception:
            pass

    def start_night(self):
        """Rotate the working log and start a dated archive for this
        poker night. Returns the archive filename."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = LOG_ARCHIVE_DIR / f"poker_{stamp}.txt"
        try:
            LOG_FILE.write_text("")
        except Exception:
            pass
        try:
            LOG_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            archive.write_text("")
        except Exception:
            pass
        with self._lock:
            self._lines = []
            self._archive_path = archive
        self.log(f"=== Poker night started {stamp} → {archive.name} ===")
        return archive.name

    def end_night(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log(f"=== Poker night ended {stamp} ===")
        # Stop appending to the per-night archive; the file on disk stays
        # for historical analysis. The next start_night() creates a new
        # archive file and takes over.
        with self._lock:
            self._archive_path = None


log = LogBuffer()

# ---------------------------------------------------------------------------
# Frame capture via ffmpeg
# ---------------------------------------------------------------------------

class FrameCapture:
    @staticmethod
    def find_index_by_name(prefer_substring):
        """Scan avfoundation devices and return the index whose name contains
        `prefer_substring` (case-insensitive). Returns None if not found or
        if ffmpeg isn't available.

        Example ffmpeg output we're parsing:
          [AVFoundation indev @ ...] AVFoundation video devices:
          [AVFoundation indev @ ...] [0] Logitech BRIO
          [AVFoundation indev @ ...] [1] MacBook Neo Camera
          [AVFoundation indev @ ...] AVFoundation audio devices:
          [AVFoundation indev @ ...] [0] ...
        """
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "avfoundation",
                 "-list_devices", "true", "-i", ""],
                capture_output=True, timeout=5,
            )
        except Exception:
            return None
        text = (proc.stderr or b"").decode(errors="replace")
        in_video = False
        needle = prefer_substring.lower()
        for line in text.splitlines():
            if "AVFoundation video devices" in line:
                in_video = True
                continue
            if "AVFoundation audio devices" in line:
                in_video = False
                continue
            if not in_video:
                continue
            m = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
            if not m:
                continue
            idx, name = int(m.group(1)), m.group(2)
            if needle in name.lower():
                log.log(f"Camera: matched '{name}' at index {idx}")
                return idx
        return None

    @staticmethod
    def find_cv_index_by_name(preferred_substring):
        """Find the OpenCV VideoCapture index for a camera whose name matches
        preferred_substring. Uses AVFoundations own device enumeration via
        PyObjC, which shares its ordering with OpenCVs AVFoundation backend
        — ffmpegs avfoundation indices do NOT match. Returns None if PyObjC
        is unavailable or no matching camera is found.
        """
        try:
            from AVFoundation import (
                AVCaptureDevice,
                AVCaptureDeviceDiscoverySession,
                AVCaptureDeviceTypeBuiltInWideAngleCamera,
                AVCaptureDeviceTypeExternal,
                AVMediaTypeVideo,
            )
        except Exception as e:
            log.log(f"[CAPTURE] pyobjc AVFoundation unavailable: {e}")
            return None
        # Prefer the discovery session (modern API, includes external cams).
        try:
            types = [
                AVCaptureDeviceTypeBuiltInWideAngleCamera,
                AVCaptureDeviceTypeExternal,
            ]
            session = AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
                types, AVMediaTypeVideo, 0
            )
            devices = list(session.devices())
        except Exception:
            devices = list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo) or [])
        needle = preferred_substring.lower()
        hit = None
        for i, dev in enumerate(devices):
            name = str(dev.localizedName())
            log.log(f"[CAPTURE] AVFoundation idx={i}: {name}")
            if hit is None and needle in name.lower():
                hit = (i, name)
        if hit is not None:
            log.log(f"[CAPTURE] Matched '{hit[1]}' at OpenCV idx {hit[0]}")
            return hit[0]
        return None

    def __init__(self, camera_index, resolution="auto", camera_name_hint=None,
                 cv_index_override=None, focus=None):
        self.camera_index = camera_index
        self.camera_name_hint = camera_name_hint
        self.cv_index_override = cv_index_override
        # None = leave autofocus on; otherwise disable AF and apply this
        # manual focus value (Logitech Brio usable range is 0..255, lower
        # = farther). Settable at runtime via set_focus().
        self.focus = focus
        self._active_cap = None
        self._check_ffmpeg()  # only used by _find_best_resolution below
        self.resolution = self._find_best_resolution() if resolution == "auto" else resolution
        w, h = self.resolution.split("x")
        self.width, self.height = int(w), int(h)
        # Persistent stream via OpenCVs AVFoundation backend. Previously
        # we piped MJPEG from a long-running ffmpeg, but that stalled
        # on the Brio after the first frame no matter which pixel format
        # we negotiated. cv2.VideoCapture keeps the camera open and
        # returns fresh frames on every read().
        self._stderr_tail = b""
        self._frame_lock = Lock()
        self._latest_frame = None
        self._stop = False
        self._stream_thread = None
        self._last_sig = -1.0
        self._unique_sigs = 0
        self._sig_err_logged = False
        self._start_stream()

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
                    "-pixel_format", "uyvy422",
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

    def _start_stream(self):
        # Two strategies to find the right OpenCV VideoCapture index:
        # 1. PyObjC AVFoundation device enumeration — shares OpenCVs
        #    ordering, and lets us match by camera name. This is the
        #    reliable path once a name hint is available.
        # 2. Resolution probe fallback — open each index, grab one frame,
        #    keep the first that delivers the requested WxH. Works when
        #    only one camera supports 4K, but confuses Brio with another
        #    4K-capable webcam (e.g., EMeet Pixy).
        self._cv_index = None
        self._initial_cap = None
        if self.cv_index_override is not None:
            log.log(f"[CAPTURE] Using forced OpenCV idx {self.cv_index_override}")
            self._cv_index = int(self.cv_index_override)
        elif self.camera_name_hint:
            self._cv_index = self.find_cv_index_by_name(self.camera_name_hint)
        if self._cv_index is None:
            self._cv_index, self._initial_cap = self._find_matching_cv_cap()
        if self._cv_index is None:
            log.log(
                f"[CAPTURE] no cv2.VideoCapture index delivered "
                f"{self.width}x{self.height}; falling back to 0"
            )
            self._cv_index = 0
        if self._initial_cap is None:
            MJPG = int(cv2.VideoWriter_fourcc(*"MJPG"))
            self._initial_cap = cv2.VideoCapture(self._cv_index, cv2.CAP_AVFOUNDATION)
            self._initial_cap.set(cv2.CAP_PROP_FOURCC, MJPG)
            self._initial_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._initial_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._initial_cap.set(cv2.CAP_PROP_FPS, 30)
        self._apply_focus(self._initial_cap)
        self._stream_thread = Thread(target=self._read_stream, daemon=True)
        self._stream_thread.start()

    def _find_matching_cv_cap(self):
        """Open each OpenCV index, request target resolution as MJPG, read
        one frame, log what came back. Return the first index that actually
        delivered the requested size along with its open VideoCapture."""
        target_w, target_h = self.width, self.height
        MJPG = int(cv2.VideoWriter_fourcc(*"MJPG"))
        first_match = None
        for idx in range(6):
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                try: cap.release()
                except Exception: pass
                log.log(f"[CAPTURE] probe idx={idx}: not opened")
                continue
            cap.set(cv2.CAP_PROP_FOURCC, MJPG)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, frame = cap.read()
            w = h = 0
            if ok and frame is not None:
                h, w = frame.shape[:2]
            log.log(f"[CAPTURE] probe idx={idx}: {w}x{h}")
            if first_match is None and w == target_w and h == target_h:
                first_match = (idx, cap)
            else:
                try: cap.release()
                except Exception: pass
        return first_match if first_match is not None else (None, None)

    def _read_stream(self):
        """Keep an AVFoundation VideoCapture open and push each decoded
        BGR frame into self._latest_frame. Much simpler and more stable
        on the Mac than piping MJPEG through ffmpeg, which stalled after
        one frame on the Brio."""
        backoff_s = 1.0
        frame_count = 0
        last_log = time.time()
        first_pass = True
        MJPG = int(cv2.VideoWriter_fourcc(*"MJPG"))
        while not self._stop:
            if first_pass and self._initial_cap is not None:
                # Reuse the already-open capture from the resolution probe.
                cap = self._initial_cap
                self._initial_cap = None
            else:
                cap = cv2.VideoCapture(self._cv_index, cv2.CAP_AVFOUNDATION)
                # IMPORTANT: the Brio delivers 4K as MJPEG on the wire.
                # Without this FOURCC hint OpenCV negotiates an
                # uncompressed pixel format, which saturates USB and
                # makes read() fail constantly at 4K.
                cap.set(cv2.CAP_PROP_FOURCC, MJPG)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                self._apply_focus(cap)
            self._active_cap = cap
            first_pass = False
            # Smallest internal buffer so read() returns the newest frame
            # rather than an old queued one.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if not cap.isOpened():
                log.log("[CAPTURE] VideoCapture failed to open — retrying")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(backoff_s)
                backoff_s = min(10.0, backoff_s * 2)
                continue
            backoff_s = 1.0
            log.log(
                f"[CAPTURE] VideoCapture opened idx={self._cv_index} "
                f"{self.width}x{self.height}"
            )
            # Tolerate a burst of transient read() failures before tearing
            # the whole capture down and reopening — AVFoundation under 4K
            # MJPEG occasionally delivers a bad packet that read() rejects.
            # A quick sleep+retry almost always recovers; a reopen costs
            # a full camera-open cycle we cannot afford.
            fail_streak = 0
            MAX_READ_FAILS = 10
            while not self._stop:
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail_streak += 1
                    if fail_streak >= MAX_READ_FAILS:
                        log.log(
                            f"[CAPTURE] read() failed {fail_streak}x in a row "
                            f"— reopening capture"
                        )
                        break
                    time.sleep(0.05)
                    continue
                fail_streak = 0
                with self._frame_lock:
                    self._latest_frame = frame
                frame_count += 1
                try:
                    sig = float(frame.sum())
                except Exception as e:
                    if not self._sig_err_logged:
                        log.log(f"[CAPTURE] sig compute failed: {e}")
                        self._sig_err_logged = True
                    sig = 0.0
                if sig != self._last_sig:
                    self._last_sig = sig
                    self._unique_sigs += 1
                now = time.time()
                if now - last_log >= 30:
                    fps = frame_count / max(1e-3, now - last_log)
                    log.log(
                        f"[CAPTURE] Brio stream: {frame_count} frames "
                        f"in {now - last_log:.0f}s ({fps:.1f} fps, "
                        f"{self._unique_sigs} unique, last_sig={sig:.0f})"
                    )
                    frame_count = 0
                    self._unique_sigs = 0
                    last_log = now
            try:
                cap.release()
            except Exception:
                pass

    def _apply_focus(self, cap):
        """Set autofocus + manual focus on a freshly-opened VideoCapture."""
        if cap is None or self.focus is None:
            return
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
            cap.set(cv2.CAP_PROP_FOCUS, float(self.focus))
            log.log(f"[CAPTURE] Autofocus off, focus={self.focus}")
        except Exception as e:
            log.log(f"[CAPTURE] focus set failed: {e}")

    def set_focus(self, value):
        """Update focus at runtime. None turns autofocus back on."""
        self.focus = value
        cap = self._active_cap
        if cap is None:
            return
        try:
            if value is None:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
                log.log("[CAPTURE] Autofocus re-enabled")
            else:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
                cap.set(cv2.CAP_PROP_FOCUS, float(value))
                log.log(f"[CAPTURE] focus={value}")
        except Exception as e:
            log.log(f"[CAPTURE] set_focus failed: {e}")

    def capture(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def close(self):
        self._stop = True

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


def _stats_bump(state, key, delta=1):
    """Increment a key in state.stats if state exists. Zone monitor uses
    this to tally YOLO vs Claude recognitions without needing a hard
    dependency on AppState being initialized yet (first-run safety)."""
    if state is None or not hasattr(state, "stats"):
        return
    state.stats[key] = state.stats.get(key, 0) + delta


def _recapture_baselines(s):
    """Capture zone baselines AND reset any watching-phase bookkeeping so
    the deal-order gate starts clean for the next round."""
    if s.cal.ok and s.latest_frame is not None:
        s.monitor.capture_baselines(s.latest_frame)
    if hasattr(s, "_zones_with_motion"):
        s._zones_with_motion = set()

# ---------------------------------------------------------------------------
# Zone monitor
# ---------------------------------------------------------------------------

class ZoneMonitor:
    def __init__(self, threshold):
        self.threshold = threshold
        self.yolo_min_conf = 0.50  # below this, fall back to Claude API
        self.baselines = {}
        self.last_card = {}
        self.zone_state = {}
        self.pending = {}
        self.recognition_details = {}  # name -> {yolo, yolo_conf, claude, final}
        self.recognition_crops = {}   # name -> crop (numpy array) at time of recognition
        self._yolo_model = None
        self._yolo_lock = Lock()  # YOLO model is not thread-safe
        self._client = None
        self._load_yolo()

    def _load_yolo(self):
        if YOLO_MODEL_PATH.exists():
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(str(YOLO_MODEL_PATH))
                log.log(f"YOLO model loaded: {YOLO_MODEL_PATH.name}")
            except Exception as e:
                log.log(f"YOLO load failed: {e}")
        else:
            log.log("No YOLO model found — will use Claude API")

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
            except ImportError:
                pass
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
        """Check all zones. YOLO runs for each changed zone, then one batched
        Claude call handles all zones where YOLO was below threshold."""
        changed = {}  # name -> crop for zones needing recognition
        for z in _state.cal.zones:
            name = z["name"]
            if name not in self.baselines or self.pending.get(name):
                continue
            # Skip corrected zones — dealer already fixed this card
            if self.zone_state.get(name) == "corrected":
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
                changed[name] = crop.copy()

        if changed:
            # Mark all as pending so we don't double-process
            for name in changed:
                self.pending[name] = True
            Thread(target=self._recognize_batch, args=(changed,), daemon=True).start()

    def _recognize_batch(self, zone_crops):
        """Run YOLO on all zones, then batch Claude call for low-confidence ones."""
        need_claude = {}  # name -> (crop, yolo_result, yolo_conf)

        # Phase 1: YOLO all zones (sequential, under lock)
        for name, crop in zone_crops.items():
            details = {"yolo": "", "yolo_conf": 0, "claude": "", "final": ""}
            t0 = time.time()
            try:
                if self._yolo_model is not None:
                    log.log(f"[{name}] YOLO inference started")
                    t_yolo = time.time()
                    result, conf = self._recognize_yolo(crop)
                    yolo_ms = (time.time() - t_yolo) * 1000
                    log.log(f"[{name}] YOLO result: {result} ({conf:.0%}) in {yolo_ms:.0f}ms")
                    details["yolo"] = result
                    details["yolo_conf"] = round(conf * 100)

                    if result != "No card" and conf >= self.yolo_min_conf:
                        # YOLO confident — accept it
                        total_ms = (time.time() - t0) * 1000
                        self.last_card[name] = result
                        self.zone_state[name] = "recognized"
                        details["final"] = result
                        details["source"] = "yolo"
                        _stats_bump(_state, "yolo_right")
                        log.log(f"[{name}] RECOGNIZED: {result} (total {total_ms:.0f}ms)")
                        self._save(name, crop, result)
                        speech.say(f"{name}, {result}")
                    else:
                        # Need Claude
                        need_claude[name] = (crop, details)
                        self.zone_state[name] = "processing"
                        continue
                else:
                    need_claude[name] = (crop, details)
                    self.zone_state[name] = "processing"
                    continue
            except Exception as e:
                log.log(f"[{name}] YOLO error: {e}")
                self.zone_state[name] = "empty"

            self.recognition_details[name] = details
            self.recognition_crops[name] = crop
            self.pending[name] = False

        # Phase 2: Batch Claude call for all low-confidence zones
        if need_claude and self.client:
            self._recognize_claude_batch(need_claude)
        else:
            # No Claude available — mark remaining as empty
            for name, (crop, details) in need_claude.items():
                yolo_result = details.get("yolo", "")
                if yolo_result and yolo_result != "No card":
                    # Accept low-conf YOLO rather than nothing
                    self.last_card[name] = yolo_result
                    self.zone_state[name] = "recognized"
                    details["final"] = yolo_result
                    details["source"] = "yolo"
                    _stats_bump(_state, "yolo_right")
                    log.log(f"[{name}] RECOGNIZED (low conf, no Claude): {yolo_result}")
                    self._save(name, crop, yolo_result)
                    speech.say(f"{name}, {yolo_result}")
                else:
                    self.zone_state[name] = "empty"
                self.recognition_details[name] = details
                self.recognition_crops[name] = crop
                self.pending[name] = False

    def _recognize_claude_batch(self, need_claude):
        """Single Claude API call with all zone images."""
        names = list(need_claude.keys())
        log.log(f"[CLAUDE] Batch call for {len(names)} zones: {', '.join(names)}")

        # Build multi-image message
        content = []
        for name in names:
            crop, details = need_claude[name]
            b64 = base64.b64encode(
                cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            ).decode()
            content.append({"type": "text", "text": f"Image {name}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
        content.append({"type": "text", "text":
            "For each labeled image above, identify the playing card. "
            "Reply with one line per image: 'Name: Rank of Suit' (e.g. 'Steve: 4 of Clubs'). "
            "If unclear or no card visible, reply 'Name: No card'."})

        # Retry with exponential backoff on 529 overloaded / 500 errors
        t0 = time.time()
        resp = None
        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=200,
                    messages=[{"role": "user", "content": content}])
                break
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_transient = "529" in err_str or "overloaded" in err_str.lower() or "500" in err_str
                if not is_transient or attempt == 2:
                    raise
                delay = 2 ** attempt  # 1s, 2s, 4s
                log.log(f"[CLAUDE] Transient error (attempt {attempt+1}/3), retrying in {delay}s: {err_str[:80]}")
                time.sleep(delay)
        try:
            claude_ms = (time.time() - t0) * 1000
            raw = resp.content[0].text.strip()
            log.log(f"[CLAUDE] Response in {claude_ms:.0f}ms: {raw}")

            # Parse response lines
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Match "Name: Rank of Suit" or "Name: No card"
                for name in names:
                    if line.lower().startswith(name.lower() + ":"):
                        value = line[len(name)+1:].strip()
                        crop, details = need_claude[name]
                        m = re.search(r'(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)', value, re.I)
                        if m:
                            result = f"{m.group(1).capitalize()} of {m.group(2).capitalize()}"
                            details["claude"] = result
                            details["final"] = result
                            details["source"] = "claude"
                            _stats_bump(_state, "claude_right")
                            self.last_card[name] = result
                            self.zone_state[name] = "recognized"
                            log.log(f"[{name}] RECOGNIZED (Claude): {result}")
                            self._save(name, crop, result)
                            speech.say(f"{name}, {result}")
                        else:
                            details["claude"] = "No card"
                            self.zone_state[name] = "empty"
                            log.log(f"[{name}] Claude: No card")
                        self.recognition_details[name] = details
                        self.recognition_crops[name] = crop
                        self.pending[name] = False
                        break

            # Handle any names not found in response
            for name in names:
                if self.pending.get(name):
                    crop, details = need_claude[name]
                    # Fall back to low-conf YOLO if available
                    yolo_result = details.get("yolo", "")
                    if yolo_result and yolo_result != "No card":
                        details["final"] = yolo_result
                        details["source"] = "yolo"
                        _stats_bump(_state, "yolo_right")
                        self.last_card[name] = yolo_result
                        self.zone_state[name] = "recognized"
                        log.log(f"[{name}] RECOGNIZED (YOLO fallback): {yolo_result}")
                        self._save(name, crop, yolo_result)
                        speech.say(f"{name}, {yolo_result}")
                    else:
                        self.zone_state[name] = "empty"
                    self.recognition_details[name] = details
                    self.recognition_crops[name] = crop
                    self.pending[name] = False

        except Exception as e:
            log.log(f"[CLAUDE] Batch error: {e}")
            # Fall back to YOLO results for all
            for name in names:
                if not self.pending.get(name):
                    continue
                crop, details = need_claude[name]
                yolo_result = details.get("yolo", "")
                if yolo_result and yolo_result != "No card":
                    details["final"] = yolo_result
                    details["source"] = "yolo"
                    _stats_bump(_state, "yolo_right")
                    self.last_card[name] = yolo_result
                    self.zone_state[name] = "recognized"
                    log.log(f"[{name}] RECOGNIZED (YOLO, Claude failed): {yolo_result}")
                    self._save(name, crop, yolo_result)
                    speech.say(f"{name}, {yolo_result}")
                else:
                    self.zone_state[name] = "empty"
                self.recognition_details[name] = details
                self.recognition_crops[name] = crop
                self.pending[name] = False

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

    def _recognize_single(self, name, crop):
        """Recognize a single zone (used by test mode and deal mode)."""
        details = {"yolo": "", "yolo_conf": 0, "claude": "", "final": ""}
        try:
            if self._yolo_model is not None:
                result, conf = self._recognize_yolo(crop)
                details["yolo"] = result
                details["yolo_conf"] = round(conf * 100)
                if result != "No card" and conf >= self.yolo_min_conf:
                    self.last_card[name] = result
                    self.zone_state[name] = "recognized"
                    details["final"] = result
                    self._save(name, crop, result)
                    return
                if self.client:
                    result = self._recognize_claude(crop)
                    details["claude"] = result
                    if result and result != "No card":
                        self.last_card[name] = result
                        self.zone_state[name] = "recognized"
                        details["final"] = result
                        self._save(name, crop, result)
                        return
            self.zone_state[name] = "empty"
        except Exception as e:
            log.log(f"[{name}] error: {e}")
            self.zone_state[name] = "empty"
        finally:
            self.recognition_details[name] = details
            self.recognition_crops[name] = crop
            self.pending[name] = False

    def _recognize_yolo(self, crop):
        with self._yolo_lock:
            results = self._yolo_model.predict(crop, conf=0.2, verbose=False)
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = results[0].names[cls_idx]
            # Convert class name like "4c" to "4 of Clubs"
            rank = cls_name[:-1]
            suit_letter = cls_name[-1]
            suit = {"c": "Clubs", "d": "Diamonds", "h": "Hearts", "s": "Spades"}.get(suit_letter, suit_letter)
            rank_name = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}.get(rank, rank)
            return f"{rank_name} of {suit}", conf
        return "No card", 0.0

    def _recognize_claude(self, crop):
        b64 = base64.b64encode(
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        ).decode()
        resp = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=20,
            messages=[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},
                {"type":"text","text":"What playing card is this? Reply ONLY: 'Rank of Suit' (e.g. '4 of Clubs'). If unclear: 'No card'"},
            ]}])
        raw = resp.content[0].text.strip()
        m = re.search(r'(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)', raw, re.I)
        return f"{m.group(1).capitalize()} of {m.group(2).capitalize()}" if m else "No card"

    def _save(self, name, crop, result):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = result.replace(" ", "_").replace("/", "-")[:30]
        base = TRAINING_DIR / f"{ts}_{name}_{safe}"
        cv2.imwrite(str(base.with_suffix(".jpg")), crop)
        base.with_suffix(".txt").write_text(result)
        # Remember so a later user correction can delete this (wrong-label)
        # pair and replace it with the correct one.
        if not hasattr(self, "last_save_base"):
            self.last_save_base = {}
        self.last_save_base[name] = base

    def _delete_last_save(self, name):
        """Remove the most recent training_data save for this zone. Used
        when the user corrects a misrecognition so the bad label doesn't
        poison future YOLO training runs."""
        base = getattr(self, "last_save_base", {}).get(name)
        if base is None:
            return False
        removed = False
        for suffix in (".jpg", ".txt"):
            p = base.with_suffix(suffix)
            try:
                if p.exists():
                    p.unlink()
                    removed = True
            except Exception as e:
                log.log(f"[{name}] failed to remove stale training file {p}: {e}")
        self.last_save_base.pop(name, None)
        return removed

# ---------------------------------------------------------------------------
# Console scan trigger — watches dealer's zone, scans all zones when dealer dealt
# ---------------------------------------------------------------------------

def _brio_player_names(s):
    """Active players whose Brio zones the overhead watcher should scan.

    For both local and remote players: the dealer places face-up cards
    in the players Brio zone for all players to see (Rodneys flipped-
    up card in 7/27, his up cards in stud games, etc.). So every active
    player contributes a Brio zone.
    """
    return set(s.console_active_players)


def _console_watch_dealer(s, frame):
    """Watch the dealer's zone. When a card appears there, wait for settle then
    scan all active player zones in one batch. Dealer deals to themselves last,
    so this guarantees all cards are placed before scanning.

    After scan, if any active players have no recognized card, watch their zones
    and rescan when they move their cards."""
    phase = s.console_scan_phase

    if phase in ("idle", "confirmed"):
        return

    ge = s.game_engine
    dealer_name = ge.get_dealer().name
    brio_names = _brio_player_names(s)

    # Handle missing-card watching: any active player with empty card
    if phase == "watching_missing":
        missing_zones = []
        for z in s.cal.zones:
            name = z["name"]
            if name not in brio_names:
                continue
            if s.monitor.zone_state.get(name) == "corrected":
                continue
            card = s.monitor.last_card.get(name, "")
            if card and card != "No card":
                continue
            missing_zones.append(z)
        # If any missing zone now has a card, trigger rescan of all missing
        retry_crops = {}
        for z in missing_zones:
            crop = s.monitor.check_single(frame, z)
            if crop is not None:
                retry_crops[z["name"]] = crop.copy()
                s.monitor.pending[z["name"]] = True
        # Watchdog: if no movement was detected for >10s, force a rescan
        # anyway. The dealer may be adjusting a card that barely moves the
        # pixel diff (thin edge inside zone). Keeps us from getting stuck
        # prompting "adjust your card" with nothing happening.
        if not retry_crops and missing_zones:
            last = getattr(s, "_missing_prompt_time", 0.0)
            if last and time.time() - last >= 10.0:
                for z in missing_zones:
                    crop = s.monitor._crop(frame, z)
                    if crop is not None:
                        retry_crops[z["name"]] = crop.copy()
                        s.monitor.pending[z["name"]] = True
                if retry_crops:
                    log.log(f"[CONSOLE] Watchdog rescan of missing zones: {', '.join(retry_crops.keys())}")
                    s._missing_prompt_time = time.time()
        if retry_crops:
            log.log(f"[CONSOLE] Movement detected in missing zones: {', '.join(retry_crops.keys())}")
            s.console_scan_phase = "scanned"  # will transition to watching_missing again if still missing
            Thread(target=_console_rescan_missing, args=(s, retry_crops), daemon=True).start()
        return

    if phase == "scanned":
        # Wait until pending scans are done before deciding anything.
        if any(s.monitor.pending.get(n) for n in brio_names):
            return
        # Dealer deals to themselves last, so the motion trigger should
        # have coincided with an actual card in the dealer's own zone.
        # If the dealer zone is still empty post-scan, the trigger was a
        # hand/arm sweep — revert to watching instead of nagging every
        # player who is also missing.
        dealer_card = s.monitor.last_card.get(dealer_name, "")
        dealer_empty = not dealer_card or dealer_card == "No card"
        is_hit_round = (
            ge.current_game and ge.current_game.name.startswith("7/27")
            and s.console_up_round >= 1
        )
        if dealer_empty and not is_hit_round:
            log.log(
                f"[CONSOLE] {dealer_name}'s zone empty after scan — "
                f"likely a false trigger (arm over zone). Resuming watch."
            )
            # Only reset zones that did NOT land a card on this scan.
            # Cards already recognized or corrected keep their state so
            # the next motion trigger does not re-scan them from scratch.
            for nm in brio_names:
                state = s.monitor.zone_state.get(nm)
                if state in ("recognized", "corrected"):
                    continue
                s.monitor.zone_state[nm] = "empty"
                s.monitor.last_card[nm] = ""
            # Clear the deal-order gate too; the arm sweep tagged zones
            # that dont actually have cards yet.
            s._zones_with_motion = set()
            s.console_scan_phase = "watching"
            return
        missing = []
        for name in brio_names:
            if s.monitor.zone_state.get(name) == "corrected":
                continue
            card = s.monitor.last_card.get(name, "")
            if not card or card == "No card":
                missing.append(name)
        # In hit rounds (7/27), dealer deals one at a time around the
        # table. The first motion fires a scan long before the later
        # players get their cards, so a "missing" zone here just means
        # not yet dealt — frozen or not. Treat any unfrozen missing
        # zone as "keep watching", so the next motion trigger fires
        # another scan pass (already-recognized zones stay locked).
        if is_hit_round:
            unfrozen_missing = [
                n for n in missing if s.freezes.get(n, 0) < 3
            ]
            if unfrozen_missing:
                log.log(
                    f"[CONSOLE] Hit-round partial scan: still waiting on "
                    f"{', '.join(unfrozen_missing)} — resuming watch"
                )
                s.console_scan_phase = "watching"
            return
        if missing:
            # Per-player "please adjust" speech, capped at 2 prompts per
            # round so we stop nagging when YOLO and Claude simply can't
            # see a card there. 10s cooldown stays — no back-to-back
            # announcements even for a new set of names.
            now = time.time()
            last_speech = getattr(s, "_missing_speech_time", 0.0)
            if not hasattr(s, "_missing_speech_count"):
                s._missing_speech_count = {}
            if now - last_speech >= 10.0:
                to_say = [n for n in missing
                          if s._missing_speech_count.get(n, 0) < 2]
                if to_say:
                    names = " and ".join(to_say)
                    log.log(f"[CONSOLE] Missing cards: {names} — prompting to adjust")
                    speech.say(f"{names}, please adjust your card")
                    s._missing_speech_time = now
                    for n in to_say:
                        s._missing_speech_count[n] = s._missing_speech_count.get(n, 0) + 1
                else:
                    log.log(
                        "[CONSOLE] Missing cards still unresolved — "
                        "2-announcement cap reached, waiting for manual entry"
                    )
            s.console_scan_phase = "watching_missing"
            s._missing_prompt_time = time.time()
        return

    if dealer_name not in brio_names:
        # Dealer is remote (Rodney) — nobody is placing cards in the
        # dealer zone so the regular trigger can't fire. Fall back to
        # any local brio zone as the motion trigger.
        alt = next((z for z in s.cal.zones if z["name"] in brio_names), None)
        if alt is None:
            return
        dealer_zone = alt
    else:
        dealer_zone = next((z for z in s.cal.zones if z["name"] == dealer_name), None)
    if dealer_zone is None:
        return

    if phase == "watching":
        # 7/27 hit rounds: dealer goes around asking each player for a card
        # or a freeze. Any LOCAL player's zone — not necessarily the dealer's
        # — may be the first to change. Skip frozen players (≥3 freezes).
        is_7_27_hit = (ge.current_game
                       and ge.current_game.name.startswith("7/27")
                       and s.console_up_round >= 1)
        if is_7_27_hit:
            trigger_zone = None
            for z in s.cal.zones:
                if z["name"] not in brio_names:
                    continue
                if s.freezes.get(z["name"], 0) >= 3:
                    continue
                if s.monitor.check_single(frame, z) is not None:
                    trigger_zone = z
                    break
            if trigger_zone is not None:
                log.log(f"[CONSOLE] Hit-round card detected in {trigger_zone['name']}'s zone — {s.brio_settle_s:.1f}s settle")
                s.console_scan_phase = "settling"
                s.console_settle_time = time.time()
            return
        # Deal-order gating: the dealer sweeps their own zone repeatedly
        # while dealing to everyone else. To avoid false triggers we only
        # treat a dealer-zone motion as "the deal is done" once every
        # OTHER active brio zone has already shown motion since the last
        # round reset. Other zones are tracked in s._zones_with_motion.
        if not hasattr(s, "_zones_with_motion"):
            s._zones_with_motion = set()
        for z in s.cal.zones:
            name = z["name"]
            if name not in brio_names:
                continue
            if name == dealer_zone["name"]:
                continue
            if name in s._zones_with_motion:
                continue
            if s.monitor.zone_state.get(name) in ("recognized", "corrected"):
                s._zones_with_motion.add(name)
                continue
            if s.monitor.check_single(frame, z) is not None:
                s._zones_with_motion.add(name)
                log.log(f"[CONSOLE] Motion seen in {name}'s zone "
                        f"({len(s._zones_with_motion)}/"
                        f"{len(brio_names) - 1} others)")
        others = brio_names - {dealer_zone["name"]}
        crop = s.monitor.check_single(frame, dealer_zone)
        if crop is not None and s._zones_with_motion >= others:
            log.log(f"[CONSOLE] Dealer card detected in {dealer_zone['name']}'s "
                    f"zone — {s.brio_settle_s:.1f}s settle")
            s.console_scan_phase = "settling"
            s.console_settle_time = time.time()
            return
        if crop is not None and not (s._zones_with_motion >= others):
            # Dealer zone moved but not all other zones have received a
            # card yet — likely dealer's arm crossing their own zone
            # during deal. Ignore, keep watching.
            pass
        # Heartbeat diagnostic: once every ~10s while we're stuck in
        # watching, log the per-zone diff from baseline for every active
        # zone so the user can tell whether Brio is seeing changes below
        # the threshold vs. not seeing changes at all (zones miscalibrated).
        now = time.time()
        if now - getattr(s, "_watch_diag_time", 0.0) >= 10.0:
            s._watch_diag_time = now
            diffs = []
            for z in s.cal.zones:
                if z["name"] not in brio_names:
                    continue
                bl = s.monitor.baselines.get(z["name"])
                cur = s.monitor._crop(frame, z)
                if bl is None or cur is None or cur.shape != bl.shape:
                    diffs.append(f"{z['name']}=?")
                    continue
                d = float(np.mean(cv2.absdiff(cur, bl)))
                diffs.append(f"{z['name']}={d:.1f}")
            log.log(
                f"[CONSOLE] watching {dealer_name}'s zone — "
                f"diffs vs baseline: {', '.join(diffs)} "
                f"(threshold {s.monitor.threshold:.0f})"
            )
        return

    if phase == "settling":
        if time.time() - s.console_settle_time < s.brio_settle_s:
            return
        # Trust the initial motion trigger — don't re-verify. The old 2s
        # re-check rejected real cards when auto-exposure drifted the
        # per-pixel diff below threshold. YOLO already filters hand-only
        # pass-overs by returning "No card" for the whole zone.
        log.log("[CONSOLE] Scanning all active zones")
        zone_crops = {}
        for z in s.cal.zones:
            name = z["name"]
            if name not in brio_names:
                continue
            # Lock already-identified cards: once a zone has a recognized
            # (or user-corrected) card for this round, skip it. Only zones
            # that are still empty get re-scanned when the dealer triggers
            # another motion event in the same round.
            if s.monitor.zone_state.get(name) in ("recognized", "corrected"):
                continue
            crop = s.monitor._crop(frame, z)
            if crop is None or crop.size == 0:
                continue
            zone_crops[name] = crop.copy()
            s.monitor.pending[name] = True
        s.console_scan_phase = "scanned"
        Thread(target=s.monitor._recognize_batch, args=(zone_crops,), daemon=True).start()


def _console_rescan_missing(s, zone_crops):
    """Rescan just the zones where cards moved. Reuses the batch pipeline."""
    s.monitor._recognize_batch(zone_crops)


# ---------------------------------------------------------------------------
# Follow the Queen tracking for overhead camera
# ---------------------------------------------------------------------------

_RANK_TO_7_27_VALUE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "J": 0.5, "Q": 0.5, "K": 0.5,
    # Ace handled separately.
}


def _compute_7_27_values(cards):
    """Given a list of (rank, suit) tuples, return the sorted list of
    possible 7/27 hand totals (<=27), one entry per distinct ace
    assignment. No aces → single-element list.
    """
    base = 0.0
    aces = 0
    for rank, _suit in cards:
        if rank == "A":
            aces += 1
            continue
        v = _RANK_TO_7_27_VALUE.get(rank)
        if v is None:
            continue
        base += v
    values = set()
    for k in range(aces + 1):
        total = base + k * 11 + (aces - k) * 1
        if total <= 27:
            # Format as int if whole, else keep .5
            values.add(total if total != int(total) else int(total))
    return sorted(values)


def _speak_value(v):
    """Render a 7/27 numeric value for speech. Half-integers become '… and a half'.

    Examples:
      0.5  -> "a half"
      7.5  -> "7 and a half"
      12   -> "12"
    """
    if isinstance(v, int) or v == int(v):
        return str(int(v))
    whole = int(v)
    frac = v - whole
    if abs(frac - 0.5) < 1e-6:
        return "a half" if whole == 0 else f"{whole} and a half"
    return f"{v:g}"


def _format_values_phrase(values):
    """Turn [2, 12, 22] into '2, 12, or 22' for speech, preserving half-speak."""
    strs = [_speak_value(v) for v in values]
    if len(strs) == 1:
        return strs[0]
    if len(strs) == 2:
        return f"{strs[0]} or {strs[1]}"
    return ", ".join(strs[:-1]) + f", or {strs[-1]}"


_ALL_CARDS = [
    f"{r_long} of {su.capitalize()}"
    for r_long in ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                   "Jack", "Queen", "King"]
    for su in ["clubs", "diamonds", "hearts", "spades"]
]


def _dedup_round_cards_against_seen(s, round_cards):
    """If a recognized up card duplicates a card we've already seen in this
    hand (prior up rounds, Rodney's scanned down cards, or another player's
    card in the same round), swap it for a random card that isn't already
    in play. Operates in place on round_cards.

    The scanner can't distinguish two identical-looking cards, and Claude
    will sometimes echo back whatever it saw last. Rodney's down cards are
    invisible to the overhead camera, so a duplicate against one of them
    is a near-certain misread.
    """
    seen = set()
    # Prior confirmed up cards
    for c in s.console_hand_cards:
        seen.add(c["card"])
    # Rodney's known down cards (verified + pending low-conf guesses).
    # Skip his flipped slot — the flipped card IS this round's up card
    # for Rodney, so including it in seen would flag his own scan as a
    # self-collision and trigger a random substitution.
    suit_full = {"c": "Clubs", "d": "Diamonds", "h": "Hearts", "s": "Spades"}
    rank_full = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}
    def _canonical(rank, suit):
        return f"{rank_full.get(rank, rank)} of {suit.capitalize()}"
    flipped_slot = (s.rodney_flipped_up or {}).get("slot")
    for slot_num, d in s.rodney_downs.items():
        if slot_num == flipped_slot:
            continue
        seen.add(_canonical(d["rank"], d["suit"]))
    for d in s.slot_pending.values():
        seen.add(_canonical(d["rank"], d["suit"]))

    for entry in round_cards:
        card = entry.get("card", "")
        player = entry.get("player", "")
        if card in seen:
            # If the user has explicitly corrected this zone, trust the
            # correction and keep the card as-is — the collision is
            # almost always a misrecognized down card in seen, not the
            # users value. Otherwise log the collision but still keep
            # the scan; randomly substituting a card corrupts wild
            # tracking (fake Queens) and hand evaluation.
            if s.monitor.zone_state.get(player) == "corrected":
                log.log(
                    f"[CONFIRM] {player}: {card} collides with seen card "
                    f"but was user-corrected — keeping as-is"
                )
            else:
                log.log(
                    f"[CONFIRM] {player}: {card} collides with seen card "
                    f"— leaving as-is (dealer can correct if wrong)"
                )
        seen.add(card)


def _update_7_27_freezes(s, round_cards):
    """Apply freeze-count changes for a 7/27 hit round.

    round_cards comes from /api/console/confirm and lists the players
    who have a recognized up card in this round's Brio scan. Players in
    that list took a card (freeze reset to 0); players not in it and
    not already frozen tick up by 1. Three-in-a-row means frozen: no
    more cards this hand.
    """
    took = {c["player"] for c in round_cards}
    newly_frozen = []
    for name in s.console_active_players:
        if s.freezes.get(name, 0) >= 3:
            continue  # already frozen
        if name in took:
            s.freezes[name] = 0
        else:
            s.freezes[name] = s.freezes.get(name, 0) + 1
            if s.freezes[name] >= 3:
                newly_frozen.append(name)
    for name in newly_frozen:
        log.log(f"[7/27] {name} is frozen")
        speech.say(f"{name} is frozen")


def _announce_7_27_hand_values(s):
    """Walk each active player's up cards and announce their 7/27 totals,
    then announce which player is first-to-bet (highest value)."""
    ge = s.game_engine
    if not ge.current_game or not ge.current_game.name.startswith("7/27"):
        return
    RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
    # Gather up cards per player from the hand-wide history.
    per_player = {}
    for entry in s.console_hand_cards:
        txt = entry.get("card", "")
        parts = txt.split(" of ")
        if len(parts) != 2:
            continue
        rank_full, suit_full = parts[0].strip(), parts[1].strip().lower()
        rank = RANK_SHORT.get(rank_full, rank_full)
        per_player.setdefault(entry["player"], []).append((rank, suit_full))

    best_player = None
    best_high = -1
    per_player_values = {}
    for name in s.console_active_players:
        cards = per_player.get(name, [])
        if not cards:
            continue
        values = _compute_7_27_values(cards)
        if not values:
            continue
        per_player_values[name] = values
        # Keep a log entry per player for debugging, but only speak the winner.
        log.log(f"[7/27] {name}: {_format_values_phrase(values)}")
        high = max(values)
        if high > best_high:
            best_high = high
            best_player = name

    if best_player is not None:
        best_values = per_player_values.get(best_player, [best_high])
        # Descending list — highest first. If the player has aces, include
        # the other valid totals ("high of 25 or 15").
        ordered = sorted(set(best_values), reverse=True)
        if len(ordered) == 1:
            tail = _speak_value(ordered[0])
        else:
            tail = _format_values_phrase(ordered)
        phrase = f"{best_player}, your bet with high of {tail}"
        log.log(f"[7/27] Bet first: {phrase}")
        speech.say(phrase)


def _announce_poker_hand_bet_first(s):
    """Announce who bets first at a poker-hand game based on best visible
    hand. Skips 7/27 (its own announcer), Challenge games, and all-down
    games (5CD, 3 Toed Pete) where nobody has an up card to compare."""
    ge = s.game_engine
    if not ge.current_game:
        return
    if ge.current_game.name.startswith("7/27"):
        return
    if any(ph.type.value == "challenge" for ph in ge.current_game.phases):
        return
    has_up_deal = any(
        ph.type.value in ("deal", "community") and "up" in ph.pattern
        for ph in ge.current_game.phases
    )
    if not has_up_deal:
        return
    try:
        from poker_hands import best_hand, HandResult, RANK_VALUE, RANK_NAME, VALUE_RANK
    except Exception as e:
        log.log(f"[POKER] best_hand unavailable: {e}")
        return

    RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
    per_player_cards = {}
    for entry in s.console_hand_cards:
        parts = entry.get("card", "").split(" of ")
        if len(parts) != 2:
            continue
        rank_full, suit_full = parts[0].strip(), parts[1].strip().lower()
        rank = RANK_SHORT.get(rank_full, rank_full)
        per_player_cards.setdefault(entry["player"], []).append((rank, suit_full))

    wild_ranks = list(getattr(ge, "wild_ranks", []) or [])
    best_player = None
    best_result = None
    for name in s.console_active_players:
        if name in s.folded_players:
            continue
        cards = per_player_cards.get(name, [])
        if not cards:
            continue
        try:
            if len(cards) == 1:
                # Single up card — treat as high-card only.
                rank, suit = cards[0]
                v = RANK_VALUE.get(rank, 0)
                result = HandResult(
                    "high_card",
                    f"{RANK_NAME.get(rank, rank)} high",
                    [v],
                    [],
                )
            else:
                result = best_hand(cards, wild_ranks=wild_ranks)
        except Exception as e:
            log.log(f"[POKER] eval {name} failed: {e}")
            continue
        log.log(f"[POKER] {name}: {result.label}")
        key = (result.rank, result.tiebreakers)
        if best_result is None or key > (best_result.rank, best_result.tiebreakers):
            best_result = result
            best_player = name

    if best_player is not None and best_result is not None:
        phrase = f"{best_player}, your bet with {best_result.label}"
        log.log(f"[POKER] Bet first: {phrase}")
        speech.say(phrase)


def _check_follow_the_queen_round(s, round_cards):
    """Check cards for Follow the Queen wild at end of round.

    Args:
        round_cards: list of {"player": name, "card": "Rank of Suit"} in deal order
    """
    ge = s.game_engine
    if not ge.current_game or ge.current_game.dynamic_wild != "follow_the_queen":
        return

    RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}

    for c in round_cards:
        parts = c["card"].split(" of ")
        if len(parts) != 2:
            continue
        rank = parts[0]
        rank_short = RANK_SHORT.get(rank, rank)

        if ge.last_up_was_queen:
            if rank_short == "Q":
                # Queen immediately after a Queen: ignore the earlier one
                # and keep watching. The second Queen's follower is what
                # becomes wild. (Avoids the "Queens and Queens are wild"
                # annunciation.)
                pass
            else:
                ge.wild_ranks = ["Q", rank_short]
                plural = f"{rank}'s" if rank.isdigit() else f"{rank}s"
                ge.wild_label = f"Queens and {plural} are wild"
                log.log(f"[WILD] {ge.wild_label}")
                speech.say(f"Queens and {plural} are now wild")

        ge.last_up_was_queen = (rank_short == "Q")

    # Always announce current wild state at end of round if non-default
    if ge.wild_label and ge.wild_label != "Queens wild":
        log.log(f"[WILD] Current: {ge.wild_label}")


def _recompute_follow_the_queen(s):
    """Replay FTQ queen-follower logic against console_hand_cards in round
    order and update ge.wild_ranks/wild_label. Used when a correction
    changes a card that may have been the follower of a Queen. Announces
    the new wild state if it differs from the current one."""
    ge = s.game_engine
    if not ge.current_game or ge.current_game.dynamic_wild != "follow_the_queen":
        return
    prior_label = ge.wild_label
    RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
    ge.wild_ranks = ["Q"]
    ge.wild_label = "Queens wild"
    ge.last_up_was_queen = False
    by_round = {}
    for e in s.console_hand_cards:
        by_round.setdefault(e.get("round", 0), []).append(e)
    for r in sorted(by_round.keys()):
        for c in by_round[r]:
            parts = c.get("card", "").split(" of ")
            if len(parts) != 2:
                continue
            rank = parts[0]
            rank_short = RANK_SHORT.get(rank, rank)
            if ge.last_up_was_queen and rank_short != "Q":
                ge.wild_ranks = ["Q", rank_short]
                plural = f"{rank}'s" if rank.isdigit() else f"{rank}s"
                ge.wild_label = f"Queens and {plural} are wild"
            ge.last_up_was_queen = (rank_short == "Q")
    if ge.wild_label != prior_label:
        log.log(f"[WILD] Recomputed after correction: {ge.wild_label}")
        if ge.wild_label == "Queens wild":
            speech.say("Correction: only queens are now wild")
        else:
            tail = ge.wild_label.replace("Queens and ", "").replace(" are wild", "")
            speech.say(f"Correction: queens and {tail} are now wild")


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
        # Deal test mode
        self.deal_mode = None
        # Data collection mode
        self.collect_mode = None  # None or {"card_idx":0, "pass":1, "captured":False}
        # Console (dealer phone UI)
        self.game_engine = GameEngine()
        self.console_active_players = list(PLAYER_NAMES)  # who's playing tonight
        self.console_last_round_cards = []  # cards from last upcard scan
        self.console_hand_cards = []  # all confirmed up cards this hand: [{player, card, round}]
        self.console_up_round = 0     # current up-card round number
        self.console_total_up_rounds = 0  # total up-card rounds in this game
        self.console_scan_phase = "idle"  # "idle" | "watching" | "settling" | "scanned" | "confirmed"
        self.console_settle_time = 0.0
        # ---- Remote-player table view ("/table") ----
        # state_version is bumped whenever anything the observer needs changes.
        # Rodney's down cards come from the Pi scanner; other players only
        # expose a down-count + up-cards on the observer view.
        self.table_state_version = 0
        # Rodney's down-card slots. Indexed by scanner slot number so a
        # fluctuating or re-scanned slot replaces its prior value instead of
        # appending a new entry. Each value is {rank, suit, confidence}.
        self.rodney_downs = {}         # slot_num -> {rank, suit, confidence} (verified / auto-accepted)
        # 7/27: when Rodney has 2 down cards the UI asks him to pick one to
        # flip face-up. Once chosen, the card moves here and the LED for
        # that slot blinks so the dealer knows which to physically lift.
        self.rodney_flipped_up = None   # None or {rank, suit, slot}
        self.slot_pending = {}         # slot_num -> {rank, suit, confidence} (latest low-conf guess, awaiting confirm)
        self.slot_empty = {}           # slot_num -> True when poller sees no card
        self.verify_queue = []         # FIFO of slot_nums that need manual verify after /api/console/confirm
        self.pending_verify = None     # None or {guess, slot, prompt}
        self.table_log = []            # [{ts, msg}]
        self.pi_base_url = os.environ.get("PI_BASE_URL", "http://pokerbuddy.local:8080")
        # Tunables loaded from ~/.cardgame_host.json if present. Setup modal
        # writes them back when the user saves, so defaults only matter on
        # first run. pi_presence_threshold is a cached mirror of the Pi's
        # own persisted value — pushed to the Pi on save.
        cfg = _load_host_config()
        self.brio_settle_s = float(cfg.get("brio_settle_s", DEFAULT_BRIO_SETTLE_S))
        self.pi_presence_threshold = float(cfg.get("pi_presence_threshold", 140.0))
        self.pi_polling = False
        self.pi_poll_thread = None
        self.pi_prev_slots = {}        # slot_num -> last-seen card code (e.g. "Ac")
        # Slot-by-slot guided dealing state. None = not guiding; otherwise
        # {expecting: int, num_slots: int}. Regular _pi_poll_loop skips its
        # work while this is set.
        self.guided_deal = None
        self.guided_deal_thread = None
        # "Poker night" flag — set by Start, cleared by Exit Poker. The
        # console UI gates the game dropdown + action controls on this.
        self.night_active = False
        # High-level console state machine surfaced to the UI.
        # "idle" | "dealing" | "betting" | "hand_over"
        self.console_state = "idle"
        # 5 Card Draw / draw-phase support: Rodney marks cards during
        # betting (a set of slot numbers). When he hits "Request cards",
        # those slots' LEDs light up and guided flow refills them. One
        # draw per hand. betting_round distinguishes pre-draw vs post-draw
        # for games with two betting rounds.
        # Per-hand recognition stats: how many cards YOLO and Claude each
        # produced, and of those how many the user corrected. Reset on
        # every /api/console/deal and logged on /api/console/end.
        self.stats = {
            "yolo_right": 0, "yolo_wrong": 0,
            "claude_right": 0, "claude_wrong": 0,
        }
        self.rodney_marked_slots: set[int] = set()
        self.rodney_drew_this_hand = False
        # Count of completed draws this hand (3 Toed Pete has 3). Reset on
        # deal; incremented after each guided replace completes. Used to
        # index into the games list of DRAW phases for max-marks, and to
        # decide when to advance to hand_over instead of another draw.
        self.rodney_draws_done = 0
        self.console_betting_round = 0
        # Games with a trailing down card (7 Card Stud's 7th street, FTQ's
        # final down): after the last up round's Pot-is-right we run a second
        # guided session for that down slot. This flag, once set, means the
        # next Pot-is-right goes straight to hand_over instead of starting
        # trailing deal again.
        self.console_trailing_done = False
        self.table_lock = Lock()       # guards rodney_downs / pending_verify / table_log
        self.pi_confidence_threshold = 0.70  # >= this → auto-accept
        # The Pi's template matcher returns low-but-nonzero confidence for
        # every slot (including empty ones), so "empty" as a confidence
        # threshold doesn't work reliably. We trust the Pi's recognized
        # flag + any nonzero confidence as "something was seen" so the
        # weak-but-present scan still ends up in slot_pending and can be
        # manually verified.
        self.pi_empty_threshold = 0.0
        self._pi_last_logged = {}            # slot_num -> last logged code, throttle log spam
        self.pi_flash_held = False           # tracked so we don't spam hold/release
        self.folded_players = set()     # Rodney's view of who's folded this hand
        self.freezes = {}               # 7/27: player_name -> freezes in a row
        # True when Deal pinged the Pi and got no answer; stays set until the
        # next Deal so we skip hitting the Pi (flash/hold, /slots, LEDs, etc).
        self.pi_offline = False

_state = None


# ---------------------------------------------------------------------------
# Observer table view ("/table") — shared with Rodney via Teams
# ---------------------------------------------------------------------------

_CARD_NAME_RE = re.compile(
    r"^\s*(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)\s*$",
    re.IGNORECASE,
)
_RANK_CANON = {"ACE": "A", "KING": "K", "QUEEN": "Q", "JACK": "J"}
_SUIT_LETTER = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}


def _parse_card_any(text):
    """Parse either 'King of Hearts' or 'Kh' / '10s' into {rank, suit} or None."""
    if not text:
        return None
    text = str(text).strip()
    m = _CARD_NAME_RE.match(text)
    if m:
        rank = m.group(1).upper()
        rank = _RANK_CANON.get(rank, rank)
        return {"rank": rank, "suit": m.group(2).lower()}
    m = re.match(r"^(10|[2-9JQKA])([hdcs])$", text, re.IGNORECASE)
    if m:
        return {"rank": m.group(1).upper(), "suit": _SUIT_LETTER[m.group(2).lower()]}
    return None


_SUIT_LETTER_CODE = {"clubs": "c", "diamonds": "d", "hearts": "h", "spades": "s"}


def _best_hand_for_cards(cards, ge):
    """Given a list of card dicts ({rank, suit, ...}), compute the best
    poker hand using the current game's wild ranks and return
    {"label": ..., "codes": [...]} for the /table UI. codes are short
    card codes ("Ah", "10s") in best-hand order so the client can reorder
    its card row. Returns None if fewer than 2 cards or evaluation fails.
    """
    tuples = []
    code_by_id = {}
    for i, c in enumerate(cards):
        rank = c.get("rank")
        suit = c.get("suit")
        if not rank or not suit:
            continue
        tuples.append((rank, suit))
        code_by_id[i] = f"{rank}{_SUIT_LETTER_CODE.get(suit, (suit or '?')[0])}"
    if len(tuples) < 2:
        return None
    try:
        from poker_hands import best_hand
    except Exception:
        return None
    try:
        wilds = list(getattr(ge, "wild_ranks", []) or [])
        result = best_hand(tuples, wild_ranks=wilds)
    except Exception:
        return None
    codes = []
    for bc in result.cards:
        suit_letter = _SUIT_LETTER_CODE.get(bc.suit, (bc.suit or "?")[0])
        codes.append(f"{bc.rank}{suit_letter}")
    return {"label": result.label, "codes": codes, "category": result.category}


def _build_table_state(s):
    """Produce the JSON doc that /table/state returns.

    Rodney sees his hand in full. Every other player is just a down-count
    plus Brio up-card scans. The log is the tail of table_log.
    """
    ge = s.game_engine
    current_game = ge.current_game.name if ge.current_game else None

    # Accumulate up-card history by player from console_hand_cards (populated
    # when the dealer confirms each up-card round). Fall back to the latest
    # zone scan for any player missing from history — useful between rounds
    # before the dealer has hit Confirm.
    up_by_player = {}
    for entry in s.console_hand_cards:
        name = entry.get("player")
        parsed = _parse_card_any(entry.get("card", ""))
        if name and parsed:
            up_by_player.setdefault(name, []).append(
                {"rank": parsed["rank"], "suit": parsed["suit"], "round": entry.get("round")}
            )

    players = []
    for p in ge.players:
        if p.name not in s.console_active_players:
            continue
        up_cards = list(up_by_player.get(p.name, []))
        if not up_cards and s.monitor:
            # Only show latest scan if we don't already have history for this player.
            latest_txt = s.monitor.last_card.get(p.name, "")
            latest_parsed = _parse_card_any(latest_txt)
            if latest_parsed:
                details = s.monitor.recognition_details.get(p.name, {})
                conf = details.get("yolo_conf")
                cur = {"rank": latest_parsed["rank"], "suit": latest_parsed["suit"]}
                if conf is not None:
                    cur["confidence"] = round(float(conf), 2)
                up_cards.append(cur)

        freezes_n = s.freezes.get(p.name, 0)
        entry = {
            "name": p.name,
            "position": p.position,
            "is_dealer": p.is_dealer,
            "is_remote": p.is_remote,
            "folded": p.name in s.folded_players,
            "freezes": freezes_n,
            "frozen": freezes_n >= 3,
        }
        if p.is_remote:
            # Rodney's hand = only down-card slots that have been recognized
            # and validated (rodney_downs). Tentative slot_pending guesses
            # are shown in the verify modal instead, not as cards in hand.
            # If Rodney flipped one of his downs face-up (7/27 2-down), the
            # card remains in rodney_downs (so Pi counting still works) but
            # we render it here as an up-card instead of a down.
            flipped_slot = (s.rodney_flipped_up or {}).get("slot")
            hand = []
            for slot_num in sorted(s.rodney_downs.keys()):
                if slot_num == flipped_slot:
                    continue
                d = s.rodney_downs[slot_num]
                hand.append({"type": "down", "rank": d["rank"],
                             "suit": d["suit"], "slot": slot_num,
                             "confidence": d.get("confidence")})
            if s.rodney_flipped_up:
                fu = s.rodney_flipped_up
                # Don't duplicate once Brio picks it up via a zone scan.
                already = any(
                    c.get("rank") == fu["rank"] and c.get("suit") == fu["suit"]
                    for c in up_cards
                )
                if not already:
                    hand.append({"type": "up", "rank": fu["rank"], "suit": fu["suit"]})
            for c in up_cards:
                hand.append({"type": "up", **c})
            entry["hand"] = hand
            entry["best_hand"] = _best_hand_for_cards(hand, ge)
        else:
            # Dealer deals the same card-type to every player in each round,
            # so every non-folded player holds as many downs as Rodney has
            # validated. In 7/27 (2-down) once Rodney has flipped, every
            # local player has also flipped one of their two — subtract the
            # flipped card from the visible down-count.
            down_count = len(s.rodney_downs)
            if s.rodney_flipped_up:
                down_count = max(0, down_count - 1)
            entry["down_count"] = down_count
            entry["up_cards"] = up_cards
            entry["best_hand"] = _best_hand_for_cards(up_cards, ge)
        players.append(entry)

    # Console flow doesn't advance game_engine.phase_index, so derive the
    # round counter from console_up_round (confirmed up rounds) + down cards
    # Rodney has actually received — including pending scans so a yet-to-be-
    # verified card still advances the counter.
    active_down_slots = set(s.rodney_downs.keys()) | set(s.slot_pending.keys())
    current_round = s.console_up_round + len(active_down_slots)
    total_rounds = _total_card_rounds(ge)
    # Open-ended games (e.g. 7/27) report total=0 so the UI drops "of N".
    if ge.current_game is not None:
        has_hit_round = any(
            ph.type.value == "hit_round" and ph.card_type == "up"
            for ph in ge.current_game.phases
        )
        if has_hit_round:
            total_rounds = 0

    # 7/27: once Rodney has 2 down cards scanned he needs to pick one to
    # flip face-up (standard 7/27 start). The /table modal consumes this
    # field. It stays None if the dealer already flipped one himself.
    flip_choice = None
    # Flip choice only applies to the 2-down variant ("7/27" proper, not
    # "7/27 (one up)" where the dealer deals a face-up directly). The
    # len==2 check naturally gates this.
    if ge.current_game is not None and ge.current_game.name == "7/27":
        if s.rodney_flipped_up is None and len(s.rodney_downs) == 2:
            downs_sorted = sorted(s.rodney_downs.items())
            flip_choice = {
                "prompt": "Pick a card to turn face-up",
                "options": [
                    {"slot": sn, "rank": d["rank"], "suit": d["suit"]}
                    for sn, d in downs_sorted
                ],
            }

    # 7/27: compute hand values per player. Rodney's value includes both
    # his up and down cards; everyone else's is up-cards-only (we don't
    # know their downs).
    is_7_27 = ge.current_game is not None and ge.current_game.name.startswith("7/27")
    if is_7_27:
        RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
        up_by_player_cards = {}
        for entry in s.console_hand_cards:
            parts = entry.get("card", "").split(" of ")
            if len(parts) != 2:
                continue
            rank_full, suit_full = parts[0].strip(), parts[1].strip().lower()
            rank = RANK_SHORT.get(rank_full, rank_full)
            up_by_player_cards.setdefault(entry["player"], []).append((rank, suit_full))
        remote_name = next((p.name for p in ge.players if p.is_remote), None)
        flipped_slot = (s.rodney_flipped_up or {}).get("slot")
        values_by_player = {}
        for p in ge.players:
            pairs = list(up_by_player_cards.get(p.name, []))
            if p.name == remote_name:
                # Skip the flipped slot — its already counted via the
                # up_by_player_cards entry fed in by /api/console/confirm
                # (flip_up sets monitor.last_card[Rodney] to that card).
                for slot_num, d in s.rodney_downs.items():
                    if slot_num == flipped_slot:
                        continue
                    if d.get("rank") and d.get("suit"):
                        pairs.append((d["rank"], d["suit"]))
            if pairs:
                values_by_player[p.name] = _compute_7_27_values(pairs)
        # Write onto each player entry so the UI can render.
        for entry in players:
            vals = values_by_player.get(entry["name"])
            if vals:
                entry["values_7_27"] = vals
    return {
        "version": s.table_state_version,
        "viewer": next((p.name for p in ge.players if p.is_remote), "Rodney"),
        "game": {
            "name": current_game or "",
            "round": getattr(ge, "draw_round", 0),
            "wild_label": ge.wild_label or "",
            "wild_ranks": list(getattr(ge, "wild_ranks", []) or []),
            "current_round": current_round,
            "total_rounds": total_rounds,
            "state": getattr(ge.state, "value", str(ge.state)),
        },
        "dealer": ge.get_dealer().name,
        "current_player": None,
        "players": players,
        "log": list(s.table_log[-30:]),
        "pending_verify": s.pending_verify,
        "flip_choice": flip_choice,
        "guided_deal": (
            dict(s.guided_deal) if s.guided_deal is not None else None
        ),
        "draw": {
            # Multi-draw games (3 Toed Pete): rodney_draws_done counts how
            # many draws are behind us; we can mark and request again as
            # long as more DRAW phases remain and the current draw has not
            # been taken yet.
            "can_mark": (
                _game_has_draw_phase(ge)
                and s.rodney_draws_done < _total_draw_phases(ge)
                and not s.rodney_drew_this_hand
                and s.console_state in ("betting", "draw")
            ),
            "can_request": (
                s.console_state == "draw"
                and s.rodney_draws_done < _total_draw_phases(ge)
                and not s.rodney_drew_this_hand
            ),
            "max_marks": _max_draw_for_game(ge, s.rodney_draws_done),
            "marked_slots": sorted(s.rodney_marked_slots),
            "drew_this_hand": s.rodney_drew_this_hand,
            "draws_done": s.rodney_draws_done,
            "total_draws": _total_draw_phases(ge),
        },
    }


def _table_state_bump(s):
    """Call when something observable changes so polling clients re-render."""
    s.table_state_version += 1


def _dealing_phase_types():
    """Tuple of phase type values that contribute a round per card."""
    from game_engine import PhaseType
    return (PhaseType.DEAL, PhaseType.COMMUNITY)


def _total_card_rounds(ge):
    """Total cards in the game's deal + community phases (per player).

    Follow the Queen and 7-Card Stud = 7; Texas Hold'em = 2 hole + 5 community = 7.
    Returns 0 when no game is active.
    """
    if ge.current_game is None:
        return 0
    allowed = _dealing_phase_types()
    total = 0
    for ph in ge.current_game.phases:
        if ph.type in allowed:
            total += len(ph.pattern)
    return total


def _cards_dealt_so_far(ge):
    """Zero-based count of cards already dealt in the current game."""
    if ge.current_game is None:
        return 0
    allowed = _dealing_phase_types()
    completed = 0
    for i, ph in enumerate(ge.current_game.phases):
        if i < ge.phase_index:
            if ph.type in allowed:
                completed += len(ph.pattern)
        elif i == ge.phase_index:
            if ph.type in allowed:
                completed += ge.card_in_phase
            break
    return completed


def _next_deal_position_type(s):
    """Returns 'down', 'up', or None for the next card about to be dealt.

    Walks the combined DEAL/COMMUNITY pattern of the current game, consuming
    one up-card-per-up-round-confirmed and one down-per-rodney_down, and
    returns the type of the next un-dealt position.
    """
    ge = s.game_engine
    if ge.current_game is None:
        return None
    pattern = []
    for ph in ge.current_game.phases:
        if ph.type in _dealing_phase_types():
            pattern.extend(ph.pattern)
    dealt_ups = s.console_up_round
    dealt_downs = len(s.rodney_downs)
    for pos in pattern:
        if pos == "up":
            if dealt_ups > 0:
                dealt_ups -= 1
                continue
            return "up"
        else:
            if dealt_downs > 0:
                dealt_downs -= 1
                continue
            return "down"
    return None


def _skip_inactive_dealer(s):
    """Rotate past any unchecked player so the dealer is always an active
    seat. Caps the loop at one full rotation to guarantee termination if
    every player ends up inactive."""
    ge = s.game_engine
    if not s.console_active_players:
        return
    for _ in range(len(ge.players)):
        if ge.get_dealer().name in s.console_active_players:
            return
        ge.advance_dealer()


def _total_downs_in_pattern(ge):
    """Total number of down cards in the current game's deal pattern.

    Each game has a fixed number of down cards per player regardless of
    where they appear in the deal order (FTQ = 3, 7-Card Stud = 3,
    Hold'em = 2, 5-Card Draw = 5). The scanner box only needs to monitor
    that many slots.
    """
    if ge.current_game is None:
        return 0
    allowed = _dealing_phase_types()
    n = 0
    for ph in ge.current_game.phases:
        if ph.type in allowed:
            n += sum(1 for t in ph.pattern if t == "down")
    return n


def _initial_down_count(ge):
    """Number of down cards in the FIRST deal phase's pattern.

    7-Card Stud's ['down','down','up'] → 2; Hold'em's ['down','down'] → 2;
    5 Card Draw's ['down']*5 → 5; Follow the Queen → 3. These are the slots
    the scanner box guides through at Deal-time. Any remaining downs
    (7CS/FTQ 7th street) are handled as a trailing guided session after
    the final up round.
    """
    if ge.current_game is None:
        return 0
    allowed = _dealing_phase_types()
    for ph in ge.current_game.phases:
        if ph.type in allowed:
            return sum(1 for t in ph.pattern if t == "down")
    return 0


def _trailing_down_slots(ge):
    """Slot numbers for down cards beyond the initial deal phase.

    For 7CS (3 total downs, 2 initial) → [3]. For FTQ similarly → [4].
    For 5CD / Hold'em → [] (no trailing downs). These slots are guided
    after the final up round's Pot-is-right.
    """
    total = _total_downs_in_pattern(ge)
    initial = _initial_down_count(ge)
    if total <= initial:
        return []
    return list(range(initial + 1, total + 1))


def _table_log_add(s, msg):
    s.table_log.append({"ts": int(time.time()), "msg": msg})
    if len(s.table_log) > 200:
        s.table_log = s.table_log[-100:]


def _parse_card_code(code):
    """Parse 'Ac' or '10h' into {rank, suit} or None."""
    if not code:
        return None
    m = re.match(r"^\s*(10|[2-9JQKA])([hdcs])\s*$", code, re.IGNORECASE)
    if not m:
        return None
    return {"rank": m.group(1).upper(), "suit": _SUIT_LETTER[m.group(2).lower()]}


_SIM_RANKS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
_SIM_SUITS = ["clubs","diamonds","hearts","spades"]


def _promote_next_verify(s) -> bool:
    """If no modal is open and a queued slot has a pending guess, open it.

    Returns True if state changed (pending_verify was set). Caller owns the
    table_lock.
    """
    if s.pending_verify is not None or not s.verify_queue:
        return False
    for slot_num in list(s.verify_queue):
        guess = s.slot_pending.get(slot_num)
        if not guess:
            continue
        s.pending_verify = {
            "slot": slot_num,
            "guess": dict(guess),
            "prompt": (
                f"Slot {slot_num} needs verification. "
                f"Remove the card, hold it up for Rodney, "
                f"then confirm or override."
            ),
            "image_url": (
                None if s.pi_offline
                else f"/api/table/slot_image/{slot_num}"
            ),
        }
        _table_log_add(s, f"Slot {slot_num}: modal opened for verify")
        return True
    return False


def _simulate_offline_slot_scans(s):
    """Fill rodney_downs' expected slots with random low-confidence guesses
    so a hand can be played end-to-end without the Pi. Each missing slot
    (not in rodney_downs, not in slot_pending) gets one random card at
    conf=0.20 — low enough to queue a verify modal on Confirm Cards where
    Rodney can override with the actual card.
    """
    max_slot = _total_downs_in_pattern(s.game_engine)
    if max_slot <= 0:
        return
    with s.table_lock:
        added = []
        for n in range(1, max_slot + 1):
            if n in s.rodney_downs or n in s.slot_pending:
                continue
            rank = random.choice(_SIM_RANKS)
            suit = random.choice(_SIM_SUITS)
            s.slot_pending[n] = {"rank": rank, "suit": suit, "confidence": 0.20}
            added.append((n, f"{rank}{suit[0]}"))
        if added:
            for (n, code) in added:
                _table_log_add(s, f"Slot {n}: simulated {code} (Pi offline, needs verify)")
            s.table_state_version += 1


def _pi_ping(s, timeout_s: float = 1.5) -> bool:
    """One quick GET /ping to test Pi reachability. True on success."""
    import urllib.request
    try:
        url = f"{s.pi_base_url.rstrip('/')}/ping"
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            resp.read()
        return True
    except Exception:
        return False


def _pi_fetch_slots(s):
    """Fetch /slots from the Pi, limiting to the slots our game uses.

    Passes max_slot so the Pi skips capturing + matching the unused ones.
    Returns the parsed dict or None on error. If s.pi_offline is set (Deal
    determined the Pi was unreachable) this returns None without making a
    network call, so simulation kicks in immediately.
    """
    if s.pi_offline:
        return None
    import urllib.request
    max_slot = _total_downs_in_pattern(s.game_engine)
    if max_slot <= 0:
        return {"slots": []}
    try:
        url = f"{s.pi_base_url.rstrip('/')}/slots?max_slot={max_slot}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.log(f"[PI] /slots error: {e}")
        return None


def _pi_flash(s, hold):
    """Hold or release the Pi's flash LEDs. Tracks state to avoid redundant calls."""
    if s.pi_offline:
        return
    if s.pi_flash_held == hold:
        return
    import urllib.request
    path = "/flash/hold" if hold else "/flash/release"
    try:
        url = f"{s.pi_base_url.rstrip('/')}{path}"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=3) as resp:
            resp.read()
        s.pi_flash_held = hold
        log.log(f"[PI] flash {'held' if hold else 'released'}")
    except Exception as e:
        # Bare repr(e) catches empty-string errors (URLError / HTTPError).
        detail = repr(e) if not str(e) else str(e)
        log.log(f"[PI] {path} error: {type(e).__name__}: {detail}")


def _update_flash_for_deal_state(s):
    """Hold LEDs while a down card is the next expected deal; release otherwise."""
    nxt = _next_deal_position_type(s)
    _pi_flash(s, nxt == "down")


def _pi_poll_loop(s):
    """Background poll: map Pi scanner detections into rodney_downs.

    - Each /slots result is compared against s.pi_prev_slots.
    - A new card (slot was empty or held a different card) becomes:
        * rodney_downs[slot] = card when confidence >= threshold, OR
        * a pending_verify prompt otherwise (poller stops advancing that
          slot until the verify modal is resolved).
    - Slots that go empty clear their rodney_downs entry too.
    """
    log.log("[PI] poll loop started")
    offline_streak = 0
    while s.pi_polling:
        # Guided dealing takes exclusive ownership of the scanner for
        # all-down games. The regular poller idles until guided completes.
        if s.guided_deal is not None:
            time.sleep(0.5)
            continue
        # Only hit the Pi when we're actually expecting a down card to be
        # dealt. Gate on the deal pattern directly (not pi_flash_held) so a
        # failed /flash/hold call doesn't also stop the scan polling.
        _update_flash_for_deal_state(s)
        if _next_deal_position_type(s) != "down":
            time.sleep(2.0)
            continue
        doc = _pi_fetch_slots(s)
        if doc is None:
            offline_streak += 1
            # After two failed fetches, assume the Pi isn't running and
            # simulate slot scans so gameplay is testable without hardware.
            # Each expected slot gets a random low-confidence guess the user
            # can override in the verify modal. Also promote any queued
            # verify into pending_verify so the modal actually opens in
            # offline mode.
            if offline_streak >= 2 or s.pi_offline:
                _simulate_offline_slot_scans(s)
                with s.table_lock:
                    if _promote_next_verify(s):
                        s.table_state_version += 1
            time.sleep(2.0)
            continue
        offline_streak = 0
        # Only scan slots the current game actually uses (FTQ=3, Hold'em=2).
        max_slot = _total_downs_in_pattern(s.game_engine)
        with s.table_lock:
            changed = False
            for entry in doc.get("slots", []):
                slot_num = entry.get("slot")
                if slot_num is None:
                    continue
                if slot_num > max_slot:
                    if slot_num in s.rodney_downs:
                        s.rodney_downs.pop(slot_num, None)
                        changed = True
                    s.pi_prev_slots.pop(slot_num, None)
                    s.slot_pending.pop(slot_num, None)
                    s.slot_empty[slot_num] = True
                    continue

                recognized = entry.get("recognized")
                rank = entry.get("rank")
                suit = entry.get("suit")
                conf = float(entry.get("confidence", 0.0))
                is_empty = (not recognized) or conf < s.pi_empty_threshold

                if is_empty:
                    # Slot physically empty: mark empty but keep slot_pending
                    # intact — the last-seen guess survives removal so the
                    # verify modal can still fire after a Confirm Cards.
                    # If there's a weak scan below threshold log it once so
                    # the user can see that the scanner IS seeing something
                    # but deciding it's noise.
                    if recognized and rank and suit and conf > 0:
                        weak_code = f"{rank}{suit[0]} ({int(conf*100)}%)"
                        if s._pi_last_logged.get(slot_num) != weak_code:
                            log.log(f"[PI] Slot {slot_num}: weak {weak_code} below empty threshold")
                            s._pi_last_logged[slot_num] = weak_code
                    elif s._pi_last_logged.get(slot_num) is not None:
                        s._pi_last_logged[slot_num] = None
                    s.slot_empty[slot_num] = True
                    s.pi_prev_slots.pop(slot_num, None)
                    continue

                # Non-empty: remember what's there now.
                s.slot_empty[slot_num] = False
                code = f"{rank}{suit[0]}" if rank and suit else ""

                if conf >= s.pi_confidence_threshold:
                    prev = s.pi_prev_slots.get(slot_num)
                    if prev == code:
                        continue
                    s.rodney_downs[slot_num] = {
                        "rank": rank, "suit": suit, "confidence": round(conf, 2),
                    }
                    s.slot_pending.pop(slot_num, None)
                    if slot_num in s.verify_queue:
                        s.verify_queue.remove(slot_num)
                    s.pi_prev_slots[slot_num] = code
                    _table_log_add(s, f"Slot {slot_num}: {code} (auto, {int(conf*100)}%)")
                    changed = True
                else:
                    # Medium confidence: hold as the latest guess but don't
                    # surface a modal until the dealer runs /api/console/confirm
                    # and the user subsequently removes the card from the slot.
                    guess = {"rank": rank, "suit": suit, "confidence": round(conf, 2)}
                    if s.slot_pending.get(slot_num) != guess:
                        s.slot_pending[slot_num] = guess
                        changed = True

            if _promote_next_verify(s):
                changed = True
            if changed:
                s.table_state_version += 1
        time.sleep(1.0)
    log.log("[PI] poll loop stopped")


def _pi_poll_start(s):
    if s.pi_polling:
        return
    s.pi_polling = True
    s.pi_prev_slots = {}
    t = Thread(target=_pi_poll_loop, args=(s,), daemon=True)
    s.pi_poll_thread = t
    t.start()


def _pi_poll_stop(s):
    s.pi_polling = False
    # Don't join — daemon thread will exit on its own


# ---------------------------------------------------------------------------
# Guided dealing for all-down games (5 Card Draw, 3 Toed Pete, etc.)
# ---------------------------------------------------------------------------

GUIDED_GOOD_CONF = 0.50   # at/above this, auto-accept the scan
GUIDED_POLL_S = 0.6       # interval between /slots/<n>/scan polls
# Require this many consecutive present=true scans before firing a verify
# modal. The first presence hit is often a finger or a half-inserted card;
# YOLO can't see it yet. A high-confidence scan short-circuits this wait.
GUIDED_STABLE_SCANS = 3
# After first detecting a card in the slot, wait this long before using any
# scan reading. Gives the dealer time to fully seat the card so YOLO isn't
# fighting motion blur on the first capture. Longer than strictly needed
# to give the dealer time to finish placing the card before YOLO reads —
# shorter values led to verify modals popping with partial-insertion reads.
GUIDED_SETTLE_S = 2.0

# Default seconds to wait after the overhead (Brio) camera trips a motion
# event in the dealer zone before firing the whole-table scan. Runtime-
# configurable via the Setup modal → persisted in the host config file.
DEFAULT_BRIO_SETTLE_S = 0.7

HOST_CONFIG_PATH = Path.home() / ".cardgame_host.json"


def _load_host_config() -> dict:
    """Read persisted host tunables. Returns {} if file is missing/bad."""
    try:
        return json.loads(HOST_CONFIG_PATH.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.log(f"[CONFIG] Could not read {HOST_CONFIG_PATH}: {e}")
        return {}


def _save_host_config(updates: dict) -> None:
    """Merge updates into the persisted host config and write back to disk."""
    cfg = _load_host_config()
    cfg.update(updates)
    try:
        HOST_CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")
    except OSError as e:
        log.log(f"[CONFIG] Could not write {HOST_CONFIG_PATH}: {e}")


def _pi_slot_led(s, slot_num: int, state: str):
    """POST /slots/<n>/led with state = on | off | blink."""
    if s.pi_offline:
        return
    import urllib.request
    url = f"{s.pi_base_url.rstrip('/')}/slots/{slot_num}/led"
    body = json.dumps({"state": state}).encode()
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2).read()
    except Exception as e:
        log.log(f"[PI] LED slot {slot_num} {state} failed: "
                f"{type(e).__name__}: {e}")


def _pi_slot_scan(s, slot_num: int):
    """POST /slots/<n>/scan — returns dict (present/card/...) or None on error."""
    if s.pi_offline:
        return None
    import urllib.request
    url = f"{s.pi_base_url.rstrip('/')}/slots/{slot_num}/scan"
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.log(f"[PI] slot_scan {slot_num} failed: "
                f"{type(e).__name__}: {e}")
        return None


def _guided_deal_loop(s):
    """Slot-by-slot dealing for all-down games.

    Turns slot-1 LED on, polls /slots/1/scan until a card is present:
    if YOLO conf >= GUIDED_GOOD_CONF, record the card + advance; if lower,
    blink the LED and open the /table verify modal for Rodney to resolve.
    Strict 1→N order: never looks at slot N+1 until slot N is resolved.

    External code stops the loop by setting s.guided_deal = None.
    """
    gd = s.guided_deal
    if gd is None:
        return
    N = gd["num_slots"]
    log.log(f"[GUIDED] Started — {N} slots")
    # Initial LED state: slot 1 solid on, the rest off.
    for n in range(1, N + 1):
        _pi_slot_led(s, n, "on" if n == 1 else "off")

    # Per-slot debounce state: how many consecutive present=true scans
    # we've seen, and the best card guess from any of them. Reset on
    # either present=false (card/finger withdrawn) or after we commit.
    stable_count = 0
    best_card = None
    settled = False  # True after GUIDED_SETTLE_S has elapsed since first present

    while True:
        gd = s.guided_deal
        if gd is None:
            log.log("[GUIDED] Stopped externally")
            return
        expecting = gd["expecting"]
        if expecting > N:
            for n in range(1, N + 1):
                _pi_slot_led(s, n, "off")
            log.log(f"[GUIDED] Complete — {N} slots filled")
            with s.table_lock:
                s.guided_deal = None
                s.table_state_version += 1
            # Per-game post-guided transitions.
            ge = s.game_engine
            first_deal_phase = next(
                (ph for ph in (ge.current_game.phases if ge.current_game else [])
                 if ph.type.value in ("deal", "community")),
                None,
            )
            first_phase_has_up = bool(
                first_deal_phase and "up" in first_deal_phase.pattern
            )
            has_hit_round_game = bool(
                ge.current_game and any(
                    ph.type.value == "hit_round"
                    for ph in ge.current_game.phases
                )
            )
            if (s.console_state == "dealing"
                    and s.console_total_up_rounds == 0
                    and not first_phase_has_up
                    and not has_hit_round_game):
                # Truly all-down games (5CD, 3 Toed Pete): every card is in,
                # auto-advance to betting so next action is Pot is right.
                s.console_state = "betting"
                if s.console_betting_round == 0:
                    s.console_betting_round = 1
                log.log(
                    f"[CONSOLE] All-down deal complete → "
                    f"betting round {s.console_betting_round}"
                )
            elif s.console_state == "dealing" and has_hit_round_game:
                # 7/27 (either variant): the local players still need to
                # flip/reveal an up card onto the table for Brio to scan.
                # Keep state in "dealing" with Brio watching; user presses
                # Confirm Cards once every player's up card is in.
                s.monitoring = True
                s.console_scan_phase = "watching"
                s._zones_with_motion = set()
                dname = ge.get_dealer().name if ge.current_game else "dealer"
                log.log(
                    f"[CONSOLE] 7/27 guided downs done → Brio watching "
                    f"{dname}'s zone for flipped-up cards"
                )
            elif s.console_state == "replacing":
                # Draw-phase refill just completed — back to the current
                # betting round (round 1 still — dealer will Pot-is-right
                # to advance to round 2 once everyone has drawn).
                s.console_state = "betting"
                log.log("[CONSOLE] Draw replacement complete → betting")
            elif (s.console_state == "dealing"
                  and s.console_total_up_rounds > 0
                  and s.console_scan_phase == "idle"):
                # Mixed game (7CS, Hold'em, FTQ): initial downs are in,
                # now hand off to Brio to watch for the up card(s). The
                # baselines captured at Deal-time are still valid — the
                # table had no up cards then and still has none now (local
                # down cards are kept in hand, not placed in up-zones).
                ge = s.game_engine
                s.monitoring = True
                s.console_scan_phase = "watching"
                s._zones_with_motion = set()
                dname = ge.get_dealer().name if ge.current_game else "dealer"
                log.log(f"[CONSOLE] Guided downs done → Brio watching {dname}'s zone")
            return

        # Was this slot's verify modal just resolved? Advance if so.
        with s.table_lock:
            already_filled = expecting in s.rodney_downs
            pv = s.pending_verify
            waiting_verify = pv is not None and pv.get("slot") == expecting

        if already_filled:
            _pi_slot_led(s, expecting, "off")
            with s.table_lock:
                gd["expecting"] = expecting + 1
                s.table_state_version += 1
            if expecting + 1 <= N:
                _pi_slot_led(s, expecting + 1, "on")
            stable_count = 0
            best_card = None
            settled = False
            continue

        if waiting_verify:
            time.sleep(0.3)
            continue

        result = _pi_slot_scan(s, expecting)
        if result is None:
            time.sleep(1.5)  # Pi unreachable — back off before retrying
            continue

        if not result.get("present"):
            stable_count = 0
            best_card = None
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        # First scan that sees "present" — the card may still be sliding
        # into place. Wait GUIDED_SETTLE_S before trusting any reading so
        # YOLO isn't hitting motion blur or a half-inserted card.
        if not settled:
            time.sleep(GUIDED_SETTLE_S)
            settled = True
            continue

        stable_count += 1
        card = result.get("card")
        if card:
            conf = float(card.get("confidence", 0.0))
            # Track the best (highest-conf) guess across debounce scans.
            if best_card is None or conf > float(best_card.get("confidence", 0.0)):
                best_card = card
            code = f"{card['rank']}{card['suit'][0]}"
            # Short-circuit: any scan above the auto-accept threshold commits
            # immediately, no need to wait for more stability.
            if conf >= GUIDED_GOOD_CONF:
                with s.table_lock:
                    s.rodney_downs[expecting] = {
                        "rank": card["rank"],
                        "suit": card["suit"],
                        "confidence": round(conf, 2),
                    }
                    s.pi_prev_slots[expecting] = code
                    gd["expecting"] = expecting + 1
                    s.table_state_version += 1
                _table_log_add(s, f"Slot {expecting}: {code} (auto, {int(conf*100)}%)")
                _pi_slot_led(s, expecting, "off")
                if expecting + 1 <= N:
                    _pi_slot_led(s, expecting + 1, "on")
                stable_count = 0
                best_card = None
                settled = False
                continue

        # Low-conf or no-card: give the card time to settle before popping
        # the verify modal. Early "present" ticks from a finger or a half-
        # inserted card would otherwise fire the modal with empty fields.
        if stable_count < GUIDED_STABLE_SCANS:
            time.sleep(GUIDED_POLL_S)
            continue

        # Debounce window elapsed without a high-confidence read. If YOLO
        # never recognized anything at all across the whole window, the
        # scanner is probably misreporting present=true for an empty slot
        # (e.g., brightness threshold too high). Don't open a modal with
        # an empty guess — log once, reset state, and keep polling.
        if best_card is None:
            log.log(
                f"[GUIDED] Slot {expecting}: present but nothing recognized "
                f"after {GUIDED_STABLE_SCANS} scans — Pi presence threshold "
                f"may be too high; continuing to poll"
            )
            stable_count = 0
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        conf = float(best_card.get("confidence", 0.0))
        guess = {
            "rank": best_card["rank"],
            "suit": best_card["suit"],
            "confidence": round(conf, 2),
        }
        prompt = (
            f"Slot {expecting}: low confidence ({int(conf*100)}%). "
            f"Confirm or correct."
        )

        with s.table_lock:
            s.pending_verify = {
                "slot": expecting,
                "guess": guess,
                "prompt": prompt,
                "image_url": f"/api/table/slot_image/{expecting}",
            }
            if guess["rank"]:
                s.slot_pending[expecting] = dict(guess)
            s.table_state_version += 1
        _table_log_add(
            s,
            f"Slot {expecting}: verify needed"
            + (f" ({int(guess['confidence']*100)}%)" if guess["rank"] else ""),
        )
        _pi_slot_led(s, expecting, "blink")


def _start_guided_deal(s, num_slots: int):
    """Kick off the guided deal thread. Safe to call again; becomes no-op
    if a guided deal is already running."""
    if s.guided_deal is not None:
        return
    with s.table_lock:
        s.guided_deal = {"expecting": 1, "num_slots": num_slots}
        s.table_state_version += 1
    # Hold the flash on for the whole guided session so every /slots/<n>/scan
    # runs under steady lighting without the 300ms warmup each pulse. The
    # Pi's capture_with_flash short-circuits its own warmup when flash.held.
    _pi_flash(s, True)
    t = Thread(target=_guided_deal_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _stop_guided_deal(s):
    """Signal the guided loop to exit and clear LEDs."""
    gd = s.guided_deal
    if gd is None:
        return
    # The guided_deal dict carries either "num_slots" (full deal) or "slots"
    # (explicit list, used by draw-phase replacement). Turn everything off.
    slots = gd.get("slots") or list(range(1, gd.get("num_slots", 0) + 1))
    with s.table_lock:
        s.guided_deal = None
        s.table_state_version += 1
    for n in slots:
        _pi_slot_led(s, n, "off")
    # Release the held flash; regular poller / idle state will manage it.
    _pi_flash(s, False)


def _start_guided_replace(s, slots, previous_cards=None):
    """Kick off the guided loop for a specific slot list (draw-phase
    replacement). Same as _start_guided_deal but iterates through the
    supplied slot numbers in order rather than 1..N.

    previous_cards maps slot_num → code string ("Ah") for the card that
    was in each slot before the replace started. The loop uses it to
    detect "card changed" in case the Pi scan misses the present=false
    moment between the swap."""
    if s.guided_deal is not None:
        return
    ordered = [int(x) for x in slots if isinstance(x, int) or str(x).isdigit()]
    ordered = sorted(set(ordered))
    if not ordered:
        return
    prev = {int(k): str(v) for k, v in (previous_cards or {}).items()}
    with s.table_lock:
        s.guided_deal = {
            "slots": ordered, "index": 0, "mode": "replace",
            "previous_cards": prev,
        }
        s.console_state = "replacing"
        s.table_state_version += 1
    _pi_flash(s, True)
    t = Thread(target=_guided_replace_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _start_guided_trailing_deal(s, slots: list[int]):
    """Kick off guided flow for trailing down cards (7 Card Stud's 7th
    street, Follow the Queen's 7th). Console stays in 'dealing' until the
    loop finishes, then transitions to 'betting' for one final Pot-is-right
    before hand_over."""
    if s.guided_deal is not None:
        return
    ordered = sorted(set(int(x) for x in slots if isinstance(x, int) or str(x).isdigit()))
    if not ordered:
        return
    with s.table_lock:
        s.guided_deal = {"slots": ordered, "index": 0, "mode": "trailing"}
        s.console_state = "dealing"
        s.table_state_version += 1
    _pi_flash(s, True)
    t = Thread(target=_guided_replace_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _guided_replace_loop(s):
    """Variant of _guided_deal_loop driven by an explicit slot list.
    LEDs light in the order given; cleared rodney_downs entries refill
    as new cards are scanned.

    Shared by draw-phase replacement (mode='replace') and the trailing-down
    deal for stud games (mode='trailing'); only the completion transition
    differs."""
    gd = s.guided_deal
    if gd is None or "slots" not in gd:
        return
    slots = list(gd["slots"])
    mode = gd.get("mode", "replace")
    log.log(f"[GUIDED/{mode}] Started — slots {slots}")
    # Strict single-slot: only the slot being processed right now has its
    # LED lit, every other slot is off. Previously upcoming slots blinked,
    # which looked like we were trying to process them in parallel.
    for i, n in enumerate(slots):
        _pi_slot_led(s, n, "on" if i == 0 else "off")
    stable_count = 0
    best_card = None
    settled = False
    # In replace mode the old card may still be physically in the slot
    # when guided starts — require a present=false transition before we
    # accept a present=true reading, otherwise the old card gets re-
    # committed as "new". Trailing mode starts from an empty slot.
    require_empty_first = (mode == "replace")
    saw_empty = not require_empty_first

    while True:
        gd = s.guided_deal
        if gd is None:
            log.log(f"[GUIDED/{mode}] Stopped externally")
            return
        idx = gd.get("index", 0)
        if idx >= len(slots):
            for n in slots:
                _pi_slot_led(s, n, "off")
            log.log(f"[GUIDED/{mode}] Complete")
            with s.table_lock:
                s.guided_deal = None
                s.table_state_version += 1
                if mode == "trailing":
                    # 7CS/FTQ 7th street done — one more betting round before
                    # hand_over. console_trailing_done tells next_round to
                    # skip the trailing branch second time through.
                    s.console_state = "betting"
                    s.console_trailing_done = True
                    log.log(
                        "[CONSOLE] Trailing down deal complete → final betting"
                    )
                elif s.console_state == "replacing":
                    # Record this draw as done. Multi-draw games (3 Toed
                    # Pete) use this to know whether another DRAW phase
                    # follows; post-draw betting round number is the
                    # count of draws completed so far + 1.
                    s.rodney_draws_done += 1
                    s.console_state = "betting"
                    s.console_betting_round = s.rodney_draws_done + 1
                    log.log(
                        f"[CONSOLE] Draw {s.rodney_draws_done} replacement "
                        f"done → betting round {s.console_betting_round}"
                    )
            return
        expecting = slots[idx]

        with s.table_lock:
            already_filled = expecting in s.rodney_downs
            pv = s.pending_verify
            waiting_verify = pv is not None and pv.get("slot") == expecting

        if already_filled:
            _pi_slot_led(s, expecting, "off")
            with s.table_lock:
                gd["index"] = idx + 1
                s.table_state_version += 1
            if idx + 1 < len(slots):
                _pi_slot_led(s, slots[idx + 1], "on")
            stable_count = 0
            best_card = None
            settled = False
            saw_empty = not require_empty_first
            continue

        if waiting_verify:
            time.sleep(0.3)
            continue

        result = _pi_slot_scan(s, expecting)
        if result is None:
            time.sleep(1.5)
            continue

        present = bool(result.get("present"))
        cur = result.get("card") or {}
        cur_code = (
            f"{cur['rank']}{cur['suit'][0]}"
            if cur.get("rank") and cur.get("suit") else ""
        )
        log.log(
            f"[GUIDED/{mode}] Slot {expecting}: present={present} "
            f"card={cur_code or '-'} "
            f"conf={cur.get('confidence', 0.0):.2f} "
            f"saw_empty={saw_empty}"
        )

        if not present:
            if not saw_empty:
                log.log(f"[GUIDED/{mode}] Slot {expecting}: empty — ready for new card")
            saw_empty = True
            stable_count = 0
            best_card = None
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        # Replace mode: the old card may still be in the slot at loop
        # start. Accept the scan only after either (a) we saw present=
        # false at some point, or (b) YOLO reads a DIFFERENT card code
        # than the one that was in the slot before the replace started
        # — user swapped the card faster than our polling.
        if not saw_empty:
            prev_code = gd.get("previous_cards", {}).get(expecting, "")
            if cur_code and prev_code and cur_code != prev_code:
                log.log(
                    f"[GUIDED/{mode}] Slot {expecting}: card changed "
                    f"{prev_code} → {cur_code} (no empty seen) — accepting"
                )
                saw_empty = True
            else:
                time.sleep(GUIDED_POLL_S)
                continue

        if not settled:
            time.sleep(GUIDED_SETTLE_S)
            settled = True
            continue

        stable_count += 1
        card = result.get("card")
        if card:
            conf = float(card.get("confidence", 0.0))
            if best_card is None or conf > float(best_card.get("confidence", 0.0)):
                best_card = card
            code = f"{card['rank']}{card['suit'][0]}"
            if conf >= GUIDED_GOOD_CONF:
                with s.table_lock:
                    s.rodney_downs[expecting] = {
                        "rank": card["rank"],
                        "suit": card["suit"],
                        "confidence": round(conf, 2),
                    }
                    s.pi_prev_slots[expecting] = code
                    gd["index"] = idx + 1
                    s.table_state_version += 1
                _table_log_add(s, f"Slot {expecting} (replace): {code} (auto, {int(conf*100)}%)")
                _pi_slot_led(s, expecting, "off")
                if idx + 1 < len(slots):
                    _pi_slot_led(s, slots[idx + 1], "on")
                stable_count = 0
                best_card = None
                settled = False
                saw_empty = not require_empty_first
                continue

        if stable_count < GUIDED_STABLE_SCANS:
            time.sleep(GUIDED_POLL_S)
            continue

        if best_card is None:
            log.log(
                f"[GUIDED/{mode}] Slot {expecting}: present but nothing "
                f"recognized after {GUIDED_STABLE_SCANS} scans — continuing"
            )
            stable_count = 0
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        conf = float(best_card.get("confidence", 0.0))
        guess = {
            "rank": best_card["rank"],
            "suit": best_card["suit"],
            "confidence": round(conf, 2),
        }
        prompt = (
            f"Slot {expecting} (replacement): low confidence "
            f"({int(conf*100)}%). Confirm or correct."
        )

        with s.table_lock:
            s.pending_verify = {
                "slot": expecting,
                "guess": guess,
                "prompt": prompt,
                "image_url": f"/api/table/slot_image/{expecting}",
            }
            if guess["rank"]:
                s.slot_pending[expecting] = dict(guess)
            s.table_state_version += 1
        _pi_slot_led(s, expecting, "blink")


def _game_has_draw_phase(ge) -> bool:
    """True if the current game has a DRAW phase somewhere in its template."""
    try:
        from game_engine import PhaseType
        if ge.current_game is None:
            return False
        return any(ph.type == PhaseType.DRAW for ph in ge.current_game.phases)
    except Exception:
        return False


def _total_draw_phases(ge) -> int:
    """Count of DRAW phases. 5 Card Draw = 1, 3 Toed Pete = 3."""
    try:
        from game_engine import PhaseType
        if ge.current_game is None:
            return 0
        return sum(1 for ph in ge.current_game.phases if ph.type == PhaseType.DRAW)
    except Exception:
        return 0


def _max_draw_for_game(ge, draws_done: int = 0) -> int:
    """Max cards Rodney can replace in the draws_done-th DRAW phase.

    Multi-draw games (3 Toed Pete) shrink the allowance each round: 3, 2,
    then 1. draws_done is the number of draws already completed — 0 for
    the first draw, 1 for the second, etc. Returns 0 if no such phase.
    """
    try:
        from game_engine import PhaseType
        if ge.current_game is None:
            return 0
        seen = 0
        for ph in ge.current_game.phases:
            if ph.type == PhaseType.DRAW:
                if seen == draws_done:
                    return int(getattr(ph, "max_draw", 0) or 0)
                seen += 1
    except Exception:
        pass
    return 0


def _enqueue_down_card_verifies(s):
    """Called at the end of an up-card round (console Confirm Cards).

    Any slot that has a pending (low-confidence) scan that isn't already
    verified in rodney_downs gets added to the FIFO verify queue — but
    only if that slot has actually been dealt. A stale slot_pending
    entry for a trailing slot that we haven't started guiding yet (e.g.
    FTQ slot 3 during round 1) would otherwise pop a verify modal for a
    card that doesn't exist yet.
    """
    ge = s.game_engine
    initial = _initial_down_count(ge)
    # Slots currently valid to verify: the initial guided range plus any
    # slot the guided loop is presently iterating. Trailing slots get
    # added once the trailing guided session starts.
    expected_slots = set(range(1, initial + 1))
    if s.guided_deal:
        gd = s.guided_deal
        if "num_slots" in gd:
            expected_slots |= set(range(1, int(gd["num_slots"]) + 1))
        for sn in gd.get("slots") or []:
            try:
                expected_slots.add(int(sn))
            except (TypeError, ValueError):
                pass
    with s.table_lock:
        newly_queued = []
        for slot_num, guess in s.slot_pending.items():
            if slot_num in s.rodney_downs:
                continue
            if slot_num in s.verify_queue:
                continue
            if slot_num not in expected_slots:
                # Haven't dealt this slot yet (FTQ/7CS trailing, etc.) —
                # don't prompt the user to verify a card that isn't there.
                continue
            s.verify_queue.append(slot_num)
            newly_queued.append(slot_num)
        if newly_queued:
            for sn in newly_queued:
                _table_log_add(s, f"Slot {sn}: queued for verify (blink LED)")
            # Open the modal immediately rather than waiting for the next
            # poller tick — nicer UX, and in offline mode the poller may
            # be sleeping between failed fetches.
            _promote_next_verify(s)
            s.table_state_version += 1
    # TODO: POST to Pi /slots/<n>/led to blink once that endpoint is wired.
    return newly_queued


def _resolve_verify(s, card_dict):
    """Set the verified card into rodney_downs[slot] and clear the modal."""
    with s.table_lock:
        pv = s.pending_verify
        if not pv:
            return False
        slot = pv.get("slot")
        if slot is None:
            return False
        s.rodney_downs[slot] = {
            "rank": card_dict["rank"],
            "suit": card_dict["suit"],
        }
        code = f"{card_dict['rank']}{card_dict['suit'][0]}"
        s.pi_prev_slots[slot] = code
        s.slot_pending.pop(slot, None)
        if slot in s.verify_queue:
            s.verify_queue.remove(slot)
        s.pending_verify = None
        _table_log_add(s, f"Slot {slot}: {code} (verified)")
        s.table_state_version += 1
    return True


# ---------------------------------------------------------------------------
# Deal mode — dictation for game name, then visual recognition for cards
# ---------------------------------------------------------------------------

# Game templates: map game name to list of deal patterns
# Each pattern is a list of "up" or "down" per card, dealt to all players in order
GAME_PATTERNS = {
    "5 Card Draw": ["down"] * 5,
    "3 Toed Pete": ["down"] * 3,
    "7 Card Stud": ["down", "down", "up", "up", "up", "up", "down"],
    "7 Stud Deuces Wild": ["down", "down", "up", "up", "up", "up", "down"],
    "Follow the Queen": ["down", "down", "up", "up", "up", "up", "down"],
    "High Chicago": ["down", "down", "up", "up", "up", "up", "down"],
    "High Low High Challenge": ["down"] * 3,
    "7 27": ["down"] * 2,
    "Texas Hold'em": ["down"] * 2,
}


def _get_deal_order(dealer_name):
    """Return player names in deal order (clockwise from left of dealer)."""
    try:
        idx = [n.lower() for n in PLAYER_NAMES].index(dealer_name.lower())
    except ValueError:
        idx = 0
    # Start with player to dealer's left (next clockwise)
    order = []
    for i in range(1, len(PLAYER_NAMES) + 1):
        order.append(PLAYER_NAMES[(idx + i) % len(PLAYER_NAMES)])
    return order


def _start_deal_mode(s):
    if s.deal_mode:
        return
    s.deal_mode = {
        "phase": "game_select",  # "game_select", "dealing", "complete"
        "game": None,
        "dealer": None,
        "deal_order": list(PLAYER_NAMES),  # updated when dealer is set
        "pattern": [],
        "cards": [],
        "round_idx": 0,
        "player_idx": 0,
        "announced": set(),
    }
    log.log("Deal mode started — select dealer and game")


def _stop_deal_mode(s):
    s.deal_mode = None
    log.log("Deal mode stopped")


def _set_deal_game(s, game_name):
    """Set the game and start dealing."""
    if not s.deal_mode:
        return
    pattern = GAME_PATTERNS.get(game_name)
    if not pattern:
        log.log(f"[DEAL] Unknown game pattern: {game_name}")
        return
    s.deal_mode["game"] = game_name
    s.deal_mode["pattern"] = pattern
    s.deal_mode["phase"] = "dealing"
    s.deal_mode["round_idx"] = 0
    s.deal_mode["cards"] = []
    s.deal_mode["round_results"] = {}
    s.deal_mode["retry_time"] = 0

    # Capture baselines before dealing starts
    if s.latest_frame is not None:
        s.monitor.capture_baselines(s.latest_frame)
        log.log("[DEAL] Baselines captured")

    # Skip initial down card rounds
    _advance_to_next_up(s)

    order = s.deal_mode["deal_order"]
    log.log(f"[DEAL] Game: {game_name}, dealer: {s.deal_mode['dealer']}")
    log.log(f"[DEAL] Deal order: {' -> '.join(order)}")
    log.log(f"[DEAL] Pattern: {pattern}")
    if s.deal_mode["phase"] == "dealing":
        dealer_name = order[-1]
        log.log(f"[DEAL] Round {s.deal_mode['round_idx']+1}: waiting for {dealer_name}'s card (last dealt)")


def _advance_to_next_up(s):
    """Advance to next up card round, skipping down cards."""
    dm = s.deal_mode
    if not dm:
        return
    pattern = dm["pattern"]

    while dm["round_idx"] < len(pattern):
        if pattern[dm["round_idx"]] == "up":
            dm["phase"] = "dealing"
            dm["round_results"] = {}
            dm["retry_time"] = 0
            return
        log.log(f"[DEAL] Skipping round {dm['round_idx']+1} (down cards — use scanner)")
        dm["round_idx"] += 1

    dm["phase"] = "complete"
    log.log("[DEAL] All rounds dealt")


def _deal_scan_all_zones(s):
    """Scan all player zones and recognize cards. No baseline comparison — just crop and recognize."""
    dm = s.deal_mode
    if not dm:
        return

    frame = s.latest_frame
    if frame is None:
        return

    order = dm["deal_order"]
    missing = []

    for player in order:
        # Skip already recognized this round
        if player in dm["round_results"]:
            continue

        zone = next((z for z in s.cal.zones if z["name"] == player), None)
        if not zone:
            continue

        # Crop zone directly — don't rely on baseline diff
        crop = s.monitor._crop(frame, zone)
        if crop is None or crop.size == 0:
            missing.append(player)
            continue

        s.monitor._recognize_single(player, crop)
        result = s.monitor.last_card.get(player, "No card")
        if result and result != "No card":
            dm["round_results"][player] = result
            dm["cards"].append({
                "player": player,
                "card": result,
                "round": dm["round_idx"] + 1,
            })
            s.monitor.zone_state[player] = "empty"
            s.monitor.last_card[player] = ""
        else:
            missing.append(player)

    # Announce recognized cards
    for player in order:
        if player in dm["round_results"] and player not in dm.get("announced_this_round", set()):
            dm.setdefault("announced_this_round", set()).add(player)

    if missing:
        names = " and ".join(missing)
        log.log(f"[DEAL] Missing: {names}")
        speech.say(f"{names}, adjust your cards please")
        dm["phase"] = "retry_missing"
        dm["retry_time"] = time.time()
    else:
        # All recognized — recapture baselines WITH cards, then wait for removal
        log.log(f"[DEAL] Round {dm['round_idx']+1} complete — all {len(order)} cards recognized")
        # Capture baselines with cards present — clearing = change from this
        if s.latest_frame is not None:
            s.monitor.capture_baselines(s.latest_frame)
        speech.say("Clear zones")
        dm["phase"] = "waiting_to_clear"
        log.log("[DEAL] Waiting for all zones to be cleared")


def _deal_check_dealer_zone(s):
    """Check if the dealer (last in order) has a card — triggers full scan."""
    dm = s.deal_mode
    if not dm or dm["phase"] != "dealing":
        return

    frame = s.latest_frame
    if frame is None:
        return

    order = dm["deal_order"]
    dealer_name = order[-1]  # dealer gets card last
    dealer_zone = next((z for z in s.cal.zones if z["name"] == dealer_name), None)
    if not dealer_zone:
        return

    crop = s.monitor.check_single(frame, dealer_zone)
    if crop is not None:
        log.log(f"[DEAL] Card detected in {dealer_name}'s zone — waiting 2s for all cards to settle")
        dm["phase"] = "settling"
        dm["settle_time"] = time.time()


def _deal_retry_missing(s):
    """After 5 seconds, rescan missing zones."""
    dm = s.deal_mode
    if not dm or dm["phase"] != "retry_missing":
        return
    if time.time() - dm["retry_time"] < 5:
        return

    log.log("[DEAL] Retrying missing zones...")
    dm["phase"] = "scanning"
    _deal_scan_all_zones(s)


def _deal_check_zones_clear(s):
    """Check if all zones are empty — players moved cards out."""
    dm = s.deal_mode
    if not dm or dm["phase"] != "waiting_to_clear":
        return

    frame = s.latest_frame
    if frame is None:
        return

    order = dm["deal_order"]
    still_occupied = []
    for player in order:
        zone = next((z for z in s.cal.zones if z["name"] == player), None)
        if not zone:
            continue
        crop = s.monitor._crop(frame, zone)
        if crop is None or crop.size == 0:
            continue
        baseline = s.monitor.baselines.get(player)
        if baseline is None or crop.shape != baseline.shape:
            continue
        diff = float(np.mean(cv2.absdiff(crop, baseline)))
        # Baseline was captured WITH cards — if diff is LOW, card is still there
        # If diff is HIGH, card was removed (zone changed)
        if diff < s.monitor.threshold:
            still_occupied.append(player)

    if not still_occupied:
        # All zones clear — recapture baselines and advance
        log.log("[DEAL] All zones cleared")
        s.monitor.capture_baselines(frame)
        log.log("[DEAL] Baselines recaptured")
        dm["phase"] = "advancing"  # prevent re-entry
        dm["round_idx"] += 1
        dm["round_results"] = {}
        dm["announced_this_round"] = set()
        _advance_to_next_up(s)
        if dm["phase"] == "dealing":
            dealer_name = order[-1]
            log.log(f"[DEAL] Round {dm['round_idx']+1}: waiting for {dealer_name}'s card")
            speech.say("Deal")


def _deal_mode_json(s):
    """Return deal mode state as JSON-serializable dict."""
    dm = s.deal_mode
    if not dm:
        return None
    order = dm.get("deal_order", [])
    dealer_name = order[-1] if order else None
    missing = []
    if dm["phase"] in ("retry_missing", "scanning"):
        for p in order:
            if p not in dm.get("round_results", {}):
                missing.append(p)
    return {
        "phase": dm["phase"],
        "game": dm["game"],
        "dealer": dm.get("dealer"),
        "deal_order": order,
        "cards": dm["cards"],
        "round_results": dm.get("round_results", {}),
        "missing": missing,
        "watching_for": dealer_name if dm["phase"] == "dealing" else None,
        "round_idx": dm.get("round_idx", 0),
        "total_rounds": len(dm.get("pattern", [])),
    }


# ---------------------------------------------------------------------------
# Data collection mode
# ---------------------------------------------------------------------------

COLLECT_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
COLLECT_SUITS = ["clubs", "diamonds", "hearts", "spades"]

SUIT_NAMES = {"clubs": "Clubs", "diamonds": "Diamonds", "hearts": "Hearts", "spades": "Spades"}
RANK_NAMES = {"A": "Ace", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
              "8": "8", "9": "9", "10": "10", "J": "Jack", "Q": "Queen", "K": "King"}

# Pass assignments: which suit goes to which player zone each pass
# 4 suits across 4 players (Bill, David, Joe, Rodney), Steve rotates
COLLECT_PASSES = [
    {"Bill": "clubs", "David": "diamonds", "Joe": "hearts", "Rodney": "spades"},
    {"Bill": "diamonds", "David": "hearts", "Joe": "spades", "Rodney": "clubs"},
    {"Bill": "hearts", "David": "spades", "Joe": "clubs", "Rodney": "diamonds"},
    {"Bill": "spades", "David": "clubs", "Joe": "diamonds", "Rodney": "hearts"},
]


def _start_collect_mode(s):
    if s.collect_mode:
        return
    s.collect_mode = {"rank_idx": 0, "pass_idx": 0, "captured": False, "countdown": 0}
    log.log("Data collection started")
    p = COLLECT_PASSES[0]
    log.log("[COLLECT] Pass 1: " + ", ".join(f"{k}={v.capitalize()}" for k, v in p.items()))


def _stop_collect_mode(s):
    s.collect_mode = None
    log.log("Data collection stopped")


def _collect_deal_info(cm):
    """Return what to deal for current state."""
    if cm["pass_idx"] >= len(COLLECT_PASSES):
        return None
    if cm["rank_idx"] >= len(COLLECT_RANKS):
        return None
    rank = COLLECT_RANKS[cm["rank_idx"]]
    assignments = COLLECT_PASSES[cm["pass_idx"]]
    cards = {}
    for player, suit in assignments.items():
        cards[player] = f"{RANK_NAMES[rank]} of {SUIT_NAMES[suit]}"
    return {"rank": rank, "cards": cards, "pass": cm["pass_idx"] + 1}


def _collect_scan(s):
    """Capture zone crops and save with correct labels per player/suit."""
    cm = s.collect_mode
    if not cm:
        return

    info = _collect_deal_info(cm)
    if not info:
        return

    frame = s.latest_frame
    if frame is None:
        log.log("[COLLECT] No frame available")
        return

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    pass_num = info["pass"]

    for zone in s.cal.zones:
        name = zone["name"]
        label = info["cards"].get(name)
        if not label:
            continue

        crop = s.monitor._crop(frame, zone)
        if crop is None or crop.size == 0:
            log.log(f"[COLLECT] {name}: crop failed")
            continue

        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(" ", "_")
        filename = f"collect_p{pass_num}_{safe_label}_{name}_{ts}"
        cv2.imwrite(str(TRAINING_DIR / f"{filename}.jpg"), crop)
        (TRAINING_DIR / f"{filename}.txt").write_text(label)
        saved += 1
        log.log(f"[COLLECT] {name}: {label}")

    log.log(f"[COLLECT] Saved {saved} images")
    cm["captured"] = True


def _collect_advance(s):
    """Advance to the next rank or pass."""
    cm = s.collect_mode
    if not cm:
        return
    cm["rank_idx"] += 1
    cm["captured"] = False
    if cm["rank_idx"] >= len(COLLECT_RANKS):
        cm["pass_idx"] += 1
        cm["rank_idx"] = 0
        if cm["pass_idx"] < len(COLLECT_PASSES):
            p = COLLECT_PASSES[cm["pass_idx"]]
            log.log(f"[COLLECT] Pass {cm['pass_idx']+1}: " +
                    ", ".join(f"{k}={v.capitalize()}" for k, v in p.items()))
            # Pause for new pass — user needs to re-sort deck
            cm["phase"] = "paused_new_pass"
            speech.say(f"Pass {cm['pass_idx']+1}. New suit assignments. Press Start when ready.")
            return
        else:
            cm["phase"] = "done"
            log.log("[COLLECT] All 4 passes complete!")
            speech.say("Data collection complete")
            return
    # Start the deal phase
    _collect_start_deal(s)


def _collect_start_deal(s):
    """Start the 5-second deal countdown."""
    cm = s.collect_mode
    if not cm:
        return
    cm["phase"] = "dealing"
    cm["timer_start"] = time.time()
    cm["timer_duration"] = 5
    speech.say("Deal")


def _collect_start_clear(s):
    """Start the 5-second clear countdown."""
    cm = s.collect_mode
    if not cm:
        return
    cm["phase"] = "clearing"
    cm["timer_start"] = time.time()
    cm["timer_duration"] = 5
    speech.say("Clear")


def _collect_redo(s):
    """Go back to previous rank and redo."""
    cm = s.collect_mode
    if not cm:
        return
    # Delete the images we just saved for this rank
    if cm["captured"]:
        info = _collect_deal_info(cm)
        if info:
            pass_num = info["pass"]
            rank = info["rank"]
            # Find and delete files matching this rank/pass
            for f in TRAINING_DIR.glob(f"collect_p{pass_num}_*_{rank}_*"):
                f.unlink()
                log.log(f"[COLLECT] Deleted: {f.name}")

    cm["captured"] = False
    cm["phase"] = "paused"
    log.log(f"[COLLECT] Redo — re-deal rank {COLLECT_RANKS[cm['rank_idx']]}")
    speech.say("Redo")


def _collect_auto_cycle(s):
    """Called from bg_loop — handles the timed phases."""
    cm = s.collect_mode
    if not cm or cm.get("phase") not in ("dealing", "clearing"):
        return

    elapsed = time.time() - cm.get("timer_start", 0)
    remaining = cm.get("timer_duration", 5) - elapsed

    if remaining <= 0:
        if cm["phase"] == "dealing":
            # Deal time is up — scan now
            _collect_scan(s)
            _collect_start_clear(s)
        elif cm["phase"] == "clearing":
            # Clear time is up — advance to next rank
            _collect_advance(s)


def _collect_start_first(s):
    """Start the first deal cycle."""
    cm = s.collect_mode
    if not cm:
        return
    _collect_start_deal(s)


def _collect_mode_json(s):
    cm = s.collect_mode
    if not cm:
        return None
    info = _collect_deal_info(cm)
    done = cm["pass_idx"] >= len(COLLECT_PASSES)
    total = len(COLLECT_RANKS) * len(COLLECT_PASSES)
    current = cm["pass_idx"] * len(COLLECT_RANKS) + cm["rank_idx"]
    phase = cm.get("phase", "paused")
    countdown = 0
    if phase in ("dealing", "clearing"):
        elapsed = time.time() - cm.get("timer_start", 0)
        countdown = max(0, int(cm.get("timer_duration", 5) - elapsed))

    return {
        "rank_idx": cm["rank_idx"],
        "pass_idx": cm["pass_idx"],
        "pass_total": len(COLLECT_PASSES),
        "cards": info["cards"] if info else {},
        "rank": COLLECT_RANKS[cm["rank_idx"]] if cm["rank_idx"] < len(COLLECT_RANKS) else None,
        "captured": cm["captured"],
        "done": done,
        "current": current,
        "total": total,
        "countdown": countdown,
        "phase": phase,
    }


def _process_deal_text(s, text):
    """Parse dictated text for game name only."""
    from speech_recognition_module import parse_speech, GameCommand

    if not s.deal_mode or s.deal_mode["phase"] != "game_select":
        return

    log.log(f"[DEAL] Parsing game name: \"{text}\"")
    commands = parse_speech(text)
    for cmd in commands:
        if isinstance(cmd, GameCommand):
            _set_deal_game(s, cmd.game_name)
            return


# ---------------------------------------------------------------------------
# Background capture
# ---------------------------------------------------------------------------

def bg_loop():
    while not _state.quit_flag:
        frame = _state.capture.capture()
        if frame is not None:
            _state.latest_frame = frame
            # Recognition/motion detection runs FIRST so card arrival
            # doesn't wait on the display-JPEG encode below. On a 4K
            # Brio frame to_jpeg was eating 0.5-1.5s per bg_loop pass.
            if _state.monitoring and _state.cal.ok:
                _console_watch_dealer(_state, frame)

            # Display JPEG is cheap once we downscale — the UI renders
            # it inside a small iframe anyway. Keep the overlay drawn
            # on the full frame (zone coordinates are in 4K space),
            # then scale the encoded output.
            disp = crop_circle(frame, _state.cal).copy()
            draw_overlay(disp, _state.cal, _state.monitor)
            small = cv2.resize(disp, (1280, 720), interpolation=cv2.INTER_AREA)
            _state.latest_jpg = to_jpeg(small, 70)

            # Data collection auto-cycle
            if _state.collect_mode:
                _collect_auto_cycle(_state)

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
                        tm["confirm_time"] = time.time()
                        speech.say(f"{zone['name']}, {result}")

            # Test mode: auto-confirm after 4 seconds
            if tm and tm["waiting"] == "confirm":
                if time.time() - tm.get("confirm_time", 0) > 4:
                    # Auto-confirm — advance to next zone
                    tm["zone_idx"] += 1
                    if tm["zone_idx"] >= len(_state.cal.zones):
                        _state.test_mode = None
                        log.log("[TEST] All zones tested")
                    else:
                        tm["waiting"] = "card"
                        tm["result"] = ""
                        next_name = _state.cal.zones[tm["zone_idx"]]["name"]
                        speech.say(f"{next_name} is next")
                        log.log(f"[TEST] Auto-confirmed. Next: {next_name}")

            # Deal mode
            dm = _state.deal_mode
            if dm and _state.cal.ok:
                if dm["phase"] == "dealing":
                    _deal_check_dealer_zone(_state)
                elif dm["phase"] == "settling":
                    if time.time() - dm.get("settle_time", 0) >= 2:
                        log.log("[DEAL] Scanning all zones")
                        dm["phase"] = "scanning"
                        dm["announced_this_round"] = set()
                        _deal_scan_all_zones(_state)
                elif dm["phase"] == "retry_missing":
                    _deal_retry_missing(_state)
                elif dm["phase"] == "waiting_to_clear":
                    _deal_check_zones_clear(_state)

        time.sleep(1)  # 1 second capture rate

TABLE_HTML = """<!DOCTYPE html>
<html><head><title>Poker Table</title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font-family:-apple-system,sans-serif;background:#0d1b2a;color:#e0e0e0;overflow:hidden;display:flex;flex-direction:column}

/* Top bar */
header{flex:0 0 auto;display:flex;justify-content:space-between;align-items:center;
  padding:10px 18px;background:#16213e;border-bottom:1px solid #1f3560;font-size:.95em}
header .group{display:flex;gap:18px;align-items:center;flex-wrap:wrap}
header .lbl{color:#8faacc;margin-right:4px;font-size:.85em}
header .val{color:#e0e0e0;font-weight:600}
header .val.game{color:#4fc3f7;font-size:1.1em}
header button{padding:6px 12px;background:#0f3460;color:#fff;border:none;border-radius:6px;
  cursor:pointer;font-size:.85em}
header button:hover{background:#1a5a9a}

/* Table area split into three rows */
.table{flex:1 1 auto;display:grid;grid-template-rows:1fr 1fr 1fr;gap:8px;padding:10px;min-height:0}
.row{display:flex;align-items:center;min-height:0}
.row.top, .row.middle{justify-content:space-between}
.row.bottom{justify-content:center}

/* Per-player hand region */
.hand-box{display:flex;flex-direction:column;min-height:0;max-width:48%;height:100%}
.hand-box.center{max-width:90%;align-items:center}
.hand-box .head{display:flex;gap:6px;align-items:baseline;margin-bottom:4px}
.hand-box .name{font-weight:700;font-size:1em}
.hand-box .name.dealer{color:#ffd54f}
.hand-box .name.remote{color:#4fc3f7}
.hand-box .sml-btn{padding:2px 8px;background:#0f3460;color:#fff;border:none;border-radius:4px;
  cursor:pointer;font-size:.75em}
.hand-box .sml-btn.active{background:#1b5e20}
.hand-box .sml-btn.folded{background:#b71c1c}
.hand-box .values{margin-left:auto;padding:2px 10px;background:#0f3460;color:#ffd54f;border-radius:4px;font-size:.85em;font-weight:600;white-space:nowrap}
.hand-box .freezes{margin-left:6px;padding:2px 8px;background:#1a5a9a;color:#e0e0e0;border-radius:4px;font-size:.8em;font-weight:600;white-space:nowrap}
.hand-box .freezes.frozen{background:#0d47a1;color:#bbdefb;outline:1px solid #e3f2fd}
.hand-box .cards{flex:1 1 auto;display:flex;gap:0;align-items:flex-start;min-height:0;overflow:hidden}
.hand-box.center .cards{justify-content:center}
.hand-box.folded .cards{opacity:.3;filter:grayscale(60%)}

/* Card art scales to 80% of the row height */
.card{height:80%;aspect-ratio:2.5/3.5;border-radius:6px;overflow:hidden;background:#fff;
  border:1px solid #333;box-shadow:0 1px 3px rgba(0,0,0,.5);flex-shrink:0;position:relative}
.card img{width:100%;height:100%;display:block;object-fit:contain;background:#fff}
.card.offset-down{transform:translateY(10px)}
.card.missing{background:#1a3a5a;display:flex;align-items:center;justify-content:center;color:#eee;font-size:.8em}
.card.markable{cursor:pointer}
.card.marked{outline:3px solid #ffb74d;outline-offset:-1px;opacity:.55}
.card.marked::after{content:'✗';position:absolute;top:4px;right:6px;color:#ffb74d;font-size:1.4em;font-weight:800;text-shadow:0 0 4px #000}
#draw-request-row{margin:10px 0;text-align:center}
#draw-request-btn{padding:10px 22px;font-size:1em;font-weight:600;background:#b26b00;color:#fff;border:none;border-radius:8px;cursor:pointer}
#draw-request-btn:disabled{opacity:.5}

/* Verify modal */
.modal{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;
  justify-content:center;z-index:50}
.modal.show{display:flex}
.modal-inner{background:#16213e;padding:20px 24px;border-radius:12px;max-width:460px;width:90%}
.modal h2{color:#4fc3f7;font-size:1.1em;margin-bottom:10px}
.modal .guess{padding:10px;background:#0f3460;border-radius:6px;margin:10px 0;font-size:1.05em}
.modal input{width:100%;padding:10px;border-radius:6px;border:1px solid #333;background:#0d1b2a;
  color:#e0e0e0;font-size:1em;font-family:inherit}
.modal .buttons{display:flex;gap:8px;margin-top:14px}
.modal button{flex:1;padding:12px;border:none;border-radius:8px;cursor:pointer;font-weight:600}
.btn-green{background:#1b5e20;color:#fff}
.btn-red{background:#b71c1c;color:#fff}
.btn-blue{background:#0f3460;color:#fff}
</style></head><body>
<header>
  <div class="group">
    <span><span class="lbl">Game</span><span class="val game" id="game-name">—</span></span>
    <span><span class="lbl">Dealer</span><span class="val" id="dealer-name">—</span></span>
    <span><span class="lbl">Round</span><span class="val" id="round-info">—</span></span>
    <span><span class="lbl">Wilds</span><span class="val" id="wilds-info">—</span></span>
  </div>
  <div class="group">
    <button onclick="openLogs()">Logs</button>
  </div>
</header>

<div id="guided-banner"
     style="display:none;background:#1b4d2a;color:#e8f5e9;padding:8px 14px;
            text-align:center;font-weight:600;letter-spacing:.02em"></div>

<div class="table">
  <div class="row top">
    <div class="hand-box" id="box-Bill"></div>
    <div class="hand-box" id="box-David"></div>
  </div>
  <div class="row middle">
    <div class="hand-box" id="box-Steve"></div>
    <div class="hand-box" id="box-Joe"></div>
  </div>
  <div class="row bottom">
    <div class="hand-box center" id="box-Rodney"></div>
  </div>
</div>

<div class="modal" id="flip-modal">
  <div class="modal-inner">
    <h2 id="flip-title">Pick a card to turn face-up</h2>
    <div id="flip-options" style="display:flex;gap:12px;justify-content:center;margin-top:12px"></div>
  </div>
</div>

<div class="modal" id="verify-modal">
  <div class="modal-inner">
    <h2>Verify card</h2>
    <div id="verify-body">—</div>
    <div style="text-align:center;margin:8px 0">
      <img id="verify-scan" alt="slot scan"
           style="max-width:120px;max-height:180px;border:1px solid #444;
                  border-radius:4px;background:#000;display:none"/>
    </div>
    <div class="guess" id="verify-guess">—</div>
    <div style="display:flex;gap:10px;align-items:center;margin:8px 0">
      <label style="width:60px">Rank</label>
      <select id="verify-rank" style="flex:1;padding:8px;border-radius:6px;border:1px solid #333;background:#0d1b2a;color:#e0e0e0">
        <option value="">—</option>
        <option value="A">Ace</option>
        <option value="2">2</option><option value="3">3</option>
        <option value="4">4</option><option value="5">5</option>
        <option value="6">6</option><option value="7">7</option>
        <option value="8">8</option><option value="9">9</option>
        <option value="10">10</option>
        <option value="J">Jack</option><option value="Q">Queen</option>
        <option value="K">King</option>
      </select>
    </div>
    <div style="display:flex;gap:10px;align-items:center;margin:8px 0">
      <label style="width:60px">Suit</label>
      <select id="verify-suit" style="flex:1;padding:8px;border-radius:6px;border:1px solid #333;background:#0d1b2a;color:#e0e0e0">
        <option value="">—</option>
        <option value="clubs">Clubs</option>
        <option value="diamonds">Diamonds</option>
        <option value="hearts">Hearts</option>
        <option value="spades">Spades</option>
      </select>
    </div>
    <div class="buttons">
      <button class="btn-green" onclick="confirmVerify()">Accept guess</button>
      <button class="btn-red" onclick="overrideVerify()">Use my values</button>
      <button class="btn-blue" onclick="rescanVerify()">Rescan</button>
    </div>
  </div>
</div>

<script>
var SUIT_SYM = {clubs:"\u2663",diamonds:"\u2666",hearts:"\u2665",spades:"\u2660"};
var RANK_FILE = {"2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9","10":"10",
                 "J":"jack","Q":"queen","K":"king","A":"ace"};
var RANK_ORDER = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"J":11,"Q":12,"K":13,"A":14};

// Per-player sort preference; default deal order.
var sortMode = {};    // name -> "deal" | "rank"
var lastVersion = -1;
var lastEtag = null;
var _logWin = null;

function openLogs() {
  if (_logWin && !_logWin.closed) { _logWin.focus(); return; }
  _logWin = window.open('/logview', '_tablelogs', 'width=900,height=500,scrollbars=yes');
}

function cardImgUrl(rank, suit) {
  var r = RANK_FILE[rank];
  if (!r || !suit) return null;
  return '/cards/' + r + '_of_' + suit + '.svg';
}

function cardEl(card, opts) {
  opts = opts || {};
  var el = document.createElement('div');
  el.className = 'card';
  if (card.type === 'down' && (card.hidden || !card.rank)) {
    var back = document.createElement('img');
    back.src = '/cards/back.svg';
    back.alt = 'card back';
    el.classList.add('offset-down');
    el.appendChild(back);
    return el;
  }
  if (card.type === 'down') el.classList.add('offset-down');
  var url = cardImgUrl(card.rank, card.suit);
  if (!url) { el.classList.add('missing'); el.textContent = '?'; return el; }
  var img = document.createElement('img');
  img.src = url;
  img.alt = (card.rank || '') + (SUIT_SYM[card.suit] || '');
  el.appendChild(img);
  // Rodney's draw-phase marking: when can_mark and this is a down card
  // backed by a slot, tapping toggles the mark.
  if (opts.markable && card.type === 'down' && card.slot) {
    el.classList.add('markable');
    if (opts.marked) el.classList.add('marked');
    el.onclick = function(ev) {
      ev.stopPropagation();
      toggleMark(card.slot, !opts.marked);
    };
  }
  return el;
}

function sortCards(cards, mode, best) {
  if (mode === "best" && best && best.codes && best.codes.length) {
    // "Best" mode: cards that form the best hand come first in the exact
    // order poker_hands.best_hand returned, then leftover cards sorted
    // by rank descending. Down-only placeholders (no rank) sink last.
    var codeOf = function(c){
      if (!c.rank || !c.suit) return "";
      var sl = {clubs:"c",diamonds:"d",hearts:"h",spades:"s"}[c.suit] || "";
      return c.rank + sl;
    };
    var used = {};
    var ordered = [];
    best.codes.forEach(function(code){
      for (var i = 0; i < cards.length; i++) {
        if (used[i]) continue;
        if (codeOf(cards[i]) === code) {
          used[i] = true;
          ordered.push(cards[i]);
          break;
        }
      }
    });
    var leftover = [];
    for (var j = 0; j < cards.length; j++) {
      if (!used[j]) leftover.push(cards[j]);
    }
    leftover.sort(function(a,b){
      var aKnown = !!a.rank, bKnown = !!b.rank;
      if (aKnown !== bKnown) return aKnown ? -1 : 1;
      var ra = RANK_ORDER[a.rank] || 0, rb = RANK_ORDER[b.rank] || 0;
      return rb - ra;
    });
    return ordered.concat(leftover);
  }
  if (mode !== "rank") return cards;
  var copy = cards.slice();
  copy.sort(function(a,b) {
    // Put down (unknown) cards first, then sort the rest by rank
    var aKnown = !!a.rank, bKnown = !!b.rank;
    if (aKnown !== bKnown) return aKnown ? 1 : -1;
    var ra = RANK_ORDER[a.rank] || 0, rb = RANK_ORDER[b.rank] || 0;
    if (ra !== rb) return ra - rb;
    return (a.suit || "").localeCompare(b.suit || "");
  });
  return copy;
}

function buildCards(p) {
  // Downs first (on the left), then up cards, matching Rodney's hand order.
  if (p.is_remote) return (p.hand || []).slice();
  var cards = [];
  for (var i = 0; i < (p.down_count || 0); i++) {
    cards.push({type:'down', hidden:true});
  }
  (p.up_cards || []).forEach(function(c){
    cards.push({type:'up', rank:c.rank, suit:c.suit});
  });
  return cards;
}

function setSortMode(name, mode) {
  sortMode[name] = mode;
  renderPlayer(window._playersByName[name]);
}
// Backward compat — some handlers might still reference toggleSort.
function toggleSort(name) {
  var cur = sortMode[name] || "deal";
  setSortMode(name, cur === "deal" ? "rank" : (cur === "rank" ? "best" : "deal"));
}

function toggleFold(name, currentlyFolded) {
  fetch('/api/table/fold', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({player: name, folded: !currentlyFolded})
  });
}

function toggleMark(slot, markOn) {
  fetch('/api/table/mark', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({slot: slot, marked: markOn})
  }).then(function(r){ return r.json(); }).then(function(d){
    if (d && !d.ok && d.error) console.warn('mark failed:', d.error);
  });
}

function requestCards() {
  fetch('/api/table/request_cards', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({})
  }).then(function(r){ return r.json(); }).then(function(d){
    if (d && !d.ok && d.error) alert('Request failed: ' + d.error);
  });
}

function renderPlayer(p) {
  if (!p) return;
  var box = document.getElementById('box-' + p.name);
  if (!box) return;
  var folded = !!p.folded;
  box.classList.toggle('folded', folded);

  // Header: name + Deal/Rank toggle + Fold button
  var head = document.createElement('div');
  head.className = 'head';
  var nm = document.createElement('span');
  nm.className = 'name' + (p.is_dealer ? ' dealer' : '') + (p.is_remote ? ' remote' : '');
  nm.textContent = p.name + (p.is_dealer ? ' (D)' : '');
  head.appendChild(nm);

  var mode = sortMode[p.name] || "deal";
  var sortSel = document.createElement('select');
  sortSel.className = 'sml-btn';
  [
    ["deal","Deal Order"],
    ["rank","Rank Order"],
    ["best","Best Hand"],
  ].forEach(function(o){
    var opt = document.createElement('option');
    opt.value = o[0]; opt.textContent = o[1];
    if (o[0] === mode) opt.selected = true;
    sortSel.appendChild(opt);
  });
  sortSel.onchange = function(){ setSortMode(p.name, sortSel.value); };
  head.appendChild(sortSel);
  if (mode === "best" && p.best_hand && p.best_hand.label) {
    var bhLabel = document.createElement('span');
    bhLabel.className = 'values';
    bhLabel.textContent = p.best_hand.label;
    head.appendChild(bhLabel);
  }

  var foldBtn = document.createElement('button');
  foldBtn.className = 'sml-btn' + (folded ? ' folded' : '');
  foldBtn.textContent = folded ? 'Folded' : 'Fold';
  foldBtn.onclick = function(){ toggleFold(p.name, folded); };
  head.appendChild(foldBtn);

  // 7/27 hand value(s), if the game engine has computed them.
  if (p.values_7_27 && p.values_7_27.length) {
    var vSpan = document.createElement('span');
    vSpan.className = 'values';
    vSpan.textContent = p.values_7_27.join(' / ');
    head.appendChild(vSpan);
  }

  // Freeze count for 7/27 hit rounds: ❄︎ N, or "FROZEN" once locked.
  if (p.freezes != null && p.freezes > 0) {
    var fSpan = document.createElement('span');
    fSpan.className = 'freezes' + (p.frozen ? ' frozen' : '');
    fSpan.textContent = p.frozen ? 'FROZEN' : ('❄ ' + p.freezes);
    head.appendChild(fSpan);
  }

  var cardRow = document.createElement('div');
  cardRow.className = 'cards';
  var cards = sortCards(buildCards(p), mode, p.best_hand);
  var drawState = (window._lastState && window._lastState.draw) || {};
  var canMark = !!drawState.can_mark && p.is_remote;
  var markedSet = {};
  (drawState.marked_slots || []).forEach(function(s){ markedSet[s] = true; });
  cards.forEach(function(c){
    var opts = {};
    if (canMark) {
      opts.markable = true;
      opts.marked = !!(c.slot && markedSet[c.slot]);
    }
    cardRow.appendChild(cardEl(c, opts));
  });

  box.innerHTML = '';
  box.appendChild(head);
  box.appendChild(cardRow);

  // Request-cards row: only visible in the "draw" state once the dealer
  // has pressed Pot is right. Pre-selection during betting still shows
  // the mark ✗ badges but the button doesn't appear until it's the draw.
  if (drawState.can_request && p.is_remote) {
    var row = document.createElement('div');
    row.id = 'draw-request-row';
    var btn = document.createElement('button');
    btn.id = 'draw-request-btn';
    var count = (drawState.marked_slots || []).length;
    btn.textContent = count
      ? ('Request ' + count + ' card' + (count === 1 ? '' : 's'))
      : 'Stand pat (no cards)';
    btn.onclick = function() { requestCards(); };
    row.appendChild(btn);
    box.appendChild(row);
  }

  // After layout settles, fan the cards: each card after the first
  // overlaps its neighbor so only ~22px of the underlying card's left
  // edge (the rank+suit indicator) remains visible. Re-apply on every
  // render so it tracks row resizes and new cards appearing.
  requestAnimationFrame(function() {
    var first = cardRow.querySelector('.card');
    if (!first) return;
    var w = first.getBoundingClientRect().width;
    if (!w) return;
    var reveal = 44;  // px of underlying card left edge to show
    var overlap = Math.max(0, w - reveal);
    Array.prototype.forEach.call(cardRow.querySelectorAll('.card'), function(c, i) {
      c.style.marginLeft = (i === 0) ? '0' : ('-' + overlap + 'px');
    });
  });
}

function render(state) {
  if (!state) return;
  window._lastState = state;
  var byName = {};
  state.players.forEach(function(p){ byName[p.name] = p; });
  window._playersByName = byName;

  document.getElementById('game-name').textContent = state.game.name || 'No game';
  document.getElementById('dealer-name').textContent = state.dealer || '—';
  var cur = state.game.current_round || 0;
  var tot = state.game.total_rounds || 0;
  document.getElementById('round-info').textContent =
    tot ? (cur + ' of ' + tot) : (cur ? String(cur) : '—');
  var wilds = state.game.wild_ranks || [];
  document.getElementById('wilds-info').textContent = wilds.length ? wilds.join(', ') : '—';

  var gd = state.guided_deal;
  var banner = document.getElementById('guided-banner');
  if (gd) {
    if (gd.slots) {
      // Replacement flow: explicit slot list + index into it.
      var idx = gd.index || 0;
      var slots = gd.slots;
      if (idx < slots.length) {
        banner.textContent = 'Replacement: place card in slot ' + slots[idx]
          + ' (' + (idx + 1) + ' of ' + slots.length + ')';
      } else {
        banner.textContent = 'Replacement done';
      }
    } else {
      var n = gd.expecting;
      var N = gd.num_slots;
      if (n <= N) {
        banner.textContent = 'Dealing: place card in slot ' + n + ' of ' + N;
      } else {
        banner.textContent = 'All ' + N + ' cards in';
      }
    }
    banner.style.display = 'block';
  } else {
    banner.style.display = 'none';
  }

  // Render every configured box; missing players just get cleared.
  ['Bill','David','Steve','Joe','Rodney'].forEach(function(nm) {
    var p = byName[nm];
    var box = document.getElementById('box-' + nm);
    if (!p) { if (box) box.innerHTML = ''; return; }
    renderPlayer(p);
  });

  var flip = state.flip_choice;
  var fmodal = document.getElementById('flip-modal');
  if (flip && flip.options && flip.options.length) {
    document.getElementById('flip-title').textContent = flip.prompt || 'Pick a card';
    var opts = document.getElementById('flip-options');
    opts.innerHTML = '';
    flip.options.forEach(function(opt) {
      var btn = document.createElement('button');
      btn.style.padding = '0';
      btn.style.border = '2px solid #333';
      btn.style.background = '#0d1b2a';
      btn.style.borderRadius = '6px';
      btn.style.cursor = 'pointer';
      var img = document.createElement('img');
      var url = cardImgUrl(opt.rank, opt.suit);
      if (url) { img.src = url; img.alt = opt.rank + ' of ' + opt.suit; }
      img.style.height = '160px';
      img.style.display = 'block';
      img.style.background = '#fff';
      img.style.borderRadius = '4px';
      btn.appendChild(img);
      var caption = document.createElement('div');
      caption.textContent = 'Slot ' + opt.slot;
      caption.style.color = '#aaa';
      caption.style.fontSize = '.85em';
      caption.style.margin = '4px 0';
      btn.appendChild(caption);
      btn.onclick = function() { pickFlip(opt.slot); };
      opts.appendChild(btn);
    });
    fmodal.classList.add('show');
  } else {
    fmodal.classList.remove('show');
  }

  var pv = state.pending_verify;
  var modal = document.getElementById('verify-modal');
  if (pv) {
    document.getElementById('verify-body').textContent = pv.prompt || '';
    var g = pv.guess || {};
    var gtxt = g.rank ? (g.rank + (SUIT_SYM[g.suit] || '')) : '—';
    if (g.confidence != null) gtxt += ' (' + Math.round(g.confidence * 100) + '%)';
    document.getElementById('verify-guess').textContent = 'Guess: ' + gtxt;
    var img = document.getElementById('verify-scan');
    // Refresh the scan image + seed selects on (re)open and whenever the
    // backend bumps rescan_id (user pressed Rescan and a new guess arrived).
    var rkey = (pv.slot || 0) + ':' + (pv.rescan_id || 0);
    if (modal.dataset.rkey !== rkey) {
      modal.dataset.rkey = rkey;
      document.getElementById('verify-rank').value = g.rank || '';
      document.getElementById('verify-suit').value = g.suit || '';
      if (pv.image_url) {
        img.src = pv.image_url + '?v=' + rkey + '&t=' + Date.now();
        img.style.display = 'inline-block';
      } else {
        img.style.display = 'none';
      }
    }
    modal.classList.add('show');
  } else {
    document.getElementById('verify-modal').dataset.rkey = '';
    modal.classList.remove('show');
  }
}

function poll() {
  var headers = {};
  if (lastEtag) headers['If-None-Match'] = lastEtag;
  fetch('/table/state', {headers: headers, cache: 'no-store'}).then(function(r) {
    if (r.status === 304) return null;
    lastEtag = r.headers.get('ETag');
    return r.json();
  }).then(function(d) {
    if (!d) return;
    if (d.version === lastVersion) return;
    lastVersion = d.version;
    render(d);
  }).catch(function(e) { console.warn('poll failed', e); });
}

function pickFlip(slot) {
  // Close the modal immediately so the user gets instant feedback instead
  // of waiting up to one poll cycle (500 ms) for the next render pass.
  var fmodal = document.getElementById('flip-modal');
  if (fmodal) fmodal.classList.remove('show');
  fetch('/api/table/flip_up', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({slot: slot})
  }).then(function(r){return r.json()}).then(function(d) {
    if (!d.ok) alert('Flip failed: ' + (d.error || 'unknown'));
    poll();
  }).catch(function(e) { console.warn('flip failed', e); });
}

function confirmVerify() {
  fetch('/api/table/verify', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action:'confirm'})
  }).then(function(r){return r.json()}).then(function(d) {
    if (!d.ok) alert('Verify failed: ' + (d.error || 'unknown'));
  });
}
function overrideVerify() {
  var rank = document.getElementById('verify-rank').value;
  var suit = document.getElementById('verify-suit').value;
  if (!rank || !suit) return alert('Pick both rank and suit');
  fetch('/api/table/verify', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action:'override', rank: rank, suit: suit})
  }).then(function(r){return r.json()}).then(function(d) {
    if (!d.ok) alert('Override failed: ' + (d.error || 'unknown'));
  });
}
function rescanVerify() {
  fetch('/api/table/verify', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action:'rescan'})
  }).then(function(r){return r.json()}).then(function(d) {
    if (!d.ok) alert('Rescan failed: ' + (d.error || 'unknown'));
  });
}

setInterval(poll, 500);
poll();
</script>
</body></html>"""


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
            "/logview": self._serve_logview,
            "/calibration": lambda s: self._r(200,"application/json",CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else self._r(404,"text/plain","none"),
            "/training": self._training_list,
            "/console": self._console_page,
            "/table": self._table_page,
            "/table/state": self._table_state,
        }
        if p in routes:
            routes[p](s)
        elif p.startswith("/zone_snap/"):
            name = p[11:]
            crop = s.monitor.recognition_crops.get(name)
            if crop is not None:
                j = to_jpeg(crop, 90)
                if j: self._r(200,"image/jpeg",j)
                else: self._r(500,"text/plain","Encode failed")
            else:
                self._zone_img(s, name)  # fallback to live
        elif p.startswith("/zone/"):
            self._zone_img(s, p[6:])
        elif p.startswith("/training/"):
            self._training_file(p[10:])
        elif p.startswith("/cards/"):
            self._card_asset(p[7:])
        elif p.startswith("/api/table/slot_image/"):
            try:
                slot_n = int(p[len("/api/table/slot_image/"):])
            except ValueError:
                return self._r(404, "text/plain", "bad slot")
            self._proxy_slot_image(s, slot_n)
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

        elif p == "/api/log/clear":
            log.clear()
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/brio/focus":
            # Live focus tuning + persistence. Body: {"value": N} where N
            # is 0..255 (lower = farther). "auto" or null re-enables AF.
            raw = data.get("value")
            new_val = None
            if isinstance(raw, (int, float)):
                new_val = max(0, min(255, int(raw)))
            elif isinstance(raw, str) and raw.strip().isdigit():
                new_val = max(0, min(255, int(raw.strip())))
            s.capture.set_focus(new_val)
            _save_host_config({"brio_focus": new_val})
            self._r(200, "application/json",
                    json.dumps({"ok": True, "focus": new_val}))

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

        elif p == "/api/deal/dealer":
            dealer = data.get("dealer", "")
            if s.deal_mode:
                s.deal_mode["dealer"] = dealer
                s.deal_mode["deal_order"] = _get_deal_order(dealer)
                log.log(f"[DEAL] Dealer: {dealer}, order: {' -> '.join(s.deal_mode['deal_order'])}")
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/text":
            text = data.get("text", "")
            _process_deal_text(s, text)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/clear":
            _stop_deal_mode(s)
            _start_deal_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/start":
            _start_collect_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/stop":
            _stop_collect_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/go":
            _collect_start_first(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/pause":
            if s.collect_mode:
                s.collect_mode["phase"] = "paused"
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/resume":
            if s.collect_mode:
                _collect_start_deal(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/redo":
            _collect_redo(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/snapshot/save":
            if s.latest_frame is not None:
                cropped = crop_circle(s.latest_frame, s.cal)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = Path(__file__).parent / f"snapshot_{ts}.jpg"
                cv2.imwrite(str(path), cropped)
                log.log(f"Snapshot saved: {path.name}")
            self._r(200,"application/json",'{"ok":true}')

        # --- Observer table view ---

        elif p == "/api/table/pi_start":
            _pi_poll_start(s)
            self._r(200, "application/json", '{"ok":true,"polling":true}')

        elif p == "/api/table/pi_stop":
            _pi_poll_stop(s)
            self._r(200, "application/json", '{"ok":true,"polling":false}')

        elif p == "/api/table/reset_hand":
            _stop_guided_deal(s)
            with s.table_lock:
                s.rodney_downs = {}
                s.rodney_flipped_up = None
                s.slot_pending = {}
                s.slot_empty = {}
                s.verify_queue = []
                s.pending_verify = None
                s.pi_prev_slots = {}
                s.folded_players = set()
                s.freezes = {}
                _table_log_add(s, "Remote hand cleared")
                s.table_state_version += 1
            _update_flash_for_deal_state(s)
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/yolo/recognize":
            # Pi scanner sends a batch of slot crops for YOLO inference on Neo's
            # model + MPS. Body: {"slots": [{"slot": N, "image": "<base64 jpeg>"}, ...]}
            items = data.get("slots", [])
            results = []
            if s.monitor._yolo_model is None:
                return self._r(503, "application/json",
                               json.dumps({"error": "YOLO model not loaded"}))
            for item in items:
                slot_n = item.get("slot")
                img_b64 = item.get("image", "")
                try:
                    raw = base64.b64decode(img_b64)
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                except Exception as e:
                    results.append({"slot": slot_n, "error": f"decode: {e}"})
                    continue
                if crop is None or crop.size == 0:
                    results.append({"slot": slot_n, "error": "empty crop"})
                    continue
                label, conf = s.monitor._recognize_yolo(crop)
                parsed = _parse_card_any(label)
                entry = {"slot": slot_n, "confidence": round(float(conf), 3)}
                if parsed and label != "No card":
                    entry["recognized"] = True
                    entry["rank"] = parsed["rank"]
                    entry["suit"] = parsed["suit"]
                else:
                    entry["recognized"] = False
                results.append(entry)
            self._r(200, "application/json", json.dumps({"slots": results}))

        elif p == "/api/table/flip_up":
            # Rodney picked which of his 2 initial down cards to flip face-up.
            # Keep the card in rodney_downs — the physical card stays in the
            # slot and still counts toward the initial deal count. Mark the
            # slot in rodney_flipped_up, feed the card into last_card so the
            # confirm flow treats it as an up card for this round, and blink
            # the slots LED so the dealer knows which physical card to pull
            # out and show the table.
            try:
                slot_num = int(data.get("slot"))
            except (TypeError, ValueError):
                return self._r(400, "application/json", '{"ok":false,"error":"bad slot"}')
            rodney = next((p2 for p2 in s.game_engine.players if p2.is_remote), None)
            RANK_TO_NAME = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}
            SUIT_TO_NAME = {"spades": "Spades", "hearts": "Hearts",
                            "diamonds": "Diamonds", "clubs": "Clubs"}
            with s.table_lock:
                d = s.rodney_downs.get(slot_num)
                if d is None:
                    return self._r(400, "application/json",
                                   '{"ok":false,"error":"slot not in rodney_downs"}')
                s.rodney_flipped_up = {
                    "rank": d["rank"], "suit": d["suit"], "slot": slot_num,
                }
                if rodney:
                    rank_nm = RANK_TO_NAME.get(d["rank"], d["rank"])
                    suit_nm = SUIT_TO_NAME.get(d["suit"], d["suit"])
                    s.monitor.last_card[rodney.name] = f"{rank_nm} of {suit_nm}"
                    # "corrected" tells the Brio batch scan + missing-card
                    # check to skip Rodney — we already know his flipped-up
                    # card from the Pi slot, no need for YOLO on his zone.
                    s.monitor.zone_state[rodney.name] = "corrected"
                _table_log_add(
                    s,
                    f"Slot {slot_num}: flipping up ({d['rank']}{d['suit'][0]})",
                )
                s.table_state_version += 1
            # Blink the slots LED so the dealer can spot the chosen physical
            # card at a glance. Skipped when the Pi is offline.
            if not s.pi_offline:
                _pi_slot_led(s, slot_num, "blink")
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/table/fold":
            name = str(data.get("player", "")).strip()
            folded = bool(data.get("folded", True))
            ge = s.game_engine
            valid = next((pl.name for pl in ge.players if pl.name.lower() == name.lower()), None)
            if not valid:
                return self._r(400, "application/json", '{"ok":false,"error":"unknown player"}')
            with s.table_lock:
                if folded:
                    s.folded_players.add(valid)
                else:
                    s.folded_players.discard(valid)
                _table_log_add(s, f"{valid} {'folded' if folded else 'unfolded'}")
                s.table_state_version += 1
            self._r(200, "application/json",
                    json.dumps({"ok": True, "player": valid, "folded": folded}))

        elif p == "/api/table/verify":
            action = data.get("action", "")
            ok = False
            err = None
            if action == "confirm":
                pv = s.pending_verify
                if pv and pv.get("guess"):
                    g = pv["guess"]
                    if g.get("rank") and g.get("suit"):
                        ok = _resolve_verify(s, {"rank": g["rank"], "suit": g["suit"]})
                    else:
                        err = "guess missing rank/suit"
                else:
                    err = "no active verify"
            elif action == "override":
                rank = str(data.get("rank", "")).upper()
                suit = str(data.get("suit", "")).lower()
                if rank in {"A","2","3","4","5","6","7","8","9","10","J","Q","K"} and \
                        suit in {"clubs","diamonds","hearts","spades"}:
                    ok = _resolve_verify(s, {"rank": rank, "suit": suit})
                else:
                    # Legacy "code" path (e.g. "Ac" / "10h")
                    parsed = _parse_card_code(data.get("code", ""))
                    if parsed:
                        ok = _resolve_verify(s, parsed)
                    else:
                        err = "invalid rank/suit"
            elif action == "rescan":
                pv = s.pending_verify
                if not pv:
                    err = "no active verify"
                else:
                    slot = pv.get("slot")
                    result = _pi_slot_scan(s, slot) if slot else None
                    if result is None:
                        err = "pi unreachable"
                    elif not result.get("present"):
                        new_guess = {"rank": "", "suit": "", "confidence": 0.0}
                        new_prompt = f"Slot {slot}: rescan — no card present."
                        with s.table_lock:
                            pv["guess"] = new_guess
                            pv["prompt"] = new_prompt
                            pv["rescan_id"] = int(pv.get("rescan_id", 0)) + 1
                            s.slot_pending.pop(slot, None)
                            s.table_state_version += 1
                        ok = True
                    else:
                        card = result.get("card") or {}
                        conf = float(card.get("confidence", 0.0))
                        if card.get("rank") and card.get("suit"):
                            new_guess = {
                                "rank": card["rank"],
                                "suit": card["suit"],
                                "confidence": round(conf, 2),
                            }
                            new_prompt = (
                                f"Slot {slot} rescan: {int(conf*100)}%. "
                                f"Confirm or correct."
                            )
                            with s.table_lock:
                                pv["guess"] = new_guess
                                pv["prompt"] = new_prompt
                                pv["rescan_id"] = int(pv.get("rescan_id", 0)) + 1
                                s.slot_pending[slot] = dict(new_guess)
                                s.table_state_version += 1
                            ok = True
                        else:
                            new_guess = {"rank": "", "suit": "", "confidence": 0.0}
                            new_prompt = f"Slot {slot}: rescan — card not recognized."
                            with s.table_lock:
                                pv["guess"] = new_guess
                                pv["prompt"] = new_prompt
                                pv["rescan_id"] = int(pv.get("rescan_id", 0)) + 1
                                s.slot_pending.pop(slot, None)
                                s.table_state_version += 1
                            ok = True
            else:
                err = "bad action"
            resp = {"ok": ok}
            if err:
                resp["error"] = err
            self._r(200, "application/json", json.dumps(resp))

        elif p == "/api/table/mark":
            # Rodney toggles a slot's "to replace" mark during betting.
            # Body: {"slot": N, "marked": true/false}
            try:
                slot_num = int(data.get("slot", 0))
            except (TypeError, ValueError):
                slot_num = 0
            marked = bool(data.get("marked", False))
            if slot_num <= 0 or slot_num not in s.rodney_downs:
                self._r(400, "application/json",
                        '{"ok":false,"error":"invalid slot"}')
            elif s.rodney_drew_this_hand:
                self._r(400, "application/json",
                        '{"ok":false,"error":"draw already taken this hand"}')
            else:
                ge = s.game_engine
                max_marks = _max_draw_for_game(ge, s.rodney_draws_done)
                with s.table_lock:
                    if marked:
                        if len(s.rodney_marked_slots) >= max_marks and \
                                slot_num not in s.rodney_marked_slots:
                            self._r(400, "application/json",
                                    json.dumps({"ok": False,
                                                "error": f"max {max_marks} cards"}))
                            return
                        s.rodney_marked_slots.add(slot_num)
                    else:
                        s.rodney_marked_slots.discard(slot_num)
                    s.table_state_version += 1
                self._r(200, "application/json", json.dumps({
                    "ok": True,
                    "marked_slots": sorted(s.rodney_marked_slots),
                }))

        elif p == "/api/table/request_cards":
            # Rodney submits his draw choice. With 0 marks the hand skips
            # straight to betting round 2; with 1+ marks the slots clear,
            # LEDs light, and the guided replace flow starts.
            if s.rodney_drew_this_hand:
                self._r(400, "application/json",
                        '{"ok":false,"error":"draw already taken"}')
            else:
                slots_to_replace = sorted(s.rodney_marked_slots)
                # Remember the old card code per slot so the guided loop can
                # detect "card changed" even when the Pi's /scan polling is
                # too slow to catch the present=false moment between swap.
                previous_cards = {}
                with s.table_lock:
                    for slot in slots_to_replace:
                        s.rodney_downs.pop(slot, None)
                        code = s.pi_prev_slots.pop(slot, None)
                        if code:
                            previous_cards[slot] = code
                    s.rodney_marked_slots = set()
                    s.rodney_drew_this_hand = True
                    s.table_state_version += 1
                if slots_to_replace:
                    _table_log_add(s,
                        f"Rodney requested {len(slots_to_replace)} card(s): "
                        f"slots {slots_to_replace}"
                    )
                    log.log(f"[CONSOLE] Rodney draw: replacing slots "
                            f"{slots_to_replace}")
                    _start_guided_replace(s, slots_to_replace, previous_cards)
                else:
                    # Rodney stood pat — no cards replaced. Mark the draw
                    # as done and advance to the post-draw betting round.
                    _table_log_add(s, "Rodney stood pat (no cards replaced)")
                    with s.table_lock:
                        s.rodney_draws_done += 1
                        s.console_state = "betting"
                        s.console_betting_round = s.rodney_draws_done + 1
                        s.table_state_version += 1
                    log.log(
                        f"[CONSOLE] Rodney stood pat — draw "
                        f"{s.rodney_draws_done} done → betting round "
                        f"{s.console_betting_round}"
                    )
                self._r(200, "application/json", json.dumps({
                    "ok": True, "slots": slots_to_replace,
                }))

        # --- Console (dealer phone UI) ---

        elif p == "/api/console/state":
            ge = s.game_engine
            # Include zone-recognized cards with details — only for players
            # checked in as active. Inactive players' zones are still
            # calibrated but not in play this hand.
            zone_cards = {}
            for z in s.cal.zones:
                name = z["name"]
                if name not in s.console_active_players:
                    continue
                card = s.monitor.last_card.get(name, "")
                details = s.monitor.recognition_details.get(name, {})
                zone_cards[name] = {
                    "card": card if card and card != "No card" else "",
                    "yolo": details.get("yolo", ""),
                    "yolo_conf": details.get("yolo_conf", 0),
                    "claude": details.get("claude", ""),
                    "duplicate": False,
                }
            # Flag duplicates against current round AND all prior rounds this hand
            seen = {}  # card -> "player round N" descriptor
            for c in s.console_hand_cards:
                seen[c["card"]] = f"{c['player']} (round {c['round']})"
            for name in s.console_active_players:
                zi = zone_cards.get(name)
                if not zi or not zi["card"]:
                    continue
                card = zi["card"]
                if card in seen:
                    zi["duplicate"] = True
                    # If the prior is in current zone_cards, flag that too
                    prior_name = seen[card].split(" ")[0]
                    if prior_name in zone_cards:
                        zone_cards[prior_name]["duplicate"] = True
                else:
                    seen[card] = name
            game_in_progress = (
                ge.current_game is not None
                and s.console_state != "idle"
            )
            has_up = s.console_total_up_rounds > 0 or (
                ge.current_game is not None and any(
                    ph.type.value == "hit_round" for ph in ge.current_game.phases
                )
            )
            # Map console_state -> phase label + action button spec for the UI.
            if not s.night_active:
                phase_label = "Night not started"
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            elif ge.current_game is None or s.console_state == "idle":
                phase_label = "Choose a game"
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            elif s.console_state == "dealing":
                phase_label = "Dealing"
                action_label = "Confirm Cards"
                action_endpoint = "/api/console/confirm"
                # Disabled for all-down rounds (auto-advances when guided
                # finishes) and whenever a guided session is in flight so
                # the dealer can't commit up cards mid-deal for the
                # leading-down guided flow in stud games.
                action_enabled = has_up and s.guided_deal is None
            elif s.console_state == "betting":
                rnd = s.console_betting_round or max(1, s.console_up_round)
                phase_label = f"Betting (round {rnd})"
                action_label = "Pot is right"
                action_endpoint = "/api/console/next_round"
                action_enabled = True
            elif s.console_state == "draw":
                phase_label = "Draw — Rodney picking"
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            elif s.console_state == "replacing":
                phase_label = "Replacing Rodney's cards"
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            elif s.console_state == "hand_over":
                phase_label = "Hand over"
                action_label = "New Hand"
                action_endpoint = "/api/console/end"
                action_enabled = True
            else:
                phase_label = s.console_state
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            self._r(200, "application/json", json.dumps({
                "active_players": s.console_active_players,
                "all_players": PLAYER_NAMES,
                "games": ge.get_game_list(),
                "game_groups": ge.get_game_groups(),
                "brio_settle_s": round(s.brio_settle_s, 2),
                "pi_presence_threshold": round(s.pi_presence_threshold, 1),
                "dealer": ge.get_dealer().name,
                "hand": ge.get_hand_state(),
                "last_round_cards": s.console_last_round_cards,
                "zone_cards": zone_cards,
                "yolo_min_conf": s.monitor.yolo_min_conf,
                "up_round": s.console_up_round,
                "total_up_rounds": s.console_total_up_rounds,
                "scan_phase": s.console_scan_phase,
                "night_active": s.night_active,
                "console_state": s.console_state,
                "game_in_progress": game_in_progress,
                "phase_label": phase_label,
                "action_label": action_label,
                "action_endpoint": action_endpoint,
                "action_enabled": action_enabled,
                "current_game": ge.current_game.name if ge.current_game else "",
            }))

        elif p == "/api/console/players":
            names = data.get("players", [])
            valid = [n for n in names if n in PLAYER_NAMES]
            s.console_active_players = valid if valid else list(PLAYER_NAMES)
            log.log(f"[CONSOLE] Active players: {', '.join(s.console_active_players)}")
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/console/set_dealer":
            ge = s.game_engine
            name = data.get("dealer", "")
            for i, p2 in enumerate(ge.players):
                if p2.name.lower() == name.lower():
                    ge.dealer_index = i
                    ge._update_dealer()
                    log.log(f"[CONSOLE] Dealer set to {p2.name}")
                    break
            self._r(200, "application/json", json.dumps({"dealer": ge.get_dealer().name}))

        elif p == "/api/console/start_night":
            # Start the night: rotate log file, flip night_active, accept
            # any initial settings (dealer/players/thresholds) at the same
            # time so the modal submits once.
            archive = log.start_night()
            s.night_active = True
            s.console_state = "idle"
            self._apply_settings(s, data)
            log.log("[CONSOLE] Poker night started")
            self._r(200, "application/json", json.dumps({
                "ok": True, "archive": archive,
            }))

        elif p == "/api/console/settings":
            # Mid-night adjustments: same payload as start_night, but doesn't
            # rotate the log or toggle night state.
            self._apply_settings(s, data)
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/console/exit_poker":
            log.log("[CONSOLE] Exit Poker — closing night")
            _stop_guided_deal(s)
            log.end_night()
            s.night_active = False
            s.console_state = "idle"
            self._r(200, "application/json", '{"ok":true,"exiting":true}')
            # Schedule a brief deferred exit so the response can flush first.
            def _bye():
                time.sleep(0.3)
                os._exit(0)
            Thread(target=_bye, daemon=True).start()

        elif p == "/api/console/force_scan":
            # Dealer clicked "Waiting for cards..." — skip motion detection
            # and scan every active, non-frozen zone right now.
            frame = s.latest_frame
            if frame is None:
                return self._r(503, "application/json",
                               '{"ok":false,"error":"no frame yet"}')
            zone_crops = {}
            for z in s.cal.zones:
                name = z["name"]
                if name not in s.console_active_players:
                    continue
                if s.monitor.zone_state.get(name) == "corrected":
                    continue
                if s.freezes.get(name, 0) >= 3:
                    continue
                crop = s.monitor._crop(frame, z)
                if crop is None or crop.size == 0:
                    continue
                zone_crops[name] = crop.copy()
                s.monitor.pending[name] = True
            if not zone_crops:
                return self._r(200, "application/json",
                               '{"ok":true,"scanned":0}')
            s.console_scan_phase = "scanned"
            Thread(target=s.monitor._recognize_batch,
                   args=(zone_crops,), daemon=True).start()
            log.log(f"[CONSOLE] Force scan of {len(zone_crops)} zones")
            self._r(200, "application/json",
                    json.dumps({"ok": True, "scanned": len(zone_crops)}))

        elif p == "/api/console/deal":
            ge = s.game_engine
            game_name = data.get("game", "")
            if game_name not in ge.templates:
                self._r(400, "application/json", json.dumps({"error": f"Unknown game: {game_name}"}))
            else:
                result = ge.new_hand(game_name)
                s.console_last_round_cards = []
                s.console_hand_cards = []
                # Count total up-card rounds from template. Games with an
                # open-ended HIT_ROUND (7/27) aren't a fixed count — mark
                # them as 0 (unbounded) so the UI stays in "confirmed /
                # Next Round" flow instead of switching to idle.
                s.console_up_round = 0
                template = ge.templates[game_name]
                has_hit_round = any(
                    phase.type.value == "hit_round" and phase.card_type == "up"
                    for phase in template.phases
                )
                if has_hit_round:
                    up_rounds = 0  # 0 = unbounded
                else:
                    up_rounds = 0
                    for phase in template.phases:
                        if phase.type.value == "deal" and "up" in phase.pattern:
                            up_rounds += 1
                        elif phase.type.value == "community":
                            up_rounds += 1
                s.console_total_up_rounds = up_rounds
                log.log(f"[CONSOLE] New hand: {game_name}, dealer: {result['dealer']}")
                if result.get("wild_label"):
                    log.log(f"[CONSOLE] {result['wild_label']}")
                # Quick Pi reachability check. If the Pi's down, set the
                # pi_offline flag so every Pi call (flash/hold, /slots,
                # slot LEDs, etc.) short-circuits for the rest of the game.
                # Flag clears on the next Deal (re-checked below).
                pi_up = _pi_ping(s)
                s.pi_offline = not pi_up
                s.pi_flash_held = False  # reset our tracker regardless
                log.log(f"[PI] Deal-time ping: {'reachable' if pi_up else 'OFFLINE (suppressing calls)'}")
                # Brio watching: triggered for any game that will produce up
                # cards, either via explicit up/community phases or an
                # open-ended HIT_ROUND (7/27). Baselines are captured now
                # while the table is empty of up cards, regardless of when
                # watching actually starts. Whether to watch immediately or
                # defer until guided finishes depends on whether the FIRST
                # deal phase itself contains an up card.
                leading_downs = _initial_down_count(ge)
                will_guide = leading_downs > 0 and not s.pi_offline
                needs_brio = up_rounds != 0 or has_hit_round
                if needs_brio and s.cal.ok and s.latest_frame is not None:
                    s.monitor.capture_baselines(s.latest_frame)
                    s.monitoring = True
                    if will_guide:
                        # Any game that uses guided Pi-slot dealing deals
                        # downs first — regardless of whether the first
                        # phase also contains an up card — so the dealer
                        # sweeps every zone multiple times before the up
                        # cards arrive. Defer Brio until guided completes;
                        # the per-game guided-completion branches (all-down
                        # vs mixed vs hit-round) take over from there.
                        s.console_scan_phase = "idle"
                        log.log(
                            "[CONSOLE] Baselines captured; Brio watching "
                            "deferred until guided downs are validated"
                        )
                    else:
                        s.console_scan_phase = "watching"
                        s._zones_with_motion = set()
                        log.log(
                            f"[CONSOLE] Watching {ge.get_dealer().name}'s "
                            f"zone for first card"
                        )
                # Start the hand fresh: clear Rodney-side state and turn the
                # scanner LEDs on so the initial down cards get good scans.
                with s.table_lock:
                    s.rodney_downs = {}
                    s.rodney_flipped_up = None
                    s.slot_pending = {}
                    s.slot_empty = {}
                    s.verify_queue = []
                    s.pending_verify = None
                    s.pi_prev_slots = {}
                    s.folded_players = set()
                    s.freezes = {name: 0 for name in s.console_active_players}
                    s.rodney_marked_slots = set()
                    s.rodney_drew_this_hand = False
                    s.rodney_draws_done = 0
                    s.console_betting_round = 0
                    s.console_trailing_done = False
                    s.stats = {"yolo_right": 0, "yolo_wrong": 0,
                               "claude_right": 0, "claude_wrong": 0}
                    s._zones_with_motion = set()
                    s._missing_speech_count = {}
                    s.table_state_version += 1
                # Make sure any stale guided session from a prior hand is gone.
                _stop_guided_deal(s)
                s.console_state = "dealing"
                # Use the slot-by-slot guided flow for the leading down cards
                # in the first deal phase — 2 for 7 Card Stud, 2 for Hold'em,
                # 3 for Follow the Queen, 5 for 5 Card Draw. Games with up
                # cards in the same phase start Brio watching only after
                # guided completes, so dealer hand motion over local zones
                # during down-card dealing doesn't trip false alarms.
                if will_guide:
                    _start_guided_deal(s, leading_downs)
                else:
                    _update_flash_for_deal_state(s)
                self._r(200, "application/json", json.dumps(result))

        elif p == "/api/console/confirm":
            ge = s.game_engine
            # Collect round cards in deal order (clockwise from dealer's left)
            dealer_idx = ge.dealer_index
            round_cards = []
            for i in range(1, len(ge.players) + 1):
                p2 = ge.players[(dealer_idx + i) % len(ge.players)]
                if p2.name not in s.console_active_players:
                    continue
                card = s.monitor.last_card.get(p2.name, "")
                if card and card != "No card":
                    round_cards.append({"player": p2.name, "card": card})
            # De-duplicate against previously-seen cards (prior up rounds,
            # Rodney's down cards) before the round is announced and
            # accumulated — a duplicate is almost always a misread.
            _dedup_round_cards_against_seen(s, round_cards)
            # Check Follow the Queen wild cards — announce before betting
            _check_follow_the_queen_round(s, round_cards)
            # Accumulate into hand-wide history, then clear the current-round
            # data so it stops re-appearing as "just dealt" cards (which were
            # triggering the duplicate detector against the exact same cards
            # now in the hand history).
            round_num = s.console_up_round + 1
            if round_cards:
                for c in round_cards:
                    s.console_hand_cards.append({"player": c["player"], "card": c["card"], "round": round_num})
            # For 7/27, announce each player's hand value(s) after the
            # up-cards have been accumulated + indicate who bets first.
            _announce_7_27_hand_values(s)
            _announce_poker_hand_bet_first(s)
            # 7/27 freeze tracking: on hit rounds (any round after the
            # first up-card round), a player who didn't take a card this
            # round increments their freeze count. Three freezes in a row
            # means they can't take another card this hand.
            if (ge.current_game and ge.current_game.name.startswith("7/27")
                    and round_num > 1):
                _update_7_27_freezes(s, round_cards)
            s.console_last_round_cards = []
            for z in s.cal.zones:
                zname = z["name"]
                s.monitor.zone_state[zname] = "empty"
                s.monitor.last_card[zname] = ""
                s.monitor.recognition_details[zname] = {}
                s.monitor.recognition_crops[zname] = None
            # If this was the last up-card round, go to idle. 0 means
            # unbounded (games with HIT_ROUND), never idle on count.
            if (s.console_total_up_rounds > 0
                    and round_num >= s.console_total_up_rounds):
                s.console_scan_phase = "idle"
                log.log(f"[CONSOLE] Final up round ({round_num}) confirmed — idle until End Hand")
            else:
                s.console_scan_phase = "confirmed"
                log.log(f"[CONSOLE] Cards confirmed for up round {round_num}")
            # Reset the per-player adjust-prompt cap so the next round
            # starts fresh (two prompts max per player per round).
            s._missing_speech_count = {}
            # Once the up-card round is confirmed, check Rodney's down slots
            # for anything below the auto-accept threshold. Those slots get
            # queued and (on LED-equipped hardware) will start blinking; the
            # /table modal will appear when the user removes each card.
            queued = _enqueue_down_card_verifies(s)
            if queued:
                log.log(f"[CONSOLE] Down-card verify queued for slots {queued}")
            # If Rodney flipped a card (7/27 2-down), stop its slots blink-
            # hint now that the round is confirmed and the physical card is
            # on the table.
            if s.rodney_flipped_up and not s.pi_offline:
                _pi_slot_led(s, int(s.rodney_flipped_up["slot"]), "off")
            _update_flash_for_deal_state(s)
            # /table polls on state version; the hand-wide up-card history
            # just grew so bump the version or clients 304 and never see
            # the new up cards.
            with s.table_lock:
                s.table_state_version += 1
            s.console_state = "betting"
            self._r(200, "application/json", json.dumps({"ok": True}))

        elif p == "/api/console/next_round":
            ge = s.game_engine
            # All-down games (5 Card Draw, 3 Toed Pete) track betting rounds
            # independently from console_up_round because they have no up
            # rounds. 5CD flow:
            #   betting round 1 + Pot-is-right → draw state (Rodney's Request
            #     Cards button appears on /table; marking still enabled)
            #   replacement completes → betting round 2
            #   betting round 2 + Pot-is-right → hand_over
            if s.console_total_up_rounds == 0 and s.console_betting_round > 0:
                has_draw = _game_has_draw_phase(ge)
                total_draws = _total_draw_phases(ge)
                # If there are more DRAW phases left (multi-draw games like
                # 3 Toed Pete, or the single draw in 5 Card Draw), loop back
                # into a draw state. Reset Rodneys drew flag + marks so the
                # /table can collect fresh picks for this draw.
                if has_draw and s.rodney_draws_done < total_draws:
                    with s.table_lock:
                        s.console_state = "draw"
                        s.rodney_drew_this_hand = False
                        s.rodney_marked_slots = set()
                        s.table_state_version += 1
                    log.log(
                        f"[CONSOLE] Betting round {s.console_betting_round} "
                        f"done → draw {s.rodney_draws_done + 1}/{total_draws}"
                    )
                    return self._r(200, "application/json", json.dumps({
                        "ok": True, "state": "draw",
                    }))
                # No draws remain (either a non-draw all-down game or the
                # last post-draw betting of a multi-draw game).
                s.console_state = "hand_over"
                log.log("[CONSOLE] All-down betting complete → hand_over")
                with s.table_lock:
                    s.table_state_version += 1
                return self._r(200, "application/json", json.dumps({
                    "ok": True, "state": "hand_over",
                }))
            # If trailing guided has already run this hand, this Pot-is-right
            # is the final betting round's — go to hand_over without touching
            # up_round / baselines.
            if s.console_trailing_done:
                s.console_scan_phase = "idle"
                s.console_state = "hand_over"
                log.log("[CONSOLE] Final betting done → hand_over")
                with s.table_lock:
                    s.table_state_version += 1
                return self._r(200, "application/json", json.dumps({
                    "ok": True, "state": "hand_over",
                }))
            # Advance round counter
            s.console_up_round += 1
            beyond_last_up = (
                s.console_total_up_rounds > 0
                and s.console_up_round >= s.console_total_up_rounds
            )
            # If we've finished all up rounds AND the game has trailing down
            # cards (7CS 7th, FTQ 7th), start a guided session for those
            # slots rather than jumping to hand_over.
            trailing = _trailing_down_slots(ge) if beyond_last_up else []
            if beyond_last_up and trailing and not s.pi_offline:
                s.console_scan_phase = "idle"
                log.log(
                    f"[CONSOLE] All up rounds done — starting trailing "
                    f"down deal for slots {trailing}"
                )
                _start_guided_trailing_deal(s, trailing)
                # State was set to "dealing" inside the starter; bump the
                # version so /table and /api/console/state re-fetch.
                with s.table_lock:
                    s.table_state_version += 1
                return self._r(200, "application/json", json.dumps({
                    "hand": ge.get_hand_state(),
                    "up_round": s.console_up_round,
                    "total_up_rounds": s.console_total_up_rounds,
                }))
            # Recapture baselines and resume watching dealer — but only if
            # there's still an up round ahead. If we've finished all up
            # rounds with no trailing downs, the hand is over.
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
                for z in s.cal.zones:
                    s.monitor.zone_state[z["name"]] = "empty"
                    s.monitor.last_card[z["name"]] = ""
                    s.monitor.recognition_details[z["name"]] = {}
                    s.monitor.recognition_crops[z["name"]] = None
                s.console_scan_phase = "idle" if beyond_last_up else "watching"
                s._zones_with_motion = set()
                if beyond_last_up:
                    log.log("[CONSOLE] No more up rounds — idle until End Hand")
                    s.console_state = "hand_over"
                else:
                    log.log(f"[CONSOLE] Baselines recaptured, watching {ge.get_dealer().name}'s zone")
                    s.console_state = "dealing"
            log.log(f"[CONSOLE] Next Round — up round {s.console_up_round}/{s.console_total_up_rounds}")
            # Also queue any pending down-card scans so games with a final
            # down (no Confirm Cards) still get a chance to verify.
            queued = _enqueue_down_card_verifies(s)
            if queued:
                log.log(f"[CONSOLE] Down-card verify queued for slots {queued}")
            # Advancing rounds may change what the next expected card is —
            # e.g. after the 4th up round the 7th-card down becomes next, so
            # the LEDs need to come back on.
            _update_flash_for_deal_state(s)
            with s.table_lock:
                s.table_state_version += 1
            self._r(200, "application/json", json.dumps({
                "hand": ge.get_hand_state(),
                "up_round": s.console_up_round,
                "total_up_rounds": s.console_total_up_rounds,
            }))

        elif p == "/api/console/end":
            _stop_guided_deal(s)
            ge = s.game_engine
            # Recognition stats for the hand just ending.
            yr = s.stats.get("yolo_right", 0)
            yw = s.stats.get("yolo_wrong", 0)
            cr = s.stats.get("claude_right", 0)
            cw = s.stats.get("claude_wrong", 0)
            yolo_total = yr + yw
            claude_total = cr + cw
            total = yolo_total + claude_total
            if total > 0:
                def _pct(n, d):
                    return f"{(100.0 * n / d):.0f}%" if d else "—"
                log.log(
                    f"[STATS] Hand recognition: {total} cards total, "
                    f"YOLO {yr}/{yolo_total} right ({_pct(yr, yolo_total)}), "
                    f"Claude {cr}/{claude_total} right ({_pct(cr, claude_total)})"
                )
            result = ge.end_hand()
            _skip_inactive_dealer(s)
            result["next_dealer"] = ge.get_dealer().name
            s.console_last_round_cards = []
            s.console_hand_cards = []
            s.console_up_round = 0
            s.console_total_up_rounds = 0
            s.console_trailing_done = False
            s.monitoring = False
            s.console_scan_phase = "idle"
            s.console_state = "idle"
            # Reset all zone states
            for z in s.cal.zones:
                s.monitor.zone_state[z["name"]] = "empty"
                s.monitor.last_card[z["name"]] = ""
                s.monitor.recognition_details[z["name"]] = {}
                s.monitor.recognition_crops[z["name"]] = None
            log.log(f"[CONSOLE] Hand over — next dealer: {result['next_dealer']}")
            # Clear Rodney-side hand state and turn scanner LEDs off now
            # that no cards are expected.
            with s.table_lock:
                s.rodney_downs = {}
                s.rodney_flipped_up = None
                s.slot_pending = {}
                s.slot_empty = {}
                s.verify_queue = []
                s.pending_verify = None
                s.pi_prev_slots = {}
                s.folded_players = set()
                s.freezes = {}
                s.table_state_version += 1
            _update_flash_for_deal_state(s)
            self._r(200, "application/json", json.dumps(result))

        elif p == "/api/console/advance_dealer":
            ge = s.game_engine
            ge.advance_dealer()
            _skip_inactive_dealer(s)
            log.log(f"[CONSOLE] Dealer advanced to {ge.get_dealer().name}")
            self._r(200, "application/json", json.dumps({"dealer": ge.get_dealer().name}))

        elif p == "/api/console/correct":
            # Batch corrections: [{player, rank, suit}, ...]
            corrections = data.get("corrections", [])
            changed_any = False
            for c in corrections:
                player = c.get("player", "")
                rank = c.get("rank", "")
                suit = c.get("suit", "")
                if player and rank and suit:
                    RANK_TO_NAME = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}
                    SUIT_TO_NAME = {"spades": "Spades", "hearts": "Hearts",
                                    "diamonds": "Diamonds", "clubs": "Clubs"}
                    rank_name = RANK_TO_NAME.get(rank, rank)
                    suit_name = SUIT_TO_NAME.get(suit, suit)
                    new_card = f"{rank_name} of {suit_name}"
                    old_card = s.monitor.last_card.get(player, "")
                    # Tally this as a miss for whichever recognizer produced
                    # the card the user just overrode.
                    prior_details = s.monitor.recognition_details.get(player, {}) or {}
                    prior_source = prior_details.get("source")
                    if old_card != new_card and prior_source in ("yolo", "claude"):
                        _stats_bump(s, f"{prior_source}_right", -1)
                        _stats_bump(s, f"{prior_source}_wrong", +1)
                    s.monitor.last_card[player] = new_card
                    s.monitor.zone_state[player] = "corrected"
                    s.monitor.recognition_details[player] = {
                        "yolo": s.monitor.recognition_details.get(player, {}).get("yolo", ""),
                        "yolo_conf": s.monitor.recognition_details.get(player, {}).get("yolo_conf", 0),
                        "claude": s.monitor.recognition_details.get(player, {}).get("claude", ""),
                        "final": new_card,
                        "corrected": True,
                    }
                    # If the corrected card already landed in console_hand_cards
                    # (post-Confirm correction), update that entry in place so
                    # dedup / wild recompute / hand value / best hand all see
                    # the new value. Match on BOTH player AND the old card
                    # value — otherwise a mid-round correction (the common
                    # case) would overwrite a previous rounds entry because
                    # the current round hasnt been appended yet.
                    for entry in reversed(s.console_hand_cards):
                        if (entry.get("player") == player
                                and entry.get("card") == old_card):
                            entry["card"] = new_card
                            break
                    # Save corrected crop to training_data for future YOLO
                    # training. Delete the prior (wrong-label) save for this
                    # zone so the bad label doesnt poison the dataset.
                    crop = s.monitor.recognition_crops.get(player)
                    if crop is not None:
                        removed = s.monitor._delete_last_save(player)
                        if removed:
                            log.log(f"[CONSOLE] Removed wrong-label training save for {player}")
                        s.monitor._save(player, crop, new_card)
                        log.log(f"[CONSOLE] Saved correction to training_data: {new_card}")
                    log.log(f"[CONSOLE] Corrected {player}: {old_card} -> {new_card}")
                    if old_card != new_card:
                        changed_any = True
            # Re-derive Follow-the-Queen wild ranks from the corrected
            # history. If the corrected card was the follower of a queen,
            # this announces the new wild rank.
            if changed_any:
                _recompute_follow_the_queen(s)
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/console/yolo_conf":
            val = data.get("value")
            if val is not None:
                s.monitor.yolo_min_conf = max(0.0, min(1.0, float(val)))
                log.log(f"[CONSOLE] YOLO min confidence: {s.monitor.yolo_min_conf:.0%}")
            self._r(200, "application/json", json.dumps({"yolo_min_conf": s.monitor.yolo_min_conf}))

        else:
            self._r(404,"text/plain","Not found")

    def _r(self, code, ct, body):
        try:
            self.send_response(code)
            # Default text/HTML/JSON responses to UTF-8 so browsers dont
            # mis-decode em-dashes and other multibyte chars as Windows-1252
            # (which is what the iPhone Safari view was showing as â€").
            if (ct.startswith("text/") or ct == "application/json") \
                    and "charset" not in ct.lower():
                ct = ct + "; charset=utf-8"
            self.send_header("Content-Type", ct)
            if ct == "image/jpeg":
                self.send_header("Cache-Control","no-store,no-cache,max-age=0")
            self.end_headers()
            self.wfile.write(body.encode() if isinstance(body,str) else body)
        except (ConnectionResetError, BrokenPipeError):
            pass

    def _jpeg(self, frame):
        if frame is None:
            log.log("[SNAPSHOT] no frame yet")
            return self._r(503, "text/plain", "No frame")
        try:
            h, w = frame.shape[:2]
        except Exception as e:
            log.log(f"[SNAPSHOT] bad frame: {e}")
            return self._r(500, "text/plain", f"bad frame: {e}")
        j = to_jpeg(frame, 80)
        if not j:
            log.log(f"[SNAPSHOT] JPEG encode failed for {w}x{h} frame")
            return self._r(500, "text/plain", "JPEG encode failed")
        log.log(f"[SNAPSHOT] served {w}x{h} JPEG ({len(j)} bytes)")
        return self._r(200, "image/jpeg", j)

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
            "deal_mode": _deal_mode_json(s),
            "collect_mode": _collect_mode_json(s),
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
  <button class="btn-blue" onclick="ensureLogWindow();startTest()">Test Recognition</button>
  <button class="btn-blue" id="btn-deal" onclick="toggleDeal()">Test Dealing</button>
  <button class="btn-blue" onclick="resetBaselines()">Reset Baselines</button>
  <button class="btn-blue" onclick="saveSnapshot()">Snapshot</button>
  <button class="btn-blue" id="btn-collect" onclick="toggleCollect()">Collect Data</button>
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
<div id="collect-panel" style="display:none;background:#1b5e20;padding:12px;border-radius:8px;margin:8px 0">
  <h3>Data Collection</h3>
  <div id="collect-info">
    <p style="font-size:1.1em;color:#fff;margin:8px 0">
      Deal rank <span id="collect-rank" style="color:#ff0;font-weight:bold;font-size:1.3em">—</span> to each player:
    </p>
    <div id="collect-assignments" style="margin:6px 0"></div>
    <p style="font-size:.9em;color:#aaa">
      Pass <span id="collect-pass">1</span>/<span id="collect-pass-total">4</span> |
      Card <span id="collect-idx">0</span>/<span id="collect-total">52</span>
    </p>
    <div style="background:#333;border-radius:4px;height:8px;margin:8px 0">
      <div id="collect-progress" style="background:#4caf50;height:100%;border-radius:4px;width:0%"></div>
    </div>
  </div>
  <div id="collect-done" style="display:none">
    <p style="font-size:1.2em;color:#4caf50">All cards collected!</p>
  </div>
  <div style="margin-top:8px">
    <button class="btn-green" id="collect-go-btn" onclick="collectGo()" style="font-size:1.1em;padding:10px 20px">Start</button>
    <button class="btn-orange" id="collect-pause-btn" onclick="collectPause()" style="display:none">Pause</button>
    <button class="btn-blue" id="collect-redo-btn" onclick="collectRedo()" style="display:none">Redo</button>
    <span id="collect-countdown" style="font-size:2.5em;color:#ff0;margin:0 16px;display:none"></span>
    <span id="collect-phase-label" style="font-size:1em;color:#aaa;display:none"></span>
    <button class="btn-red" onclick="toggleCollect()" style="margin-left:8px">Stop</button>
  </div>
</div>
<div id="deal-panel" style="display:none;background:#0f3460;padding:12px;border-radius:8px;margin:8px 0">
  <h3>Test Dealing</h3>
  <p style="margin:4px 0">Game: <span id="deal-game" style="color:#4fc3f7;font-size:1.1em">—</span></p>
  <div id="deal-game-input" style="margin:6px 0">
    <div style="margin:4px 0">
      <label style="font-size:.85em;color:#888">Dealer: </label>
      <select id="deal-dealer" style="padding:6px;border-radius:4px;border:1px solid #444;background:#1a1a2e;color:#e0e0e0"
        onchange="setDealer()">
      </select>
      <span id="deal-order-display" style="font-size:.8em;color:#888;margin-left:8px"></span>
    </div>
    <div style="margin:4px 0">
      <label style="font-size:.85em;color:#888">Game (dictate or type): </label>
      <input id="deal-input" type="text" placeholder="e.g. 'Follow the Queen'"
        style="width:60%;padding:8px;border-radius:6px;border:1px solid #444;background:#1a1a2e;color:#e0e0e0;font-size:1em">
      <button class="btn-green" onclick="submitGameName()" style="margin-left:4px">Start</button>
    </div>
  </div>
  <div id="deal-status" style="margin:6px 0;display:none">
    <p style="color:#4fc3f7;font-size:.95em">Watching: <span id="deal-active-player" style="font-weight:bold">—</span>'s zone
      (round <span id="deal-round">—</span>/<span id="deal-total">—</span>)</p>
  </div>
  <div id="deal-cards" style="margin:8px 0"></div>
  <button class="btn-orange" onclick="clearDeal()">Clear</button>
  <button class="btn-red" onclick="toggleDeal()">Stop Dealing</button>
</div>
<div id="main">
  <div id="left">
    <img id="tableimg" src="/snapshot/cropped" style="width:100%;cursor:pointer"
         onclick="this.src='/snapshot/cropped?'+Date.now()">
    <div style="margin-top:8px">
      <button class="btn-off" style="padding:3px 8px;font-size:.75em" onclick="openLogWindow()">Log Window</button>
      <a href="/training" style="color:#4fc3f7;font-size:.8em;margin-left:8px">Training data</a>
    </div>
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
  if(!monitoring) ensureLogWindow();
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
    else {{ ensureLogWindow(); api('/api/deal/start').then(function(){{
      update();
      setTimeout(function(){{
        var inp=document.getElementById('deal-input');
        if(inp) inp.focus();
      }}, 300);
    }}); }}
  }});
}}
function clearDeal(){{
  api('/api/deal/clear').then(function(){{
    var inp=document.getElementById('deal-input');
    if(inp) inp.value='';
    update();
  }});
}}
function submitGameName(){{
  var inp=document.getElementById('deal-input');
  if(inp && inp.value.trim()) {{
    api('/api/deal/text',{{text:inp.value.trim()}}).then(update);
  }}
}}
function setDealer(){{
  var sel=document.getElementById('deal-dealer');
  if(sel) {{
    api('/api/deal/dealer',{{dealer:sel.value}}).then(function(){{
      // Show deal order
      var idx=players.indexOf(sel.value);
      var order=[];
      for(var i=1;i<=players.length;i++) order.push(players[(idx+i)%players.length]);
      var el=document.getElementById('deal-order-display');
      if(el) el.textContent='Deal: '+order.join(' → ');
    }});
  }}
}}
// Allow Enter key to submit game name
document.addEventListener('keydown',function(e){{
  if(e.key==='Enter' && document.activeElement && document.activeElement.id==='deal-input'){{
    submitGameName();
  }}
}});

function resetBaselines(){{
  if(!confirm('Clear all cards, then click OK')) return;
  api('/api/baselines').then(function(){{ log.log && update(); }});
}}
function saveSnapshot(){{ api('/api/snapshot/save'); }}

function toggleCollect(){{
  fetch('/api/state').then(function(r){{return r.json()}}).then(function(d){{
    if(d.collect_mode) api('/api/collect/stop').then(update);
    else {{ ensureLogWindow(); api('/api/collect/start').then(update); }}
  }});
}}
function collectGo(){{
  api('/api/collect/go').then(update);
}}
function collectPause(){{
  api('/api/collect/pause').then(update);
}}
function collectResume(){{
  api('/api/collect/resume').then(update);
}}
function collectRedo(){{
  api('/api/collect/redo').then(update);
}}
var _logWin=null;
function openLogWindow(){{
  if(!_logWin || _logWin.closed)
    _logWin=window.open('/logview','_logview','width=800,height=400,scrollbars=yes');
  else
    _logWin.focus();
}}
function ensureLogWindow(){{
  if(!_logWin || _logWin.closed)
    _logWin=window.open('/logview','_logview','width=800,height=400,scrollbars=yes');
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

    // Collect mode panel
    var cp=document.getElementById('collect-panel');
    var cbtn=document.getElementById('btn-collect');
    if(d.collect_mode){{
      cp.style.display='block';
      cbtn.textContent='Stop Collect';cbtn.className='btn-red';
      var cm=d.collect_mode;
      if(cm.done){{
        document.getElementById('collect-info').style.display='none';
        document.getElementById('collect-done').style.display='';
        document.getElementById('collect-scan-btn').style.display='none';
        document.getElementById('collect-next-btn').style.display='none';
      }} else {{
        document.getElementById('collect-info').style.display='';
        document.getElementById('collect-done').style.display='none';
        document.getElementById('collect-rank').textContent=cm.rank||'?';
        document.getElementById('collect-pass').textContent=cm.pass_idx+1;
        document.getElementById('collect-pass-total').textContent=cm.pass_total;
        document.getElementById('collect-idx').textContent=cm.current+1;
        document.getElementById('collect-total').textContent=cm.total;
        // Show per-player assignments
        var ah='';
        Object.keys(cm.cards||{{}}).forEach(function(p){{
          ah+='<span style="display:inline-block;margin:3px;padding:4px 10px;background:#0f3460;border-radius:4px">'
            +p+': <b>'+cm.cards[p]+'</b></span>';
        }});
        document.getElementById('collect-assignments').innerHTML=ah;
        var pct=Math.round(cm.current/cm.total*100);
        document.getElementById('collect-progress').style.width=pct+'%';
        var goBtn=document.getElementById('collect-go-btn');
        var pauseBtn=document.getElementById('collect-pause-btn');
        var redoBtn=document.getElementById('collect-redo-btn');
        var cdSpan=document.getElementById('collect-countdown');
        var phLabel=document.getElementById('collect-phase-label');
        if(cm.phase=='dealing'||cm.phase=='clearing'){{
          goBtn.style.display='none';
          pauseBtn.style.display='';
          redoBtn.style.display=cm.phase=='clearing'?'':'none';
          cdSpan.style.display='';cdSpan.textContent=cm.countdown;
          phLabel.style.display='';
          phLabel.textContent=cm.phase=='dealing'?'DEAL':'CLEAR';
          phLabel.style.color=cm.phase=='dealing'?'#4caf50':'#ff9800';
        }} else if(cm.phase=='paused'||cm.phase=='paused_new_pass'){{
          goBtn.style.display='';goBtn.textContent=cm.phase=='paused_new_pass'?'Start New Pass':'Resume';
          goBtn.onclick=cm.phase=='paused_new_pass'?collectGo:collectResume;
          pauseBtn.style.display='none';
          redoBtn.style.display='';
          cdSpan.style.display='none';
          phLabel.style.display='';phLabel.textContent='PAUSED';phLabel.style.color='#888';
        }} else {{
          goBtn.style.display='';goBtn.textContent='Start';goBtn.onclick=collectGo;
          pauseBtn.style.display='none';
          redoBtn.style.display='none';
          cdSpan.style.display='none';
          phLabel.style.display='none';
        }}
      }}
    }} else {{
      cp.style.display='none';
      cbtn.textContent='Collect Data';cbtn.className='btn-blue';
    }}

    // Deal mode panel
    var dp=document.getElementById('deal-panel');
    var dbtn=document.getElementById('btn-deal');
    if(d.deal_mode){{
      dp.style.display='block';
      dbtn.textContent='Stop Dealing';dbtn.className='btn-red';
      var dm=d.deal_mode;
      document.getElementById('deal-game').textContent=dm.game||'—';
      // Show/hide game input vs dealing status
      var gi=document.getElementById('deal-game-input');
      var ds=document.getElementById('deal-status');
      if(dm.phase=='game_select'){{
        gi.style.display='';ds.style.display='none';
        // Populate dealer dropdown if empty
        var sel=document.getElementById('deal-dealer');
        if(sel && sel.options.length==0){{
          players.forEach(function(n){{
            var opt=document.createElement('option');opt.value=n;opt.textContent=n;
            sel.appendChild(opt);
          }});
          setDealer();  // set initial deal order
        }}
      }} else {{
        gi.style.display='none';ds.style.display='';
        document.getElementById('deal-active-player').textContent=dm.active_player||'—';
        document.getElementById('deal-round').textContent=(dm.round_idx||0)+1;
        document.getElementById('deal-total').textContent=dm.total_rounds||'?';
      }}
      if(dm.phase=='complete'){{
        ds.innerHTML='<p style="color:#4caf50;font-size:1.1em">All rounds dealt!</p>';
      }}
      var ch='';
      (dm.cards||[]).forEach(function(c){{
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

}}

setInterval(update, 2000);
update();
</script></body></html>""")

    def _calibrate_page(self, s):
        players_js = json.dumps(PLAYER_NAMES)
        current_focus = getattr(s.capture, "focus", None)
        focus_init_js = json.dumps(current_focus)
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
<div style="margin:8px 0;display:flex;align-items:center;gap:10px">
  <label>Brio focus <span id="focus-val">auto</span></label>
  <input id="focus" type="range" min="0" max="255" value="0"
         style="flex:1;max-width:400px" oninput="onFocus(this.value)">
  <button onclick="onFocus(null)" style="padding:4px 10px">Auto</button>
  <button onclick="reloadImage()" style="padding:4px 10px">Refresh Image</button>
</div>
<canvas id="canvas"></canvas>
<button onclick="location.href='/'">Back</button>
<script>
var c=document.getElementById('canvas'),ctx=c.getContext('2d');
var players={players_js};
var _initFocus={focus_init_js};
(function(){{
  var slider=document.getElementById('focus');
  var label=document.getElementById('focus-val');
  if(_initFocus!==null&&_initFocus!==undefined){{
    slider.value=_initFocus;label.textContent=_initFocus;
  }} else {{
    label.textContent='auto';
  }}
}})();
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
img.onerror=function(){{
  document.getElementById('status').textContent=
    'Image load FAILED — /snapshot returned an error. Check host log.';
}};
var _focusTimer=null;
function onFocus(v){{
  var label=document.getElementById('focus-val');
  label.textContent=(v===null||v==='')?'auto':v;
  if(_focusTimer) clearTimeout(_focusTimer);
  _focusTimer=setTimeout(function(){{
    fetch('/api/brio/focus',{{
      method:'POST',headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{value:(v===null||v==='')?null:parseInt(v,10)}})
    }}).then(function(){{
      // Give the camera ~300 ms to settle on the new focus, then
      // refetch the snapshot so the user sees the effect.
      setTimeout(reloadImage,300);
    }});
  }},150);
}}
function reloadImage(){{
  var src='/snapshot?'+Date.now();
  var tmp=new Image();
  tmp.onload=function(){{
    img=tmp;
    // Canvas stays same size; just re-draw everything.
    draw();
  }};
  tmp.src=src;
}}
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

    def _serve_logview(self, s):
        self._r(200, "text/html", """<!DOCTYPE html>
<html><head><title>Log</title>
<style>
body{font-family:monospace;background:#0d1117;color:#e0e0e0;padding:0;margin:0}
#toolbar{padding:6px 12px;background:#16213e;position:sticky;top:0}
button{padding:4px 12px;border:none;border-radius:4px;cursor:pointer;background:#333;color:#e0e0e0;font-size:0.85em}
button:hover{background:#555}
#status{font-size:0.8em;color:#888;margin-left:12px}
pre{white-space:pre-wrap;font-size:0.85em;line-height:1.4;padding:8px 12px;margin:0}
</style>
<script>
function refresh(){
  fetch('/api/log').then(function(r){return r.json()}).then(function(d){
    var pre=document.getElementById('log');
    var atBottom=pre.scrollHeight-pre.scrollTop-pre.clientHeight<50;
    pre.innerHTML=d.lines.join('\\n');
    if(atBottom) pre.scrollTop=pre.scrollHeight;
  }).catch(function(){});
  setTimeout(refresh,2000);
}
function saveLog(){
  fetch('/log').then(function(r){return r.text()}).then(function(t){
    var blob=new Blob([t],{type:'text/plain'});
    var a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='log.txt';
    a.click();
    URL.revokeObjectURL(a.href);
    var btn=document.getElementById('savebtn');
    btn.textContent='Saved!';
    setTimeout(function(){btn.textContent='Save Log'},2000);
  });
}
function clearLog(){
  fetch('/api/log/clear',{method:'POST'}).then(function(){
    var pre=document.getElementById('log');
    pre.innerHTML='';
    var btn=document.getElementById('clrbtn');
    btn.textContent='Cleared';
    setTimeout(function(){btn.textContent='Clear Log'},1500);
  });
}
refresh();
</script></head><body>
<div id="toolbar">
  <button id="savebtn" onclick="saveLog()">Save Log</button>
  <button id="clrbtn" onclick="clearLog()">Clear Log</button>
  <span id="status">Auto-refreshing every 2s</span>
</div>
<pre id="log">Loading...</pre>
</body></html>""")

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

    def _table_state(self, s):
        try:
            doc = _build_table_state(s)
            body = json.dumps(doc)
        except Exception as e:
            body = json.dumps({"error": str(e)})
            return self._r(500, "application/json", body)
        # ETag short-circuit: if client already has this version, return 304.
        etag = f'W/"v{doc.get("version", 0)}"'
        inm = self.headers.get("If-None-Match")
        if inm == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _table_page(self, s):
        self._r(200, "text/html; charset=utf-8", TABLE_HTML)

    def _console_page(self, s):
        self._r(200, "text/html", """<!DOCTYPE html>
<html><head><title>Poker Console</title>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;
  padding:10px;padding-bottom:24px;-webkit-user-select:none;user-select:none}
h1{font-size:1.25em;text-align:center;margin:2px 0 8px;color:#e0e0e0}
button{font-family:inherit;font-weight:600;border:none;border-radius:8px;cursor:pointer;
  -webkit-tap-highlight-color:transparent}
select{padding:10px;border-radius:8px;border:1px solid #444;background:#16213e;color:#e0e0e0;
  font-size:1em;width:100%;-webkit-appearance:none;appearance:none}
select:disabled{opacity:.55}
.hdr{display:flex;gap:8px;margin-bottom:10px}
.hdr button{flex:1;padding:10px;font-size:.95em}
.btn-start{background:#1b5e20;color:#fff}
.btn-setup{background:#0f3460;color:#fff}
.btn-exit{background:#b71c1c;color:#fff}
.dim{opacity:.45;pointer-events:none}
.field{margin:8px 0}
.status-row{display:flex;gap:8px;align-items:stretch;margin:10px 0}
.status-label{flex:1.2;background:#16213e;padding:12px;border-radius:8px;font-weight:600;
  color:#4fc3f7;display:flex;align-items:center}
.action-btn{flex:1;padding:12px;font-size:1em;background:#1b5e20;color:#fff}
.action-btn:disabled{background:#333;color:#666}
h2{font-size:.85em;color:#888;text-transform:uppercase;letter-spacing:1px;margin:14px 0 6px}
.zone-row{display:flex;align-items:center;padding:10px;margin:4px 0;border-radius:8px;
  background:#16213e;cursor:pointer;-webkit-tap-highlight-color:transparent}
.zone-row:active{background:#1a3a6e}
.zone-name{width:80px;font-weight:600}
.zone-card{flex:1;color:#4caf50}
.zone-empty{color:#555}
.zone-dup{color:#e53935 !important}
.zone-arrow{color:#555;font-size:1.2em}
.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:100;
  overflow-y:auto;padding:16px}
.modal-inner{background:#16213e;border-radius:12px;padding:14px;max-width:400px;margin:10px auto}
.modal-inner h2{color:#4fc3f7;font-size:1.1em;letter-spacing:0;text-transform:none;margin:0 0 10px}
.modal-inner label{display:block;font-size:.85em;color:#aaa;margin-bottom:4px}
.modal-inner input[type=range]{width:100%;accent-color:#4fc3f7}
.modal-inner input[type=number]{width:80px;padding:6px;background:#0d1b2a;color:#fff;
  border:1px solid #333;border-radius:4px}
.modal-btns{display:flex;gap:8px;margin-top:14px}
.modal-btns button{flex:1;padding:10px}
.modal-save{background:#1b5e20;color:#fff}
.modal-cancel{background:#333;color:#ccc}
.player-row{display:flex;align-items:center;padding:6px 0;border-bottom:1px solid #222}
.player-row input{width:22px;height:22px;accent-color:#4fc3f7;margin-right:10px}
.player-row label{flex:1;font-size:1em}
#correct-img{width:40%;border-radius:8px;margin:6px auto;display:block;border:1px solid #333;background:#000}
.picker-row{display:flex;gap:8px;margin:8px 0;align-items:center}
.picker-row label{width:50px;font-size:.9em;color:#888}
.picker-row select{flex:1}
.hint{font-size:.75em;color:#666;margin-top:3px}
</style></head><body>

<h1>Poker Console</h1>

<div class="hdr">
  <button id="start-btn" class="btn-start" onclick="onStart()">Start</button>
  <button id="exit-btn" class="btn-exit" onclick="onExit()" style="display:none">Exit Poker</button>
</div>

<div id="main" class="dim">
  <div class="field">
    <select id="game-select" onchange="onPickGame()" disabled>
      <option value="">-- choose game --</option>
    </select>
  </div>

  <div class="status-row">
    <div class="status-label" id="state-text">Night not started</div>
    <button class="action-btn" id="action-btn" onclick="onAction()" disabled>—</button>
  </div>

  <h2>Up cards seen</h2>
  <div id="zone-cards"></div>
</div>

<!-- Setup / Adjust modal -->
<div id="setup-modal" class="modal">
  <div class="modal-inner">
    <h2 id="setup-title">Start night</h2>
    <div class="field">
      <label>Dealer</label>
      <select id="setup-dealer"></select>
    </div>
    <div class="field">
      <label>Active players</label>
      <div id="setup-players"></div>
    </div>
    <div class="field">
      <label>YOLO min confidence: <span id="yolo-val">50%</span></label>
      <input id="setup-yolo" type="range" min="0" max="100" value="50"
             oninput="document.getElementById('yolo-val').textContent = this.value + '%'">
      <div class="hint">Below this, falls back to Claude AI.</div>
    </div>
    <div class="field">
      <label>Pi presence brightness ceiling</label>
      <input id="setup-presence" type="number" min="0" max="255" step="1" value="140">
      <div class="hint">Slot is "present" when its brightness falls below this.</div>
    </div>
    <div class="field">
      <label>Brio scan settle (seconds)</label>
      <input id="setup-settle" type="number" min="0" max="10" step="0.1" value="0.7">
      <div class="hint">Delay after the dealer zone trips before the whole-table scan fires.</div>
    </div>
    <div class="modal-btns">
      <button class="modal-save" onclick="saveSetup()">Save</button>
      <button class="modal-cancel" onclick="closeModal('setup-modal')">Cancel</button>
    </div>
  </div>
</div>

<!-- Correction modal -->
<div id="correct-modal" class="modal" onclick="if(event.target===this)closeModal('correct-modal')">
  <div class="modal-inner">
    <h2 id="correct-title">—</h2>
    <img id="correct-img" src="">
    <div class="picker-row">
      <label>Rank</label>
      <select id="correct-rank">
        <option value="">—</option>
        <option value="A">Ace</option>
        <option value="2">2</option><option value="3">3</option>
        <option value="4">4</option><option value="5">5</option>
        <option value="6">6</option><option value="7">7</option>
        <option value="8">8</option><option value="9">9</option>
        <option value="10">10</option>
        <option value="J">Jack</option><option value="Q">Queen</option>
        <option value="K">King</option>
      </select>
    </div>
    <div class="picker-row">
      <label>Suit</label>
      <select id="correct-suit">
        <option value="">—</option>
        <option value="clubs">Clubs</option>
        <option value="diamonds">Diamonds</option>
        <option value="hearts">Hearts</option>
        <option value="spades">Spades</option>
      </select>
    </div>
    <div class="modal-btns">
      <button class="modal-save" onclick="saveCorrection()">Save</button>
      <button class="modal-cancel" onclick="closeModal('correct-modal')">Cancel</button>
    </div>
  </div>
</div>

<script>
var ST=null;
var correctPlayer=null;
var gameOptionsBuilt=false;

function api(path,data){
  return fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(data||{})}).then(function(r){return r.json()});
}

function refresh(){
  api('/api/console/state').then(function(d){ST=d;render();}).catch(function(){});
}

function buildGameOptions(){
  if(gameOptionsBuilt||!ST||!ST.game_groups) return;
  var gs=document.getElementById('game-select');
  // Keep the placeholder, if any, that the HTML already rendered.
  ST.game_groups.forEach(function(g){
    if(g.variants && g.variants.length){
      var og=document.createElement('optgroup');og.label=g.name;
      [g.name].concat(g.variants).forEach(function(v){
        var o=document.createElement('option');o.value=v;o.textContent=v;og.appendChild(o);
      });
      gs.appendChild(og);
    } else {
      var o=document.createElement('option');o.value=g.name;o.textContent=g.name;gs.appendChild(o);
    }
  });
  gameOptionsBuilt=true;
}

var lastZoneKey='';
function buildZoneRows(){
  if(!ST||!ST.active_players) return;
  // Rebuild when the active-player list changes between hands.
  var key=(ST.active_players||[]).join(',');
  if(key===lastZoneKey) return;
  lastZoneKey=key;
  var zc=document.getElementById('zone-cards');
  zc.innerHTML='';
  ST.active_players.forEach(function(n){
    var div=document.createElement('div');
    div.className='zone-row';div.id='zr-'+n;
    div.onclick=(function(player){return function(){openCorrect(player);};})(n);
    var name=document.createElement('span');name.className='zone-name';name.textContent=n;
    var card=document.createElement('span');card.className='zone-card zone-empty';card.id='zc-'+n;card.textContent='—';
    var arr=document.createElement('span');arr.className='zone-arrow';arr.textContent='›';
    div.appendChild(name);div.appendChild(card);div.appendChild(arr);
    zc.appendChild(div);
  });
}

function render(){
  if(!ST) return;
  buildGameOptions();
  buildZoneRows();

  var gs=document.getElementById('game-select');
  gs.value=ST.current_game||'';
  gs.disabled=!ST.night_active||ST.game_in_progress;

  var sbtn=document.getElementById('start-btn');
  sbtn.textContent=ST.night_active?'Setup':'Start';
  sbtn.className=ST.night_active?'btn-setup':'btn-start';

  var ebtn=document.getElementById('exit-btn');
  ebtn.style.display=ST.night_active?'block':'none';
  ebtn.textContent=ST.game_in_progress?'Exit Game':'Exit Poker';

  var mc=document.getElementById('main');
  if(ST.night_active) mc.classList.remove('dim'); else mc.classList.add('dim');

  document.getElementById('state-text').textContent=ST.phase_label||'—';
  var abtn=document.getElementById('action-btn');
  if(ST.action_label){
    abtn.style.visibility='visible';
    abtn.textContent=ST.action_label;
    abtn.disabled=!ST.action_enabled;
  } else {
    abtn.style.visibility='hidden';
  }

  (ST.active_players||[]).forEach(function(n){
    var zi=(ST.zone_cards||{})[n]||{};
    var cs=document.getElementById('zc-'+n);
    if(!cs) return;
    var txt=zi.card||'';
    cs.textContent=txt?(zi.duplicate?txt+' DUP!':txt):'—';
    cs.className='zone-card'+(txt?'':' zone-empty')+(zi.duplicate?' zone-dup':'');
  });
}

function populateSetupModal(){
  if(!ST) return;
  var ds=document.getElementById('setup-dealer');
  ds.innerHTML='';
  (ST.all_players||[]).forEach(function(n){
    var o=document.createElement('option');o.value=n;o.textContent=n;
    if(ST.dealer===n) o.selected=true;
    ds.appendChild(o);
  });
  var pl=document.getElementById('setup-players');
  pl.innerHTML='';
  (ST.all_players||[]).forEach(function(n){
    var active=(ST.active_players||[]).indexOf(n)!==-1;
    var row=document.createElement('div');row.className='player-row';
    var cb=document.createElement('input');cb.type='checkbox';cb.id='sp-'+n;cb.checked=active;
    var lbl=document.createElement('label');lbl.htmlFor='sp-'+n;lbl.textContent=n;
    row.appendChild(cb);row.appendChild(lbl);
    pl.appendChild(row);
  });
  var yolo=document.getElementById('setup-yolo');
  yolo.value=Math.round((ST.yolo_min_conf||0.5)*100);
  document.getElementById('yolo-val').textContent=yolo.value+'%';
  if(ST.pi_presence_threshold!=null){
    document.getElementById('setup-presence').value=Math.round(ST.pi_presence_threshold);
  }
  if(ST.brio_settle_s!=null){
    document.getElementById('setup-settle').value=Number(ST.brio_settle_s).toFixed(1);
  }
}

function onStart(){
  populateSetupModal();
  document.getElementById('setup-title').textContent=
    (ST&&ST.night_active)?'Adjust settings':'Start night';
  document.getElementById('setup-modal').style.display='block';
}

function saveSetup(){
  var dealer=document.getElementById('setup-dealer').value;
  var players=[];
  (ST.all_players||[]).forEach(function(n){
    if(document.getElementById('sp-'+n).checked) players.push(n);
  });
  var yolo=parseInt(document.getElementById('setup-yolo').value,10)/100.0;
  var presence=parseFloat(document.getElementById('setup-presence').value);
  var settle=parseFloat(document.getElementById('setup-settle').value);
  var body={dealer:dealer,players:players,yolo_min_conf:yolo,presence_threshold:presence,brio_settle_s:settle};
  var path=(ST&&ST.night_active)?'/api/console/settings':'/api/console/start_night';
  api(path,body).then(function(){closeModal('setup-modal');refresh();});
}

function onExit(){
  if(!ST) return;
  if(ST.game_in_progress){
    if(!confirm('End current hand?')) return;
    api('/api/console/end').then(refresh);
  } else {
    if(!confirm('Exit poker night?')) return;
    api('/api/console/exit_poker').then(function(){
      document.body.innerHTML='<h1 style="text-align:center;margin-top:40vh">Goodnight!</h1>';
    });
  }
}

function onPickGame(){
  var g=document.getElementById('game-select').value;
  if(!g) return;
  api('/api/console/deal',{game:g}).then(refresh);
}

function onAction(){
  if(!ST||!ST.action_endpoint) return;
  var ep=ST.action_endpoint;
  // Confirm Cards with nothing scanned yet → trigger a force_scan first,
  // then confirm once recognition has settled.
  if(ep==='/api/console/confirm' && ST.scan_phase==='watching'){
    api('/api/console/force_scan').then(function(){
      setTimeout(function(){ api('/api/console/confirm').then(refresh); },2500);
    });
  } else {
    api(ep).then(refresh);
  }
}

var RANK_MAP={'Ace':'A','King':'K','Queen':'Q','Jack':'J',
  '2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'10'};
var SUIT_MAP={'Clubs':'clubs','Diamonds':'diamonds','Hearts':'hearts','Spades':'spades'};

function parseCard(card){
  if(!card) return {rank:'',suit:''};
  var m=card.match(/^(.+) of (.+)$/);
  if(!m) return {rank:'',suit:''};
  return {rank:RANK_MAP[m[1]]||'',suit:SUIT_MAP[m[2]]||''};
}

function openCorrect(player){
  if(!ST) return;
  correctPlayer=player;
  var zi=(ST.zone_cards||{})[player]||{};
  document.getElementById('correct-title').textContent=player+' — tap to correct';
  document.getElementById('correct-img').src='/zone_snap/'+player+'?t='+Date.now();
  var parsed=parseCard(zi.card);
  document.getElementById('correct-rank').value=parsed.rank;
  document.getElementById('correct-suit').value=parsed.suit;
  document.getElementById('correct-modal').style.display='block';
}

function saveCorrection(){
  var rank=document.getElementById('correct-rank').value;
  var suit=document.getElementById('correct-suit').value;
  if(!rank||!suit){alert('Pick both rank and suit');return;}
  api('/api/console/correct',{corrections:[{player:correctPlayer,rank:rank,suit:suit}]})
    .then(function(){closeModal('correct-modal');refresh();});
}

function closeModal(id){document.getElementById(id).style.display='none';}

setInterval(refresh,2000);
refresh();
</script></body></html>""")

    def _training_file(self, name):
        p = TRAINING_DIR / name
        if not p.exists(): return self._r(404,"text/plain","Not found")
        self._r(200, "image/jpeg" if p.suffix==".jpg" else "text/plain",
                p.read_bytes() if p.suffix==".jpg" else p.read_text())

    def _apply_settings(self, s, data):
        """Apply the Setup modal payload: dealer, players, YOLO min-conf,
        Pi presence threshold. Every field is optional."""
        import urllib.request
        ge = s.game_engine
        dealer_name = (data.get("dealer") or "").strip()
        if dealer_name:
            for i, p2 in enumerate(ge.players):
                if p2.name.lower() == dealer_name.lower():
                    ge.dealer_index = i
                    ge._update_dealer()
                    log.log(f"[CONSOLE] Dealer set to {p2.name}")
                    break
        players = data.get("players")
        if isinstance(players, list):
            valid = [n for n in players if n in PLAYER_NAMES]
            if valid:
                s.console_active_players = valid
                log.log(f"[CONSOLE] Active players: {', '.join(valid)}")
        yolo = data.get("yolo_min_conf")
        if yolo is not None:
            try:
                s.monitor.yolo_min_conf = max(0.0, min(1.0, float(yolo)))
                log.log(f"[CONSOLE] YOLO min conf → {s.monitor.yolo_min_conf:.2f}")
            except (TypeError, ValueError):
                pass
        presence = data.get("presence_threshold")
        if presence is not None:
            try:
                pval = float(presence)
                s.pi_presence_threshold = pval
                url = f"{s.pi_base_url.rstrip('/')}/presence_threshold"
                body = json.dumps({"value": pval}).encode()
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=3).read()
                log.log(f"[CONSOLE] Pi presence_threshold → {pval}")
            except Exception as e:
                log.log(f"[CONSOLE] presence_threshold push failed: {e}")
        settle = data.get("brio_settle_s")
        if settle is not None:
            try:
                s.brio_settle_s = max(0.0, min(10.0, float(settle)))
                log.log(f"[CONSOLE] Brio settle → {s.brio_settle_s:.2f}s")
            except (TypeError, ValueError):
                pass
        # Persist host-managed tunables so restarts keep the user's choices.
        _save_host_config({
            "brio_settle_s": s.brio_settle_s,
            "pi_presence_threshold": s.pi_presence_threshold,
            "yolo_min_conf": s.monitor.yolo_min_conf,
        })

    def _proxy_slot_image(self, s, slot_num: int):
        """Proxy the Pi's /slots/<n>/image through Neo so the browser sees
        a same-origin URL. Avoids cross-origin/HTTPS-upgrade issues that
        leave the verify modal showing a broken-image placeholder."""
        import urllib.request
        url = f"{s.pi_base_url.rstrip('/')}/slots/{slot_num}/image"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = resp.read()
                ct = resp.headers.get("Content-Type", "image/jpeg")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            log.log(f"[TABLE] slot_image proxy failed for slot {slot_num}: "
                    f"{type(e).__name__}: {e}")
            self._r(502, "text/plain", "pi image unavailable")

    def _card_asset(self, name):
        """Serve a pretty card image (SVG or PNG) from host/static/cards/.
        Guards against path traversal."""
        root = (Path(__file__).parent / "static" / "cards").resolve()
        p = (root / name).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            return self._r(404, "text/plain", "Not found")
        ext = p.suffix.lower()
        if ext not in (".svg", ".png") or not p.exists():
            return self._r(404, "text/plain", "Not found")
        mime = "image/svg+xml" if ext == ".svg" else "image/png"
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _state
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help=f"avfoundation camera index; default is auto-detected "
                             f"by name (looks for '{DEFAULT_CAMERA_NAME}')")
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME,
                        help="Substring of the avfoundation device name to prefer "
                             "when auto-selecting the camera")
    parser.add_argument("--cv-camera-index", type=int, default=None,
                        help="Force a specific OpenCV VideoCapture index, skipping "
                             "name-based lookup. Use this when multiple 4K cameras "
                             "are connected and the auto-picker opens the wrong one.")
    parser.add_argument("--brio-focus", type=int, default=None,
                        help="Manual focus value for the Brio (0..255, lower = "
                             "farther). Omitting the flag leaves autofocus on.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--voice", type=str, default=None,
                        help="Base voice name for `say`. The actual voice used "
                             "is the highest-quality installed variant "
                             "(Premium > Enhanced > base). Overrides SPEECH_VOICE env.")
    args = parser.parse_args()

    if args.voice:
        speech.voice = _resolve_best_voice(args.voice)
        log.log(f"Speech voice overridden to: {speech.voice}")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    camera_index = args.camera
    if camera_index is None:
        camera_index = FrameCapture.find_index_by_name(args.camera_name)
        if camera_index is None:
            camera_index = DEFAULT_CAMERA_INDEX
            log.log(f"Camera: '{args.camera_name}' not found in avfoundation devices, "
                    f"falling back to index {camera_index}")

    # OpenCV VideoCapture index can differ from AVFoundations enumeration
    # when multiple 4K cameras are attached. Persist whatever value the user
    # passes via --cv-camera-index so the next run picks up the Brio without
    # having to re-specify it.
    _persisted_cfg = _load_host_config()
    cv_idx = args.cv_camera_index
    if cv_idx is not None:
        _save_host_config({"cv_camera_index": cv_idx})
        log.log(f"[CAPTURE] Saved cv_camera_index={cv_idx} to host config")
    elif "cv_camera_index" in _persisted_cfg:
        cv_idx = _persisted_cfg["cv_camera_index"]
        log.log(f"[CAPTURE] Loaded cv_camera_index={cv_idx} from host config")

    # Brio manual focus override — autofocus hunts on the low-contrast
    # felt background, so pin a focus position once and keep it.
    brio_focus = args.brio_focus
    if brio_focus is not None:
        _save_host_config({"brio_focus": brio_focus})
        log.log(f"[CAPTURE] Saved brio_focus={brio_focus} to host config")
    elif "brio_focus" in _persisted_cfg:
        brio_focus = _persisted_cfg["brio_focus"]
        log.log(f"[CAPTURE] Loaded brio_focus={brio_focus} from host config")

    capture = FrameCapture(camera_index, args.resolution,
                           camera_name_hint=args.camera_name,
                           cv_index_override=cv_idx,
                           focus=brio_focus)
    log.log(f"Camera {camera_index}, resolution {capture.resolution}")

    # Wait for the persistent ffmpeg stream to warm up enough to produce
    # a frame. AVFoundation can take several seconds to open the Brio.
    print("  Waiting for first frame from camera stream...")
    frame = None
    deadline = time.time() + 15.0
    while time.time() < deadline:
        frame = capture.capture()
        if frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        tail = getattr(capture, "_stderr_tail", b"").decode(errors="replace").strip()
        hint = f"\n  ffmpeg stderr tail: {tail}" if tail else ""
        sys.exit(
            f"  ERROR: No frames from camera after 15s. "
            f"Is another app holding the Brio?{hint}"
        )
    print(f"  OK: {frame.shape[1]}x{frame.shape[0]}")

    cal = Calibration()
    cal.load()

    monitor = ZoneMonitor(threshold=args.threshold)
    _state = AppState(capture, cal, monitor)
    _state.latest_frame = frame
    # Apply any persisted YOLO min-confidence now that the monitor exists.
    _persisted = _load_host_config()
    if "yolo_min_conf" in _persisted:
        try:
            monitor.yolo_min_conf = max(0.0, min(1.0, float(_persisted["yolo_min_conf"])))
        except (TypeError, ValueError):
            pass

    # Start server. ThreadingHTTPServer gives each client connection its own
    # thread so a browser's keep-alive polling (e.g. /table/state every 500ms)
    # can't starve other clients like /console or /logview.
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
    server.daemon_threads = True
    Thread(target=server.serve_forever, daemon=True).start()

    # Auto-start Pi slot poller so /table populates Rodney's hand without a
    # manual kick. The loop handles Pi-unreachable with a retry delay, so
    # starting it here is safe even if the Pi is off.
    _pi_poll_start(_state)
    log.log(f"Pi poller started against {_state.pi_base_url}")
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

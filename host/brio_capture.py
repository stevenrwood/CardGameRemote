"""
Brio (overhead) camera frame capture.

``FrameCapture`` opens a persistent AVFoundation VideoCapture against
the Logitech Brio and streams decoded BGR frames into ``_latest_frame``
for the main loop's ``capture()`` poll. Also handles:

- PyObjC-backed camera-name lookup so the Brio is selected by name
  rather than by OpenCV's unstable enumeration order (which would
  otherwise pick up whichever 4K webcam is plugged in today).
- MJPG fourcc negotiation required for the Brio to deliver 4K over
  USB without saturating the bus with uncompressed frames.
- Manual focus control (Logitech Brio usable range 0..255, lower
  = farther).
"""

import re
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock, Thread

import cv2

from log_buffer import log


CAPTURE_FILE = Path("/tmp/card_scanner_frame.jpg")


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

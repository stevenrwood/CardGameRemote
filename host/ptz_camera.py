"""PTZ control for the EMEET PIXY (and any other UVC-PTZ webcam) via
the `uvc-util` CLI [1]. The Pi side uses a different scanner camera —
this module is host-only and only meaningful on macOS where the camera
is plugged into Neo and Teams shares it as a video source.

The standard `uvc-util` install locations live in PTZ_BINARY_PATHS;
the first one that exists wins. If none exist (e.g. fresh Neo, no
build done yet), `available()` returns False and every action becomes
a no-op that surfaces a clear error to the UI rather than crashing.

[1] https://github.com/jtfrey/uvc-util — built from source, not in
Homebrew. README has macOS build instructions.

UVC angular units are arcseconds: 3600 = 1°. We keep the public API
in degrees and convert at the boundary so callers (and the /table UI
text input) don't have to deal with arcseconds.
"""

import os
import re
import shutil
import subprocess
from threading import Lock

from log_buffer import log


PTZ_BINARY_PATHS = [
    "/opt/homebrew/bin/uvc-util",
    "/usr/local/bin/uvc-util",
    "/tmp/uvc-util/src/uvc-util",
    os.path.expanduser("~/bin/uvc-util"),
    os.path.expanduser("~/uvc-util/uvc-util"),
    os.path.expanduser("~/src/uvc-util/uvc-util"),
]

CAMERA_NAME_HINTS = ("EMEET PIXY", "PIXY")

ARCSEC_PER_DEG = 3600

MIN_STEP_DEG = 1
MAX_STEP_DEG = 20
DEFAULT_STEP_DEG = 8

ZOOM_STEP_FRACTION = 0.10


_lock = Lock()
_binary: str | None = None
_device_index: int | None = None
_resolved = False
_pan_range: tuple[int, int] | None = None
_tilt_range: tuple[int, int] | None = None
_zoom_range: tuple[int, int] | None = None
_pan_home: int = 0
_tilt_home: int = 0
_zoom_home: int = 0


def _find_binary() -> str | None:
    on_path = shutil.which("uvc-util")
    if on_path:
        return on_path
    for path in PTZ_BINARY_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _run(args: list[str], timeout: float = 2.0) -> tuple[int, str, str]:
    if _binary is None:
        return 1, "", "uvc-util not installed"
    try:
        proc = subprocess.run(
            [_binary] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "uvc-util timed out"
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def _enumerate_devices() -> list[tuple[int, str]]:
    rc, out, _ = _run(["-d"])
    if rc != 0:
        return []
    devices: list[tuple[int, str]] = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(.*)", line.strip())
        if m:
            devices.append((int(m.group(1)), m.group(2).strip()))
    return devices


def _pick_device(devices: list[tuple[int, str]]) -> int | None:
    for idx, name in devices:
        for hint in CAMERA_NAME_HINTS:
            if hint.lower() in name.lower():
                return idx
    return None


def _parse_range(text: str) -> tuple[int, int] | None:
    m = re.search(r"(-?\d+)\s*(?:to|\.\.)\s*(-?\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"-?\d+", text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None


def _parse_pan_tilt(text: str) -> tuple[int, int] | None:
    nums = re.findall(r"-?\d+", text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None


def _read_range(ctrl: str) -> tuple[int, int] | None:
    rc, out, _ = _run(["-I", str(_device_index), "-r", ctrl])
    if rc != 0:
        return None
    return _parse_range(out)


def _resolve():
    """Locate the binary, the EMEET PIXY device, and read each
    control's valid range. Cached after first call. Re-run by
    calling resolve(force=True)."""
    global _binary, _device_index, _resolved
    global _pan_range, _tilt_range, _zoom_range
    global _pan_home, _tilt_home, _zoom_home

    _binary = _find_binary()
    if _binary is None:
        log.log("[PTZ] uvc-util not found on PATH or known locations")
        _resolved = True
        return

    devices = _enumerate_devices()
    _device_index = _pick_device(devices)
    if _device_index is None:
        names = ", ".join(n for _, n in devices) or "(none)"
        log.log(f"[PTZ] EMEET PIXY not found among UVC devices: {names}")
        _resolved = True
        return

    log.log(f"[PTZ] using device {_device_index} (uvc-util at {_binary})")

    rc, out, _ = _run(["-I", str(_device_index), "-r", "pan-tilt-abs"])
    if rc == 0:
        nums = re.findall(r"-?\d+", out)
        if len(nums) >= 4:
            _pan_range = (int(nums[0]), int(nums[1]))
            _tilt_range = (int(nums[2]), int(nums[3]))

    _zoom_range = _read_range("zoom-abs")

    if _pan_range:
        _pan_home = (_pan_range[0] + _pan_range[1]) // 2
    if _tilt_range:
        _tilt_home = (_tilt_range[0] + _tilt_range[1]) // 2
    if _zoom_range:
        _zoom_home = _zoom_range[0]

    log.log(
        f"[PTZ] ranges: pan={_pan_range} tilt={_tilt_range} "
        f"zoom={_zoom_range}"
    )
    _resolved = True


def _ensure_resolved():
    if not _resolved:
        _resolve()


def available() -> bool:
    _ensure_resolved()
    return _binary is not None and _device_index is not None


def status() -> dict:
    _ensure_resolved()
    return {
        "available": available(),
        "binary": _binary,
        "device_index": _device_index,
        "pan_range": list(_pan_range) if _pan_range else None,
        "tilt_range": list(_tilt_range) if _tilt_range else None,
        "zoom_range": list(_zoom_range) if _zoom_range else None,
        "min_step_deg": MIN_STEP_DEG,
        "max_step_deg": MAX_STEP_DEG,
        "default_step_deg": DEFAULT_STEP_DEG,
    }


def _read_pan_tilt() -> tuple[int, int] | None:
    rc, out, _ = _run(["-I", str(_device_index), "-g", "pan-tilt-abs"])
    if rc != 0:
        return None
    return _parse_pan_tilt(out)


def _read_zoom() -> int | None:
    rc, out, _ = _run(["-I", str(_device_index), "-g", "zoom-abs"])
    if rc != 0:
        return None
    nums = re.findall(r"-?\d+", out)
    if not nums:
        return None
    return int(nums[0])


def _clamp(val: int, rng: tuple[int, int] | None) -> int:
    if rng is None:
        return val
    lo, hi = rng
    return max(lo, min(hi, val))


def _set_pan_tilt(pan: int, tilt: int) -> tuple[bool, str]:
    pan = _clamp(pan, _pan_range)
    tilt = _clamp(tilt, _tilt_range)
    rc, _, err = _run([
        "-I", str(_device_index),
        "-s", f"pan-tilt-abs={pan},{tilt}",
    ])
    if rc != 0:
        return False, err.strip() or "uvc-util failed"
    return True, ""


def _set_zoom(zoom: int) -> tuple[bool, str]:
    zoom = _clamp(zoom, _zoom_range)
    rc, _, err = _run([
        "-I", str(_device_index),
        "-s", f"zoom-abs={zoom}",
    ])
    if rc != 0:
        return False, err.strip() or "uvc-util failed"
    return True, ""


def _clamp_step_deg(step_deg: float) -> float:
    try:
        s = float(step_deg)
    except (TypeError, ValueError):
        s = DEFAULT_STEP_DEG
    if s < MIN_STEP_DEG:
        s = MIN_STEP_DEG
    if s > MAX_STEP_DEG:
        s = MAX_STEP_DEG
    return s


def step(action: str, step_deg: float = DEFAULT_STEP_DEG) -> tuple[bool, str]:
    """Execute one PTZ step. `action` is one of pan_left, pan_right,
    tilt_up, tilt_down, zoom_in, zoom_out, home. `step_deg` is degrees
    of pan/tilt per click; ignored for zoom (which uses a fraction of
    its range) and home."""
    _ensure_resolved()
    if not available():
        return False, "PTZ unavailable (uvc-util or camera missing)"

    with _lock:
        if action == "home":
            ok1, e1 = _set_pan_tilt(_pan_home, _tilt_home)
            ok2, e2 = _set_zoom(_zoom_home)
            if not (ok1 and ok2):
                return False, (e1 or e2 or "home failed")
            return True, ""

        if action in ("pan_left", "pan_right", "tilt_up", "tilt_down"):
            cur = _read_pan_tilt()
            if cur is None:
                return False, "could not read current pan/tilt"
            pan, tilt = cur
            delta = int(_clamp_step_deg(step_deg) * ARCSEC_PER_DEG)
            if action == "pan_left":
                pan -= delta
            elif action == "pan_right":
                pan += delta
            elif action == "tilt_up":
                tilt += delta
            elif action == "tilt_down":
                tilt -= delta
            return _set_pan_tilt(pan, tilt)

        if action in ("zoom_in", "zoom_out"):
            if _zoom_range is None:
                return False, "zoom range unknown"
            cur = _read_zoom()
            if cur is None:
                return False, "could not read current zoom"
            lo, hi = _zoom_range
            zstep = max(1, int((hi - lo) * ZOOM_STEP_FRACTION))
            new = cur + zstep if action == "zoom_in" else cur - zstep
            return _set_zoom(new)

        return False, f"unknown action: {action}"

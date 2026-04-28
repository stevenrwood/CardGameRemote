"""Pure cv2/numpy helpers for the overhead camera frame.

No AppState dependency — these are the bits the bg_loop uses to
crop and annotate raw frames before encoding them. Kept separate
from overhead_test.py so they can be reused (and unit-tested)
without dragging the whole game-flow module in.
"""

import cv2
import numpy as np


def crop_circle(frame, cal):
    """Mask out everything outside the felt circle. Returns the
    original frame unchanged when calibration is incomplete so the
    caller can still encode something instead of crashing on a None
    mask center."""
    if not cal.circle_center or not cal.circle_radius:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, cal.circle_center, cal.circle_radius, 255, -1)
    return cv2.bitwise_and(frame, frame, mask=mask)


def draw_overlay(frame, cal, monitor):
    """Draw the felt circle + per-zone circles + zone names + last
    recognized card on top of ``frame`` in place. Used by the
    display-JPEG path that the main page's iframe renders."""
    if cal.circle_center and cal.circle_radius:
        cv2.circle(frame, cal.circle_center, cal.circle_radius, (255, 255, 255), 2)
    for z in cal.zones:
        name, cx, cy, r = z["name"], z["cx"], z["cy"], z["r"]
        zs = monitor.zone_state.get(name, "empty")
        color = {"recognized": (0, 255, 0), "processing": (0, 255, 255)}.get(
            zs, (255, 255, 255)
        )
        cv2.circle(frame, (cx, cy), r, color, 2)
        cv2.putText(
            frame, name, (cx - 30, cy - r - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
        card = monitor.last_card.get(name, "")
        if card:
            cv2.putText(
                frame, card, (cx - 60, cy + r + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )


def to_jpeg(frame, q=85):
    """Encode ``frame`` as JPEG bytes at quality ``q``. Returns
    ``None`` on encode failure rather than raising — callers
    surface a 500 to the client without unwinding the bg_loop."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else None

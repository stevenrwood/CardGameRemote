"""Calibration data for the overhead camera.

The felt circle (its image-space center + radius), the pixel radius
of an 8" zone marker (so all 5 player zones get the same physical
size), and the per-player zone list itself. Persisted to
``calibration.json`` next to this module.
"""

import json

from log_buffer import log
from host_constants import CALIBRATION_FILE, NUM_ZONES


class Calibration:
    def __init__(self):
        self.circle_center = None
        self.circle_radius = None
        # Pixel radius of an 8-inch-diameter zone, measured once per
        # calibration via two clicks on a ruler / known marks 8"
        # apart on the felt. Every zone placed in the calibration
        # wizard inherits this radius so they are physically uniform.
        self.zone_radius_px = None
        self.zones = []

    def save(self):
        data = {
            "circle_center": (
                list(self.circle_center) if self.circle_center else None
            ),
            "circle_radius": self.circle_radius,
            "zone_radius_px": self.zone_radius_px,
            "zones": self.zones,
        }
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
        self.zone_radius_px = data.get("zone_radius_px")
        self.zones = data.get("zones", [])
        if self.zones and "cx" not in self.zones[0]:
            self.zones = []
            return False
        # Backwards compat: derive a sensible default zone_radius_px
        # from the existing zones' average radius so the new
        # calibration UI has something to fall back on if the dealer
        # only re-runs the wizard partially.
        if self.zone_radius_px is None and self.zones:
            radii = [int(z.get("r", 0) or 0) for z in self.zones]
            radii = [r for r in radii if r > 0]
            if radii:
                self.zone_radius_px = sum(radii) // len(radii)
        return True

    @property
    def ok(self):
        return (
            self.circle_center
            and self.circle_radius
            and len(self.zones) == NUM_ZONES
        )

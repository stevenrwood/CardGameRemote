"""Shared constants for the host process.

Pulled out of overhead_test.py so smaller modules (calibration,
frame_utils, app_state, etc.) don't have to round-trip through the
big module to reach a couple of names. overhead_test.py re-exports
everything here for backward compatibility with http_server.py's
``from overhead_test import …`` block.
"""

from pathlib import Path


PLAYER_NAMES = ["Steve", "Bill", "David", "Joe", "Rodney"]
NUM_ZONES = len(PLAYER_NAMES)

DEFAULT_CAMERA_INDEX = 1  # Fallback if we can't find the Brio by name
DEFAULT_CAMERA_NAME = "BRIO"  # avfoundation device name substring to prefer
DEFAULT_THRESHOLD = 30.0
DEFAULT_RESOLUTION = "auto"

# Default seconds to wait after the overhead (Brio) camera trips a motion
# event in the dealer zone before firing the whole-table scan. Runtime-
# configurable via the Setup modal → persisted in the host config file.
DEFAULT_BRIO_SETTLE_S = 0.7

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"

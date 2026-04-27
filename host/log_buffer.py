"""
Log buffer — in-memory tail + dual-stream file mirror.

Two streams:

  log.txt                 The "between games" stream. Holds
                          host startup, settings changes, calibration
                          activity, voice noise — everything that
                          happens while no game is in progress.

  YYYY-MM-DD HH:MM:SS X   A per-game file opened when a hand starts
                          and closed when that hand ends or the next
                          one begins. ``X`` is a filename-safe
                          abbreviation of the game name (FTQ, 7CS,
                          7-27, etc).

The host calls ``log.start_game(name)`` from /api/console/deal and
``log.end_game()`` from any path that ends a hand (End Hand,
Exit Poker, etc). While a game file is open every ``log.log()``
line goes to the game file ONLY — log.txt sees nothing until the
game ends. A ``most_recent_game`` symlink in the log directory
always points at the freshest game file.

A single module-level ``log = LogBuffer()`` is shared by the rest
of the host code. Logs live in ~/Library/Logs/cardgame-host/ so
they stay outside macOS Transparency Consent and Control (TCC)
protected directories — sshd / scp can read them without Full Disk
Access.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from threading import Lock


LOG_DIR = Path.home() / "Library" / "Logs" / "cardgame-host"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "log.txt"
RECENT_LINK = LOG_DIR / "most_recent_game"


# Hand-curated abbreviations for the templates we ship; anything not
# in this map falls through to a sanitized version of the full name.
_GAME_ABBREVS = {
    "5 Card Draw": "5CD",
    "5 Card Double Draw": "5CDD",
    "3 Toed Pete": "3TP",
    "7 Card Stud": "7CS",
    "7 Stud Deuces Wild": "7SDW",
    "Follow the Queen": "FTQ",
    "High Chicago": "HC",
    "Eight or Better": "8B",
    "Challenge": "CHL",
    "High, Low, High": "HLH",
    "Low, High, Low": "LHL",
    "Low, Low, High": "LLH",
    "7/27": "7-27",
    "7/27 (one up)": "7-27 1up",
    "Texas Hold'em": "TXH",
}


def _game_abbrev(name: str) -> str:
    if name in _GAME_ABBREVS:
        return _GAME_ABBREVS[name]
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (name or "")).strip("_")
    return cleaned or "game"


class LogBuffer:
    def __init__(self, maxlines=500):
        self._lines = []
        self._lock = Lock()
        # Path to the active per-game log file when a hand is in
        # progress; None otherwise. While set, log() writes to this
        # file INSTEAD of log.txt.
        self._game_path = None
        # Overwrite log file on startup so each host run starts fresh.
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
            target = self._game_path or LOG_FILE
        try:
            with open(target, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def get(self, n=50):
        with self._lock:
            return list(self._lines[-n:])

    def clear(self):
        """Wipe the in-memory buffer and the live log file."""
        with self._lock:
            self._lines = []
        try:
            LOG_FILE.write_text("")
        except Exception:
            pass

    def start_game(self, game_name: str) -> str:
        """Open a fresh per-game log file for ``game_name``. If a
        previous game file is still open, close it first. Updates the
        ``most_recent_game`` symlink to point at the new file. Returns
        the new file's name.
        """
        # Close any prior game first so its trailing line lands in
        # the right file before we switch.
        self.end_game()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        abbrev = _game_abbrev(game_name)
        path = LOG_DIR / f"{stamp} {abbrev}.txt"
        try:
            with open(path, "w") as f:
                f.write("")
        except Exception:
            pass
        with self._lock:
            self._game_path = path
        try:
            if RECENT_LINK.is_symlink() or RECENT_LINK.exists():
                RECENT_LINK.unlink()
            os.symlink(path.name, RECENT_LINK)
        except Exception:
            pass
        self.log(f"=== Game started: {game_name} ===")
        return path.name

    def end_game(self) -> None:
        """Close the current per-game log file (if any). Subsequent
        log() lines flow back into log.txt until start_game() runs
        again. Idempotent."""
        with self._lock:
            active = self._game_path
        if active is None:
            return
        # Write the close marker to the game file BEFORE clearing the
        # pointer so the marker lands in the right stream.
        self.log("=== Game ended ===")
        with self._lock:
            self._game_path = None

    # -- Legacy compatibility shims ----------------------------------
    # Existing callers still hit start_night() / end_night() to
    # bracket a poker night. The old behavior wrote a poker_*.txt
    # archive that mirrored every line; that's superseded by the
    # per-game files. We keep the names so the rest of the codebase
    # doesn't have to be edited in lockstep, but they're now no-ops
    # beyond a couple of marker lines in log.txt.

    def start_night(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log(f"=== Poker night started {stamp} ===")
        return ""

    def end_night(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Make sure no per-game file is left open if a hand was mid-
        # flight at exit.
        self.end_game()
        self.log(f"=== Poker night ended {stamp} ===")


log = LogBuffer()

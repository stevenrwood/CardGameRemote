"""
Log buffer — in-memory tail + dual-stream file mirror.

Three layers:

  ~/Library/Logs/cardgame-host/log.txt
        Pre-session "scratch" log — host startup, errors before any
        Start Poker / Start Testing button has been pressed.

  ~/Library/Logs/cardgame-host/<session>/log.txt
        Inside each session folder, the "between-games" stream for
        that session. Holds settings changes, voice noise, anything
        that happens while no game is active during this session.

  ~/Library/Logs/cardgame-host/<session>/YYYY-MM-DD HH:MM:SS X.txt
        One per-game file inside the session, opened at /api/console/
        deal and closed at hand end. ``X`` is a filename-safe
        abbreviation of the game name (FTQ, 7CS, 7-27, etc).

  ~/Library/Logs/cardgame-host/<session>/most_recent_game
        Symlink inside the session folder, points at the freshest
        game log file.

Session folder names:
  Start Poker   → "YYYY-MM-DD HH:MM:SS Poker Night"  (per-night history)
  Start Testing → "Testing"                          (rolling, wiped each start)

A single module-level ``log = LogBuffer()`` is shared by the rest
of the host code. Logs live in ~/Library/Logs/cardgame-host/ so
they stay outside macOS Transparency Consent and Control (TCC)
protected directories — sshd / scp can read them without Full Disk
Access.
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock


LOG_DIR = Path.home() / "Library" / "Logs" / "cardgame-host"
LOG_DIR.mkdir(parents=True, exist_ok=True)
# Pre-session "before any session was started" log file. Sessions
# rotate to their own log.txt inside their session folder.
LOG_FILE = LOG_DIR / "log.txt"


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
        # file INSTEAD of the session log.txt.
        self._game_path = None
        # Session folder. None before Start Poker / Start Testing —
        # logging falls through to the root log.txt.
        self._session_dir = None
        # Overwrite the root log.txt on startup so the next pre-
        # session run starts fresh.
        LOG_FILE.write_text("")

    def _session_log_path(self):
        if self._session_dir is None:
            return LOG_FILE
        return self._session_dir / "log.txt"

    def log(self, msg):
        # ms precision so we can time sub-second pipeline stages.
        now = datetime.now()
        ts = now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"
        line = f"[{ts}] {msg}"
        print(line)
        with self._lock:
            self._lines.append(line)
            self._lines = self._lines[-500:]
            target = self._game_path or self._session_log_path()
        try:
            with open(target, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def get(self, n=50):
        with self._lock:
            return list(self._lines[-n:])

    def clear(self):
        """Wipe the in-memory buffer and the active log file."""
        with self._lock:
            self._lines = []
            target = self._session_log_path()
        try:
            target.write_text("")
        except Exception:
            pass

    @property
    def session_dir(self):
        return self._session_dir

    def start_session(self, kind: str) -> str:
        """Open a new session folder and route subsequent logs into
        it. ``kind`` is "poker" or "testing":
          poker   → "YYYY-MM-DD HH:MM:SS Poker Night/"  (per-night)
          testing → "Testing/"  (wiped each start, single rolling slot)

        Closes any prior session / per-game file first. Returns the
        session folder's name."""
        # Close any in-flight game from the previous session so its
        # trailing line lands in the right file.
        self.end_game()
        if kind == "testing":
            folder = LOG_DIR / "Testing"
            # Wipe and recreate so each Start Testing run is a
            # clean slate.
            try:
                if folder.exists():
                    shutil.rmtree(folder)
            except Exception:
                pass
        else:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            folder = LOG_DIR / f"{stamp} Poker Night"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "log.txt").write_text("")
        except Exception:
            pass
        with self._lock:
            self._session_dir = folder
        self.log(f"=== Session started: {folder.name} ===")
        return folder.name

    def end_session(self) -> None:
        """Close the active session. Subsequent log() lines flow
        back into the root log.txt until the next start_session().
        Idempotent."""
        with self._lock:
            active = self._session_dir
        if active is None:
            return
        self.end_game()
        self.log(f"=== Session ended: {active.name} ===")
        with self._lock:
            self._session_dir = None

    def start_game(self, game_name: str, abbrev: str = "") -> str:
        """Open a fresh per-game log file inside the active session
        folder (or LOG_DIR if no session). If a previous game file
        is still open, close it first. Updates the
        ``most_recent_game`` symlink to point at the new file.

        ``abbrev`` is the filename-safe game abbreviation. Callers
        with a GameTemplate should pass template.log_abbrev — falls
        back to the in-module mapping (then to a sanitized form of
        game_name) if blank.

        Returns the new file's name."""
        self.end_game()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Trust an explicit caller-provided abbrev (likely from
        # GameTemplate.log_abbrev); fall back to the local map.
        abbrev = (abbrev or "").strip() or _game_abbrev(game_name)
        parent = self._session_dir or LOG_DIR
        path = parent / f"{stamp} {abbrev}.txt"
        try:
            with open(path, "w") as f:
                f.write("")
        except Exception:
            pass
        with self._lock:
            self._game_path = path
        link = parent / "most_recent_game"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(path.name, link)
        except Exception:
            pass
        self.log(f"=== Game started: {game_name} ===")
        return path.name

    def end_game(self) -> None:
        """Close the current per-game log file (if any). Subsequent
        log() lines flow back into the session log.txt until
        start_game() runs again. Idempotent."""
        with self._lock:
            active = self._game_path
        if active is None:
            return
        self.log("=== Game ended ===")
        with self._lock:
            self._game_path = None

    # -- Legacy compatibility shims ----------------------------------
    # Older callers used start_night() / end_night() to bracket the
    # session. They now just delegate to the new session methods so
    # the rest of the codebase keeps working.

    def start_night(self, kind: str = "poker") -> str:
        return self.start_session(kind)

    def end_night(self) -> None:
        self.end_session()


log = LogBuffer()

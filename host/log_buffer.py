"""
Log buffer — in-memory tail + live file mirror + per-night archive.

A single module-level ``log = LogBuffer()`` is shared by the rest of
the host code. Logs live in ~/Library/Logs/cardgame-host/ rather than
~/Downloads so they stay outside macOS Transparency Consent and
Control (TCC) protected directories, letting SSH/scp read them
without granting sshd Full Disk Access.
"""

from datetime import datetime
from pathlib import Path
from threading import Lock


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

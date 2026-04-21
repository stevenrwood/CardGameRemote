#!/usr/bin/env python3
"""
Retroactively remove wrong-label training_data pairs.

Every time YOLO or Claude finalized a card, overhead_test.py saved the
crop + label to training_data/. If the user later corrected the
recognition, the wrong-label pair was left behind (the bug this patches
forward). This script walks the archived poker logs and deletes those
stale wrong-label files so future YOLO training isn't poisoned.

Usage:
    python3 cleanup_training_data.py          # dry run — list matches
    python3 cleanup_training_data.py --apply  # actually delete

Matches a training_data file when:
  - player name matches a correction event in some archived log,
  - label (safe-encoded) matches the corrections "old label",
  - training files timestamp is before the correction timestamp
    and within a 3-hour window (a hand never runs longer than that).
"""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING_DIR = HERE / "training_data"

# New canonical location (post-fix) + the old ~/Downloads spot in case
# pre-fix archive files still live there.
LOG_DIRS = [
    Path.home() / "Library" / "Logs" / "cardgame-host",
    Path.home() / "Downloads",
]

ARCHIVE_RE = re.compile(r"^poker_(\d{8})_(\d{6})\.txt$")
CORRECTION_RE = re.compile(
    r"\[(\d\d):(\d\d):(\d\d)(?:\.\d+)?\]\s+\[CONSOLE\]\s+"
    r"Corrected\s+(\w+):\s+(.+?)\s+->\s+(.+?)\s*$"
)
FILE_RE = re.compile(r"^(\d{8}_\d{6})_([^_]+)_(.+)$")


def _label_to_safe(label: str) -> str:
    return label.replace(" ", "_").replace("/", "-")[:30]


def load_corrections():
    """Return list of (correction_datetime, player, old_label, new_label)."""
    corrections = []
    for log_dir in LOG_DIRS:
        if not log_dir.exists():
            continue
        for f in sorted(log_dir.glob("poker_*.txt")):
            m = ARCHIVE_RE.match(f.name)
            if not m:
                continue
            try:
                base_date = datetime.strptime(m.group(1), "%Y%m%d").date()
            except ValueError:
                continue
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                cm = CORRECTION_RE.search(line)
                if not cm:
                    continue
                h, mi, se = int(cm.group(1)), int(cm.group(2)), int(cm.group(3))
                ts = datetime.combine(
                    base_date, datetime.min.time()
                ).replace(hour=h, minute=mi, second=se)
                corrections.append(
                    (ts, cm.group(4), cm.group(5).strip(), cm.group(6).strip())
                )
    return corrections


def parse_training_basename(stem: str):
    """'20260420_221005_David_5_of_Hearts' -> (datetime, 'David', '5_of_Hearts')"""
    m = FILE_RE.match(stem)
    if not m:
        return None
    try:
        ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return ts, m.group(2), m.group(3)


def main():
    apply = "--apply" in sys.argv
    if not TRAINING_DIR.exists():
        print(f"No training dir at {TRAINING_DIR}")
        return
    corrections = load_corrections()
    if not corrections:
        print("No correction events found in any archived log.")
        return
    print(f"Loaded {len(corrections)} correction events from archives.")

    window = timedelta(hours=3)
    targets = []
    for jpg in TRAINING_DIR.glob("*.jpg"):
        parsed = parse_training_basename(jpg.stem)
        if parsed is None:
            continue
        f_ts, f_player, f_safe = parsed
        for c_ts, c_player, c_old, _c_new in corrections:
            if c_player != f_player:
                continue
            if _label_to_safe(c_old) != f_safe:
                continue
            if not (f_ts <= c_ts and (c_ts - f_ts) <= window):
                continue
            targets.append((jpg, c_ts, c_old))
            break

    if not targets:
        print("No wrong-label training files matched any correction.")
        return

    targets.sort(key=lambda t: t[0].name)
    print(f"\n{len(targets)} training pair(s) match a correction:")
    for jpg, c_ts, c_old in targets:
        print(f"  {jpg.name}  (corrected at {c_ts:%Y-%m-%d %H:%M:%S}, was {c_old!r})")

    if not apply:
        print("\nDry run. Re-run with --apply to delete.")
        return

    removed = 0
    for jpg, _c_ts, _c_old in targets:
        for suffix in (".jpg", ".txt"):
            p = jpg.with_suffix(suffix)
            try:
                if p.exists():
                    p.unlink()
                    removed += 1
            except OSError as e:
                print(f"  could not remove {p}: {e}")
    print(f"\nRemoved {removed} files.")


if __name__ == "__main__":
    main()

"""
Speech queue — say(text) enqueues phrases; a background thread runs
``say -v <voice> <phrase>`` per item and dedupes rapid-fire duplicates
so an overlong queue collapses to one utterance each.

A single module-level ``speech = SpeechQueue()`` is shared by the
rest of the host code.
"""

import os
import re
import subprocess
import sys
from queue import Queue, Empty
from threading import Thread


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
        # `log` isn't constructed until after SpeechQueue in the old layout,
        # and we want this module self-contained anyway, so use stderr here.
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

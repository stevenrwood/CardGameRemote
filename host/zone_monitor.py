"""
ZoneMonitor — per-zone card recognition.

Runs YOLO on each changed zone and batches a single multi-image
Claude call for zones below the YOLO confidence bar. Saves every
recognized crop to training_data/ for later model retraining, and
remembers the most recent save so a user correction can delete the
wrong-labeled pair before it poisons training.

Dependencies are injected:
  - ``get_zones`` — callable returning the current zone list (dicts
    with name/cx/cy/r). Lets the monitor stay oblivious to AppState.
  - ``stats_cb`` — callable(key: str) -> None, invoked for each
    recognition so the host can tally YOLO- vs Claude-sourced calls.
"""

import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

import cv2
import numpy as np

from log_buffer import log
from speech import speech


HOST_DIR = Path(__file__).parent
TRAINING_DIR = HOST_DIR / "training_data"
YOLO_MODEL_PATH = HOST_DIR / "models" / "card_detector.pt"
CONFIG_FILE = HOST_DIR.parent / "local" / "config.json"

CLAUDE_MODEL = "claude-sonnet-4-20250514"


class ZoneMonitor:
    def __init__(self, threshold, get_zones, stats_cb=None,
                 speech_formatter=None):
        self.threshold = threshold
        self.yolo_min_conf = 0.50  # below this, fall back to Claude API
        self._get_zones = get_zones
        self._stats_cb = stats_cb or (lambda key: None)
        # Optional callback (name, card_text) -> speech string. Lets
        # the active game class customize the per-card announcement —
        # e.g. 7/27 appending "with N or less down below" when a
        # player's visible total starts edging toward 27. Default
        # just speaks "{name}, {card}" as before.
        self._speech_formatter = (
            speech_formatter or (lambda name, card: f"{name}, {card}")
        )
        self.baselines = {}
        self.last_card = {}
        # Tracks the most recent value spoken for each zone, so a
        # repeat force_scan over a still-recognized zone (e.g. when
        # the dealer hits Confirm with the scan_phase reverted to
        # "watching" and the JS auto-fires force_scan first) doesn't
        # re-announce a card the table already heard. Cleared with
        # last_card on round confirm and on rescan_all, so genuine
        # new-round / explicit-rescan flows still announce.
        self.last_announced = {}
        # Per-zone stability tracker for the streaming watcher: how
        # many consecutive non-empty frames have been roughly
        # identical to the previous frame. Once this hits a small
        # threshold the host queues that one zone for recognition
        # without waiting for any other zone.
        self.stable_count = {}
        self.prev_crop = {}
        self.zone_state = {}
        self.pending = {}
        self.recognition_details = {}  # name -> {yolo, yolo_conf, claude, final}
        self.recognition_crops = {}   # name -> crop (numpy array) at time of recognition
        self._yolo_model = None
        self._yolo_lock = Lock()  # YOLO model is not thread-safe
        self._client = None
        self._load_yolo()

    def _load_yolo(self):
        if YOLO_MODEL_PATH.exists():
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(str(YOLO_MODEL_PATH))
                log.log(f"YOLO model loaded: {YOLO_MODEL_PATH.name}")
            except Exception as e:
                log.log(f"YOLO load failed: {e}")
        else:
            log.log("No YOLO model found — will use Claude API")

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key and CONFIG_FILE.exists():
                    key = json.loads(CONFIG_FILE.read_text()).get("anthropic_api_key")
                if key and key != "YOUR_KEY_HERE":
                    self._client = anthropic.Anthropic(api_key=key)
            except ImportError:
                pass
        return self._client

    def capture_baselines(self, frame):
        for z in self._get_zones():
            crop = self._crop(frame, z)
            if crop is not None and crop.size > 0:
                self.baselines[z["name"]] = crop.copy()
                self.zone_state[z["name"]] = "empty"
                self.last_card[z["name"]] = ""
                self.last_announced[z["name"]] = ""
                self.pending[z["name"]] = False
                self.stable_count[z["name"]] = 0
                self.prev_crop[z["name"]] = None
        log.log("Baselines captured")

    # No-op shims kept so callers that opt to "force-open" the gate
    # (force_scan / rescan_all / dealer-zone-done) don't have to be
    # edited in lockstep. Speech now fires immediately per-zone.
    def open_speech_gate(self):
        return

    def close_speech_gate(self):
        return

    def _announce_card(self, name, result):
        """Speak the recognized card immediately unless we already
        announced this same value for this zone (which would happen
        if force_scan re-runs over already-recognized zones)."""
        if self.last_announced.get(name) == result:
            return
        self.last_announced[name] = result
        speech.say(self._speech_formatter(name, result))

    def check_zones(self, frame):
        """Check all zones. YOLO runs for each changed zone, then one batched
        Claude call handles all zones where YOLO was below threshold."""
        changed = {}  # name -> crop for zones needing recognition
        for z in self._get_zones():
            name = z["name"]
            if name not in self.baselines or self.pending.get(name):
                continue
            # Skip corrected zones — dealer already fixed this card
            if self.zone_state.get(name) == "corrected":
                continue
            crop = self._crop(frame, z)
            if crop is None or crop.size == 0:
                continue
            bl = self.baselines[name]
            if crop.shape != bl.shape:
                continue
            diff = float(np.mean(cv2.absdiff(crop, bl)))

            if self.zone_state.get(name) == "recognized":
                if diff < self.threshold:
                    self.zone_state[name] = "empty"
                    self.last_card[name] = ""
                continue

            if diff > self.threshold:
                changed[name] = crop.copy()

        if changed:
            # Mark all as pending so we don't double-process
            for name in changed:
                self.pending[name] = True
            Thread(target=self._recognize_batch, args=(changed,), daemon=True).start()

    def _recognize_batch(self, zone_crops, force_claude_names=None):
        """Run YOLO on all zones, then batch Claude call for low-confidence ones.

        ``force_claude_names`` — optional iterable of zone names that must
        route through Claude even when YOLO passes the confidence bar.
        Used for zones that already came back empty earlier in this round;
        a subsequent YOLO hit on those zones is almost always a
        hallucination (phantom 2 of Spades on an empty zone), so Claude's
        verdict wins before we commit a card that never existed."""
        force_claude = set(force_claude_names or ())
        need_claude = {}  # name -> (crop, yolo_result, yolo_conf)

        # Phase 1: YOLO all zones (sequential, under lock)
        for name, crop in zone_crops.items():
            details = {"yolo": "", "yolo_conf": 0, "claude": "", "final": ""}
            t0 = time.time()
            try:
                if self._yolo_model is not None:
                    log.log(f"[{name}] YOLO inference started")
                    t_yolo = time.time()
                    result, conf = self._recognize_yolo(crop)
                    yolo_ms = (time.time() - t_yolo) * 1000
                    log.log(f"[{name}] YOLO result: {result} ({conf:.0%}) in {yolo_ms:.0f}ms")
                    details["yolo"] = result
                    details["yolo_conf"] = round(conf * 100)

                    if (result != "No card" and conf >= self.yolo_min_conf
                            and name not in force_claude):
                        # YOLO confident — accept it
                        total_ms = (time.time() - t0) * 1000
                        self.last_card[name] = result
                        self.zone_state[name] = "recognized"
                        details["final"] = result
                        details["source"] = "yolo"
                        self._stats_cb("yolo_right")
                        log.log(f"[{name}] RECOGNIZED: {result} (total {total_ms:.0f}ms)")
                        self._save(name, crop, result)
                        self._announce_card(name, result)
                    else:
                        # Need Claude
                        need_claude[name] = (crop, details)
                        self.zone_state[name] = "processing"
                        continue
                else:
                    need_claude[name] = (crop, details)
                    self.zone_state[name] = "processing"
                    continue
            except Exception as e:
                log.log(f"[{name}] YOLO error: {e}")
                self.zone_state[name] = "empty"

            self.recognition_details[name] = details
            self.recognition_crops[name] = crop
            self.pending[name] = False

        # Phase 2: Batch Claude call for all low-confidence zones
        if need_claude and self.client:
            self._recognize_claude_batch(need_claude)
        else:
            # No Claude available — mark remaining as empty
            for name, (crop, details) in need_claude.items():
                yolo_result = details.get("yolo", "")
                if yolo_result and yolo_result != "No card":
                    # Accept low-conf YOLO rather than nothing
                    self.last_card[name] = yolo_result
                    self.zone_state[name] = "recognized"
                    details["final"] = yolo_result
                    details["source"] = "yolo"
                    self._stats_cb("yolo_right")
                    log.log(f"[{name}] RECOGNIZED (low conf, no Claude): {yolo_result}")
                    self._save(name, crop, yolo_result)
                    self._announce_card(name, yolo_result)
                else:
                    self.zone_state[name] = "empty"
                self.recognition_details[name] = details
                self.recognition_crops[name] = crop
                self.pending[name] = False

    def _recognize_claude_batch(self, need_claude):
        """Single Claude API call with all zone images."""
        names = list(need_claude.keys())
        log.log(f"[CLAUDE] Batch call for {len(names)} zones: {', '.join(names)}")

        # Build multi-image message
        content = []
        for name in names:
            crop, details = need_claude[name]
            b64 = base64.b64encode(
                cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            ).decode()
            content.append({"type": "text", "text": f"Image {name}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
        content.append({"type": "text", "text":
            "For each labeled image above, identify the playing card. "
            "Reply with one line per image: 'Name: Rank of Suit' (e.g. 'Steve: 4 of Clubs'). "
            "If unclear or no card visible, reply 'Name: No card'."})

        # Retry with exponential backoff on 529 overloaded / 500 errors
        t0 = time.time()
        resp = None
        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=200,
                    messages=[{"role": "user", "content": content}])
                break
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_transient = "529" in err_str or "overloaded" in err_str.lower() or "500" in err_str
                if not is_transient or attempt == 2:
                    raise
                delay = 2 ** attempt  # 1s, 2s, 4s
                log.log(f"[CLAUDE] Transient error (attempt {attempt+1}/3), retrying in {delay}s: {err_str[:80]}")
                time.sleep(delay)
        try:
            claude_ms = (time.time() - t0) * 1000
            raw = resp.content[0].text.strip()
            log.log(f"[CLAUDE] Response in {claude_ms:.0f}ms: {raw}")

            # Parse response lines
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Match "Name: Rank of Suit" or "Name: No card"
                for name in names:
                    if line.lower().startswith(name.lower() + ":"):
                        value = line[len(name)+1:].strip()
                        crop, details = need_claude[name]
                        m = re.search(r'(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)', value, re.I)
                        if m:
                            result = f"{m.group(1).capitalize()} of {m.group(2).capitalize()}"
                            details["claude"] = result
                            details["final"] = result
                            details["source"] = "claude"
                            self._stats_cb("claude_right")
                            self.last_card[name] = result
                            self.zone_state[name] = "recognized"
                            log.log(f"[{name}] RECOGNIZED (Claude): {result}")
                            self._save(name, crop, result)
                            self._announce_card(name, result)
                        else:
                            details["claude"] = "No card"
                            self.zone_state[name] = "empty"
                            log.log(f"[{name}] Claude: No card")
                        self.recognition_details[name] = details
                        self.recognition_crops[name] = crop
                        self.pending[name] = False
                        break

            # Handle any names not found in response
            for name in names:
                if self.pending.get(name):
                    crop, details = need_claude[name]
                    # Fall back to low-conf YOLO if available
                    yolo_result = details.get("yolo", "")
                    if yolo_result and yolo_result != "No card":
                        details["final"] = yolo_result
                        details["source"] = "yolo"
                        self._stats_cb("yolo_right")
                        self.last_card[name] = yolo_result
                        self.zone_state[name] = "recognized"
                        log.log(f"[{name}] RECOGNIZED (YOLO fallback): {yolo_result}")
                        self._save(name, crop, yolo_result)
                        self._announce_card(name, yolo_result)
                    else:
                        self.zone_state[name] = "empty"
                    self.recognition_details[name] = details
                    self.recognition_crops[name] = crop
                    self.pending[name] = False

        except Exception as e:
            log.log(f"[CLAUDE] Batch error: {e}")
            # Fall back to YOLO results for all
            for name in names:
                if not self.pending.get(name):
                    continue
                crop, details = need_claude[name]
                yolo_result = details.get("yolo", "")
                if yolo_result and yolo_result != "No card":
                    details["final"] = yolo_result
                    details["source"] = "yolo"
                    self._stats_cb("yolo_right")
                    self.last_card[name] = yolo_result
                    self.zone_state[name] = "recognized"
                    log.log(f"[{name}] RECOGNIZED (YOLO, Claude failed): {yolo_result}")
                    self._save(name, crop, yolo_result)
                    self._announce_card(name, yolo_result)
                else:
                    self.zone_state[name] = "empty"
                self.recognition_details[name] = details
                self.recognition_crops[name] = crop
                self.pending[name] = False

    def check_single(self, frame, zone):
        name = zone["name"]
        if name not in self.baselines:
            return None
        crop = self._crop(frame, zone)
        if crop is None or crop.size == 0:
            return None
        bl = self.baselines[name]
        if crop.shape != bl.shape:
            return None
        if float(np.mean(cv2.absdiff(crop, bl))) > self.threshold:
            return crop.copy()
        return None

    def _crop(self, frame, z):
        h, w = frame.shape[:2]
        cx, cy, r = z["cx"], z["cy"], z["r"]
        x1, y1 = max(0, cx-r), max(0, cy-r)
        x2, y2 = min(w, cx+r), min(h, cy+r)
        return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None

    def _recognize_single(self, name, crop):
        """Recognize a single zone (used by test mode and deal mode)."""
        details = {"yolo": "", "yolo_conf": 0, "claude": "", "final": ""}
        try:
            if self._yolo_model is not None:
                result, conf = self._recognize_yolo(crop)
                details["yolo"] = result
                details["yolo_conf"] = round(conf * 100)
                if result != "No card" and conf >= self.yolo_min_conf:
                    self.last_card[name] = result
                    self.zone_state[name] = "recognized"
                    details["final"] = result
                    self._save(name, crop, result)
                    return
                if self.client:
                    result = self._recognize_claude(crop)
                    details["claude"] = result
                    if result and result != "No card":
                        self.last_card[name] = result
                        self.zone_state[name] = "recognized"
                        details["final"] = result
                        self._save(name, crop, result)
                        return
            self.zone_state[name] = "empty"
        except Exception as e:
            log.log(f"[{name}] error: {e}")
            self.zone_state[name] = "empty"
        finally:
            self.recognition_details[name] = details
            self.recognition_crops[name] = crop
            self.pending[name] = False

    def _recognize_yolo(self, crop):
        with self._yolo_lock:
            results = self._yolo_model.predict(crop, conf=0.2, verbose=False)
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = results[0].names[cls_idx]
            # Convert class name like "4c" to "4 of Clubs"
            rank = cls_name[:-1]
            suit_letter = cls_name[-1]
            suit = {"c": "Clubs", "d": "Diamonds", "h": "Hearts", "s": "Spades"}.get(suit_letter, suit_letter)
            rank_name = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}.get(rank, rank)
            return f"{rank_name} of {suit}", conf
        return "No card", 0.0

    def _recognize_claude(self, crop):
        b64 = base64.b64encode(
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        ).decode()
        resp = self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=20,
            messages=[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},
                {"type":"text","text":"What playing card is this? Reply ONLY: 'Rank of Suit' (e.g. '4 of Clubs'). If unclear: 'No card'"},
            ]}])
        raw = resp.content[0].text.strip()
        m = re.search(r'(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)', raw, re.I)
        return f"{m.group(1).capitalize()} of {m.group(2).capitalize()}" if m else "No card"

    def _save(self, name, crop, result):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = result.replace(" ", "_").replace("/", "-")[:30]
        base = TRAINING_DIR / f"{ts}_{name}_{safe}"
        cv2.imwrite(str(base.with_suffix(".jpg")), crop)
        base.with_suffix(".txt").write_text(result)
        # Remember so a later user correction can delete this (wrong-label)
        # pair and replace it with the correct one.
        if not hasattr(self, "last_save_base"):
            self.last_save_base = {}
        self.last_save_base[name] = base

    def _delete_last_save(self, name):
        """Remove the most recent training_data save for this zone. Used
        when the user corrects a misrecognition so the bad label doesn't
        poison future YOLO training runs."""
        base = getattr(self, "last_save_base", {}).get(name)
        if base is None:
            return False
        removed = False
        for suffix in (".jpg", ".txt"):
            p = base.with_suffix(suffix)
            try:
                if p.exists():
                    p.unlink()
                    removed = True
            except Exception as e:
                log.log(f"[{name}] failed to remove stale training file {p}: {e}")
        self.last_save_base.pop(name, None)
        return removed

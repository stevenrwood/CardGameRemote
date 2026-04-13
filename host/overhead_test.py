#!/usr/bin/env python3
"""
Overhead Camera Card Recognition Test Harness

Single-page browser UI at http://localhost:8888
Terminal is only used for startup — all interaction in the browser.

Usage:
    python overhead_test.py [--camera 0] [--threshold 30.0] [--resolution auto]
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock
from queue import Queue, Empty

import cv2
import http.server
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAYER_NAMES = ["Steve", "Bill", "David", "Joe", "Rodney"]
NUM_ZONES = len(PLAYER_NAMES)

DEFAULT_CAMERA_INDEX = 0
DEFAULT_THRESHOLD = 30.0
DEFAULT_RESOLUTION = "auto"

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
TRAINING_DIR = Path(__file__).parent / "training_data"
CONFIG_FILE = Path(__file__).parent.parent / "local" / "config.json"
CAPTURE_FILE = Path("/tmp/card_scanner_frame.jpg")
YOLO_MODEL_PATH = Path(__file__).parent / "models" / "card_detector.pt"

CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Speech queue
# ---------------------------------------------------------------------------

class SpeechQueue:
    def __init__(self):
        self._queue = Queue()
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
                subprocess.run(["say", p], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)

speech = SpeechQueue()

# ---------------------------------------------------------------------------
# Log buffer
# ---------------------------------------------------------------------------

LOG_FILE = Path.home() / "Downloads" / "log.txt"

class LogBuffer:
    def __init__(self, maxlines=500):
        self._lines = []
        self._lock = Lock()
        # Clear log file on startup
        LOG_FILE.write_text("")

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(f"  {msg}")
        with self._lock:
            self._lines.append(line)
            self._lines = self._lines[-500:]
        # Append to file
        try:
            with open(LOG_FILE, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def get(self, n=50):
        with self._lock:
            return list(self._lines[-n:])

log = LogBuffer()

# ---------------------------------------------------------------------------
# Frame capture via ffmpeg
# ---------------------------------------------------------------------------

class FrameCapture:
    def __init__(self, camera_index, resolution="auto"):
        self.camera_index = camera_index
        self._check_ffmpeg()
        self.resolution = self._find_best_resolution() if resolution == "auto" else resolution
        w, h = self.resolution.split("x")
        self.width, self.height = int(w), int(h)

    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        except FileNotFoundError:
            print("  ERROR: ffmpeg not found. Install with: brew install ffmpeg")
            sys.exit(1)

    def _find_best_resolution(self):
        log.log("Auto-detecting resolution...")
        for res in ["3840x2160", "2560x1440", "1920x1080", "1280x720"]:
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
                    "-f", "avfoundation", "-video_size", res, "-framerate", "5",
                    "-i", f"{self.camera_index}:none", "-frames:v", "1",
                    "-q:v", "2", str(CAPTURE_FILE)
                ], capture_output=True, timeout=10, stdin=subprocess.DEVNULL)
                frame = cv2.imread(str(CAPTURE_FILE))
                if frame is not None and f"{frame.shape[1]}x{frame.shape[0]}" == res:
                    log.log(f"  {res} — OK")
                    return res
                log.log(f"  {res} — skipped")
            except Exception:
                log.log(f"  {res} — failed")
        return "1920x1080"

    def capture(self):
        try:
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
                "-f", "avfoundation", "-video_size", self.resolution,
                "-framerate", "5", "-i", f"{self.camera_index}:none",
                "-frames:v", "1", "-q:v", "2", str(CAPTURE_FILE)
            ], capture_output=True, timeout=10, stdin=subprocess.DEVNULL)
            return cv2.imread(str(CAPTURE_FILE))
        except Exception as e:
            log.log(f"Capture error: {e}")
            return None

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class Calibration:
    def __init__(self):
        self.circle_center = None
        self.circle_radius = None
        self.zones = []

    def save(self):
        data = {"circle_center": list(self.circle_center) if self.circle_center else None,
                "circle_radius": self.circle_radius, "zones": self.zones}
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
        self.zones = data.get("zones", [])
        if self.zones and "cx" not in self.zones[0]:
            self.zones = []
            return False
        return True

    @property
    def ok(self):
        return self.circle_center and self.circle_radius and len(self.zones) == NUM_ZONES

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def crop_circle(frame, cal):
    if not cal.circle_center or not cal.circle_radius:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, cal.circle_center, cal.circle_radius, 255, -1)
    return cv2.bitwise_and(frame, frame, mask=mask)

def draw_overlay(frame, cal, monitor):
    if cal.circle_center and cal.circle_radius:
        cv2.circle(frame, cal.circle_center, cal.circle_radius, (255,255,255), 2)
    for z in cal.zones:
        name, cx, cy, r = z["name"], z["cx"], z["cy"], z["r"]
        zs = monitor.zone_state.get(name, "empty")
        color = {"recognized":(0,255,0), "processing":(0,255,255)}.get(zs, (255,255,255))
        cv2.circle(frame, (cx,cy), r, color, 2)
        cv2.putText(frame, name, (cx-30, cy-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        card = monitor.last_card.get(name, "")
        if card:
            cv2.putText(frame, card, (cx-60, cy+r+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def to_jpeg(frame, q=85):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else None

# ---------------------------------------------------------------------------
# Zone monitor
# ---------------------------------------------------------------------------

class ZoneMonitor:
    def __init__(self, threshold):
        self.threshold = threshold
        self.baselines = {}
        self.last_card = {}
        self.zone_state = {}
        self.pending = {}
        self._yolo_model = None
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
        for z in _state.cal.zones:
            crop = self._crop(frame, z)
            if crop is not None and crop.size > 0:
                self.baselines[z["name"]] = crop.copy()
                self.zone_state[z["name"]] = "empty"
                self.last_card[z["name"]] = ""
                self.pending[z["name"]] = False
        log.log("Baselines captured")

    def check_zones(self, frame):
        for z in _state.cal.zones:
            name = z["name"]
            if name not in self.baselines or self.pending.get(name):
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
                self.zone_state[name] = "processing"
                self.pending[name] = True
                Thread(target=self._recognize, args=(name, crop.copy()), daemon=True).start()

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

    def _recognize(self, name, crop):
        t0 = time.time()
        try:
            result = None

            # Try YOLO first
            if self._yolo_model is not None:
                log.log(f"[{name}] YOLO inference started")
                t_yolo = time.time()
                result, conf = self._recognize_yolo(crop)
                yolo_ms = (time.time() - t_yolo) * 1000
                log.log(f"[{name}] YOLO result: {result} ({conf:.0%}) in {yolo_ms:.0f}ms")

                if result == "No card" or conf < 0.5:
                    yolo_result = result
                    if self.client:
                        log.log(f"[{name}] YOLO low confidence, calling Claude API...")
                        t_claude = time.time()
                        result = self._recognize_claude(crop)
                        claude_ms = (time.time() - t_claude) * 1000
                        log.log(f"[{name}] Claude result: {result} in {claude_ms:.0f}ms")
                    elif result == "No card":
                        result = None
            elif self.client:
                log.log(f"[{name}] No YOLO model, calling Claude API...")
                t_claude = time.time()
                result = self._recognize_claude(crop)
                claude_ms = (time.time() - t_claude) * 1000
                log.log(f"[{name}] Claude result: {result} in {claude_ms:.0f}ms")
            else:
                self.zone_state[name] = "empty"
                return

            total_ms = (time.time() - t0) * 1000
            if result and result != "No card":
                self.last_card[name] = result
                self.zone_state[name] = "recognized"
                log.log(f"[{name}] RECOGNIZED: {result} (total {total_ms:.0f}ms)")
                self._save(name, crop, result)
                speech.say(f"{name}, {result}")
            else:
                log.log(f"[{name}] No card (total {total_ms:.0f}ms)")
                self.zone_state[name] = "empty"

        except Exception as e:
            log.log(f"[{name}] error: {e}")
            self.zone_state[name] = "empty"
        finally:
            self.pending[name] = False

    def _recognize_yolo(self, crop):
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
        safe = result.replace(" ","_").replace("/","-")[:30]
        cv2.imwrite(str(TRAINING_DIR / f"{ts}_{name}_{safe}.jpg"), crop)
        (TRAINING_DIR / f"{ts}_{name}_{safe}.txt").write_text(result)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, capture, cal, monitor):
        self.capture = capture
        self.cal = cal
        self.monitor = monitor
        self.monitoring = False
        self.latest_frame = None
        self.latest_jpg = None  # cropped + overlay
        self.quit_flag = False
        self.test_mode = None   # None or {"zone_idx":0, "waiting":"card"|"confirm", "result":""}
        # Deal test mode
        self.deal_mode = None
        # Data collection mode
        self.collect_mode = None  # None or {"card_idx":0, "pass":1, "captured":False}

_state = None

# ---------------------------------------------------------------------------
# Deal mode — dictation for game name, then visual recognition for cards
# ---------------------------------------------------------------------------

# Game templates: map game name to list of deal patterns
# Each pattern is a list of "up" or "down" per card, dealt to all players in order
GAME_PATTERNS = {
    "5 Card Draw": ["down"] * 5,
    "3 Toed Pete": ["down"] * 3,
    "7 Card Stud": ["down", "down", "up", "up", "up", "up", "down"],
    "7 Stud Deuces Wild": ["down", "down", "up", "up", "up", "up", "down"],
    "Follow the Queen": ["down", "down", "up", "up", "up", "up", "down"],
    "High Chicago": ["down", "down", "up", "up", "up", "up", "down"],
    "High Low High Challenge": ["down"] * 3,
    "7 27": ["down"] * 2,
    "Texas Hold'em": ["down"] * 2,
}


def _get_deal_order(dealer_name):
    """Return player names in deal order (clockwise from left of dealer)."""
    try:
        idx = [n.lower() for n in PLAYER_NAMES].index(dealer_name.lower())
    except ValueError:
        idx = 0
    # Start with player to dealer's left (next clockwise)
    order = []
    for i in range(1, len(PLAYER_NAMES) + 1):
        order.append(PLAYER_NAMES[(idx + i) % len(PLAYER_NAMES)])
    return order


def _start_deal_mode(s):
    if s.deal_mode:
        return
    s.deal_mode = {
        "phase": "game_select",  # "game_select", "dealing", "complete"
        "game": None,
        "dealer": None,
        "deal_order": list(PLAYER_NAMES),  # updated when dealer is set
        "pattern": [],
        "cards": [],
        "round_idx": 0,
        "player_idx": 0,
        "announced": set(),
    }
    log.log("Deal mode started — select dealer and game")


def _stop_deal_mode(s):
    s.deal_mode = None
    log.log("Deal mode stopped")


def _set_deal_game(s, game_name):
    """Set the game and start dealing."""
    if not s.deal_mode:
        return
    pattern = GAME_PATTERNS.get(game_name)
    if not pattern:
        log.log(f"[DEAL] Unknown game pattern: {game_name}")
        return
    s.deal_mode["game"] = game_name
    s.deal_mode["pattern"] = pattern
    s.deal_mode["phase"] = "dealing"
    s.deal_mode["round_idx"] = 0
    s.deal_mode["cards"] = []
    s.deal_mode["round_results"] = {}
    s.deal_mode["retry_time"] = 0

    # Capture baselines before dealing starts
    if s.latest_frame is not None:
        s.monitor.capture_baselines(s.latest_frame)
        log.log("[DEAL] Baselines captured")

    # Skip initial down card rounds
    _advance_to_next_up(s)

    order = s.deal_mode["deal_order"]
    log.log(f"[DEAL] Game: {game_name}, dealer: {s.deal_mode['dealer']}")
    log.log(f"[DEAL] Deal order: {' -> '.join(order)}")
    log.log(f"[DEAL] Pattern: {pattern}")
    if s.deal_mode["phase"] == "dealing":
        dealer_name = order[-1]
        log.log(f"[DEAL] Round {s.deal_mode['round_idx']+1}: waiting for {dealer_name}'s card (last dealt)")


def _advance_to_next_up(s):
    """Advance to next up card round, skipping down cards."""
    dm = s.deal_mode
    if not dm:
        return
    pattern = dm["pattern"]

    while dm["round_idx"] < len(pattern):
        if pattern[dm["round_idx"]] == "up":
            dm["phase"] = "dealing"
            dm["round_results"] = {}
            dm["retry_time"] = 0
            return
        log.log(f"[DEAL] Skipping round {dm['round_idx']+1} (down cards — use scanner)")
        dm["round_idx"] += 1

    dm["phase"] = "complete"
    log.log("[DEAL] All rounds dealt")


def _deal_scan_all_zones(s):
    """Scan all player zones and recognize cards. No baseline comparison — just crop and recognize."""
    dm = s.deal_mode
    if not dm:
        return

    frame = s.latest_frame
    if frame is None:
        return

    order = dm["deal_order"]
    missing = []

    for player in order:
        # Skip already recognized this round
        if player in dm["round_results"]:
            continue

        zone = next((z for z in s.cal.zones if z["name"] == player), None)
        if not zone:
            continue

        # Crop zone directly — don't rely on baseline diff
        crop = s.monitor._crop(frame, zone)
        if crop is None or crop.size == 0:
            missing.append(player)
            continue

        s.monitor._recognize(player, crop)
        result = s.monitor.last_card.get(player, "No card")
        if result and result != "No card":
            dm["round_results"][player] = result
            dm["cards"].append({
                "player": player,
                "card": result,
                "round": dm["round_idx"] + 1,
            })
            s.monitor.zone_state[player] = "empty"
            s.monitor.last_card[player] = ""
        else:
            missing.append(player)

    # Announce recognized cards
    for player in order:
        if player in dm["round_results"] and player not in dm.get("announced_this_round", set()):
            dm.setdefault("announced_this_round", set()).add(player)

    if missing:
        names = " and ".join(missing)
        log.log(f"[DEAL] Missing: {names}")
        speech.say(f"{names}, adjust your cards please")
        dm["phase"] = "retry_missing"
        dm["retry_time"] = time.time()
    else:
        # All recognized — recapture baselines WITH cards, then wait for removal
        log.log(f"[DEAL] Round {dm['round_idx']+1} complete — all {len(order)} cards recognized")
        # Capture baselines with cards present — clearing = change from this
        if s.latest_frame is not None:
            s.monitor.capture_baselines(s.latest_frame)
        speech.say("Clear zones")
        dm["phase"] = "waiting_to_clear"
        log.log("[DEAL] Waiting for all zones to be cleared")


def _deal_check_dealer_zone(s):
    """Check if the dealer (last in order) has a card — triggers full scan."""
    dm = s.deal_mode
    if not dm or dm["phase"] != "dealing":
        return

    frame = s.latest_frame
    if frame is None:
        return

    order = dm["deal_order"]
    dealer_name = order[-1]  # dealer gets card last
    dealer_zone = next((z for z in s.cal.zones if z["name"] == dealer_name), None)
    if not dealer_zone:
        return

    crop = s.monitor.check_single(frame, dealer_zone)
    if crop is not None:
        log.log(f"[DEAL] Card detected in {dealer_name}'s zone — scanning all zones")
        dm["phase"] = "scanning"
        dm["announced_this_round"] = set()
        _deal_scan_all_zones(s)


def _deal_retry_missing(s):
    """After 5 seconds, rescan missing zones."""
    dm = s.deal_mode
    if not dm or dm["phase"] != "retry_missing":
        return
    if time.time() - dm["retry_time"] < 5:
        return

    log.log("[DEAL] Retrying missing zones...")
    dm["phase"] = "scanning"
    _deal_scan_all_zones(s)


def _deal_check_zones_clear(s):
    """Check if all zones are empty — players moved cards out."""
    dm = s.deal_mode
    if not dm or dm["phase"] != "waiting_to_clear":
        return

    frame = s.latest_frame
    if frame is None:
        return

    order = dm["deal_order"]
    still_occupied = []
    for player in order:
        zone = next((z for z in s.cal.zones if z["name"] == player), None)
        if not zone:
            continue
        crop = s.monitor._crop(frame, zone)
        if crop is None or crop.size == 0:
            continue
        baseline = s.monitor.baselines.get(player)
        if baseline is None or crop.shape != baseline.shape:
            continue
        diff = float(np.mean(cv2.absdiff(crop, baseline)))
        # Baseline was captured WITH cards — if diff is LOW, card is still there
        # If diff is HIGH, card was removed (zone changed)
        if diff < s.monitor.threshold:
            still_occupied.append(player)

    if not still_occupied:
        # All zones clear — recapture baselines and advance
        log.log("[DEAL] All zones cleared")
        s.monitor.capture_baselines(frame)
        log.log("[DEAL] Baselines recaptured")
        dm["phase"] = "advancing"  # prevent re-entry
        dm["round_idx"] += 1
        dm["round_results"] = {}
        dm["announced_this_round"] = set()
        _advance_to_next_up(s)
        if dm["phase"] == "dealing":
            dealer_name = order[-1]
            log.log(f"[DEAL] Round {dm['round_idx']+1}: waiting for {dealer_name}'s card")
            speech.say("Deal")


def _deal_mode_json(s):
    """Return deal mode state as JSON-serializable dict."""
    dm = s.deal_mode
    if not dm:
        return None
    order = dm.get("deal_order", [])
    dealer_name = order[-1] if order else None
    missing = []
    if dm["phase"] in ("retry_missing", "scanning"):
        for p in order:
            if p not in dm.get("round_results", {}):
                missing.append(p)
    return {
        "phase": dm["phase"],
        "game": dm["game"],
        "dealer": dm.get("dealer"),
        "deal_order": order,
        "cards": dm["cards"],
        "round_results": dm.get("round_results", {}),
        "missing": missing,
        "watching_for": dealer_name if dm["phase"] == "dealing" else None,
        "round_idx": dm.get("round_idx", 0),
        "total_rounds": len(dm.get("pattern", [])),
    }


# ---------------------------------------------------------------------------
# Data collection mode
# ---------------------------------------------------------------------------

COLLECT_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
COLLECT_SUITS = ["clubs", "diamonds", "hearts", "spades"]

SUIT_NAMES = {"clubs": "Clubs", "diamonds": "Diamonds", "hearts": "Hearts", "spades": "Spades"}
RANK_NAMES = {"A": "Ace", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
              "8": "8", "9": "9", "10": "10", "J": "Jack", "Q": "Queen", "K": "King"}

# Pass assignments: which suit goes to which player zone each pass
# 4 suits across 4 players (Bill, David, Joe, Rodney), Steve rotates
COLLECT_PASSES = [
    {"Bill": "clubs", "David": "diamonds", "Joe": "hearts", "Rodney": "spades"},
    {"Bill": "diamonds", "David": "hearts", "Joe": "spades", "Rodney": "clubs"},
    {"Bill": "hearts", "David": "spades", "Joe": "clubs", "Rodney": "diamonds"},
    {"Bill": "spades", "David": "clubs", "Joe": "diamonds", "Rodney": "hearts"},
]


def _start_collect_mode(s):
    if s.collect_mode:
        return
    s.collect_mode = {"rank_idx": 0, "pass_idx": 0, "captured": False, "countdown": 0}
    log.log("Data collection started")
    p = COLLECT_PASSES[0]
    log.log("[COLLECT] Pass 1: " + ", ".join(f"{k}={v.capitalize()}" for k, v in p.items()))


def _stop_collect_mode(s):
    s.collect_mode = None
    log.log("Data collection stopped")


def _collect_deal_info(cm):
    """Return what to deal for current state."""
    if cm["pass_idx"] >= len(COLLECT_PASSES):
        return None
    if cm["rank_idx"] >= len(COLLECT_RANKS):
        return None
    rank = COLLECT_RANKS[cm["rank_idx"]]
    assignments = COLLECT_PASSES[cm["pass_idx"]]
    cards = {}
    for player, suit in assignments.items():
        cards[player] = f"{RANK_NAMES[rank]} of {SUIT_NAMES[suit]}"
    return {"rank": rank, "cards": cards, "pass": cm["pass_idx"] + 1}


def _collect_scan(s):
    """Capture zone crops and save with correct labels per player/suit."""
    cm = s.collect_mode
    if not cm:
        return

    info = _collect_deal_info(cm)
    if not info:
        return

    frame = s.latest_frame
    if frame is None:
        log.log("[COLLECT] No frame available")
        return

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    pass_num = info["pass"]

    for zone in s.cal.zones:
        name = zone["name"]
        label = info["cards"].get(name)
        if not label:
            continue

        crop = s.monitor._crop(frame, zone)
        if crop is None or crop.size == 0:
            log.log(f"[COLLECT] {name}: crop failed")
            continue

        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(" ", "_")
        filename = f"collect_p{pass_num}_{safe_label}_{name}_{ts}"
        cv2.imwrite(str(TRAINING_DIR / f"{filename}.jpg"), crop)
        (TRAINING_DIR / f"{filename}.txt").write_text(label)
        saved += 1
        log.log(f"[COLLECT] {name}: {label}")

    log.log(f"[COLLECT] Saved {saved} images")
    cm["captured"] = True


def _collect_advance(s):
    """Advance to the next rank or pass."""
    cm = s.collect_mode
    if not cm:
        return
    cm["rank_idx"] += 1
    cm["captured"] = False
    if cm["rank_idx"] >= len(COLLECT_RANKS):
        cm["pass_idx"] += 1
        cm["rank_idx"] = 0
        if cm["pass_idx"] < len(COLLECT_PASSES):
            p = COLLECT_PASSES[cm["pass_idx"]]
            log.log(f"[COLLECT] Pass {cm['pass_idx']+1}: " +
                    ", ".join(f"{k}={v.capitalize()}" for k, v in p.items()))
            # Pause for new pass — user needs to re-sort deck
            cm["phase"] = "paused_new_pass"
            speech.say(f"Pass {cm['pass_idx']+1}. New suit assignments. Press Start when ready.")
            return
        else:
            cm["phase"] = "done"
            log.log("[COLLECT] All 4 passes complete!")
            speech.say("Data collection complete")
            return
    # Start the deal phase
    _collect_start_deal(s)


def _collect_start_deal(s):
    """Start the 5-second deal countdown."""
    cm = s.collect_mode
    if not cm:
        return
    cm["phase"] = "dealing"
    cm["timer_start"] = time.time()
    cm["timer_duration"] = 5
    speech.say("Deal")


def _collect_start_clear(s):
    """Start the 5-second clear countdown."""
    cm = s.collect_mode
    if not cm:
        return
    cm["phase"] = "clearing"
    cm["timer_start"] = time.time()
    cm["timer_duration"] = 5
    speech.say("Clear")


def _collect_redo(s):
    """Go back to previous rank and redo."""
    cm = s.collect_mode
    if not cm:
        return
    # Delete the images we just saved for this rank
    if cm["captured"]:
        info = _collect_deal_info(cm)
        if info:
            pass_num = info["pass"]
            rank = info["rank"]
            # Find and delete files matching this rank/pass
            for f in TRAINING_DIR.glob(f"collect_p{pass_num}_*_{rank}_*"):
                f.unlink()
                log.log(f"[COLLECT] Deleted: {f.name}")

    cm["captured"] = False
    cm["phase"] = "paused"
    log.log(f"[COLLECT] Redo — re-deal rank {COLLECT_RANKS[cm['rank_idx']]}")
    speech.say("Redo")


def _collect_auto_cycle(s):
    """Called from bg_loop — handles the timed phases."""
    cm = s.collect_mode
    if not cm or cm.get("phase") not in ("dealing", "clearing"):
        return

    elapsed = time.time() - cm.get("timer_start", 0)
    remaining = cm.get("timer_duration", 5) - elapsed

    if remaining <= 0:
        if cm["phase"] == "dealing":
            # Deal time is up — scan now
            _collect_scan(s)
            _collect_start_clear(s)
        elif cm["phase"] == "clearing":
            # Clear time is up — advance to next rank
            _collect_advance(s)


def _collect_start_first(s):
    """Start the first deal cycle."""
    cm = s.collect_mode
    if not cm:
        return
    _collect_start_deal(s)


def _collect_mode_json(s):
    cm = s.collect_mode
    if not cm:
        return None
    info = _collect_deal_info(cm)
    done = cm["pass_idx"] >= len(COLLECT_PASSES)
    total = len(COLLECT_RANKS) * len(COLLECT_PASSES)
    current = cm["pass_idx"] * len(COLLECT_RANKS) + cm["rank_idx"]
    phase = cm.get("phase", "paused")
    countdown = 0
    if phase in ("dealing", "clearing"):
        elapsed = time.time() - cm.get("timer_start", 0)
        countdown = max(0, int(cm.get("timer_duration", 5) - elapsed))

    return {
        "rank_idx": cm["rank_idx"],
        "pass_idx": cm["pass_idx"],
        "pass_total": len(COLLECT_PASSES),
        "cards": info["cards"] if info else {},
        "rank": COLLECT_RANKS[cm["rank_idx"]] if cm["rank_idx"] < len(COLLECT_RANKS) else None,
        "captured": cm["captured"],
        "done": done,
        "current": current,
        "total": total,
        "countdown": countdown,
        "phase": phase,
    }


def _process_deal_text(s, text):
    """Parse dictated text for game name only."""
    from speech_recognition_module import parse_speech, GameCommand

    if not s.deal_mode or s.deal_mode["phase"] != "game_select":
        return

    log.log(f"[DEAL] Parsing game name: \"{text}\"")
    commands = parse_speech(text)
    for cmd in commands:
        if isinstance(cmd, GameCommand):
            _set_deal_game(s, cmd.game_name)
            return


# ---------------------------------------------------------------------------
# Background capture
# ---------------------------------------------------------------------------

_last_capture_log = [0]  # throttle capture timing logs

def bg_loop():
    while not _state.quit_flag:
        t_cap = time.time()
        frame = _state.capture.capture()
        cap_ms = (time.time() - t_cap) * 1000
        if frame is not None:
            # Log capture timing every 30 seconds
            now = time.time()
            if now - _last_capture_log[0] > 30:
                log.log(f"[CAPTURE] frame grabbed in {cap_ms:.0f}ms")
                _last_capture_log[0] = now
            _state.latest_frame = frame
            disp = crop_circle(frame, _state.cal).copy()
            draw_overlay(disp, _state.cal, _state.monitor)
            _state.latest_jpg = to_jpeg(disp)

            if _state.monitoring and _state.cal.ok:
                _state.monitor.check_zones(frame)

            # Data collection auto-cycle
            if _state.collect_mode:
                _collect_auto_cycle(_state)

            # Test mode: check if card appeared in the active zone
            tm = _state.test_mode
            if tm and tm["waiting"] == "card" and _state.cal.ok:
                zone = _state.cal.zones[tm["zone_idx"]]
                crop = _state.monitor.check_single(frame, zone)
                if crop is not None:
                    log.log(f"[{zone['name']}] Card detected, recognizing...")
                    result = _state.monitor.last_card.get(zone["name"], "")
                    if not result or result == "No card":
                        _state.monitor._recognize(zone["name"], crop)
                        result = _state.monitor.last_card.get(zone["name"], "No card")
                    if result and result != "No card":
                        tm["result"] = result
                        tm["waiting"] = "confirm"
                        tm["confirm_time"] = time.time()
                        speech.say(f"{zone['name']}, {result}")

            # Test mode: auto-confirm after 4 seconds
            if tm and tm["waiting"] == "confirm":
                if time.time() - tm.get("confirm_time", 0) > 4:
                    # Auto-confirm — advance to next zone
                    tm["zone_idx"] += 1
                    if tm["zone_idx"] >= len(_state.cal.zones):
                        _state.test_mode = None
                        log.log("[TEST] All zones tested")
                    else:
                        tm["waiting"] = "card"
                        tm["result"] = ""
                        next_name = _state.cal.zones[tm["zone_idx"]]["name"]
                        speech.say(f"{next_name} is next")
                        log.log(f"[TEST] Auto-confirmed. Next: {next_name}")

            # Deal mode
            dm = _state.deal_mode
            if dm and _state.cal.ok:
                if dm["phase"] == "dealing":
                    _deal_check_dealer_zone(_state)
                elif dm["phase"] == "retry_missing":
                    _deal_retry_missing(_state)
                elif dm["phase"] == "waiting_to_clear":
                    _deal_check_zones_clear(_state)

        time.sleep(1)  # 1 second capture rate

# ---------------------------------------------------------------------------
# Web server — single page app
# ---------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        s = _state
        if not s: return self._r(500,"text/plain","Not ready")
        p = self.path.split("?")[0]
        routes = {
            "/": self._page, "/app": self._page,
            "/calibrate": self._calibrate_page,
            "/snapshot": lambda s: self._jpeg(s.latest_frame),
            "/snapshot/cropped": lambda s: self._r(200,"image/jpeg",s.latest_jpg) if s.latest_jpg else self._r(503,"text/plain","wait"),
            "/api/state": self._api_state,
            "/api/log": lambda s: self._r(200,"application/json",json.dumps({"lines":log.get(100)})),
            "/log": lambda s: self._r(200,"text/plain","\n".join(log.get(200))),
            "/logview": self._serve_logview,
            "/calibration": lambda s: self._r(200,"application/json",CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else self._r(404,"text/plain","none"),
            "/training": self._training_list,
        }
        if p in routes:
            routes[p](s)
        elif p.startswith("/zone/"):
            self._zone_img(s, p[6:])
        elif p.startswith("/training/"):
            self._training_file(p[10:])
        else:
            self._r(404,"text/plain","Not found")

    def do_POST(self):
        s = _state
        if not s: return self._r(500,"text/plain","Not ready")
        body = self.rfile.read(int(self.headers.get("Content-Length",0))).decode()
        data = json.loads(body) if body else {}
        p = self.path

        if p == "/api/calibrate/save":
            cc = data.get("circle_center")
            s.cal.circle_center = tuple(cc) if cc else None
            s.cal.circle_radius = data.get("circle_radius")
            s.cal.zones = data.get("zones", [])
            s.cal.save()
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/monitor/start":
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
                s.monitoring = True
            self._r(200,"application/json",json.dumps({"monitoring":s.monitoring}))

        elif p == "/api/monitor/stop":
            s.monitoring = False
            self._r(200,"application/json",'{"monitoring":false}')

        elif p == "/api/baselines":
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/start":
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
                s.test_mode = {"zone_idx": 0, "waiting": "card", "result": ""}
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/confirm":
            tm = s.test_mode
            if tm:
                correct = data.get("correct", True)
                if correct:
                    tm["zone_idx"] += 1
                    if tm["zone_idx"] >= len(s.cal.zones):
                        s.test_mode = None
                    else:
                        tm["waiting"] = "card"
                        tm["result"] = ""
                else:
                    # Retry same zone
                    tm["waiting"] = "card"
                    tm["result"] = ""
                    name = s.cal.zones[tm["zone_idx"]]["name"]
                    s.monitor.zone_state[name] = "empty"
                    s.monitor.last_card[name] = ""
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/skip":
            tm = s.test_mode
            if tm:
                tm["zone_idx"] += 1
                if tm["zone_idx"] >= len(s.cal.zones):
                    s.test_mode = None
                else:
                    tm["waiting"] = "card"
                    tm["result"] = ""
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/stop":
            s.test_mode = None
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/start":
            _start_deal_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/stop":
            _stop_deal_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/dealer":
            dealer = data.get("dealer", "")
            if s.deal_mode:
                s.deal_mode["dealer"] = dealer
                s.deal_mode["deal_order"] = _get_deal_order(dealer)
                log.log(f"[DEAL] Dealer: {dealer}, order: {' -> '.join(s.deal_mode['deal_order'])}")
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/text":
            text = data.get("text", "")
            _process_deal_text(s, text)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/clear":
            _stop_deal_mode(s)
            _start_deal_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/start":
            _start_collect_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/stop":
            _stop_collect_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/go":
            _collect_start_first(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/pause":
            if s.collect_mode:
                s.collect_mode["phase"] = "paused"
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/resume":
            if s.collect_mode:
                _collect_start_deal(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/redo":
            _collect_redo(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/snapshot/save":
            if s.latest_frame is not None:
                cropped = crop_circle(s.latest_frame, s.cal)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = Path(__file__).parent / f"snapshot_{ts}.jpg"
                cv2.imwrite(str(path), cropped)
                log.log(f"Snapshot saved: {path.name}")
            self._r(200,"application/json",'{"ok":true}')

        else:
            self._r(404,"text/plain","Not found")

    def _r(self, code, ct, body):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ct)
            if ct == "image/jpeg":
                self.send_header("Cache-Control","no-store,no-cache,max-age=0")
            self.end_headers()
            self.wfile.write(body.encode() if isinstance(body,str) else body)
        except (ConnectionResetError, BrokenPipeError):
            pass

    def _jpeg(self, frame):
        if frame is None: return self._r(503,"text/plain","No frame")
        j = to_jpeg(frame, 80)
        if j: self._r(200,"image/jpeg",j)

    def _api_state(self, s):
        tm = s.test_mode
        test_info = None
        if tm:
            idx = tm["zone_idx"]
            test_info = {
                "zone": s.cal.zones[idx]["name"] if idx < len(s.cal.zones) else None,
                "zone_idx": idx,
                "total": len(s.cal.zones),
                "waiting": tm["waiting"],
                "result": tm["result"],
            }
        self._r(200,"application/json",json.dumps({
            "monitoring": s.monitoring,
            "calibrated": s.cal.ok,
            "resolution": s.capture.resolution,
            "test_mode": test_info,
            "deal_mode": _deal_mode_json(s),
            "collect_mode": _collect_mode_json(s),
            "zones": {z["name"]: {"state": s.monitor.zone_state.get(z["name"],"empty"),
                                   "card": s.monitor.last_card.get(z["name"],"")}
                      for z in s.cal.zones},
        }))

    def _page(self, s):
        players_js = json.dumps(PLAYER_NAMES)
        self._r(200,"text/html",f"""<!DOCTYPE html>
<html><head><title>Card Scanner</title><meta name="viewport" content="width=device-width">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:12px}}
button{{padding:8px 14px;border:none;border-radius:6px;cursor:pointer;font-size:.9em;margin:3px}}
.btn-blue{{background:#0f3460;color:#fff}} .btn-blue:hover{{background:#1a4a7a}}
.btn-green{{background:#1b5e20;color:#fff}} .btn-green:hover{{background:#2e7d32}}
.btn-red{{background:#b71c1c;color:#fff}} .btn-red:hover{{background:#c62828}}
.btn-orange{{background:#e65100;color:#fff}}
.btn-off{{background:#333;color:#888}}
img{{border:1px solid #333;border-radius:4px}}
#toolbar{{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px;padding:8px;background:#16213e;border-radius:8px}}
#main{{display:flex;gap:12px;flex-wrap:wrap}}
#left{{flex:1;min-width:300px}} #right{{width:320px;flex-shrink:0}}
.zone{{padding:8px;margin:4px 0;border:2px solid #444;border-radius:8px;display:flex;align-items:center;gap:8px}}
.zone img{{width:100px;height:100px;object-fit:cover;border-radius:4px}}
.zone-info{{flex:1}}
.zone-name{{font-weight:bold;font-size:1.1em}}
.zone-card{{font-size:1.2em;margin-top:2px}}
pre{{background:#0d1117;padding:8px;border-radius:6px;font-size:.8em;max-height:200px;overflow:auto;margin-top:8px}}
#test-panel{{background:#0f3460;padding:12px;border-radius:8px;margin:8px 0;display:none}}
#test-panel h3{{margin-bottom:8px}}
#status-bar{{font-size:.85em;color:#888;margin-top:8px}}
</style></head><body>
<div id="toolbar">
  <button class="btn-blue" onclick="location.href='/calibrate'">Calibrate</button>
  <button class="btn-green" id="btn-monitor" onclick="toggleMonitor()">Start Monitor</button>
  <button class="btn-blue" onclick="ensureLogWindow();startTest()">Test Recognition</button>
  <button class="btn-blue" id="btn-deal" onclick="toggleDeal()">Test Dealing</button>
  <button class="btn-blue" onclick="resetBaselines()">Reset Baselines</button>
  <button class="btn-blue" onclick="saveSnapshot()">Snapshot</button>
  <button class="btn-blue" id="btn-collect" onclick="toggleCollect()">Collect Data</button>
  <span id="status-bar">Loading...</span>
</div>
<div id="test-panel">
  <h3 id="test-title">Test Mode</h3>
  <p id="test-prompt"></p>
  <div id="test-buttons" style="margin-top:8px">
    <button class="btn-green" onclick="testConfirm(true)">Correct</button>
    <button class="btn-red" onclick="testConfirm(false)">Incorrect</button>
    <button class="btn-orange" onclick="testSkip()">Skip</button>
    <button class="btn-off" onclick="testStop()">Stop Test</button>
  </div>
</div>
<div id="collect-panel" style="display:none;background:#1b5e20;padding:12px;border-radius:8px;margin:8px 0">
  <h3>Data Collection</h3>
  <div id="collect-info">
    <p style="font-size:1.1em;color:#fff;margin:8px 0">
      Deal rank <span id="collect-rank" style="color:#ff0;font-weight:bold;font-size:1.3em">—</span> to each player:
    </p>
    <div id="collect-assignments" style="margin:6px 0"></div>
    <p style="font-size:.9em;color:#aaa">
      Pass <span id="collect-pass">1</span>/<span id="collect-pass-total">4</span> |
      Card <span id="collect-idx">0</span>/<span id="collect-total">52</span>
    </p>
    <div style="background:#333;border-radius:4px;height:8px;margin:8px 0">
      <div id="collect-progress" style="background:#4caf50;height:100%;border-radius:4px;width:0%"></div>
    </div>
  </div>
  <div id="collect-done" style="display:none">
    <p style="font-size:1.2em;color:#4caf50">All cards collected!</p>
  </div>
  <div style="margin-top:8px">
    <button class="btn-green" id="collect-go-btn" onclick="collectGo()" style="font-size:1.1em;padding:10px 20px">Start</button>
    <button class="btn-orange" id="collect-pause-btn" onclick="collectPause()" style="display:none">Pause</button>
    <button class="btn-blue" id="collect-redo-btn" onclick="collectRedo()" style="display:none">Redo</button>
    <span id="collect-countdown" style="font-size:2.5em;color:#ff0;margin:0 16px;display:none"></span>
    <span id="collect-phase-label" style="font-size:1em;color:#aaa;display:none"></span>
    <button class="btn-red" onclick="toggleCollect()" style="margin-left:8px">Stop</button>
  </div>
</div>
<div id="deal-panel" style="display:none;background:#0f3460;padding:12px;border-radius:8px;margin:8px 0">
  <h3>Test Dealing</h3>
  <p style="margin:4px 0">Game: <span id="deal-game" style="color:#4fc3f7;font-size:1.1em">—</span></p>
  <div id="deal-game-input" style="margin:6px 0">
    <div style="margin:4px 0">
      <label style="font-size:.85em;color:#888">Dealer: </label>
      <select id="deal-dealer" style="padding:6px;border-radius:4px;border:1px solid #444;background:#1a1a2e;color:#e0e0e0"
        onchange="setDealer()">
      </select>
      <span id="deal-order-display" style="font-size:.8em;color:#888;margin-left:8px"></span>
    </div>
    <div style="margin:4px 0">
      <label style="font-size:.85em;color:#888">Game (dictate or type): </label>
      <input id="deal-input" type="text" placeholder="e.g. 'Follow the Queen'"
        style="width:60%;padding:8px;border-radius:6px;border:1px solid #444;background:#1a1a2e;color:#e0e0e0;font-size:1em">
      <button class="btn-green" onclick="submitGameName()" style="margin-left:4px">Start</button>
    </div>
  </div>
  <div id="deal-status" style="margin:6px 0;display:none">
    <p style="color:#4fc3f7;font-size:.95em">Watching: <span id="deal-active-player" style="font-weight:bold">—</span>'s zone
      (round <span id="deal-round">—</span>/<span id="deal-total">—</span>)</p>
  </div>
  <div id="deal-cards" style="margin:8px 0"></div>
  <button class="btn-orange" onclick="clearDeal()">Clear</button>
  <button class="btn-red" onclick="toggleDeal()">Stop Dealing</button>
</div>
<div id="main">
  <div id="left">
    <img id="tableimg" src="/snapshot/cropped" style="width:100%;cursor:pointer"
         onclick="this.src='/snapshot/cropped?'+Date.now()">
    <div style="margin-top:8px">
      <button class="btn-off" style="padding:3px 8px;font-size:.75em" onclick="openLogWindow()">Log Window</button>
      <a href="/training" style="color:#4fc3f7;font-size:.8em;margin-left:8px">Training data</a>
    </div>
  </div>
  <div id="right">
    <h3>Zones</h3>
    <div id="zones"></div>
  </div>
</div>
<script>
var players={players_js};
var monitoring=false, testMode=null;

function api(path, data){{
  return fetch(path,{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify(data||{{}})}}).then(function(r){{return r.json()}});
}}

function toggleMonitor(){{
  if(!monitoring) ensureLogWindow();
  if(monitoring) api('/api/monitor/stop').then(update);
  else api('/api/monitor/start').then(update);
}}
function startTest(){{
  if(!confirm('Clear all cards from table, then click OK')) return;
  api('/api/test/start').then(update);
}}
function testConfirm(correct){{ api('/api/test/confirm',{{correct:correct}}).then(update); }}
function testSkip(){{ api('/api/test/skip').then(update); }}
function testStop(){{ api('/api/test/stop').then(update); }}
function toggleDeal(){{
  fetch('/api/state').then(function(r){{return r.json()}}).then(function(d){{
    if(d.deal_mode) api('/api/deal/stop').then(update);
    else {{ ensureLogWindow(); api('/api/deal/start').then(function(){{
      update();
      setTimeout(function(){{
        var inp=document.getElementById('deal-input');
        if(inp) inp.focus();
      }}, 300);
    }}); }}
  }});
}}
function clearDeal(){{
  api('/api/deal/clear').then(function(){{
    var inp=document.getElementById('deal-input');
    if(inp) inp.value='';
    update();
  }});
}}
function submitGameName(){{
  var inp=document.getElementById('deal-input');
  if(inp && inp.value.trim()) {{
    api('/api/deal/text',{{text:inp.value.trim()}}).then(update);
  }}
}}
function setDealer(){{
  var sel=document.getElementById('deal-dealer');
  if(sel) {{
    api('/api/deal/dealer',{{dealer:sel.value}}).then(function(){{
      // Show deal order
      var idx=players.indexOf(sel.value);
      var order=[];
      for(var i=1;i<=players.length;i++) order.push(players[(idx+i)%players.length]);
      var el=document.getElementById('deal-order-display');
      if(el) el.textContent='Deal: '+order.join(' → ');
    }});
  }}
}}
// Allow Enter key to submit game name
document.addEventListener('keydown',function(e){{
  if(e.key==='Enter' && document.activeElement && document.activeElement.id==='deal-input'){{
    submitGameName();
  }}
}});

function resetBaselines(){{
  if(!confirm('Clear all cards, then click OK')) return;
  api('/api/baselines').then(function(){{ log.log && update(); }});
}}
function saveSnapshot(){{ api('/api/snapshot/save'); }}

function toggleCollect(){{
  fetch('/api/state').then(function(r){{return r.json()}}).then(function(d){{
    if(d.collect_mode) api('/api/collect/stop').then(update);
    else {{ ensureLogWindow(); api('/api/collect/start').then(update); }}
  }});
}}
function collectGo(){{
  api('/api/collect/go').then(update);
}}
function collectPause(){{
  api('/api/collect/pause').then(update);
}}
function collectResume(){{
  api('/api/collect/resume').then(update);
}}
function collectRedo(){{
  api('/api/collect/redo').then(update);
}}
var _logWin=null;
function openLogWindow(){{
  if(!_logWin || _logWin.closed)
    _logWin=window.open('/logview','_logview','width=800,height=400,scrollbars=yes');
  else
    _logWin.focus();
}}
function ensureLogWindow(){{
  if(!_logWin || _logWin.closed)
    _logWin=window.open('/logview','_logview','width=800,height=400,scrollbars=yes');
}}

function update(){{
  // Table image
  var ti=new Image();
  ti.onload=function(){{document.getElementById('tableimg').src=ti.src}};
  ti.src='/snapshot/cropped?'+Date.now()+Math.random();

  // State
  fetch('/api/state').then(function(r){{return r.json()}}).then(function(d){{
    monitoring=d.monitoring;
    var btn=document.getElementById('btn-monitor');
    btn.textContent=monitoring?'Stop Monitor':'Start Monitor';
    btn.className=monitoring?'btn-red':'btn-green';

    // Status bar
    var sb='Resolution: '+d.resolution+' | '+(d.calibrated?'Calibrated':'NOT calibrated');
    sb+=' | Monitor: '+(d.monitoring?'ON':'OFF');
    document.getElementById('status-bar').textContent=sb;

    // Zones
    var zh='';
    players.forEach(function(name){{
      var z=d.zones[name]||{{}};
      var st=z.state||'empty';
      var card=z.card||'';
      var bc={{'recognized':'#4caf50','processing':'#ff9800'}}[st]||'#444';
      zh+='<div class="zone" style="border-color:'+bc+'">'
        +'<img src="/zone/'+name+'?'+Date.now()+'" onerror="this.style.display=\\'none\\'">'
        +'<div class="zone-info"><div class="zone-name">'+name+'</div>'
        +'<div style="color:#888;font-size:.8em">'+st+'</div>'
        +'<div class="zone-card" style="color:'+bc+'">'+card+'</div></div></div>';
    }});
    document.getElementById('zones').innerHTML=zh;

    // Collect mode panel
    var cp=document.getElementById('collect-panel');
    var cbtn=document.getElementById('btn-collect');
    if(d.collect_mode){{
      cp.style.display='block';
      cbtn.textContent='Stop Collect';cbtn.className='btn-red';
      var cm=d.collect_mode;
      if(cm.done){{
        document.getElementById('collect-info').style.display='none';
        document.getElementById('collect-done').style.display='';
        document.getElementById('collect-scan-btn').style.display='none';
        document.getElementById('collect-next-btn').style.display='none';
      }} else {{
        document.getElementById('collect-info').style.display='';
        document.getElementById('collect-done').style.display='none';
        document.getElementById('collect-rank').textContent=cm.rank||'?';
        document.getElementById('collect-pass').textContent=cm.pass_idx+1;
        document.getElementById('collect-pass-total').textContent=cm.pass_total;
        document.getElementById('collect-idx').textContent=cm.current+1;
        document.getElementById('collect-total').textContent=cm.total;
        // Show per-player assignments
        var ah='';
        Object.keys(cm.cards||{{}}).forEach(function(p){{
          ah+='<span style="display:inline-block;margin:3px;padding:4px 10px;background:#0f3460;border-radius:4px">'
            +p+': <b>'+cm.cards[p]+'</b></span>';
        }});
        document.getElementById('collect-assignments').innerHTML=ah;
        var pct=Math.round(cm.current/cm.total*100);
        document.getElementById('collect-progress').style.width=pct+'%';
        var goBtn=document.getElementById('collect-go-btn');
        var pauseBtn=document.getElementById('collect-pause-btn');
        var redoBtn=document.getElementById('collect-redo-btn');
        var cdSpan=document.getElementById('collect-countdown');
        var phLabel=document.getElementById('collect-phase-label');
        if(cm.phase=='dealing'||cm.phase=='clearing'){{
          goBtn.style.display='none';
          pauseBtn.style.display='';
          redoBtn.style.display=cm.phase=='clearing'?'':'none';
          cdSpan.style.display='';cdSpan.textContent=cm.countdown;
          phLabel.style.display='';
          phLabel.textContent=cm.phase=='dealing'?'DEAL':'CLEAR';
          phLabel.style.color=cm.phase=='dealing'?'#4caf50':'#ff9800';
        }} else if(cm.phase=='paused'||cm.phase=='paused_new_pass'){{
          goBtn.style.display='';goBtn.textContent=cm.phase=='paused_new_pass'?'Start New Pass':'Resume';
          goBtn.onclick=cm.phase=='paused_new_pass'?collectGo:collectResume;
          pauseBtn.style.display='none';
          redoBtn.style.display='';
          cdSpan.style.display='none';
          phLabel.style.display='';phLabel.textContent='PAUSED';phLabel.style.color='#888';
        }} else {{
          goBtn.style.display='';goBtn.textContent='Start';goBtn.onclick=collectGo;
          pauseBtn.style.display='none';
          redoBtn.style.display='none';
          cdSpan.style.display='none';
          phLabel.style.display='none';
        }}
      }}
    }} else {{
      cp.style.display='none';
      cbtn.textContent='Collect Data';cbtn.className='btn-blue';
    }}

    // Deal mode panel
    var dp=document.getElementById('deal-panel');
    var dbtn=document.getElementById('btn-deal');
    if(d.deal_mode){{
      dp.style.display='block';
      dbtn.textContent='Stop Dealing';dbtn.className='btn-red';
      var dm=d.deal_mode;
      document.getElementById('deal-game').textContent=dm.game||'—';
      // Show/hide game input vs dealing status
      var gi=document.getElementById('deal-game-input');
      var ds=document.getElementById('deal-status');
      if(dm.phase=='game_select'){{
        gi.style.display='';ds.style.display='none';
        // Populate dealer dropdown if empty
        var sel=document.getElementById('deal-dealer');
        if(sel && sel.options.length==0){{
          players.forEach(function(n){{
            var opt=document.createElement('option');opt.value=n;opt.textContent=n;
            sel.appendChild(opt);
          }});
          setDealer();  // set initial deal order
        }}
      }} else {{
        gi.style.display='none';ds.style.display='';
        document.getElementById('deal-active-player').textContent=dm.active_player||'—';
        document.getElementById('deal-round').textContent=(dm.round_idx||0)+1;
        document.getElementById('deal-total').textContent=dm.total_rounds||'?';
      }}
      if(dm.phase=='complete'){{
        ds.innerHTML='<p style="color:#4caf50;font-size:1.1em">All rounds dealt!</p>';
      }}
      var ch='';
      (dm.cards||[]).forEach(function(c){{
        ch+='<span style="display:inline-block;margin:3px;padding:4px 8px;background:#1b5e20;border-radius:4px">'
          +c.player+': '+c.card+'</span>';
      }});
      document.getElementById('deal-cards').innerHTML=ch;
    }} else {{
      dp.style.display='none';
      dbtn.textContent='Test Dealing';dbtn.className='btn-blue';
    }}

    // Test mode panel
    var tp=document.getElementById('test-panel');
    var tb=document.getElementById('test-buttons');
    if(d.test_mode){{
      tp.style.display='block';
      var tm=d.test_mode;
      if(!tm.zone){{
        document.getElementById('test-title').textContent='Test Complete!';
        document.getElementById('test-prompt').textContent='';
        tb.innerHTML='<button class="btn-blue" onclick="testStop()">Done</button>';
      }} else if(tm.waiting=='card'){{
        document.getElementById('test-title').textContent='Testing: '+tm.zone+' ('+
          (tm.zone_idx+1)+'/'+tm.total+')';
        document.getElementById('test-prompt').textContent='Place a face-up card in '+tm.zone+'\\'s zone...';
        tb.innerHTML='<button class="btn-orange" onclick="testSkip()">Skip</button> '+
          '<button class="btn-off" onclick="testStop()">Stop Test</button>';
      }} else if(tm.waiting=='confirm'){{
        document.getElementById('test-title').textContent='Testing: '+tm.zone;
        document.getElementById('test-prompt').textContent='Recognized: '+tm.result;
        tb.innerHTML='<button class="btn-green" onclick="testConfirm(true)">Correct</button> '+
          '<button class="btn-red" onclick="testConfirm(false)">Incorrect</button> '+
          '<button class="btn-orange" onclick="testSkip()">Skip</button> '+
          '<button class="btn-off" onclick="testStop()">Stop Test</button>';
      }}
    }} else {{
      tp.style.display='none';
    }}
  }}).catch(function(){{}});

}}

setInterval(update, 2000);
update();
</script></body></html>""")

    def _calibrate_page(self, s):
        players_js = json.dumps(PLAYER_NAMES)
        self._r(200,"text/html",f"""<!DOCTYPE html>
<html><head><title>Calibrate</title>
<style>
body{{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px;margin:0}}
canvas{{border:1px solid #444;cursor:crosshair;display:block;margin:10px 0;max-width:100%}}
#status{{font-size:1.2em;color:#4fc3f7;margin:10px 0}}
button{{padding:8px 16px;background:#e94560;color:#fff;border:none;border-radius:6px;cursor:pointer;margin:5px}}
</style></head><body>
<h1>Calibration</h1>
<div id="status">Loading image...</div>
<canvas id="canvas"></canvas>
<button onclick="location.href='/'">Back</button>
<script>
var c=document.getElementById('canvas'),ctx=c.getContext('2d');
var players={players_js};
var steps=[],step=0,cc=null,cr=null,zones=[],pc=null,imgW=0,imgH=0;
steps.push({{p:'Click CENTER of felt circle',t:'cc'}});
steps.push({{p:"Click EDGE of felt circle (at Bill's position)",t:'ce'}});
players.forEach(function(n){{
  steps.push({{p:'Click CENTER of '+n+"'s zone",t:'zc',n:n}});
  steps.push({{p:'Set '+n+"'s zone size, then click",t:'ze',n:n}});
}});
var img=new Image();
img.onload=function(){{
  imgW=img.naturalWidth;imgH=img.naturalHeight;
  var sc=Math.min((window.innerWidth-40)/imgW,1);
  c.width=Math.round(imgW*sc);c.height=Math.round(imgH*sc);c.dataset.s=sc;
  draw();document.getElementById('status').textContent=steps[0].p;
}};
img.src='/snapshot?'+Date.now();
function S(){{return parseFloat(c.dataset.s)||1}}
function draw(){{
  var s=S();ctx.drawImage(img,0,0,c.width,c.height);
  if(cc&&cr){{ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(cc[0]*s,cc[1]*s,cr*s,0,Math.PI*2);ctx.stroke()}}
  zones.forEach(function(z){{ctx.strokeStyle='#0f0';ctx.lineWidth=2;ctx.beginPath();ctx.arc(z.cx*s,z.cy*s,z.r*s,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='#0f0';ctx.font='16px sans-serif';ctx.fillText(z.name,z.cx*s-20,z.cy*s-z.r*s-8)}});
  if(pc){{ctx.strokeStyle='#ff0';ctx.lineWidth=2;ctx.setLineDash([5,5]);ctx.beginPath();ctx.arc(pc[0]*S(),pc[1]*S(),(pc[2]||0)*S(),0,Math.PI*2);ctx.stroke();ctx.setLineDash([])}}
}}
function xy(e){{var r=c.getBoundingClientRect();return[Math.round((e.clientX-r.left)*imgW/r.width),Math.round((e.clientY-r.top)*imgH/r.height)]}}
c.onclick=function(e){{
  var p=xy(e),x=p[0],y=p[1];if(step>=steps.length)return;var st=steps[step];
  if(st.t=='cc'){{cc=[x,y];step++}}
  else if(st.t=='ce'){{cr=Math.round(Math.sqrt(Math.pow(x-cc[0],2)+Math.pow(y-cc[1],2)));step++}}
  else if(st.t=='zc'){{pc=[x,y,0];step++}}
  else if(st.t=='ze'){{var r=Math.round(Math.sqrt(Math.pow(x-pc[0],2)+Math.pow(y-pc[1],2)));zones.push({{name:st.n,cx:pc[0],cy:pc[1],r:r}});pc=null;step++}}
  draw();
  if(step<steps.length)document.getElementById('status').textContent=steps[step].p;
  else{{document.getElementById('status').textContent='Saving...';
    fetch('/api/calibrate/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{circle_center:cc,circle_radius:cr,zones:zones}})}})
    .then(function(){{document.getElementById('status').textContent='Done! Return to main page.'}})}}
}};
c.onmousemove=function(e){{if(!pc)return;var p=xy(e);pc[2]=Math.round(Math.sqrt(Math.pow(p[0]-pc[0],2)+Math.pow(p[1]-pc[1],2)));draw()}};
</script></body></html>""")

    def _serve_logview(self, s):
        self._r(200, "text/html", """<!DOCTYPE html>
<html><head><title>Log</title>
<style>
body{font-family:monospace;background:#0d1117;color:#e0e0e0;padding:0;margin:0}
#toolbar{padding:6px 12px;background:#16213e;position:sticky;top:0}
button{padding:4px 12px;border:none;border-radius:4px;cursor:pointer;background:#333;color:#e0e0e0;font-size:0.85em}
button:hover{background:#555}
#status{font-size:0.8em;color:#888;margin-left:12px}
pre{white-space:pre-wrap;font-size:0.85em;line-height:1.4;padding:8px 12px;margin:0}
</style>
<script>
function refresh(){
  fetch('/api/log').then(function(r){return r.json()}).then(function(d){
    var pre=document.getElementById('log');
    var atBottom=pre.scrollHeight-pre.scrollTop-pre.clientHeight<50;
    pre.innerHTML=d.lines.join('\\n');
    if(atBottom) pre.scrollTop=pre.scrollHeight;
  }).catch(function(){});
  setTimeout(refresh,2000);
}
function saveLog(){
  fetch('/log').then(function(r){return r.text()}).then(function(t){
    var blob=new Blob([t],{type:'text/plain'});
    var a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='log.txt';
    a.click();
    URL.revokeObjectURL(a.href);
    var btn=document.getElementById('savebtn');
    btn.textContent='Saved!';
    setTimeout(function(){btn.textContent='Save Log'},2000);
  });
}
refresh();
</script></head><body>
<div id="toolbar">
  <button id="savebtn" onclick="saveLog()">Save Log</button>
  <span id="status">Auto-refreshing every 2s</span>
</div>
<pre id="log">Loading...</pre>
</body></html>""")

    def _zone_img(self, s, name):
        if not s.latest_frame is not None:
            return self._r(503,"text/plain","No frame")
        z = next((z for z in s.cal.zones if z["name"]==name), None)
        if not z: return self._r(404,"text/plain","Not found")
        crop = s.monitor._crop(s.latest_frame, z)
        if crop is None: return self._r(500,"text/plain","Crop failed")
        j = to_jpeg(crop, 90)
        if j: self._r(200,"image/jpeg",j)

    def _training_list(self, s):
        if not TRAINING_DIR.exists():
            return self._r(200,"text/html","<p>No data</p>")
        files = sorted(TRAINING_DIR.iterdir(), reverse=True)
        h = "<html><body style='font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px'><h1>Training Data</h1>"
        for f in files[:100]:
            if f.suffix == ".jpg":
                lbl = f.with_suffix(".txt").read_text() if f.with_suffix(".txt").exists() else ""
                h += f'<div style="display:inline-block;margin:8px;text-align:center"><img src="/training/{f.name}" width="150"><br><small>{f.stem[:30]}</small><br>{lbl}</div>'
        self._r(200,"text/html",h+"</body></html>")

    def _training_file(self, name):
        p = TRAINING_DIR / name
        if not p.exists(): return self._r(404,"text/plain","Not found")
        self._r(200, "image/jpeg" if p.suffix==".jpg" else "text/plain",
                p.read_bytes() if p.suffix==".jpg" else p.read_text())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _state
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    capture = FrameCapture(args.camera, args.resolution)
    log.log(f"Camera {args.camera}, resolution {capture.resolution}")

    print("  Testing capture...")
    frame = capture.capture()
    if frame is None:
        sys.exit("  ERROR: Could not capture. Check camera and ffmpeg.")
    print(f"  OK: {frame.shape[1]}x{frame.shape[0]}")

    cal = Calibration()
    cal.load()

    monitor = ZoneMonitor(threshold=args.threshold)
    _state = AppState(capture, cal, monitor)
    _state.latest_frame = frame

    # Start server
    server = http.server.HTTPServer(("0.0.0.0", 8888), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    log.log("Server at http://localhost:8888")

    # Start background capture
    Thread(target=bg_loop, daemon=True).start()

    # Open browser
    time.sleep(1)
    subprocess.Popen(["open", "http://localhost:8888"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if cal.ok:
        print(f"  Calibration: {len(cal.zones)} zones")
    else:
        print("  No calibration — use browser to calibrate")

    print("  All UI is in the browser. Press Ctrl+C to quit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        _state.quit_flag = True

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Overhead Camera Card Recognition Test Harness

Single-page browser UI at http://localhost:8888
Terminal is only used for startup — all interaction in the browser.

Usage:
    python overhead_test.py [--camera 0] [--threshold 30.0] [--resolution auto]
"""

import sys as _sys
# When this file is run as `python overhead_test.py` it loads as the
# module `__main__`, not `overhead_test`. Our own http_server.py does
# `import overhead_test as ot` at module load to reach game-flow
# helpers that still live here — without this alias, that triggers a
# SECOND execution of this file under the name `overhead_test`, which
# then re-hits `from http_server import Handler` mid-load and crashes
# with "cannot import name 'Handler' from partially initialized
# module 'http_server'". Aliasing the main module under its filename
# name makes the nested import resolve to the same (partial) module
# we are already executing, so nothing re-runs.
if __name__ == "__main__":
    _sys.modules.setdefault("overhead_test", _sys.modules[__name__])

import argparse
import base64
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock, Timer
from queue import Queue, Empty

import cv2
import http.server
import numpy as np

from game_engine import GameEngine
from log_buffer import log, LOG_DIR, LOG_FILE, LOG_ARCHIVE_DIR
from speech import speech, _resolve_best_voice
from brio_capture import FrameCapture, CAPTURE_FILE
from zone_monitor import ZoneMonitor, TRAINING_DIR, CONFIG_FILE, CLAUDE_MODEL, YOLO_MODEL_PATH
from pi_scanner import (
    HOST_CONFIG_PATH,
    _load_host_config,
    _save_host_config,
    _pi_ping,
    _pi_fetch_slots,
    _pi_flash,
    _pi_slot_led,
    _pi_slot_scan,
)
from ui_templates import (
    TABLE_HTML, LOGVIEW_HTML, CONSOLE_HTML,
    SCANNER_TMPL, CALIBRATE_TMPL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAYER_NAMES = ["Steve", "Bill", "David", "Joe", "Rodney"]
NUM_ZONES = len(PLAYER_NAMES)

DEFAULT_CAMERA_INDEX = 1  # Fallback if we can't find the Brio by name
DEFAULT_CAMERA_NAME = "BRIO"  # avfoundation device name substring to prefer
DEFAULT_THRESHOLD = 30.0
DEFAULT_RESOLUTION = "auto"

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"


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


def _stats_bump(state, key, delta=1):
    """Increment a key in state.stats if state exists. Zone monitor uses
    this to tally YOLO vs Claude recognitions without needing a hard
    dependency on AppState being initialized yet (first-run safety)."""
    if state is None or not hasattr(state, "stats"):
        return
    state.stats[key] = state.stats.get(key, 0) + delta


def _recapture_baselines(s):
    """Capture zone baselines AND reset any watching-phase bookkeeping so
    the deal-order gate starts clean for the next round."""
    if s.cal.ok and s.latest_frame is not None:
        s.monitor.capture_baselines(s.latest_frame)
    if hasattr(s, "_zones_with_motion"):
        s._zones_with_motion = set()

# ---------------------------------------------------------------------------
# Console scan trigger — watches dealer's zone, scans all zones when dealer dealt
# ---------------------------------------------------------------------------

def _brio_player_names(s):
    """Active players whose Brio zones the overhead watcher should scan.

    For both local and remote players: the dealer places face-up cards
    in the players Brio zone for all players to see (Rodneys flipped-
    up card in 7/27, his up cards in stud games, etc.). So every active
    player contributes a Brio zone.
    """
    return set(s.console_active_players)


def _console_watch_dealer(s, frame):
    """Watch the dealer's zone. When a card appears there, wait for settle then
    scan all active player zones in one batch. Dealer deals to themselves last,
    so this guarantees all cards are placed before scanning.

    After scan, if any active players have no recognized card, watch their zones
    and rescan when they move their cards."""
    phase = s.console_scan_phase

    if phase in ("idle", "confirmed"):
        return

    ge = s.game_engine
    dealer_name = ge.get_dealer().name
    brio_names = _brio_player_names(s)

    # Handle missing-card watching: any active player with empty card
    if phase == "watching_missing":
        missing_zones = []
        for z in s.cal.zones:
            name = z["name"]
            if name not in brio_names:
                continue
            if s.monitor.zone_state.get(name) == "corrected":
                continue
            card = s.monitor.last_card.get(name, "")
            if card and card != "No card":
                continue
            missing_zones.append(z)
        # If any missing zone now has a card, trigger rescan of all missing
        retry_crops = {}
        for z in missing_zones:
            crop = s.monitor.check_single(frame, z)
            if crop is not None:
                retry_crops[z["name"]] = crop.copy()
                s.monitor.pending[z["name"]] = True
        # Watchdog: if no movement was detected for >10s, force a rescan
        # anyway. The dealer may be adjusting a card that barely moves the
        # pixel diff (thin edge inside zone). Keeps us from getting stuck
        # prompting "adjust your card" with nothing happening.
        if not retry_crops and missing_zones:
            last = getattr(s, "_missing_prompt_time", 0.0)
            if last and time.time() - last >= 10.0:
                for z in missing_zones:
                    crop = s.monitor._crop(frame, z)
                    if crop is not None:
                        retry_crops[z["name"]] = crop.copy()
                        s.monitor.pending[z["name"]] = True
                if retry_crops:
                    log.log(f"[CONSOLE] Watchdog rescan of missing zones: {', '.join(retry_crops.keys())}")
                    s._missing_prompt_time = time.time()
        if retry_crops:
            log.log(f"[CONSOLE] Movement detected in missing zones: {', '.join(retry_crops.keys())}")
            s.console_scan_phase = "scanned"  # will transition to watching_missing again if still missing
            Thread(target=_console_rescan_missing, args=(s, retry_crops), daemon=True).start()
        return

    if phase == "scanned":
        # Wait until pending scans are done before deciding anything.
        if any(s.monitor.pending.get(n) for n in brio_names):
            return
        # Ask the per-game class which zones to evaluate and whether
        # empty zones are legitimate (standing). stand_allowed=True
        # subsumes the old "is this a 7/27 hit round?" check: the same
        # trigger + missing-handling logic falls out of one flag.
        impl = s.current_game_impl
        if impl is not None:
            scan_names, stand_allowed = impl.zones_to_scan(s)
        else:
            scan_names, stand_allowed = list(brio_names), False
        watched = set(scan_names) & set(brio_names)
        # Dealer deals to themselves last, so the motion trigger should
        # have coincided with an actual card in the dealer's own zone.
        # If the dealer zone is still empty post-scan, the trigger was a
        # hand/arm sweep — revert to watching instead of nagging every
        # player who is also missing. Skip this check when standing is
        # allowed because the trigger could have come from any zone in
        # watched, not specifically the dealer's.
        dealer_card = s.monitor.last_card.get(dealer_name, "")
        dealer_empty = not dealer_card or dealer_card == "No card"
        if dealer_empty and not stand_allowed:
            log.log(
                f"[CONSOLE] {dealer_name}'s zone empty after scan — "
                f"likely a false trigger (arm over zone). Resuming watch."
            )
            # Only reset zones that did NOT land a card on this scan.
            # Cards already recognized or corrected keep their state so
            # the next motion trigger does not re-scan them from scratch.
            for nm in brio_names:
                state = s.monitor.zone_state.get(nm)
                if state in ("recognized", "corrected"):
                    continue
                s.monitor.zone_state[nm] = "empty"
                s.monitor.last_card[nm] = ""
            # Clear the deal-order gate too; the arm sweep tagged zones
            # that dont actually have cards yet.
            s._zones_with_motion = set()
            s.console_scan_phase = "watching"
            return
        missing = []
        if not hasattr(s, "_empty_scan_count"):
            s._empty_scan_count = {}
        MAX_EMPTY_SCANS_SCANNED = 3
        for name in watched:
            if s.monitor.zone_state.get(name) == "corrected":
                continue
            card = s.monitor.last_card.get(name, "")
            if not card or card == "No card":
                s._empty_scan_count[name] = s._empty_scan_count.get(name, 0) + 1
                # Stop treating a zone as "missing" once it has come back
                # empty enough times — otherwise a standing player (e.g.
                # Rodney passing in a 7/27 hit round) keeps re-triggering
                # "still waiting on…" every time another zone sees motion.
                # User can still correct the zone if the player actually
                # did place a card we never saw.
                if s._empty_scan_count[name] >= MAX_EMPTY_SCANS_SCANNED:
                    continue
                missing.append(name)
        # Standing allowed: dealer deals one at a time so a missing
        # zone might just mean "dealt to someone else first, we'll see
        # this player next". Always return to watching so late-
        # arriving cards keep triggering scans; the empty-scan cap
        # merely hides "still waiting on X" spam for genuinely-
        # standing players (they stay out of the missing list).
        if stand_allowed:
            if missing:
                log.log(
                    f"[CONSOLE] Partial scan: still waiting on "
                    f"{', '.join(missing)} — resuming watch"
                )
            s.console_scan_phase = "watching"
            return
        if missing:
            # Per-player "please adjust" speech, capped at 2 prompts per
            # round so we stop nagging when YOLO and Claude simply can't
            # see a card there. 10s cooldown stays — no back-to-back
            # announcements even for a new set of names.
            now = time.time()
            last_speech = getattr(s, "_missing_speech_time", 0.0)
            if not hasattr(s, "_missing_speech_count"):
                s._missing_speech_count = {}
            if now - last_speech >= 10.0:
                to_say = [n for n in missing
                          if s._missing_speech_count.get(n, 0) < 2]
                if to_say:
                    names = " and ".join(to_say)
                    log.log(f"[CONSOLE] Missing cards: {names} — prompting to adjust")
                    speech.say(f"{names}, please adjust your card")
                    s._missing_speech_time = now
                    for n in to_say:
                        s._missing_speech_count[n] = s._missing_speech_count.get(n, 0) + 1
                else:
                    log.log(
                        "[CONSOLE] Missing cards still unresolved — "
                        "2-announcement cap reached, waiting for manual entry"
                    )
            s.console_scan_phase = "watching_missing"
            s._missing_prompt_time = time.time()
        return

    if dealer_name not in brio_names:
        # Dealer is remote (Rodney) — nobody is placing cards in the
        # dealer zone so the regular trigger can't fire. Fall back to
        # any local brio zone as the motion trigger.
        alt = next((z for z in s.cal.zones if z["name"] in brio_names), None)
        if alt is None:
            return
        dealer_zone = alt
    else:
        dealer_zone = next((z for z in s.cal.zones if z["name"] == dealer_name), None)
    if dealer_zone is None:
        return

    if phase == "watching":
        # stand_allowed rounds (e.g. 7/27 hit rounds): the dealer
        # hands to one player at a time, so any zone in the scan set
        # can be the first to change. Frozen / out-of-game players
        # are omitted from zones_to_scan so their motion is ignored.
        impl = s.current_game_impl
        if impl is not None:
            scan_names, stand_allowed = impl.zones_to_scan(s)
        else:
            scan_names, stand_allowed = list(brio_names), False
        watched = set(scan_names) & set(brio_names)
        if stand_allowed:
            trigger_zone = None
            for z in s.cal.zones:
                if z["name"] not in watched:
                    continue
                if s.monitor.check_single(frame, z) is not None:
                    trigger_zone = z
                    break
            if trigger_zone is not None:
                log.log(f"[CONSOLE] Hit-round card detected in {trigger_zone['name']}'s zone — {s.brio_settle_s:.1f}s settle")
                s.console_scan_phase = "settling"
                s.console_settle_time = time.time()
            return
        # "Wait for all zones" gate: hold off on settling until every
        # watched zone currently shows motion above its baseline
        # threshold (i.e., a physical card is present in each zone RIGHT
        # NOW, not just "has shown motion at some point this round"). An
        # arm sweeping through a zone produces transient motion that the
        # old accumulate-over-time gate mistook for a card landing —
        # this version requires the diff to be above threshold at the
        # moment of the decision, so transient arm-crossings don't fire
        # scans prematurely. Used for 7 Card Stud initial deal AND 7/27
        # round-1 flip-up, where stand_allowed=False.
        if not hasattr(s, "_zones_with_motion"):
            s._zones_with_motion = set()
        all_present = True
        for z in s.cal.zones:
            name = z["name"]
            if name not in brio_names:
                continue
            if s.monitor.zone_state.get(name) in ("recognized", "corrected"):
                s._zones_with_motion.add(name)
                continue
            if s.monitor.check_single(frame, z) is not None:
                if name not in s._zones_with_motion:
                    s._zones_with_motion.add(name)
                    log.log(
                        f"[CONSOLE] Card present in {name}'s zone "
                        f"({len(s._zones_with_motion)}/{len(brio_names)})"
                    )
            else:
                all_present = False
                s._zones_with_motion.discard(name)
        if all_present and brio_names:
            log.log(
                f"[CONSOLE] All {len(brio_names)} zones have cards — "
                f"{s.brio_settle_s:.1f}s settle"
            )
            s.console_scan_phase = "settling"
            s.console_settle_time = time.time()
            return
        # Heartbeat diagnostic: once every ~10s while we're stuck in
        # watching, log the per-zone diff from baseline for every active
        # zone so the user can tell whether Brio is seeing changes below
        # the threshold vs. not seeing changes at all (zones miscalibrated).
        now = time.time()
        if now - getattr(s, "_watch_diag_time", 0.0) >= 10.0:
            s._watch_diag_time = now
            diffs = []
            for z in s.cal.zones:
                if z["name"] not in brio_names:
                    continue
                bl = s.monitor.baselines.get(z["name"])
                cur = s.monitor._crop(frame, z)
                if bl is None or cur is None or cur.shape != bl.shape:
                    diffs.append(f"{z['name']}=?")
                    continue
                d = float(np.mean(cv2.absdiff(cur, bl)))
                diffs.append(f"{z['name']}={d:.1f}")
            log.log(
                f"[CONSOLE] watching {dealer_name}'s zone — "
                f"diffs vs baseline: {', '.join(diffs)} "
                f"(threshold {s.monitor.threshold:.0f})"
            )
        return

    if phase == "settling":
        if time.time() - s.console_settle_time < s.brio_settle_s:
            return
        # Trust the initial motion trigger — don't re-verify. The old 2s
        # re-check rejected real cards when auto-exposure drifted the
        # per-pixel diff below threshold. YOLO already filters hand-only
        # pass-overs by returning "No card" for the whole zone.
        log.log("[CONSOLE] Scanning all active zones")
        zone_crops = {}
        # Per-round empty-scan counter. If a zones scan keeps coming back
        # as "No card" (YOLO below threshold + Claude sees nothing) we
        # eventually stop including it in the batch so YOLO cant finally
        # hallucinate a random card on its Nth retry. Reset on Confirm
        # Cards and on next round.
        if not hasattr(s, "_empty_scan_count"):
            s._empty_scan_count = {}
        MAX_EMPTY_SCANS = 3
        for z in s.cal.zones:
            name = z["name"]
            if name not in brio_names:
                continue
            # Lock already-identified cards: once a zone has a recognized
            # (or user-corrected) card for this round, skip it. Only zones
            # that are still empty get re-scanned when the dealer triggers
            # another motion event in the same round.
            if s.monitor.zone_state.get(name) in ("recognized", "corrected"):
                continue
            # No hard cap-based skip here anymore: a new settling event
            # means the watcher saw something new worth scanning, and
            # force_claude (below) routes previously-empty zones
            # through Claude so a late YOLO hallucination can't slip
            # through. The MAX_EMPTY_SCANS cap still lives in the
            # scanned phase where it suppresses "still waiting on X"
            # log spam for genuinely-standing players.
            crop = s.monitor._crop(frame, z)
            if crop is None or crop.size == 0:
                continue
            zone_crops[name] = crop.copy()
            s.monitor.pending[name] = True
        # Zones that came back empty on a prior scan this round can't
        # trust YOLO for their next recognition — it has a strong habit
        # of hallucinating "2 of Spades" on an empty felt crop at 50-55%
        # confidence, which is just above the auto-accept bar. Send
        # those zones through Claude to sanity-check YOLO's pick.
        force_claude = {
            name for name in zone_crops
            if s._empty_scan_count.get(name, 0) > 0
        }
        s.console_scan_phase = "scanned"
        Thread(
            target=s.monitor._recognize_batch,
            args=(zone_crops, force_claude),
            daemon=True,
        ).start()


def _console_rescan_missing(s, zone_crops):
    """Rescan just the zones where cards moved. Reuses the batch pipeline."""
    s.monitor._recognize_batch(zone_crops)


# ---------------------------------------------------------------------------
# Follow the Queen tracking for overhead camera
# ---------------------------------------------------------------------------

_ALL_CARDS = [
    f"{r_long} of {su.capitalize()}"
    for r_long in ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                   "Jack", "Queen", "King"]
    for su in ["clubs", "diamonds", "hearts", "spades"]
]


def _dedup_round_cards_against_seen(s, round_cards):
    """If a recognized up card duplicates a card we've already seen in this
    hand (prior up rounds, Rodney's scanned down cards, or another player's
    card in the same round), swap it for a random card that isn't already
    in play. Operates in place on round_cards.

    The scanner can't distinguish two identical-looking cards, and Claude
    will sometimes echo back whatever it saw last. Rodney's down cards are
    invisible to the overhead camera, so a duplicate against one of them
    is a near-certain misread.
    """
    seen = set()
    # Prior confirmed up cards
    for c in s.console_hand_cards:
        seen.add(c["card"])
    # Rodney's known down cards (verified + pending low-conf guesses).
    # Skip his flipped slot — the flipped card IS this round's up card
    # for Rodney, so including it in seen would flag his own scan as a
    # self-collision and trigger a random substitution.
    suit_full = {"c": "Clubs", "d": "Diamonds", "h": "Hearts", "s": "Spades"}
    rank_full = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}
    def _canonical(rank, suit):
        return f"{rank_full.get(rank, rank)} of {suit.capitalize()}"
    flipped_slot = (s.rodney_flipped_up or {}).get("slot")
    for slot_num, d in s.rodney_downs.items():
        if slot_num == flipped_slot:
            continue
        seen.add(_canonical(d["rank"], d["suit"]))
    for d in s.slot_pending.values():
        seen.add(_canonical(d["rank"], d["suit"]))

    for entry in round_cards:
        card = entry.get("card", "")
        player = entry.get("player", "")
        if card in seen:
            # If the user has explicitly corrected this zone, trust the
            # correction and keep the card as-is — the collision is
            # almost always a misrecognized down card in seen, not the
            # users value. Otherwise log the collision but still keep
            # the scan; randomly substituting a card corrupts wild
            # tracking (fake Queens) and hand evaluation.
            if s.monitor.zone_state.get(player) == "corrected":
                log.log(
                    f"[CONFIRM] {player}: {card} collides with seen card "
                    f"but was user-corrected — keeping as-is"
                )
            else:
                log.log(
                    f"[CONFIRM] {player}: {card} collides with seen card "
                    f"— leaving as-is (dealer can correct if wrong)"
                )
        seen.add(card)


def _announce_poker_hand_bet_first(s):
    """Announce who bets first at a poker-hand game based on best visible
    hand. Skips 7/27 (its own announcer), Challenge games, and all-down
    games (5CD, 3 Toed Pete) where nobody has an up card to compare."""
    ge = s.game_engine
    if not ge.current_game:
        return
    if ge.current_game.name.startswith("7/27"):
        return
    if any(ph.type.value == "challenge" for ph in ge.current_game.phases):
        return
    has_up_deal = any(
        ph.type.value in ("deal", "community") and "up" in ph.pattern
        for ph in ge.current_game.phases
    )
    if not has_up_deal:
        return
    try:
        from poker_hands import best_hand, HandResult, RANK_VALUE, RANK_NAME, VALUE_RANK
    except Exception as e:
        log.log(f"[POKER] best_hand unavailable: {e}")
        return

    RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
    per_player_cards = {}
    for entry in s.console_hand_cards:
        parts = entry.get("card", "").split(" of ")
        if len(parts) != 2:
            continue
        rank_full, suit_full = parts[0].strip(), parts[1].strip().lower()
        rank = RANK_SHORT.get(rank_full, rank_full)
        per_player_cards.setdefault(entry["player"], []).append((rank, suit_full))

    wild_ranks = list(getattr(ge, "wild_ranks", []) or [])
    best_player = None
    best_result = None
    for name in s.console_active_players:
        if name in s.folded_players:
            continue
        cards = per_player_cards.get(name, [])
        if not cards:
            continue
        try:
            if len(cards) == 1:
                # Single up card — treat as high-card only.
                rank, suit = cards[0]
                v = RANK_VALUE.get(rank, 0)
                result = HandResult(
                    "high_card",
                    f"{RANK_NAME.get(rank, rank)} high",
                    [v],
                    [],
                )
            else:
                result = best_hand(cards, wild_ranks=wild_ranks)
        except Exception as e:
            log.log(f"[POKER] eval {name} failed: {e}")
            continue
        log.log(f"[POKER] {name}: {result.label}")
        key = (result.rank, result.tiebreakers)
        if best_result is None or key > (best_result.rank, best_result.tiebreakers):
            best_result = result
            best_player = name

    if best_player is not None and best_result is not None:
        try:
            from poker_hands import RANK_PLURAL
        except Exception:
            RANK_PLURAL = {}
        cat = best_result.category
        tb = best_result.tiebreakers
        name_of = lambda v: RANK_NAME.get(VALUE_RANK.get(v, ""), "")
        plural_of = lambda v: RANK_PLURAL.get(VALUE_RANK.get(v, ""), name_of(v) + "s")
        if cat == "five_of_a_kind":
            hand_phrase = f"Five {plural_of(tb[0])}"
        elif cat == "four_of_a_kind":
            hand_phrase = f"Four {plural_of(tb[0])}"
        elif cat == "three_of_a_kind":
            hand_phrase = f"Three {plural_of(tb[0])}"
        elif cat == "full_house":
            hand_phrase = f"Full house, {plural_of(tb[0])} over {plural_of(tb[1])}"
        elif cat == "two_pair":
            hand_phrase = f"Two {plural_of(tb[0])} and two {plural_of(tb[1])}"
        elif cat == "pair":
            hand_phrase = f"Two {plural_of(tb[0])}"
        elif cat == "straight_flush" and tb[0] == 14:
            hand_phrase = "Royal flush"
        elif cat == "straight_flush":
            hand_phrase = f"{name_of(tb[0])} high straight flush"
        elif cat == "flush":
            hand_phrase = f"{name_of(tb[0])} high flush"
        elif cat == "straight":
            hand_phrase = f"{name_of(tb[0])} high straight"
        else:
            # high card: list every card the player actually shows, in
            # descending effective rank (wilds speak as Ace).
            wild_set = set(wild_ranks)
            values = []
            for rank, _suit in per_player_cards.get(best_player, []):
                if rank in wild_set:
                    values.append(14)
                else:
                    v = RANK_VALUE.get(rank)
                    if v:
                        values.append(v)
            values.sort(reverse=True)
            ranks_spoken = [name_of(v) for v in values if v]
            hand_phrase = ", ".join(ranks_spoken) if ranks_spoken else "no card"
        phrase = f"{best_player}, {hand_phrase} is high. Your bet."
        log.log(f"[POKER] Bet first: {phrase} ({best_result.label})")
        speech.say(phrase)


def _check_follow_the_queen_round(s, round_cards, announce=True):
    """Check cards for Follow the Queen wild at end of round.

    Args:
        round_cards: list of {"player": name, "card": "Rank of Suit"} in deal order
        announce: when False, update wild state silently. Used for the
            last up-card round of stud games — we defer the speech until
            after the trailing 7th (down) card has been dealt, so the
            final wild state is announced once, alongside the high-hand
            bet-first call. State updates still happen either way; only
            speech.say is gated.
    """
    ge = s.game_engine
    if not ge.current_game or ge.current_game.dynamic_wild != "follow_the_queen":
        return

    RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}

    for c in round_cards:
        parts = c["card"].split(" of ")
        if len(parts) != 2:
            continue
        rank = parts[0]
        rank_short = RANK_SHORT.get(rank, rank)

        if ge.last_up_was_queen:
            if rank_short == "Q":
                # Queen immediately after a Queen: ignore the earlier one
                # and keep watching. The second Queen's follower is what
                # becomes wild. (Avoids the "Queens and Queens are wild"
                # annunciation.)
                pass
            else:
                ge.wild_ranks = ["Q", rank_short]
                plural = f"{rank}'s" if rank.isdigit() else f"{rank}s"
                ge.wild_label = f"Queens and {plural} are wild"
                log.log(f"[WILD] {ge.wild_label}")
                if announce:
                    speech.say(f"Queens and {plural} are now wild")

        ge.last_up_was_queen = (rank_short == "Q")

    # Always announce current wild state at end of round if non-default
    if ge.wild_label and ge.wild_label != "Queens wild":
        log.log(f"[WILD] Current: {ge.wild_label}")


def _announce_trailing_done(s):
    """Speak the final wild state (if any) then the bet-first player.

    Called after the trailing down card (7th street for 7CS/FTQ) has
    been dealt to every active player, so the final betting round gets
    one consolidated announcement instead of one stale one after the
    6th-street confirm.
    """
    ge = s.game_engine
    if not ge or not ge.current_game:
        return
    # Re-state the current wild label if it's non-default. For FTQ we
    # suppressed mid-round speech on the last up round; this is where
    # the player hears the final wild mapping. ge.wild_label already
    # contains the "are wild" suffix (set as "Queens and {X}s are
    # wild") so speak it verbatim — the earlier bug appended a second
    # " are wild" producing "Queens and sevens are wild are wild".
    label = getattr(ge, "wild_label", "") or ""
    if label and label != "Queens wild":
        speech.say(label)
    _announce_poker_hand_bet_first(s)


def _recompute_follow_the_queen(s):
    """Replay FTQ queen-follower logic against console_hand_cards in round
    order and update ge.wild_ranks/wild_label. Used when a correction
    changes a card that may have been the follower of a Queen. Announces
    the new wild state if it differs from the current one."""
    ge = s.game_engine
    if not ge.current_game or ge.current_game.dynamic_wild != "follow_the_queen":
        return
    prior_label = ge.wild_label
    RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
    ge.wild_ranks = ["Q"]
    ge.wild_label = "Queens wild"
    ge.last_up_was_queen = False
    by_round = {}
    for e in s.console_hand_cards:
        by_round.setdefault(e.get("round", 0), []).append(e)
    for r in sorted(by_round.keys()):
        for c in by_round[r]:
            parts = c.get("card", "").split(" of ")
            if len(parts) != 2:
                continue
            rank = parts[0]
            rank_short = RANK_SHORT.get(rank, rank)
            if ge.last_up_was_queen and rank_short != "Q":
                ge.wild_ranks = ["Q", rank_short]
                plural = f"{rank}'s" if rank.isdigit() else f"{rank}s"
                ge.wild_label = f"Queens and {plural} are wild"
            ge.last_up_was_queen = (rank_short == "Q")
    if ge.wild_label != prior_label:
        log.log(f"[WILD] Recomputed after correction: {ge.wild_label}")
        if ge.wild_label == "Queens wild":
            speech.say("Correction: only queens are now wild")
        else:
            tail = ge.wild_label.replace("Queens and ", "").replace(" are wild", "")
            speech.say(f"Correction: queens and {tail} are now wild")


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
        # Console (dealer phone UI)
        self.game_engine = GameEngine()
        self.console_active_players = list(PLAYER_NAMES)  # who's playing tonight
        self.console_last_round_cards = []  # cards from last upcard scan
        self.console_hand_cards = []  # all confirmed up cards this hand: [{player, card, round}]
        self.console_up_round = 0     # current up-card round number
        self.console_total_up_rounds = 0  # total up-card rounds in this game
        self.console_scan_phase = "idle"  # "idle" | "watching" | "settling" | "scanned" | "confirmed"
        self.console_settle_time = 0.0
        # Name of the most recently-dealt game this session. Used by the
        # "Same game again" / "Let's run that back" voice command to
        # repeat the previous hand without having to say its name again.
        self.last_game_name = ""
        # ---- Remote-player table view ("/table") ----
        # state_version is bumped whenever anything the observer needs changes.
        # Rodney's down cards come from the Pi scanner; other players only
        # expose a down-count + up-cards on the observer view.
        self.table_state_version = 0
        # Rodney's down-card slots. Indexed by scanner slot number so a
        # fluctuating or re-scanned slot replaces its prior value instead of
        # appending a new entry. Each value is {rank, suit, confidence}.
        self.rodney_downs = {}         # slot_num -> {rank, suit, confidence} (verified / auto-accepted)
        # 7/27: when Rodney has 2 down cards the UI asks him to pick one to
        # flip face-up. Once chosen, the card moves here and the LED for
        # that slot blinks so the dealer knows which to physically lift.
        self.rodney_flipped_up = None   # None or {rank, suit, slot}
        self.slot_pending = {}         # slot_num -> {rank, suit, confidence} (latest low-conf guess, awaiting confirm)
        self.slot_empty = {}           # slot_num -> True when poller sees no card
        self.verify_queue = []         # FIFO of slot_nums that need manual verify after /api/console/confirm
        self.pending_verify = None     # None or {guess, slot, prompt}
        self.table_log = []            # [{ts, msg}]
        self.pi_base_url = os.environ.get("PI_BASE_URL", "http://pokerbuddy.local:8080")
        # Tunables loaded from ~/.cardgame_host.json if present. Setup modal
        # writes them back when the user saves, so defaults only matter on
        # first run. pi_presence_threshold is a cached mirror of the Pi's
        # own persisted value — pushed to the Pi on save.
        cfg = _load_host_config()
        self.brio_settle_s = float(cfg.get("brio_settle_s", DEFAULT_BRIO_SETTLE_S))
        self.pi_presence_threshold = float(cfg.get("pi_presence_threshold", 140.0))
        self.pi_polling = False
        self.pi_poll_thread = None
        self.pi_prev_slots = {}        # slot_num -> last-seen card code (e.g. "Ac")
        # Slot-by-slot guided dealing state. None = not guiding; otherwise
        # {expecting: int, num_slots: int}. Regular _pi_poll_loop skips its
        # work while this is set.
        self.guided_deal = None
        self.guided_deal_thread = None
        # "Poker night" flag — set by Start, cleared by Exit Poker. The
        # console UI gates the game dropdown + action controls on this.
        self.night_active = False
        # High-level console state machine surfaced to the UI.
        # "idle" | "dealing" | "betting" | "hand_over"
        self.console_state = "idle"
        # 5 Card Draw / draw-phase support: Rodney marks cards during
        # betting (a set of slot numbers). When he hits "Request cards",
        # those slots' LEDs light up and guided flow refills them. One
        # draw per hand. betting_round distinguishes pre-draw vs post-draw
        # for games with two betting rounds.
        # Per-hand recognition stats: how many cards YOLO and Claude each
        # produced, and of those how many the user corrected. Reset on
        # every /api/console/deal and logged on /api/console/end.
        # pi_auto: count of Pi guided-deal scans that landed ≥ GUIDED_GOOD_CONF
        #   and were auto-committed without the user seeing a verify modal.
        # pi_verify_right: user opened the verify modal on a low-conf guess
        #   and accepted the Pi's suggestion unchanged → Pi was right.
        # pi_verify_wrong: same modal, but user edited rank/suit → Pi was wrong.
        # Together these tell us whether GUIDED_GOOD_CONF is set too aggressive
        # or too conservative across a night of play.
        self.stats = {
            "yolo_right": 0, "yolo_wrong": 0,
            "claude_right": 0, "claude_wrong": 0,
            "pi_auto": 0,
            "pi_verify_right": 0, "pi_verify_wrong": 0,
        }
        self.rodney_marked_slots: set[int] = set()
        self.rodney_drew_this_hand = False
        # Count of completed draws this hand (3 Toed Pete has 3). Reset on
        # deal; incremented after each guided replace completes. Used to
        # index into the games list of DRAW phases for max-marks, and to
        # decide when to advance to hand_over instead of another draw.
        self.rodney_draws_done = 0
        self.console_betting_round = 0
        # Games with a trailing down card (7 Card Stud's 7th street, FTQ's
        # final down): after the last up round's Pot-is-right we run a second
        # guided session for that down slot. This flag, once set, means the
        # next Pot-is-right goes straight to hand_over instead of starting
        # trailing deal again.
        self.console_trailing_done = False
        self.table_lock = Lock()       # guards rodney_downs / pending_verify / table_log
        self.pi_confidence_threshold = 0.70  # >= this → auto-accept
        # The Pi's template matcher returns low-but-nonzero confidence for
        # every slot (including empty ones), so "empty" as a confidence
        # threshold doesn't work reliably. We trust the Pi's recognized
        # flag + any nonzero confidence as "something was seen" so the
        # weak-but-present scan still ends up in slot_pending and can be
        # manually verified.
        self.pi_empty_threshold = 0.0
        self._pi_last_logged = {}            # slot_num -> last logged code, throttle log spam
        self.pi_flash_held = False           # tracked so we don't spam hold/release
        self.folded_players = set()     # Rodney's view of who's folded this hand
        self.freezes = {}               # 7/27: player_name -> freezes in a row
        # True when Deal pinged the Pi and got no answer; stays set until the
        # next Deal so we skip hitting the Pi (flash/hold, /slots, LEDs, etc).
        self.pi_offline = False
        # Per-hand game class instance (subclass of games.BaseGame).
        # Created in /api/console/deal from the template's class_name and
        # cleared on end_hand. None when idle / between hands.
        self.current_game_impl = None

_state = None


# ---------------------------------------------------------------------------
# Observer table view ("/table") — shared with Rodney via Teams
# ---------------------------------------------------------------------------

_CARD_NAME_RE = re.compile(
    r"^\s*(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)\s*$",
    re.IGNORECASE,
)
_RANK_CANON = {"ACE": "A", "KING": "K", "QUEEN": "Q", "JACK": "J"}
_SUIT_LETTER = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}


def _parse_card_any(text):
    """Parse either 'King of Hearts' or 'Kh' / '10s' into {rank, suit} or None."""
    if not text:
        return None
    text = str(text).strip()
    m = _CARD_NAME_RE.match(text)
    if m:
        rank = m.group(1).upper()
        rank = _RANK_CANON.get(rank, rank)
        return {"rank": rank, "suit": m.group(2).lower()}
    m = re.match(r"^(10|[2-9JQKA])([hdcs])$", text, re.IGNORECASE)
    if m:
        return {"rank": m.group(1).upper(), "suit": _SUIT_LETTER[m.group(2).lower()]}
    return None


_SUIT_LETTER_CODE = {"clubs": "c", "diamonds": "d", "hearts": "h", "spades": "s"}


def _best_hand_for_cards(cards, ge):
    """Given a list of card dicts ({rank, suit, ...}), compute the best
    poker hand using the current game's wild ranks and return
    {"label": ..., "codes": [...]} for the /table UI. codes are short
    card codes ("Ah", "10s") in best-hand order so the client can reorder
    its card row. Returns None if fewer than 2 cards or evaluation fails.
    """
    tuples = []
    code_by_id = {}
    for i, c in enumerate(cards):
        rank = c.get("rank")
        suit = c.get("suit")
        if not rank or not suit:
            continue
        tuples.append((rank, suit))
        code_by_id[i] = f"{rank}{_SUIT_LETTER_CODE.get(suit, (suit or '?')[0])}"
    if len(tuples) < 2:
        return None
    try:
        from poker_hands import best_hand
    except Exception:
        return None
    try:
        wilds = list(getattr(ge, "wild_ranks", []) or [])
        result = best_hand(tuples, wild_ranks=wilds)
    except Exception:
        return None
    codes = []
    for bc in result.cards:
        suit_letter = _SUIT_LETTER_CODE.get(bc.suit, (bc.suit or "?")[0])
        codes.append(f"{bc.rank}{suit_letter}")
    return {"label": result.label, "codes": codes, "category": result.category}


def _build_table_state(s):
    """Produce the JSON doc that /table/state returns.

    Rodney sees his hand in full. Every other player is just a down-count
    plus Brio up-card scans. The log is the tail of table_log.
    """
    ge = s.game_engine
    current_game = ge.current_game.name if ge.current_game else None

    # Accumulate up-card history by player from console_hand_cards (populated
    # when the dealer confirms each up-card round). Fall back to the latest
    # zone scan for any player missing from history — useful between rounds
    # before the dealer has hit Confirm.
    up_by_player = {}
    for entry in s.console_hand_cards:
        name = entry.get("player")
        parsed = _parse_card_any(entry.get("card", ""))
        if name and parsed:
            up_by_player.setdefault(name, []).append(
                {"rank": parsed["rank"], "suit": parsed["suit"], "round": entry.get("round")}
            )

    players = []
    for p in ge.players:
        if p.name not in s.console_active_players:
            continue
        up_cards = list(up_by_player.get(p.name, []))
        if not up_cards and s.monitor:
            # Only show latest scan if we don't already have history for this player.
            latest_txt = s.monitor.last_card.get(p.name, "")
            latest_parsed = _parse_card_any(latest_txt)
            if latest_parsed:
                details = s.monitor.recognition_details.get(p.name, {})
                conf = details.get("yolo_conf")
                cur = {"rank": latest_parsed["rank"], "suit": latest_parsed["suit"]}
                if conf is not None:
                    cur["confidence"] = round(float(conf), 2)
                up_cards.append(cur)

        freezes_n = s.freezes.get(p.name, 0)
        entry = {
            "name": p.name,
            "position": p.position,
            "is_dealer": p.is_dealer,
            "is_remote": p.is_remote,
            "folded": p.name in s.folded_players,
            "freezes": freezes_n,
            "frozen": freezes_n >= 3,
        }
        if p.is_remote:
            # Rodney's hand = only down-card slots that have been recognized
            # and validated (rodney_downs). Tentative slot_pending guesses
            # are shown in the verify modal instead, not as cards in hand.
            # If Rodney flipped one of his downs face-up (7/27 2-down), the
            # card remains in rodney_downs (so Pi counting still works) but
            # we render it here as an up-card instead of a down.
            flipped_slot = (s.rodney_flipped_up or {}).get("slot")
            hand = []
            for slot_num in sorted(s.rodney_downs.keys()):
                if slot_num == flipped_slot:
                    continue
                d = s.rodney_downs[slot_num]
                hand.append({"type": "down", "rank": d["rank"],
                             "suit": d["suit"], "slot": slot_num,
                             "confidence": d.get("confidence")})
            if s.rodney_flipped_up:
                fu = s.rodney_flipped_up
                # Don't duplicate once Brio picks it up via a zone scan.
                already = any(
                    c.get("rank") == fu["rank"] and c.get("suit") == fu["suit"]
                    for c in up_cards
                )
                if not already:
                    hand.append({"type": "up", "rank": fu["rank"], "suit": fu["suit"]})
            for c in up_cards:
                hand.append({"type": "up", **c})
            entry["hand"] = hand
            entry["best_hand"] = _best_hand_for_cards(hand, ge)
        else:
            # Dealer deals the same card-type to every player in each round,
            # so every non-folded player holds as many downs as Rodney has
            # validated. In 7/27 (2-down) once Rodney has flipped, every
            # local player has also flipped one of their two — subtract the
            # flipped card from the visible down-count.
            down_count = len(s.rodney_downs)
            if s.rodney_flipped_up:
                down_count = max(0, down_count - 1)
            entry["down_count"] = down_count
            entry["up_cards"] = up_cards
            entry["best_hand"] = _best_hand_for_cards(up_cards, ge)
        players.append(entry)

    # Console flow doesn't advance game_engine.phase_index, so derive the
    # round counter from console_up_round (confirmed up rounds) + down cards
    # Rodney has actually received — including pending scans so a yet-to-be-
    # verified card still advances the counter.
    active_down_slots = set(s.rodney_downs.keys()) | set(s.slot_pending.keys())
    current_round = s.console_up_round + len(active_down_slots)
    total_rounds = _total_card_rounds(ge)
    # Open-ended games (e.g. 7/27) report total=0 so the UI drops "of N".
    if ge.current_game is not None:
        has_hit_round = any(
            ph.type.value == "hit_round" and ph.card_type == "up"
            for ph in ge.current_game.phases
        )
        if has_hit_round:
            total_rounds = 0

    doc = {
        "version": s.table_state_version,
        "viewer": next((p.name for p in ge.players if p.is_remote), "Rodney"),
        "game": {
            "name": current_game or "",
            "round": getattr(ge, "draw_round", 0),
            "wild_label": ge.wild_label or "",
            "wild_ranks": list(getattr(ge, "wild_ranks", []) or []),
            "current_round": current_round,
            "total_rounds": total_rounds,
            "state": getattr(ge.state, "value", str(ge.state)),
        },
        "dealer": ge.get_dealer().name,
        "current_player": None,
        "players": players,
        "log": list(s.table_log[-30:]),
        "pending_verify": s.pending_verify,
        "flip_choice": None,
        "guided_deal": (
            dict(s.guided_deal) if s.guided_deal is not None else None
        ),
        "draw": {
            # Multi-draw games (3 Toed Pete): rodney_draws_done counts how
            # many draws are behind us; we can mark and request again as
            # long as more DRAW phases remain and the current draw has not
            # been taken yet.
            "can_mark": (
                _game_has_draw_phase(ge)
                and s.rodney_draws_done < _total_draw_phases(ge)
                and not s.rodney_drew_this_hand
                and s.console_state in ("betting", "draw")
            ),
            "can_request": (
                s.console_state == "draw"
                and s.rodney_draws_done < _total_draw_phases(ge)
                and not s.rodney_drew_this_hand
            ),
            "max_marks": _max_draw_for_game(ge, s.rodney_draws_done),
            "marked_slots": sorted(s.rodney_marked_slots),
            "drew_this_hand": s.rodney_drew_this_hand,
            "draws_done": s.rodney_draws_done,
            "total_draws": _total_draw_phases(ge),
        },
    }
    # Per-game decorations: the game class adds its own fields to the
    # document (7/27 injects values_7_27 per player and the flip-choice
    # prompt; base class / stud / draw are no-ops).
    impl = s.current_game_impl
    if impl is not None:
        impl.decorate_table_players(players, s)
        impl.decorate_table_state(doc, s)
    return doc


def _table_state_bump(s):
    """Call when something observable changes so polling clients re-render."""
    s.table_state_version += 1


def _dealing_phase_types():
    """Tuple of phase type values that contribute a round per card."""
    from game_engine import PhaseType
    return (PhaseType.DEAL, PhaseType.COMMUNITY)


def _total_card_rounds(ge):
    """Total cards in the game's deal + community phases (per player).

    Follow the Queen and 7-Card Stud = 7; Texas Hold'em = 2 hole + 5 community = 7.
    Returns 0 when no game is active.
    """
    if ge.current_game is None:
        return 0
    allowed = _dealing_phase_types()
    total = 0
    for ph in ge.current_game.phases:
        if ph.type in allowed:
            total += len(ph.pattern)
    return total


def _cards_dealt_so_far(ge):
    """Zero-based count of cards already dealt in the current game."""
    if ge.current_game is None:
        return 0
    allowed = _dealing_phase_types()
    completed = 0
    for i, ph in enumerate(ge.current_game.phases):
        if i < ge.phase_index:
            if ph.type in allowed:
                completed += len(ph.pattern)
        elif i == ge.phase_index:
            if ph.type in allowed:
                completed += ge.card_in_phase
            break
    return completed


def _next_deal_position_type(s):
    """Returns 'down', 'up', or None for the next card about to be dealt.

    Walks the combined DEAL/COMMUNITY pattern of the current game, consuming
    one up-card-per-up-round-confirmed and one down-per-rodney_down, and
    returns the type of the next un-dealt position.
    """
    ge = s.game_engine
    if ge.current_game is None:
        return None
    pattern = []
    for ph in ge.current_game.phases:
        if ph.type in _dealing_phase_types():
            pattern.extend(ph.pattern)
    dealt_ups = s.console_up_round
    dealt_downs = len(s.rodney_downs)
    for pos in pattern:
        if pos == "up":
            if dealt_ups > 0:
                dealt_ups -= 1
                continue
            return "up"
        else:
            if dealt_downs > 0:
                dealt_downs -= 1
                continue
            return "down"
    return None


def _skip_inactive_dealer(s):
    """Rotate past any unchecked player so the dealer is always an active
    seat. Caps the loop at one full rotation to guarantee termination if
    every player ends up inactive."""
    ge = s.game_engine
    if not s.console_active_players:
        return
    for _ in range(len(ge.players)):
        if ge.get_dealer().name in s.console_active_players:
            return
        ge.advance_dealer()


def _total_downs_in_pattern(ge):
    """Total number of down cards in the current game's deal pattern.

    Each game has a fixed number of down cards per player regardless of
    where they appear in the deal order (FTQ = 3, 7-Card Stud = 3,
    Hold'em = 2, 5-Card Draw = 5). The scanner box only needs to monitor
    that many slots.
    """
    if ge.current_game is None:
        return 0
    allowed = _dealing_phase_types()
    n = 0
    for ph in ge.current_game.phases:
        if ph.type in allowed:
            n += sum(1 for t in ph.pattern if t == "down")
    return n


def _initial_down_count(ge):
    """Number of down cards in the FIRST deal phase's pattern.

    7-Card Stud's ['down','down','up'] → 2; Hold'em's ['down','down'] → 2;
    5 Card Draw's ['down']*5 → 5; Follow the Queen → 3. These are the slots
    the scanner box guides through at Deal-time. Any remaining downs
    (7CS/FTQ 7th street) are handled as a trailing guided session after
    the final up round.
    """
    if ge.current_game is None:
        return 0
    allowed = _dealing_phase_types()
    for ph in ge.current_game.phases:
        if ph.type in allowed:
            return sum(1 for t in ph.pattern if t == "down")
    return 0


def _trailing_down_slots(ge):
    """Slot numbers for down cards beyond the initial deal phase.

    For 7CS (3 total downs, 2 initial) → [3]. For FTQ similarly → [4].
    For 5CD / Hold'em → [] (no trailing downs). These slots are guided
    after the final up round's Pot-is-right.
    """
    total = _total_downs_in_pattern(ge)
    initial = _initial_down_count(ge)
    if total <= initial:
        return []
    return list(range(initial + 1, total + 1))


def _table_log_add(s, msg):
    s.table_log.append({"ts": int(time.time()), "msg": msg})
    if len(s.table_log) > 200:
        s.table_log = s.table_log[-100:]


def _parse_card_code(code):
    """Parse 'Ac' or '10h' into {rank, suit} or None."""
    if not code:
        return None
    m = re.match(r"^\s*(10|[2-9JQKA])([hdcs])\s*$", code, re.IGNORECASE)
    if not m:
        return None
    return {"rank": m.group(1).upper(), "suit": _SUIT_LETTER[m.group(2).lower()]}


_SIM_RANKS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
_SIM_SUITS = ["clubs","diamonds","hearts","spades"]


def _promote_next_verify(s) -> bool:
    """If no modal is open and a queued slot has a pending guess, open it.

    Returns True if state changed (pending_verify was set). Caller owns the
    table_lock.
    """
    if s.pending_verify is not None or not s.verify_queue:
        return False
    for slot_num in list(s.verify_queue):
        guess = s.slot_pending.get(slot_num)
        if not guess:
            continue
        s.pending_verify = {
            "slot": slot_num,
            "guess": dict(guess),
            "prompt": (
                f"Slot {slot_num} needs verification. "
                f"Remove the card, hold it up for Rodney, "
                f"then confirm or override."
            ),
            "image_url": (
                None if s.pi_offline
                else f"/api/table/slot_image/{slot_num}"
            ),
        }
        _table_log_add(s, f"Slot {slot_num}: modal opened for verify")
        return True
    return False


def _simulate_offline_slot_scans(s):
    """Fill rodney_downs' expected slots with random low-confidence guesses
    so a hand can be played end-to-end without the Pi. Each missing slot
    (not in rodney_downs, not in slot_pending) gets one random card at
    conf=0.20 — low enough to queue a verify modal on Confirm Cards where
    Rodney can override with the actual card.
    """
    max_slot = _total_downs_in_pattern(s.game_engine)
    if max_slot <= 0:
        return
    with s.table_lock:
        added = []
        for n in range(1, max_slot + 1):
            if n in s.rodney_downs or n in s.slot_pending:
                continue
            rank = random.choice(_SIM_RANKS)
            suit = random.choice(_SIM_SUITS)
            s.slot_pending[n] = {"rank": rank, "suit": suit, "confidence": 0.20}
            added.append((n, f"{rank}{suit[0]}"))
        if added:
            for (n, code) in added:
                _table_log_add(s, f"Slot {n}: simulated {code} (Pi offline, needs verify)")
            s.table_state_version += 1


def _update_flash_for_deal_state(s):
    """Hold LEDs while a down card is the next expected deal; release otherwise."""
    nxt = _next_deal_position_type(s)
    _pi_flash(s, nxt == "down")


def _pi_poll_loop(s):
    """Background poll: map Pi scanner detections into rodney_downs.

    - Each /slots result is compared against s.pi_prev_slots.
    - A new card (slot was empty or held a different card) becomes:
        * rodney_downs[slot] = card when confidence >= threshold, OR
        * a pending_verify prompt otherwise (poller stops advancing that
          slot until the verify modal is resolved).
    - Slots that go empty clear their rodney_downs entry too.
    """
    log.log("[PI] poll loop started")
    offline_streak = 0
    while s.pi_polling:
        # Guided dealing takes exclusive ownership of the scanner for
        # all-down games. The regular poller idles until guided completes.
        if s.guided_deal is not None:
            time.sleep(0.5)
            continue
        # Only hit the Pi when we're actually expecting a down card to be
        # dealt. Gate on the deal pattern directly (not pi_flash_held) so a
        # failed /flash/hold call doesn't also stop the scan polling.
        _update_flash_for_deal_state(s)
        if _next_deal_position_type(s) != "down":
            time.sleep(2.0)
            continue
        doc = _pi_fetch_slots(s)
        if doc is None:
            offline_streak += 1
            # After two failed fetches, assume the Pi isn't running and
            # simulate slot scans so gameplay is testable without hardware.
            # Each expected slot gets a random low-confidence guess the user
            # can override in the verify modal. Also promote any queued
            # verify into pending_verify so the modal actually opens in
            # offline mode.
            if offline_streak >= 2 or s.pi_offline:
                _simulate_offline_slot_scans(s)
                with s.table_lock:
                    if _promote_next_verify(s):
                        s.table_state_version += 1
            time.sleep(2.0)
            continue
        offline_streak = 0
        # Only scan slots the current game actually uses (FTQ=3, Hold'em=2).
        max_slot = _total_downs_in_pattern(s.game_engine)
        with s.table_lock:
            changed = False
            for entry in doc.get("slots", []):
                slot_num = entry.get("slot")
                if slot_num is None:
                    continue
                if slot_num > max_slot:
                    if slot_num in s.rodney_downs:
                        s.rodney_downs.pop(slot_num, None)
                        changed = True
                    s.pi_prev_slots.pop(slot_num, None)
                    s.slot_pending.pop(slot_num, None)
                    s.slot_empty[slot_num] = True
                    continue

                recognized = entry.get("recognized")
                rank = entry.get("rank")
                suit = entry.get("suit")
                conf = float(entry.get("confidence", 0.0))
                is_empty = (not recognized) or conf < s.pi_empty_threshold

                if is_empty:
                    # Slot physically empty: mark empty but keep slot_pending
                    # intact — the last-seen guess survives removal so the
                    # verify modal can still fire after a Confirm Cards.
                    # If there's a weak scan below threshold log it once so
                    # the user can see that the scanner IS seeing something
                    # but deciding it's noise.
                    if recognized and rank and suit and conf > 0:
                        weak_code = f"{rank}{suit[0]} ({int(conf*100)}%)"
                        if s._pi_last_logged.get(slot_num) != weak_code:
                            log.log(f"[PI] Slot {slot_num}: weak {weak_code} below empty threshold")
                            s._pi_last_logged[slot_num] = weak_code
                    elif s._pi_last_logged.get(slot_num) is not None:
                        s._pi_last_logged[slot_num] = None
                    s.slot_empty[slot_num] = True
                    s.pi_prev_slots.pop(slot_num, None)
                    continue

                # Non-empty: remember what's there now.
                s.slot_empty[slot_num] = False
                code = f"{rank}{suit[0]}" if rank and suit else ""

                if conf >= s.pi_confidence_threshold:
                    prev = s.pi_prev_slots.get(slot_num)
                    if prev == code:
                        continue
                    s.rodney_downs[slot_num] = {
                        "rank": rank, "suit": suit, "confidence": round(conf, 2),
                    }
                    s.slot_pending.pop(slot_num, None)
                    if slot_num in s.verify_queue:
                        s.verify_queue.remove(slot_num)
                    s.pi_prev_slots[slot_num] = code
                    _table_log_add(s, f"Slot {slot_num}: {code} (auto, {int(conf*100)}%)")
                    changed = True
                else:
                    # Medium confidence: hold as the latest guess but don't
                    # surface a modal until the dealer runs /api/console/confirm
                    # and the user subsequently removes the card from the slot.
                    guess = {"rank": rank, "suit": suit, "confidence": round(conf, 2)}
                    if s.slot_pending.get(slot_num) != guess:
                        s.slot_pending[slot_num] = guess
                        changed = True

            if _promote_next_verify(s):
                changed = True
            if changed:
                s.table_state_version += 1
        time.sleep(1.0)
    log.log("[PI] poll loop stopped")


def _pi_poll_start(s):
    if s.pi_polling:
        return
    s.pi_polling = True
    s.pi_prev_slots = {}
    t = Thread(target=_pi_poll_loop, args=(s,), daemon=True)
    s.pi_poll_thread = t
    t.start()


def _pi_poll_stop(s):
    s.pi_polling = False
    # Don't join — daemon thread will exit on its own


# ---------------------------------------------------------------------------
# Guided dealing for all-down games (5 Card Draw, 3 Toed Pete, etc.)
# ---------------------------------------------------------------------------

GUIDED_GOOD_CONF = 0.50   # at/above this, auto-accept the scan
GUIDED_POLL_S = 0.6       # interval between /slots/<n>/scan polls
# Require this many consecutive present=true scans before firing a verify
# modal. The first presence hit is often a finger or a half-inserted card;
# YOLO can't see it yet. A high-confidence scan short-circuits this wait.
GUIDED_STABLE_SCANS = 3
# After first detecting a card in the slot, wait this long before using any
# scan reading. Gives the dealer time to fully seat the card so YOLO isn't
# fighting motion blur on the first capture. Longer than strictly needed
# to give the dealer time to finish placing the card before YOLO reads —
# shorter values led to verify modals popping with partial-insertion reads.
GUIDED_SETTLE_S = 2.0

# Default seconds to wait after the overhead (Brio) camera trips a motion
# event in the dealer zone before firing the whole-table scan. Runtime-
# configurable via the Setup modal → persisted in the host config file.
DEFAULT_BRIO_SETTLE_S = 0.7


def _guided_deal_loop(s):
    """Slot-by-slot dealing for all-down games.

    Turns slot-1 LED on, polls /slots/1/scan until a card is present:
    if YOLO conf >= GUIDED_GOOD_CONF, record the card + advance; if lower,
    blink the LED and open the /table verify modal for Rodney to resolve.
    Strict 1→N order: never looks at slot N+1 until slot N is resolved.

    External code stops the loop by setting s.guided_deal = None.
    """
    gd = s.guided_deal
    if gd is None:
        return
    N = gd["num_slots"]
    log.log(f"[GUIDED] Started — {N} slots")
    # Initial LED state: slot 1 solid on, the rest off.
    for n in range(1, N + 1):
        _pi_slot_led(s, n, "on" if n == 1 else "off")

    # Per-slot debounce state: how many consecutive present=true scans
    # we've seen, and the best card guess from any of them. Reset on
    # either present=false (card/finger withdrawn) or after we commit.
    stable_count = 0
    best_card = None
    settled = False  # True after GUIDED_SETTLE_S has elapsed since first present

    while True:
        gd = s.guided_deal
        if gd is None:
            log.log("[GUIDED] Stopped externally")
            return
        expecting = gd["expecting"]
        if expecting > N:
            for n in range(1, N + 1):
                _pi_slot_led(s, n, "off")
            log.log(f"[GUIDED] Complete — {N} slots filled")
            with s.table_lock:
                s.guided_deal = None
                s.table_state_version += 1
            # Per-game post-guided transitions.
            ge = s.game_engine
            first_deal_phase = next(
                (ph for ph in (ge.current_game.phases if ge.current_game else [])
                 if ph.type.value in ("deal", "community")),
                None,
            )
            first_phase_has_up = bool(
                first_deal_phase and "up" in first_deal_phase.pattern
            )
            has_hit_round_game = bool(
                ge.current_game and any(
                    ph.type.value == "hit_round"
                    for ph in ge.current_game.phases
                )
            )
            if (s.console_state == "dealing"
                    and s.console_total_up_rounds == 0
                    and not first_phase_has_up
                    and not has_hit_round_game):
                # Truly all-down games (5CD, 3 Toed Pete): every card is in,
                # auto-advance to betting so next action is Pot is right.
                s.console_state = "betting"
                if s.console_betting_round == 0:
                    s.console_betting_round = 1
                log.log(
                    f"[CONSOLE] All-down deal complete → "
                    f"betting round {s.console_betting_round}"
                )
            elif s.console_state == "dealing" and has_hit_round_game:
                # 7/27 (either variant): the local players still need to
                # flip/reveal an up card onto the table for Brio to scan.
                # Keep state in "dealing" with Brio watching; user presses
                # Confirm Cards once every player's up card is in.
                s.monitoring = True
                s.console_scan_phase = "watching"
                s._zones_with_motion = set()
                dname = ge.get_dealer().name if ge.current_game else "dealer"
                log.log(
                    f"[CONSOLE] 7/27 guided downs done → Brio watching "
                    f"{dname}'s zone for flipped-up cards"
                )
            elif s.console_state == "replacing":
                # Draw-phase refill just completed — back to the current
                # betting round (round 1 still — dealer will Pot-is-right
                # to advance to round 2 once everyone has drawn).
                s.console_state = "betting"
                log.log("[CONSOLE] Draw replacement complete → betting")
            elif (s.console_state == "dealing"
                  and s.console_total_up_rounds > 0
                  and s.console_scan_phase == "idle"):
                # Mixed game (7CS, Hold'em, FTQ): initial downs are in,
                # now hand off to Brio to watch for the up card(s). The
                # baselines captured at Deal-time are still valid — the
                # table had no up cards then and still has none now (local
                # down cards are kept in hand, not placed in up-zones).
                ge = s.game_engine
                s.monitoring = True
                s.console_scan_phase = "watching"
                s._zones_with_motion = set()
                dname = ge.get_dealer().name if ge.current_game else "dealer"
                log.log(f"[CONSOLE] Guided downs done → Brio watching {dname}'s zone")
            return

        # Was this slot's verify modal just resolved? Advance if so.
        with s.table_lock:
            already_filled = expecting in s.rodney_downs
            pv = s.pending_verify
            waiting_verify = pv is not None and pv.get("slot") == expecting

        if already_filled:
            _pi_slot_led(s, expecting, "off")
            with s.table_lock:
                gd["expecting"] = expecting + 1
                s.table_state_version += 1
            if expecting + 1 <= N:
                _pi_slot_led(s, expecting + 1, "on")
            stable_count = 0
            best_card = None
            settled = False
            continue

        if waiting_verify:
            time.sleep(0.3)
            continue

        result = _pi_slot_scan(s, expecting)
        if result is None:
            time.sleep(1.5)  # Pi unreachable — back off before retrying
            continue

        if not result.get("present"):
            stable_count = 0
            best_card = None
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        # First scan that sees "present" — the card may still be sliding
        # into place. Wait GUIDED_SETTLE_S before trusting any reading so
        # YOLO isn't hitting motion blur or a half-inserted card.
        if not settled:
            time.sleep(GUIDED_SETTLE_S)
            settled = True
            continue

        stable_count += 1
        card = result.get("card")
        if card:
            conf = float(card.get("confidence", 0.0))
            # Track the best (highest-conf) guess across debounce scans.
            if best_card is None or conf > float(best_card.get("confidence", 0.0)):
                best_card = card
            code = f"{card['rank']}{card['suit'][0]}"
            # Short-circuit: any scan above the auto-accept threshold commits
            # immediately, no need to wait for more stability.
            if conf >= GUIDED_GOOD_CONF:
                with s.table_lock:
                    s.rodney_downs[expecting] = {
                        "rank": card["rank"],
                        "suit": card["suit"],
                        "confidence": round(conf, 2),
                    }
                    s.pi_prev_slots[expecting] = code
                    gd["expecting"] = expecting + 1
                    s.table_state_version += 1
                _stats_bump(s, "pi_auto")
                _table_log_add(s, f"Slot {expecting}: {code} (auto, {int(conf*100)}%)")
                _pi_slot_led(s, expecting, "off")
                if expecting + 1 <= N:
                    _pi_slot_led(s, expecting + 1, "on")
                stable_count = 0
                best_card = None
                settled = False
                continue

        # Low-conf or no-card: give the card time to settle before popping
        # the verify modal. Early "present" ticks from a finger or a half-
        # inserted card would otherwise fire the modal with empty fields.
        if stable_count < GUIDED_STABLE_SCANS:
            time.sleep(GUIDED_POLL_S)
            continue

        # Debounce window elapsed without a high-confidence read. If YOLO
        # never recognized anything at all across the whole window, the
        # scanner is probably misreporting present=true for an empty slot
        # (e.g., brightness threshold too high). Don't open a modal with
        # an empty guess — log once, reset state, and keep polling.
        if best_card is None:
            log.log(
                f"[GUIDED] Slot {expecting}: present but nothing recognized "
                f"after {GUIDED_STABLE_SCANS} scans — Pi presence threshold "
                f"may be too high; continuing to poll"
            )
            stable_count = 0
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        conf = float(best_card.get("confidence", 0.0))
        guess = {
            "rank": best_card["rank"],
            "suit": best_card["suit"],
            "confidence": round(conf, 2),
        }
        prompt = (
            f"Slot {expecting}: low confidence ({int(conf*100)}%). "
            f"Confirm or correct."
        )

        with s.table_lock:
            s.pending_verify = {
                "slot": expecting,
                "guess": guess,
                "prompt": prompt,
                "image_url": f"/api/table/slot_image/{expecting}",
            }
            if guess["rank"]:
                s.slot_pending[expecting] = dict(guess)
            s.table_state_version += 1
        _table_log_add(
            s,
            f"Slot {expecting}: verify needed"
            + (f" ({int(guess['confidence']*100)}%)" if guess["rank"] else ""),
        )
        _pi_slot_led(s, expecting, "blink")


def _start_guided_deal(s, num_slots: int):
    """Kick off the guided deal thread. Safe to call again; becomes no-op
    if a guided deal is already running."""
    if s.guided_deal is not None:
        return
    with s.table_lock:
        s.guided_deal = {"expecting": 1, "num_slots": num_slots}
        s.table_state_version += 1
    # Hold the flash on for the whole guided session so every /slots/<n>/scan
    # runs under steady lighting without the 300ms warmup each pulse. The
    # Pi's capture_with_flash short-circuits its own warmup when flash.held.
    _pi_flash(s, True)
    t = Thread(target=_guided_deal_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _stop_guided_deal(s):
    """Signal the guided loop to exit and clear LEDs."""
    gd = s.guided_deal
    if gd is None:
        return
    # The guided_deal dict carries either "num_slots" (full deal) or "slots"
    # (explicit list, used by draw-phase replacement). Turn everything off.
    slots = gd.get("slots") or list(range(1, gd.get("num_slots", 0) + 1))
    with s.table_lock:
        s.guided_deal = None
        s.table_state_version += 1
    for n in slots:
        _pi_slot_led(s, n, "off")
    # Release the held flash; regular poller / idle state will manage it.
    _pi_flash(s, False)


def _start_guided_replace(s, slots, previous_cards=None):
    """Kick off the guided loop for a specific slot list (draw-phase
    replacement). Same as _start_guided_deal but iterates through the
    supplied slot numbers in order rather than 1..N.

    previous_cards maps slot_num → code string ("Ah") for the card that
    was in each slot before the replace started. The loop uses it to
    detect "card changed" in case the Pi scan misses the present=false
    moment between the swap."""
    if s.guided_deal is not None:
        return
    ordered = [int(x) for x in slots if isinstance(x, int) or str(x).isdigit()]
    ordered = sorted(set(ordered))
    if not ordered:
        return
    prev = {int(k): str(v) for k, v in (previous_cards or {}).items()}
    with s.table_lock:
        s.guided_deal = {
            "slots": ordered, "index": 0, "mode": "replace",
            "previous_cards": prev,
        }
        s.console_state = "replacing"
        s.table_state_version += 1
    _pi_flash(s, True)
    t = Thread(target=_guided_replace_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _start_guided_trailing_deal(s, slots: list[int]):
    """Kick off guided flow for trailing down cards (7 Card Stud's 7th
    street, Follow the Queen's 7th). Console stays in 'dealing' until the
    loop finishes, then transitions to 'betting' for one final Pot-is-right
    before hand_over."""
    if s.guided_deal is not None:
        return
    ordered = sorted(set(int(x) for x in slots if isinstance(x, int) or str(x).isdigit()))
    if not ordered:
        return
    with s.table_lock:
        s.guided_deal = {"slots": ordered, "index": 0, "mode": "trailing"}
        s.console_state = "dealing"
        s.table_state_version += 1
    _pi_flash(s, True)
    t = Thread(target=_guided_replace_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _guided_replace_loop(s):
    """Variant of _guided_deal_loop driven by an explicit slot list.
    LEDs light in the order given; cleared rodney_downs entries refill
    as new cards are scanned.

    Shared by draw-phase replacement (mode='replace') and the trailing-down
    deal for stud games (mode='trailing'); only the completion transition
    differs."""
    gd = s.guided_deal
    if gd is None or "slots" not in gd:
        return
    slots = list(gd["slots"])
    mode = gd.get("mode", "replace")
    log.log(f"[GUIDED/{mode}] Started — slots {slots}")
    # Strict single-slot: only the slot being processed right now has its
    # LED lit, every other slot is off. Previously upcoming slots blinked,
    # which looked like we were trying to process them in parallel.
    for i, n in enumerate(slots):
        _pi_slot_led(s, n, "on" if i == 0 else "off")
    stable_count = 0
    best_card = None
    settled = False
    # In replace mode the old card may still be physically in the slot
    # when guided starts — require a present=false transition before we
    # accept a present=true reading, otherwise the old card gets re-
    # committed as "new". Trailing mode starts from an empty slot.
    require_empty_first = (mode == "replace")
    saw_empty = not require_empty_first

    while True:
        gd = s.guided_deal
        if gd is None:
            log.log(f"[GUIDED/{mode}] Stopped externally")
            return
        idx = gd.get("index", 0)
        if idx >= len(slots):
            for n in slots:
                _pi_slot_led(s, n, "off")
            log.log(f"[GUIDED/{mode}] Complete")
            with s.table_lock:
                s.guided_deal = None
                s.table_state_version += 1
                if mode == "trailing":
                    # 7CS/FTQ 7th street done — one more betting round before
                    # hand_over. console_trailing_done tells next_round to
                    # skip the trailing branch second time through.
                    s.console_state = "betting"
                    s.console_trailing_done = True
                    log.log(
                        "[CONSOLE] Trailing down deal complete → final betting"
                    )
                    # The wild-card and bet-first announcements were
                    # deferred from the last-up-round Confirm Cards so
                    # the player hears them only once, after the final
                    # card is on the table.
                    _announce_trailing_done(s)
                elif s.console_state == "replacing":
                    # Record this draw as done. Multi-draw games (3 Toed
                    # Pete) use this to know whether another DRAW phase
                    # follows; post-draw betting round number is the
                    # count of draws completed so far + 1.
                    s.rodney_draws_done += 1
                    s.console_state = "betting"
                    s.console_betting_round = s.rodney_draws_done + 1
                    log.log(
                        f"[CONSOLE] Draw {s.rodney_draws_done} replacement "
                        f"done → betting round {s.console_betting_round}"
                    )
            return
        expecting = slots[idx]

        with s.table_lock:
            already_filled = expecting in s.rodney_downs
            pv = s.pending_verify
            waiting_verify = pv is not None and pv.get("slot") == expecting

        if already_filled:
            _pi_slot_led(s, expecting, "off")
            with s.table_lock:
                gd["index"] = idx + 1
                s.table_state_version += 1
            if idx + 1 < len(slots):
                _pi_slot_led(s, slots[idx + 1], "on")
            stable_count = 0
            best_card = None
            settled = False
            saw_empty = not require_empty_first
            continue

        if waiting_verify:
            time.sleep(0.3)
            continue

        result = _pi_slot_scan(s, expecting)
        if result is None:
            time.sleep(1.5)
            continue

        present = bool(result.get("present"))
        cur = result.get("card") or {}
        cur_code = (
            f"{cur['rank']}{cur['suit'][0]}"
            if cur.get("rank") and cur.get("suit") else ""
        )
        log.log(
            f"[GUIDED/{mode}] Slot {expecting}: present={present} "
            f"card={cur_code or '-'} "
            f"conf={cur.get('confidence', 0.0):.2f} "
            f"saw_empty={saw_empty}"
        )

        if not present:
            if not saw_empty:
                log.log(f"[GUIDED/{mode}] Slot {expecting}: empty — ready for new card")
            saw_empty = True
            stable_count = 0
            best_card = None
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        # Replace mode: the old card may still be in the slot at loop
        # start. Accept the scan only after either (a) we saw present=
        # false at some point, or (b) YOLO reads a DIFFERENT card code
        # than the one that was in the slot before the replace started
        # — user swapped the card faster than our polling.
        if not saw_empty:
            prev_code = gd.get("previous_cards", {}).get(expecting, "")
            if cur_code and prev_code and cur_code != prev_code:
                log.log(
                    f"[GUIDED/{mode}] Slot {expecting}: card changed "
                    f"{prev_code} → {cur_code} (no empty seen) — accepting"
                )
                saw_empty = True
            else:
                time.sleep(GUIDED_POLL_S)
                continue

        if not settled:
            time.sleep(GUIDED_SETTLE_S)
            settled = True
            continue

        stable_count += 1
        card = result.get("card")
        if card:
            conf = float(card.get("confidence", 0.0))
            if best_card is None or conf > float(best_card.get("confidence", 0.0)):
                best_card = card
            code = f"{card['rank']}{card['suit'][0]}"
            if conf >= GUIDED_GOOD_CONF:
                with s.table_lock:
                    s.rodney_downs[expecting] = {
                        "rank": card["rank"],
                        "suit": card["suit"],
                        "confidence": round(conf, 2),
                    }
                    s.pi_prev_slots[expecting] = code
                    gd["index"] = idx + 1
                    s.table_state_version += 1
                _stats_bump(s, "pi_auto")
                _table_log_add(s, f"Slot {expecting} (replace): {code} (auto, {int(conf*100)}%)")
                _pi_slot_led(s, expecting, "off")
                if idx + 1 < len(slots):
                    _pi_slot_led(s, slots[idx + 1], "on")
                stable_count = 0
                best_card = None
                settled = False
                saw_empty = not require_empty_first
                continue

        if stable_count < GUIDED_STABLE_SCANS:
            time.sleep(GUIDED_POLL_S)
            continue

        if best_card is None:
            log.log(
                f"[GUIDED/{mode}] Slot {expecting}: present but nothing "
                f"recognized after {GUIDED_STABLE_SCANS} scans — continuing"
            )
            stable_count = 0
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        conf = float(best_card.get("confidence", 0.0))
        guess = {
            "rank": best_card["rank"],
            "suit": best_card["suit"],
            "confidence": round(conf, 2),
        }
        prompt = (
            f"Slot {expecting} (replacement): low confidence "
            f"({int(conf*100)}%). Confirm or correct."
        )

        with s.table_lock:
            s.pending_verify = {
                "slot": expecting,
                "guess": guess,
                "prompt": prompt,
                "image_url": f"/api/table/slot_image/{expecting}",
            }
            if guess["rank"]:
                s.slot_pending[expecting] = dict(guess)
            s.table_state_version += 1
        _pi_slot_led(s, expecting, "blink")


def _game_has_draw_phase(ge) -> bool:
    """True if the current game has a DRAW phase somewhere in its template."""
    try:
        from game_engine import PhaseType
        if ge.current_game is None:
            return False
        return any(ph.type == PhaseType.DRAW for ph in ge.current_game.phases)
    except Exception:
        return False


def _total_draw_phases(ge) -> int:
    """Count of DRAW phases. 5 Card Draw = 1, 3 Toed Pete = 3."""
    try:
        from game_engine import PhaseType
        if ge.current_game is None:
            return 0
        return sum(1 for ph in ge.current_game.phases if ph.type == PhaseType.DRAW)
    except Exception:
        return 0


def _max_draw_for_game(ge, draws_done: int = 0) -> int:
    """Max cards Rodney can replace in the draws_done-th DRAW phase.

    Multi-draw games (3 Toed Pete) shrink the allowance each round: 3, 2,
    then 1. draws_done is the number of draws already completed — 0 for
    the first draw, 1 for the second, etc. Returns 0 if no such phase.
    """
    try:
        from game_engine import PhaseType
        if ge.current_game is None:
            return 0
        seen = 0
        for ph in ge.current_game.phases:
            if ph.type == PhaseType.DRAW:
                if seen == draws_done:
                    return int(getattr(ph, "max_draw", 0) or 0)
                seen += 1
    except Exception:
        pass
    return 0


def _enqueue_down_card_verifies(s):
    """Called at the end of an up-card round (console Confirm Cards).

    Any slot that has a pending (low-confidence) scan that isn't already
    verified in rodney_downs gets added to the FIFO verify queue — but
    only if that slot has actually been dealt. A stale slot_pending
    entry for a trailing slot that we haven't started guiding yet (e.g.
    FTQ slot 3 during round 1) would otherwise pop a verify modal for a
    card that doesn't exist yet.
    """
    ge = s.game_engine
    initial = _initial_down_count(ge)
    # Slots currently valid to verify: the initial guided range plus any
    # slot the guided loop is presently iterating. Trailing slots get
    # added once the trailing guided session starts.
    expected_slots = set(range(1, initial + 1))
    if s.guided_deal:
        gd = s.guided_deal
        if "num_slots" in gd:
            expected_slots |= set(range(1, int(gd["num_slots"]) + 1))
        for sn in gd.get("slots") or []:
            try:
                expected_slots.add(int(sn))
            except (TypeError, ValueError):
                pass
    with s.table_lock:
        newly_queued = []
        for slot_num, guess in s.slot_pending.items():
            if slot_num in s.rodney_downs:
                continue
            if slot_num in s.verify_queue:
                continue
            if slot_num not in expected_slots:
                # Haven't dealt this slot yet (FTQ/7CS trailing, etc.) —
                # don't prompt the user to verify a card that isn't there.
                continue
            s.verify_queue.append(slot_num)
            newly_queued.append(slot_num)
        if newly_queued:
            for sn in newly_queued:
                _table_log_add(s, f"Slot {sn}: queued for verify (blink LED)")
            # Open the modal immediately rather than waiting for the next
            # poller tick — nicer UX, and in offline mode the poller may
            # be sleeping between failed fetches.
            _promote_next_verify(s)
            s.table_state_version += 1
    # TODO: POST to Pi /slots/<n>/led to blink once that endpoint is wired.
    return newly_queued


def _resolve_verify(s, card_dict):
    """Set the verified card into rodney_downs[slot] and clear the modal.

    Also tallies whether the Pi's guess matched what the user finally
    submitted — that's our accuracy signal for the low-confidence path
    (and by extension, a hint about whether GUIDED_GOOD_CONF is set too
    low or too high).
    """
    with s.table_lock:
        pv = s.pending_verify
        if not pv:
            return False
        slot = pv.get("slot")
        if slot is None:
            return False
        guess = pv.get("guess") or {}
        guess_matched = (
            bool(guess.get("rank"))
            and guess.get("rank") == card_dict.get("rank")
            and guess.get("suit") == card_dict.get("suit")
        )
        s.rodney_downs[slot] = {
            "rank": card_dict["rank"],
            "suit": card_dict["suit"],
        }
        code = f"{card_dict['rank']}{card_dict['suit'][0]}"
        s.pi_prev_slots[slot] = code
        s.slot_pending.pop(slot, None)
        if slot in s.verify_queue:
            s.verify_queue.remove(slot)
        s.pending_verify = None
        _table_log_add(s, f"Slot {slot}: {code} (verified)")
        s.table_state_version += 1
    _stats_bump(s, "pi_verify_right" if guess_matched else "pi_verify_wrong")
    return True


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

        s.monitor._recognize_single(player, crop)
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
        log.log(f"[DEAL] Card detected in {dealer_name}'s zone — waiting 2s for all cards to settle")
        dm["phase"] = "settling"
        dm["settle_time"] = time.time()


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
# Voice command dispatcher
# ---------------------------------------------------------------------------

def _derive_voice_phase(s):
    """Map the current console state to the voice-grammar phase.

    Returns one of:
      'pre_game'     — no game in progress; accepts game-selection commands
      'up_round'     — cards being scanned; accepts "{player}, {card}"
      'pre_confirm'  — every watched zone has a card; accepts Correction / Confirmed
      'pre_pot'      — betting round in progress; accepts Pot Is Right + Fold
      'other'        — anything else (draw / replacing / hand_over / idle mid-setup)
    """
    ge = s.game_engine
    if ge is None or ge.current_game is None:
        return "pre_game"
    if s.console_state == "dealing":
        if s.console_scan_phase == "scanned":
            return "pre_confirm"
        return "up_round"
    if s.console_state == "betting":
        return "pre_pot"
    return "other"


def _voice_post(path, body=None):
    """Internal-HTTP helper so voice commands go through the same
    endpoints the buttons do — guarantees identical side effects
    (state transitions, announce, stats, table version bumps)."""
    import urllib.request
    url = f"http://localhost:8888{path}"
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=3).read()
        return True
    except Exception as e:
        log.log(f"[VOICE] POST {path} failed: {type(e).__name__}: {e}")
        return False


# Debounce timer for the voice-readback. After any voice-driven card
# call we re-arm this timer; on 0.8 s of quiet it walks every active
# zone in deal order and speaks back what's currently recognized.
# This lets the dealer verify the round by ear instead of visually
# scanning the console while they're still calling the next card.
_voice_status_timer = None
_voice_status_lock = Lock()


def _schedule_voice_status_speech(delay_s: float = 0.8) -> None:
    """Arm (or re-arm) the debounce timer that will speak which
    players are still waiting once the flurry of voice card calls
    goes quiet."""
    global _voice_status_timer
    with _voice_status_lock:
        if _voice_status_timer is not None:
            _voice_status_timer.cancel()
        _voice_status_timer = Timer(delay_s, _speak_voice_status)
        _voice_status_timer.daemon = True
        _voice_status_timer.start()


def _speak_voice_status() -> None:
    """Delta-only readback. Each card is announced at most once per
    round: Brio's own recognition speech covers zones it scans,
    voice-driven corrections get announced here when the debounce
    fires, and corrections that change a card's value re-announce the
    new value. When every active zone is filled, announce 'All cards
    in' once so the dealer knows to say Confirmed.

    Runs on the debounce Timer thread — the global _state may have
    moved on between the voice calls and the timer firing, so always
    re-check phase and re-derive the watched set before speaking."""
    s = _state
    if s is None:
        return
    phase = _derive_voice_phase(s)
    if phase not in ("up_round", "pre_confirm"):
        return  # round probably ended; nothing to announce
    impl = getattr(s, "current_game_impl", None)
    if impl is not None:
        scan_names, _stand = impl.zones_to_scan(s)
    else:
        scan_names = list(s.console_active_players)
    # Walk in deal order (clockwise from dealer's left) so multiple
    # deltas (rare but possible) play back in a natural sequence.
    ge = s.game_engine
    dealer_idx = ge.dealer_index
    deal_order = [
        ge.players[(dealer_idx + i) % len(ge.players)].name
        for i in range(1, len(ge.players) + 1)
    ]
    ordered = [n for n in deal_order if n in scan_names]

    # Reset the per-round announce-tracker when the round advances.
    current_round = s.console_up_round + 1
    if getattr(s, "_voice_announced_round", -1) != current_round:
        s._voice_announced_cards = {}
        s._voice_announced_all_in = False
        s._voice_announced_round = current_round

    waiting = []
    for name in ordered:
        card = s.monitor.last_card.get(name, "")
        zstate = s.monitor.zone_state.get(name, "")
        if not card or card == "No card":
            waiting.append(name)
            continue
        # Brio's recognition path already called speech.say on its
        # own; just track that the card was announced so a later
        # voice-correction of the same zone can detect the delta.
        if zstate == "recognized":
            s._voice_announced_cards[name] = card
            continue
        # zone_state = "corrected" — voice call or manual console
        # correction. Speak only if the value differs from what we
        # already announced this round.
        if s._voice_announced_cards.get(name) != card:
            speech.say(f"{name}, {card}")
            log.log(f"[VOICE] Announce: {name}, {card}")
            s._voice_announced_cards[name] = card

    # "All cards in" fires once per round, the first time every
    # active zone has a card.
    if not waiting and not s._voice_announced_all_in:
        speech.say("All cards in")
        log.log("[VOICE] All cards in")
        s._voice_announced_all_in = True


def _process_voice_command(cmd):
    """Phase-filtered dispatch for a single parsed voice command.

    Commands spoken in the wrong phase are logged and ignored — no
    action fires. That way a stray "Confirmed" during a betting round
    doesn't accidentally skip ahead, and a "Pot is right" during deal
    doesn't short-circuit the scan phase.
    """
    s = _state
    if s is None:
        return
    from speech_recognition_module import (
        GameCommand, RepeatGameCommand, CardCallCommand, CorrectionCommand,
        ConfirmCommand, PotIsRightCommand, FoldCommand, UnrecognizedCommand,
    )
    phase = _derive_voice_phase(s)

    if isinstance(cmd, GameCommand):
        if phase != "pre_game":
            log.log(f"[VOICE] Ignoring 'game is {cmd.game_name}' in phase {phase}")
            return
        log.log(f"[VOICE] Starting new hand: {cmd.game_name}")
        _voice_post("/api/console/deal", {"game": cmd.game_name})
        return

    if isinstance(cmd, RepeatGameCommand):
        if phase != "pre_game":
            log.log(f"[VOICE] Ignoring 'same game again' in phase {phase}")
            return
        last = getattr(s, "last_game_name", "") or ""
        if not last:
            log.log("[VOICE] 'Same game again' heard but no previous game recorded")
            speech.say("No previous game to repeat")
            return
        log.log(f"[VOICE] Repeating previous game: {last}")
        _voice_post("/api/console/deal", {"game": last})
        return

    if isinstance(cmd, CardCallCommand):
        # Voice-assigning a card during an up-card round (or pre-confirm
        # if the user didn't use the "Correction:" prefix). Routes
        # through /api/console/correct so the monitor picks it up
        # exactly like a typed correction — locks the zone from the
        # next Brio scan, updates training_data, etc.
        if phase not in ("up_round", "pre_confirm"):
            log.log(
                f"[VOICE] Ignoring card call "
                f"'{cmd.player}, {cmd.rank}{cmd.suit[0]}' in phase {phase}"
            )
            return
        log.log(f"[VOICE] {cmd.player}: {cmd.rank} of {cmd.suit}")
        _voice_post("/api/console/correct", {
            "corrections": [{"player": cmd.player, "rank": cmd.rank, "suit": cmd.suit}],
        })
        # Arm the debounced readback. Dealer says all the cards
        # (quickly, in sequence); once they pause for 0.8 s, the
        # system walks every active zone in deal order and speaks
        # back each player's current card so the dealer can verify
        # by ear without looking at the console. No per-card echo —
        # that was overlapping with their next utterance.
        _schedule_voice_status_speech()
        return

    if isinstance(cmd, CorrectionCommand):
        if phase != "pre_confirm":
            log.log(
                f"[VOICE] Ignoring 'Correction: {cmd.player}, "
                f"{cmd.rank}{cmd.suit[0]}' in phase {phase}"
            )
            return
        log.log(f"[VOICE] Correction: {cmd.player} {cmd.rank} of {cmd.suit}")
        _voice_post("/api/console/correct", {
            "corrections": [{"player": cmd.player, "rank": cmd.rank, "suit": cmd.suit}],
        })
        _schedule_voice_status_speech()
        return

    if isinstance(cmd, ConfirmCommand):
        if phase != "pre_confirm":
            log.log(f"[VOICE] Ignoring 'Confirmed' in phase {phase}")
            return
        log.log("[VOICE] Confirmed → /api/console/confirm")
        _voice_post("/api/console/confirm")
        return

    if isinstance(cmd, PotIsRightCommand):
        if phase != "pre_pot":
            log.log(f"[VOICE] Ignoring 'Pot is right' in phase {phase}")
            return
        log.log("[VOICE] Pot is right → /api/console/next_round")
        _voice_post("/api/console/next_round")
        return

    if isinstance(cmd, FoldCommand):
        if phase != "pre_pot":
            log.log(f"[VOICE] Ignoring '{cmd.player} folds' in phase {phase}")
            return
        log.log(f"[VOICE] {cmd.player} folds → /api/table/fold")
        _voice_post("/api/table/fold", {"player": cmd.player, "folded": True})
        return

    if isinstance(cmd, UnrecognizedCommand):
        log.log(f"[VOICE] ? {cmd.raw_text!r}")
        return


# ---------------------------------------------------------------------------
# Background capture
# ---------------------------------------------------------------------------

def bg_loop():
    while not _state.quit_flag:
        frame = _state.capture.capture()
        if frame is not None:
            _state.latest_frame = frame
            # Recognition/motion detection runs FIRST so card arrival
            # doesn't wait on the display-JPEG encode below. On a 4K
            # Brio frame to_jpeg was eating 0.5-1.5s per bg_loop pass.
            if _state.monitoring and _state.cal.ok:
                _console_watch_dealer(_state, frame)

            # Display JPEG is cheap once we downscale — the UI renders
            # it inside a small iframe anyway. Keep the overlay drawn
            # on the full frame (zone coordinates are in 4K space),
            # then scale the encoded output.
            disp = crop_circle(frame, _state.cal).copy()
            draw_overlay(disp, _state.cal, _state.monitor)
            small = cv2.resize(disp, (1280, 720), interpolation=cv2.INTER_AREA)
            _state.latest_jpg = to_jpeg(small, 70)

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
                elif dm["phase"] == "settling":
                    if time.time() - dm.get("settle_time", 0) >= 2:
                        log.log("[DEAL] Scanning all zones")
                        dm["phase"] = "scanning"
                        dm["announced_this_round"] = set()
                        _deal_scan_all_zones(_state)
                elif dm["phase"] == "retry_missing":
                    _deal_retry_missing(_state)
                elif dm["phase"] == "waiting_to_clear":
                    _deal_check_zones_clear(_state)

        time.sleep(1)  # 1 second capture rate



# ---------------------------------------------------------------------------
# Web server — single page app
# ---------------------------------------------------------------------------

# The HTTP handler lives in its own module. Import happens HERE,
# after every helper it needs is already defined, so its
# `from overhead_test import ...` block sees a fully-loaded module.
from http_server import Handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _state
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help=f"avfoundation camera index; default is auto-detected "
                             f"by name (looks for '{DEFAULT_CAMERA_NAME}')")
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME,
                        help="Substring of the avfoundation device name to prefer "
                             "when auto-selecting the camera")
    parser.add_argument("--cv-camera-index", type=int, default=None,
                        help="Force a specific OpenCV VideoCapture index, skipping "
                             "name-based lookup. Use this when multiple 4K cameras "
                             "are connected and the auto-picker opens the wrong one.")
    parser.add_argument("--brio-focus", type=int, default=None,
                        help="Manual focus value for the Brio (0..255, lower = "
                             "farther). Omitting the flag leaves autofocus on.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--voice", type=str, default=None,
                        help="Base voice name for `say`. The actual voice used "
                             "is the highest-quality installed variant "
                             "(Premium > Enhanced > base). Overrides SPEECH_VOICE env.")
    parser.add_argument("--listen", action="store_true",
                        help="Enable phase-filtered speech-input commands via "
                             "MLX Whisper on the Mac mic (\"The game is …\", "
                             "\"{player}, {card}\", \"Correction: …\", "
                             "\"Confirmed\", \"Pot is right\", \"{player} folds\", "
                             "\"Same game again\"). Requires mlx-whisper, "
                             "SpeechRecognition, pyaudio, and portaudio.")
    args = parser.parse_args()

    if args.voice:
        speech.voice = _resolve_best_voice(args.voice)
        log.log(f"Speech voice overridden to: {speech.voice}")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    camera_index = args.camera
    if camera_index is None:
        camera_index = FrameCapture.find_index_by_name(args.camera_name)
        if camera_index is None:
            camera_index = DEFAULT_CAMERA_INDEX
            log.log(f"Camera: '{args.camera_name}' not found in avfoundation devices, "
                    f"falling back to index {camera_index}")

    # OpenCV VideoCapture index can differ from AVFoundations enumeration
    # when multiple 4K cameras are attached. Persist whatever value the user
    # passes via --cv-camera-index so the next run picks up the Brio without
    # having to re-specify it.
    _persisted_cfg = _load_host_config()
    cv_idx = args.cv_camera_index
    if cv_idx is not None:
        _save_host_config({"cv_camera_index": cv_idx})
        log.log(f"[CAPTURE] Saved cv_camera_index={cv_idx} to host config")
    elif "cv_camera_index" in _persisted_cfg:
        cv_idx = _persisted_cfg["cv_camera_index"]
        log.log(f"[CAPTURE] Loaded cv_camera_index={cv_idx} from host config")

    # Brio manual focus override — autofocus hunts on the low-contrast
    # felt background, so pin a focus position once and keep it.
    brio_focus = args.brio_focus
    if brio_focus is not None:
        _save_host_config({"brio_focus": brio_focus})
        log.log(f"[CAPTURE] Saved brio_focus={brio_focus} to host config")
    elif "brio_focus" in _persisted_cfg:
        brio_focus = _persisted_cfg["brio_focus"]
        log.log(f"[CAPTURE] Loaded brio_focus={brio_focus} from host config")

    capture = FrameCapture(camera_index, args.resolution,
                           camera_name_hint=args.camera_name,
                           cv_index_override=cv_idx,
                           focus=brio_focus)
    log.log(f"Camera {camera_index}, resolution {capture.resolution}")

    # Wait for the persistent ffmpeg stream to warm up enough to produce
    # a frame. AVFoundation can take several seconds to open the Brio.
    print("  Waiting for first frame from camera stream...")
    frame = None
    deadline = time.time() + 15.0
    while time.time() < deadline:
        frame = capture.capture()
        if frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        tail = getattr(capture, "_stderr_tail", b"").decode(errors="replace").strip()
        hint = f"\n  ffmpeg stderr tail: {tail}" if tail else ""
        sys.exit(
            f"  ERROR: No frames from camera after 15s. "
            f"Is another app holding the Brio?{hint}"
        )
    print(f"  OK: {frame.shape[1]}x{frame.shape[0]}")

    cal = Calibration()
    cal.load()

    monitor = ZoneMonitor(
        threshold=args.threshold,
        get_zones=lambda: cal.zones,
        stats_cb=lambda key: _stats_bump(_state, key),
    )
    _state = AppState(capture, cal, monitor)
    _state.latest_frame = frame
    # Apply any persisted YOLO min-confidence now that the monitor exists.
    _persisted = _load_host_config()
    if "yolo_min_conf" in _persisted:
        try:
            monitor.yolo_min_conf = max(0.0, min(1.0, float(_persisted["yolo_min_conf"])))
        except (TypeError, ValueError):
            pass

    # Start server. ThreadingHTTPServer gives each client connection its own
    # thread so a browser's keep-alive polling (e.g. /table/state every 500ms)
    # can't starve other clients like /console or /logview.
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
    server.daemon_threads = True
    Thread(target=server.serve_forever, daemon=True).start()

    # Auto-start Pi slot poller so /table populates Rodney's hand without a
    # manual kick. The loop handles Pi-unreachable with a retry delay, so
    # starting it here is safe even if the Pi is off.
    _pi_poll_start(_state)
    log.log(f"Pi poller started against {_state.pi_base_url}")
    log.log("Server at http://localhost:8888")

    # Start background capture
    Thread(target=bg_loop, daemon=True).start()

    # Optional speech-input listener. The SpeechListener runs on its
    # own background thread; _process_voice_command is the callback
    # for each parsed command. Mic calibration takes ~2s at startup
    # and model load ~30s on first run (downloads a whisper-small
    # checkpoint to ~/.cache), so this is gated behind --listen to
    # keep the normal boot path fast.
    if args.listen:
        try:
            from speech_recognition_module import (
                SpeechListener, set_log_function,
            )
            set_log_function(log.log)
            listener = SpeechListener(callback=_process_voice_command)
            listener.start()
            log.log("[VOICE] speech-input listener started (--listen)")
        except Exception as e:
            log.log(f"[VOICE] could not start listener: {type(e).__name__}: {e}")
            log.log("[VOICE] install deps with: pip3 install mlx-whisper SpeechRecognition pyaudio")

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

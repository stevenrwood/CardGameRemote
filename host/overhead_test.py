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
import re
import subprocess
import sys
import time
from datetime import datetime
from threading import Thread, Lock, Timer
from queue import Queue, Empty

import cv2
import http.server
import numpy as np

from log_buffer import log, LOG_DIR, LOG_FILE
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
    _pi_buzz,
    _pi_slot_led,
    _pi_slot_scan,
)
from ui_templates import (
    TABLE_HTML, LOGVIEW_HTML, CONSOLE_HTML,
    SCANNER_TMPL, CALIBRATE_TMPL,
)

# ---------------------------------------------------------------------------
# Re-exports — see host_constants.py / calibration.py / frame_utils.py /
# app_state.py. http_server.py still imports these names from
# overhead_test, so re-export them here.
# ---------------------------------------------------------------------------

from host_constants import (
    PLAYER_NAMES,
    NUM_ZONES,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CAMERA_NAME,
    DEFAULT_THRESHOLD,
    DEFAULT_RESOLUTION,
    DEFAULT_BRIO_SETTLE_S,
    CALIBRATION_FILE,
)
from calibration import Calibration
from frame_utils import crop_circle, draw_overlay, to_jpeg


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
# Console scan trigger — see brio_watcher.py. Re-exported here so the
# bg_loop iter (later in this file) can call _console_watch_dealer.
# ---------------------------------------------------------------------------

from brio_watcher import (  # noqa: E402,F401
    ZONE_DIAG_INTERVAL_S,
    ZONE_MAX_EMPTY_SCANS,
    ZONE_PRESENCE_FRACTION,
    ZONE_PRESENCE_PIXEL_DIFF,
    ZONE_STABILITY_THRESHOLD,
    ZONE_STABLE_FRAMES,
    _all_active_zones_present,
    _auto_scan_all_zones,
    _brio_player_names,
    _console_rescan_missing,
    _console_watch_dealer,
    _zone_presence_metric,
)


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
    """Detect duplicate cards at confirm time.

    Two flavors of collision get treated differently:

    1. SAME-SCAN duplicate — two zones in *this* Brio scan returned
       the same card. Almost always a YOLO hallucination (the model
       has a bias toward a handful of cards on ambiguous inputs;
       Ace of Diamonds especially). The auto-recognized card gets
       cleared so the dealer is forced to re-scan or hand-correct
       that player. User-corrected zones are trusted.

    2. CROSS-ROUND duplicate — this round's card matches one already
       in console_hand_cards (a prior round) or Rodney's known
       downs. We can't tell which side is wrong: maybe a previous
       round had a hallucination and this round is right; maybe
       this round is the hallucination. Log it and leave as-is so
       the genuine card still shows in the player's hand. The
       dealer can correct after the fact via the zone-tap modal if
       needed.

    Operates in place on round_cards.
    """
    history_seen = set()
    for c in s.console_hand_cards:
        history_seen.add(c["card"])
    suit_full = {"c": "Clubs", "d": "Diamonds", "h": "Hearts", "s": "Spades"}
    rank_full = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}

    def _canonical(rank, suit):
        return f"{rank_full.get(rank, rank)} of {suit.capitalize()}"

    flipped_slot = (s.rodney_flipped_up or {}).get("slot")
    for slot_num, d in s.rodney_downs.items():
        if slot_num == flipped_slot:
            continue
        history_seen.add(_canonical(d["rank"], d["suit"]))
    for d in s.slot_pending.values():
        history_seen.add(_canonical(d["rank"], d["suit"]))

    same_scan_seen = set()
    cleared = []
    for entry in list(round_cards):
        card = entry.get("card", "")
        player = entry.get("player", "")
        if not card:
            continue
        if card in same_scan_seen:
            # Same-scan dup → almost certainly a hallucination.
            if s.monitor.zone_state.get(player) == "corrected":
                log.log(
                    f"[CONFIRM] {player}: {card} duplicates same-scan "
                    f"card but was user-corrected — keeping as-is"
                )
                continue
            log.log(
                f"[CONFIRM] {player}: {card} duplicates same-scan "
                f"card — clearing recognition; dealer must re-scan "
                f"or correct"
            )
            s.monitor.last_card[player] = ""
            s.monitor.zone_state[player] = "empty"
            try:
                round_cards.remove(entry)
            except ValueError:
                pass
            cleared.append(player)
            continue
        if card in history_seen:
            # Cross-round dup → log only, leave the genuine-looking
            # card in place. The history entry could just as easily
            # be the hallucination.
            log.log(
                f"[CONFIRM] {player}: {card} duplicates a prior round / "
                f"Rodney down — keeping as-is (dealer can correct via "
                f"zone tap if wrong)"
            )
        same_scan_seen.add(card)
    if cleared:
        log.log(
            f"[CONFIRM] Cleared same-scan duplicates for: "
            f"{', '.join(cleared)}"
        )


# ---------------------------------------------------------------------------
# Poker bet-first announcer — see poker_announce.py. Re-exported here for
# http_server.py's `from overhead_test import …` block.
# ---------------------------------------------------------------------------

from poker_announce import _announce_poker_hand_bet_first  # noqa: E402,F401


# Follow the Queen wild-card tracking — see games/follow_the_queen.py.
# Re-exported below for http_server.py's `from overhead_test import …`
# block.
from games.follow_the_queen import (  # noqa: E402,F401
    _check_follow_the_queen_round,
    _recompute_follow_the_queen,
)


# ---------------------------------------------------------------------------
# Guided Pi-slot dealing — see guided_deal.py. Re-exported here because
# challenge.py / http_server.py / bg_loop callers expect to find these
# symbols on overhead_test.
# ---------------------------------------------------------------------------

from guided_deal import (  # noqa: E402,F401
    GUIDED_GOOD_CONF,
    GUIDED_POLL_S,
    GUIDED_SETTLE_S,
    GUIDED_STABLE_SCANS,
    _announce_trailing_done,
    _guided_deal_loop,
    _guided_replace_loop,
    _start_guided_deal,
    _start_guided_deal_range,
    _start_guided_replace,
    _start_guided_trailing_deal,
    _stop_guided_deal,
)


# ---------------------------------------------------------------------------
# App state — see app_state.py. AppState is re-exported here for any
# external caller; _state is the module-level singleton main() fills in.
# ---------------------------------------------------------------------------

from app_state import AppState  # noqa: E402,F401

_state = None


# ---------------------------------------------------------------------------
# Observer table view ("/table") — see table_state.py. Re-exported here
# for http_server.py's `from overhead_test import …` block.
# ---------------------------------------------------------------------------

from table_state import (  # noqa: E402,F401
    _build_table_state,
    _redact_remote_downs,
    _table_state_bump,
    _parse_card_any,
    _best_hand_for_cards,
)



# Verify-queue helpers — see verify_queue.py. Re-exported below for
# http_server.py + the rest of overhead_test that still calls them.
from verify_queue import (  # noqa: E402,F401
    _enqueue_down_card_verifies,
    _parse_card_code,
    _promote_next_verify,
    _resolve_verify,
    _table_log_add,
)


# Game-template introspection — see game_meta.py. Re-exported here so
# http_server.py and the older callers in this file still see the
# helpers on overhead_test.
from game_meta import (  # noqa: E402,F401
    BETTING_LIMIT_LABELS,
    BETTING_LIMIT_SPOKEN,
    FORCED_POT_LIMIT_GAMES,
    _betting_limit_label,
    _betting_limit_spoken,
    _cards_dealt_so_far,
    _dealing_phase_types,
    _forced_betting_limit,
    _game_has_draw_phase,
    _get_deal_order,
    _initial_down_count,
    _max_draw_for_game,
    _next_deal_position_type,
    _skip_inactive_dealer,
    _speak_ante,
    _total_card_rounds,
    _total_downs_in_pattern,
    _total_draw_phases,
    _trailing_down_slots,
)


# ---------------------------------------------------------------------------
# Pi poll loop + stuck-cards alert — see pi_poll.py. Re-exported here
# for http_server.py's `from overhead_test import …` block.
# ---------------------------------------------------------------------------

from pi_poll import (  # noqa: E402,F401
    _pi_poll_loop,
    _pi_poll_start,
    _pi_poll_stop,
    _simulate_offline_slot_scans,
    _update_flash_for_deal_state,
    _stuck_slots_at_new_hand,
    _alert_stuck_cards_at_new_hand,
)


# ---------------------------------------------------------------------------
# Challenge-game state machine — see games/challenge.py. Re-exported
# here for http_server.py's `from overhead_test import …` block.
# ---------------------------------------------------------------------------

from games.challenge import (  # noqa: E402,F401
    CHALLENGE_SUBSEQUENT_ANTE_CENTS,
    MAX_PASSES_PER_ROUND,
    _begin_challenge_vote,
    _bump_table_version,
    _challenge_ante_cents_for,
    _challenge_can_mark,
    _challenge_first_voter,
    _challenge_phase_label,
    _challenge_required_cards,
    _clear_rodney_challenge_leds,
    _fmt_money,
    _format_name_list,
    _game_is_challenge,
    _handle_challenge_winner,
    _log_and_speak,
    _reset_round_passes,
    _resolve_challenge_round,
    _set_challenge_vote,
    _start_next_challenge_round,
)



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
# Voice command dispatcher — see voice_dispatch.py. _process_voice_command
# is wired into the SpeechListener as its callback (see main()).
# ---------------------------------------------------------------------------

from voice_dispatch import (  # noqa: E402,F401
    _derive_voice_phase,
    _process_voice_command,
    _schedule_voice_status_speech,
    _speak_voice_status,
    _voice_post,
)


def bg_loop():
    # The original loop had no exception handling, so a single bad
    # iteration (e.g. a cv2 op throwing on a transient frame) silently
    # killed the thread — recognition stopped, /snapshot froze on the
    # last good frame, and the only way out was a host restart. Wrap
    # the body so the loop survives and logs the problem instead.
    while not _state.quit_flag:
        try:
            _bg_loop_iter()
        except Exception as e:
            log.log(f"[BG_LOOP] iteration error: {type(e).__name__}: {e}")
            time.sleep(1)


def _bg_loop_iter():
    frame = _state.capture.capture()
    if frame is None:
        time.sleep(1)
        return
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
    parser.add_argument("--brio-zoom", type=int, default=None,
                        help="UVC zoom-absolute for the Brio (100..500, "
                             "100 = 1x / fully zoomed out / widest FOV). "
                             "Omitting leaves whatever the camera was on; "
                             "set 100 once to lock max field of view.")
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

    # Brio UVC zoom — 100 (1×, full FOV) is the widest setting.
    brio_zoom = args.brio_zoom
    if brio_zoom is not None:
        _save_host_config({"brio_zoom": brio_zoom})
        log.log(f"[CAPTURE] Saved brio_zoom={brio_zoom} to host config")
    elif "brio_zoom" in _persisted_cfg:
        brio_zoom = _persisted_cfg["brio_zoom"]
        log.log(f"[CAPTURE] Loaded brio_zoom={brio_zoom} from host config")

    capture = FrameCapture(camera_index, args.resolution,
                           camera_name_hint=args.camera_name,
                           cv_index_override=cv_idx,
                           focus=brio_focus,
                           zoom=brio_zoom)
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

    def _per_card_speech(name, card_text):
        # Default: just "{name}, {card}". The active game class can
        # extend this — 7/27 appends "with N or less down below"
        # when the player's running up-card total nears 27.
        default = f"{name}, {card_text}"
        impl = getattr(_state, "current_game_impl", None) if _state else None
        if impl is None:
            return default
        try:
            return impl.annotate_card_speech(_state, name, card_text, default)
        except Exception as e:
            log.log(f"[SPEECH] annotate_card_speech failed: {e!r}")
            return default

    monitor = ZoneMonitor(
        threshold=args.threshold,
        get_zones=lambda: cal.zones,
        stats_cb=lambda key: _stats_bump(_state, key),
        speech_formatter=_per_card_speech,
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
        log.log("[VOICE] --listen set; initialising speech listener")
        try:
            from speech_recognition_module import (
                SpeechListener, set_log_function,
            )
            set_log_function(log.log)
            # Pull the live game catalog so Whisper's bias prompt
            # covers every game name the dealer might say (including
            # ones Whisper would otherwise hallucinate, like
            # "Hodgecargo" for "High Chicago").
            game_names = list(_state.game_engine.templates.keys())
            listener = SpeechListener(
                callback=_process_voice_command,
                game_names=game_names,
                # Re-read the live AppState value each iteration so
                # the dealer can adjust the floor mid-night via the
                # Setup modal without restarting the host.
                min_energy_threshold_fn=(
                    lambda: _state.whisper_min_energy_threshold
                ),
            )
            listener.start()
            _state.whisper_listener = listener
            log.log("[VOICE] speech-input listener started (--listen)")
        except Exception as e:
            import traceback
            log.log(f"[VOICE] FAILED to start listener: "
                    f"{type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines():
                log.log(f"[VOICE]   {line}")
            log.log("[VOICE] install deps with: "
                    "pip3 install mlx-whisper SpeechRecognition pyaudio")
    else:
        log.log("[VOICE] --listen NOT set; speech input disabled")

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

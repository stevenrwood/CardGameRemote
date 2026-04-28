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
    wild_set = set(wild_ranks)
    all_results = []  # list of (name, HandResult), pre-sort
    for name in s.console_active_players:
        if name in s.folded_players:
            continue
        cards = per_player_cards.get(name, [])
        if not cards:
            continue
        try:
            if len(cards) == 1:
                # Single up card — treat as high-card only. A wild
                # card here serves as an Ace, otherwise the natural
                # rank value applies.
                rank, suit = cards[0]
                if rank in wild_set:
                    v = 14
                    rank_name = "Ace"
                else:
                    v = RANK_VALUE.get(rank, 0)
                    rank_name = RANK_NAME.get(rank, rank)
                result = HandResult(
                    "high_card",
                    f"{rank_name} high",
                    [v],
                    [],
                )
            else:
                result = best_hand(cards, wild_ranks=wild_ranks)
        except Exception as e:
            log.log(f"[POKER] eval {name} failed: {e}")
            continue
        log.log(f"[POKER] {name}: {result.label}")
        all_results.append((name, result))

    # Stable two-pass sort: first by deal order ascending so a tie on
    # hand strength resolves to the player dealt first ("first Ace
    # bets" convention), then by (category rank, tiebreakers)
    # descending so [0] is the best hand and [1] (if present) is the
    # runner-up. The runner-up is needed below to decide how many
    # kicker cards actually have to be announced — saying "Ace, Jack,
    # Eight is high" when the next hand is just "King high" is more
    # than the dealer needs.
    dealer_name = ge.get_dealer().name if ge.players else ""
    deal_order_index = {
        nm: i for i, nm in enumerate(_get_deal_order(dealer_name))
    }
    all_results.sort(key=lambda nr: deal_order_index.get(nr[0], 999))
    all_results.sort(
        key=lambda nr: (nr[1].rank, nr[1].tiebreakers),
        reverse=True,
    )
    if all_results:
        best_player, best_result = all_results[0]
        if len(all_results) > 1:
            runner_up_player, runner_up_result = all_results[1]
        else:
            runner_up_player = None
            runner_up_result = None
    else:
        best_player = None
        best_result = None
        runner_up_player = None
        runner_up_result = None

    if best_player is not None and best_result is not None:
        try:
            from poker_hands import RANK_PLURAL
        except Exception:
            RANK_PLURAL = {}
        cat = best_result.category
        tb = best_result.tiebreakers
        name_of = lambda v: RANK_NAME.get(VALUE_RANK.get(v, ""), "")
        plural_of = lambda v: RANK_PLURAL.get(VALUE_RANK.get(v, ""), name_of(v) + "s")

        # Per-category, how many of `tb`'s leading positions are
        # "primary" (already part of the spoken phrase). Anything
        # beyond is a kicker that only gets announced when the
        # runner-up has the same primary and the kicker actually
        # decides the hand. high_card has no fixed primary — every
        # value is just a tiebreaker the dealer narrates one by one.
        primary_count = {
            "five_of_a_kind": 1,
            "four_of_a_kind": 1,
            "three_of_a_kind": 1,
            "full_house": 2,
            "flush": 1,
            "two_pair": 2,
            "pair": 1,
            "straight_flush": 1,
            "straight": 1,
            "high_card": 0,
        }.get(cat, 1)

        def _extended_compare(result, cards):
            """Return descending [primary..., kickers...]. For
            categories whose tiebreakers don't carry kickers (pair,
            two_pair, three / four of a kind), pull kickers from the
            visible cards minus the cards used in the primary hand,
            with wild cards counted as Ace=14."""
            primary = list(result.tiebreakers)
            kc = result.category
            if kc in ("five_of_a_kind", "straight_flush", "straight",
                      "full_house", "flush", "high_card"):
                return primary
            used_set = {(c.rank, c.suit.lower()) for c in result.cards}
            kickers = []
            for rank, suit in cards:
                if (rank, suit.lower()) in used_set:
                    continue
                if rank in wild_set:
                    kickers.append(14)
                else:
                    rv = RANK_VALUE.get(rank, 0)
                    if rv:
                        kickers.append(rv)
            kickers.sort(reverse=True)
            return primary + kickers

        best_cards = per_player_cards.get(best_player, [])
        best_compare = _extended_compare(best_result, best_cards)
        if (runner_up_result is not None
                and runner_up_result.category == cat):
            ru_cards = per_player_cards.get(runner_up_player, [])
            ru_compare = _extended_compare(runner_up_result, ru_cards)
        else:
            ru_compare = []

        # How many positions of best_compare actually need to be
        # spoken to differentiate from the runner-up. Always at least
        # 1, and never less than the category's primary_count (since
        # the spoken phrase already names that many).
        needed = 1
        for i, v in enumerate(best_compare):
            needed = i + 1
            if i >= len(ru_compare) or v > ru_compare[i]:
                break
        if needed < primary_count:
            needed = primary_count
        extra = max(0, needed - primary_count)
        # True high-card tie with the runner-up (e.g. round 1 of stud
        # where Joe shows a wild 2 and Rodney shows a real Ace —
        # both score Ace high). Bet goes to the player dealt first;
        # the dealer announces it as "first <Rank> is high".
        is_high_card_tie = (
            cat == "high_card"
            and ru_compare
            and best_compare == list(ru_compare)
        )

        def _kicker_suffix(values):
            names = [name_of(v) for v in values if v]
            if not names:
                return ""
            joined = ", ".join(names)
            return f"{joined} kicker" if len(names) == 1 else f"{joined} kickers"

        def _maybe_append_kickers(base_phrase, kicker_start):
            if extra <= 0:
                return base_phrase
            suffix = _kicker_suffix(best_compare[kicker_start:kicker_start + extra])
            return f"{base_phrase}, {suffix}" if suffix else base_phrase

        if cat == "five_of_a_kind":
            hand_phrase = f"Five {plural_of(tb[0])}"
        elif cat == "four_of_a_kind":
            hand_phrase = _maybe_append_kickers(
                f"Four {plural_of(tb[0])}", 1
            )
        elif cat == "three_of_a_kind":
            hand_phrase = _maybe_append_kickers(
                f"Three {plural_of(tb[0])}", 1
            )
        elif cat == "full_house":
            hand_phrase = (
                f"Full house, {plural_of(tb[0])} over {plural_of(tb[1])}"
            )
        elif cat == "two_pair":
            hand_phrase = _maybe_append_kickers(
                f"Two {plural_of(tb[0])} and two {plural_of(tb[1])}", 2
            )
        elif cat == "pair":
            hand_phrase = _maybe_append_kickers(
                f"Two {plural_of(tb[0])}", 1
            )
        elif cat == "straight_flush" and tb[0] == 14:
            hand_phrase = "Royal flush"
        elif cat == "straight_flush":
            hand_phrase = f"{name_of(tb[0])} high straight flush"
        elif cat == "flush":
            # Flush extends the "X high flush" phrase rather than
            # using the "kicker" suffix — convention is "Ace, Jack
            # high flush" when the second card breaks the tie.
            ranks = [name_of(v) for v in best_compare[:needed] if v]
            top_part = ", ".join(ranks) if ranks else name_of(tb[0])
            hand_phrase = f"{top_part} high flush"
        elif cat == "straight":
            hand_phrase = f"{name_of(tb[0])} high straight"
        else:
            # high_card: phrase IS the (truncated) tiebreaker list.
            if not best_compare:
                hand_phrase = "no card"
            else:
                ranks = [name_of(v) for v in best_compare[:needed] if v]
                hand_phrase = ", ".join(ranks) if ranks else "no card"
        if is_high_card_tie and hand_phrase != "no card":
            # "Ace" → "first Ace", "Ace, Jack" → "first Ace, Jack" —
            # the wild-card-equals-real-card tie idiom carries through
            # any kicker positions that also tied.
            hand_phrase = f"first {hand_phrase[0].lower()}{hand_phrase[1:]}"
        phrase = f"{best_player}, {hand_phrase} is high. Your bet."
        log.log(f"[POKER] Bet first: {phrase} ({best_result.label})")
        speech.say(phrase)
        s.last_bet_first = best_player
    else:
        s.last_bet_first = None


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


def _max_draw_for_game(ge, draws_done: int = 0, s=None) -> int:
    """Max cards Rodney can replace in the draws_done-th DRAW phase.

    Multi-draw games (3 Toed Pete) shrink the allowance each round: 3, 2,
    then 1. draws_done is the number of draws already completed — 0 for
    the first draw, 1 for the second, etc. Returns 0 if no such phase.

    ``s`` is an optional AppState used for runtime caps that depend on
    active player count. 5 Card Double Draw at a 5-handed table can't
    fit a second 3-card draw (5×5 + 5×3 + 5×3 = 55 > 52), so the
    second-draw allowance is capped at 2 when five players are still
    in. With 4 or fewer the full 3 is fine.
    """
    try:
        from game_engine import PhaseType
        if ge.current_game is None:
            return 0
        seen = 0
        for ph in ge.current_game.phases:
            if ph.type == PhaseType.DRAW:
                if seen == draws_done:
                    base = int(getattr(ph, "max_draw", 0) or 0)
                    if (s is not None
                            and ge.current_game.name == "5 Card Double Draw"
                            and seen == 1
                            and len(getattr(s, "console_active_players", [])) >= 5):
                        return min(base, 2)
                    return base
                seen += 1
    except Exception:
        pass
    return 0


# Games whose betting limit is always Pot limit (dealer can't pick
# anything else). Everything else defaults to $1/$2 but the dealer
# can override from the console dropdown.
FORCED_POT_LIMIT_GAMES = {
    "3 Toed Pete",
    "High, Low, High",
    "Low, High, Low",
    "Low, Low, High",
    "Texas Hold'em",
}

BETTING_LIMIT_LABELS = {
    "1_2": "$1 / $2",
    "1_all_way": "$1 all the way",
    "pot": "Pot limit",
}

# Spoken phrasing for TTS — "$1 / $2" doesn't read well, so use
# word form for the Start-Game announcement.
BETTING_LIMIT_SPOKEN = {
    "1_2": "Betting is one and two",
    "1_all_way": "Betting is one all the way",
    "pot": "Betting is pot limit",
}


def _forced_betting_limit(game_name: str):
    """Return 'pot' if the given game forces Pot limit, None otherwise."""
    return "pot" if game_name in FORCED_POT_LIMIT_GAMES else None


def _betting_limit_label(code: str) -> str:
    return BETTING_LIMIT_LABELS.get(code, code)


def _betting_limit_spoken(code: str) -> str:
    return BETTING_LIMIT_SPOKEN.get(code, _betting_limit_label(code))


def _speak_ante(cents: int) -> str:
    """TTS-friendly ante phrasing — '50 cent', '75 cent', '1 dollar',
    '2 dollar'. Whole dollars take the singular 'dollar'; sub-dollar
    amounts use the integer + 'cent'."""
    if cents == 0:
        return "no ante"
    if cents % 100 == 0:
        return f"{cents // 100} dollar"
    return f"{cents} cent"


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
      'other'        — anything else (draw / replacing / hand_over / idle mid-setup,
                       OR poker night not yet started — Whisper is already
                       transcribing in --listen mode and a stray hallucination
                       would otherwise auto-deal a hand before the dealer hits
                       Start)
    """
    if not getattr(s, "night_active", False):
        return "other"
    ge = s.game_engine
    if ge is None or ge.current_game is None:
        return "pre_game"
    if s.console_state == "dealing":
        if s.console_scan_phase == "scanned":
            return "pre_confirm"
        return "up_round"
    if s.console_state == "betting":
        return "pre_pot"
    if s.console_state == "challenge_vote":
        return "challenge_vote"
    if s.console_state == "challenge_resolve":
        return "challenge_resolve"
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
        s._voice_inferred_this_round = set()
        s._voice_announced_round = current_round
    if not hasattr(s, "_voice_inferred_this_round"):
        s._voice_inferred_this_round = set()

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
        # already announced this round. Prefix "Inferred:" when the
        # card was resolved from an orphan voice call (no player
        # name spoken), so the dealer knows this one is a guess and
        # worth verifying before they Confirm.
        if s._voice_announced_cards.get(name) != card:
            if name in s._voice_inferred_this_round:
                speech.say(f"Inferred: {name}, {card}")
                log.log(f"[VOICE] Announce: Inferred: {name}, {card}")
                s._voice_inferred_this_round.discard(name)
            else:
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
        GameCommand, RepeatGameCommand, CardCallCommand, InferredCardCommand,
        CorrectionCommand, ConfirmCommand, PotIsRightCommand,
        ScanCardsCommand, FoldCommand,
        PassCommand, GoOutCommand, ChallengeWinnerCommand,
        UnrecognizedCommand,
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

    if isinstance(cmd, InferredCardCommand):
        # Dealer spoke just rank+suit without a name. Resolve to the
        # next unfilled zone in deal order (clockwise from dealer's
        # left, dealer last). Skip players who've folded/busted or
        # whose zone already has a card.
        if phase not in ("up_round", "pre_confirm"):
            log.log(
                f"[VOICE] Ignoring orphan card '{cmd.rank}{cmd.suit[0]}' "
                f"in phase {phase}"
            )
            return
        ge = s.game_engine
        dealer_idx = ge.dealer_index
        deal_order = [
            ge.players[(dealer_idx + i) % len(ge.players)].name
            for i in range(1, len(ge.players) + 1)
        ]
        impl = getattr(s, "current_game_impl", None)
        if impl is not None:
            scan_names, _stand = impl.zones_to_scan(s)
        else:
            scan_names = list(s.console_active_players)
        scan_set = set(scan_names)
        target = None
        for name in deal_order:
            if name not in scan_set:
                continue
            existing = s.monitor.last_card.get(name, "")
            # Only target zones that haven't been resolved yet
            # (last_card == ""). Zones marked "No card" are
            # explicit Claude/YOLO/user verdicts that the zone is
            # empty — routing an orphan rank/suit there caused the
            # 7/27 round-2 phantom 7-of-Hearts that sent Steve a
            # card he never received.
            if not existing:
                target = name
                break
        if target is None:
            log.log(
                f"[VOICE] Can't infer player for '{cmd.rank}{cmd.suit[0]}' — "
                f"every active zone already has a card"
            )
            # Audible cue: dealer said a card we can't place. Usually
            # means Whisper dropped a player name earlier and the
            # subsequent orphan has no home. Short "orphan card" lets
            # them know without staring at the log.
            speech.say("Orphan card")
            return
        log.log(
            f"[VOICE] Inferred {target} for '{cmd.rank}{cmd.suit[0]}' "
            f"(next in deal order)"
        )
        # Immediate warning so the dealer catches the orphan before
        # the debounced readback fires. The readback will then speak
        # "Inferred: <target>, <rank> of <suit>" (the Inferred prefix
        # is applied via the _voice_inferred_this_round tracker).
        speech.say("Orphan card")
        if not hasattr(s, "_voice_inferred_this_round"):
            s._voice_inferred_this_round = set()
        s._voice_inferred_this_round.add(target)
        _voice_post("/api/console/correct", {
            "corrections": [{"player": target, "rank": cmd.rank, "suit": cmd.suit}],
        })
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

    if isinstance(cmd, ScanCardsCommand):
        # Manual scan trigger. Allow during any phase that has a
        # game running and zones to scan — the dealer might want
        # to re-scan during up_round (initial scan) or pre_confirm
        # (re-scan after a missing-card prompt).
        if phase not in ("up_round", "pre_confirm"):
            log.log(f"[VOICE] Ignoring 'Scan cards' in phase {phase}")
            return
        log.log("[VOICE] Scan cards → /api/console/force_scan")
        _voice_post("/api/console/force_scan")
        return

    if isinstance(cmd, FoldCommand):
        if phase != "pre_pot":
            log.log(f"[VOICE] Ignoring '{cmd.player} folds' in phase {phase}")
            return
        log.log(f"[VOICE] {cmd.player} folds → /api/table/fold")
        _voice_post("/api/table/fold", {"player": cmd.player, "folded": True})
        return

    if isinstance(cmd, (PassCommand, GoOutCommand)):
        # Challenge votes are now dealer-driven via console buttons;
        # voice is too unreliable to drive state. Log the utterance
        # so we can review what Whisper heard but don't act on it.
        log.log(
            f"[VOICE] {type(cmd).__name__} from voice ignored — "
            f"use console Pass/Out buttons instead "
            f"(heard: {cmd.raw_text!r})"
        )
        return

    if isinstance(cmd, ChallengeWinnerCommand):
        if phase != "challenge_resolve":
            log.log(
                f"[VOICE] Ignoring '{cmd.player} wins' in phase {phase}"
            )
            return
        _handle_challenge_winner(s, cmd.player)
        return

    if isinstance(cmd, UnrecognizedCommand):
        log.log(f"[VOICE] ? {cmd.raw_text!r}")
        return


# ---------------------------------------------------------------------------
# Background capture
# ---------------------------------------------------------------------------

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

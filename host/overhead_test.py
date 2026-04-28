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

from log_buffer import log

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
# Bench / training-data harnesses — see test_modes.py. Re-exported for
# http_server.py.
# ---------------------------------------------------------------------------

from test_modes import (  # noqa: E402,F401
    GAME_PATTERNS,
    _advance_to_next_up,
    _collect_advance,
    _collect_auto_cycle,
    _collect_deal_info,
    _collect_mode_json,
    _collect_redo,
    _collect_scan,
    _collect_start_clear,
    _collect_start_deal,
    _collect_start_first,
    _deal_check_dealer_zone,
    _deal_check_zones_clear,
    _deal_mode_json,
    _deal_retry_missing,
    _deal_scan_all_zones,
    _process_deal_text,
    _set_deal_game,
    _start_collect_mode,
    _start_deal_mode,
    _stop_collect_mode,
    _stop_deal_mode,
)


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


# ---------------------------------------------------------------------------
# Launcher delegation — see main.py. Keeps `python3 overhead_test.py` as
# the canonical entry point so existing launch scripts and the user's
# `--listen` muscle memory keep working.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from main import main
    main()

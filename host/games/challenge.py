"""Challenge-game state machine (High, Low, High and its siblings).

Procedural module — the Challenge templates currently fall through
to ``BaseGame``, so this is a flat collection of free functions
operating on AppState rather than a subclass. The dealer drives
each round via console Pass/Out buttons; Rodney drives his own
votes via /table buttons; the engine just records the votes and
advances the state machine.

The four small phrasing helpers (``_fmt_money``,
``_format_name_list``, ``_log_and_speak``, ``_bump_table_version``)
moved with this module because every caller is inside the
challenge flow. ``_guided_replace_loop`` and
``_start_guided_deal_range`` live in ``overhead_test.py`` and are
imported lazily inside ``_start_next_challenge_round`` so we can
keep this module on the leaf side of the import graph.
"""

from threading import Thread

from log_buffer import log
from speech import speech
from host_constants import PLAYER_NAMES
from pi_scanner import _pi_flash, _pi_slot_led


# After the first round of a Challenge hand, every subsequent
# round's ante (rounds 2, 3, and any post-reshuffle rounds) is a
# flat 50c per player regardless of what the dealer picked on the
# Start Game form. Only the very first round honours the dropdown
# value — that's the "buy in" ante; the rest are "keep playing"
# antes that feed the pot at a fixed rate.
CHALLENGE_SUBSEQUENT_ANTE_CENTS = 50

MAX_PASSES_PER_ROUND = 2


# ---------------------------------------------------------------------------
# Small phrasing / state helpers used throughout the module
# ---------------------------------------------------------------------------


def _fmt_money(cents: int) -> str:
    """$1.50 from 150 cents."""
    dollars = cents // 100
    rem = cents % 100
    return f"${dollars}.{rem:02d}"


def _format_name_list(names) -> str:
    """'Joe' / 'Bill and David' / 'Bill, David and Joe' for readback."""
    names = list(names)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _log_and_speak(s, msg: str):
    """Log a [CHALLENGE] line and say the same string."""
    log.log(f"[CHALLENGE] {msg}")
    try:
        speech.say(msg)
    except Exception:
        pass


def _bump_table_version(s):
    with s.table_lock:
        s.table_state_version += 1


# ---------------------------------------------------------------------------
# Round / phase introspection
# ---------------------------------------------------------------------------


def _game_is_challenge(ge) -> bool:
    """True if the current game has at least one CHALLENGE phase."""
    try:
        from game_engine import PhaseType
        if ge.current_game is None:
            return False
        return any(ph.type == PhaseType.CHALLENGE for ph in ge.current_game.phases)
    except Exception:
        return False


def _challenge_can_mark(s, ge) -> bool:
    """True when Rodney can mark cards toward a Go Out. Rounds 1-2
    only (round 3 takes all 7 cards, no selection). Disabled once
    Rodney has committed (went_out) or locked in a pass.

    Marking is also allowed during the guided deal for rounds 2/3
    so Rodney can pre-select cards while the dealer is still laying
    down physical cards — the selection carries into the vote."""
    if not _game_is_challenge(ge):
        return False
    if s.challenge_round_index is None:
        return False
    if s.console_state not in ("challenge_vote", "dealing"):
        return False
    if s.rodney_out_slots:
        return False
    if (s.challenge_round_index or 0) >= 2:
        return False
    rodney_state = s.challenge_per_player.get("Rodney") or {}
    if rodney_state.get("went_out"):
        return False
    return True


def _challenge_required_cards(s) -> int:
    """How many cards Rodney must select to Go Out in the current round.
    Returns 2/3/5 based on the challenge_round_index, by looking at the
    game engine's CHALLENGE phases in order. 0 if unavailable."""
    ge = s.game_engine
    try:
        from game_engine import PhaseType
        if ge.current_game is None or s.challenge_round_index is None:
            return 0
        idx = 0
        for ph in ge.current_game.phases:
            if ph.type == PhaseType.CHALLENGE:
                if idx == s.challenge_round_index:
                    return int(getattr(ph, "select_cards", 0) or 0)
                idx += 1
    except Exception:
        pass
    return 0


def _challenge_phase_label(s) -> str:
    """Return the current round's CHALLENGE phase label with params
    substituted (e.g. 'Best 2-card High hand'), or a fallback that
    uses _challenge_required_cards + the round's letter label."""
    ge = s.game_engine
    try:
        from game_engine import PhaseType
        if ge.current_game is None or s.challenge_round_index is None:
            return ""
        idx = 0
        for ph in ge.current_game.phases:
            if ph.type == PhaseType.CHALLENGE:
                if idx == s.challenge_round_index:
                    return str(getattr(ph, "label", "") or "")
                idx += 1
    except Exception:
        pass
    return f"Best {_challenge_required_cards(s)}-card hand"


def _challenge_first_voter(s) -> str:
    """Name of the player who votes first this round. Rotates through
    the active players in dealer-left order: dealer's left kicks off
    round 1, the next seat starts round 2, etc. Position is computed
    from total rounds played this hand (including reshuffle loops):
    shuffle_count * 3 + challenge_round_index, modulo the active-
    player count."""
    try:
        ge = s.game_engine
        dealer = ge.get_dealer().name
        active = list(s.console_active_players)
        if not active:
            return ""
        # Build deal-order list: dealer's left first, dealer last,
        # filtered to active players.
        d_idx = PLAYER_NAMES.index(dealer) if dealer in PLAYER_NAMES else 0
        rotated = []
        for offset in range(1, len(PLAYER_NAMES) + 1):
            cand = PLAYER_NAMES[(d_idx + offset) % len(PLAYER_NAMES)]
            if cand in active:
                rotated.append(cand)
        if not rotated:
            return active[0]
        total_rounds = (
            (s.challenge_shuffle_count or 0) * 3
            + (s.challenge_round_index or 0)
        )
        return rotated[total_rounds % len(rotated)]
    except Exception:
        return ""


def _challenge_ante_cents_for(s, shuffle_count: int, round_index: int) -> int:
    """How much each player antes for a given Challenge round. The
    very first round of the hand (no reshuffles yet, round 0) uses
    the dealer-selected ante from Start Game; everything after is a
    flat CHALLENGE_SUBSEQUENT_ANTE_CENTS."""
    if shuffle_count == 0 and round_index == 0:
        return s.ante_cents
    return CHALLENGE_SUBSEQUENT_ANTE_CENTS


# ---------------------------------------------------------------------------
# Round transitions
# ---------------------------------------------------------------------------


def _begin_challenge_vote(s):
    """Enter the challenge_vote state after a round's deal/replace
    completes. Dealer drives the round via console Pass/Out buttons
    and the "End Round" button; Rodney uses his /table buttons.

    Speaks an opening announcement that names the round's hand type
    (e.g. "Best 2-card High hand") and the first player to vote (the
    dealer's left) so everyone at the table knows whose turn it is.

    Does NOT call _reset_round_passes — the reset happens before the
    deal in _start_next_challenge_round (round 2+) or at hand-start
    in /api/console/deal (round 1). Resetting here would wipe any
    mid-deal commit Rodney made via his /table before the deal fully
    finished."""
    s.console_state = "challenge_vote"
    round_label = (s.challenge_round_index or 0) + 1
    hand_label = _challenge_phase_label(s) or "challenge hand"
    first_voter = _challenge_first_voter(s)
    opener = f"Challenge round. {hand_label}."
    if first_voter:
        opener += f" {first_voter}, you are first."
    _log_and_speak(s, opener)
    log.log(f"[CHALLENGE] Round {round_label} vote begins — "
            "use console Pass/Out buttons")
    _bump_table_version(s)


def _clear_rodney_challenge_leds(s):
    """Turn off any green slot LEDs lit for Rodney's current-round
    commit and clear rodney_out_slots. Safe to call from any
    challenge-state transition — no-op when nothing's lit."""
    for slot in list(s.rodney_out_slots or []):
        try:
            _pi_slot_led(s, int(slot), "off")
        except Exception:
            pass
    s.rodney_out_slots = []


def _reset_round_passes(s):
    """Start-of-round reset: every player starts each round fresh.
    Pass counter → 0, went_out → False, committed out_slots cleared.
    Rodney's committed-slot LEDs are turned off (he can pick a new
    set of cards from his now-larger hand in the next round)."""
    _clear_rodney_challenge_leds(s)
    for st in s.challenge_per_player.values():
        st["passes"] = 0
        st["went_out"] = False
        st["out_round"] = None
        st["out_slots"] = []


def _set_challenge_vote(s, name: str, vote: str) -> tuple:
    """Manual per-player vote setter — called by dealer buttons on the
    console and by Rodney's /table buttons. vote ∈ {"pass", "out",
    "clear"}.

    Pass semantics: each Pass click increments the counter up to 2.
    Once a player hits 2 passes they're locked for the round — no
    further passes or outs. Clear resets the counter to 0.

    For Rodney + "out", validates his card selection (rounds 1-2 need
    exact match of _challenge_required_cards; round 3 takes all 7)
    and lights the LEDs on his committed slots.

    Returns (ok: bool, error: str).
    """
    ge = s.game_engine
    in_challenge_deal = (
        s.console_state == "dealing"
        and _game_is_challenge(ge)
        and s.challenge_round_index is not None
    )
    if s.console_state != "challenge_vote" and not in_challenge_deal:
        return False, "not in challenge vote"
    st = s.challenge_per_player.get(name)
    if st is None:
        return False, f"unknown player {name}"
    if st.get("went_out") and vote != "clear":
        return False, f"{name} already out for this hand"
    passes = int(st.get("passes", 0))
    locked = (passes >= MAX_PASSES_PER_ROUND)
    if locked and vote != "clear":
        return False, (
            f"{name} already passed {MAX_PASSES_PER_ROUND} times this round "
            f"— locked until next round"
        )
    if vote == "clear":
        st["passes"] = 0
        _log_and_speak(s, f"{name} pass counter cleared")
        _bump_table_version(s)
        return True, ""
    if vote == "pass":
        st["passes"] = passes + 1
        if name == "Rodney":
            with s.table_lock:
                s.rodney_marked_slots = set()
        _log_and_speak(s,
            f"{name} passes ({st['passes']} of {MAX_PASSES_PER_ROUND})")
        _bump_table_version(s)
        return True, ""
    if vote == "out":
        round_idx = s.challenge_round_index or 0
        if name == "Rodney":
            if round_idx != 2:
                required = _challenge_required_cards(s)
                marks = sorted(s.rodney_marked_slots)
                if len(marks) != required:
                    return False, (
                        f"Rodney must select exactly {required} cards "
                        f"before going out"
                    )
                st["out_slots"] = marks
                s.rodney_out_slots = list(marks)
                for slot in marks:
                    _pi_slot_led(s, slot, "on")
                with s.table_lock:
                    s.rodney_marked_slots = set()
            else:
                slots = [1, 2, 3, 4, 5]
                st["out_slots"] = slots
                s.rodney_out_slots = list(slots)
                for slot in slots:
                    _pi_slot_led(s, slot, "on")
                with s.table_lock:
                    s.rodney_marked_slots = set()
        st["went_out"] = True
        st["out_round"] = round_idx
        _log_and_speak(s, f"{name} is out")
        _bump_table_version(s)
        return True, ""
    return False, f"unknown vote kind {vote!r}"


def _resolve_challenge_round(s):
    """End-of-round-two-laps resolve."""
    outs = [nm for nm, st in s.challenge_per_player.items() if st["went_out"]]
    if len(outs) == 0:
        next_idx = (s.challenge_round_index or 0) + 1
        if next_idx >= 3:
            # End of shuffle cycle with no outs — reshuffle and redeal.
            _clear_rodney_challenge_leds(s)
            s.console_state = "reshuffle"
            _log_and_speak(s, "No one went out. Reshuffle and redeal.")
            _bump_table_version(s)
            return
        s.challenge_round_index = next_idx
        _start_next_challenge_round(s)
        return
    if len(outs) == 1:
        # 1-out-all-pass wins the pot; hand ends.
        winner = outs[0]
        amount = s.pot_cents
        _clear_rodney_challenge_leds(s)
        _log_and_speak(s, f"{winner} wins the pot: {_fmt_money(amount)}.")
        s.pot_cents = 0
        s.console_state = "hand_over"
        _bump_table_version(s)
        return
    # 2+ out — dealer announces winner in challenge_resolve. Rodney's
    # LEDs stay on through the compare so the dealer can see which
    # cards he committed; they get cleared by _handle_challenge_winner
    # when the compare settles.
    s.console_state = "challenge_resolve"
    _log_and_speak(s, f"Challenge between {', '.join(outs)}. Call the winner.")
    _bump_table_version(s)


def _start_next_challenge_round(s):
    """Auto-ante + deal the next round's cards. Assumes challenge_round_index
    is already set to the round we're entering (1 or 2 — round 0 is handled
    by new-hand init).

    Every active player antes every round (only folding would exempt
    someone and that isn't part of this game). The per-player
    pass/out state reset happens inside _reset_round_passes — players
    who committed in the previous round get to re-vote this round
    with the new cards added to their hand.

    Only the very first round of the hand honours the dealer-selected
    ante; every subsequent round (rounds 2+, reshuffled rounds) uses
    the flat CHALLENGE_SUBSEQUENT_ANTE_CENTS."""
    # Lazy import: _start_guided_deal_range and _guided_replace_loop
    # live in overhead_test.py. Importing here at runtime keeps the
    # module on the leaf side of the import graph (overhead_test
    # imports challenge, not the other way around).
    from guided_deal import _guided_replace_loop, _start_guided_deal_range

    per_player_cents = _challenge_ante_cents_for(
        s, s.challenge_shuffle_count or 0, s.challenge_round_index or 0,
    )
    n = len(s.console_active_players)
    ante_cents = per_player_cents * n
    s.pot_cents += ante_cents
    _log_and_speak(s,
        f"Round {s.challenge_round_index + 1} ante: "
        f"{_fmt_money(per_player_cents)} each. "
        f"Pot is now {_fmt_money(s.pot_cents)}.")
    _reset_round_passes(s)
    if s.challenge_round_index == 1:
        _start_guided_deal_range(s, [4, 5])
    elif s.challenge_round_index == 2:
        # Round 3: slots 4 and 5 get physically replaced. Save the old
        # cards as rodney_overflow (for the resolve UI), pop them from
        # rodney_downs + pi_prev_slots, and set s.guided_deal all in
        # ONE lock window so the Pi poll loop can't re-populate slots
        # 4/5 with stale scan data between the pop and the replace
        # loop taking ownership.
        prev = {}
        overflow = []
        ordered = [4, 5]
        if s.guided_deal is None:
            with s.table_lock:
                for slot in ordered:
                    code = s.pi_prev_slots.get(slot, "")
                    if code:
                        prev[slot] = code
                    card = s.rodney_downs.get(slot)
                    if card:
                        overflow.append({"slot": slot, "card": dict(card)})
                    s.rodney_downs.pop(slot, None)
                    s.pi_prev_slots.pop(slot, None)
                    s.slot_pending.pop(slot, None)
                s.rodney_overflow = overflow
                s.guided_deal = {
                    "slots": list(ordered), "index": 0, "mode": "replace",
                    "previous_cards": dict(prev),
                }
                s.console_state = "replacing"
                s.table_state_version += 1
            _pi_flash(s, True)
            t = Thread(target=_guided_replace_loop, args=(s,), daemon=True)
            s.guided_deal_thread = t
            t.start()


def _handle_challenge_winner(s, winner_name: str) -> bool:
    """Dealer-announced winner for a 2+ compare. Each loser pays the
    winner the current pot amount as a side payment; the pot itself
    stays on the table and the hand continues — only a 1-out-all-
    pass actually awards (and zeros) the pot.

    In rounds 1-2 that means advancing into the next round. Round 3
    always ends the shuffle cycle: 2+ compare there triggers a
    reshuffle so the hand keeps going toward a single winner.

    Returns True on success."""
    if s.console_state != "challenge_resolve":
        return False
    outs = [nm for nm, st in s.challenge_per_player.items() if st["went_out"]]
    if winner_name not in outs:
        _log_and_speak(s, f"{winner_name} was not out. Try again.")
        return False
    losers = [nm for nm in outs if nm != winner_name]
    per_loser = s.pot_cents
    verb = "pays" if len(losers) == 1 else "pay"
    _log_and_speak(s,
        f"{winner_name} wins. "
        f"{_format_name_list(losers)} {verb} {winner_name} "
        f"{_fmt_money(per_loser)}.")
    _clear_rodney_challenge_leds(s)
    next_idx = (s.challenge_round_index or 0) + 1
    if next_idx >= 3:
        # End of the shuffle cycle with the pot still unawarded —
        # reshuffle and redeal.
        s.console_state = "reshuffle"
        _log_and_speak(s, "Deck exhausted. Reshuffle and redeal.")
        _bump_table_version(s)
        return True
    # Advance into the next round — everyone's passes/out reset inside
    # _start_next_challenge_round so previously-committed players
    # rejoin the vote with the new cards added to their hand.
    s.challenge_round_index = next_idx
    _start_next_challenge_round(s)
    return True

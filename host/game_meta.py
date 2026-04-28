"""Read-only helpers that interrogate the current game template.

These walk ``ge.current_game.phases`` to answer questions like
"how many up cards total?" / "what's the next deal position?" /
"is this a draw game?" / "how many cards can Rodney replace in
the second draw of 3 Toed Pete?". No side effects, no AppState
mutation — just template introspection.

Plus the betting-limit + ante phrasing tables that the dealer's
"Start game" announcement reads from.
"""

from host_constants import PLAYER_NAMES


# ---------------------------------------------------------------------------
# Phase walking — total cards / position introspection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Draw-phase helpers (5 Card Draw, 3 Toed Pete, etc.)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Betting-limit / ante phrasing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Deal-order rotation
# ---------------------------------------------------------------------------


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

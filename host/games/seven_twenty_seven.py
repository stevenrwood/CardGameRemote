"""
7/27 — a high-low split where each player picks how many cards to
take, with hand value closest to 7 (low half) or 27 (high half)
winning. Aces count as 1 or 11; face cards count as 0.5.

The game class owns:
- per-player freeze tracking (three consecutive passes → frozen)
- per-hand-value scoring (half-integers and multiple ace totals)
- the flip-choice prompt for the 2-down variant
- the ``values_7_27`` decoration fed to /table for UI rendering

Both ``7/27`` (two-down + pick-one-to-flip) and ``7/27 (one up)``
share this class; the flip-choice method is the only place the two
variants diverge.
"""

from __future__ import annotations

from log_buffer import log
from speech import speech

from . import BaseGame, ScoreResult, register


# Face cards and 10 contribute either their point value or 0.5 (for
# J/Q/K — the 7/27 convention). Ace is handled separately because it
# can be 1 OR 11, so a player with N aces has N+1 candidate totals.
_RANK_TO_VALUE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "J": 0.5, "Q": 0.5, "K": 0.5,
}


def compute_values(cards):
    """Given an iterable of (rank, suit) tuples, return the sorted
    list of possible 7/27 totals ≤ 27 (one per ace assignment).

    Only totals that stay ≤27 count — once a hand is over 27 with
    aces at 1, that assignment doesn't qualify. The empty result
    means the player has already busted.
    """
    base = 0.0
    aces = 0
    for rank, _suit in cards:
        if rank == "A":
            aces += 1
            continue
        v = _RANK_TO_VALUE.get(rank)
        if v is None:
            continue
        base += v
    values = set()
    for k in range(aces + 1):
        total = base + k * 11 + (aces - k) * 1
        if total <= 27:
            # Collapse 17.0 → 17 so speech says "17" not "17.0".
            values.add(total if total != int(total) else int(total))
    return sorted(values)


def _speak_value(v):
    """Render a 7/27 numeric value for speech. Half-integers become '… and a half'."""
    if isinstance(v, int) or v == int(v):
        return str(int(v))
    whole = int(v)
    frac = v - whole
    if abs(frac - 0.5) < 1e-6:
        return "a half" if whole == 0 else f"{whole} and a half"
    return f"{v:g}"


def _format_values_phrase(values):
    """Turn [2, 12, 22] into '2, 12, or 22' for speech, preserving half-speak."""
    strs = [_speak_value(v) for v in values]
    if len(strs) == 1:
        return strs[0]
    if len(strs) == 2:
        return f"{strs[0]} or {strs[1]}"
    return ", ".join(strs[:-1]) + f", or {strs[-1]}"


@register
class SevenTwentySevenGame(BaseGame):
    def __init__(self, template, engine):
        super().__init__(template, engine)
        # player_name -> consecutive freezes. A player with 3 is frozen
        # for the remainder of the hand (cannot take more cards).
        self.freezes: dict[str, int] = {}

    # --- lifecycle ---

    def on_hand_start(self, state) -> None:
        # Initialize every active player at zero freezes. Share the
        # same dict with ``state.freezes`` so legacy call sites that
        # still read from AppState see the same data — tech debt that
        # later steps can scrub.
        self.freezes = {name: 0 for name in state.console_active_players}
        state.freezes = self.freezes

    # --- per-round hook (state + speech) ---

    def on_round_confirmed(
        self, state, round_cards, *, announce: bool = True
    ) -> None:
        # Freeze tracking only kicks in starting round 2: round 1 is
        # the initial deal where every player must accept a card.
        round_num = state.console_up_round + 1
        if round_num > 1:
            took = {c["player"] for c in round_cards}
            newly_frozen = []
            for name in state.console_active_players:
                if self.freezes.get(name, 0) >= 3:
                    continue  # already frozen
                if name in took:
                    self.freezes[name] = 0
                else:
                    self.freezes[name] = self.freezes.get(name, 0) + 1
                    if self.freezes[name] >= 3:
                        newly_frozen.append(name)
            for name in newly_frozen:
                log.log(f"[7/27] {name} is frozen")
                speech.say(f"{name} is frozen")
        # Let the base class speak the per-player totals + bet first
        # via score_hand / _announce_round.
        super().on_round_confirmed(state, round_cards, announce=announce)

    # --- scoring ---

    def score_hand(self, cards, wild_ranks):
        if not cards:
            return None
        values = compute_values(cards)
        if not values:
            return None
        best = max(values)
        ordered = sorted(set(values), reverse=True)
        tail = (_speak_value(ordered[0]) if len(ordered) == 1
                else _format_values_phrase(ordered))
        return ScoreResult(
            value=best,
            speech=f"your bet with high of {tail}",
        )

    def _player_visible_cards(self, state, player_name):
        """Rodney's score includes his known down cards (minus any
        flipped one). Everyone else uses the default ``console_hand_cards``
        view."""
        cards = super()._player_visible_cards(state, player_name)
        remote = next(
            (p for p in self.engine.players if p.is_remote), None
        )
        if remote is None or player_name != remote.name:
            return cards
        flipped_slot = (state.rodney_flipped_up or {}).get("slot")
        for slot_num, d in state.rodney_downs.items():
            if slot_num == flipped_slot:
                continue
            if d.get("rank") and d.get("suit"):
                cards.append((d["rank"], d["suit"]))
        return cards

    # --- scan policy ---

    def is_frozen(self, state, player_name: str) -> bool:
        return self.freezes.get(player_name, 0) >= 3

    # --- table decorations ---

    def decorate_table_players(self, entries, state) -> None:
        """Add ``values_7_27`` (list of possible totals ≤27) to each
        player entry. Uses the same per-player card view as scoring
        but exposes every candidate total rather than just the max."""
        for entry in entries:
            pairs = self._player_visible_cards(state, entry["name"])
            if not pairs:
                continue
            values = compute_values(pairs)
            if values:
                entry["values_7_27"] = values

    def flip_choice(self, state):
        """2-down variant only: prompt Rodney to pick which of his two
        down cards to flip face-up. Returns None for ``7/27 (one up)``
        (dealer dealt the up card directly) and once a flip has been
        resolved."""
        if self.template.name != "7/27":
            return None
        if state.rodney_flipped_up is not None:
            return None
        if len(state.rodney_downs) != 2:
            return None
        downs_sorted = sorted(state.rodney_downs.items())
        return {
            "prompt": "Pick a card to turn face-up",
            "options": [
                {"slot": sn, "rank": d["rank"], "suit": d["suit"]}
                for sn, d in downs_sorted
            ],
        }

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


def compute_values(cards, max_total: float = 27):
    """Given an iterable of (rank, suit) tuples, return the sorted
    list of possible 7/27 totals ≤ max_total (one per ace assignment).

    Only totals that stay within the ceiling count — assignments that
    exceed it don't qualify. An empty result means the player has
    busted relative to the ceiling used.

    ``max_total`` defaults to 27 (the natural high-hand cap for a
    complete hand). Callers scoring VISIBLE up-cards only should pass
    26.5 — any higher visible total can't reach a legal ≤27 hand
    because the down card contributes at least 0.5 (face card), so
    an interpretation that already exceeds 26.5 visible is impossible
    to recover from and shouldn't be announced.
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
        if total <= max_total:
            # Collapse 17.0 → 17 so speech says "17" not "17.0".
            values.add(total if total != int(total) else int(total))
    return sorted(values)


# Visible-only ceiling: any higher total + min down card (0.5) exceeds 27.
VISIBLE_MAX = 26.5


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
        # Bust detection: any player whose visible up-cards exceed 26.5
        # can't reach a legal ≤27 final hand no matter what their down
        # card is. Auto-fold them and announce so play can continue
        # without waiting for them to stand.
        self._check_busts(state, announce=announce)
        # Let the base class speak the per-player totals + bet first
        # via score_hand / _announce_round. Busted players are now in
        # folded_players so the base class skips them.
        super().on_round_confirmed(state, round_cards, announce=announce)

    def _check_busts(self, state, announce: bool = True) -> None:
        """Fold any player whose visible up-cards already exceed the
        26.5 bust ceiling. Rodney is excluded — his real hand total is
        evaluated against the normal 27 ceiling because we know his
        down cards; a bust there is checked by downstream code that
        has access to the full hand."""
        remote = next(
            (p for p in self.engine.players if p.is_remote), None
        )
        remote_name = remote.name if remote else None
        for name in list(state.console_active_players):
            if name == remote_name:
                continue
            if name in state.folded_players:
                continue
            cards = self._player_visible_cards(state, name)
            if not cards:
                continue
            # Bust iff every possible ace assignment leaves the
            # visible total above 26.5.
            values = compute_values(cards, max_total=VISIBLE_MAX)
            if values:
                continue
            state.folded_players.add(name)
            log.log(f"[7/27] {name} busted out — auto-folded")
            if announce:
                speech.say(
                    f"{name}, you busted out with more than "
                    f"{_speak_value(VISIBLE_MAX)} showing"
                )

    # --- scoring ---

    def score_hand(self, cards, wild_ranks):
        # Visible-only scoring: cap at 26.5. An ace-as-11 interpretation
        # (or any interpretation) that puts visible cards above that is
        # an impossible-to-win hand regardless of the down card, so
        # don't announce it as their "high".
        if not cards:
            return None
        values = compute_values(cards, max_total=VISIBLE_MAX)
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

    # NOTE: no _player_visible_cards override. The bet-first
    # announcement compares VISIBLE up-cards only — the same view every
    # player at the table sees. Rodney's down cards are known to the
    # host (they came through the Pi scanner) but must NOT leak into
    # the public comparison. His personal values_7_27 total — which
    # includes his down cards — is built separately in
    # decorate_table_players below so Rodney's own UI sees his real
    # hand total, but nobody else gets spoilers.

    def _rodney_all_cards(self, state):
        """Rodney's complete hand (up + his known downs, minus any
        flipped slot already counted via console_hand_cards). For
        use inside decorate_table_players when filling Rodney's
        personal values_7_27 — never for public scoring."""
        remote = next(
            (p for p in self.engine.players if p.is_remote), None
        )
        if remote is None:
            return []
        cards = super()._player_visible_cards(state, remote.name)
        flipped_slot = (state.rodney_flipped_up or {}).get("slot")
        for slot_num, d in state.rodney_downs.items():
            if slot_num == flipped_slot:
                continue
            if d.get("rank") and d.get("suit"):
                cards.append((d["rank"], d["suit"]))
        return cards

    # --- scan policy ---

    def zones_to_scan(self, state):
        # Round 1 is the initial flip/deal — everyone must place a card
        # and standing isn't allowed yet. Rounds 2+ are hit rounds: any
        # non-frozen active player may take or stand.
        round_num = state.console_up_round + 1
        if round_num <= 1:
            return list(state.console_active_players), False
        active = [
            n for n in state.console_active_players
            if self.freezes.get(n, 0) < 3
        ]
        return active, True

    # --- table decorations ---

    def decorate_table_players(self, entries, state) -> None:
        """Add ``values_7_27`` (list of possible totals ≤ the relevant
        ceiling) to each player entry.

        Rodney's entry uses his full private hand (up + known downs)
        with the natural 27 ceiling because this is HIS display and he
        already knows his downs — the total shown is a real candidate
        final hand value.

        Every other player's entry uses the VISIBLE-only view and the
        26.5 ceiling: anything higher visible can't reach a legal ≤27
        final hand once you add the mandatory down card, so it's a
        busted interpretation that shouldn't clutter the UI."""
        remote = next(
            (p for p in self.engine.players if p.is_remote), None
        )
        remote_name = remote.name if remote else None
        for entry in entries:
            if entry["name"] == remote_name:
                pairs = self._rodney_all_cards(state)
                values = compute_values(pairs) if pairs else []
            else:
                pairs = self._player_visible_cards(state, entry["name"])
                values = (
                    compute_values(pairs, max_total=VISIBLE_MAX)
                    if pairs else []
                )
            if values:
                entry["values_7_27"] = values

    def decorate_table_state(self, doc, state) -> None:
        """2-down variant only: once Rodney has both down cards scanned
        and validated (``len(rodney_downs) == 2``), inject the flip-
        choice prompt into the /table state doc. Cleared automatically
        as soon as ``rodney_flipped_up`` is set. No-op for the
        ``(one up)`` variant (dealer dealt a face-up directly)."""
        if self.template.name != "7/27":
            return
        if state.rodney_flipped_up is not None:
            return
        if len(state.rodney_downs) != 2:
            return
        downs_sorted = sorted(state.rodney_downs.items())
        doc["flip_choice"] = {
            "prompt": "Pick a card to turn face-up",
            "options": [
                {"slot": sn, "rank": d["rank"], "suit": d["suit"]}
                for sn, d in downs_sorted
            ],
        }

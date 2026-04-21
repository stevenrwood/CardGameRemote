"""
Per-game classes plugged into the host.

Each GameTemplate can point at a class name via its ``class_name``
field. At hand-start the host instantiates that class and stashes it
on ``AppState.current_game_impl``; the HTTP handler and bg loop call
into the instance for game-specific behavior (wild tracking, freeze
counting, split-pot scoring, etc.). Templates with an empty
``class_name`` fall back to :class:`BaseGame`, which is a bag of
no-ops — games that only differ from a base stud/draw template in
data (wild_ranks, dynamic_wild, with_params) don't need a class at
all.

The interface is intentionally narrow: a single per-round hook
(``on_round_confirmed``) for state updates, and a ``score_hand``
primitive that the base class uses to walk each active player,
rank them, and speak the bet-first announcement. Games that need
non-default speech (7/27's numeric totals, stud's "is high. Your
bet.") return a fully-rendered phrase from ``score_hand``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from log_buffer import log
from speech import speech


@dataclass
class ScoreResult:
    """What a game's ``score_hand`` returns for one player's cards.

    - ``value`` is the comparable score (higher wins). Used by the
      base class to pick the bet-first player.
    - ``speech`` is the fully-rendered phrase the announcer appends
      after the player's name, e.g. ``"Three Jacks is high. Your
      bet."`` or ``"your bet with high of 16 or 26"``. Games own
      their sentence shape since templates differ per family.
    """
    value: float
    speech: str


# Short rank tokens used when parsing "Rank of Suit" strings from
# console_hand_cards back into (rank, suit) pairs. Games that need
# these can import from here rather than redefining per-file.
RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}


def parse_hand_card(text: str):
    """Split a ``"Rank of Suit"`` string into a ``(rank_short, suit_lc)``
    tuple, or return None if the text doesn't match."""
    parts = text.split(" of ")
    if len(parts) != 2:
        return None
    rank_full, suit_full = parts[0].strip(), parts[1].strip().lower()
    return RANK_SHORT.get(rank_full, rank_full), suit_full


class BaseGame:
    """Default behavior: nothing game-specific, plus a generic
    bet-first announcer that drives off ``score_hand``.

    Subclasses:
      - override ``score_hand`` to score a player's visible cards
        (return ``None`` for games that don't announce a round
        winner — all-down games, Challenge, etc.)
      - override ``on_round_confirmed`` to advance internal state
        (freezes, dynamic wilds) and call ``super().on_round_confirmed``
        so the base class's speech still fires
      - override ``_player_visible_cards`` only if the game
        includes more than hand-history up cards for scoring (7/27
        adds Rodney's known down cards to his own total)
    """

    def __init__(self, template, engine):
        self.template = template
        self.engine = engine

    # --- per-hand lifecycle ---

    def on_hand_start(self, state) -> None:
        """Called once after ``GameEngine.new_hand`` returns. Reset
        any per-hand state that belongs on the game instance."""
        pass

    def on_hand_end(self, state) -> None:
        """Called at hand-over, before the engine rotates the dealer.
        Most games need nothing here."""
        pass

    # --- the one per-round hook ---

    def on_round_confirmed(
        self, state, round_cards: list, *, announce: bool = True
    ) -> None:
        """Called from ``/api/console/confirm`` after the round's
        up-cards are committed to ``console_hand_cards``.

        Subclasses update internal state (freezes, wilds) and then
        call super() so the bet-first speech fires. ``announce=False``
        is the stud/FTQ "defer past trailing down" case: state updates
        still happen, but speech is held until the trailing down deal
        completes (and another hook re-speaks).
        """
        if announce:
            self._announce_round(state)

    # --- scoring + announce (plumbed through on_round_confirmed) ---

    def score_hand(
        self, cards: list, wild_ranks: list
    ) -> Optional[ScoreResult]:
        """Return a ScoreResult for the given (rank, suit) cards, or
        None if no announcement applies (all-down games, Challenge,
        player with no visible cards, 7/27 bust, etc.). Default: None."""
        return None

    def _player_visible_cards(self, state, player_name: str):
        """The (rank, suit) tuples used for scoring this player's
        hand. Default: every card in ``console_hand_cards`` belonging
        to this player. Subclasses (e.g. 7/27) extend this when the
        score includes additional cards the host knows about."""
        out = []
        for entry in state.console_hand_cards:
            if entry.get("player") != player_name:
                continue
            parsed = parse_hand_card(entry.get("card", ""))
            if parsed is not None:
                out.append(parsed)
        return out

    def _announce_round(self, state) -> None:
        """Walk every active non-folded player, ask ``score_hand`` for
        each, log the per-player result, and speak the winner. Base
        class method — subclasses should never need to override this
        directly; customize via ``score_hand`` + ``_player_visible_cards``
        instead."""
        ge = self.engine
        wild_ranks = list(getattr(ge, "wild_ranks", []) or [])
        scored: dict[str, ScoreResult] = {}
        for name in state.console_active_players:
            if name in state.folded_players:
                continue
            cards = self._player_visible_cards(state, name)
            result = self.score_hand(cards, wild_ranks)
            if result is None:
                continue
            scored[name] = result
            log.log(f"[{type(self).__name__}] {name}: {result.speech}")
        if not scored:
            return
        best_name, best_result = max(
            scored.items(), key=lambda kv: kv[1].value
        )
        phrase = f"{best_name}, {best_result.speech}"
        log.log(f"[{type(self).__name__}] Bet first: {phrase}")
        speech.say(phrase)

    # --- zone + scan policy ---

    def zones_to_scan(self, state) -> tuple[list[str], bool]:
        """Return ``(zone_names, stand_allowed)`` for this round.

        - ``zone_names``: player names whose Brio zones the watcher
          should trigger on and include in the batch scan. Frozen /
          out-of-game players are simply omitted from the list.
        - ``stand_allowed``: True if an empty zone is a legitimate
          outcome (the player chose not to take a card — 7/27 hit
          round "stand"). False if every zone in the list is
          expected to land a card (stud/draw deal rounds) — empty
          zones there mean "please adjust, we missed a scan".

        Default: every active player, cards required.
        """
        return list(state.console_active_players), False

    # --- table UI decorations ---

    def decorate_table_players(self, entries: list, state) -> None:
        """Mutate the per-player dicts in ``entries`` in place to add
        game-specific fields (7/27 adds ``values_7_27``). Called from
        ``_build_table_state``."""
        pass

    def decorate_table_state(self, doc: dict, state) -> None:
        """Mutate the full ``/table/state`` document in place to add
        game-level prompts or annotations (7/27 emits the flip-choice
        dialog this way — the base class and stud/draw games have no
        such concept so they no-op)."""
        pass


# Module-level registry. Each real game class inserts itself here via
# the ``register`` helper so GameTemplate.class_name -> class lookup is
# one dict hit with no import-time ordering pitfalls.
GAME_CLASSES: dict[str, type[BaseGame]] = {
    "BaseGame": BaseGame,
}


def register(cls: type[BaseGame]) -> type[BaseGame]:
    """Class decorator that adds a game class to GAME_CLASSES keyed by
    its own __name__. Idempotent so re-imports don't clobber."""
    GAME_CLASSES.setdefault(cls.__name__, cls)
    return cls


def make_game(template, engine) -> BaseGame:
    """Construct the per-template game instance. Empty ``class_name``
    falls back to BaseGame."""
    name = template.class_name or "BaseGame"
    cls = GAME_CLASSES.get(name)
    if cls is None:
        raise ValueError(
            f"Template {template.name!r} points at class_name={name!r}, "
            f"which is not registered. Known: {sorted(GAME_CLASSES)}"
        )
    return cls(template, engine)


# Import concrete game modules here so their @register decorators run
# at package-import time. Keep this block at the bottom so BaseGame and
# register exist before the submodule tries to subclass/decorate.
from . import seven_twenty_seven  # noqa: E402,F401

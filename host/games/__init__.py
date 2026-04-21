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

Everything here is intentionally the minimum needed for step 1 of
the refactor. Later steps add methods as the call sites move in.
"""

from __future__ import annotations

from typing import Optional


class BaseGame:
    """Default behavior: nothing game-specific. Override only the
    hooks where your game actually diverges from the generic stud /
    draw flow. Every method has a safe no-op default so an empty
    subclass stays bootable."""

    def __init__(self, template, engine):
        self.template = template
        self.engine = engine

    # --- per-hand lifecycle ---

    def on_hand_start(self, state) -> None:
        """Called once after ``GameEngine.new_hand`` returns. Reset
        any per-hand state that belongs on the game instance (e.g.
        freeze counters for 7/27)."""
        pass

    def on_hand_end(self, state) -> None:
        """Called once at hand-over, before the engine rotates the
        dealer. Use for cleanup; most games need nothing here."""
        pass

    # --- Confirm Cards hooks ---

    def on_round_confirmed(self, state, round_cards: list) -> None:
        """Called from ``/api/console/confirm`` after a round's
        up-cards are committed to ``console_hand_cards``. 7/27 ticks
        freeze counters here; FTQ updates the wild mapping."""
        pass

    def on_history_corrected(self, state) -> None:
        """Called after a user-correction that rewrites a past entry
        in ``console_hand_cards``. FTQ replays queen-follower logic
        here; most games need nothing."""
        pass

    def announce_round_summary(self, state) -> None:
        """Called after ``on_round_confirmed``. Speaks the end-of-round
        summary (bet-first player, hand values). 7/27 announces numeric
        totals; stud/draw announces best poker hand; Challenge is
        silent."""
        pass

    # --- zone + scan policy ---

    def is_frozen(self, state, player_name: str) -> bool:
        """True if the player is skipped from this round's scan
        because they have opted out (7/27 three-freeze rule)."""
        return False

    # --- table UI decorations ---

    def decorate_table_players(self, entries: list, state) -> None:
        """Mutate the per-player dicts in ``entries`` in place to add
        game-specific fields (7/27 adds ``values_7_27`` + ``freezes``).
        Called from ``_build_table_state``."""
        pass

    def flip_choice(self, state) -> Optional[dict]:
        """Return a ``{slots: [a,b]}`` prompt payload when the game
        is waiting on the remote player to pick which down card to
        flip up (7/27 two-down variant). Return None otherwise."""
        return None


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

from __future__ import annotations
"""
Game state management and template engine.

Manages the current hand state, processes game templates, and determines
what should happen with each card (slot vs table, draw prompts, etc.).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PhaseType(str, Enum):
    DEAL = "deal"
    BETTING = "betting"
    DRAW = "draw"
    CHALLENGE = "challenge"
    HIT_ROUND = "hit_round"
    COMMUNITY = "community"


@dataclass
class Phase:
    type: PhaseType
    pattern: list[str] = field(default_factory=list)  # ["down", "up", ...]
    max_draw: int = 0
    select_cards: int = 0
    label: str = ""
    card_type: str = "up"  # for hit_round


@dataclass
class GameTemplate:
    name: str
    phases: list[Phase] = field(default_factory=list)
    # When set, copy phases from the named template at engine-init time.
    # Mutually exclusive with phases=. Used to declare 7 Card Stud variants
    # (Deuces Wild, Follow the Queen, High Chicago, Eight or Better) that
    # share the stud deal/bet cadence but differ only in wild-card or
    # winning-hand rules.
    phases_from: str = ""
    wild_cards: dict = field(default_factory=dict)  # {"ranks": ["2"], "label": "..."}
    dynamic_wild: str = ""  # e.g., "follow_the_queen"
    repeatable: bool = False
    notes: str = ""


@dataclass
class Slot:
    slot_number: int
    rank: str
    suit: str
    card_type: str  # "down" or "up"
    status: str = "active"  # "active", "discarded", "challenged"

    def to_dict(self) -> dict:
        return {
            "slot_number": self.slot_number,
            "card": {"rank": self.rank, "suit": self.suit},
            "card_type": self.card_type,
            "status": self.status,
        }


class GameState(str, Enum):
    IDLE = "idle"                    # No game in progress
    WAITING_FOR_CARD = "dealing"     # Waiting for next card on scanner
    WAITING_FOR_BETTING = "betting"  # Paused for betting round
    WAITING_FOR_DRAW = "draw"        # Waiting for remote player to discard
    WAITING_FOR_CHALLENGE = "challenge"  # Waiting for challenge selection
    HIT_ROUND = "hit_round"          # Open-ended hit round
    HAND_OVER = "hand_over"          # Hand complete


@dataclass
class Player:
    name: str
    position: int       # 1-5, clockwise seating order
    is_remote: bool = False
    is_dealer: bool = False


# Default player configuration
DEFAULT_PLAYERS = [
    Player(name="Steve", position=1),
    Player(name="Bill", position=2),
    Player(name="David", position=3),
    Player(name="Joe", position=4),
    Player(name="Rodney", position=5, is_remote=True),
]


class GameEngine:
    """Manages game flow driven by templates."""

    def __init__(self, players: list[Player] | None = None):
        self.players = players or list(DEFAULT_PLAYERS)
        self.dealer_index = 0  # index into self.players for current dealer
        self.templates: dict[str, GameTemplate] = {}
        self.current_game: GameTemplate | None = None
        self.state = GameState.IDLE
        self.slots: list[Slot] = []
        self.next_slot = 1
        self.phase_index = 0
        self.card_in_phase = 0  # which card within current deal phase
        self.draw_round = 0
        self.wild_ranks: list[str] = []
        self.wild_label: str = ""
        self.last_up_was_queen = False  # for Follow the Queen tracking
        self._load_default_templates()
        self._update_dealer()

    def _load_default_templates(self):
        """Load the built-in game templates and resolve phases_from refs."""
        self.templates = {t.name: t for t in _default_templates()}
        self._resolve_phases_from()

    def _resolve_phases_from(self):
        """For any template with phases_from set, copy phases from the
        referenced template. Validates that each template has exactly one
        of phases / phases_from. Raises on unknown parent or cycle."""
        for name, t in self.templates.items():
            has_phases = bool(t.phases)
            has_parent = bool(t.phases_from)
            if has_phases and has_parent:
                raise ValueError(
                    f"Template {name!r} sets both phases and phases_from — pick one"
                )
            if not has_phases and not has_parent:
                raise ValueError(
                    f"Template {name!r} has neither phases nor phases_from"
                )
        for name, t in self.templates.items():
            if not t.phases_from:
                continue
            parent = self.templates.get(t.phases_from)
            if parent is None:
                raise ValueError(
                    f"Template {name!r} references unknown parent {t.phases_from!r}"
                )
            if parent.phases_from:
                raise ValueError(
                    f"Template {name!r} parents {parent.name!r} which itself "
                    f"uses phases_from — chained inheritance not supported"
                )
            t.phases = list(parent.phases)

    def _update_dealer(self):
        """Set the is_dealer flag on the current dealer."""
        for p in self.players:
            p.is_dealer = False
        self.players[self.dealer_index].is_dealer = True

    def advance_dealer(self):
        """Rotate dealer to next player clockwise."""
        self.dealer_index = (self.dealer_index + 1) % len(self.players)
        self._update_dealer()

    def get_dealer(self) -> Player:
        """Return the current dealer."""
        return self.players[self.dealer_index]

    def get_remote_player(self) -> Player:
        """Return the remote player."""
        return next(p for p in self.players if p.is_remote)

    def get_player_by_name(self, name: str) -> Player | None:
        """Find a player by name (case-insensitive)."""
        name_lower = name.lower()
        return next((p for p in self.players if p.name.lower() == name_lower), None)

    def get_players_info(self) -> list[dict]:
        """Return player list for UI display."""
        return [
            {
                "name": p.name,
                "position": p.position,
                "is_remote": p.is_remote,
                "is_dealer": p.is_dealer,
            }
            for p in self.players
        ]

    def get_game_list(self) -> list[str]:
        """Return list of available game names."""
        return sorted(self.templates.keys())

    def get_game_groups(self) -> list[dict]:
        """Return games grouped by the template their phases came from.

        A template that defines its own phases is a group leader; any
        templates pointing to it via phases_from are listed as variants
        under that leader. Games with no variants still appear as a
        leader with an empty variants list.

        [{"name": "7 Card Stud", "variants": [...]}, ...] sorted by name.
        """
        variants_of: dict[str, list[str]] = {}
        for t in self.templates.values():
            if t.phases_from:
                variants_of.setdefault(t.phases_from, []).append(t.name)
        out = []
        for name in sorted(self.templates.keys()):
            t = self.templates[name]
            if t.phases_from:
                continue  # appears as a variant under its parent
            out.append({
                "name": name,
                "variants": sorted(variants_of.get(name, [])),
            })
        return out

    def new_hand(self, game_name: str) -> dict:
        """Start a new hand with the specified game."""
        if game_name not in self.templates:
            raise ValueError(f"Unknown game: {game_name}")

        self.current_game = self.templates[game_name]
        self.slots = []
        self.next_slot = 1
        self.phase_index = 0
        self.card_in_phase = 0
        self.draw_round = 0
        self.last_up_was_queen = False

        # Set up wild cards
        if self.current_game.wild_cards:
            self.wild_ranks = list(self.current_game.wild_cards.get("ranks", []))
            self.wild_label = self.current_game.wild_cards.get("label", "")
        else:
            self.wild_ranks = []
            self.wild_label = ""

        self._advance_to_next_actionable_phase()

        return {
            "type": "new_hand",
            "game_name": game_name,
            "dealer": self.get_dealer().name,
            "wild_ranks": self.wild_ranks,
            "wild_label": self.wild_label,
        }

    def get_expected_card_type(self) -> str | None:
        """
        Returns "down" or "up" for the next expected card, or None if
        not currently in a dealing state.
        """
        if self.state not in (GameState.WAITING_FOR_CARD, GameState.HIT_ROUND):
            return None

        phase = self._current_phase()
        if phase is None:
            return None

        if phase.type == PhaseType.DEAL or phase.type == PhaseType.COMMUNITY:
            if self.card_in_phase < len(phase.pattern):
                return phase.pattern[self.card_in_phase]
            return None
        elif phase.type == PhaseType.HIT_ROUND:
            return phase.card_type
        elif phase.type == PhaseType.DRAW:
            return "down"  # replacement cards are always down

        return None

    def card_scanned(self, rank: str, suit: str) -> dict:
        """
        Process a scanned card. Returns action dict with:
        - direction: "slot" or "table"
        - card_type: "down" or "up"
        - slot_number: assigned slot (if direction is "slot")
        - messages: list of messages to send to remote
        """
        card_type = self.get_expected_card_type()
        if card_type is None:
            card_type = "down"  # default

        if card_type == "down":
            direction = "slot"
            slot_number = self.next_slot
            self.slots.append(Slot(slot_number, rank, suit, "down"))
            self.next_slot += 1
        else:
            direction = "table"
            slot_number = self.next_slot
            self.slots.append(Slot(slot_number, rank, suit, "up"))
            self.next_slot += 1

        messages = []

        # Card dealt message
        messages.append({
            "type": "card_dealt",
            "slot_number": slot_number,
            "card": {"rank": rank, "suit": suit},
            "card_type": card_type,
        })

        # Follow the Queen wild card tracking
        if self.current_game and self.current_game.dynamic_wild == "follow_the_queen":
            if card_type == "up":
                if self.last_up_was_queen:
                    # This card's rank becomes the new wild
                    self.wild_ranks = ["Q", rank]
                    plural = f"{rank}'s" if rank.isdigit() else f"{rank}s"
                    self.wild_label = f"Queens and {plural} are wild"
                    messages.append({
                        "type": "wild_card_update",
                        "wild_ranks": self.wild_ranks,
                        "label": self.wild_label,
                    })
                self.last_up_was_queen = (rank == "Q")

        # Advance phase tracking
        self.card_in_phase += 1
        phase = self._current_phase()
        if phase and phase.type in (PhaseType.DEAL, PhaseType.COMMUNITY):
            if self.card_in_phase >= len(phase.pattern):
                self.phase_index += 1
                self.card_in_phase = 0
                self._advance_to_next_actionable_phase()
                # Check if we've entered a new phase that needs a message
                messages.extend(self._phase_entry_messages())

        return {
            "direction": direction,
            "card_type": card_type,
            "slot_number": slot_number,
            "messages": messages,
        }

    def process_discard(self, slot_numbers: list[int]) -> dict:
        """Process a discard request from the remote player."""
        for slot in self.slots:
            if slot.slot_number in slot_numbers:
                slot.status = "discarded"

        # Move to dealing replacement cards
        self.state = GameState.WAITING_FOR_CARD
        # The current phase (draw) expects replacement cards
        # Number of replacements = number discarded

        return {
            "type": "discard_acknowledged",
            "slot_numbers": slot_numbers,
        }

    def process_challenge(self, slot_numbers: list[int]) -> dict:
        """Process a challenge selection from the remote player."""
        for slot in self.slots:
            if slot.slot_number in slot_numbers:
                slot.status = "challenged"

        self.phase_index += 1
        self.card_in_phase = 0
        self._advance_to_next_actionable_phase()

        return {
            "type": "challenge_acknowledged",
            "slot_numbers": slot_numbers,
        }

    def process_pass_challenge(self) -> dict:
        """Remote player passes on the challenge."""
        self.phase_index += 1
        self.card_in_phase = 0
        self._advance_to_next_actionable_phase()
        return {"type": "pass_challenge_acknowledged"}

    def continue_after_betting(self) -> list[dict]:
        """Host clicks 'Continue' after a betting round."""
        if self.state != GameState.WAITING_FOR_BETTING:
            return []

        self.phase_index += 1
        self.card_in_phase = 0
        self._advance_to_next_actionable_phase()
        return self._phase_entry_messages()

    def end_hand(self) -> dict:
        """End the current hand and rotate dealer."""
        self.state = GameState.HAND_OVER
        self.current_game = None
        self.phase_index = 0
        self.card_in_phase = 0
        self.draw_round = 0
        self.wild_ranks = []
        self.wild_label = ""
        self.last_up_was_queen = False
        self.slots = []
        self.advance_dealer()
        return {
            "type": "hand_over",
            "next_dealer": self.get_dealer().name,
        }

    def get_hand_state(self) -> dict:
        """Return the full current hand state for syncing."""
        # Count deal/community phases up to current position (for round display)
        deal_round = 0
        if self.current_game:
            for i, ph in enumerate(self.current_game.phases):
                if i > self.phase_index:
                    break
                if ph.type in (PhaseType.DEAL, PhaseType.COMMUNITY):
                    deal_round += 1
        return {
            "game_name": self.current_game.name if self.current_game else None,
            "state": self.state.value,
            "slots": [s.to_dict() for s in self.slots],
            "next_slot": self.next_slot,
            "wild_ranks": self.wild_ranks,
            "wild_label": self.wild_label,
            "expected_card_type": self.get_expected_card_type(),
            "current_phase": self._describe_current_phase(),
            "deal_round": deal_round,
            "dealer": self.get_dealer().name,
            "players": self.get_players_info(),
        }

    def get_active_hand(self) -> list[dict]:
        """Return only active (non-discarded) cards for display."""
        return [s.to_dict() for s in self.slots if s.status == "active"]

    # --- Peek scan (not part of game flow) ---

    def peek_card(self, rank: str, suit: str, label: str = "") -> dict:
        """Create a peek card message (doesn't affect hand state)."""
        return {
            "type": "peek_card",
            "card": {"rank": rank, "suit": suit},
            "label": label,
        }

    # --- Internal ---

    def _current_phase(self) -> Phase | None:
        if self.current_game is None:
            return None
        if self.phase_index >= len(self.current_game.phases):
            return None
        return self.current_game.phases[self.phase_index]

    def _advance_to_next_actionable_phase(self):
        """Skip to the next phase that requires action, setting state accordingly."""
        while True:
            phase = self._current_phase()
            if phase is None:
                self.state = GameState.WAITING_FOR_BETTING  # end of template
                return

            if phase.type in (PhaseType.DEAL, PhaseType.COMMUNITY):
                self.state = GameState.WAITING_FOR_CARD
                return
            elif phase.type == PhaseType.BETTING:
                self.state = GameState.WAITING_FOR_BETTING
                return
            elif phase.type == PhaseType.DRAW:
                self.state = GameState.WAITING_FOR_DRAW
                self.draw_round += 1
                return
            elif phase.type == PhaseType.CHALLENGE:
                self.state = GameState.WAITING_FOR_CHALLENGE
                return
            elif phase.type == PhaseType.HIT_ROUND:
                self.state = GameState.HIT_ROUND
                return

            self.phase_index += 1
            self.card_in_phase = 0

    def _phase_entry_messages(self) -> list[dict]:
        """Generate messages when entering a new phase."""
        phase = self._current_phase()
        if phase is None:
            return []

        if phase.type == PhaseType.DRAW:
            return [{
                "type": "draw_prompt",
                "draw_round": self.draw_round,
                "max_draw": phase.max_draw,
            }]
        elif phase.type == PhaseType.CHALLENGE:
            return [{
                "type": "challenge_prompt",
                "select_cards": phase.select_cards,
                "label": phase.label,
            }]

        return []

    def _describe_current_phase(self) -> str:
        """Human-readable description of current phase."""
        phase = self._current_phase()
        if phase is None:
            return "End of template"

        if phase.type == PhaseType.DEAL:
            remaining = len(phase.pattern) - self.card_in_phase
            next_type = phase.pattern[self.card_in_phase] if self.card_in_phase < len(phase.pattern) else "?"
            return f"Deal: {remaining} cards left (next: {next_type})"
        elif phase.type == PhaseType.BETTING:
            return "Betting round"
        elif phase.type == PhaseType.DRAW:
            return f"Draw round {self.draw_round} (max {phase.max_draw})"
        elif phase.type == PhaseType.CHALLENGE:
            return f"Challenge: {phase.label}"
        elif phase.type == PhaseType.HIT_ROUND:
            return "Hit round (open-ended)"
        elif phase.type == PhaseType.COMMUNITY:
            return f"Community: {phase.label}"
        return phase.type.value


def _default_templates() -> list[GameTemplate]:
    """Built-in game templates."""
    return [
        GameTemplate(
            name="5 Card Draw",
            phases=[
                Phase(type=PhaseType.DEAL, pattern=["down"] * 5),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.DRAW, max_draw=3),
                Phase(type=PhaseType.BETTING),
            ],
        ),
        GameTemplate(
            name="3 Toed Pete",
            phases=[
                Phase(type=PhaseType.DEAL, pattern=["down"] * 3),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.DRAW, max_draw=3),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.DRAW, max_draw=2),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.DRAW, max_draw=1),
                Phase(type=PhaseType.BETTING),
            ],
        ),
        GameTemplate(
            name="7 Card Stud",
            phases=[
                Phase(type=PhaseType.DEAL, pattern=["down", "down", "up"]),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.DEAL, pattern=["up"]),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.DEAL, pattern=["up"]),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.DEAL, pattern=["up"]),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.DEAL, pattern=["down"]),
                Phase(type=PhaseType.BETTING),
            ],
        ),
        GameTemplate(
            name="7 Stud Deuces Wild",
            phases_from="7 Card Stud",
            wild_cards={"ranks": ["2"], "label": "Deuces Wild"},
        ),
        GameTemplate(
            name="Follow the Queen",
            phases_from="7 Card Stud",
            wild_cards={"ranks": ["Q"], "label": "Queens wild"},
            dynamic_wild="follow_the_queen",
        ),
        GameTemplate(
            name="High Chicago",
            phases_from="7 Card Stud",
            notes="Split pot: best poker hand + highest spade in the hole",
        ),
        GameTemplate(
            name="Eight or Better",
            phases_from="7 Card Stud",
            notes="Split pot: best high hand + best low hand (highest card "
                  "8 or under, no pair) qualifies for the low half",
        ),
        GameTemplate(
            name="High/Low/High Challenge",
            phases=[
                Phase(type=PhaseType.DEAL, pattern=["down"] * 3),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.CHALLENGE, select_cards=2, label="Best 2-card high hand"),
                Phase(type=PhaseType.DEAL, pattern=["down"] * 2),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.CHALLENGE, select_cards=3, label="Best 3-card low hand"),
                Phase(type=PhaseType.DEAL, pattern=["down"] * 2),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.CHALLENGE, select_cards=5, label="Best 5-card poker hand"),
            ],
            repeatable=True,
        ),
        GameTemplate(
            name="7/27",
            phases=[
                Phase(type=PhaseType.DEAL, pattern=["down", "down"]),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.HIT_ROUND, card_type="up"),
            ],
            notes="Each player dealt 2 down initially; Rodney flips one of "
                  "the two face-up, then hit rounds (card or freeze). "
                  "Freeze 3x → frozen.",
        ),
        GameTemplate(
            name="7/27 (one up)",
            phases=[
                Phase(type=PhaseType.DEAL, pattern=["down", "up"]),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.HIT_ROUND, card_type="up"),
            ],
            notes="Each player dealt 1 down + 1 up initially, then hit "
                  "rounds (card or freeze). Freeze 3x → frozen.",
        ),
        GameTemplate(
            name="Texas Hold'em",
            phases=[
                Phase(type=PhaseType.DEAL, pattern=["down"] * 2),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.COMMUNITY, pattern=["up"] * 3, label="Flop"),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.COMMUNITY, pattern=["up"], label="Turn"),
                Phase(type=PhaseType.BETTING),
                Phase(type=PhaseType.COMMUNITY, pattern=["up"], label="River"),
                Phase(type=PhaseType.BETTING),
            ],
        ),
    ]

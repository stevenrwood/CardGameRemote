"""
Poker hand evaluator with wild-card support.

Given a list of cards and a list of ranks that are wild, compute the best
poker hand. Works for any number of cards >= 2. For hands of fewer than
5 cards, straights and flushes are not considered (can't form five
sequential / same-suit cards).

Card input formats accepted:
    - tuples: ("K", "spades")
    - dicts:  {"rank": "K", "suit": "spades"}
    - strings (short codes): "Ks", "10h", "Ad"

Rank strings are "2"..."10", "J", "Q", "K", "A". Suit strings are the full
name (lowercase): "clubs", "diamonds", "hearts", "spades".

Usage:
    >>> from poker_hands import best_hand
    >>> best_hand(["As", "Ah", "Kd", "Kh", "2s"], wild_ranks=["2"])
    HandResult(category='four_of_a_kind', label='Four of a Kind, Aces', ...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


RANK_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANK_ORDER)}  # "2"→2 .. "A"→14
VALUE_RANK = {v: r for r, v in RANK_VALUE.items()}

RANK_NAME = {
    "A": "Ace", "K": "King", "Q": "Queen", "J": "Jack",
    "10": "Ten", "9": "Nine", "8": "Eight", "7": "Seven",
    "6": "Six", "5": "Five", "4": "Four", "3": "Three", "2": "Two",
}

SUITS = {"clubs", "diamonds", "hearts", "spades"}
SUIT_FROM_LETTER = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}

CATEGORY_RANK = {
    "high_card":       0,
    "pair":            1,
    "two_pair":        2,
    "three_of_a_kind": 3,
    "straight":        4,
    "flush":           5,
    "full_house":      6,
    "four_of_a_kind":  7,
    "straight_flush":  8,
    "five_of_a_kind":  9,
}


@dataclass
class Card:
    rank: str
    suit: str
    is_wild: bool = False

    def __str__(self):
        suit_char = self.suit[0].upper()
        marker = "*" if self.is_wild else ""
        return f"{self.rank}{suit_char}{marker}"


@dataclass
class HandResult:
    category: str
    label: str
    tiebreakers: list = field(default_factory=list)
    cards: list = field(default_factory=list)

    @property
    def rank(self) -> int:
        """Higher wins. Use tiebreakers as secondary when equal."""
        return CATEGORY_RANK[self.category]

    def __str__(self):
        return self.label


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

_CARD_CODE_RE = re.compile(r"^(10|[2-9JQKA])([cdhs])$", re.IGNORECASE)


def _parse_one(card) -> Card:
    if isinstance(card, Card):
        return Card(card.rank, card.suit, card.is_wild)
    if isinstance(card, dict):
        return Card(str(card["rank"]).upper(), str(card["suit"]).lower())
    if isinstance(card, (tuple, list)) and len(card) >= 2:
        return Card(str(card[0]).upper(), str(card[1]).lower())
    if isinstance(card, str):
        m = _CARD_CODE_RE.match(card.strip())
        if not m:
            raise ValueError(f"Unrecognized card code: {card!r}")
        rank = m.group(1).upper()
        suit = SUIT_FROM_LETTER[m.group(2).lower()]
        return Card(rank, suit)
    raise TypeError(f"Cannot parse card: {card!r}")


def _parse_cards(cards, wild_ranks) -> list[Card]:
    wild_set = {str(r).upper() for r in (wild_ranks or [])}
    parsed = []
    for c in cards:
        card = _parse_one(c)
        if card.rank not in RANK_VALUE:
            raise ValueError(f"Invalid rank: {card.rank!r}")
        if card.suit not in SUITS:
            raise ValueError(f"Invalid suit: {card.suit!r}")
        card.is_wild = card.rank in wild_set
        parsed.append(card)
    return parsed


def _rank_name(v: int) -> str:
    return RANK_NAME[VALUE_RANK[v]]


# ---------------------------------------------------------------------------
# Hand-category evaluators
# ---------------------------------------------------------------------------

def _rank_groups(cards):
    """Return (regular_groups_by_rank_value, list_of_wild_cards)."""
    groups: dict[int, list[Card]] = {}
    wilds: list[Card] = []
    for c in cards:
        if c.is_wild:
            wilds.append(c)
        else:
            groups.setdefault(RANK_VALUE[c.rank], []).append(c)
    return groups, wilds


def _best_n_of_a_kind(cards, n):
    """Best group of exactly n cards of one rank using wilds to fill."""
    groups, wilds = _rank_groups(cards)
    # Try the highest rank with enough regulars + wilds
    for rv in sorted(groups.keys(), reverse=True):
        regulars = groups[rv]
        if len(regulars) + len(wilds) < n:
            # Not enough even with all wilds; but maybe a lower rank has more
            # regulars — keep scanning.
            continue
        wilds_needed = max(0, n - len(regulars))
        used = regulars[:n - wilds_needed] + wilds[:wilds_needed]
        return rv, used
    # All-wilds: pick Aces
    if len(wilds) >= n:
        return 14, wilds[:n]
    return None


def _best_two_pair(cards):
    """Best two-pair. Uses regulars/wilds. Returns (hi, lo, used) or None."""
    groups, wilds = _rank_groups(cards)
    # Enumerate every (hi, lo) rank pair, greedy-use wilds.
    best = None
    ranks = sorted(groups.keys(), reverse=True)
    for i, hi in enumerate(ranks):
        hi_count = len(groups[hi])
        hi_wilds = max(0, 2 - hi_count)
        if hi_wilds > len(wilds):
            continue
        for lo in ranks[i + 1:]:
            lo_count = len(groups[lo])
            lo_wilds = max(0, 2 - lo_count)
            if hi_wilds + lo_wilds > len(wilds):
                continue
            cand = (hi, lo)
            if best is None or cand > best[:2]:
                # Build used cards
                used = list(groups[hi][:min(2, hi_count)])
                wstack = list(wilds)
                for _ in range(hi_wilds):
                    used.append(wstack.pop(0))
                used.extend(groups[lo][:min(2, lo_count)])
                for _ in range(lo_wilds):
                    used.append(wstack.pop(0))
                best = (hi, lo, used)
    return best


def _best_full_house(cards):
    """Best full house (3 of one rank + 2 of another)."""
    groups, wilds = _rank_groups(cards)
    ranks = sorted(groups.keys(), reverse=True)
    best = None
    for trips in ranks:
        tc = len(groups[trips])
        tw = max(0, 3 - tc)
        if tw > len(wilds):
            continue
        for pair in ranks:
            if pair == trips:
                continue
            pc = len(groups[pair])
            pw = max(0, 2 - pc)
            if tw + pw > len(wilds):
                continue
            cand = (trips, pair)
            if best is None or cand > best[:2]:
                wstack = list(wilds)
                used = list(groups[trips][:min(3, tc)])
                for _ in range(tw):
                    used.append(wstack.pop(0))
                used.extend(groups[pair][:min(2, pc)])
                for _ in range(pw):
                    used.append(wstack.pop(0))
                best = (trips, pair, used)
    return best


def _best_flush(cards):
    """Best 5-card flush. Wilds count as any suit (use the best one)."""
    wilds = [c for c in cards if c.is_wild]
    regulars = [c for c in cards if not c.is_wild]
    by_suit: dict[str, list[Card]] = {}
    for c in regulars:
        by_suit.setdefault(c.suit, []).append(c)
    best = None
    for suit, suit_cards in by_suit.items():
        if len(suit_cards) + len(wilds) < 5:
            continue
        # Highest-ranked regulars first; wilds act as Aces for scoring.
        suit_sorted = sorted(suit_cards, key=lambda c: RANK_VALUE[c.rank], reverse=True)
        wilds_needed = max(0, 5 - len(suit_sorted))
        used = suit_sorted[:5 - wilds_needed] + wilds[:wilds_needed]
        values = sorted(
            [RANK_VALUE[c.rank] if not c.is_wild else 14 for c in used],
            reverse=True,
        )
        cand = (values, suit)
        if best is None or values > best[0]:
            best = (values, used, suit)
    return best


def _best_straight_from_ranks(ranks_available: dict, wilds: list) -> tuple | None:
    """Find the highest 5-card straight given a rank->regular-card map and
    a pool of wild cards. Returns (top_value, used_cards) or None."""
    for top in range(14, 4, -1):
        needed = [top - i for i in range(5)]
        missing = [v for v in needed if v not in ranks_available]
        if len(missing) <= len(wilds):
            used = []
            wstack = list(wilds)
            for v in needed:
                if v in ranks_available:
                    used.append(ranks_available[v][0])
                else:
                    used.append(wstack.pop(0))
            return top, used
    # A-low straight (5-4-3-2-A)
    alow = [5, 4, 3, 2, 14]
    missing = [v for v in alow if v not in ranks_available]
    if len(missing) <= len(wilds):
        used = []
        wstack = list(wilds)
        for v in alow:
            if v in ranks_available:
                used.append(ranks_available[v][0])
            else:
                used.append(wstack.pop(0))
        return 5, used
    return None


def _best_straight(cards):
    groups, wilds = _rank_groups(cards)
    return _best_straight_from_ranks(groups, wilds)


def _best_straight_flush(cards):
    wilds = [c for c in cards if c.is_wild]
    regulars = [c for c in cards if not c.is_wild]
    by_suit: dict[str, list[Card]] = {}
    for c in regulars:
        by_suit.setdefault(c.suit, []).append(c)
    best = None
    for suit, suit_cards in by_suit.items():
        if len(suit_cards) + len(wilds) < 5:
            continue
        ranks_available = {}
        for c in suit_cards:
            ranks_available.setdefault(RANK_VALUE[c.rank], []).append(c)
        r = _best_straight_from_ranks(ranks_available, wilds)
        if r and (best is None or r[0] > best[0]):
            best = (r[0], r[1], suit)
    # All-wilds straight-flush (5+ wilds): royal flush of any suit.
    if best is None and len(wilds) >= 5:
        best = (14, wilds[:5], wilds[0].suit if wilds[0].suit in SUITS else "spades")
    return best


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------

def best_hand(cards, wild_ranks=None) -> HandResult:
    """Evaluate the best poker hand from the given cards.

    cards: iterable of Card / dict / (rank, suit) / code-string.
    wild_ranks: optional iterable of ranks considered wild (e.g., ["2"]).
    """
    parsed = _parse_cards(cards, wild_ranks)
    n = len(parsed)
    if n < 2:
        raise ValueError("Need at least 2 cards")

    allow_fiveplus = n >= 5
    allow_fourplus = n >= 4
    allow_threeplus = n >= 3

    # Five of a kind
    if allow_fiveplus:
        r = _best_n_of_a_kind(parsed, 5)
        if r:
            rv, used = r
            return HandResult(
                "five_of_a_kind",
                f"Five of a Kind, {_rank_name(rv)}s",
                [rv],
                used,
            )

    # Straight flush
    if allow_fiveplus:
        r = _best_straight_flush(parsed)
        if r:
            top, used, suit = r
            label = "Royal Flush" if top == 14 else f"Straight Flush, {_rank_name(top)}-high {suit}"
            return HandResult("straight_flush", label, [top], used)

    # Four of a kind
    if allow_fourplus:
        r = _best_n_of_a_kind(parsed, 4)
        if r:
            rv, used = r
            return HandResult(
                "four_of_a_kind",
                f"Four of a Kind, {_rank_name(rv)}s",
                [rv],
                used,
            )

    # Full house
    if allow_fiveplus:
        r = _best_full_house(parsed)
        if r:
            trips, pair, used = r
            return HandResult(
                "full_house",
                f"Full House, {_rank_name(trips)}s over {_rank_name(pair)}s",
                [trips, pair],
                used,
            )

    # Flush
    if allow_fiveplus:
        r = _best_flush(parsed)
        if r:
            values, used, suit = r
            return HandResult(
                "flush",
                f"Flush, {_rank_name(values[0])}-high {suit}",
                values,
                used,
            )

    # Straight
    if allow_fiveplus:
        r = _best_straight(parsed)
        if r:
            top, used = r
            return HandResult(
                "straight",
                f"Straight, {_rank_name(top)}-high",
                [top],
                used,
            )

    # Three of a kind
    if allow_threeplus:
        r = _best_n_of_a_kind(parsed, 3)
        if r:
            rv, used = r
            return HandResult(
                "three_of_a_kind",
                f"Three of a Kind, {_rank_name(rv)}s",
                [rv],
                used,
            )

    # Two pair
    if allow_fourplus:
        r = _best_two_pair(parsed)
        if r:
            hi, lo, used = r
            return HandResult(
                "two_pair",
                f"Two Pair, {_rank_name(hi)}s and {_rank_name(lo)}s",
                [hi, lo],
                used,
            )

    # Pair
    r = _best_n_of_a_kind(parsed, 2)
    if r:
        rv, used = r
        return HandResult(
            "pair",
            f"Pair of {_rank_name(rv)}s",
            [rv],
            used,
        )

    # High card — treat each wild as an Ace (best it can be).
    def _v(c):
        return 14 if c.is_wild else RANK_VALUE[c.rank]
    sorted_cards = sorted(parsed, key=_v, reverse=True)
    values = [_v(c) for c in sorted_cards]
    return HandResult(
        "high_card",
        f"{_rank_name(values[0])} high",
        values,
        sorted_cards,
    )


# ---------------------------------------------------------------------------
# Quick self-tests when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        # (cards, wild_ranks, expected_category)
        (["As", "Ah"], None, "pair"),
        (["As", "Kh"], None, "high_card"),
        (["As", "Ah", "Ad"], None, "three_of_a_kind"),
        (["As", "Ah", "Ad", "Ac"], None, "four_of_a_kind"),
        (["As", "Ah", "Ad", "Ac", "Kh"], None, "four_of_a_kind"),
        (["As", "Ah", "Kd", "Kh", "Kc"], None, "full_house"),
        (["As", "Ks", "Qs", "Js", "10s"], None, "straight_flush"),  # royal
        (["5s", "4d", "3h", "2c", "Ah"], None, "straight"),          # A-low
        (["9s", "8s", "7s", "6s", "5s"], None, "straight_flush"),
        (["9s", "8s", "7s", "6s", "5h"], None, "straight"),
        (["9s", "8s", "7s", "6s", "2s"], None, "flush"),
        (["As", "Ah", "2s"], ["2"], "three_of_a_kind"),              # wild deuce
        (["As", "Ah", "Kd", "2s"], ["2"], "three_of_a_kind"),        # AAA+K (wild→A)
        (["As", "Ah", "Kd", "Kh", "2s"], ["2"], "full_house"),       # AAA KK
        (["As", "Ah", "Kd", "Kh", "2s", "2d"], ["2"], "four_of_a_kind"),  # A A A A K K
        (["As", "2s", "2d"], ["2"], "three_of_a_kind"),              # AAA (both wilds→A)
        (["2s", "2d", "2h", "2c", "As"], ["2"], "five_of_a_kind"),   # 5 aces
        (["Ks", "Qs", "Js", "2s", "2h"], ["2"], "straight_flush"),   # K-high with wild 10 + wild 9 or A?
        (["As", "Kh"], ["2"], "high_card"),
        (["3h", "3d", "4s", "4c"], None, "two_pair"),
        (["3h", "3d", "4s"], None, "pair"),                          # 2pair needs 4 cards
        # 4-card: no straight/flush
        (["As", "Ks", "Qs", "Js"], None, "high_card"),               # 4 cards, no straight
    ]
    passes = fails = 0
    for cards, wilds, expected in cases:
        got = best_hand(cards, wilds)
        ok = got.category == expected
        status = "OK " if ok else "FAIL"
        if ok:
            passes += 1
        else:
            fails += 1
        wilds_str = f" wilds={wilds}" if wilds else ""
        print(f"  {status}  {cards}{wilds_str} → {got.category:18s}  {got.label}")
    print(f"\n{passes} pass, {fails} fail")

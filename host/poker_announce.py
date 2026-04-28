"""Poker bet-first announcer — stud, draw, FTQ, Hold'em, etc.

Walks every active non-folded player's visible up-cards through
``poker_hands.best_hand`` (with the current game's wild ranks),
finds the highest hand, and speaks "X, <hand> is high. Your bet."
The runner-up's category is consulted to decide how many kickers
to read aloud — saying "Ace, Jack, Eight is high" is more than
the dealer needs when the next-best hand is just "King high".

7/27, Challenge variants, and all-down games (5CD, 3 Toed Pete)
short-circuit at the top because they have their own announce
paths or no up cards to compare.

The single overhead_test cross-reference (``_get_deal_order``)
is imported lazily so this module stays leaf-importable.
"""

from log_buffer import log
from speech import speech


def _announce_poker_hand_bet_first(s):
    """Announce who bets first at a poker-hand game based on best visible
    hand. Skips 7/27 (its own announcer), Challenge games, and all-down
    games (5CD, 3 Toed Pete) where nobody has an up card to compare."""
    from overhead_test import _get_deal_order

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

"""Follow the Queen — dynamic wild-card tracking.

FTQ's wild rule: every up-card after a Queen also becomes wild.
The first Queen out makes Queens wild (the default for the
template). The card immediately *following* a Queen joins the
wild set ("Queens and Sevens are wild"). Two queens in a row
reset the watch — only the most recent Queen's follower wins
the second wild slot.

These two procedures are called from the round-confirm path:

  - ``_check_follow_the_queen_round`` updates the wild state
    forward as new up-cards land.
  - ``_recompute_follow_the_queen`` replays the rule from
    scratch over ``console_hand_cards`` after a user
    correction, in case the corrected card was a follower
    that should have made (or stopped making) something wild.
"""

from log_buffer import log
from speech import speech


_RANK_SHORT = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}


def _check_follow_the_queen_round(s, round_cards, announce=True):
    """Check cards for Follow the Queen wild at end of round.

    Args:
        round_cards: list of {"player": name, "card": "Rank of Suit"} in deal order
        announce: when False, update wild state silently. Used for the
            last up-card round of stud games — we defer the speech until
            after the trailing 7th (down) card has been dealt, so the
            final wild state is announced once, alongside the high-hand
            bet-first call. State updates still happen either way; only
            speech.say is gated.
    """
    ge = s.game_engine
    if not ge.current_game or ge.current_game.dynamic_wild != "follow_the_queen":
        return

    for c in round_cards:
        parts = c["card"].split(" of ")
        if len(parts) != 2:
            continue
        rank = parts[0]
        rank_short = _RANK_SHORT.get(rank, rank)

        if ge.last_up_was_queen:
            if rank_short == "Q":
                # Queen immediately after a Queen: ignore the earlier one
                # and keep watching. The second Queen's follower is what
                # becomes wild. (Avoids the "Queens and Queens are wild"
                # annunciation.)
                pass
            else:
                ge.wild_ranks = ["Q", rank_short]
                plural = f"{rank}'s" if rank.isdigit() else f"{rank}s"
                ge.wild_label = f"Queens and {plural} are wild"
                log.log(f"[WILD] {ge.wild_label}")
                if announce:
                    speech.say(f"Queens and {plural} are now wild")

        ge.last_up_was_queen = (rank_short == "Q")

    # Always announce current wild state at end of round if non-default
    if ge.wild_label and ge.wild_label != "Queens wild":
        log.log(f"[WILD] Current: {ge.wild_label}")


def _recompute_follow_the_queen(s):
    """Replay FTQ queen-follower logic against console_hand_cards in round
    order and update ge.wild_ranks/wild_label. Used when a correction
    changes a card that may have been the follower of a Queen. Announces
    the new wild state if it differs from the current one."""
    ge = s.game_engine
    if not ge.current_game or ge.current_game.dynamic_wild != "follow_the_queen":
        return
    prior_label = ge.wild_label
    ge.wild_ranks = ["Q"]
    ge.wild_label = "Queens wild"
    ge.last_up_was_queen = False
    by_round = {}
    for e in s.console_hand_cards:
        by_round.setdefault(e.get("round", 0), []).append(e)
    for r in sorted(by_round.keys()):
        for c in by_round[r]:
            parts = c.get("card", "").split(" of ")
            if len(parts) != 2:
                continue
            rank = parts[0]
            rank_short = _RANK_SHORT.get(rank, rank)
            if ge.last_up_was_queen and rank_short != "Q":
                ge.wild_ranks = ["Q", rank_short]
                plural = f"{rank}'s" if rank.isdigit() else f"{rank}s"
                ge.wild_label = f"Queens and {plural} are wild"
            ge.last_up_was_queen = (rank_short == "Q")
    if ge.wild_label != prior_label:
        log.log(f"[WILD] Recomputed after correction: {ge.wild_label}")
        if ge.wild_label == "Queens wild":
            speech.say("Correction: only queens are now wild")
        else:
            tail = ge.wild_label.replace("Queens and ", "").replace(" are wild", "")
            speech.say(f"Correction: queens and {tail} are now wild")

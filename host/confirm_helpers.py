"""Helpers that run at /api/console/confirm time, just before
the round's up-cards get appended to ``console_hand_cards``.

Currently one function — ``_dedup_round_cards_against_seen`` —
which catches two kinds of duplicate-card collisions that
recognition can produce. Lives in its own module because it's
single-purpose, called once per Confirm, and doesn't fit
cleanly into either the recognition pipeline (brio_watcher /
zone_monitor) or the announce path (poker_announce).
"""

from log_buffer import log


def _dedup_round_cards_against_seen(s, round_cards):
    """Detect duplicate cards at confirm time.

    Two flavors of collision get treated differently:

    1. SAME-SCAN duplicate — two zones in *this* Brio scan returned
       the same card. Almost always a YOLO hallucination (the model
       has a bias toward a handful of cards on ambiguous inputs;
       Ace of Diamonds especially). The auto-recognized card gets
       cleared so the dealer is forced to re-scan or hand-correct
       that player. User-corrected zones are trusted.

    2. CROSS-ROUND duplicate — this round's card matches one already
       in console_hand_cards (a prior round) or Rodney's known
       downs. We can't tell which side is wrong: maybe a previous
       round had a hallucination and this round is right; maybe
       this round is the hallucination. Log it and leave as-is so
       the genuine card still shows in the player's hand. The
       dealer can correct after the fact via the zone-tap modal if
       needed.

    Operates in place on round_cards.
    """
    history_seen = set()
    for c in s.console_hand_cards:
        history_seen.add(c["card"])
    suit_full = {"c": "Clubs", "d": "Diamonds", "h": "Hearts", "s": "Spades"}
    rank_full = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}

    def _canonical(rank, suit):
        return f"{rank_full.get(rank, rank)} of {suit.capitalize()}"

    flipped_slot = (s.rodney_flipped_up or {}).get("slot")
    for slot_num, d in s.rodney_downs.items():
        if slot_num == flipped_slot:
            continue
        history_seen.add(_canonical(d["rank"], d["suit"]))
    for d in s.slot_pending.values():
        history_seen.add(_canonical(d["rank"], d["suit"]))

    same_scan_seen = set()
    cleared = []
    for entry in list(round_cards):
        card = entry.get("card", "")
        player = entry.get("player", "")
        if not card:
            continue
        if card in same_scan_seen:
            # Same-scan dup → almost certainly a hallucination.
            if s.monitor.zone_state.get(player) == "corrected":
                log.log(
                    f"[CONFIRM] {player}: {card} duplicates same-scan "
                    f"card but was user-corrected — keeping as-is"
                )
                continue
            log.log(
                f"[CONFIRM] {player}: {card} duplicates same-scan "
                f"card — clearing recognition; dealer must re-scan "
                f"or correct"
            )
            s.monitor.last_card[player] = ""
            s.monitor.zone_state[player] = "empty"
            try:
                round_cards.remove(entry)
            except ValueError:
                pass
            cleared.append(player)
            continue
        if card in history_seen:
            # Cross-round dup → log only, leave the genuine-looking
            # card in place. The history entry could just as easily
            # be the hallucination.
            log.log(
                f"[CONFIRM] {player}: {card} duplicates a prior round / "
                f"Rodney down — keeping as-is (dealer can correct via "
                f"zone tap if wrong)"
            )
        same_scan_seen.add(card)
    if cleared:
        log.log(
            f"[CONFIRM] Cleared same-scan duplicates for: "
            f"{', '.join(cleared)}"
        )

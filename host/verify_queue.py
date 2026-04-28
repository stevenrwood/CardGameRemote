"""Per-slot verify-modal queue and the verify-resolve path.

When the Pi poller (or guided-deal loop) sees a low-confidence
scan it parks the guess in ``s.slot_pending`` and queues the slot
in ``s.verify_queue``. ``_promote_next_verify`` opens the modal
on the next /api/console/confirm; ``_resolve_verify`` commits
the user's choice into ``s.rodney_downs``.

Plus two small helpers that everything in the slot-handling
pipeline reaches for: ``_table_log_add`` (append to the rolling
table log) and ``_parse_card_code`` (turn 'Ac' / '10h' into
{rank, suit}).

Cross-module helpers (``_initial_down_count``, ``_stats_bump``)
are imported lazily inside the entry points so this module
stays leaf-importable.
"""

import re
import time

# Mirror of table_state._SUIT_LETTER — private here too, used by
# _parse_card_code's "Ac" / "10h" short-form parser. Kept inline
# rather than imported so this module has no overhead_test
# back-reference at import time.
_SUIT_LETTER = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}


def _table_log_add(s, msg):
    s.table_log.append({"ts": int(time.time()), "msg": msg})
    if len(s.table_log) > 200:
        s.table_log = s.table_log[-100:]


def _parse_card_code(code):
    """Parse 'Ac' or '10h' into {rank, suit} or None."""
    if not code:
        return None
    m = re.match(r"^\s*(10|[2-9JQKA])([hdcs])\s*$", code, re.IGNORECASE)
    if not m:
        return None
    return {"rank": m.group(1).upper(), "suit": _SUIT_LETTER[m.group(2).lower()]}


def _promote_next_verify(s) -> bool:
    """If no modal is open and a queued slot has a pending guess, open it.

    Returns True if state changed (pending_verify was set). Caller owns the
    table_lock.
    """
    if s.pending_verify is not None or not s.verify_queue:
        return False
    for slot_num in list(s.verify_queue):
        guess = s.slot_pending.get(slot_num)
        if not guess:
            continue
        s.pending_verify = {
            "slot": slot_num,
            "guess": dict(guess),
            "prompt": (
                f"Slot {slot_num} needs verification. "
                f"Remove the card, hold it up for Rodney, "
                f"then confirm or override."
            ),
            "image_url": (
                None if s.pi_offline
                else f"/api/table/slot_image/{slot_num}"
            ),
        }
        _table_log_add(s, f"Slot {slot_num}: modal opened for verify")
        return True
    return False


def _enqueue_down_card_verifies(s):
    """Called at the end of an up-card round (console Confirm Cards).

    Any slot that has a pending (low-confidence) scan that isn't already
    verified in rodney_downs gets added to the FIFO verify queue — but
    only if that slot has actually been dealt. A stale slot_pending
    entry for a trailing slot that we haven't started guiding yet (e.g.
    FTQ slot 3 during round 1) would otherwise pop a verify modal for a
    card that doesn't exist yet.
    """
    from game_meta import _initial_down_count

    ge = s.game_engine
    initial = _initial_down_count(ge)
    # Slots currently valid to verify: the initial guided range plus any
    # slot the guided loop is presently iterating. Trailing slots get
    # added once the trailing guided session starts.
    expected_slots = set(range(1, initial + 1))
    if s.guided_deal:
        gd = s.guided_deal
        if "num_slots" in gd:
            expected_slots |= set(range(1, int(gd["num_slots"]) + 1))
        for sn in gd.get("slots") or []:
            try:
                expected_slots.add(int(sn))
            except (TypeError, ValueError):
                pass
    with s.table_lock:
        newly_queued = []
        for slot_num, guess in s.slot_pending.items():
            if slot_num in s.rodney_downs:
                continue
            if slot_num in s.verify_queue:
                continue
            if slot_num not in expected_slots:
                # Haven't dealt this slot yet (FTQ/7CS trailing, etc.) —
                # don't prompt the user to verify a card that isn't there.
                continue
            s.verify_queue.append(slot_num)
            newly_queued.append(slot_num)
        if newly_queued:
            for sn in newly_queued:
                _table_log_add(s, f"Slot {sn}: queued for verify (blink LED)")
            # Open the modal immediately rather than waiting for the next
            # poller tick — nicer UX, and in offline mode the poller may
            # be sleeping between failed fetches.
            _promote_next_verify(s)
            s.table_state_version += 1
    # TODO: POST to Pi /slots/<n>/led to blink once that endpoint is wired.
    return newly_queued


def _resolve_verify(s, card_dict):
    """Set the verified card into rodney_downs[slot] and clear the modal.

    Also tallies whether the Pi's guess matched what the user finally
    submitted — that's our accuracy signal for the low-confidence path
    (and by extension, a hint about whether GUIDED_GOOD_CONF is set too
    low or too high).
    """
    from app_state import _stats_bump

    with s.table_lock:
        pv = s.pending_verify
        if not pv:
            return False
        slot = pv.get("slot")
        if slot is None:
            return False
        guess = pv.get("guess") or {}
        guess_matched = (
            bool(guess.get("rank"))
            and guess.get("rank") == card_dict.get("rank")
            and guess.get("suit") == card_dict.get("suit")
        )
        s.rodney_downs[slot] = {
            "rank": card_dict["rank"],
            "suit": card_dict["suit"],
        }
        code = f"{card_dict['rank']}{card_dict['suit'][0]}"
        s.pi_prev_slots[slot] = code
        s.slot_pending.pop(slot, None)
        if slot in s.verify_queue:
            s.verify_queue.remove(slot)
        s.pending_verify = None
        _table_log_add(s, f"Slot {slot}: {code} (verified)")
        s.table_state_version += 1
    _stats_bump(s, "pi_verify_right" if guess_matched else "pi_verify_wrong")
    return True

"""Pi scanner background poller and stuck-cards alert.

Owns the daemon thread that keeps polling /slots on the Pi and
maps recognized scans into ``s.rodney_downs`` (high confidence,
auto-accepted) or ``s.slot_pending`` (medium confidence, awaits
verify modal). Also owns the one-shot "10 beeps over 20 s"
alert that fires at the start of a new hand if the previous
hand's cards are still seated in the scanner.

Cross-module helpers (``_next_deal_position_type``,
``_promote_next_verify``, ``_table_log_add``) live in
overhead_test.py and are imported lazily inside the loop to
avoid a circular import — by the time the daemon thread first
runs, overhead_test is fully loaded.
"""

import random
import time
from threading import Thread

from log_buffer import log
from pi_scanner import (
    _pi_buzz,
    _pi_fetch_slots,
    _pi_flash,
)


_SIM_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
_SIM_SUITS = ["clubs", "diamonds", "hearts", "spades"]


def _simulate_offline_slot_scans(s):
    """Fill rodney_downs' expected slots with random low-confidence guesses
    so a hand can be played end-to-end without the Pi. Each missing slot
    (not in rodney_downs, not in slot_pending) gets one random card at
    conf=0.20 — low enough to queue a verify modal on Confirm Cards where
    Rodney can override with the actual card.
    """
    from game_meta import _total_downs_in_pattern
    from verify_queue import _table_log_add

    max_slot = _total_downs_in_pattern(s.game_engine)
    if max_slot <= 0:
        return
    with s.table_lock:
        added = []
        for n in range(1, max_slot + 1):
            if n in s.rodney_downs or n in s.slot_pending:
                continue
            rank = random.choice(_SIM_RANKS)
            suit = random.choice(_SIM_SUITS)
            s.slot_pending[n] = {"rank": rank, "suit": suit, "confidence": 0.20}
            added.append((n, f"{rank}{suit[0]}"))
        if added:
            for (n, code) in added:
                _table_log_add(s, f"Slot {n}: simulated {code} (Pi offline, needs verify)")
            s.table_state_version += 1


def _update_flash_for_deal_state(s):
    """Hold LEDs while a down card is the next expected deal; release otherwise."""
    from game_meta import _next_deal_position_type

    nxt = _next_deal_position_type(s)
    _pi_flash(s, nxt == "down")


def _pi_poll_loop(s):
    """Background poll: map Pi scanner detections into rodney_downs.

    - Each /slots result is compared against s.pi_prev_slots.
    - A new card (slot was empty or held a different card) becomes:
        * rodney_downs[slot] = card when confidence >= threshold, OR
        * a pending_verify prompt otherwise (poller stops advancing that
          slot until the verify modal is resolved).
    - Slots that go empty clear their rodney_downs entry too.
    """
    from game_meta import _next_deal_position_type, _total_downs_in_pattern
    from verify_queue import _promote_next_verify, _table_log_add

    log.log("[PI] poll loop started")
    offline_streak = 0
    while s.pi_polling:
        # Guided dealing takes exclusive ownership of the scanner for
        # all-down games. The regular poller idles until guided completes.
        if s.guided_deal is not None:
            time.sleep(0.5)
            continue
        # Only hit the Pi when we're actually expecting a down card to be
        # dealt. Gate on the deal pattern directly (not pi_flash_held) so a
        # failed /flash/hold call doesn't also stop the scan polling.
        # During the stuck-cards-after-final-round window the buzzer
        # thread does its own once-per-30s /slots refresh — no need
        # for this loop to keep capturing+inferring constantly, which
        # was overloading a marginal Pi.
        expecting_down = _next_deal_position_type(s) == "down"
        if expecting_down:
            _update_flash_for_deal_state(s)
        else:
            # Make sure flash isn't held while we're not expecting a
            # deal — the Pi spends less effort on capture cycles.
            _pi_flash(s, False)
            time.sleep(2.0)
            continue
        doc = _pi_fetch_slots(s)
        if doc is None:
            offline_streak += 1
            # After two failed fetches, assume the Pi isn't running and
            # simulate slot scans so gameplay is testable without hardware.
            # Each expected slot gets a random low-confidence guess the user
            # can override in the verify modal. Also promote any queued
            # verify into pending_verify so the modal actually opens in
            # offline mode.
            if offline_streak >= 2 or s.pi_offline:
                _simulate_offline_slot_scans(s)
                with s.table_lock:
                    if _promote_next_verify(s):
                        s.table_state_version += 1
            time.sleep(2.0)
            continue
        offline_streak = 0
        # Only scan slots the current game actually uses (FTQ=3, Hold'em=2).
        max_slot = _total_downs_in_pattern(s.game_engine)
        with s.table_lock:
            # Re-check guided ownership under the lock. The top-of-loop
            # check is outside the lock, so a guided session that started
            # during our HTTP fetch would otherwise see this (stale)
            # scan overwrite rodney_downs — that's the round-3 Challenge
            # bug where slots 4/5 got instantly "already filled" with the
            # old cards right after _start_next_challenge_round popped
            # them.
            if s.guided_deal is not None:
                continue
            changed = False
            for entry in doc.get("slots", []):
                slot_num = entry.get("slot")
                if slot_num is None:
                    continue
                if slot_num > max_slot:
                    if slot_num in s.rodney_downs:
                        s.rodney_downs.pop(slot_num, None)
                        changed = True
                    s.pi_prev_slots.pop(slot_num, None)
                    s.slot_pending.pop(slot_num, None)
                    s.slot_empty[slot_num] = True
                    continue

                recognized = entry.get("recognized")
                rank = entry.get("rank")
                suit = entry.get("suit")
                conf = float(entry.get("confidence", 0.0))
                is_empty = (not recognized) or conf < s.pi_empty_threshold

                if is_empty:
                    # Slot physically empty: mark empty but keep slot_pending
                    # intact — the last-seen guess survives removal so the
                    # verify modal can still fire after a Confirm Cards.
                    # If there's a weak scan below threshold log it once so
                    # the user can see that the scanner IS seeing something
                    # but deciding it's noise.
                    if recognized and rank and suit and conf > 0:
                        weak_code = f"{rank}{suit[0]} ({int(conf*100)}%)"
                        if s._pi_last_logged.get(slot_num) != weak_code:
                            log.log(f"[PI] Slot {slot_num}: weak {weak_code} below empty threshold")
                            s._pi_last_logged[slot_num] = weak_code
                    elif s._pi_last_logged.get(slot_num) is not None:
                        s._pi_last_logged[slot_num] = None
                    s.slot_empty[slot_num] = True
                    s.pi_prev_slots.pop(slot_num, None)
                    continue

                # Non-empty: remember what's there now.
                s.slot_empty[slot_num] = False
                code = f"{rank}{suit[0]}" if rank and suit else ""

                if conf >= s.pi_confidence_threshold:
                    prev = s.pi_prev_slots.get(slot_num)
                    if prev == code:
                        continue
                    s.rodney_downs[slot_num] = {
                        "rank": rank, "suit": suit, "confidence": round(conf, 2),
                    }
                    s.slot_pending.pop(slot_num, None)
                    if slot_num in s.verify_queue:
                        s.verify_queue.remove(slot_num)
                    s.pi_prev_slots[slot_num] = code
                    _table_log_add(s, f"Slot {slot_num}: {code} (auto, {int(conf*100)}%)")
                    changed = True
                else:
                    # Medium confidence: hold as the latest guess but don't
                    # surface a modal until the dealer runs /api/console/confirm
                    # and the user subsequently removes the card from the slot.
                    guess = {"rank": rank, "suit": suit, "confidence": round(conf, 2)}
                    if s.slot_pending.get(slot_num) != guess:
                        s.slot_pending[slot_num] = guess
                        changed = True

            if _promote_next_verify(s):
                changed = True
            if changed:
                s.table_state_version += 1
        time.sleep(1.0)
    log.log("[PI] poll loop stopped")


def _pi_poll_start(s):
    if s.pi_polling:
        return
    s.pi_polling = True
    s.pi_prev_slots = {}
    t = Thread(target=_pi_poll_loop, args=(s,), daemon=True)
    s.pi_poll_thread = t
    t.start()


def _pi_poll_stop(s):
    s.pi_polling = False
    # Don't join — daemon thread will exit on its own


# ---------------------------------------------------------------------------
# Stuck-cards alert at new-hand start
# ---------------------------------------------------------------------------
#
# When a new hand begins, any card still in the scanner from the
# previous hand is a problem — the dealer forgot to collect it. We
# beep the Pi piezo 10 times over 20 s so the dealer can pull them
# out before the deal continues. Replaces the old every-30-s nag
# loop, which kept buzzing the table for the whole post-game window.


def _stuck_slots_at_new_hand(s) -> list[int]:
    """Fetch /slots once and return slot numbers that look like they
    actually hold a card. Returns [] when the Pi is unreachable so
    we don't false-buzz on transient timeouts.

    Uses ``pi_confidence_threshold`` (the same auto-accept bar the
    main poll loop uses to commit cards into ``rodney_downs``)
    rather than just ``recognized=True``. The Pi will happily flag
    an empty slot as recognized with 5 % confidence on shadows /
    fingerprints / dust; alerting on those was the false-positive
    that buzzed the table 10× at the start of an empty FTQ deal."""
    doc = _pi_fetch_slots(s)
    if doc is None:
        return []
    occupied = []
    threshold = getattr(s, "pi_confidence_threshold", 0.70)
    for entry in doc.get("slots", []):
        slot_num = entry.get("slot")
        if slot_num is None:
            continue
        if not (entry.get("recognized") and entry.get("rank") and entry.get("suit")):
            continue
        try:
            conf = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < threshold:
            continue
        occupied.append(slot_num)
    return sorted(occupied)


def _alert_stuck_cards_at_new_hand(s):
    """If new-hand init found cards still seated in scanner slots,
    spawn a daemon thread that beeps the piezo 10 times spaced
    roughly 2 s apart. Caller must have already verified that we're
    starting a hand and the Pi is reachable."""
    occupied = _stuck_slots_at_new_hand(s)
    if not occupied:
        return
    log.log(
        f"[CARDS-STUCK] New hand starting with cards in slots "
        f"{occupied} — alerting dealer (10 beeps over 20s)"
    )

    def _run():
        # 10 beeps: 1 s on / 1 s off each = 20 s total.
        for _ in range(10):
            try:
                _pi_buzz(s, n=1, on_time=1.0, off_time=1.0)
            except Exception as e:
                log.log(
                    f"[CARDS-STUCK] beep failed: {type(e).__name__}: {e}"
                )
                return
            # Bail early if the dealer cleared the slots mid-alert.
            occ = _stuck_slots_at_new_hand(s)
            if not occ:
                log.log("[CARDS-STUCK] Slots cleared — alert stopping early")
                return

    Thread(target=_run, daemon=True).start()

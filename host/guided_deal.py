"""Guided dealing for Pi scanner slots.

Two loop variants — each runs as a daemon thread and owns the
per-slot LEDs + Pi /slots/<n>/scan polling for the duration of
its session:

  - ``_guided_deal_loop`` — strict 1→N order. Used for the
    leading down cards of any game with downs (5 Card Draw =
    5 slots, 7 Card Stud = 2, FTQ = 3, etc.).

  - ``_guided_replace_loop`` — explicit slot list, all LEDs on
    simultaneously, scans whatever arrives in any order. Used
    by draw-phase replacement (5CD / 3 Toed Pete), the
    trailing 7th-street down (7 Card Stud, FTQ), Challenge
    round-2 deal, and Challenge round-3 replace.

The completion transitions branch on game type — challenge
games hand off to the vote phase, mixed up/down games hand off
to the Brio watcher, all-down games go straight to betting.
The cross-module helpers needed by those transitions
(``_announce_poker_hand_bet_first``, ``_table_log_add``,
``_stats_bump``, ``_begin_challenge_vote``,
``_game_is_challenge``) are imported lazily inside the loops
so this module stays leaf-importable.
"""

import time
from threading import Thread

from log_buffer import log
from speech import speech
from pi_scanner import _pi_flash, _pi_slot_led, _pi_slot_scan


GUIDED_GOOD_CONF = 0.50   # at/above this, auto-accept the scan
GUIDED_POLL_S = 0.6       # interval between /slots/<n>/scan polls
# Require this many consecutive present=true scans before firing a verify
# modal. The first presence hit is often a finger or a half-inserted card;
# YOLO can't see it yet. A high-confidence scan short-circuits this wait.
GUIDED_STABLE_SCANS = 3
# After first detecting a card in the slot, wait this long before using any
# scan reading. Gives the dealer time to fully seat the card so YOLO isn't
# fighting motion blur on the first capture. Longer than strictly needed
# to give the dealer time to finish placing the card before YOLO reads —
# shorter values led to verify modals popping with partial-insertion reads.
GUIDED_SETTLE_S = 2.0


def _announce_trailing_done(s):
    """Speak the final wild state (if any) then the bet-first player.

    Called after the trailing down card (7th street for 7CS/FTQ) has
    been dealt to every active player, so the final betting round gets
    one consolidated announcement instead of one stale one after the
    6th-street confirm.
    """
    from poker_announce import _announce_poker_hand_bet_first

    ge = s.game_engine
    if not ge or not ge.current_game:
        return
    # Re-state the current wild label if it's non-default. For FTQ we
    # suppressed mid-round speech on the last up round; this is where
    # the player hears the final wild mapping. ge.wild_label already
    # contains the "are wild" suffix (set as "Queens and {X}s are
    # wild") so speak it verbatim — the earlier bug appended a second
    # " are wild" producing "Queens and sevens are wild are wild".
    label = getattr(ge, "wild_label", "") or ""
    if label and label != "Queens wild":
        speech.say(label)
    _announce_poker_hand_bet_first(s)


def _guided_deal_loop(s):
    """Slot-by-slot dealing for all-down games.

    Turns slot-1 LED on, polls /slots/1/scan until a card is present:
    if YOLO conf >= GUIDED_GOOD_CONF, record the card + advance; if lower,
    blink the LED and open the /table verify modal for Rodney to resolve.
    Strict 1→N order: never looks at slot N+1 until slot N is resolved.

    External code stops the loop by setting s.guided_deal = None.
    """
    from app_state import _stats_bump
    from verify_queue import _table_log_add
    from games.challenge import _begin_challenge_vote, _game_is_challenge

    gd = s.guided_deal
    if gd is None:
        return
    N = gd["num_slots"]
    log.log(f"[GUIDED] Started — {N} slots")
    # Initial LED state: slot 1 solid on, the rest off.
    for n in range(1, N + 1):
        _pi_slot_led(s, n, "on" if n == 1 else "off")

    # Per-slot debounce state: how many consecutive present=true scans
    # we've seen, and the best card guess from any of them. Reset on
    # either present=false (card/finger withdrawn) or after we commit.
    stable_count = 0
    best_card = None
    settled = False  # True after GUIDED_SETTLE_S has elapsed since first present

    while True:
        gd = s.guided_deal
        if gd is None:
            log.log("[GUIDED] Stopped externally")
            return
        expecting = gd["expecting"]
        if expecting > N:
            for n in range(1, N + 1):
                _pi_slot_led(s, n, "off")
            log.log(f"[GUIDED] Complete — {N} slots filled")
            with s.table_lock:
                s.guided_deal = None
                s.table_state_version += 1
            # Per-game post-guided transitions.
            ge = s.game_engine
            first_deal_phase = next(
                (ph for ph in (ge.current_game.phases if ge.current_game else [])
                 if ph.type.value in ("deal", "community")),
                None,
            )
            first_phase_has_up = bool(
                first_deal_phase and "up" in first_deal_phase.pattern
            )
            has_hit_round_game = bool(
                ge.current_game and any(
                    ph.type.value == "hit_round"
                    for ph in ge.current_game.phases
                )
            )
            if s.console_state == "dealing" and _game_is_challenge(ge):
                # Challenge games (High Low High etc.): skip betting, go
                # straight to the vote phase.
                _begin_challenge_vote(s)
            elif (s.console_state == "dealing"
                    and s.console_total_up_rounds == 0
                    and not first_phase_has_up
                    and not has_hit_round_game):
                # Truly all-down games (5CD, 3 Toed Pete): every card is in,
                # auto-advance to betting so next action is Pot is right.
                s.console_state = "betting"
                if s.console_betting_round == 0:
                    s.console_betting_round = 1
                log.log(
                    f"[CONSOLE] All-down deal complete → "
                    f"betting round {s.console_betting_round}"
                )
            elif s.console_state == "dealing" and has_hit_round_game:
                # 7/27 (either variant): the local players still need to
                # flip/reveal an up card onto the table for Brio to scan.
                # Keep state in "dealing" with Brio watching; user presses
                # Confirm Cards once every player's up card is in.
                s.monitoring = True
                s.console_scan_phase = "watching"
                s._zones_with_motion = set()
                dname = ge.get_dealer().name if ge.current_game else "dealer"
                log.log(
                    f"[CONSOLE] 7/27 guided downs done → Brio watching "
                    f"{dname}'s zone for flipped-up cards"
                )
            elif s.console_state == "replacing":
                # Draw-phase refill just completed — back to the current
                # betting round (round 1 still — dealer will Pot-is-right
                # to advance to round 2 once everyone has drawn).
                s.console_state = "betting"
                log.log("[CONSOLE] Draw replacement complete → betting")
            elif (s.console_state == "dealing"
                  and s.console_total_up_rounds > 0
                  and s.console_scan_phase == "idle"):
                # Mixed game (7CS, Hold'em, FTQ): initial downs are in,
                # now hand off to Brio to watch for the up card(s). The
                # baselines captured at Deal-time are still valid — the
                # table had no up cards then and still has none now (local
                # down cards are kept in hand, not placed in up-zones).
                ge = s.game_engine
                s.monitoring = True
                s.console_scan_phase = "watching"
                s._zones_with_motion = set()
                dname = ge.get_dealer().name if ge.current_game else "dealer"
                log.log(f"[CONSOLE] Guided downs done → Brio watching {dname}'s zone")
            return

        # Was this slot's verify modal just resolved? Advance if so.
        with s.table_lock:
            already_filled = expecting in s.rodney_downs
            pv = s.pending_verify
            waiting_verify = pv is not None and pv.get("slot") == expecting

        if already_filled:
            _pi_slot_led(s, expecting, "off")
            with s.table_lock:
                gd["expecting"] = expecting + 1
                s.table_state_version += 1
            if expecting + 1 <= N:
                _pi_slot_led(s, expecting + 1, "on")
            stable_count = 0
            best_card = None
            settled = False
            continue

        if waiting_verify:
            time.sleep(0.3)
            continue

        result = _pi_slot_scan(s, expecting)
        if result is None:
            time.sleep(1.5)  # Pi unreachable — back off before retrying
            continue

        if not result.get("present"):
            stable_count = 0
            best_card = None
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        # First scan that sees "present" — the card may still be sliding
        # into place. Wait GUIDED_SETTLE_S before trusting any reading so
        # YOLO isn't hitting motion blur or a half-inserted card.
        if not settled:
            time.sleep(GUIDED_SETTLE_S)
            settled = True
            continue

        stable_count += 1
        card = result.get("card")
        if card:
            conf = float(card.get("confidence", 0.0))
            # Track the best (highest-conf) guess across debounce scans.
            if best_card is None or conf > float(best_card.get("confidence", 0.0)):
                best_card = card
            code = f"{card['rank']}{card['suit'][0]}"
            # Short-circuit: any scan above the auto-accept threshold commits
            # immediately, no need to wait for more stability.
            if conf >= GUIDED_GOOD_CONF:
                with s.table_lock:
                    s.rodney_downs[expecting] = {
                        "rank": card["rank"],
                        "suit": card["suit"],
                        "confidence": round(conf, 2),
                    }
                    s.pi_prev_slots[expecting] = code
                    gd["expecting"] = expecting + 1
                    s.table_state_version += 1
                _stats_bump(s, "pi_auto")
                _table_log_add(s, f"Slot {expecting}: {code} (auto, {int(conf*100)}%)")
                _pi_slot_led(s, expecting, "off")
                if expecting + 1 <= N:
                    _pi_slot_led(s, expecting + 1, "on")
                stable_count = 0
                best_card = None
                settled = False
                continue

        # Low-conf or no-card: give the card time to settle before popping
        # the verify modal. Early "present" ticks from a finger or a half-
        # inserted card would otherwise fire the modal with empty fields.
        if stable_count < GUIDED_STABLE_SCANS:
            time.sleep(GUIDED_POLL_S)
            continue

        # Debounce window elapsed without a high-confidence read. If YOLO
        # never recognized anything at all across the whole window, the
        # scanner is probably misreporting present=true for an empty slot
        # (e.g., brightness threshold too high). Don't open a modal with
        # an empty guess — log once, reset state, and keep polling.
        if best_card is None:
            log.log(
                f"[GUIDED] Slot {expecting}: present but nothing recognized "
                f"after {GUIDED_STABLE_SCANS} scans — Pi presence threshold "
                f"may be too high; continuing to poll"
            )
            stable_count = 0
            settled = False
            time.sleep(GUIDED_POLL_S)
            continue

        conf = float(best_card.get("confidence", 0.0))
        guess = {
            "rank": best_card["rank"],
            "suit": best_card["suit"],
            "confidence": round(conf, 2),
        }
        prompt = (
            f"Slot {expecting}: low confidence ({int(conf*100)}%). "
            f"Confirm or correct."
        )

        with s.table_lock:
            s.pending_verify = {
                "slot": expecting,
                "guess": guess,
                "prompt": prompt,
                "image_url": f"/api/table/slot_image/{expecting}",
            }
            if guess["rank"]:
                s.slot_pending[expecting] = dict(guess)
            s.table_state_version += 1
        _table_log_add(
            s,
            f"Slot {expecting}: verify needed"
            + (f" ({int(guess['confidence']*100)}%)" if guess["rank"] else ""),
        )
        _pi_slot_led(s, expecting, "blink")


def _start_guided_deal(s, num_slots: int):
    """Kick off the guided deal thread. Safe to call again; becomes no-op
    if a guided deal is already running."""
    if s.guided_deal is not None:
        return
    with s.table_lock:
        s.guided_deal = {"expecting": 1, "num_slots": num_slots}
        s.table_state_version += 1
    # Hold the flash on for the whole guided session so every /slots/<n>/scan
    # runs under steady lighting without the 300ms warmup each pulse. The
    # Pi's capture_with_flash short-circuits its own warmup when flash.held.
    _pi_flash(s, True)
    t = Thread(target=_guided_deal_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _stop_guided_deal(s):
    """Signal the guided loop to exit and clear LEDs."""
    gd = s.guided_deal
    if gd is None:
        return
    # The guided_deal dict carries either "num_slots" (full deal) or "slots"
    # (explicit list, used by draw-phase replacement). Turn everything off.
    slots = gd.get("slots") or list(range(1, gd.get("num_slots", 0) + 1))
    with s.table_lock:
        s.guided_deal = None
        s.table_state_version += 1
    for n in slots:
        _pi_slot_led(s, n, "off")
    # Release the held flash; regular poller / idle state will manage it.
    _pi_flash(s, False)


def _start_guided_replace(s, slots, previous_cards=None):
    """Kick off the guided loop for a specific slot list (draw-phase
    replacement). Same as _start_guided_deal but iterates through the
    supplied slot numbers in order rather than 1..N.

    previous_cards maps slot_num → code string ("Ah") for the card that
    was in each slot before the replace started. The loop uses it to
    detect "card changed" in case the Pi scan misses the present=false
    moment between the swap."""
    if s.guided_deal is not None:
        return
    ordered = [int(x) for x in slots if isinstance(x, int) or str(x).isdigit()]
    ordered = sorted(set(ordered))
    if not ordered:
        return
    prev = {int(k): str(v) for k, v in (previous_cards or {}).items()}
    with s.table_lock:
        s.guided_deal = {
            "slots": ordered, "index": 0, "mode": "replace",
            "previous_cards": prev,
        }
        s.console_state = "replacing"
        s.table_state_version += 1
    _pi_flash(s, True)
    t = Thread(target=_guided_replace_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _start_guided_trailing_deal(s, slots: list[int]):
    """Kick off guided flow for trailing down cards (7 Card Stud's 7th
    street, Follow the Queen's 7th). Console stays in 'dealing' until the
    loop finishes, then transitions to 'betting' for one final Pot-is-right
    before hand_over."""
    if s.guided_deal is not None:
        return
    ordered = sorted(set(int(x) for x in slots if isinstance(x, int) or str(x).isdigit()))
    if not ordered:
        return
    with s.table_lock:
        s.guided_deal = {"slots": ordered, "index": 0, "mode": "trailing"}
        s.console_state = "dealing"
        s.table_state_version += 1
    _pi_flash(s, True)
    t = Thread(target=_guided_replace_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _start_guided_deal_range(s, slots: list[int]):
    """Challenge round 2: deal new cards into an explicit slot list.
    Slots start empty (no require_empty_first). Completion transitions to
    challenge_vote for the next round of voting."""
    if s.guided_deal is not None:
        return
    ordered = sorted(set(int(x) for x in slots if isinstance(x, int) or str(x).isdigit()))
    if not ordered:
        return
    with s.table_lock:
        s.guided_deal = {"slots": ordered, "index": 0, "mode": "challenge_deal"}
        s.console_state = "dealing"
        s.table_state_version += 1
    _pi_flash(s, True)
    t = Thread(target=_guided_replace_loop, args=(s,), daemon=True)
    s.guided_deal_thread = t
    t.start()


def _guided_replace_loop(s):
    """Variant of _guided_deal_loop driven by an explicit slot list.
    All selected slots light up simultaneously; replacements are scanned
    as they arrive in any order.

    Shared by draw-phase replacement (mode='replace') and the trailing-down
    deal for stud games (mode='trailing'); only the completion transition
    differs."""
    from app_state import _stats_bump
    from verify_queue import _table_log_add
    from games.challenge import _begin_challenge_vote, _game_is_challenge

    gd = s.guided_deal
    if gd is None or "slots" not in gd:
        return
    slots = list(gd["slots"])
    mode = gd.get("mode", "replace")
    log.log(f"[GUIDED/{mode}] Started — slots {slots}")
    # Light every selected slot at once. The main loop polls each
    # uncommitted slot in round-robin, so the dealer can place cards in
    # any order.
    now0 = time.time()
    for n in slots:
        _pi_slot_led(s, n, "on")

    # In replace mode the old card may still be physically in the slot
    # when guided starts — require a present=false transition before we
    # accept a present=true reading, otherwise the old card gets re-
    # committed as "new". Trailing mode starts from an empty slot.
    require_empty_first = (mode == "replace")
    per_slot = {
        n: {
            "saw_empty": not require_empty_first,
            "stable_count": 0,
            "best_card": None,
            "settle_until": None,
            "settled": False,
            "committed": False,
            # Last LED-on POST timestamp. Re-sent periodically so a Pi
            # restart mid-replace doesn't strand the LED off.
            "led_on_ts": now0,
        }
        for n in slots
    }
    LED_REFRESH_S = 10.0

    def _remaining():
        return [n for n in slots if not per_slot[n]["committed"]]

    with s.table_lock:
        gd["remaining"] = _remaining()
        gd["total"] = len(slots)
        s.table_state_version += 1

    while True:
        gd = s.guided_deal
        if gd is None:
            log.log(f"[GUIDED/{mode}] Stopped externally")
            return

        if all(st["committed"] for st in per_slot.values()):
            for n in slots:
                _pi_slot_led(s, n, "off")
            log.log(f"[GUIDED/{mode}] Complete")
            ge = s.game_engine
            is_challenge = _game_is_challenge(ge)
            with s.table_lock:
                s.guided_deal = None
                s.table_state_version += 1
                if mode == "challenge_deal":
                    # Challenge round 2 deal done — enter vote.
                    pass  # call below, after lock released
                elif mode == "trailing":
                    # 7CS/FTQ 7th street done — one more betting round before
                    # hand_over. console_trailing_done tells next_round to
                    # skip the trailing branch second time through.
                    s.console_state = "betting"
                    s.console_trailing_done = True
                    log.log(
                        "[CONSOLE] Trailing down deal complete → final betting"
                    )
                    # The wild-card and bet-first announcements were
                    # deferred from the last-up-round Confirm Cards so
                    # the player hears them only once, after the final
                    # card is on the table.
                    _announce_trailing_done(s)
                elif s.console_state == "replacing":
                    if is_challenge:
                        # Challenge round 3 replace done — enter final vote.
                        pass  # call below, after lock released
                    else:
                        # Record this draw as done. Multi-draw games (3 Toed
                        # Pete) use this to know whether another DRAW phase
                        # follows; post-draw betting round number is the
                        # count of draws completed so far + 1.
                        s.rodney_draws_done += 1
                        s.console_state = "betting"
                        s.console_betting_round = s.rodney_draws_done + 1
                        log.log(
                            f"[CONSOLE] Draw {s.rodney_draws_done} replacement "
                            f"done → betting round {s.console_betting_round}"
                        )
            # Challenge transitions call _begin_challenge_vote outside the
            # lock (it takes the lock internally via _bump_table_version).
            if mode == "challenge_deal" or (
                    mode == "replace" and is_challenge):
                _begin_challenge_vote(s)
            return

        progress = False
        now = time.time()
        for n in slots:
            st = per_slot[n]
            if st["committed"]:
                continue

            # Periodic LED refresh — if the Pi restarted between the
            # initial LED-on POST and now, the slot LED would have gone
            # dark. Re-send every LED_REFRESH_S seconds for any slot
            # that's not currently in the verify modal (where it should
            # be blinking instead).
            if (st["stable_count"] == 0
                    and now - st["led_on_ts"] >= LED_REFRESH_S):
                _pi_slot_led(s, n, "on")
                st["led_on_ts"] = now

            # Settle window — skip scans while the dealer is still seating
            # the card we just saw arrive.
            if st["settle_until"] is not None and now < st["settle_until"]:
                continue

            with s.table_lock:
                already_filled = n in s.rodney_downs
                pv = s.pending_verify
                waiting_verify_here = pv is not None and pv.get("slot") == n
                any_pending_verify = pv is not None

            # Frontend correction arrived via /api/console/correct —
            # rodney_downs[n] was filled out from under us.
            if already_filled and not waiting_verify_here:
                _pi_slot_led(s, n, "off")
                st["committed"] = True
                with s.table_lock:
                    gd["remaining"] = _remaining()
                    s.table_state_version += 1
                progress = True
                continue

            # This slot is currently parked in the verify modal.
            if waiting_verify_here:
                continue

            result = _pi_slot_scan(s, n)
            if result is None:
                continue

            present = bool(result.get("present"))
            cur = result.get("card") or {}
            cur_code = (
                f"{cur['rank']}{cur['suit'][0]}"
                if cur.get("rank") and cur.get("suit") else ""
            )
            log.log(
                f"[GUIDED/{mode}] Slot {n}: present={present} "
                f"card={cur_code or '-'} "
                f"conf={cur.get('confidence', 0.0):.2f} "
                f"saw_empty={st['saw_empty']}"
            )

            if not present:
                if not st["saw_empty"]:
                    log.log(
                        f"[GUIDED/{mode}] Slot {n}: empty — ready for new card"
                    )
                st["saw_empty"] = True
                st["stable_count"] = 0
                st["best_card"] = None
                st["settle_until"] = None
                st["settled"] = False
                continue

            # Replace mode: old card may still be in the slot at loop
            # start. Accept the scan only after (a) present=false was
            # seen, or (b) YOLO reads a DIFFERENT code than the previous
            # — user swapped faster than our polling caught empty.
            if not st["saw_empty"]:
                prev_code = gd.get("previous_cards", {}).get(n, "")
                if cur_code and prev_code and cur_code != prev_code:
                    log.log(
                        f"[GUIDED/{mode}] Slot {n}: card changed "
                        f"{prev_code} → {cur_code} (no empty seen) — accepting"
                    )
                    st["saw_empty"] = True
                else:
                    continue

            # First present=true after empty → start the settle clock and
            # skip this scan. Subsequent passes wait until it expires.
            if not st["settled"]:
                if st["settle_until"] is None:
                    st["settle_until"] = now + GUIDED_SETTLE_S
                    continue
                if now < st["settle_until"]:
                    continue
                st["settled"] = True
                continue

            st["stable_count"] += 1
            card = result.get("card")
            if card:
                conf = float(card.get("confidence", 0.0))
                if (st["best_card"] is None
                        or conf > float(st["best_card"].get("confidence", 0.0))):
                    st["best_card"] = card
                code = f"{card['rank']}{card['suit'][0]}"
                if conf >= GUIDED_GOOD_CONF:
                    with s.table_lock:
                        s.rodney_downs[n] = {
                            "rank": card["rank"],
                            "suit": card["suit"],
                            "confidence": round(conf, 2),
                        }
                        s.pi_prev_slots[n] = code
                        s.table_state_version += 1
                    _stats_bump(s, "pi_auto")
                    _table_log_add(
                        s,
                        f"Slot {n} (replace): {code} (auto, {int(conf*100)}%)",
                    )
                    _pi_slot_led(s, n, "off")
                    st["committed"] = True
                    with s.table_lock:
                        gd["remaining"] = _remaining()
                        s.table_state_version += 1
                    progress = True
                    continue

            if st["stable_count"] < GUIDED_STABLE_SCANS:
                continue

            if st["best_card"] is None:
                log.log(
                    f"[GUIDED/{mode}] Slot {n}: present but nothing "
                    f"recognized after {GUIDED_STABLE_SCANS} scans — continuing"
                )
                st["stable_count"] = 0
                st["settle_until"] = None
                st["settled"] = False
                continue

            # Low-confidence handoff to the verify modal. Only one slot
            # can own pending_verify at a time; if another slot already
            # does, wait our turn.
            if any_pending_verify:
                continue

            best = st["best_card"]
            conf = float(best.get("confidence", 0.0))
            guess = {
                "rank": best["rank"],
                "suit": best["suit"],
                "confidence": round(conf, 2),
            }
            prompt = (
                f"Slot {n} (replacement): low confidence "
                f"({int(conf*100)}%). Confirm or correct."
            )
            with s.table_lock:
                s.pending_verify = {
                    "slot": n,
                    "guess": guess,
                    "prompt": prompt,
                    "image_url": f"/api/table/slot_image/{n}",
                }
                if guess["rank"]:
                    s.slot_pending[n] = dict(guess)
                s.table_state_version += 1
            _pi_slot_led(s, n, "blink")

        if not progress:
            time.sleep(GUIDED_POLL_S)

"""Voice command pipeline.

Three pieces:

  - ``_derive_voice_phase`` — maps the current console state
    onto a voice-grammar phase (``pre_game`` / ``up_round`` /
    ``pre_confirm`` / ``pre_pot`` / ``challenge_*`` / ``other``).
    Phase is the gate that decides which utterances are
    actionable right now and which are logged-and-ignored.

  - ``_process_voice_command`` — the SpeechListener's callback.
    Takes a parsed command and dispatches it to the matching
    /api/console/* endpoint via _voice_post, applying the
    phase filter so a stray "Pot is right" mid-deal doesn't
    short-circuit anything.

  - ``_speak_voice_status`` + ``_schedule_voice_status_speech``
    — debounced delta-readback. After the dealer voice-calls a
    flurry of cards, 0.8 s of quiet triggers a per-zone
    readback so they can verify the round by ear.

Cross-module helpers (``_handle_challenge_winner``,
``_state``) are imported lazily inside the entry points so
this module stays leaf-importable.
"""

import json
from threading import Lock, Timer

from log_buffer import log
from speech import speech


def _derive_voice_phase(s):
    """Map the current console state to the voice-grammar phase.

    Returns one of:
      'pre_game'     — no game in progress; accepts game-selection commands
      'up_round'     — cards being scanned; accepts "{player}, {card}"
      'pre_confirm'  — every watched zone has a card; accepts Correction / Confirmed
      'pre_pot'      — betting round in progress; accepts Pot Is Right + Fold
      'other'        — anything else (draw / replacing / hand_over / idle mid-setup,
                       OR poker night not yet started — Whisper is already
                       transcribing in --listen mode and a stray hallucination
                       would otherwise auto-deal a hand before the dealer hits
                       Start)
    """
    if not getattr(s, "night_active", False):
        return "other"
    ge = s.game_engine
    if ge is None or ge.current_game is None:
        return "pre_game"
    if s.console_state == "dealing":
        if s.console_scan_phase == "scanned":
            return "pre_confirm"
        return "up_round"
    if s.console_state == "betting":
        return "pre_pot"
    if s.console_state == "challenge_vote":
        return "challenge_vote"
    if s.console_state == "challenge_resolve":
        return "challenge_resolve"
    return "other"


def _voice_post(path, body=None):
    """Internal-HTTP helper so voice commands go through the same
    endpoints the buttons do — guarantees identical side effects
    (state transitions, announce, stats, table version bumps)."""
    import urllib.request
    url = f"http://localhost:8888{path}"
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=3).read()
        return True
    except Exception as e:
        log.log(f"[VOICE] POST {path} failed: {type(e).__name__}: {e}")
        return False


# Debounce timer for the voice-readback. After any voice-driven card
# call we re-arm this timer; on 0.8 s of quiet it walks every active
# zone in deal order and speaks back what's currently recognized.
# This lets the dealer verify the round by ear instead of visually
# scanning the console while they're still calling the next card.
_voice_status_timer = None
_voice_status_lock = Lock()


def _schedule_voice_status_speech(delay_s: float = 0.8) -> None:
    """Arm (or re-arm) the debounce timer that will speak which
    players are still waiting once the flurry of voice card calls
    goes quiet."""
    global _voice_status_timer
    with _voice_status_lock:
        if _voice_status_timer is not None:
            _voice_status_timer.cancel()
        _voice_status_timer = Timer(delay_s, _speak_voice_status)
        _voice_status_timer.daemon = True
        _voice_status_timer.start()


def _speak_voice_status() -> None:
    """Delta-only readback. Each card is announced at most once per
    round: Brio's own recognition speech covers zones it scans,
    voice-driven corrections get announced here when the debounce
    fires, and corrections that change a card's value re-announce the
    new value. When every active zone is filled, announce 'All cards
    in' once so the dealer knows to say Confirmed.

    Runs on the debounce Timer thread — the global _state may have
    moved on between the voice calls and the timer firing, so always
    re-check phase and re-derive the watched set before speaking."""
    import overhead_test
    s = overhead_test._state
    if s is None:
        return
    phase = _derive_voice_phase(s)
    if phase not in ("up_round", "pre_confirm"):
        return  # round probably ended; nothing to announce
    impl = getattr(s, "current_game_impl", None)
    if impl is not None:
        scan_names, _stand = impl.zones_to_scan(s)
    else:
        scan_names = list(s.console_active_players)
    # Walk in deal order (clockwise from dealer's left) so multiple
    # deltas (rare but possible) play back in a natural sequence.
    ge = s.game_engine
    dealer_idx = ge.dealer_index
    deal_order = [
        ge.players[(dealer_idx + i) % len(ge.players)].name
        for i in range(1, len(ge.players) + 1)
    ]
    ordered = [n for n in deal_order if n in scan_names]

    # Reset the per-round announce-tracker when the round advances.
    current_round = s.console_up_round + 1
    if getattr(s, "_voice_announced_round", -1) != current_round:
        s._voice_announced_cards = {}
        s._voice_announced_all_in = False
        s._voice_inferred_this_round = set()
        s._voice_announced_round = current_round
    if not hasattr(s, "_voice_inferred_this_round"):
        s._voice_inferred_this_round = set()

    waiting = []
    for name in ordered:
        card = s.monitor.last_card.get(name, "")
        zstate = s.monitor.zone_state.get(name, "")
        if not card or card == "No card":
            waiting.append(name)
            continue
        # Brio's recognition path already called speech.say on its
        # own; just track that the card was announced so a later
        # voice-correction of the same zone can detect the delta.
        if zstate == "recognized":
            s._voice_announced_cards[name] = card
            continue
        # zone_state = "corrected" — voice call or manual console
        # correction. Speak only if the value differs from what we
        # already announced this round. Prefix "Inferred:" when the
        # card was resolved from an orphan voice call (no player
        # name spoken), so the dealer knows this one is a guess and
        # worth verifying before they Confirm.
        if s._voice_announced_cards.get(name) != card:
            if name in s._voice_inferred_this_round:
                speech.say(f"Inferred: {name}, {card}")
                log.log(f"[VOICE] Announce: Inferred: {name}, {card}")
                s._voice_inferred_this_round.discard(name)
            else:
                speech.say(f"{name}, {card}")
                log.log(f"[VOICE] Announce: {name}, {card}")
            s._voice_announced_cards[name] = card

    # "All cards in" fires once per round, the first time every
    # active zone has a card.
    if not waiting and not s._voice_announced_all_in:
        speech.say("All cards in")
        log.log("[VOICE] All cards in")
        s._voice_announced_all_in = True


def _process_voice_command(cmd):
    """Phase-filtered dispatch for a single parsed voice command.

    Commands spoken in the wrong phase are logged and ignored — no
    action fires. That way a stray "Confirmed" during a betting round
    doesn't accidentally skip ahead, and a "Pot is right" during deal
    doesn't short-circuit the scan phase.
    """
    import overhead_test
    s = overhead_test._state
    if s is None:
        return
    from speech_recognition_module import (
        GameCommand, RepeatGameCommand, CardCallCommand, InferredCardCommand,
        CorrectionCommand, ConfirmCommand, PotIsRightCommand,
        ScanCardsCommand, FoldCommand,
        PassCommand, GoOutCommand, ChallengeWinnerCommand,
        UnrecognizedCommand,
    )
    from games.challenge import _handle_challenge_winner
    phase = _derive_voice_phase(s)

    if isinstance(cmd, GameCommand):
        if phase != "pre_game":
            log.log(f"[VOICE] Ignoring 'game is {cmd.game_name}' in phase {phase}")
            return
        log.log(f"[VOICE] Starting new hand: {cmd.game_name}")
        _voice_post("/api/console/deal", {"game": cmd.game_name})
        return

    if isinstance(cmd, RepeatGameCommand):
        if phase != "pre_game":
            log.log(f"[VOICE] Ignoring 'same game again' in phase {phase}")
            return
        last = getattr(s, "last_game_name", "") or ""
        if not last:
            log.log("[VOICE] 'Same game again' heard but no previous game recorded")
            speech.say("No previous game to repeat")
            return
        log.log(f"[VOICE] Repeating previous game: {last}")
        _voice_post("/api/console/deal", {"game": last})
        return

    if isinstance(cmd, CardCallCommand):
        # Voice-assigning a card during an up-card round (or pre-confirm
        # if the user didn't use the "Correction:" prefix). Routes
        # through /api/console/correct so the monitor picks it up
        # exactly like a typed correction — locks the zone from the
        # next Brio scan, updates training_data, etc.
        if phase not in ("up_round", "pre_confirm"):
            log.log(
                f"[VOICE] Ignoring card call "
                f"'{cmd.player}, {cmd.rank}{cmd.suit[0]}' in phase {phase}"
            )
            return
        log.log(f"[VOICE] {cmd.player}: {cmd.rank} of {cmd.suit}")
        _voice_post("/api/console/correct", {
            "corrections": [{"player": cmd.player, "rank": cmd.rank, "suit": cmd.suit}],
        })
        # Arm the debounced readback. Dealer says all the cards
        # (quickly, in sequence); once they pause for 0.8 s, the
        # system walks every active zone in deal order and speaks
        # back each player's current card so the dealer can verify
        # by ear without looking at the console. No per-card echo —
        # that was overlapping with their next utterance.
        _schedule_voice_status_speech()
        return

    if isinstance(cmd, InferredCardCommand):
        # Dealer spoke just rank+suit without a name. Resolve to the
        # next unfilled zone in deal order (clockwise from dealer's
        # left, dealer last). Skip players who've folded/busted or
        # whose zone already has a card.
        if phase not in ("up_round", "pre_confirm"):
            log.log(
                f"[VOICE] Ignoring orphan card '{cmd.rank}{cmd.suit[0]}' "
                f"in phase {phase}"
            )
            return
        ge = s.game_engine
        dealer_idx = ge.dealer_index
        deal_order = [
            ge.players[(dealer_idx + i) % len(ge.players)].name
            for i in range(1, len(ge.players) + 1)
        ]
        impl = getattr(s, "current_game_impl", None)
        if impl is not None:
            scan_names, _stand = impl.zones_to_scan(s)
        else:
            scan_names = list(s.console_active_players)
        scan_set = set(scan_names)
        target = None
        for name in deal_order:
            if name not in scan_set:
                continue
            existing = s.monitor.last_card.get(name, "")
            # Only target zones that haven't been resolved yet
            # (last_card == ""). Zones marked "No card" are
            # explicit Claude/YOLO/user verdicts that the zone is
            # empty — routing an orphan rank/suit there caused the
            # 7/27 round-2 phantom 7-of-Hearts that sent Steve a
            # card he never received.
            if not existing:
                target = name
                break
        if target is None:
            log.log(
                f"[VOICE] Can't infer player for '{cmd.rank}{cmd.suit[0]}' — "
                f"every active zone already has a card"
            )
            # Audible cue: dealer said a card we can't place. Usually
            # means Whisper dropped a player name earlier and the
            # subsequent orphan has no home. Short "orphan card" lets
            # them know without staring at the log.
            speech.say("Orphan card")
            return
        log.log(
            f"[VOICE] Inferred {target} for '{cmd.rank}{cmd.suit[0]}' "
            f"(next in deal order)"
        )
        # Immediate warning so the dealer catches the orphan before
        # the debounced readback fires. The readback will then speak
        # "Inferred: <target>, <rank> of <suit>" (the Inferred prefix
        # is applied via the _voice_inferred_this_round tracker).
        speech.say("Orphan card")
        if not hasattr(s, "_voice_inferred_this_round"):
            s._voice_inferred_this_round = set()
        s._voice_inferred_this_round.add(target)
        _voice_post("/api/console/correct", {
            "corrections": [{"player": target, "rank": cmd.rank, "suit": cmd.suit}],
        })
        _schedule_voice_status_speech()
        return

    if isinstance(cmd, CorrectionCommand):
        if phase != "pre_confirm":
            log.log(
                f"[VOICE] Ignoring 'Correction: {cmd.player}, "
                f"{cmd.rank}{cmd.suit[0]}' in phase {phase}"
            )
            return
        log.log(f"[VOICE] Correction: {cmd.player} {cmd.rank} of {cmd.suit}")
        _voice_post("/api/console/correct", {
            "corrections": [{"player": cmd.player, "rank": cmd.rank, "suit": cmd.suit}],
        })
        _schedule_voice_status_speech()
        return

    if isinstance(cmd, ConfirmCommand):
        if phase != "pre_confirm":
            log.log(f"[VOICE] Ignoring 'Confirmed' in phase {phase}")
            return
        log.log("[VOICE] Confirmed → /api/console/confirm")
        _voice_post("/api/console/confirm")
        return

    if isinstance(cmd, PotIsRightCommand):
        if phase != "pre_pot":
            log.log(f"[VOICE] Ignoring 'Pot is right' in phase {phase}")
            return
        log.log("[VOICE] Pot is right → /api/console/next_round")
        _voice_post("/api/console/next_round")
        return

    if isinstance(cmd, ScanCardsCommand):
        # Manual scan trigger. Allow during any phase that has a
        # game running and zones to scan — the dealer might want
        # to re-scan during up_round (initial scan) or pre_confirm
        # (re-scan after a missing-card prompt).
        if phase not in ("up_round", "pre_confirm"):
            log.log(f"[VOICE] Ignoring 'Scan cards' in phase {phase}")
            return
        log.log("[VOICE] Scan cards → /api/console/force_scan")
        _voice_post("/api/console/force_scan")
        return

    if isinstance(cmd, FoldCommand):
        if phase != "pre_pot":
            log.log(f"[VOICE] Ignoring '{cmd.player} folds' in phase {phase}")
            return
        log.log(f"[VOICE] {cmd.player} folds → /api/table/fold")
        _voice_post("/api/table/fold", {"player": cmd.player, "folded": True})
        return

    if isinstance(cmd, (PassCommand, GoOutCommand)):
        # Challenge votes are now dealer-driven via console buttons;
        # voice is too unreliable to drive state. Log the utterance
        # so we can review what Whisper heard but don't act on it.
        log.log(
            f"[VOICE] {type(cmd).__name__} from voice ignored — "
            f"use console Pass/Out buttons instead "
            f"(heard: {cmd.raw_text!r})"
        )
        return

    if isinstance(cmd, ChallengeWinnerCommand):
        if phase != "challenge_resolve":
            log.log(
                f"[VOICE] Ignoring '{cmd.player} wins' in phase {phase}"
            )
            return
        _handle_challenge_winner(s, cmd.player)
        return

    if isinstance(cmd, UnrecognizedCommand):
        log.log(f"[VOICE] ? {cmd.raw_text!r}")
        return

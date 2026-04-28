"""Overhead Brio camera dealer-zone watcher.

Owns the per-frame logic that drives the auto-scan trigger and
the missing-card prompt:

  - ``_brio_player_names`` — which player zones the watcher
    should be looking at this hand.
  - ``_zone_presence_metric`` — fraction of high-contrast
    pixels in a zone vs its baseline. Used both as a "is
    something there?" check and reused by zone_monitor's
    YOLO-hallucination guard (mirrored constants there).
  - ``_all_active_zones_present`` — gate that ensures every
    non-dealer zone has card content before the auto-trigger
    is allowed to fire (covers 7/27 hits where the dealer may
    place their own card before everyone else's).
  - ``_auto_scan_all_zones`` — kick the same batch-scan
    pipeline that /api/console/force_scan uses.
  - ``_console_watch_dealer`` — bg_loop's per-frame entry
    point. Drives both the stability counter on the dealer's
    zone and the once-per-round "Missing cards" prompt.
  - ``_console_rescan_missing`` — thin wrapper for legacy
    code paths that re-scan a small set of zones.

No dependencies on overhead_test internals; this is a leaf
module the rest of the system imports from.
"""

from threading import Thread

import cv2
import numpy as np

from log_buffer import log
from speech import speech


ZONE_STABLE_FRAMES = 3
ZONE_MAX_EMPTY_SCANS = 3
# Frame-to-frame stability check uses a tighter threshold than the
# baseline-comparison check. A still scene at 30 fps has mean
# frame-to-frame diff of ~1–5; auto-exposure jitter can spike to
# 10–15, but a real card placement / hand sweep is well above that.
# Keeping this lower than monitor.threshold (which is for "is the
# zone non-empty?") lets us tolerate exposure jitter without
# resetting the stability count.
ZONE_STABILITY_THRESHOLD = 15
ZONE_DIAG_INTERVAL_S = 5.0
# Fraction-of-high-contrast-pixels presence detector. The 8" zone
# is much larger than the card (~17 % area), so mean-diff vs
# baseline is dominated by unchanged felt and barely moves when a
# card is added. Instead, count pixels whose max-channel diff
# exceeds a per-pixel cutoff and require a meaningful fraction of
# the zone to be card-like. Cards produce well over the cutoff in
# the card area (suit / number prints); empty-zone sensor noise
# stays well below the fraction.
ZONE_PRESENCE_PIXEL_DIFF = 40
ZONE_PRESENCE_FRACTION = 0.04


def _brio_player_names(s):
    """Active players whose Brio zones the overhead watcher should scan.

    For both local and remote players: the dealer places face-up cards
    in the players Brio zone for all players to see (Rodneys flipped-
    up card in 7/27, his up cards in stud games, etc.). So every active
    player contributes a Brio zone.
    """
    return set(s.console_active_players)


def _zone_presence_metric(crop, baseline):
    """Return (fraction_high, mean_diff) for crop vs its baseline.
    fraction_high is the share of pixels whose max-channel absolute
    diff exceeds ZONE_PRESENCE_PIXEL_DIFF; mean_diff is the legacy
    scalar kept for diagnostic logging."""
    absdiff = cv2.absdiff(crop, baseline)
    if absdiff.ndim == 3:
        per_pixel = absdiff.max(axis=2)
    else:
        per_pixel = absdiff
    total = per_pixel.size
    if total == 0:
        return 0.0, 0.0
    high = int((per_pixel > ZONE_PRESENCE_PIXEL_DIFF).sum())
    return high / total, float(absdiff.mean())


def _all_active_zones_present(s, frame, watched, dealer_name):
    """Return True iff every watched zone (excluding the dealer's
    own) is either already recognized / corrected OR has non-empty
    content (fraction-of-high-contrast pixels above the presence
    threshold).

    Used as a second gate on the dealer-zone auto-scan trigger:
    the dealer-zone-stable signal alone isn't enough in games
    where the dealer doesn't reliably fill last (e.g. 7/27 where
    the dealer may flip their own card before dealing to others).
    Waiting for every other zone to actually contain a card stops
    the auto-scan from running over mostly-empty zones and asking
    Claude to identify nothing.
    """
    monitor = s.monitor
    for nm in watched:
        if nm == dealer_name:
            continue
        if monitor.zone_state.get(nm) in ("recognized", "corrected"):
            continue
        z = next((zz for zz in s.cal.zones if zz["name"] == nm), None)
        if z is None:
            continue
        baseline = monitor.baselines.get(nm)
        if baseline is None:
            continue
        crop = monitor._crop(frame, z)
        if crop is None or crop.size == 0 or crop.shape != baseline.shape:
            continue
        fr, _db = _zone_presence_metric(crop, baseline)
        if fr < ZONE_PRESENCE_FRACTION:
            return False
    return True


def _auto_scan_all_zones(s, frame, watched):
    """Internal helper: fire the same batch scan that
    /api/console/force_scan does. Called when the dealer's zone
    auto-trigger fires."""
    monitor = s.monitor
    s._dealer_zone_trigger_fired = True
    log.log(
        f"[CONSOLE] {s.game_engine.get_dealer().name}'s zone stable "
        f"— auto-scanning all zones"
    )
    zone_crops = {}
    for z in s.cal.zones:
        nm = z["name"]
        if nm not in watched:
            continue
        if monitor.zone_state.get(nm) == "corrected":
            continue
        c = monitor._crop(frame, z)
        if c is None or c.size == 0:
            continue
        zone_crops[nm] = c.copy()
        monitor.pending[nm] = True
    if not zone_crops:
        return
    s.console_scan_phase = "scanned"
    monitor.open_speech_gate()
    s._dealer_zone_done = True
    Thread(
        target=monitor._recognize_batch,
        args=(zone_crops,),
        daemon=True,
    ).start()


def _console_watch_dealer(s, frame):
    """Watch the dealer's zone for an auto-scan trigger AND keep
    state for the missing-card prompt.

    Two scan paths now coexist:
      1. Manual — dealer hits the Scan button or says "Scan cards",
         which posts /api/console/force_scan. Always available.
      2. Auto — when the dealer's own zone shows ZONE_STABLE_FRAMES
         consecutive stable, non-empty frames, fire the same batch
         scan. The dealer-zone gate keeps arm sweeps over PLAYER
         zones from triggering anything (the original problem with
         per-zone streaming) — in stud / FTQ / one-up 7/27 the
         dealer fills their own zone last, so dealer-zone-stable is
         a clean "deal complete" signal. (Even when Rodney is the
         engine-side dealer, the physical dealer at the table fills
         Rodney's Brio zone last in the rotation — same trigger.)

    Auto-trigger is suppressed for stand-allowed rounds (7/27 hit)
    where the dealer never places a card — those rely on manual
    Scan. Otherwise it runs whenever the dealer's name is among
    the active Brio zones (which covers every game variant).

    Either path gets the same downstream behavior: speech gate
    opens, recognition runs, and the missing-card prompt fires
    once if any zones came back empty.
    """
    phase = s.console_scan_phase
    if phase in ("idle", "confirmed"):
        return

    ge = s.game_engine
    dealer_name = ge.get_dealer().name
    brio_names = _brio_player_names(s)
    impl = s.current_game_impl
    if impl is not None:
        scan_names, stand_allowed = impl.zones_to_scan(s)
    else:
        scan_names, stand_allowed = list(brio_names), False
    watched = set(scan_names) & set(brio_names)
    if not watched:
        return

    monitor = s.monitor

    # Auto-trigger: dealer's zone has been stable for the threshold
    # frame count → batch-scan everything. One-shot per round (the
    # _dealer_zone_trigger_fired flag), and only while no scan is
    # already running (phase=="watching" — flips to "scanned" the
    # moment we dispatch).
    if (phase == "watching"
            and not stand_allowed
            and dealer_name in brio_names
            and not s._dealer_zone_trigger_fired):
        zone = next(
            (z for z in s.cal.zones if z["name"] == dealer_name), None
        )
        if zone is not None:
            baseline = monitor.baselines.get(dealer_name)
            crop = monitor._crop(frame, zone)
            if (baseline is not None
                    and crop is not None
                    and crop.size > 0
                    and crop.shape == baseline.shape):
                fr, _db = _zone_presence_metric(crop, baseline)
                if fr < ZONE_PRESENCE_FRACTION:
                    monitor.stable_count[dealer_name] = 0
                    monitor.prev_crop[dealer_name] = None
                else:
                    prev = monitor.prev_crop.get(dealer_name)
                    monitor.prev_crop[dealer_name] = crop.copy()
                    if prev is None or prev.shape != crop.shape:
                        monitor.stable_count[dealer_name] = 1
                    else:
                        diff_prev = float(np.mean(cv2.absdiff(crop, prev)))
                        if diff_prev > ZONE_STABILITY_THRESHOLD:
                            monitor.stable_count[dealer_name] = 1
                        else:
                            monitor.stable_count[dealer_name] = (
                                monitor.stable_count.get(dealer_name, 0) + 1
                            )
                            if (monitor.stable_count[dealer_name]
                                    >= ZONE_STABLE_FRAMES):
                                if _all_active_zones_present(
                                        s, frame, watched, dealer_name):
                                    _auto_scan_all_zones(s, frame, watched)

    # Mark "dealer has placed their card" for the missing-card
    # prompt gate. The speech gate itself is gone (announcements
    # now fire immediately per zone), but _dealer_zone_done still
    # gates the "Missing cards: X and Y" prompt below. Stand-
    # allowed rounds (7/27 hit) and remote-dealer hands don't
    # need to wait on the dealer placing — flip the flag right
    # away. Otherwise wait until the dealer's zone is recognized.
    needs_gate = (not stand_allowed) and (dealer_name in brio_names)
    if not needs_gate:
        if not s._dealer_zone_done:
            s._dealer_zone_done = True
    elif not s._dealer_zone_done:
        if monitor.zone_state.get(dealer_name) == "recognized":
            s._dealer_zone_done = True
            log.log("[CONSOLE] Dealer zone recognized")

    # Missing-zone prompt fires once per round, AFTER:
    #   - the speech gate is open (dealer done / stand-allowed), AND
    #   - every queued recognition has resolved (no zone pending).
    # Waiting on pending zones is what stops "Missing cards: Rodney"
    # from speaking 400ms before Rodney's own recognition lands.
    # Skipped entirely on stand-allowed rounds (7/27 hits) — empty
    # zones are expected when a player chooses to stand, so we wait
    # for the dealer's explicit "Scan cards" / Force Scan instead.
    if (s._dealer_zone_done
            and not s._missing_prompt_fired
            and not stand_allowed
            and not any(monitor.pending.get(nm) for nm in watched)):
        s._missing_prompt_fired = True
        missing = []
        for nm in watched:
            if monitor.zone_state.get(nm) in ("recognized", "corrected"):
                continue
            if s._missing_speech_count.get(nm, 0) >= 1:
                continue
            missing.append(nm)
        if missing:
            names = " and ".join(missing)
            log.log(
                f"[CONSOLE] Missing cards: {names} — "
                f"prompting to adjust"
            )
            speech.say(f"{names}, please adjust your card")
            for nm in missing:
                s._missing_speech_count[nm] = 1


def _console_rescan_missing(s, zone_crops):
    """Rescan just the zones where cards moved. Reuses the batch pipeline."""
    s.monitor._recognize_batch(zone_crops)

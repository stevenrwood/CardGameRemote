"""
HTTP server — BaseHTTPRequestHandler routing for the overhead test app.

Every endpoint the browser hits is a method or dispatch arm below.
Handler stays thin: each route either renders a page (loaded from
ui_templates), proxies a Pi call (pi_scanner), or delegates to a
game-flow helper still defined in overhead_test.

Module load order note: this file is imported from the bottom of
overhead_test.py, after every helper it names has been defined. That
is what makes `from overhead_test import ...` below work without a
circular-import failure. ``_state`` is accessed as ``ot._state`` at
request time because the module global is reassigned inside main()
and a `from` import would snapshot the pre-main None.
"""

import base64
import http.server
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2

import overhead_test as ot
from log_buffer import log
from speech import speech
from ui_templates import (
    TABLE_HTML, LOGVIEW_HTML, CONSOLE_HTML,
    SCANNER_TMPL, CALIBRATE_TMPL,
)
from pi_scanner import (
    HOST_CONFIG_PATH,
    _load_host_config, _save_host_config,
    _pi_ping, _pi_flash, _pi_slot_led, _pi_slot_scan,
)
from zone_monitor import TRAINING_DIR
from games import make_game
from overhead_test import (
    PLAYER_NAMES,
    _announce_poker_hand_bet_first,
    _build_table_state,
    _redact_remote_downs,
    _check_follow_the_queen_round,
    _collect_mode_json,
    _collect_redo,
    _collect_start_deal,
    _collect_start_first,
    _deal_mode_json,
    _dedup_round_cards_against_seen,
    _enqueue_down_card_verifies,
    _game_has_draw_phase,
    _get_deal_order,
    _initial_down_count,
    _max_draw_for_game,
    _parse_card_any,
    _parse_card_code,
    _pi_poll_start,
    _pi_poll_stop,
    _process_deal_text,
    _recompute_follow_the_queen,
    _resolve_verify,
    _skip_inactive_dealer,
    _start_collect_mode,
    _start_deal_mode,
    _start_guided_deal,
    _start_guided_deal_range,
    _start_guided_replace,
    _start_guided_trailing_deal,
    _game_is_challenge,
    _fmt_money,
    _log_and_speak,
    _set_challenge_vote,
    _resolve_challenge_round,
    _handle_challenge_winner,
    _begin_challenge_vote,
    _challenge_required_cards,
    _challenge_ante_cents_for,
    _clear_rodney_challenge_leds,
    _forced_betting_limit,
    _betting_limit_label,
    _betting_limit_spoken,
    _speak_ante,
    FORCED_POT_LIMIT_GAMES,
    BETTING_LIMIT_LABELS,
    CHALLENGE_SUBSEQUENT_ANTE_CENTS,
    MAX_PASSES_PER_ROUND,
    _stats_bump,
    _stop_collect_mode,
    _stop_deal_mode,
    _stop_guided_deal,
    _table_log_add,
    _total_draw_phases,
    _trailing_down_slots,
    _update_flash_for_deal_state,
    crop_circle,
    to_jpeg,
)
from overhead_test import CALIBRATION_FILE


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        s = ot._state
        if not s: return self._r(500,"text/plain","Not ready")
        p = self.path.split("?")[0]
        routes = {
            "/": self._page, "/app": self._page,
            "/calibrate": self._calibrate_page,
            "/snapshot": lambda s: self._jpeg(s.latest_frame),
            "/snapshot/cropped": lambda s: self._r(200,"image/jpeg",s.latest_jpg) if s.latest_jpg else self._r(503,"text/plain","wait"),
            "/api/state": self._api_state,
            "/api/log": lambda s: self._r(200,"application/json",json.dumps({"lines":log.get(100)})),
            "/log": lambda s: self._r(200,"text/plain","\n".join(log.get(200))),
            "/logview": self._serve_logview,
            "/calibration": lambda s: self._r(200,"application/json",CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else self._r(404,"text/plain","none"),
            "/training": self._training_list,
            "/console": self._console_page,
            "/table": self._table_page,
            "/table/state": self._table_state,
        }
        if p in routes:
            routes[p](s)
        elif p.startswith("/zone_snap/"):
            name = p[11:]
            crop = s.monitor.recognition_crops.get(name)
            if crop is not None:
                j = to_jpeg(crop, 90)
                if j: self._r(200,"image/jpeg",j)
                else: self._r(500,"text/plain","Encode failed")
            else:
                self._zone_img(s, name)  # fallback to live
        elif p.startswith("/zone/"):
            self._zone_img(s, p[6:])
        elif p.startswith("/training/"):
            self._training_file(p[10:])
        elif p.startswith("/cards/"):
            self._card_asset(p[7:])
        elif p.startswith("/api/table/slot_image/"):
            try:
                slot_n = int(p[len("/api/table/slot_image/"):])
            except ValueError:
                return self._r(404, "text/plain", "bad slot")
            self._proxy_slot_image(s, slot_n)
        else:
            self._r(404,"text/plain","Not found")

    def do_POST(self):
        s = ot._state
        if not s: return self._r(500,"text/plain","Not ready")
        body = self.rfile.read(int(self.headers.get("Content-Length",0))).decode()
        data = json.loads(body) if body else {}
        p = self.path

        if p == "/api/calibrate/save":
            cc = data.get("circle_center")
            s.cal.circle_center = tuple(cc) if cc else None
            s.cal.circle_radius = data.get("circle_radius")
            zr = data.get("zone_radius_px")
            if zr is not None:
                try:
                    s.cal.zone_radius_px = int(zr)
                except (TypeError, ValueError):
                    pass
            s.cal.zones = data.get("zones", [])
            s.cal.save()
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/log/clear":
            log.clear()
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/brio/focus":
            # Live focus tuning + persistence. Body: {"value": N} where N
            # is 0..255 (lower = farther). "auto" or null re-enables AF.
            raw = data.get("value")
            new_val = None
            if isinstance(raw, (int, float)):
                new_val = max(0, min(255, int(raw)))
            elif isinstance(raw, str) and raw.strip().isdigit():
                new_val = max(0, min(255, int(raw.strip())))
            s.capture.set_focus(new_val)
            _save_host_config({"brio_focus": new_val})
            self._r(200, "application/json",
                    json.dumps({"ok": True, "focus": new_val}))

        elif p == "/api/brio/zoom":
            # Live UVC zoom. Body: {"value": N} where N is 100..500.
            # 100 = 1× (fully zoomed out, widest FOV).
            raw = data.get("value")
            new_val = None
            if isinstance(raw, (int, float)):
                new_val = max(100, min(500, int(raw)))
            elif isinstance(raw, str) and raw.strip().isdigit():
                new_val = max(100, min(500, int(raw.strip())))
            if new_val is not None:
                s.capture.set_zoom(new_val)
                _save_host_config({"brio_zoom": new_val})
            self._r(200, "application/json",
                    json.dumps({"ok": True, "zoom": new_val}))

        elif p == "/api/monitor/start":
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
                s.monitoring = True
            self._r(200,"application/json",json.dumps({"monitoring":s.monitoring}))

        elif p == "/api/monitor/stop":
            s.monitoring = False
            self._r(200,"application/json",'{"monitoring":false}')

        elif p == "/api/baselines":
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/start":
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
                s.test_mode = {"zone_idx": 0, "waiting": "card", "result": ""}
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/confirm":
            tm = s.test_mode
            if tm:
                correct = data.get("correct", True)
                if correct:
                    tm["zone_idx"] += 1
                    if tm["zone_idx"] >= len(s.cal.zones):
                        s.test_mode = None
                    else:
                        tm["waiting"] = "card"
                        tm["result"] = ""
                else:
                    # Retry same zone
                    tm["waiting"] = "card"
                    tm["result"] = ""
                    name = s.cal.zones[tm["zone_idx"]]["name"]
                    s.monitor.zone_state[name] = "empty"
                    s.monitor.last_card[name] = ""
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/skip":
            tm = s.test_mode
            if tm:
                tm["zone_idx"] += 1
                if tm["zone_idx"] >= len(s.cal.zones):
                    s.test_mode = None
                else:
                    tm["waiting"] = "card"
                    tm["result"] = ""
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/test/stop":
            s.test_mode = None
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/start":
            _start_deal_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/stop":
            _stop_deal_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/dealer":
            dealer = data.get("dealer", "")
            if s.deal_mode:
                s.deal_mode["dealer"] = dealer
                s.deal_mode["deal_order"] = _get_deal_order(dealer)
                log.log(f"[DEAL] Dealer: {dealer}, order: {' -> '.join(s.deal_mode['deal_order'])}")
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/text":
            text = data.get("text", "")
            _process_deal_text(s, text)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/deal/clear":
            _stop_deal_mode(s)
            _start_deal_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/start":
            _start_collect_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/stop":
            _stop_collect_mode(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/go":
            _collect_start_first(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/pause":
            if s.collect_mode:
                s.collect_mode["phase"] = "paused"
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/resume":
            if s.collect_mode:
                _collect_start_deal(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/collect/redo":
            _collect_redo(s)
            self._r(200,"application/json",'{"ok":true}')

        elif p == "/api/snapshot/save":
            if s.latest_frame is not None:
                cropped = crop_circle(s.latest_frame, s.cal)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = Path(__file__).parent / f"snapshot_{ts}.jpg"
                cv2.imwrite(str(path), cropped)
                log.log(f"Snapshot saved: {path.name}")
            self._r(200,"application/json",'{"ok":true}')

        # --- Observer table view ---

        elif p == "/api/table/pi_start":
            _pi_poll_start(s)
            self._r(200, "application/json", '{"ok":true,"polling":true}')

        elif p == "/api/table/pi_stop":
            _pi_poll_stop(s)
            self._r(200, "application/json", '{"ok":true,"polling":false}')

        elif p == "/api/table/reset_hand":
            _stop_guided_deal(s)
            with s.table_lock:
                s.rodney_downs = {}
                s.rodney_flipped_up = None
                s.slot_pending = {}
                s.slot_empty = {}
                s.verify_queue = []
                s.pending_verify = None
                s.pi_prev_slots = {}
                s.folded_players = set()
                s.freezes = {}
                _table_log_add(s, "Remote hand cleared")
                s.table_state_version += 1
            _update_flash_for_deal_state(s)
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/yolo/recognize":
            # Pi scanner sends a batch of slot crops for YOLO inference on Neo's
            # model + MPS. Body: {"slots": [{"slot": N, "image": "<base64 jpeg>"}, ...]}
            items = data.get("slots", [])
            results = []
            if s.monitor._yolo_model is None:
                return self._r(503, "application/json",
                               json.dumps({"error": "YOLO model not loaded"}))
            for item in items:
                slot_n = item.get("slot")
                img_b64 = item.get("image", "")
                try:
                    raw = base64.b64decode(img_b64)
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                except Exception as e:
                    results.append({"slot": slot_n, "error": f"decode: {e}"})
                    continue
                if crop is None or crop.size == 0:
                    results.append({"slot": slot_n, "error": "empty crop"})
                    continue
                label, conf = s.monitor._recognize_yolo(crop)
                parsed = _parse_card_any(label)
                entry = {"slot": slot_n, "confidence": round(float(conf), 3)}
                if parsed and label != "No card":
                    entry["recognized"] = True
                    entry["rank"] = parsed["rank"]
                    entry["suit"] = parsed["suit"]
                else:
                    entry["recognized"] = False
                results.append(entry)
            self._r(200, "application/json", json.dumps({"slots": results}))

        elif p == "/api/table/flip_up":
            # Rodney picked which of his 2 initial down cards to flip face-up.
            # Keep the card in rodney_downs — the physical card stays in the
            # slot and still counts toward the initial deal count. Mark the
            # slot in rodney_flipped_up, feed the card into last_card so the
            # confirm flow treats it as an up card for this round, and blink
            # the slots LED so the dealer knows which physical card to pull
            # out and show the table.
            try:
                slot_num = int(data.get("slot"))
            except (TypeError, ValueError):
                return self._r(400, "application/json", '{"ok":false,"error":"bad slot"}')
            rodney = next((p2 for p2 in s.game_engine.players if p2.is_remote), None)
            RANK_TO_NAME = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}
            SUIT_TO_NAME = {"spades": "Spades", "hearts": "Hearts",
                            "diamonds": "Diamonds", "clubs": "Clubs"}
            with s.table_lock:
                d = s.rodney_downs.get(slot_num)
                if d is None:
                    return self._r(400, "application/json",
                                   '{"ok":false,"error":"slot not in rodney_downs"}')
                s.rodney_flipped_up = {
                    "rank": d["rank"], "suit": d["suit"], "slot": slot_num,
                }
                if rodney:
                    rank_nm = RANK_TO_NAME.get(d["rank"], d["rank"])
                    suit_nm = SUIT_TO_NAME.get(d["suit"], d["suit"])
                    s.monitor.last_card[rodney.name] = f"{rank_nm} of {suit_nm}"
                    # "corrected" tells the Brio batch scan + missing-card
                    # check to skip Rodney — we already know his flipped-up
                    # card from the Pi slot, no need for YOLO on his zone.
                    s.monitor.zone_state[rodney.name] = "corrected"
                _table_log_add(
                    s,
                    f"Slot {slot_num}: flipping up ({d['rank']}{d['suit'][0]})",
                )
                s.table_state_version += 1
            # Blink the slots LED so the dealer can spot the chosen physical
            # card at a glance. Skipped when the Pi is offline.
            if not s.pi_offline:
                _pi_slot_led(s, slot_num, "blink")
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/table/fold":
            name = str(data.get("player", "")).strip()
            folded = bool(data.get("folded", True))
            ge = s.game_engine
            valid = next((pl.name for pl in ge.players if pl.name.lower() == name.lower()), None)
            if not valid:
                return self._r(400, "application/json", '{"ok":false,"error":"unknown player"}')
            with s.table_lock:
                if folded:
                    s.folded_players.add(valid)
                else:
                    s.folded_players.discard(valid)
                _table_log_add(s, f"{valid} {'folded' if folded else 'unfolded'}")
                s.table_state_version += 1
            self._r(200, "application/json",
                    json.dumps({"ok": True, "player": valid, "folded": folded}))

        elif p == "/api/table/verify":
            action = data.get("action", "")
            ok = False
            err = None
            if action == "confirm":
                pv = s.pending_verify
                if pv and pv.get("guess"):
                    g = pv["guess"]
                    if g.get("rank") and g.get("suit"):
                        ok = _resolve_verify(s, {"rank": g["rank"], "suit": g["suit"]})
                    else:
                        err = "guess missing rank/suit"
                else:
                    err = "no active verify"
            elif action == "override":
                rank = str(data.get("rank", "")).upper()
                suit = str(data.get("suit", "")).lower()
                if rank in {"A","2","3","4","5","6","7","8","9","10","J","Q","K"} and \
                        suit in {"clubs","diamonds","hearts","spades"}:
                    ok = _resolve_verify(s, {"rank": rank, "suit": suit})
                else:
                    # Legacy "code" path (e.g. "Ac" / "10h")
                    parsed = _parse_card_code(data.get("code", ""))
                    if parsed:
                        ok = _resolve_verify(s, parsed)
                    else:
                        err = "invalid rank/suit"
            elif action == "rescan":
                pv = s.pending_verify
                if not pv:
                    err = "no active verify"
                else:
                    slot = pv.get("slot")
                    result = _pi_slot_scan(s, slot) if slot else None
                    if result is None:
                        err = "pi unreachable"
                    elif not result.get("present"):
                        new_guess = {"rank": "", "suit": "", "confidence": 0.0}
                        new_prompt = f"Slot {slot}: rescan — no card present."
                        with s.table_lock:
                            pv["guess"] = new_guess
                            pv["prompt"] = new_prompt
                            pv["rescan_id"] = int(pv.get("rescan_id", 0)) + 1
                            s.slot_pending.pop(slot, None)
                            s.table_state_version += 1
                        ok = True
                    else:
                        card = result.get("card") or {}
                        conf = float(card.get("confidence", 0.0))
                        if card.get("rank") and card.get("suit"):
                            new_guess = {
                                "rank": card["rank"],
                                "suit": card["suit"],
                                "confidence": round(conf, 2),
                            }
                            new_prompt = (
                                f"Slot {slot} rescan: {int(conf*100)}%. "
                                f"Confirm or correct."
                            )
                            with s.table_lock:
                                pv["guess"] = new_guess
                                pv["prompt"] = new_prompt
                                pv["rescan_id"] = int(pv.get("rescan_id", 0)) + 1
                                s.slot_pending[slot] = dict(new_guess)
                                s.table_state_version += 1
                            ok = True
                        else:
                            new_guess = {"rank": "", "suit": "", "confidence": 0.0}
                            new_prompt = f"Slot {slot}: rescan — card not recognized."
                            with s.table_lock:
                                pv["guess"] = new_guess
                                pv["prompt"] = new_prompt
                                pv["rescan_id"] = int(pv.get("rescan_id", 0)) + 1
                                s.slot_pending.pop(slot, None)
                                s.table_state_version += 1
                            ok = True
            else:
                err = "bad action"
            resp = {"ok": ok}
            if err:
                resp["error"] = err
            self._r(200, "application/json", json.dumps(resp))

        elif p == "/api/table/mark":
            # Rodney toggles a slot's "to replace" mark during betting,
            # or a "to go out with" mark during a Challenge vote.
            # Body: {"slot": N, "marked": true/false}
            try:
                slot_num = int(data.get("slot", 0))
            except (TypeError, ValueError):
                slot_num = 0
            marked = bool(data.get("marked", False))
            ge = s.game_engine
            in_challenge_vote = (
                s.console_state == "challenge_vote"
                and _game_is_challenge(ge)
                and not s.rodney_out_slots
            )
            if slot_num <= 0 or slot_num not in s.rodney_downs:
                self._r(400, "application/json",
                        '{"ok":false,"error":"invalid slot"}')
            elif (not in_challenge_vote) and s.rodney_drew_this_hand:
                self._r(400, "application/json",
                        '{"ok":false,"error":"draw already taken this hand"}')
            else:
                if in_challenge_vote:
                    max_marks = _challenge_required_cards(s)
                else:
                    max_marks = _max_draw_for_game(ge, s.rodney_draws_done, s)
                with s.table_lock:
                    if marked:
                        if len(s.rodney_marked_slots) >= max_marks and \
                                slot_num not in s.rodney_marked_slots:
                            self._r(400, "application/json",
                                    json.dumps({"ok": False,
                                                "error": f"max {max_marks} cards"}))
                            return
                        s.rodney_marked_slots.add(slot_num)
                    else:
                        s.rodney_marked_slots.discard(slot_num)
                    s.table_state_version += 1
                self._r(200, "application/json", json.dumps({
                    "ok": True,
                    "marked_slots": sorted(s.rodney_marked_slots),
                }))

        elif p == "/api/table/request_cards":
            # Rodney submits his draw choice. With 0 marks the hand skips
            # straight to betting round 2; with 1+ marks the slots clear,
            # LEDs light, and the guided replace flow starts.
            if s.rodney_drew_this_hand:
                self._r(400, "application/json",
                        '{"ok":false,"error":"draw already taken"}')
            else:
                slots_to_replace = sorted(s.rodney_marked_slots)
                # Remember the old card code per slot so the guided loop can
                # detect "card changed" even when the Pi's /scan polling is
                # too slow to catch the present=false moment between swap.
                previous_cards = {}
                with s.table_lock:
                    for slot in slots_to_replace:
                        s.rodney_downs.pop(slot, None)
                        code = s.pi_prev_slots.pop(slot, None)
                        if code:
                            previous_cards[slot] = code
                    s.rodney_marked_slots = set()
                    s.rodney_drew_this_hand = True
                    s.table_state_version += 1
                if slots_to_replace:
                    _table_log_add(s,
                        f"Rodney requested {len(slots_to_replace)} card(s): "
                        f"slots {slots_to_replace}"
                    )
                    log.log(f"[CONSOLE] Rodney draw: replacing slots "
                            f"{slots_to_replace}")
                    _start_guided_replace(s, slots_to_replace, previous_cards)
                else:
                    # Rodney stood pat — no cards replaced. Mark the draw
                    # as done and advance to the post-draw betting round.
                    _table_log_add(s, "Rodney stood pat (no cards replaced)")
                    with s.table_lock:
                        s.rodney_draws_done += 1
                        s.console_state = "betting"
                        s.console_betting_round = s.rodney_draws_done + 1
                        s.table_state_version += 1
                    log.log(
                        f"[CONSOLE] Rodney stood pat — draw "
                        f"{s.rodney_draws_done} done → betting round "
                        f"{s.console_betting_round}"
                    )
                self._r(200, "application/json", json.dumps({
                    "ok": True, "slots": slots_to_replace,
                }))

        elif p == "/api/table/pass":
            # Rodney's Pass button. No turn ordering in the
            # button-driven Challenge flow — dealer drives the round
            # and hits End Round when ready.
            ok, err = _set_challenge_vote(s, "Rodney", "pass")
            if ok:
                self._r(200, "application/json", '{"ok":true}')
            else:
                self._r(400, "application/json", json.dumps({
                    "ok": False, "error": err,
                }))

        elif p == "/api/table/go_out":
            # Rodney's Go Out button. _set_challenge_vote validates
            # mark count for rounds 1-2 and returns an error if it's
            # off; round 3 has no count requirement.
            ok, err = _set_challenge_vote(s, "Rodney", "out")
            if ok:
                self._r(200, "application/json", '{"ok":true}')
            else:
                self._r(400, "application/json", json.dumps({
                    "ok": False, "error": err,
                }))

        elif p == "/api/console/challenge_vote":
            # Dealer-driven per-player Pass / Out / Clear buttons.
            # Body: {"player": "Bill", "vote": "pass"|"out"|"clear"}
            player = str(data.get("player", "")).strip()
            vote = str(data.get("vote", "")).strip().lower()
            ok, err = _set_challenge_vote(s, player, vote)
            if ok:
                self._r(200, "application/json", '{"ok":true}')
            else:
                self._r(400, "application/json", json.dumps({
                    "ok": False, "error": err,
                }))

        elif p == "/api/console/challenge_end_round":
            # Dealer clicks End Round — resolve the current challenge
            # round. Players not explicitly marked pass or out are
            # treated as pass (per user's spec: "anybody not going
            # out is passing").
            if s.console_state != "challenge_vote":
                self._r(400, "application/json",
                        '{"ok":false,"error":"not in challenge vote"}')
            else:
                # Per-player pass counts already reflect what was
                # clicked. _resolve_challenge_round only looks at
                # went_out, so pass counts don't affect resolve math —
                # but we log any incomplete ones so it's clear they
                # didn't hit the 2-pass max before End Round fired.
                for nm, st in s.challenge_per_player.items():
                    if not st.get("went_out"):
                        passes = int(st.get("passes", 0))
                        if passes < MAX_PASSES_PER_ROUND:
                            log.log(
                                f"[CHALLENGE] End Round with {nm} at "
                                f"{passes}/{MAX_PASSES_PER_ROUND} passes — "
                                f"counted as pass"
                            )
                _resolve_challenge_round(s)
                self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/console/reshuffled":
            # Dealer clicks Reshuffled after round 3 ended with 0 out.
            if s.console_state != "reshuffle":
                self._r(400, "application/json",
                        '{"ok":false,"error":"not in reshuffle state"}')
            else:
                with s.table_lock:
                    s.rodney_downs = {}
                    s.rodney_marked_slots = set()
                    _clear_rodney_challenge_leds(s)
                    s.rodney_overflow = []
                    s.pi_prev_slots = {}
                    s.slot_pending = {}
                    s.slot_empty = {}
                    s.pending_verify = None
                    s.challenge_round_index = 0
                    s.challenge_shuffle_count += 1
                    for st in s.challenge_per_player.values():
                        st["passes"] = 0
                        st["went_out"] = False
                        st["out_round"] = None
                        st["out_slots"] = []
                    s.table_state_version += 1
                n_players = len(s.console_active_players)
                # Reshuffle rounds use the flat subsequent-round ante,
                # not the first-round dropdown value.
                per_player = _challenge_ante_cents_for(
                    s, s.challenge_shuffle_count, 0,
                )
                s.pot_cents += per_player * n_players
                _log_and_speak(s,
                    f"Reshuffle #{s.challenge_shuffle_count}. "
                    f"Round 1 ante: {_fmt_money(per_player)} each. "
                    f"Pot is now {_fmt_money(s.pot_cents)}.")
                s.console_state = "dealing"
                _start_guided_deal(s, 3)
                self._r(200, "application/json", '{"ok":true}')

        # --- Console (dealer phone UI) ---

        elif p == "/api/console/state":
            ge = s.game_engine
            # Include zone-recognized cards with details — only for players
            # checked in as active. Inactive players' zones are still
            # calibrated but not in play this hand.
            zone_cards = {}
            for z in s.cal.zones:
                name = z["name"]
                if name not in s.console_active_players:
                    continue
                card = s.monitor.last_card.get(name, "")
                details = s.monitor.recognition_details.get(name, {})
                zone_cards[name] = {
                    "card": card if card and card != "No card" else "",
                    "yolo": details.get("yolo", ""),
                    "yolo_conf": details.get("yolo_conf", 0),
                    "claude": details.get("claude", ""),
                    "duplicate": False,
                }
            # Flag duplicates against current round AND all prior rounds this hand
            seen = {}  # card -> "player round N" descriptor
            for c in s.console_hand_cards:
                seen[c["card"]] = f"{c['player']} (round {c['round']})"
            for name in s.console_active_players:
                zi = zone_cards.get(name)
                if not zi or not zi["card"]:
                    continue
                card = zi["card"]
                if card in seen:
                    zi["duplicate"] = True
                    # If the prior is in current zone_cards, flag that too
                    prior_name = seen[card].split(" ")[0]
                    if prior_name in zone_cards:
                        zone_cards[prior_name]["duplicate"] = True
                else:
                    seen[card] = name
            game_in_progress = (
                ge.current_game is not None
                and s.console_state != "idle"
            )
            has_up = s.console_total_up_rounds > 0 or (
                ge.current_game is not None and any(
                    ph.type.value == "hit_round" for ph in ge.current_game.phases
                )
            )
            # Map console_state -> phase label + action button spec for the UI.
            if not s.night_active:
                phase_label = "Night not started"
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            elif ge.current_game is None or s.console_state == "idle":
                phase_label = "Choose a game"
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            elif s.console_state == "dealing":
                phase_label = "Dealing"
                action_label = "Confirm Cards"
                action_endpoint = "/api/console/confirm"
                # Disabled for all-down rounds (auto-advances when guided
                # finishes) and whenever a guided session is in flight so
                # the dealer can't commit up cards mid-deal for the
                # leading-down guided flow in stud games.
                action_enabled = has_up and s.guided_deal is None
            elif s.console_state == "betting":
                rnd = s.console_betting_round or max(1, s.console_up_round)
                phase_label = f"Betting (round {rnd})"
                action_label = "Pot is right"
                action_endpoint = "/api/console/next_round"
                action_enabled = True
            elif s.console_state == "draw":
                phase_label = "Draw — Rodney picking"
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            elif s.console_state == "replacing":
                phase_label = "Replacing Rodney's cards"
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            elif s.console_state == "challenge_vote":
                outs = sum(1 for st in s.challenge_per_player.values()
                           if st.get("went_out"))
                round_lbl = (s.challenge_round_index or 0) + 1
                phase_label = (
                    f"Challenge vote — round {round_lbl} of 3 "
                    f"({outs} out so far)"
                )
                action_label = "End Round"
                action_endpoint = "/api/console/challenge_end_round"
                action_enabled = True
            elif s.console_state == "challenge_resolve":
                outs = [nm for nm, st in s.challenge_per_player.items()
                        if st.get("went_out")]
                phase_label = (
                    f"Challenge resolve — say \"{{name}} wins\" "
                    f"({', '.join(outs)})"
                )
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            elif s.console_state == "reshuffle":
                phase_label = "Collect + shuffle + cut, then Reshuffled"
                action_label = "Reshuffled"
                action_endpoint = "/api/console/reshuffled"
                action_enabled = True
            elif s.console_state == "hand_over":
                phase_label = "Hand over"
                action_label = "New Hand"
                action_endpoint = "/api/console/end"
                action_enabled = True
            else:
                phase_label = s.console_state
                action_label = ""
                action_endpoint = ""
                action_enabled = False
            self._r(200, "application/json", json.dumps({
                "active_players": s.console_active_players,
                "all_players": PLAYER_NAMES,
                "games": ge.get_game_list(),
                "game_groups": ge.get_game_groups(),
                "brio_settle_s": round(s.brio_settle_s, 2),
                "pi_presence_threshold": round(s.pi_presence_threshold, 1),
                "whisper_min_energy_threshold": round(
                    s.whisper_min_energy_threshold, 0
                ),
                "whisper_current_threshold": (
                    round(s.whisper_listener.current_energy_threshold, 0)
                    if (s.whisper_listener is not None
                        and s.whisper_listener.current_energy_threshold is not None)
                    else None
                ),
                "whisper_calibration_threshold": (
                    round(s.whisper_listener.calibration_threshold, 0)
                    if (s.whisper_listener is not None
                        and s.whisper_listener.calibration_threshold is not None)
                    else None
                ),
                "dealer": ge.get_dealer().name,
                "hand": ge.get_hand_state(),
                "last_round_cards": s.console_last_round_cards,
                "zone_cards": zone_cards,
                "yolo_min_conf": s.monitor.yolo_min_conf,
                "up_round": s.console_up_round,
                "total_up_rounds": s.console_total_up_rounds,
                "scan_phase": s.console_scan_phase,
                "night_active": s.night_active,
                "session_kind": getattr(s, "session_kind", "poker"),
                "console_state": s.console_state,
                "game_in_progress": game_in_progress,
                "phase_label": phase_label,
                "action_label": action_label,
                "action_endpoint": action_endpoint,
                "action_enabled": action_enabled,
                "current_game": ge.current_game.name if ge.current_game else "",
                "last_game_name": s.last_game_name,
                "ante_cents": s.ante_cents,
                "betting_limit": s.betting_limit,
                "betting_limit_label": _betting_limit_label(s.betting_limit),
                "forced_pot_limit_games": sorted(FORCED_POT_LIMIT_GAMES),
                "ante_options": [
                    {"value": 25, "label": "$0.25"},
                    {"value": 50, "label": "$0.50"},
                    {"value": 75, "label": "$0.75"},
                    {"value": 100, "label": "$1.00"},
                    {"value": 200, "label": "$2.00"},
                ],
                "limit_options": [
                    {"value": k, "label": v}
                    for k, v in BETTING_LIMIT_LABELS.items()
                ],
                "challenge": (
                    {
                        "round_index": s.challenge_round_index,
                        "shuffle_count": s.challenge_shuffle_count,
                        "pot_cents": s.pot_cents,
                        "required_cards": _challenge_required_cards(s),
                        "per_player": {
                            nm: {
                                "went_out": st["went_out"],
                                "passes": int(st.get("passes", 0)),
                                "out_round": st["out_round"],
                                "out_slots": list(st["out_slots"]),
                            } for nm, st in s.challenge_per_player.items()
                        },
                        "max_passes": MAX_PASSES_PER_ROUND,
                    }
                    if (_game_is_challenge(ge)
                        and s.challenge_round_index is not None)
                    else None
                ),
            }))

        elif p == "/api/console/players":
            names = data.get("players", [])
            valid = [n for n in names if n in PLAYER_NAMES]
            s.console_active_players = valid if valid else list(PLAYER_NAMES)
            log.log(f"[CONSOLE] Active players: {', '.join(s.console_active_players)}")
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/console/set_dealer":
            ge = s.game_engine
            name = data.get("dealer", "")
            for i, p2 in enumerate(ge.players):
                if p2.name.lower() == name.lower():
                    ge.dealer_index = i
                    ge._update_dealer()
                    log.log(f"[CONSOLE] Dealer set to {p2.name}")
                    break
            self._r(200, "application/json", json.dumps({"dealer": ge.get_dealer().name}))

        elif p == "/api/console/start_night":
            # Start a session: open a session folder, flip
            # night_active, accept any initial settings so the modal
            # submits once. The Console UI surfaces two buttons —
            # Start Poker / Start Testing — that hit this endpoint
            # with body.kind = "poker" or "testing".
            kind = (data.get("kind") or "poker").lower()
            if kind not in ("poker", "testing"):
                kind = "poker"
            archive = log.start_session(kind)
            s.night_active = True
            s.session_kind = kind
            # Fresh night → forget any tunnel-seen flag from a prior
            # session so the local view starts unredacted until Rodney
            # actually connects this night.
            s.rodney_tunnel_seen = False
            s.console_state = "idle"
            self._apply_settings(s, data)
            log.log("[CONSOLE] Poker night started")
            self._r(200, "application/json", json.dumps({
                "ok": True, "archive": archive,
            }))

        elif p == "/api/console/settings":
            # Mid-night adjustments: same payload as start_night, but doesn't
            # rotate the log or toggle night state.
            self._apply_settings(s, data)
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/console/exit_poker":
            log.log("[CONSOLE] Exit Poker — closing night")
            _stop_guided_deal(s)
            log.end_night()
            s.night_active = False
            s.console_state = "idle"
            self._r(200, "application/json", '{"ok":true,"exiting":true}')
            # Schedule a brief deferred exit so the response can flush first.
            def _bye():
                time.sleep(0.3)
                os._exit(0)
            Thread(target=_bye, daemon=True).start()

        elif p == "/api/console/rescan_all":
            # Wipe the auto-recognized state for every active zone in
            # the current round and rescan from the latest frame.
            # User-corrected zones are preserved — once the dealer has
            # typed a value into a zone, only an explicit re-correction
            # changes it. (If you really want to clear a correction
            # too, tap that zone and pick "—" or the right card.)
            frame = s.latest_frame
            if frame is None:
                return self._r(503, "application/json",
                               '{"ok":false,"error":"no frame yet"}')
            impl = s.current_game_impl
            if impl is not None:
                scan_names, _ = impl.zones_to_scan(s)
            else:
                scan_names = list(s.console_active_players)
            watched = set(scan_names)
            wiped = 0
            preserved = 0
            zone_crops = {}
            for z in s.cal.zones:
                name = z["name"]
                if name not in watched:
                    continue
                if s.monitor.zone_state.get(name) == "corrected":
                    preserved += 1
                    continue
                # Wipe so /api/console/state stops showing the old
                # card while the new scan is in flight.
                s.monitor.last_card[name] = ""
                s.monitor.last_announced[name] = ""
                s.monitor.zone_state[name] = "empty"
                s.monitor.recognition_details[name] = {}
                s.monitor.recognition_crops[name] = None
                wiped += 1
                crop = s.monitor._crop(frame, z)
                if crop is None or crop.size == 0:
                    continue
                zone_crops[name] = crop.copy()
                s.monitor.pending[name] = True
            log.log(
                f"[CONSOLE] Rescan all — wiped {wiped} zone(s), "
                f"scanning {len(zone_crops)}, preserved {preserved} "
                f"corrected"
            )
            with s.table_lock:
                s.table_state_version += 1
            if not zone_crops:
                return self._r(200, "application/json",
                               '{"ok":true,"scanned":0}')
            s.console_scan_phase = "scanned"
            # Manual rescan-all is an explicit dealer override —
            # don't hold the announcements behind the dealer-zone
            # speech gate.
            s.monitor.open_speech_gate()
            s._dealer_zone_done = True
            Thread(target=s.monitor._recognize_batch,
                   args=(zone_crops,), daemon=True).start()
            self._r(200, "application/json",
                    json.dumps({"ok": True, "scanned": len(zone_crops)}))

        elif p == "/api/console/force_scan":
            # Dealer clicked "Waiting for cards..." — skip motion detection
            # and scan every zone the game says is in play right now.
            frame = s.latest_frame
            if frame is None:
                return self._r(503, "application/json",
                               '{"ok":false,"error":"no frame yet"}')
            impl = s.current_game_impl
            if impl is not None:
                scan_names, _ = impl.zones_to_scan(s)
            else:
                scan_names = list(s.console_active_players)
            watched = set(scan_names)
            zone_crops = {}
            for z in s.cal.zones:
                name = z["name"]
                if name not in watched:
                    continue
                if s.monitor.zone_state.get(name) == "corrected":
                    continue
                crop = s.monitor._crop(frame, z)
                if crop is None or crop.size == 0:
                    continue
                zone_crops[name] = crop.copy()
                s.monitor.pending[name] = True
            if not zone_crops:
                return self._r(200, "application/json",
                               '{"ok":true,"scanned":0}')
            s.console_scan_phase = "scanned"
            # Manual scan is an explicit dealer override — open the
            # speech gate so cards announce immediately.
            s.monitor.open_speech_gate()
            s._dealer_zone_done = True
            Thread(target=s.monitor._recognize_batch,
                   args=(zone_crops,), daemon=True).start()
            log.log(f"[CONSOLE] Force scan of {len(zone_crops)} zones")
            self._r(200, "application/json",
                    json.dumps({"ok": True, "scanned": len(zone_crops)}))

        elif p == "/api/console/deal":
            ge = s.game_engine
            game_name = data.get("game", "")
            if game_name not in ge.templates:
                self._r(400, "application/json", json.dumps({"error": f"Unknown game: {game_name}"}))
            else:
                # Accept optional ante + betting-limit from the Start
                # Game form. Omitting them (e.g. voice "Same game
                # again") keeps the previous hand's values.
                raw_ante = data.get("ante_cents")
                if raw_ante is not None:
                    try:
                        new_ante = int(raw_ante)
                        if new_ante > 0:
                            s.ante_cents = new_ante
                    except (TypeError, ValueError):
                        pass
                raw_limit = data.get("betting_limit")
                if raw_limit is not None:
                    limit = str(raw_limit).strip()
                    if limit in BETTING_LIMIT_LABELS:
                        s.betting_limit = limit
                # Force pot limit for games that require it.
                forced = _forced_betting_limit(game_name)
                if forced:
                    s.betting_limit = forced
                result = ge.new_hand(game_name)
                s.last_game_name = game_name
                # Stand up the per-game class instance. Empty class_name
                # falls back to BaseGame, so templates that have no
                # dedicated class behave exactly as before.
                s.current_game_impl = make_game(ge.current_game, ge)
                s.console_last_round_cards = []
                s.console_hand_cards = []
                # Count total up-card rounds from template. Games with an
                # open-ended HIT_ROUND (7/27) aren't a fixed count — mark
                # them as 0 (unbounded) so the UI stays in "confirmed /
                # Next Round" flow instead of switching to idle.
                s.console_up_round = 0
                template = ge.templates[game_name]
                has_hit_round = any(
                    phase.type.value == "hit_round" and phase.card_type == "up"
                    for phase in template.phases
                )
                if has_hit_round:
                    up_rounds = 0  # 0 = unbounded
                else:
                    up_rounds = 0
                    for phase in template.phases:
                        if phase.type.value == "deal" and "up" in phase.pattern:
                            up_rounds += 1
                        elif phase.type.value == "community":
                            up_rounds += 1
                s.console_total_up_rounds = up_rounds
                # Open a per-game log file before the New hand line
                # is written so the log line itself lands in the
                # right file.
                log.start_game(game_name)
                log.log(f"[CONSOLE] New hand: {game_name}, dealer: {result['dealer']}")
                if result.get("wild_label"):
                    log.log(f"[CONSOLE] {result['wild_label']}")
                # Quick Pi reachability check. If the Pi's down, set the
                # pi_offline flag so every Pi call (flash/hold, /slots,
                # slot LEDs, etc.) short-circuits for the rest of the game.
                # Flag clears on the next Deal (re-checked below).
                pi_up = _pi_ping(s)
                s.pi_offline = not pi_up
                s.pi_flash_held = False  # reset our tracker regardless
                log.log(f"[PI] Deal-time ping: {'reachable' if pi_up else 'OFFLINE (suppressing calls)'}")
                # Brio watching: triggered for any game that will produce up
                # cards, either via explicit up/community phases or an
                # open-ended HIT_ROUND (7/27). Baselines are captured now
                # while the table is empty of up cards, regardless of when
                # watching actually starts. Whether to watch immediately or
                # defer until guided finishes depends on whether the FIRST
                # deal phase itself contains an up card.
                leading_downs = _initial_down_count(ge)
                will_guide = leading_downs > 0 and not s.pi_offline
                needs_brio = up_rounds != 0 or has_hit_round
                if needs_brio and s.cal.ok and s.latest_frame is not None:
                    s.monitor.capture_baselines(s.latest_frame)
                    s.monitoring = True
                    if will_guide:
                        # Any game that uses guided Pi-slot dealing deals
                        # downs first — regardless of whether the first
                        # phase also contains an up card — so the dealer
                        # sweeps every zone multiple times before the up
                        # cards arrive. Defer Brio until guided completes;
                        # the per-game guided-completion branches (all-down
                        # vs mixed vs hit-round) take over from there.
                        s.console_scan_phase = "idle"
                        log.log(
                            "[CONSOLE] Baselines captured; Brio watching "
                            "deferred until guided downs are validated"
                        )
                    else:
                        s.console_scan_phase = "watching"
                        s._zones_with_motion = set()
                        log.log(
                            f"[CONSOLE] Watching {ge.get_dealer().name}'s "
                            f"zone for first card"
                        )
                # Start the hand fresh: clear Rodney-side state and turn the
                # scanner LEDs on so the initial down cards get good scans.
                with s.table_lock:
                    s.rodney_downs = {}
                    s.rodney_flipped_up = None
                    s.slot_pending = {}
                    s.slot_empty = {}
                    s.verify_queue = []
                    s.pending_verify = None
                    s.pi_prev_slots = {}
                    s.folded_players = set()
                    s.freezes = {name: 0 for name in s.console_active_players}
                    s.rodney_marked_slots = set()
                    s.rodney_drew_this_hand = False
                    s.rodney_draws_done = 0
                    s.console_betting_round = 0
                    s.console_trailing_done = False
                    s.stats = {"yolo_right": 0, "yolo_wrong": 0,
                               "claude_right": 0, "claude_wrong": 0,
                               "pi_auto": 0,
                               "pi_verify_right": 0, "pi_verify_wrong": 0}
                    s._zones_with_motion = set()
                    s._missing_speech_count = {}
                    s._empty_scan_count = {}
                    s._dealer_zone_done = False
                    s._missing_prompt_fired = False
                    s._zone_prev_pending = {}
                    s.table_state_version += 1
                # Challenge-game per-hand reset. pot_cents is NOT reset —
                # it accumulates across hands until a 1-out-all-pass award.
                challenge_hand = _game_is_challenge(ge)
                if challenge_hand:
                    s.challenge_round_index = 0
                    s.challenge_shuffle_count = 0
                    s.challenge_per_player = {
                        nm: {"went_out": False, "passes": 0,
                             "out_round": None, "out_slots": []}
                        for nm in s.console_active_players
                    }
                    # Turn off any leftover challenge LEDs from a
                    # prior hand before the new guided deal lights
                    # its own slots.
                    _clear_rodney_challenge_leds(s)
                    s.rodney_overflow = []
                    n_players = len(s.console_active_players)
                    s.pot_cents += s.ante_cents * n_players
                else:
                    s.challenge_round_index = None
                # Single Start-Game announcement — dealer + game + ante
                # + betting limit + current pot. The pot line runs for
                # every game so the dealer can cross-check the physical
                # chip stack against what we've tracked before the deal
                # starts. "No pot carryover" means this should always
                # read $0.00 at a fresh hand start; non-zero pot means
                # the previous hand didn't resolve cleanly.
                dealer_name = ge.get_dealer().name
                _log_and_speak(s,
                    f"Dealer is {dealer_name}. Game is {game_name}. "
                    f"{_speak_ante(s.ante_cents)} ante. "
                    f"{_betting_limit_spoken(s.betting_limit)}. "
                    f"Pot is {_fmt_money(s.pot_cents)}.")
                # Let the per-game class wire up its own per-hand state
                # (freeze counters, local flags) now that common state is
                # reset and the engine knows the current game.
                s.current_game_impl.on_hand_start(s)
                # Make sure any stale guided session from a prior hand is gone.
                _stop_guided_deal(s)
                s.console_state = "dealing"
                # Use the slot-by-slot guided flow for the leading down cards
                # in the first deal phase — 2 for 7 Card Stud, 2 for Hold'em,
                # 3 for Follow the Queen, 5 for 5 Card Draw. Games with up
                # cards in the same phase start Brio watching only after
                # guided completes, so dealer hand motion over local zones
                # during down-card dealing doesn't trip false alarms.
                if will_guide:
                    _start_guided_deal(s, leading_downs)
                else:
                    _update_flash_for_deal_state(s)
                self._r(200, "application/json", json.dumps(result))

        elif p == "/api/console/confirm":
            ge = s.game_engine
            # Idempotency guard: after confirm processes a round, scan
            # phase transitions to "confirmed" (mid-hand) or "idle"
            # (last up round). A second click before the next scan
            # arrives would otherwise re-advance console_up_round,
            # re-announce bet-first, and re-tick 7/27 freezes — a
            # single spurious double-tap could flip a player from 1
            # freeze to 3 (frozen) in one round. Treat repeat confirms
            # as a no-op until new scan data comes in.
            if s.console_scan_phase in ("confirmed", "idle"):
                self._r(200, "application/json",
                        '{"ok":true,"duplicate":true}')
                return
            # Collect round cards in deal order (clockwise from dealer's left)
            dealer_idx = ge.dealer_index
            round_cards = []
            for i in range(1, len(ge.players) + 1):
                p2 = ge.players[(dealer_idx + i) % len(ge.players)]
                if p2.name not in s.console_active_players:
                    continue
                card = s.monitor.last_card.get(p2.name, "")
                if card and card != "No card":
                    round_cards.append({"player": p2.name, "card": card})
            # De-duplicate against previously-seen cards (prior up rounds,
            # Rodney's down cards) before the round is announced and
            # accumulated — a duplicate is almost always a misread.
            _dedup_round_cards_against_seen(s, round_cards)
            round_num = s.console_up_round + 1
            # Announce every round's wild updates + bet-first. Earlier
            # code deferred the final-up-round announce past the 7th-
            # street deal, but that silenced the 4th-street betting
            # round's bet-first call entirely — the dealer had no
            # audio cue that bet 4 had started. _announce_trailing_done
            # still fires after the 7th card lands so the final betting
            # round gets its own cue (usually the same high hand).
            _check_follow_the_queen_round(s, round_cards)
            # Accumulate into hand-wide history, then clear the current-round
            # data so it stops re-appearing as "just dealt" cards (which were
            # triggering the duplicate detector against the exact same cards
            # now in the hand history).
            if round_cards:
                for c in round_cards:
                    s.console_hand_cards.append({"player": c["player"], "card": c["card"], "round": round_num})
            # Per-round hook: game class updates state (freezes, wild
            # tracking) and speaks its bet-first announcement via the
            # base class's score_hand flow. Stud/draw games additionally
            # route through _announce_poker_hand_bet_first below until
            # the poker-hand announcer migrates into the BaseGame path.
            impl = s.current_game_impl
            if impl is not None:
                impl.on_round_confirmed(s, round_cards)
            _announce_poker_hand_bet_first(s)
            s.console_last_round_cards = []
            for z in s.cal.zones:
                zname = z["name"]
                s.monitor.zone_state[zname] = "empty"
                s.monitor.last_card[zname] = ""
                s.monitor.last_announced[zname] = ""
                s.monitor.recognition_details[zname] = {}
                s.monitor.recognition_crops[zname] = None
            # If this was the last up-card round, go to idle. 0 means
            # unbounded (games with HIT_ROUND), never idle on count.
            if (s.console_total_up_rounds > 0
                    and round_num >= s.console_total_up_rounds):
                s.console_scan_phase = "idle"
                # Message the dealer on what comes next — trailing-
                # down games (7 Card Stud / FTQ) still owe a 7th
                # street + one more betting round before hand_over,
                # so "click End Hand" would skip those steps.
                has_trailing = bool(_trailing_down_slots(ge))
                if has_trailing:
                    log.log(
                        f"[CONSOLE] Final up round ({round_num}) confirmed — "
                        f"click Pot is right to start 7th-street trailing deal"
                    )
                else:
                    log.log(
                        f"[CONSOLE] Final up round ({round_num}) confirmed — "
                        f"click Pot is right to advance to End Hand"
                    )
            else:
                s.console_scan_phase = "confirmed"
                log.log(f"[CONSOLE] Cards confirmed for up round {round_num}")
            # Reset the per-round per-zone flags so the next round
            # starts with a closed speech gate, no missing-zone
            # prompts spent yet, and no empty-scan tally carried over.
            s._missing_speech_count = {}
            s._empty_scan_count = {}
            s._dealer_zone_done = False
            s._missing_prompt_fired = False
            s._zone_prev_pending = {}
            # Once the up-card round is confirmed, check Rodney's down slots
            # for anything below the auto-accept threshold. Those slots get
            # queued and (on LED-equipped hardware) will start blinking; the
            # /table modal will appear when the user removes each card.
            queued = _enqueue_down_card_verifies(s)
            if queued:
                log.log(f"[CONSOLE] Down-card verify queued for slots {queued}")
            # If Rodney flipped a card (7/27 2-down), stop its slots blink-
            # hint now that the round is confirmed and the physical card is
            # on the table.
            if s.rodney_flipped_up and not s.pi_offline:
                _pi_slot_led(s, int(s.rodney_flipped_up["slot"]), "off")
            _update_flash_for_deal_state(s)
            # /table polls on state version; the hand-wide up-card history
            # just grew so bump the version or clients 304 and never see
            # the new up cards.
            with s.table_lock:
                s.table_state_version += 1
            s.console_state = "betting"
            self._r(200, "application/json", json.dumps({"ok": True}))

        elif p == "/api/console/next_round":
            ge = s.game_engine
            # All-down games (5 Card Draw, 3 Toed Pete) track betting rounds
            # independently from console_up_round because they have no up
            # rounds. 5CD flow:
            #   betting round 1 + Pot-is-right → draw state (Rodney's Request
            #     Cards button appears on /table; marking still enabled)
            #   replacement completes → betting round 2
            #   betting round 2 + Pot-is-right → hand_over
            if s.console_total_up_rounds == 0 and s.console_betting_round > 0:
                has_draw = _game_has_draw_phase(ge)
                total_draws = _total_draw_phases(ge)
                # If there are more DRAW phases left (multi-draw games like
                # 3 Toed Pete, or the single draw in 5 Card Draw), loop back
                # into a draw state. Reset Rodneys drew flag + marks so the
                # /table can collect fresh picks for this draw.
                if has_draw and s.rodney_draws_done < total_draws:
                    with s.table_lock:
                        s.console_state = "draw"
                        # On the FIRST draw transition, preserve any marks
                        # Rodney set during dealing/betting. Only clear
                        # stale marks between draws in multi-draw games
                        # (3 Toed Pete) where rodney_drew_this_hand is
                        # True from the previous draw.
                        if s.rodney_drew_this_hand:
                            s.rodney_marked_slots = set()
                        s.rodney_drew_this_hand = False
                        s.table_state_version += 1
                    log.log(
                        f"[CONSOLE] Betting round {s.console_betting_round} "
                        f"done → draw {s.rodney_draws_done + 1}/{total_draws}"
                    )
                    return self._r(200, "application/json", json.dumps({
                        "ok": True, "state": "draw",
                    }))
                # No draws remain (either a non-draw all-down game or the
                # last post-draw betting of a multi-draw game).
                s.console_state = "hand_over"
                log.log("[CONSOLE] All-down betting complete → hand_over")
                with s.table_lock:
                    s.table_state_version += 1
                return self._r(200, "application/json", json.dumps({
                    "ok": True, "state": "hand_over",
                }))
            # If trailing guided has already run this hand, this Pot-is-right
            # is the final betting round's — go to hand_over without touching
            # up_round / baselines.
            if s.console_trailing_done:
                s.console_scan_phase = "idle"
                s.console_state = "hand_over"
                log.log("[CONSOLE] Final betting done → hand_over")
                with s.table_lock:
                    s.table_state_version += 1
                return self._r(200, "application/json", json.dumps({
                    "ok": True, "state": "hand_over",
                }))
            # Advance round counter
            s.console_up_round += 1
            beyond_last_up = (
                s.console_total_up_rounds > 0
                and s.console_up_round >= s.console_total_up_rounds
            )
            # If we've finished all up rounds AND the game has trailing down
            # cards (7CS 7th, FTQ 7th), start a guided session for those
            # slots rather than jumping to hand_over.
            trailing = _trailing_down_slots(ge) if beyond_last_up else []
            if beyond_last_up and trailing and not s.pi_offline:
                s.console_scan_phase = "idle"
                log.log(
                    f"[CONSOLE] All up rounds done — starting trailing "
                    f"down deal for slots {trailing}"
                )
                _start_guided_trailing_deal(s, trailing)
                # State was set to "dealing" inside the starter; bump the
                # version so /table and /api/console/state re-fetch.
                with s.table_lock:
                    s.table_state_version += 1
                return self._r(200, "application/json", json.dumps({
                    "hand": ge.get_hand_state(),
                    "up_round": s.console_up_round,
                    "total_up_rounds": s.console_total_up_rounds,
                }))
            # Recapture baselines and resume watching dealer — but only if
            # there's still an up round ahead. If we've finished all up
            # rounds with no trailing downs, the hand is over.
            if s.cal.ok and s.latest_frame is not None:
                s.monitor.capture_baselines(s.latest_frame)
                for z in s.cal.zones:
                    s.monitor.zone_state[z["name"]] = "empty"
                    s.monitor.last_card[z["name"]] = ""
                    s.monitor.recognition_details[z["name"]] = {}
                    s.monitor.recognition_crops[z["name"]] = None
                s.console_scan_phase = "idle" if beyond_last_up else "watching"
                s._zones_with_motion = set()
                s._empty_scan_count = {}
                s._missing_speech_count = {}
                s._dealer_zone_done = False
                s._missing_prompt_fired = False
                s._zone_prev_pending = {}
                if beyond_last_up:
                    log.log("[CONSOLE] No more up rounds — idle until End Hand")
                    s.console_state = "hand_over"
                else:
                    log.log(f"[CONSOLE] Baselines recaptured, watching {ge.get_dealer().name}'s zone")
                    s.console_state = "dealing"
            log.log(f"[CONSOLE] Next Round — up round {s.console_up_round}/{s.console_total_up_rounds}")
            # Also queue any pending down-card scans so games with a final
            # down (no Confirm Cards) still get a chance to verify.
            queued = _enqueue_down_card_verifies(s)
            if queued:
                log.log(f"[CONSOLE] Down-card verify queued for slots {queued}")
            # Advancing rounds may change what the next expected card is —
            # e.g. after the 4th up round the 7th-card down becomes next, so
            # the LEDs need to come back on.
            _update_flash_for_deal_state(s)
            with s.table_lock:
                s.table_state_version += 1
            self._r(200, "application/json", json.dumps({
                "hand": ge.get_hand_state(),
                "up_round": s.console_up_round,
                "total_up_rounds": s.console_total_up_rounds,
            }))

        elif p == "/api/console/end":
            _stop_guided_deal(s)
            ge = s.game_engine
            # Recognition stats for the hand just ending.
            yr = s.stats.get("yolo_right", 0)
            yw = s.stats.get("yolo_wrong", 0)
            cr = s.stats.get("claude_right", 0)
            cw = s.stats.get("claude_wrong", 0)
            yolo_total = yr + yw
            claude_total = cr + cw
            total = yolo_total + claude_total

            def _pct(n, d):
                return f"{(100.0 * n / d):.0f}%" if d else "—"

            if total > 0:
                log.log(
                    f"[STATS] Hand recognition: {total} cards total, "
                    f"YOLO {yr}/{yolo_total} right ({_pct(yr, yolo_total)}), "
                    f"Claude {cr}/{claude_total} right ({_pct(cr, claude_total)})"
                )
            # Pi guided-deal (down cards) stats — helps us decide whether
            # the GUIDED_GOOD_CONF auto-accept bar of 0.50 is too high
            # (lots of verify modals, most resolved unchanged) or too low
            # (auto-accepts we would have corrected).
            pa = s.stats.get("pi_auto", 0)
            pvr = s.stats.get("pi_verify_right", 0)
            pvw = s.stats.get("pi_verify_wrong", 0)
            pv_total = pvr + pvw
            if pa or pv_total:
                log.log(
                    f"[STATS] Pi guided: {pa} auto-accepted, "
                    f"{pv_total} verified "
                    f"({pvr} right / {pvw} wrong, {_pct(pvr, pv_total)})"
                )
            if s.current_game_impl is not None:
                s.current_game_impl.on_hand_end(s)
            # Body flag: advance_dealer=false keeps the deal with the
            # current player. Used by the dropdown-cancel flow so the
            # dealer who got a mis-heard game doesn't lose their turn.
            advance = bool(data.get("advance_dealer", True))
            result = ge.end_hand(advance_dealer=advance)
            s.current_game_impl = None
            if advance:
                _skip_inactive_dealer(s)
            result["next_dealer"] = ge.get_dealer().name
            s.console_last_round_cards = []
            s.console_hand_cards = []
            s.console_up_round = 0
            s.console_total_up_rounds = 0
            s.console_trailing_done = False
            s.monitoring = False
            s.console_scan_phase = "idle"
            s.console_state = "idle"
            # Reset all zone states
            for z in s.cal.zones:
                s.monitor.zone_state[z["name"]] = "empty"
                s.monitor.last_card[z["name"]] = ""
                s.monitor.recognition_details[z["name"]] = {}
                s.monitor.recognition_crops[z["name"]] = None
            log.log(f"[CONSOLE] Hand over — next dealer: {result['next_dealer']}")
            # Close the per-game log file. Subsequent log lines flow
            # back into log.txt until the next /api/console/deal.
            log.end_game()
            # Clear Rodney-side hand state and turn scanner LEDs off now
            # that no cards are expected.
            with s.table_lock:
                s.rodney_downs = {}
                s.rodney_flipped_up = None
                s.slot_pending = {}
                s.slot_empty = {}
                s.verify_queue = []
                s.pending_verify = None
                s.pi_prev_slots = {}
                s.folded_players = set()
                s.freezes = {}
                # Clear any round-3 Challenge overflow so /table doesn't
                # leave the 2 displaced R2 cards visible after hand end.
                s.rodney_overflow = []
                # On cancel (advance=False) we also reset Challenge
                # state + roll back any antes added for this aborted
                # hand, so the dealer can redeal cleanly. Normal end
                # (resolution path) leaves pot_cents alone — resolve
                # logic handles that.
                if not advance and s.challenge_round_index is not None:
                    # Total per-player rollback sums every ante added
                    # so far across this hand (including reshuffles).
                    # Round 1 of shuffle 0 is s.ante_cents; every
                    # other round is CHALLENGE_SUBSEQUENT_ANTE_CENTS.
                    per_player_total = 0
                    shuffles = s.challenge_shuffle_count or 0
                    current_round = s.challenge_round_index or 0
                    for sh in range(shuffles + 1):
                        max_r = 2 if sh < shuffles else current_round
                        for r in range(max_r + 1):
                            per_player_total += _challenge_ante_cents_for(
                                s, sh, r,
                            )
                    rollback = per_player_total * len(s.console_active_players)
                    s.pot_cents = max(0, s.pot_cents - rollback)
                    s.challenge_round_index = None
                    s.challenge_per_player = {}
                    _clear_rodney_challenge_leds(s)
                    s.rodney_overflow = []
                    s.rodney_marked_slots = set()
                    log.log(
                        f"[CHALLENGE] Hand cancelled — rolled back "
                        f"{_fmt_money(rollback)} of antes. "
                        f"Pot now {_fmt_money(s.pot_cents)}."
                    )
                s.table_state_version += 1
            _update_flash_for_deal_state(s)
            self._r(200, "application/json", json.dumps(result))

        elif p == "/api/console/advance_dealer":
            ge = s.game_engine
            ge.advance_dealer()
            _skip_inactive_dealer(s)
            log.log(f"[CONSOLE] Dealer advanced to {ge.get_dealer().name}")
            self._r(200, "application/json", json.dumps({"dealer": ge.get_dealer().name}))

        elif p == "/api/console/correct":
            # Batch corrections: [{player, rank, suit}, ...]
            corrections = data.get("corrections", [])
            changed_any = False
            for c in corrections:
                player = c.get("player", "")
                rank = c.get("rank", "")
                suit = c.get("suit", "")
                if player and rank and suit:
                    RANK_TO_NAME = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}
                    SUIT_TO_NAME = {"spades": "Spades", "hearts": "Hearts",
                                    "diamonds": "Diamonds", "clubs": "Clubs"}
                    rank_name = RANK_TO_NAME.get(rank, rank)
                    suit_name = SUIT_TO_NAME.get(suit, suit)
                    new_card = f"{rank_name} of {suit_name}"
                    old_card = s.monitor.last_card.get(player, "")
                    # Tally this as a miss for whichever recognizer produced
                    # the card the user just overrode.
                    prior_details = s.monitor.recognition_details.get(player, {}) or {}
                    prior_source = prior_details.get("source")
                    if old_card != new_card and prior_source in ("yolo", "claude"):
                        _stats_bump(s, f"{prior_source}_right", -1)
                        _stats_bump(s, f"{prior_source}_wrong", +1)
                    s.monitor.last_card[player] = new_card
                    s.monitor.zone_state[player] = "corrected"
                    s.monitor.recognition_details[player] = {
                        "yolo": s.monitor.recognition_details.get(player, {}).get("yolo", ""),
                        "yolo_conf": s.monitor.recognition_details.get(player, {}).get("yolo_conf", 0),
                        "claude": s.monitor.recognition_details.get(player, {}).get("claude", ""),
                        "final": new_card,
                        "corrected": True,
                    }
                    # If the corrected card already landed in console_hand_cards
                    # (post-Confirm correction), update that entry in place so
                    # dedup / wild recompute / hand value / best hand all see
                    # the new value. Match on BOTH player AND the old card
                    # value — otherwise a mid-round correction (the common
                    # case) would overwrite a previous rounds entry because
                    # the current round hasnt been appended yet.
                    for entry in reversed(s.console_hand_cards):
                        if (entry.get("player") == player
                                and entry.get("card") == old_card):
                            entry["card"] = new_card
                            break
                    # Save corrected crop to training_data for future YOLO
                    # training. Delete the prior (wrong-label) save for this
                    # zone so the bad label doesnt poison the dataset.
                    crop = s.monitor.recognition_crops.get(player)
                    if crop is not None:
                        removed = s.monitor._delete_last_save(player)
                        if removed:
                            log.log(f"[CONSOLE] Removed wrong-label training save for {player}")
                        s.monitor._save(player, crop, new_card)
                        log.log(f"[CONSOLE] Saved correction to training_data: {new_card}")
                    log.log(f"[CONSOLE] Corrected {player}: {old_card} -> {new_card}")
                    if old_card != new_card:
                        changed_any = True
            # Re-derive Follow-the-Queen wild ranks from the corrected
            # history. If the corrected card was the follower of a queen,
            # this announces the new wild rank.
            if changed_any:
                _recompute_follow_the_queen(s)
                # Bump the /table version so the remote /table poll picks
                # up the new card immediately. Without this, voice-driven
                # card calls that arrive before any Brio scan this round
                # would leave Rodney's UI showing the stale pre-correction
                # state until the next natural version bump.
                with s.table_lock:
                    s.table_state_version += 1
            self._r(200, "application/json", '{"ok":true}')

        elif p == "/api/console/yolo_conf":
            val = data.get("value")
            if val is not None:
                s.monitor.yolo_min_conf = max(0.0, min(1.0, float(val)))
                log.log(f"[CONSOLE] YOLO min confidence: {s.monitor.yolo_min_conf:.0%}")
            self._r(200, "application/json", json.dumps({"yolo_min_conf": s.monitor.yolo_min_conf}))

        else:
            self._r(404,"text/plain","Not found")

    def _r(self, code, ct, body):
        try:
            self.send_response(code)
            # Default text/HTML/JSON responses to UTF-8 so browsers dont
            # mis-decode em-dashes and other multibyte chars as Windows-1252
            # (which is what the iPhone Safari view was showing as â€").
            if (ct.startswith("text/") or ct == "application/json") \
                    and "charset" not in ct.lower():
                ct = ct + "; charset=utf-8"
            self.send_header("Content-Type", ct)
            if ct == "image/jpeg":
                self.send_header("Cache-Control","no-store,no-cache,max-age=0")
            self.end_headers()
            self.wfile.write(body.encode() if isinstance(body,str) else body)
        except (ConnectionResetError, BrokenPipeError):
            pass

    def _jpeg(self, frame):
        if frame is None:
            log.log("[SNAPSHOT] no frame yet")
            return self._r(503, "text/plain", "No frame")
        try:
            h, w = frame.shape[:2]
        except Exception as e:
            log.log(f"[SNAPSHOT] bad frame: {e}")
            return self._r(500, "text/plain", f"bad frame: {e}")
        j = to_jpeg(frame, 80)
        if not j:
            log.log(f"[SNAPSHOT] JPEG encode failed for {w}x{h} frame")
            return self._r(500, "text/plain", "JPEG encode failed")
        log.log(f"[SNAPSHOT] served {w}x{h} JPEG ({len(j)} bytes)")
        return self._r(200, "image/jpeg", j)

    def _api_state(self, s):
        tm = s.test_mode
        test_info = None
        if tm:
            idx = tm["zone_idx"]
            test_info = {
                "zone": s.cal.zones[idx]["name"] if idx < len(s.cal.zones) else None,
                "zone_idx": idx,
                "total": len(s.cal.zones),
                "waiting": tm["waiting"],
                "result": tm["result"],
            }
        self._r(200,"application/json",json.dumps({
            "monitoring": s.monitoring,
            "calibrated": s.cal.ok,
            "resolution": s.capture.resolution,
            "test_mode": test_info,
            "deal_mode": _deal_mode_json(s),
            "collect_mode": _collect_mode_json(s),
            "zones": {z["name"]: {"state": s.monitor.zone_state.get(z["name"],"empty"),
                                   "card": s.monitor.last_card.get(z["name"],"")}
                      for z in s.cal.zones},
        }))

    def _page(self, s):
        players_js = json.dumps(PLAYER_NAMES)
        self._r(200, "text/html", SCANNER_TMPL.safe_substitute(players_js=players_js))


    def _calibrate_page(self, s):
        players_js = json.dumps(PLAYER_NAMES)
        current_focus = getattr(s.capture, "focus", None)
        focus_init_js = json.dumps(current_focus)
        self._r(200, "text/html", CALIBRATE_TMPL.safe_substitute(
            players_js=players_js, focus_init_js=focus_init_js,
        ))


    def _serve_logview(self, s):
        self._r(200, "text/html", LOGVIEW_HTML)


    def _zone_img(self, s, name):
        if not s.latest_frame is not None:
            return self._r(503,"text/plain","No frame")
        z = next((z for z in s.cal.zones if z["name"]==name), None)
        if not z: return self._r(404,"text/plain","Not found")
        crop = s.monitor._crop(s.latest_frame, z)
        if crop is None: return self._r(500,"text/plain","Crop failed")
        j = to_jpeg(crop, 90)
        if j: self._r(200,"image/jpeg",j)

    def _training_list(self, s):
        if not TRAINING_DIR.exists():
            return self._r(200,"text/html","<p>No data</p>")
        files = sorted(TRAINING_DIR.iterdir(), reverse=True)
        h = "<html><body style='font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px'><h1>Training Data</h1>"
        for f in files[:100]:
            if f.suffix == ".jpg":
                lbl = f.with_suffix(".txt").read_text() if f.with_suffix(".txt").exists() else ""
                h += f'<div style="display:inline-block;margin:8px;text-align:center"><img src="/training/{f.name}" width="150"><br><small>{f.stem[:30]}</small><br>{lbl}</div>'
        self._r(200,"text/html",h+"</body></html>")

    def _table_state(self, s):
        try:
            doc = _build_table_state(s)
        except Exception as e:
            body = json.dumps({"error": str(e)})
            return self._r(500, "application/json", body)
        # Cloudflare adds a cf-ray header to every proxied request, so
        # its presence reliably distinguishes Rodney (remote, via the
        # tunnel) from the dealer's local /table monitor. Once a tunnel
        # poll has been seen this night, every non-tunnel poll gets a
        # redacted view that hides Rodney's hole cards from anyone
        # sitting at the table. The flag is cleared at start_night.
        is_tunnel = bool(self.headers.get("cf-ray"))
        if is_tunnel:
            s.rodney_tunnel_seen = True
            variant = "r"
        elif getattr(s, "rodney_tunnel_seen", False):
            _redact_remote_downs(doc)
            variant = "l"
        else:
            variant = "f"
        body = json.dumps(doc)
        # ETag short-circuit: include the variant so a cached redacted
        # body never gets served as the full doc (and vice versa).
        etag = f'W/"v{doc.get("version", 0)}-{variant}"'
        inm = self.headers.get("If-None-Match")
        if inm == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _table_page(self, s):
        self._r(200, "text/html; charset=utf-8", TABLE_HTML)

    def _console_page(self, s):
        self._r(200, "text/html", CONSOLE_HTML)


    def _training_file(self, name):
        p = TRAINING_DIR / name
        if not p.exists(): return self._r(404,"text/plain","Not found")
        self._r(200, "image/jpeg" if p.suffix==".jpg" else "text/plain",
                p.read_bytes() if p.suffix==".jpg" else p.read_text())

    def _apply_settings(self, s, data):
        """Apply the Setup modal payload: dealer, players, YOLO min-conf,
        Pi presence threshold. Every field is optional."""
        import urllib.request
        ge = s.game_engine
        dealer_name = (data.get("dealer") or "").strip()
        if dealer_name:
            for i, p2 in enumerate(ge.players):
                if p2.name.lower() == dealer_name.lower():
                    ge.dealer_index = i
                    ge._update_dealer()
                    log.log(f"[CONSOLE] Dealer set to {p2.name}")
                    break
        players = data.get("players")
        if isinstance(players, list):
            valid = [n for n in players if n in PLAYER_NAMES]
            if valid:
                s.console_active_players = valid
                log.log(f"[CONSOLE] Active players: {', '.join(valid)}")
        yolo = data.get("yolo_min_conf")
        if yolo is not None:
            try:
                s.monitor.yolo_min_conf = max(0.0, min(1.0, float(yolo)))
                log.log(f"[CONSOLE] YOLO min conf → {s.monitor.yolo_min_conf:.2f}")
            except (TypeError, ValueError):
                pass
        presence = data.get("presence_threshold")
        if presence is not None:
            try:
                pval = float(presence)
                s.pi_presence_threshold = pval
                url = f"{s.pi_base_url.rstrip('/')}/presence_threshold"
                body = json.dumps({"value": pval}).encode()
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=3).read()
                log.log(f"[CONSOLE] Pi presence_threshold → {pval}")
            except Exception as e:
                log.log(f"[CONSOLE] presence_threshold push failed: {e}")
        settle = data.get("brio_settle_s")
        if settle is not None:
            try:
                s.brio_settle_s = max(0.0, min(10.0, float(settle)))
                log.log(f"[CONSOLE] Brio settle → {s.brio_settle_s:.2f}s")
            except (TypeError, ValueError):
                pass
        whisper_min = data.get("whisper_min_energy_threshold")
        if whisper_min is not None:
            try:
                # 0..4000 covers Whisper's natural range; clamp anything
                # absurd to keep "off" (= 0) reachable.
                s.whisper_min_energy_threshold = max(
                    0.0, min(4000.0, float(whisper_min))
                )
                log.log(
                    f"[CONSOLE] Whisper min energy threshold → "
                    f"{s.whisper_min_energy_threshold:.0f}"
                )
            except (TypeError, ValueError):
                pass
        # Persist host-managed tunables so restarts keep the user's choices.
        _save_host_config({
            "brio_settle_s": s.brio_settle_s,
            "pi_presence_threshold": s.pi_presence_threshold,
            "yolo_min_conf": s.monitor.yolo_min_conf,
            "whisper_min_energy_threshold": s.whisper_min_energy_threshold,
        })

    def _proxy_slot_image(self, s, slot_num: int):
        """Proxy the Pi's /slots/<n>/image through Neo so the browser sees
        a same-origin URL. Avoids cross-origin/HTTPS-upgrade issues that
        leave the verify modal showing a broken-image placeholder."""
        import urllib.request
        url = f"{s.pi_base_url.rstrip('/')}/slots/{slot_num}/image"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = resp.read()
                ct = resp.headers.get("Content-Type", "image/jpeg")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            log.log(f"[TABLE] slot_image proxy failed for slot {slot_num}: "
                    f"{type(e).__name__}: {e}")
            self._r(502, "text/plain", "pi image unavailable")

    def _card_asset(self, name):
        """Serve a pretty card image (SVG or PNG) from host/static/cards/.
        Guards against path traversal."""
        root = (Path(__file__).parent / "static" / "cards").resolve()
        p = (root / name).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            return self._r(404, "text/plain", "Not found")
        ext = p.suffix.lower()
        if ext not in (".svg", ".png") or not p.exists():
            return self._r(404, "text/plain", "Not found")
        mime = "image/svg+xml" if ext == ".svg" else "image/png"
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)


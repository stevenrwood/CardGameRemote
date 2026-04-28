"""Process entry point — argv parsing, capture/state/server bootstrap,
the bg_loop background thread, and the optional speech-input listener.

Stays a thin module: the heavy lifting lives in the topic-named
modules (brio_watcher, guided_deal, voice_dispatch, etc.). Every
runtime singleton (the BrioCapture, the ZoneMonitor, the AppState)
is constructed here and stashed on ``runtime_state._state`` so the
re-exports across the codebase keep resolving to a single instance.
"""

import argparse
import http.server
import subprocess
import sys
import time
from threading import Thread

import cv2

from log_buffer import log
from speech import speech, _resolve_best_voice
from brio_capture import FrameCapture
from zone_monitor import ZoneMonitor, TRAINING_DIR
from pi_scanner import _load_host_config, _save_host_config
from host_constants import (
    DEFAULT_BRIO_SETTLE_S,  # noqa: F401  (kept in case CLI flags reference it)
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CAMERA_NAME,
    DEFAULT_RESOLUTION,
    DEFAULT_THRESHOLD,
)
from calibration import Calibration
from app_state import AppState
from frame_utils import crop_circle, draw_overlay, to_jpeg
from brio_watcher import _console_watch_dealer
from pi_poll import _pi_poll_start
from test_modes import (
    _collect_auto_cycle,
    _deal_check_dealer_zone,
    _deal_check_zones_clear,
    _deal_retry_missing,
    _deal_scan_all_zones,
)
from voice_dispatch import _process_voice_command
import runtime_state


def _stats_bump(state, key, delta=1):
    """Increment a key in state.stats if state exists. Zone monitor uses
    this to tally YOLO vs Claude recognitions without needing a hard
    dependency on AppState being initialized yet (first-run safety)."""
    if state is None or not hasattr(state, "stats"):
        return
    state.stats[key] = state.stats.get(key, 0) + delta


def bg_loop():
    # The original loop had no exception handling, so a single bad
    # iteration (e.g. a cv2 op throwing on a transient frame) silently
    # killed the thread — recognition stopped, /snapshot froze on the
    # last good frame, and the only way out was a host restart. Wrap
    # the body so the loop survives and logs the problem instead.
    while not runtime_state._state.quit_flag:
        try:
            _bg_loop_iter()
        except Exception as e:
            log.log(f"[BG_LOOP] iteration error: {type(e).__name__}: {e}")
            time.sleep(1)


def _bg_loop_iter():
    s = runtime_state._state
    frame = s.capture.capture()
    if frame is None:
        time.sleep(1)
        return
    s.latest_frame = frame
    # Recognition/motion detection runs FIRST so card arrival
    # doesn't wait on the display-JPEG encode below. On a 4K
    # Brio frame to_jpeg was eating 0.5-1.5s per bg_loop pass.
    if s.monitoring and s.cal.ok:
        _console_watch_dealer(s, frame)

    # Display JPEG is cheap once we downscale — the UI renders
    # it inside a small iframe anyway. Keep the overlay drawn
    # on the full frame (zone coordinates are in 4K space),
    # then scale the encoded output.
    disp = crop_circle(frame, s.cal).copy()
    draw_overlay(disp, s.cal, s.monitor)
    small = cv2.resize(disp, (1280, 720), interpolation=cv2.INTER_AREA)
    s.latest_jpg = to_jpeg(small, 70)

    # Data collection auto-cycle
    if s.collect_mode:
        _collect_auto_cycle(s)

    # Test mode: check if card appeared in the active zone
    tm = s.test_mode
    if tm and tm["waiting"] == "card" and s.cal.ok:
        zone = s.cal.zones[tm["zone_idx"]]
        crop = s.monitor.check_single(frame, zone)
        if crop is not None:
            log.log(f"[{zone['name']}] Card detected, recognizing...")
            result = s.monitor.last_card.get(zone["name"], "")
            if not result or result == "No card":
                s.monitor._recognize(zone["name"], crop)
                result = s.monitor.last_card.get(zone["name"], "No card")
            if result and result != "No card":
                tm["result"] = result
                tm["waiting"] = "confirm"
                tm["confirm_time"] = time.time()
                speech.say(f"{zone['name']}, {result}")

    # Test mode: auto-confirm after 4 seconds
    if tm and tm["waiting"] == "confirm":
        if time.time() - tm.get("confirm_time", 0) > 4:
            # Auto-confirm — advance to next zone
            tm["zone_idx"] += 1
            if tm["zone_idx"] >= len(s.cal.zones):
                s.test_mode = None
                log.log("[TEST] All zones tested")
            else:
                tm["waiting"] = "card"
                tm["result"] = ""
                next_name = s.cal.zones[tm["zone_idx"]]["name"]
                speech.say(f"{next_name} is next")
                log.log(f"[TEST] Auto-confirmed. Next: {next_name}")

    # Deal mode
    dm = s.deal_mode
    if dm and s.cal.ok:
        if dm["phase"] == "dealing":
            _deal_check_dealer_zone(s)
        elif dm["phase"] == "settling":
            if time.time() - dm.get("settle_time", 0) >= 2:
                log.log("[DEAL] Scanning all zones")
                dm["phase"] = "scanning"
                dm["announced_this_round"] = set()
                _deal_scan_all_zones(s)
        elif dm["phase"] == "retry_missing":
            _deal_retry_missing(s)
        elif dm["phase"] == "waiting_to_clear":
            _deal_check_zones_clear(s)

    time.sleep(1)  # 1 second capture rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help=f"avfoundation camera index; default is auto-detected "
                             f"by name (looks for '{DEFAULT_CAMERA_NAME}')")
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME,
                        help="Substring of the avfoundation device name to prefer "
                             "when auto-selecting the camera")
    parser.add_argument("--cv-camera-index", type=int, default=None,
                        help="Force a specific OpenCV VideoCapture index, skipping "
                             "name-based lookup. Use this when multiple 4K cameras "
                             "are connected and the auto-picker opens the wrong one.")
    parser.add_argument("--brio-focus", type=int, default=None,
                        help="Manual focus value for the Brio (0..255, lower = "
                             "farther). Omitting the flag leaves autofocus on.")
    parser.add_argument("--brio-zoom", type=int, default=None,
                        help="UVC zoom-absolute for the Brio (100..500, "
                             "100 = 1x / fully zoomed out / widest FOV). "
                             "Omitting leaves whatever the camera was on; "
                             "set 100 once to lock max field of view.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--voice", type=str, default=None,
                        help="Base voice name for `say`. The actual voice used "
                             "is the highest-quality installed variant "
                             "(Premium > Enhanced > base). Overrides SPEECH_VOICE env.")
    parser.add_argument("--listen", action="store_true",
                        help="Enable phase-filtered speech-input commands via "
                             "MLX Whisper on the Mac mic (\"The game is …\", "
                             "\"{player}, {card}\", \"Correction: …\", "
                             "\"Confirmed\", \"Pot is right\", \"{player} folds\", "
                             "\"Same game again\"). Requires mlx-whisper, "
                             "SpeechRecognition, pyaudio, and portaudio.")
    args = parser.parse_args()

    if args.voice:
        speech.voice = _resolve_best_voice(args.voice)
        log.log(f"Speech voice overridden to: {speech.voice}")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    camera_index = args.camera
    if camera_index is None:
        camera_index = FrameCapture.find_index_by_name(args.camera_name)
        if camera_index is None:
            camera_index = DEFAULT_CAMERA_INDEX
            log.log(f"Camera: '{args.camera_name}' not found in avfoundation devices, "
                    f"falling back to index {camera_index}")

    # OpenCV VideoCapture index can differ from AVFoundations enumeration
    # when multiple 4K cameras are attached. Persist whatever value the user
    # passes via --cv-camera-index so the next run picks up the Brio without
    # having to re-specify it.
    _persisted_cfg = _load_host_config()
    cv_idx = args.cv_camera_index
    if cv_idx is not None:
        _save_host_config({"cv_camera_index": cv_idx})
        log.log(f"[CAPTURE] Saved cv_camera_index={cv_idx} to host config")
    elif "cv_camera_index" in _persisted_cfg:
        cv_idx = _persisted_cfg["cv_camera_index"]
        log.log(f"[CAPTURE] Loaded cv_camera_index={cv_idx} from host config")

    # Brio manual focus override — autofocus hunts on the low-contrast
    # felt background, so pin a focus position once and keep it.
    brio_focus = args.brio_focus
    if brio_focus is not None:
        _save_host_config({"brio_focus": brio_focus})
        log.log(f"[CAPTURE] Saved brio_focus={brio_focus} to host config")
    elif "brio_focus" in _persisted_cfg:
        brio_focus = _persisted_cfg["brio_focus"]
        log.log(f"[CAPTURE] Loaded brio_focus={brio_focus} from host config")

    # Brio UVC zoom — 100 (1×, full FOV) is the widest setting.
    brio_zoom = args.brio_zoom
    if brio_zoom is not None:
        _save_host_config({"brio_zoom": brio_zoom})
        log.log(f"[CAPTURE] Saved brio_zoom={brio_zoom} to host config")
    elif "brio_zoom" in _persisted_cfg:
        brio_zoom = _persisted_cfg["brio_zoom"]
        log.log(f"[CAPTURE] Loaded brio_zoom={brio_zoom} from host config")

    capture = FrameCapture(camera_index, args.resolution,
                           camera_name_hint=args.camera_name,
                           cv_index_override=cv_idx,
                           focus=brio_focus,
                           zoom=brio_zoom)
    log.log(f"Camera {camera_index}, resolution {capture.resolution}")

    # Wait for the persistent ffmpeg stream to warm up enough to produce
    # a frame. AVFoundation can take several seconds to open the Brio.
    print("  Waiting for first frame from camera stream...")
    frame = None
    deadline = time.time() + 15.0
    while time.time() < deadline:
        frame = capture.capture()
        if frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        tail = getattr(capture, "_stderr_tail", b"").decode(errors="replace").strip()
        hint = f"\n  ffmpeg stderr tail: {tail}" if tail else ""
        sys.exit(
            f"  ERROR: No frames from camera after 15s. "
            f"Is another app holding the Brio?{hint}"
        )
    print(f"  OK: {frame.shape[1]}x{frame.shape[0]}")

    cal = Calibration()
    cal.load()

    def _per_card_speech(name, card_text):
        # Default: just "{name}, {card}". The active game class can
        # extend this — 7/27 appends "with N or less down below"
        # when the player's running up-card total nears 27.
        default = f"{name}, {card_text}"
        s = runtime_state._state
        impl = getattr(s, "current_game_impl", None) if s else None
        if impl is None:
            return default
        try:
            return impl.annotate_card_speech(s, name, card_text, default)
        except Exception as e:
            log.log(f"[SPEECH] annotate_card_speech failed: {e!r}")
            return default

    monitor = ZoneMonitor(
        threshold=args.threshold,
        get_zones=lambda: cal.zones,
        stats_cb=lambda key: _stats_bump(runtime_state._state, key),
        speech_formatter=_per_card_speech,
    )
    runtime_state._state = AppState(capture, cal, monitor)
    runtime_state._state.latest_frame = frame
    # Apply any persisted YOLO min-confidence now that the monitor exists.
    _persisted = _load_host_config()
    if "yolo_min_conf" in _persisted:
        try:
            monitor.yolo_min_conf = max(0.0, min(1.0, float(_persisted["yolo_min_conf"])))
        except (TypeError, ValueError):
            pass

    # Import Handler now that runtime_state._state is populated. The
    # http_server module's `from overhead_test import …` block runs
    # at this point and sees a fully-loaded helper surface.
    from http_server import Handler

    # Start server. ThreadingHTTPServer gives each client connection its own
    # thread so a browser's keep-alive polling (e.g. /table/state every 500ms)
    # can't starve other clients like /console or /logview.
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
    server.daemon_threads = True
    Thread(target=server.serve_forever, daemon=True).start()

    # Auto-start Pi slot poller so /table populates Rodney's hand without a
    # manual kick. The loop handles Pi-unreachable with a retry delay, so
    # starting it here is safe even if the Pi is off.
    _pi_poll_start(runtime_state._state)
    log.log(f"Pi poller started against {runtime_state._state.pi_base_url}")
    log.log("Server at http://localhost:8888")

    # Start background capture
    Thread(target=bg_loop, daemon=True).start()

    # Optional speech-input listener. The SpeechListener runs on its
    # own background thread; _process_voice_command is the callback
    # for each parsed command. Mic calibration takes ~2s at startup
    # and model load ~30s on first run (downloads a whisper-small
    # checkpoint to ~/.cache), so this is gated behind --listen to
    # keep the normal boot path fast.
    if args.listen:
        log.log("[VOICE] --listen set; initialising speech listener")
        try:
            from speech_recognition_module import (
                SpeechListener, set_log_function,
            )
            set_log_function(log.log)
            # Pull the live game catalog so Whisper's bias prompt
            # covers every game name the dealer might say (including
            # ones Whisper would otherwise hallucinate, like
            # "Hodgecargo" for "High Chicago").
            game_names = list(runtime_state._state.game_engine.templates.keys())
            listener = SpeechListener(
                callback=_process_voice_command,
                game_names=game_names,
                # Re-read the live AppState value each iteration so
                # the dealer can adjust the floor mid-night via the
                # Setup modal without restarting the host.
                min_energy_threshold_fn=(
                    lambda: runtime_state._state.whisper_min_energy_threshold
                ),
            )
            listener.start()
            runtime_state._state.whisper_listener = listener
            log.log("[VOICE] speech-input listener started (--listen)")
        except Exception as e:
            import traceback
            log.log(f"[VOICE] FAILED to start listener: "
                    f"{type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines():
                log.log(f"[VOICE]   {line}")
            log.log("[VOICE] install deps with: "
                    "pip3 install mlx-whisper SpeechRecognition pyaudio")
    else:
        log.log("[VOICE] --listen NOT set; speech input disabled")

    # Open browser
    time.sleep(1)
    subprocess.Popen(["open", "http://localhost:8888"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if cal.ok:
        print(f"  Calibration: {len(cal.zones)} zones")
    else:
        print("  No calibration — use browser to calibrate")

    print("  All UI is in the browser. Press Ctrl+C to quit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        runtime_state._state.quit_flag = True


if __name__ == "__main__":
    main()

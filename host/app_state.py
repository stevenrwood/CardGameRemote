"""AppState — the single big mutable object that ties together
calibration, the Brio capture thread, the ZoneMonitor, the game
engine, and the per-hand console / table flow.
"""

import os
from threading import Lock

from game_engine import GameEngine
from pi_scanner import _load_host_config

from host_constants import DEFAULT_BRIO_SETTLE_S, PLAYER_NAMES


def _stats_bump(state, key, delta=1):
    """Increment a key in state.stats if state exists. Zone monitor uses
    this to tally YOLO vs Claude recognitions without needing a hard
    dependency on AppState being initialized yet (first-run safety)."""
    if state is None or not hasattr(state, "stats"):
        return
    state.stats[key] = state.stats.get(key, 0) + delta


class AppState:
    def __init__(self, capture, cal, monitor):
        self.capture = capture
        self.cal = cal
        self.monitor = monitor
        self.monitoring = False
        self.latest_frame = None
        self.latest_jpg = None  # cropped + overlay
        self.quit_flag = False
        self.test_mode = None   # None or {"zone_idx":0, "waiting":"card"|"confirm", "result":""}
        # Deal test mode
        self.deal_mode = None
        # Data collection mode
        self.collect_mode = None  # None or {"card_idx":0, "pass":1, "captured":False}
        # Console (dealer phone UI)
        self.game_engine = GameEngine()
        self.console_active_players = list(PLAYER_NAMES)  # who's playing tonight
        self.console_last_round_cards = []  # cards from last upcard scan
        self.console_hand_cards = []  # all confirmed up cards this hand: [{player, card, round}]
        self.console_up_round = 0     # current up-card round number
        self.console_total_up_rounds = 0  # total up-card rounds in this game
        self.console_scan_phase = "idle"  # "idle" | "watching" | "scanned" | "confirmed"
        self.console_settle_time = 0.0
        # Per-round flags driven by the watcher. Reset on confirm /
        # next-round so the gate closes again and the auto-scan
        # trigger can re-fire next round.
        self._dealer_zone_done = False
        self._dealer_zone_trigger_fired = False
        self._missing_prompt_fired = False
        self._missing_speech_count = {}
        self._empty_scan_count = {}
        self._zone_prev_pending = {}
        # Name of the most recently-dealt game this session. Used by the
        # "Same game again" / "Let's run that back" voice command to
        # repeat the previous hand without having to say its name again.
        self.last_game_name = ""
        # ---- Remote-player table view ("/table") ----
        # state_version is bumped whenever anything the observer needs changes.
        # Rodney's down cards come from the Pi scanner; other players only
        # expose a down-count + up-cards on the observer view.
        self.table_state_version = 0
        # Set the first time a /table/state poll arrives bearing a
        # Cloudflare tunnel header (cf-ray). Once true, every
        # non-tunnel /table/state response is redacted to hide
        # Rodney's hole cards — the whole point of running
        # scripts/start_rodney_tunnel.sh is that only Rodney sees his
        # own down cards. Cleared at start_night so a tunnel from a
        # prior session doesn't carry over.
        self.rodney_tunnel_seen = False
        # Rodney's down-card slots. Indexed by scanner slot number so a
        # fluctuating or re-scanned slot replaces its prior value instead of
        # appending a new entry. Each value is {rank, suit, confidence}.
        self.rodney_downs = {}         # slot_num -> {rank, suit, confidence} (verified / auto-accepted)
        # 7/27: when Rodney has 2 down cards the UI asks him to pick one to
        # flip face-up. Once chosen, the card moves here and the LED for
        # that slot blinks so the dealer knows which to physically lift.
        self.rodney_flipped_up = None   # None or {rank, suit, slot}
        self.slot_pending = {}         # slot_num -> {rank, suit, confidence} (latest low-conf guess, awaiting confirm)
        self.slot_empty = {}           # slot_num -> True when poller sees no card
        self.verify_queue = []         # FIFO of slot_nums that need manual verify after /api/console/confirm
        self.pending_verify = None     # None or {guess, slot, prompt}
        self.table_log = []            # [{ts, msg}]
        self.pi_base_url = os.environ.get("PI_BASE_URL", "http://pokerbuddy.local:8080")
        # Tunables loaded from ~/.cardgame_host.json if present. Setup modal
        # writes them back when the user saves, so defaults only matter on
        # first run. pi_presence_threshold is a cached mirror of the Pi's
        # own persisted value — pushed to the Pi on save.
        cfg = _load_host_config()
        self.brio_settle_s = float(cfg.get("brio_settle_s", DEFAULT_BRIO_SETTLE_S))
        self.pi_presence_threshold = float(cfg.get("pi_presence_threshold", 140.0))
        # Floor for the SpeechListener's recognizer.energy_threshold.
        # 0 = no floor (auto-calibrate to ambient noise as before).
        # Dealers in quiet rooms see calibration land at 8-12 which
        # makes Whisper trigger on near-silence and produce looped
        # hallucinations; setting this to 25-40 dampens that.
        self.whisper_min_energy_threshold = float(
            cfg.get("whisper_min_energy_threshold", 0.0)
        )
        # Set in main() after SpeechListener is constructed (when
        # --listen). Lets /api/console/state expose the recognizer's
        # live energy_threshold so the dealer can see what it
        # actually settled on tonight.
        self.whisper_listener = None
        self.pi_polling = False
        self.pi_poll_thread = None
        self.pi_prev_slots = {}        # slot_num -> last-seen card code (e.g. "Ac")
        # Slot-by-slot guided dealing state. None = not guiding; otherwise
        # {expecting: int, num_slots: int}. Regular _pi_poll_loop skips its
        # work while this is set.
        self.guided_deal = None
        self.guided_deal_thread = None
        # "Poker night" flag — set by Start Poker / Start Testing,
        # cleared by Exit Poker. The console UI gates the game
        # dropdown + action controls on this.
        self.night_active = False
        # Which kind of session is active: "poker" or "testing".
        # Drives the Setup-modal title and any UI hint that wants
        # to surface the mode.
        self.session_kind = "poker"
        # High-level console state machine surfaced to the UI.
        # "idle" | "dealing" | "betting" | "hand_over"
        self.console_state = "idle"
        # 5 Card Draw / draw-phase support: Rodney marks cards during
        # betting (a set of slot numbers). When he hits "Request cards",
        # those slots' LEDs light up and guided flow refills them. One
        # draw per hand. betting_round distinguishes pre-draw vs post-draw
        # for games with two betting rounds.
        # Per-hand recognition stats: how many cards YOLO and Claude each
        # produced, and of those how many the user corrected. Reset on
        # every /api/console/deal and logged on /api/console/end.
        # pi_auto: count of Pi guided-deal scans that landed ≥ GUIDED_GOOD_CONF
        #   and were auto-committed without the user seeing a verify modal.
        # pi_verify_right: user opened the verify modal on a low-conf guess
        #   and accepted the Pi's suggestion unchanged → Pi was right.
        # pi_verify_wrong: same modal, but user edited rank/suit → Pi was wrong.
        # Together these tell us whether GUIDED_GOOD_CONF is set too aggressive
        # or too conservative across a night of play.
        self.stats = {
            "yolo_right": 0, "yolo_wrong": 0,
            "claude_right": 0, "claude_wrong": 0,
            "pi_auto": 0,
            "pi_verify_right": 0, "pi_verify_wrong": 0,
        }
        self.rodney_marked_slots: set[int] = set()
        self.rodney_drew_this_hand = False
        # Count of completed draws this hand (3 Toed Pete has 3). Reset on
        # deal; incremented after each guided replace completes. Used to
        # index into the games list of DRAW phases for max-marks, and to
        # decide when to advance to hand_over instead of another draw.
        self.rodney_draws_done = 0
        self.console_betting_round = 0
        # Games with a trailing down card (7 Card Stud's 7th street, FTQ's
        # final down): after the last up round's Pot-is-right we run a second
        # guided session for that down slot. This flag, once set, means the
        # next Pot-is-right goes straight to hand_over instead of starting
        # trailing deal again.
        self.console_trailing_done = False
        self.table_lock = Lock()       # guards rodney_downs / pending_verify / table_log
        self.pi_confidence_threshold = 0.70  # >= this → auto-accept
        # The Pi's template matcher returns low-but-nonzero confidence for
        # every slot (including empty ones), so "empty" as a confidence
        # threshold doesn't work reliably. We trust the Pi's recognized
        # flag + any nonzero confidence as "something was seen" so the
        # weak-but-present scan still ends up in slot_pending and can be
        # manually verified.
        self.pi_empty_threshold = 0.0
        self._pi_last_logged = {}            # slot_num -> last logged code, throttle log spam
        self.pi_flash_held = False           # tracked so we don't spam hold/release
        self.folded_players = set()     # Rodney's view of who's folded this hand
        self.freezes = {}               # 7/27: player_name -> freezes in a row
        # Last name spoken as "Bet first" by _announce_round; consulted
        # by /api/table/fold to decide whether to re-announce the next-
        # highest hand once the previously-announced high hand folds.
        self.last_bet_first: str | None = None
        # True when Deal pinged the Pi and got no answer; stays set until the
        # next Deal so we skip hitting the Pi (flash/hold, /slots, LEDs, etc).
        self.pi_offline = False
        # Per-hand game class instance (subclass of games.BaseGame).
        # Created in /api/console/deal from the template's class_name and
        # cleared on end_hand. None when idle / between hands.
        self.current_game_impl = None
        # ---- Challenge-game state (High, Low, High etc.) ----
        # Persists across hands; only resets on the 1-out-all-pass award.
        self.pot_cents: int = 0
        # Hand config set via the console Start Game button. Persists
        # across hands — the next Start Game pre-fills with these so
        # the dealer only has to change what's actually different.
        self.ante_cents: int = 50          # $0.50 default
        self.betting_limit: str = "1_2"    # "1_2" | "1_all_way" | "pot"
        # 5-second voice-override window between /api/console/deal and
        # the actual deal kickoff. ante_timer fires
        # /api/console/finish_ante if the dealer doesn't override the
        # default ante by speech in that window. Cancelled when an
        # AnteCommand arrives or the dealer clicks through manually.
        self.ante_timer = None
        # None when the current hand isn't a Challenge variant. Otherwise
        # 0/1/2 for rounds 1/2/3. Resets to 0 on reshuffle.
        self.challenge_round_index = None
        self.challenge_shuffle_count = 0
        # Per-player vote state: {name: {"went_out": bool,
        #   "passes": int (0-2), "out_round": int|None,
        #   "out_slots": list[int]}}.
        # Dealer clicks Pass or Out buttons on the console to set the
        # per-player state, or Rodney uses his /table buttons. End
        # Round button resolves the round based on how many are out.
        self.challenge_per_player: dict = {}
        # Rodney's committed go-out slots (empty until he commits).
        self.rodney_out_slots: list = []
        # Round 3 overflow: the two cards displaced from slots 4/5 sit
        # face-down in front of Rodney off-scanner. We log them so the
        # resolve UI can surface what was there.
        self.rodney_overflow: list = []

"""
Continuous speech recognition for poker game commands.

Uses Apple's Speech framework (SFSpeechRecognizer) via pyobjc for
low-latency, on-device, streaming recognition on Apple Silicon.

Recognized command types:
    - Game selection: "The game is Follow the Queen"
    - Up-card calls:  "David, 4 of clubs"

Usage:
    # Standalone test:
    python speech_recognition_module.py

    # Integration:
    from speech_recognition_module import SpeechListener
    listener = SpeechListener(callback=my_handler)
    listener.start()   # non-blocking, runs on background thread
    listener.stop()

Requirements:
    pip install pyobjc-framework-Speech pyobjc-framework-AVFoundation
    macOS 14+ (Sonoma) for on-device speech recognition
    Microphone permission must be granted in System Settings > Privacy
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

# Apple frameworks — imported lazily in SpeechListener to allow
# parsing logic to work without pyobjc installed.
AVFoundation = None
Speech = None
NSRunLoop = None
NSDate = None

log = logging.getLogger(__name__)

# External log function — set by the app to route messages to the web UI
_external_log = None

def set_log_function(fn):
    """Set an external log function (e.g., LogBuffer.log) for web UI output."""
    global _external_log
    _external_log = fn

def _log(msg):
    """Log to both Python logging and external log buffer."""
    log.info(msg)
    if _external_log:
        _external_log(f"[SPEECH] {msg}")


def _ensure_apple_frameworks():
    """Import Apple Speech/AV frameworks. Raises ImportError with install hint."""
    global AVFoundation, Speech, NSRunLoop, NSDate
    if Speech is not None:
        return
    try:
        import AVFoundation as _av
        import Speech as _sp
        from Foundation import NSRunLoop as _rl, NSDate as _nd
        AVFoundation = _av
        Speech = _sp
        NSRunLoop = _rl
        NSDate = _nd
    except ImportError as e:
        raise ImportError(
            "Apple Speech frameworks not found. Install with:\n"
            "  pip install pyobjc-framework-Speech pyobjc-framework-AVFoundation"
        ) from e

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

GAME_NAMES = [
    "5 Card Draw",
    "3 Toed Pete",
    "7 Card Stud",
    "7 Stud Deuces Wild",
    "Follow the Queen",
    "High Chicago",
    "High Low High Challenge",
    "7 27",
    "Texas Hold'em",
]

PLAYER_NAMES = ["Steve", "Bill", "David", "Joe", "Rodney"]

RANKS = {
    "ace": "A", "one": "A", "1": "A",
    "two": "2", "deuce": "2", "2": "2",
    "three": "3", "3": "3",
    "four": "4", "4": "4",
    "five": "5", "5": "5",
    "six": "6", "6": "6",
    "seven": "7", "7": "7",
    "eight": "8", "8": "8",
    "nine": "9", "9": "9",
    "ten": "10", "10": "10",
    "jack": "J",
    "queen": "Q",
    "king": "K",
}

SUITS = {
    "clubs": "clubs",
    "club": "clubs",
    "diamonds": "diamonds",
    "diamond": "diamonds",
    "hearts": "hearts",
    "heart": "hearts",
    "spades": "spades",
    "spade": "spades",
}

# Patterns Whisper/Apple Speech commonly mistranscribes
GAME_ALIASES = {
    "three toed pete": "3 Toed Pete",
    "3 toad pete": "3 Toed Pete",
    "three toad pete": "3 Toed Pete",
    "five card draw": "5 Card Draw",
    "5 card draw": "5 Card Draw",
    "seven card stud": "7 Card Stud",
    "7 card stud": "7 Card Stud",
    "seven stud deuces wild": "7 Stud Deuces Wild",
    "7 stud deuces wild": "7 Stud Deuces Wild",
    "follow the queen": "Follow the Queen",
    "high chicago": "High Chicago",
    "high low high challenge": "High Low High Challenge",
    "high low high": "High Low High Challenge",
    "seven twenty-seven": "7 27",
    "seven twenty seven": "7 27",
    "7 27": "7 27",
    "texas hold'em": "Texas Hold'em",
    "texas holdem": "Texas Hold'em",
    "texas hold them": "Texas Hold'em",
}


# ---------------------------------------------------------------------------
# Parsed command types
# ---------------------------------------------------------------------------

@dataclass
class GameCommand:
    """Recognized game selection command."""
    game_name: str        # Canonical game name from GAME_NAMES
    raw_text: str         # What was actually heard
    confidence: float     # 0.0 - 1.0 fuzzy match score


@dataclass
class CardCallCommand:
    """Recognized up-card call command."""
    player: str           # Canonical player name
    rank: str             # Normalized rank (A, 2-10, J, Q, K)
    suit: str             # Normalized suit (clubs, diamonds, hearts, spades)
    raw_text: str
    confidence: float


@dataclass
class UnrecognizedCommand:
    """Speech was detected but didn't match any known pattern."""
    raw_text: str


# Type alias for the callback
CommandCallback = Callable  # callback(GameCommand | CardCallCommand | UnrecognizedCommand)


# ---------------------------------------------------------------------------
# Text parsing / fuzzy matching
# ---------------------------------------------------------------------------

def _fuzzy_match_game(text):
    """Try to match text against known game names. Returns (name, score) or None."""
    text_lower = text.lower().strip()

    # Direct alias lookup
    for alias, canonical in GAME_ALIASES.items():
        if alias in text_lower:
            return canonical, 1.0

    # Fuzzy match against all game names
    best_score = 0.0
    best_name = None
    for name in GAME_NAMES:
        score = SequenceMatcher(None, text_lower, name.lower()).ratio()
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= 0.6 and best_name is not None:
        return best_name, best_score

    return None


def _parse_card_call(text):
    """
    Try to parse "[player], [rank] of [suit]" from text.
    Handles variations like:
        "David four of clubs"
        "David, 4 of clubs"
        "steve ace of spades"
    """
    text_lower = text.lower().strip()

    # Find player name
    matched_player = None
    remaining = text_lower
    for name in PLAYER_NAMES:
        pattern = re.compile(r'\b' + re.escape(name.lower()) + r'\b')
        match = pattern.search(text_lower)
        if match:
            matched_player = name
            remaining = text_lower[match.end():].strip().lstrip(",").strip()
            break

    if matched_player is None:
        return None

    # Find rank
    matched_rank = None
    rank_end = 0
    for word, abbrev in RANKS.items():
        pattern = re.compile(r'\b' + re.escape(word) + r'\b')
        match = pattern.search(remaining)
        if match:
            matched_rank = abbrev
            rank_end = match.end()
            break

    if matched_rank is None:
        return None

    # Find suit (should appear after "of")
    after_rank = remaining[rank_end:].strip()
    # Remove "of" if present
    after_rank = re.sub(r'^of\s+', '', after_rank).strip()

    matched_suit = None
    for word, canonical in SUITS.items():
        if word in after_rank:
            matched_suit = canonical
            break

    if matched_suit is None:
        return None

    return CardCallCommand(
        player=matched_player,
        rank=matched_rank,
        suit=matched_suit,
        raw_text=text,
        confidence=1.0,
    )


def parse_speech(text):
    """
    Parse a transcribed speech string into a structured command.
    Tries game commands first, then card calls.
    """
    # Check for game selection pattern: "the game is ___"
    game_match = re.search(r'(?:the\s+)?game\s+is\s+(.+)', text, re.IGNORECASE)
    if game_match:
        game_text = game_match.group(1)
        result = _fuzzy_match_game(game_text)
        if result:
            name, score = result
            return GameCommand(game_name=name, raw_text=text, confidence=score)

    # Also try matching the whole utterance as a game name
    # (someone might just say "Follow the Queen" without "the game is")
    result = _fuzzy_match_game(text)
    if result:
        name, score = result
        if score >= 0.75:  # higher threshold when there's no "the game is" prefix
            return GameCommand(game_name=name, raw_text=text, confidence=score)

    # Try card call
    card = _parse_card_call(text)
    if card:
        return card

    return UnrecognizedCommand(raw_text=text)


# ---------------------------------------------------------------------------
# Apple Speech Framework listener
# ---------------------------------------------------------------------------

class SpeechListener:
    """
    Continuous speech recognition using Apple's SFSpeechRecognizer.

    Streams audio from the default microphone, delivers partial and final
    transcription results, parses them into game commands, and calls the
    provided callback.

    The recognition task auto-restarts when Apple's ~60s limit is reached.
    """

    def __init__(self, callback=None, locale="en-US"):
        _ensure_apple_frameworks()
        self._callback = callback or self._default_callback
        self._locale = locale
        self._recognizer = None
        self._audio_engine = None
        self._request = None
        self._task = None
        self._running = False
        self._thread = None
        self._last_final_text = ""
        self._last_partial_text = ""
        self._restart_count = 0

    @staticmethod
    def _default_callback(command):
        """Default callback prints to console."""
        if isinstance(command, GameCommand):
            print(f"[GAME] {command.game_name} (confidence: {command.confidence:.2f}) -- heard: \"{command.raw_text}\"")
        elif isinstance(command, CardCallCommand):
            print(f"[CARD] {command.player}: {command.rank} of {command.suit} -- heard: \"{command.raw_text}\"")
        elif isinstance(command, UnrecognizedCommand):
            print(f"[????] \"{command.raw_text}\"")

    def start(self):
        """Start listening in a background thread."""
        if self._running:
            _log("SpeechListener already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="SpeechListener")
        self._thread.start()
        _log("SpeechListener started")

    def stop(self):
        """Stop listening and clean up."""
        self._running = False
        self._stop_recognition()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        _log("SpeechListener stopped")

    def _run_loop(self):
        """Background thread: start recognition and auto-restart on timeout."""
        while self._running:
            try:
                self._start_recognition()
                # Run the NSRunLoop to process audio callbacks
                while self._running and self._task is not None:
                    NSRunLoop.currentRunLoop().runUntilDate_(
                        NSDate.dateWithTimeIntervalSinceNow_(0.1)
                    )
            except Exception as e:
                _log(f"Speech recognition error: {e}")
                self._stop_recognition()
                if self._running:
                    time.sleep(1.0)  # brief pause before restart

    def _start_recognition(self):
        """Set up and start a new recognition task."""
        # Create recognizer
        locale = Speech.NSLocale.alloc().initWithLocaleIdentifier_(self._locale)
        self._recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)

        _log(f"Recognizer available: {self._recognizer.isAvailable()}")
        if not self._recognizer.isAvailable():
            raise RuntimeError("Speech recognizer not available")

        # Request authorization (will prompt user on first run)
        auth_status = Speech.SFSpeechRecognizer.authorizationStatus()
        _log(f"Speech auth status: {auth_status} (3=authorized)")
        if auth_status != Speech.SFSpeechRecognizerAuthorizationStatusAuthorized:
            _log("Requesting speech recognition authorization...")
            event = threading.Event()
            result_status = [None]

            def auth_handler(status):
                result_status[0] = status
                event.set()

            Speech.SFSpeechRecognizer.requestAuthorization_(auth_handler)
            event.wait(timeout=30.0)

            if result_status[0] != Speech.SFSpeechRecognizerAuthorizationStatusAuthorized:
                raise RuntimeError(
                    f"Speech recognition not authorized (status={result_status[0]}). "
                    "Grant access in System Settings > Privacy > Speech Recognition."
                )

        # Create recognition request
        self._request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
        self._request.setShouldReportPartialResults_(True)

        # Use server-based recognition (on-device model may not be downloaded)
        _log("Using server-based recognition")

        # Set up audio engine
        self._audio_engine = AVFoundation.AVAudioEngine.alloc().init()
        input_node = self._audio_engine.inputNode()
        record_format = input_node.outputFormatForBus_(0)

        _log(f"Audio format: {record_format.sampleRate()}Hz, {record_format.channelCount()}ch")

        # Install tap to feed audio to the recognizer
        self._tap_count = 0
        def audio_tap(buffer, when):
            self._tap_count += 1
            if self._tap_count == 1:
                _log("First audio buffer received from mic")
            elif self._tap_count == 50:
                _log(f"Audio flowing — {self._tap_count} buffers received")
            self._request.appendAudioPCMBuffer_(buffer)

        input_node.installTapOnBus_bufferSize_format_block_(
            0, 1024, record_format, audio_tap
        )

        # Start audio engine
        self._audio_engine.prepare()
        _log("Starting audio engine...")
        success, error = self._audio_engine.startAndReturnError_(None)
        if not success:
            raise RuntimeError(f"Failed to start audio engine: {error}")
        _log("Audio engine started — listening for speech")

        # Start recognition task
        self._last_final_text = ""
        self._last_partial_text = ""

        self._result_count = 0

        # Store reference to prevent garbage collection by ObjC runtime
        def _result_handler(result, error):
            try:
                if error:
                    error_desc = str(error)
                    if any(code in error_desc for code in ["216", "209", "1110"]):
                        _log(f"Recognition ended (will restart): {error_desc}")
                    else:
                        _log(f"Recognition error: {error_desc}")
                    self._task = None
                    return

                if result is None:
                    return

                self._result_count += 1
                text = result.bestTranscription().formattedString()
                is_final = result.isFinal()

                # Log all results for debugging
                _log(f"Result #{self._result_count} (final={is_final}): \"{text}\"")

                if is_final:
                    new_text = text[len(self._last_final_text):].strip()
                    if new_text and new_text != self._last_partial_text:
                        self._process_text(new_text)
                    self._last_final_text = text
                    self._last_partial_text = ""
                else:
                    new_text = text[len(self._last_final_text):].strip()
                    if new_text != self._last_partial_text:
                        self._last_partial_text = new_text
                        self._try_partial(new_text)
            except Exception as e:
                _log(f"result_handler exception: {e}")

        # Keep reference alive
        self._result_handler_ref = _result_handler

        self._task = self._recognizer.recognitionTaskWithRequest_resultHandler_(
            self._request, self._result_handler_ref
        )

        self._restart_count += 1
        _log(f"Recognition task started (#{self._restart_count})")

    def _stop_recognition(self):
        """Stop the current recognition task and audio engine."""
        if self._task:
            self._task.cancel()
            self._task = None

        if self._audio_engine:
            self._audio_engine.stop()
            self._audio_engine.inputNode().removeTapOnBus_(0)
            self._audio_engine = None

        if self._request:
            self._request.endAudio()
            self._request = None

    def _try_partial(self, text: str):
        """
        Check partial results for high-confidence card calls.
        Card calls have a rigid structure so we can fire early.
        """
        card = _parse_card_call(text)
        if card:
            # We have player + rank + suit — that's a complete card call
            _log(f"Partial match: {text}")
            self._callback(card)
            # Reset partial tracking so we don't double-fire
            self._last_final_text += text
            self._last_partial_text = ""

    def _process_text(self, text):
        """Process a final transcription result."""
        _log(f"Heard: \"{text}\"")
        command = parse_speech(text)
        self._callback(command)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Speech Recognition Test")
    print("=" * 60)
    print()
    print("Say things like:")
    print('  "The game is Follow the Queen"')
    print('  "David, four of clubs"')
    print('  "Steve, ace of spades"')
    print()
    print("Press Ctrl+C to stop.")
    print()

    listener = SpeechListener()

    try:
        listener.start()
        # Keep main thread alive
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()
        print("Done.")

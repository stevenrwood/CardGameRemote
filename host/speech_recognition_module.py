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
CommandCallback = Callable[[GameCommand | CardCallCommand | UnrecognizedCommand], None]


# ---------------------------------------------------------------------------
# Text parsing / fuzzy matching
# ---------------------------------------------------------------------------

def _fuzzy_match_game(text: str) -> tuple[str, float] | None:
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


def _parse_card_call(text: str) -> CardCallCommand | None:
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


def parse_speech(text: str) -> GameCommand | CardCallCommand | UnrecognizedCommand:
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

    def __init__(self, callback: CommandCallback | None = None, locale: str = "en-US"):
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
            log.warning("SpeechListener already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="SpeechListener")
        self._thread.start()
        log.info("SpeechListener started")

    def stop(self):
        """Stop listening and clean up."""
        self._running = False
        self._stop_recognition()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("SpeechListener stopped")

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
                log.error(f"Speech recognition error: {e}")
                self._stop_recognition()
                if self._running:
                    time.sleep(1.0)  # brief pause before restart

    def _start_recognition(self):
        """Set up and start a new recognition task."""
        # Create recognizer
        locale = Speech.NSLocale.alloc().initWithLocaleIdentifier_(self._locale)
        self._recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)

        if not self._recognizer.isAvailable():
            raise RuntimeError("Speech recognizer not available")

        # Request authorization (will prompt user on first run)
        auth_status = Speech.SFSpeechRecognizer.authorizationStatus()
        if auth_status != Speech.SFSpeechRecognizerAuthorizationStatusAuthorized:
            log.info("Requesting speech recognition authorization...")
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

        # Enable on-device recognition if available (macOS 14+)
        if hasattr(self._request, 'requiresOnDeviceRecognition'):
            self._request.setRequiresOnDeviceRecognition_(True)

        # Set up audio engine
        self._audio_engine = AVFoundation.AVAudioEngine.alloc().init()
        input_node = self._audio_engine.inputNode()
        record_format = input_node.outputFormatForBus_(0)

        # Install tap to feed audio to the recognizer
        def audio_tap(buffer, when):
            self._request.appendAudioPCMBuffer_(buffer)

        input_node.installTapOnBus_bufferSize_format_block_(
            0, 1024, record_format, audio_tap
        )

        # Start audio engine
        self._audio_engine.prepare()
        success, error = self._audio_engine.startAndReturnError_(None)
        if not success:
            raise RuntimeError(f"Failed to start audio engine: {error}")

        # Start recognition task
        self._last_final_text = ""
        self._last_partial_text = ""

        def result_handler(result, error):
            if error:
                error_desc = str(error)
                # Code 216 = "Retry" (normal timeout after ~60s of silence)
                # Code 209 = recognition task finished
                # Code 1110 = no speech detected
                if any(code in error_desc for code in ["216", "209", "1110"]):
                    log.debug(f"Recognition ended (expected): {error_desc}")
                else:
                    log.warning(f"Recognition error: {error_desc}")
                self._task = None  # triggers restart in _run_loop
                return

            if result is None:
                return

            text = result.bestTranscription().formattedString()

            if result.isFinal():
                # Only process the new portion of text
                new_text = text[len(self._last_final_text):].strip()
                if new_text and new_text != self._last_partial_text:
                    self._process_text(new_text)
                self._last_final_text = text
                self._last_partial_text = ""
            else:
                # Partial result — check if we have a complete-looking command
                new_text = text[len(self._last_final_text):].strip()
                if new_text != self._last_partial_text:
                    self._last_partial_text = new_text
                    # Only fire on partials that look like complete commands
                    # (helps reduce latency for card calls)
                    self._try_partial(new_text)

        self._task = self._recognizer.recognitionTaskWithRequest_resultHandler_(
            self._request, result_handler
        )

        self._restart_count += 1
        log.info(f"Recognition task started (#{self._restart_count})")

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
            log.info(f"Partial match (card call): {text}")
            self._callback(card)
            # Reset partial tracking so we don't double-fire
            self._last_final_text += text
            self._last_partial_text = ""

    def _process_text(self, text: str):
        """Process a final transcription result."""
        log.info(f"Final transcription: \"{text}\"")
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

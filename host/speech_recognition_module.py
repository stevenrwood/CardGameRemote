"""
Continuous speech recognition for poker game commands.

Uses the SpeechRecognition library with macOS's built-in recognizer
for continuous listening.

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
    pip install SpeechRecognition pyaudio
    brew install portaudio
"""

import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

# External log function — set by the app to route messages to the web UI
_external_log = None

def set_log_function(fn):
    global _external_log
    _external_log = fn

def _log(msg):
    print(f"  [SPEECH] {msg}")
    if _external_log:
        _external_log(f"[SPEECH] {msg}")

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
    "four": "4", "4": "4", "for": "4",
    "five": "5", "5": "5",
    "six": "6", "6": "6",
    "seven": "7", "7": "7",
    "eight": "8", "8": "8",
    "nine": "9", "9": "9",
    "ten": "10", "10": "10",
    "jack": "J", "jacks": "J", "jacket": "J", "jackets": "J",
    "queen": "Q", "queens": "Q",
    "king": "K", "kings": "K",
}

SUITS = {
    "clubs": "clubs", "club": "clubs",
    "diamonds": "diamonds", "diamond": "diamonds",
    "hearts": "hearts", "heart": "hearts",
    "spades": "spades", "spade": "spades",
    "space": "spades", "spaces": "spades", "face": "spades", "fades": "spades",
    "private": "spades",  # common misrecognitions
}

GAME_ALIASES = {
    "three toed pete": "3 Toed Pete",
    "3 toad pete": "3 Toed Pete",
    "three toad pete": "3 Toed Pete",
    "five card draw": "5 Card Draw",
    "5 card draw": "5 Card Draw",
    "five car draw": "5 Card Draw",
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
    game_name: str
    raw_text: str
    confidence: float

@dataclass
class CardCallCommand:
    player: str
    rank: str
    suit: str
    raw_text: str
    confidence: float

@dataclass
class UnrecognizedCommand:
    raw_text: str


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

def _fuzzy_match_game(text):
    text_lower = text.lower().strip()
    for alias, canonical in GAME_ALIASES.items():
        if alias in text_lower:
            return canonical, 1.0
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
    text_lower = text.lower().strip()
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

    after_rank = remaining[rank_end:].strip()
    after_rank = re.sub(r'^of\s+', '', after_rank).strip()

    matched_suit = None
    for word, canonical in SUITS.items():
        if word in after_rank:
            matched_suit = canonical
            break
    if matched_suit is None:
        return None

    return CardCallCommand(
        player=matched_player, rank=matched_rank, suit=matched_suit,
        raw_text=text, confidence=1.0,
    )


def parse_speech(text):
    """Parse speech text. Returns a list of commands (may find multiple card calls in one chunk)."""
    results = []

    # Check for game selection
    game_match = re.search(r'(?:the\s+)?game\s+is\s+(.+)', text, re.IGNORECASE)
    if game_match:
        result = _fuzzy_match_game(game_match.group(1))
        if result:
            results.append(GameCommand(game_name=result[0], raw_text=text, confidence=result[1]))
            return results

    result = _fuzzy_match_game(text)
    if result and result[1] >= 0.75:
        results.append(GameCommand(game_name=result[0], raw_text=text, confidence=result[1]))
        return results

    # Try to extract multiple card calls from one chunk
    # Split on player names to find individual calls
    cards = _parse_multiple_card_calls(text)
    if cards:
        return cards

    # Single card call
    card = _parse_card_call(text)
    if card:
        return [card]

    return [UnrecognizedCommand(raw_text=text)]


def _parse_multiple_card_calls(text):
    """Try to split text into multiple player card calls."""
    text_lower = text.lower()

    # Find all player name positions
    positions = []
    for name in PLAYER_NAMES:
        for m in re.finditer(r'\b' + re.escape(name.lower()) + r'\b', text_lower):
            positions.append((m.start(), name))

    if len(positions) < 2:
        return None  # single or no player names, use regular parse

    # Sort by position
    positions.sort()

    # Split text at each player name and try to parse each segment
    cards = []
    for i, (pos, name) in enumerate(positions):
        if i + 1 < len(positions):
            segment = text[pos:positions[i+1][0]]
        else:
            segment = text[pos:]

        card = _parse_card_call(segment.strip())
        if card:
            cards.append(card)

    return cards if cards else None


# ---------------------------------------------------------------------------
# Speech listener using SpeechRecognition library
# ---------------------------------------------------------------------------

class SpeechListener:
    """
    Continuous speech recognition using the SpeechRecognition library.
    Listens in short chunks, transcribes each, and parses for commands.
    """

    def __init__(self, callback=None, locale="en-US"):
        self._callback = callback or self._default_callback
        self._running = False
        self._thread = None

    @staticmethod
    def _default_callback(command):
        if isinstance(command, GameCommand):
            print(f"[GAME] {command.game_name} (confidence: {command.confidence:.2f}) -- heard: \"{command.raw_text}\"")
        elif isinstance(command, CardCallCommand):
            print(f"[CARD] {command.player}: {command.rank} of {command.suit} -- heard: \"{command.raw_text}\"")
        elif isinstance(command, UnrecognizedCommand):
            print(f"[????] \"{command.raw_text}\"")

    def start(self):
        if self._running:
            _log("Already running")
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        _log("SpeechListener started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        _log("SpeechListener stopped")

    def _listen_loop(self):
        try:
            import speech_recognition as sr
        except ImportError:
            _log("ERROR: pip install SpeechRecognition pyaudio")
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5  # shorter pause = split phrases faster

        try:
            mic = sr.Microphone()
        except Exception as e:
            _log(f"ERROR: Could not open microphone: {e}")
            return

        _log("Microphone opened")

        # Calibrate for ambient noise
        with mic as source:
            _log("Calibrating for ambient noise (2 seconds)...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            _log(f"Ambient noise calibration done (threshold: {recognizer.energy_threshold:.0f})")

        _log("Listening for speech...")

        while self._running:
            try:
                with mic as source:
                    _log("Waiting for speech...")
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)

                _log("Speech detected, recognizing...")

                try:
                    # Use Apple's built-in recognizer (no internet needed)
                    text = recognizer.recognize_google(audio)
                    _log(f"Heard: \"{text}\"")

                    commands = parse_speech(text)
                    for command in commands:
                        self._callback(command)

                except sr.UnknownValueError:
                    _log("Could not understand audio")
                except sr.RequestError as e:
                    _log(f"Recognition service error: {e}")

            except sr.WaitTimeoutError:
                # No speech detected in timeout period — just loop
                pass
            except Exception as e:
                _log(f"Listen error: {e}")
                time.sleep(1)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()
        print("Done.")

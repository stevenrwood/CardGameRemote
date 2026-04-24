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

# Whisper often mishears player names
PLAYER_ALIASES = {
    "steve": "Steve", "eve": "Steve", "steep": "Steve",
    "bill": "Bill", "phil": "Bill", "built": "Bill", "pill": "Bill",
    "david": "David", "dave": "David", "give it": "David",
    "joe": "Joe", "jo": "Joe", "show": "Joe", "jo ": "Joe",
    "rodney": "Rodney", "rod": "Rodney", "ronnie": "Rodney", "honey": "Rodney",
}

RANKS = {
    "ace": "A", "one": "A", "1": "A", "aces": "A",
    "two": "2", "deuce": "2", "2": "2", "to": "2", "too": "2",
    "three": "3", "3": "3",
    "four": "4", "4": "4", "for": "4",
    "five": "5", "5": "5",
    "six": "6", "6": "6",
    "seven": "7", "7": "7",
    "eight": "8", "8": "8", "80": "8", "ate": "8",
    "nine": "9", "9": "9", "nana": "9",
    "ten": "10", "10": "10", "tennis": "10",
    "jack": "J", "jacks": "J", "jacket": "J", "jackets": "J",
    "queen": "Q", "queens": "Q",
    "king": "K", "kings": "K",
}

SUITS = {
    "clubs": "clubs", "club": "clubs",
    "diamonds": "diamonds", "diamond": "diamonds", "dime": "diamonds",
    "hearts": "hearts", "heart": "hearts", "hart": "hearts", "harts": "hearts", "hurts": "hearts",
    "spades": "spades", "spade": "spades",
    "space": "spades", "spaces": "spades", "face": "spades", "fades": "spades",
    "faze": "spades", "phase": "spades", "spain": "spades",
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
class RepeatGameCommand:
    """'Same game again' / 'Let's run that back' — the host looks up the
    last game that was dealt and starts a fresh hand of it."""
    raw_text: str

@dataclass
class CardCallCommand:
    """'{player}, {card}' spoken during an up-card round to fill in a
    card the Brio scanner missed or to pre-empt a bad recognition."""
    player: str
    rank: str
    suit: str
    raw_text: str
    confidence: float

@dataclass
class InferredCardCommand:
    """'{card}' spoken without a player name — dealer called just the
    rank+suit and expects the host to resolve which player the card
    goes to (next unfilled zone in deal order). The parser emits this
    whenever it sees a card-rank+suit with no player prefix; the
    dispatcher in overhead_test does the deal-order resolution since
    only it has access to the live console state."""
    rank: str
    suit: str
    raw_text: str
    confidence: float

@dataclass
class CorrectionCommand:
    """'Correction: {player}, {card}' — same semantic as CardCallCommand
    but valid post-scan / pre-confirm when the scanner got a zone wrong
    and the dealer wants to overwrite it before hitting Confirm."""
    player: str
    rank: str
    suit: str
    raw_text: str
    confidence: float

@dataclass
class ConfirmCommand:
    """'Confirmed' — voice equivalent of the Confirm Cards button."""
    raw_text: str

@dataclass
class PotIsRightCommand:
    """'Pot is right' — voice equivalent of the Pot Is Right button
    that ends the betting round."""
    raw_text: str

@dataclass
class FoldCommand:
    """'{player}, folds' — mark a player as folded during betting."""
    player: str
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


# Whisper-mishear → canonical text rewrites, applied before rank/suit
# extraction in both `_parse_card_call` and `_extract_card_only`. Each
# entry is (pattern, replacement) — replacement can be a string or a
# callable (per `re.sub`). Keep rules narrow (`\b` word boundaries) to
# avoid eating legitimate content.
_WHISPER_FIXES = [
    # Speech-dictation artifacts + rank substitutions we've seen in the
    # wild. Preserved from the original inline tables.
    (r'^oh,?\s+', ''),
    (r'\bto a\b', 'two of'),
    (r'\bto your\b', 'two of'),
    (r'\b80\b', 'eight of'),
    (r'\bat a\b', 'eight of'),
    (r'\bfive at\b', 'five of'),
    (r"\bo'clock\b", 'of clubs'),
    (r'\bfly with\b', 'five of'),
    (r"\bit's a\b", 'ace of'),
    (r"\bin his\b", 'ace of'),
    (r'\band diamond\b', 'ace of diamonds'),
    (r'\band spade\b', 'ace of spades'),
    (r'\band heart\b', 'ace of hearts'),
    (r'\band club\b', 'ace of clubs'),
    (r'\bi spade\b', 'ace of spades'),
    (r'\bi heart\b', 'ace of hearts'),
    (r'\bi diamond\b', 'ace of diamonds'),
    (r'\bi club\b', 'ace of clubs'),
    (r'\bin space\b', 'nine of spades'),
    (r'\bfor diamond\b', 'four of diamonds'),
    (r'\bfor heart\b', 'four of hearts'),
    (r'\bfor spade\b', 'four of spades'),
    (r'\bfor club\b', 'four of clubs'),
    # New rules targeting the 2026-04-24 FTQ log drops:
    #   "Fix the spades" → "Six of Spades"  (Whisper mishear of "Six")
    #   "Four words of spades" → "Four of Spades"
    (r'\bfix\b', 'six'),
    (r'\bsticks\b', 'six'),
    (r'\bsix the\b', 'six of'),
    (r'\bfour the\b', 'four of'),
    (r'\bseven the\b', 'seven of'),
    (r'\bten the\b', 'ten of'),
    (r'\bmine\b', 'nine'),
    (r'\bline\b', 'nine'),
    (r'\bsent\b', 'seven'),
    (r'\bset\b', 'seven'),
    # "four words of spades" / "4 word of hearts" — strip the spurious
    # "word"/"words" that Whisper inserts between the rank and "of".
    (r'\b(\d+|ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two)\s+words?\s+of\b',
     r'\1 of'),
]


def _apply_whisper_fixes(text_lower: str) -> str:
    """Pre-process a lowercased transcript by applying the shared
    Whisper-mishear rewrite table. Returns a string ready for rank +
    suit extraction."""
    for pattern, replacement in _WHISPER_FIXES:
        text_lower = re.sub(pattern, replacement, text_lower)
    return text_lower


def _parse_card_call(text):
    text_lower = text.lower().strip()
    # Fix common dictation/Whisper substitutions before parsing
    text_lower = _apply_whisper_fixes(text_lower)
    matched_player = None
    remaining = text_lower
    # Try exact player names first, then aliases
    for name in PLAYER_NAMES:
        pattern = re.compile(r'\b' + re.escape(name.lower()) + r'\b')
        match = pattern.search(text_lower)
        if match:
            matched_player = name
            remaining = text_lower[match.end():].strip().lstrip(",").strip()
            break
    if matched_player is None:
        for alias, name in PLAYER_ALIASES.items():
            pattern = re.compile(r'\b' + re.escape(alias) + r'\b')
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
    text_stripped = text.strip()
    text_lower = text_stripped.lower()

    # --- Single-phrase commands first (cheapest to check) --------------

    # "Confirmed" — voice equivalent of the Confirm Cards button. Also
    # tolerate slight mis-transcriptions like "confirm" or "confirmed."
    # standalone, but NOT inside a longer phrase like "confirmed bill 4
    # of clubs" (which would be a weird sentence anyway).
    if re.fullmatch(r"(confirmed?|confirm cards?)[.!]?", text_lower):
        return [ConfirmCommand(raw_text=text)]

    # "Pot is right" (sometimes mis-heard as "pot is writes" etc.)
    if re.fullmatch(r"(the\s+)?pot[\s,]*is\s+(right|write|ripe)[.!]?", text_lower):
        return [PotIsRightCommand(raw_text=text)]

    # "Same game again" / "Let's run that back" / "Run that back"
    if re.fullmatch(
        r"(same game( again)?|let'?s run that back|run that back)[.!]?",
        text_lower,
    ):
        return [RepeatGameCommand(raw_text=text)]

    # --- Fold: "{player}, folds" / "{player} folds out" etc. -----------
    fold_match = re.match(
        r"^([a-z]+)[\s,]+fold(s|ed|ing)?\b", text_lower
    )
    if fold_match:
        player = _canonical_player(fold_match.group(1))
        if player is not None:
            return [FoldCommand(player=player, raw_text=text, confidence=1.0)]

    # --- Correction prefix: "Correction: {player}, {card}" -------------
    if re.match(r"^correction[\s,:]+", text_lower):
        stripped = re.sub(r"^correction[\s,:]+", "", text_stripped, flags=re.IGNORECASE)
        card = _parse_card_call(stripped)
        if card is not None:
            return [CorrectionCommand(
                player=card.player, rank=card.rank, suit=card.suit,
                raw_text=text, confidence=card.confidence,
            )]
        # Fall through — maybe Whisper heard "correction" at the start
        # of a game phrase. Don't swallow it silently.

    # --- Game selection -----------------------------------------------

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

    # --- Card calls ---------------------------------------------------

    # Try to extract multiple card calls from one chunk
    # Split on player names to find individual calls
    cards = _parse_multiple_card_calls(text)
    if cards:
        return cards

    # Single card call with a player named
    card = _parse_card_call(text)
    if card:
        return [card]

    # Single card call with NO player — dealer said just "4 of Clubs".
    # Emit InferredCardCommand; the dispatcher resolves which player
    # it goes to against live deal order + current zone state.
    orphan = _extract_card_only(text)
    if orphan is not None:
        r, sv = orphan
        return [InferredCardCommand(
            rank=r, suit=sv, raw_text=text, confidence=0.8,
        )]

    return [UnrecognizedCommand(raw_text=text)]


def _canonical_player(name_lower):
    """Map a fuzzily-heard first name back to the canonical player list.
    Returns the capitalized canonical name, or None if no player looks
    remotely like what we heard."""
    if not name_lower:
        return None
    # Prefer an exact alias match (handles common Whisper mis-hears
    # like "eve" → "Steve") before falling back to a fuzzy ratio.
    alias_hit = PLAYER_ALIASES.get(name_lower)
    if alias_hit:
        return alias_hit
    best = None
    best_score = 0.0
    for canonical in PLAYER_NAMES:
        score = SequenceMatcher(None, name_lower, canonical.lower()).ratio()
        if score > best_score:
            best_score = score
            best = canonical
    if best is not None and best_score >= 0.75:
        return best
    return None


def _extract_card_only(text):
    """Try to extract rank+suit from text that has no player name. Returns (rank, suit) or None."""
    text_lower = text.lower().strip()
    text_lower = _apply_whisper_fixes(text_lower)

    matched_rank = None
    rank_end = 0
    for word, abbrev in RANKS.items():
        pattern = re.compile(r'\b' + re.escape(word) + r'\b')
        match = pattern.search(text_lower)
        if match:
            matched_rank = abbrev
            rank_end = match.end()
            break
    if not matched_rank:
        return None

    after_rank = text_lower[rank_end:].strip()
    after_rank = re.sub(r'^of\s+', '', after_rank).strip()

    matched_suit = None
    for word, canonical in SUITS.items():
        if word in after_rank:
            matched_suit = canonical
            break
    if not matched_suit:
        return None

    return (matched_rank, matched_suit)


def _extract_player_only(text):
    """Check if text is just a player name (with no card). Returns player name or None."""
    text_lower = text.lower().strip().rstrip(".,!?")
    # Check exact names
    for name in PLAYER_NAMES:
        if text_lower == name.lower():
            return name
    # Check aliases
    for alias, name in PLAYER_ALIASES.items():
        if text_lower == alias:
            return name
    return None


def _parse_multiple_card_calls(text):
    """Try to split text into multiple player card calls."""
    text_lower = text.lower()

    # Find all player name positions (exact names + aliases)
    positions = []
    for name in PLAYER_NAMES:
        for m in re.finditer(r'\b' + re.escape(name.lower()) + r'\b', text_lower):
            positions.append((m.start(), name))
    for alias, name in PLAYER_ALIASES.items():
        for m in re.finditer(r'\b' + re.escape(alias) + r'\b', text_lower):
            # Don't add if we already have a match at this position
            if not any(abs(p[0] - m.start()) < 3 for p in positions):
                positions.append((m.start(), name))

    if len(positions) < 2:
        return None  # single or no player names, use regular parse

    # Sort by position
    positions.sort()

    # If the utterance starts with a card-rank+suit BEFORE the first
    # player name (Whisper drops the first name surprisingly often —
    # "Jack of clubs. Joe, 6 of diamonds..."), prepend an
    # InferredCardCommand. The dispatcher will resolve it against
    # live deal order.
    cards = []
    lead = text[:positions[0][0]].strip()
    if lead:
        rc = _extract_card_only(lead)
        if rc is not None:
            r, sv = rc
            cards.append(InferredCardCommand(
                rank=r, suit=sv, raw_text=text, confidence=0.8,
            ))

    # Split text at each player name and try to parse each segment
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
        self._pending_player = None   # player name heard without a card
        self._pending_time = 0

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
            import mlx_whisper
        except ImportError as e:
            _log(f"ERROR: missing package: {e}")
            _log("Install: pip install mlx-whisper SpeechRecognition pyaudio")
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.2  # keep name + card together
        recognizer.phrase_threshold = 0.3
        recognizer.non_speaking_duration = 0.8

        try:
            mic = sr.Microphone()
        except Exception as e:
            _log(f"ERROR: Could not open microphone: {e}")
            return

        _log("Microphone opened")

        with mic as source:
            _log("Calibrating for ambient noise (2 seconds)...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            _log(f"Calibration done (threshold: {recognizer.energy_threshold:.0f})")

        # Load whisper model
        model_name = "mlx-community/whisper-small.en-mlx"
        # Prompt biases Whisper toward our vocabulary
        whisper_prompt = (
            "Steve, Bill, David, Joe, Rodney. "
            "Ace of spades. King of hearts. Queen of diamonds. Jack of clubs. "
            "Two of hearts. Three of spades. Four of clubs. Five of diamonds. "
            "Six of hearts. Seven of spades. Eight of clubs. Nine of diamonds. Ten of hearts. "
            "Ranks are Ace, King, Queen, Jack, Ten, Nine, Eight, Seven, Six, "
            "Five, Four, Three, Two — never Fix, Sticks, Mine, Line, Sent, or "
            "Words in place of a rank. "
            "The game is Follow the Queen. The game is 5 Card Draw. "
            "The game is 7 Card Stud. The game is 3 Toed Pete. "
            "Same game again. Let's run that back. "
            "Correction: David, 4 of clubs. "
            "Confirmed. Pot is right. "
            "Bill folds. Steve folds."
        )
        _log("Loading Whisper model (first run downloads ~500MB)...")
        try:
            import numpy as np
            import tempfile
            import wave
            import os
            # Warm up with a short silent WAV
            warmup_path = tempfile.mktemp(suffix=".wav")
            with wave.open(warmup_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
            mlx_whisper.transcribe(warmup_path, path_or_hf_repo=model_name)
            os.unlink(warmup_path)
            _log("Whisper model loaded and ready")
        except Exception as e:
            _log(f"Whisper warmup error: {e}")

        _log("Listening for speech...")

        while self._running:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=8)

                _log("Speech detected, transcribing with Whisper...")

                import tempfile
                import os
                wav_path = tempfile.mktemp(suffix=".wav")
                try:
                    wav_data = audio.get_wav_data()
                    with open(wav_path, 'wb') as f:
                        f.write(wav_data)

                    t0 = time.time()
                    result = mlx_whisper.transcribe(
                        wav_path,
                        path_or_hf_repo=model_name,
                        language="en",
                        initial_prompt=whisper_prompt,
                    )
                    text = result.get("text", "").strip()
                    elapsed = time.time() - t0

                    if text:
                        _log(f"Heard ({elapsed:.1f}s): \"{text}\"")
                        commands = parse_speech(text)
                        for command in commands:
                            if isinstance(command, UnrecognizedCommand):
                                card_only = _extract_card_only(command.raw_text)
                                player_only = _extract_player_only(command.raw_text)

                                if card_only:
                                    # We have a card — try to find a player
                                    player = None
                                    # 1. Check if pending player from previous chunk
                                    if self._pending_player and (time.time() - self._pending_time) < 5:
                                        player = self._pending_player
                                        _log(f"Matched pending player '{player}' with card")
                                    self._pending_player = None

                                    if player:
                                        self._callback(CardCallCommand(
                                            player=player, rank=card_only[0], suit=card_only[1],
                                            raw_text=f"(matched) {player} + {command.raw_text}",
                                            confidence=0.8))
                                    else:
                                        _log(f"Card without player: {card_only[0]} of {card_only[1]}")
                                        self._callback(command)
                                elif player_only:
                                    self._pending_player = player_only
                                    self._pending_time = time.time()
                                    _log(f"Pending player: {player_only}")
                                else:
                                    self._callback(command)
                            else:
                                self._pending_player = None
                                self._callback(command)
                    else:
                        _log(f"No speech in audio ({elapsed:.1f}s)")

                finally:
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)

            except Exception as e:
                if "WaitTimeoutError" in type(e).__name__:
                    pass
                else:
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

"""
Per-slot template-matching card recognizer.

Used as the second-tier fallback in scan_controller's recognition
pipeline (YOLO → slot templates → nothing). The web UI captures one
template per (slot × card) under controlled scanner-box lighting,
preprocesses each crop into an "ink" image (binarized + suit-aware),
and matches new captures against those templates with normalized
cross-correlation.

The older whole-card "corner template" path (a 52-template dictionary
matched against the top-left corner of the card) was removed once
YOLO became the primary recognizer — synthetic samples produced
useless templates that confidently mis-identified everything as a
single card, so the fallback was actively harmful rather than a
safety net.

Usage:
    detector = CardDetector("reference/")
    result = detector.identify_slot(crop_bgr, slot_num=3)  # CardResult or None
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


logger = logging.getLogger(__name__)


RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["hearts", "diamonds", "clubs", "spades"]
SUIT_ABBREV = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades"}

SUIT_SYMBOLS = {
    "hearts": "♥",
    "diamonds": "♦",
    "clubs": "♣",
    "spades": "♠",
}


@dataclass
class CardResult:
    rank: str
    suit: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "suit": self.suit,
            "confidence": round(self.confidence, 3),
        }

    def __str__(self) -> str:
        sym = SUIT_SYMBOLS.get(self.suit, self.suit)
        return f"{self.rank}{sym} ({self.confidence:.0%})"


class CardDetector:
    """Identifies playing cards by matching pre-trained slot templates."""

    # Standard size for slot-template matching. Slot crops are resized
    # to this for correlation; chosen tall+narrow to match the slot
    # aspect ratio.
    SLOT_TEMPLATE_SIZE = (96, 256)  # (width, height)

    def __init__(self, reference_dir: str):
        self.reference_dir = Path(reference_dir)
        # Per-slot template cache: slot_num -> {(rank, suit): prep_image}
        self.slot_templates: dict[int, dict[tuple[str, str], np.ndarray]] = {}
        self.reload_slot_templates()

    def reload_slot_templates(self):
        """Load per-slot templates from reference/slot_templates/slot<N>/<card>.png.

        Also migrates any legacy flat layout (slot_templates/<card>.png) into
        slot4/ on first load, so existing training data isn't lost.
        """
        root = self.reference_dir / "slot_templates"
        if not root.exists():
            self.slot_templates = {}
            return
        # One-time migration of legacy flat files → slot4/
        legacy = [f for f in root.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpg")]
        if legacy:
            dest = root / "slot4"
            dest.mkdir(exist_ok=True)
            for f in legacy:
                if self._parse_card_name(f.stem) is None:
                    continue
                f.rename(dest / f.name)
            print(f"Migrated {len(legacy)} legacy slot templates into slot4/")

        self.slot_templates = {}
        for slot_dir in sorted(root.iterdir()):
            if not slot_dir.is_dir():
                continue
            m = re.match(r"^slot(\d+)$", slot_dir.name)
            if not m:
                continue
            slot_num = int(m.group(1))
            entries: dict[tuple[str, str], np.ndarray] = {}
            for f in sorted(slot_dir.iterdir()):
                if f.suffix.lower() not in (".png", ".jpg"):
                    continue
                parsed = self._parse_card_name(f.stem)
                if parsed is None:
                    continue
                bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                entries[parsed] = self._prep_slot_crop(bgr)
            if entries:
                self.slot_templates[slot_num] = entries
        total = sum(len(v) for v in self.slot_templates.values())
        print(f"Loaded slot templates: {total} across {len(self.slot_templates)} slots")

    def slot_template_path(self, slot_num: int, rank: str, suit: str) -> Path:
        d = self.reference_dir / "slot_templates" / f"slot{slot_num}"
        d.mkdir(parents=True, exist_ok=True)
        suit_letter = suit[0].lower()
        return d / f"{rank}{suit_letter}.png"

    def save_slot_template(self, slot_num: int, rank: str, suit: str, crop_bgr: np.ndarray):
        """Persist a BGR slot crop as a reference template for a specific slot."""
        path = self.slot_template_path(slot_num, rank, suit)
        cv2.imwrite(str(path), crop_bgr)
        self.slot_templates.setdefault(slot_num, {})[(rank, suit)] = self._prep_slot_crop(crop_bgr)

    def list_slot_templates(self, slot_num: int | None = None) -> dict[int, list[tuple[str, str]]]:
        """Return {slot_num: [(rank, suit), ...]}. If slot_num given, only that slot."""
        if slot_num is not None:
            return {slot_num: sorted(self.slot_templates.get(slot_num, {}).keys())}
        return {n: sorted(v.keys()) for n, v in self.slot_templates.items()}

    def has_any_slot_templates(self) -> bool:
        return any(self.slot_templates.values())

    def _prep_slot_crop(self, bgr: np.ndarray) -> np.ndarray:
        """Convert a slot crop to the normalized ink image used for matching."""
        ink = self._preprocess_ink(bgr)
        return cv2.resize(ink, self.SLOT_TEMPLATE_SIZE)

    def identify_slot(self, crop_bgr: np.ndarray, slot_num: int | None = None) -> CardResult | None:
        """Identify a card from a pre-cropped slot image.

        Templates are captured through the same mirror as the runtime crops,
        so they share one orientation — we do NOT try a 180° rotated version.
        Doing so previously caused 6 and 9 to read as each other because a
        point-reflected 6 is visually identical to a 9.

        If slot_num is given and that slot has trained templates, match against
        only those. Otherwise fall back to matching against every loaded
        template (union across slots).
        """
        if slot_num is not None and slot_num in self.slot_templates:
            templates = self.slot_templates[slot_num]
        else:
            templates = {}
            for per_slot in self.slot_templates.values():
                templates.update(per_slot)
        if not templates:
            return None
        base = self._prep_slot_crop(crop_bgr)
        best = None
        best_score = -1.0
        for (rank, suit), tmpl in templates.items():
            r = cv2.matchTemplate(base, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(r[0][0])
            if score > best_score:
                best_score = score
                best = (rank, suit)
        if best is None:
            return None
        return CardResult(rank=best[0], suit=best[1], confidence=max(0.0, best_score))

    def _parse_card_name(self, name: str) -> tuple[str, str] | None:
        """Parse 'Kh' or '10s' into ('K', 'hearts')."""
        match = re.match(r'^(10|[2-9JQKA])([hdcs])$', name, re.IGNORECASE)
        if not match:
            return None
        rank = match.group(1).upper()
        suit = SUIT_ABBREV.get(match.group(2).lower())
        return (rank, suit) if suit else None

    def _preprocess_ink(self, bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR card region to a normalized "ink intensity" image.

        Handles both red (high-saturation) and black (low-value) ink in a
        single channel by combining inverted-value and saturation, then
        masks out the dark card border so it doesn't distort correlation.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Black ink: low value → bright after inversion.
        # Red ink: high saturation → already bright.
        inv_v = 255 - v
        ink = np.maximum(inv_v, s)

        # Mask out the dark card border (rounded-rect outline) so only
        # the lit card face contributes to correlation. The face has
        # high value; the border is dark and low-saturation.
        card_face = cv2.inRange(v, 160, 255)
        card_face = cv2.dilate(card_face, np.ones((7, 7), np.uint8), iterations=3)
        card_face = cv2.morphologyEx(
            card_face, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)
        )
        ink = cv2.bitwise_and(ink, card_face)

        return ink

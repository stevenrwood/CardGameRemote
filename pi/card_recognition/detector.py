"""
Card recognition using OpenCV template matching.

Uses whole-corner template matching: compares the top-left corner of the card
against 52 reference corner images (one per card). Simple and reliable in a
controlled environment where card position and lighting are consistent.

Usage:
    detector = CardDetector("reference/")
    result = detector.identify(image)
    # result = {"rank": "K", "suit": "hearts", "confidence": 0.95}
"""

import logging
import os
import re
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


# Standard poker card dimensions (ratio used for perspective correction)
CARD_WIDTH = 250
CARD_HEIGHT = 350

# Corner region to extract (top-left of card)
CORNER_WIDTH = 70
CORNER_HEIGHT = 105

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["hearts", "diamonds", "clubs", "spades"]
SUIT_ABBREV = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades"}

SUIT_SYMBOLS = {
    "hearts": "\u2665",
    "diamonds": "\u2666",
    "clubs": "\u2663",
    "spades": "\u2660",
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
    """Identifies playing cards by matching corner regions against references."""

    # Standard size for slot-template matching. Slot crops are resized to this
    # for correlation; chosen tall+narrow to match our slot aspect ratio.
    SLOT_TEMPLATE_SIZE = (96, 256)  # (width, height)

    def __init__(self, reference_dir: str):
        self.reference_dir = Path(reference_dir)
        self.corner_templates: dict[tuple[str, str], np.ndarray] = {}
        # Per-slot template cache: slot_num -> {(rank, suit): prep_image}
        self.slot_templates: dict[int, dict[tuple[str, str], np.ndarray]] = {}
        self._load_templates()
        self.reload_slot_templates()

    def _load_templates(self):
        """Load corner reference templates. Expected structure: reference/corners/Kh.png"""
        corners_dir = self.reference_dir / "corners"
        if not corners_dir.exists():
            print(f"No corners directory at {corners_dir}")
            return

        for f in sorted(corners_dir.iterdir()):
            if f.suffix not in (".png", ".jpg"):
                continue
            parsed = self._parse_card_name(f.stem)
            if parsed is None:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.corner_templates[parsed] = img

        print(f"Loaded {len(self.corner_templates)} corner templates")

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
        ink = self._preprocess_corner(bgr)   # works on any BGR card region
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
        import re
        match = re.match(r'^(10|[2-9JQKA])([hdcs])$', name, re.IGNORECASE)
        if not match:
            return None
        rank = match.group(1).upper()
        suit = SUIT_ABBREV.get(match.group(2).lower())
        return (rank, suit) if suit else None

    def has_any_corner_templates(self) -> bool:
        """True if at least one corner template has been loaded."""
        return bool(self.corner_templates)

    def identify(self, image: np.ndarray) -> CardResult | None:
        """
        Identify a playing card in the given image.

        Args:
            image: BGR image (from cv2.imread or camera capture)

        Returns:
            CardResult with rank, suit, and confidence, or None if no
            card found OR if no corner templates have been trained yet.
            Returning None on the no-templates case (with a one-time
            log warning) lets the service stay up on a fresh image —
            routes that do their own thing on None already cope, and
            /ping reports detector availability so the user sees why
            recognition is silent.
        """
        if not self.corner_templates:
            if not getattr(self, "_warned_no_templates", False):
                logger.warning(
                    "identify() called with no corner templates loaded — "
                    "returning None. Run train.py to capture templates."
                )
                self._warned_no_templates = True
            return None

        card_img = self._extract_card(image)
        if card_img is None:
            return None

        # Try both orientations (card could be placed 180° rotated)
        results = []
        for rotation in [0, 180]:
            if rotation == 180:
                rotated = cv2.rotate(card_img, cv2.ROTATE_180)
            else:
                rotated = card_img

            corner = self._extract_corner(rotated)
            if corner is None:
                continue

            corner_gray = self._preprocess_corner(corner)
            match = self._match_corner(corner_gray)
            if match:
                results.append(match)

        if not results:
            return None

        return max(results, key=lambda r: r.confidence)

    def identify_from_file(self, image_path: str) -> CardResult | None:
        """Convenience method to identify a card from a file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.identify(image)

    def _preprocess_corner(self, corner: np.ndarray) -> np.ndarray:
        """
        Convert corner to a normalized grayscale image suitable for matching.
        Handles both red and black ink by using saturation + value channels.
        """
        hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Create an "ink intensity" image:
        # - For black ink: low value → high intensity
        # - For red ink: high saturation → high intensity
        # Invert value so dark pixels become bright
        inv_v = 255 - v

        # Combine: max of inverted-value and saturation
        # This makes both black ink and red ink appear as bright pixels
        ink = np.maximum(inv_v, s)

        # Mask out the card border (very dark corners from rounded rect SVG)
        # The card face has high value; the border has low value AND low saturation
        card_face = cv2.inRange(v, 160, 255)
        card_face = cv2.dilate(card_face, np.ones((7, 7), np.uint8), iterations=3)
        card_face = cv2.morphologyEx(
            card_face, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)
        )
        ink = cv2.bitwise_and(ink, card_face)

        return ink

    def _match_corner(self, corner_gray: np.ndarray) -> CardResult | None:
        """Match a preprocessed corner against all 52 templates."""
        best_match = None
        best_score = -1.0

        target_size = (CORNER_WIDTH, CORNER_HEIGHT)
        resized_input = cv2.resize(corner_gray, target_size)

        for (rank, suit), template in self.corner_templates.items():
            resized_tmpl = cv2.resize(template, target_size)

            # Normalized cross-correlation
            result = cv2.matchTemplate(
                resized_input, resized_tmpl, cv2.TM_CCOEFF_NORMED
            )
            score = result[0][0]

            if score > best_score:
                best_score = score
                best_match = (rank, suit)

        if best_match is None:
            return None

        return CardResult(
            rank=best_match[0],
            suit=best_match[1],
            confidence=max(0.0, best_score),
        )

    def _extract_card(self, image: np.ndarray) -> np.ndarray | None:
        """
        Find the card in the image and return a flattened, perspective-corrected
        card image of standard size (CARD_WIDTH x CARD_HEIGHT).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find the largest rectangular contour (the card)
        card_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                card_contour = approx
                max_area = area

        if card_contour is None:
            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            card_contour = np.int32(box).reshape(-1, 1, 2)

        pts = self._order_points(card_contour.reshape(4, 2).astype(np.float32))

        width = max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3])
        )
        height = max(
            np.linalg.norm(pts[0] - pts[3]),
            np.linalg.norm(pts[1] - pts[2])
        )

        if width > height:
            dst = np.array([
                [0, 0], [CARD_HEIGHT - 1, 0],
                [CARD_HEIGHT - 1, CARD_WIDTH - 1], [0, CARD_WIDTH - 1],
            ], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(image, transform, (CARD_HEIGHT, CARD_WIDTH))
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            dst = np.array([
                [0, 0], [CARD_WIDTH - 1, 0],
                [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1],
            ], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(image, transform, (CARD_WIDTH, CARD_HEIGHT))

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    def _extract_corner(self, card_img: np.ndarray) -> np.ndarray | None:
        """Extract the top-left corner region."""
        h, w = card_img.shape[:2]
        if w < CORNER_WIDTH or h < CORNER_HEIGHT:
            return None
        return card_img[0:CORNER_HEIGHT, 0:CORNER_WIDTH]

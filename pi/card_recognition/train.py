"""
Training script to generate corner reference templates from card images.

For each card image, extracts the top-left corner and saves a preprocessed
template. These templates are used by the detector for whole-corner matching.

Workflow:
1. Place card photos in the samples/ directory.
   Naming: "Kh.jpg" (King of hearts), "10s.png" (10 of spades)
   Or use SVG cards converted to PNG.

2. Run: python -m card_recognition.train --samples samples/ --output reference/

3. Templates are saved to reference/corners/ (52 files, one per card).

Usage:
    python -m card_recognition.train --samples samples/ --output reference/
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

from .detector import (
    CARD_WIDTH, CARD_HEIGHT,
    CORNER_WIDTH, CORNER_HEIGHT,
    RANKS, SUITS, SUIT_ABBREV,
    CardDetector,
)


def parse_card_filename(filename: str) -> tuple[str, str] | None:
    """Parse 'Kh.jpg' or '10s.png' into (rank, suit)."""
    stem = os.path.splitext(filename)[0]
    match = re.match(r'^(10|[2-9JQKA])([hdcs])$', stem, re.IGNORECASE)
    if not match:
        return None
    rank = match.group(1).upper()
    suit = SUIT_ABBREV.get(match.group(2).lower())
    return (rank, suit) if suit else None


def train_from_samples(samples_dir: str, output_dir: str):
    """Generate corner templates from sample card images."""
    corners_dir = os.path.join(output_dir, "corners")
    os.makedirs(corners_dir, exist_ok=True)

    # Create a detector instance for its extraction/preprocessing methods
    det = CardDetector.__new__(CardDetector)
    det.corner_templates = {}

    files = sorted(os.listdir(samples_dir))
    processed = 0
    skipped = 0

    for filename in files:
        filepath = os.path.join(samples_dir, filename)
        parsed = parse_card_filename(filename)
        if parsed is None:
            continue

        rank, suit = parsed
        image = cv2.imread(filepath)
        if image is None:
            print(f"  Could not read {filename}")
            continue

        card_img = det._extract_card(image)
        if card_img is None:
            card_img = cv2.resize(image, (CARD_WIDTH, CARD_HEIGHT))

        # Try both orientations, pick the one with more content in the corner
        best_corner = None
        best_score = -1

        for rotation in [0, 180]:
            if rotation == 180:
                rotated = cv2.rotate(card_img, cv2.ROTATE_180)
            else:
                rotated = card_img

            corner = det._extract_corner(rotated)
            if corner is None:
                continue

            preprocessed = det._preprocess_corner(corner)
            score = np.sum(preprocessed > 30)  # amount of "ink" content

            if score > best_score:
                best_score = score
                best_corner = preprocessed

        if best_corner is not None:
            suit_abbrev = {v: k for k, v in SUIT_ABBREV.items()}[suit]
            out_name = f"{rank}{suit_abbrev}.png"
            out_path = os.path.join(corners_dir, out_name)
            cv2.imwrite(out_path, best_corner)
            processed += 1
            print(f"  {filename} -> {out_name}")
        else:
            skipped += 1
            print(f"  {filename} -> SKIPPED (no corner found)")

    print(f"\nDone! {processed} corner templates saved to {corners_dir}/")
    if skipped:
        print(f"Skipped: {skipped}")

    # Check completeness
    expected = set()
    for rank in RANKS:
        for suit_abbrev in SUIT_ABBREV:
            expected.add(f"{rank}{suit_abbrev}.png")
    actual = set(os.listdir(corners_dir))
    missing = expected - actual
    if missing:
        print(f"Missing: {sorted(missing)}")
    else:
        print("All 52 cards present!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate card recognition corner templates"
    )
    parser.add_argument(
        "--samples", type=str, default="samples/",
        help="Directory containing card images (default: samples/)"
    )
    parser.add_argument(
        "--output", type=str, default="reference/",
        help="Output directory for templates (default: reference/)"
    )
    args = parser.parse_args()

    print(f"Training from samples in: {args.samples}")
    train_from_samples(args.samples, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare training data for YOLO card recognition.

Reads zone crop images + labels from training_data/,
filters out "No card" and bad labels, converts to YOLO format,
and creates a dataset ready for training.

Usage:
    python prepare_training.py [--review]  # --review to manually check labels
"""

import json
import os
import re
import shutil
import random
from pathlib import Path

TRAINING_DIR = Path(__file__).parent / "training_data"
DATASET_DIR = Path(__file__).parent / "yolo_dataset"

# 52 card classes
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["clubs", "diamonds", "hearts", "spades"]

# Build class list: "2c", "2d", "2h", "2s", "3c", ... "As"
CLASSES = []
for rank in RANKS:
    for suit in SUITS:
        CLASSES.append(f"{rank}{suit[0]}")

CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}


def parse_label(text):
    """Parse a label like '4 of Clubs' into ('4', 'clubs') or None."""
    text = text.strip().strip(".")
    if not text or "no card" in text.lower():
        return None
    # Extract "Rank of Suit" pattern
    m = re.search(
        r'(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)',
        text, re.IGNORECASE
    )
    if not m:
        return None
    rank = m.group(1).capitalize()
    suit = m.group(2).lower()
    # Normalize rank
    rank_map = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
    rank = rank_map.get(rank, rank)
    return rank, suit


def label_to_class(rank, suit):
    """Convert rank+suit to class string like '4c'."""
    return f"{rank}{suit[0]}"


def scan_training_data():
    """Scan training_data/ and return list of (image_path, label_text) pairs."""
    pairs = []
    if not TRAINING_DIR.exists():
        print("No training_data/ directory found")
        return pairs

    for jpg in sorted(TRAINING_DIR.glob("*.jpg")):
        txt = jpg.with_suffix(".txt")
        if txt.exists():
            label = txt.read_text().strip()
            # Handle multi-line labels (take first line that matches)
            for line in label.split("\n"):
                parsed = parse_label(line)
                if parsed:
                    label = line.strip()
                    break
            pairs.append((jpg, label))
        else:
            pairs.append((jpg, ""))

    return pairs


def prepare_dataset(pairs, val_split=0.2):
    """Create YOLO dataset from image/label pairs."""
    # Filter to valid cards only
    valid = []
    skipped = 0
    for img_path, label in pairs:
        parsed = parse_label(label)
        if parsed:
            valid.append((img_path, parsed[0], parsed[1]))
        else:
            skipped += 1

    print(f"Total images: {len(pairs)}")
    print(f"Valid cards: {len(valid)}")
    print(f"Skipped (No card / invalid): {skipped}")

    if not valid:
        print("No valid training data!")
        return

    # Count per class
    class_counts = {}
    for _, rank, suit in valid:
        cls = label_to_class(rank, suit)
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"\nUnique cards seen: {len(class_counts)} / 52")
    print("Distribution:")
    for cls in sorted(class_counts.keys()):
        idx = CLASS_MAP.get(cls, -1)
        rank = cls[:-1]
        suit = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}[cls[-1]]
        print(f"  {rank:>3} of {suit:<10} ({cls}): {class_counts[cls]} images")

    # Shuffle and split
    random.shuffle(valid)
    split_idx = int(len(valid) * (1 - val_split))
    train_set = valid[:split_idx]
    val_set = valid[split_idx:]

    print(f"\nTrain: {len(train_set)}, Val: {len(val_set)}")

    # Create dataset directory structure
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split, data in [("train", train_set), ("val", val_set)]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i, (img_path, rank, suit) in enumerate(data):
            cls = label_to_class(rank, suit)
            cls_idx = CLASS_MAP[cls]

            # Copy image
            dst_img = img_dir / f"{i:04d}.jpg"
            shutil.copy2(img_path, dst_img)

            # For YOLO: each image has ONE card filling most of the frame
            # Since we're cropping zones, the card IS the image
            # Use a centered bounding box covering most of the image
            # Format: class_id x_center y_center width height (normalized 0-1)
            dst_lbl = lbl_dir / f"{i:04d}.txt"
            dst_lbl.write_text(f"{cls_idx} 0.5 0.5 0.8 0.8\n")

    # Write data.yaml
    yaml_content = f"""path: {DATASET_DIR.resolve()}
train: train/images
val: val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (DATASET_DIR / "data.yaml").write_text(yaml_content)

    print(f"\nDataset created at: {DATASET_DIR}")
    print(f"Config: {DATASET_DIR / 'data.yaml'}")
    print(f"\nTo train:")
    print(f"  pip install ultralytics")
    print(f"  python -c \"")
    print(f"from ultralytics import YOLO")
    print(f"model = YOLO('yolov8n.pt')")
    print(f"model.train(data='{DATASET_DIR / 'data.yaml'}', epochs=100, imgsz=640)\"")


def review_labels(pairs):
    """Interactive review of labels."""
    import cv2

    print(f"\nReviewing {len(pairs)} images...")
    print("For each image: Enter=keep, type new label=replace, 'd'=delete, 'q'=quit\n")

    changes = 0
    for img_path, label in pairs:
        parsed = parse_label(label)
        if not parsed:
            continue  # skip No card

        rank, suit = parsed
        display_label = f"{rank} of {suit}"

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Show image
        cv2.imshow("Review", img)
        print(f"  {img_path.name}: {display_label}  [Enter=ok, new label, d=delete, q=quit]", end=" ")

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            img_path.unlink()
            img_path.with_suffix(".txt").unlink(missing_ok=True)
            print("DELETED")
            changes += 1
        elif key == 13:  # Enter
            print("ok")
        else:
            cv2.destroyAllWindows()
            new_label = input(f"  New label (e.g. '4 of Clubs'): ").strip()
            if new_label:
                img_path.with_suffix(".txt").write_text(new_label)
                print(f"  Updated to: {new_label}")
                changes += 1

    cv2.destroyAllWindows()
    print(f"\n{changes} changes made")


if __name__ == "__main__":
    import sys

    pairs = scan_training_data()

    if "--review" in sys.argv:
        review_labels(pairs)
        pairs = scan_training_data()  # rescan after review

    prepare_dataset(pairs)

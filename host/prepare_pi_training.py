#!/usr/bin/env python3
"""
Prepare a YOLO training dataset from Pi scanner slot templates.

The Pi captures reference crops during /train into:
    <src_dir>/slot1/Ac.png
    <src_dir>/slot1/10d.png
    ...
    <src_dir>/slotN/...

Each PNG is a single-card image taken through the scanner box mirror
with the LEDs held on. This script walks those files, treats the whole
image as a YOLO bounding box for the card's rank/suit class, and writes
a stratified 80/20 train/val dataset under `--out`.

Usage (first sync the Pi's templates to the Mac):
    rsync -av srw@pokerbuddy.local:~/CardGameRemote/pi/card_recognition/reference/slot_templates/ \
          ~/pi_slot_templates/

    python3 host/prepare_pi_training.py \
        --src ~/pi_slot_templates \
        --out host/pi_yolo_dataset
"""

import argparse
import random
import re
import shutil
from pathlib import Path

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]

CLASSES = [f"{r}{s}" for r in RANKS for s in SUITS]  # 52
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}


def gather_templates(src: Path):
    """Walk src/slotN/*.png, return list of (path, class_code, slot_num)."""
    items = []
    for slot_dir in sorted(src.iterdir()):
        m = re.match(r"^slot(\d+)$", slot_dir.name)
        if not m or not slot_dir.is_dir():
            continue
        slot_num = int(m.group(1))
        for f in sorted(slot_dir.glob("*.png")):
            code = f.stem
            if code not in CLASS_MAP:
                continue
            items.append((f, code, slot_num))
    return items


def prepare(items, out_dir: Path, val_split: float):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    by_class = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)

    print(f"Classes seen: {len(by_class)} / 52")
    missing = sorted(set(CLASSES) - set(by_class.keys()))
    if missing:
        print(f"Missing classes: {missing}")
    counts = [len(v) for v in by_class.values()]
    print(f"Per-class counts: min={min(counts)}, max={max(counts)}, "
          f"total={sum(counts)}")

    train, val = [], []
    for cls in sorted(by_class.keys()):
        group = by_class[cls][:]
        random.shuffle(group)
        split_idx = max(1, int(len(group) * (1 - val_split)))
        train.extend(group[:split_idx])
        val.extend(group[split_idx:])
    random.shuffle(train)
    random.shuffle(val)
    print(f"Train: {len(train)}, Val: {len(val)}")

    for split_name, data in [("train", train), ("val", val)]:
        img_dir = out_dir / split_name / "images"
        lbl_dir = out_dir / split_name / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i, (src_path, code, slot_num) in enumerate(data):
            cls_idx = CLASS_MAP[code]
            dst_img = img_dir / f"{i:04d}_slot{slot_num}_{code}.png"
            shutil.copy2(src_path, dst_img)
            dst_lbl = lbl_dir / f"{i:04d}_slot{slot_num}_{code}.txt"
            # Card fills the whole crop; label covers ~90% centered so YOLO
            # gets a tiny bit of margin to learn the object boundary.
            dst_lbl.write_text(f"{cls_idx} 0.5 0.5 0.9 0.9\n")

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )

    print(f"\nDataset ready: {out_dir}")
    print(f"Config: {yaml_path}")
    print()
    print("Train example:")
    print(f"  python3 -c 'from ultralytics import YOLO; "
          f"YOLO(\"yolov8n.pt\").train(data=\"{yaml_path}\", "
          f"epochs=100, imgsz=320, batch=32, patience=25, "
          f"name=\"pi_card_detector\", project=\"pi_yolo_runs\", "
          f"device=\"mps\", exist_ok=True)'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path,
                   help="Directory containing slotN/*.png (e.g. rsynced from Pi)")
    p.add_argument("--out", default=Path(__file__).parent / "pi_yolo_dataset",
                   type=Path, help="Output YOLO dataset directory")
    p.add_argument("--val-split", default=0.2, type=float,
                   help="Fraction of images per class reserved for validation")
    p.add_argument("--seed", default=0, type=int)
    args = p.parse_args()

    random.seed(args.seed)
    items = gather_templates(args.src)
    if not items:
        raise SystemExit(f"No slot*/*.png files found under {args.src}")
    print(f"Found {len(items)} template images under {args.src}")
    prepare(items, args.out, args.val_split)


if __name__ == "__main__":
    main()

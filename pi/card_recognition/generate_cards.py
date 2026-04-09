"""
Generate synthetic playing card images for testing card recognition.

Creates all 52 cards as images that mimic real card layout:
- White card with rounded corners
- Rank and suit in top-left and bottom-right corners
- Large centered suit symbol
- Red for hearts/diamonds, black for clubs/spades

These are used to:
1. Train reference templates (run train.py after generating)
2. Test the recognition pipeline without real cards or a camera

Usage:
    python -m card_recognition.generate_cards --output samples/
    python -m card_recognition.generate_cards --output samples/ --size 400x560
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont


RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = {
    "h": {"name": "hearts", "symbol": "\u2665", "color": (220, 30, 30)},
    "d": {"name": "diamonds", "symbol": "\u2666", "color": (220, 30, 30)},
    "c": {"name": "clubs", "symbol": "\u2663", "color": (30, 30, 30)},
    "s": {"name": "spades", "symbol": "\u2660", "color": (30, 30, 30)},
}

# Card dimensions (pixels)
DEFAULT_WIDTH = 250
DEFAULT_HEIGHT = 350


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a clean font, fall back to default."""
    font_paths = [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # Raspberry Pi OS
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


def generate_card(rank: str, suit_key: str, width: int, height: int) -> Image.Image:
    """Generate a single playing card image."""
    suit = SUITS[suit_key]
    color = suit["color"]
    symbol = suit["symbol"]

    # Create white card
    card = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    # Draw border
    draw.rounded_rectangle(
        [1, 1, width - 2, height - 2],
        radius=12,
        outline=(180, 180, 180),
        width=2,
    )

    # Font sizes — sized to fill the corner extraction regions in detector.py
    # Rank region: y=5-55, x=2-48 (50px tall, 46px wide)
    # Suit region: y=55-95, x=5-45 (40px tall, 40px wide)
    rank_font_size = max(16, int(height * 0.12))
    suit_font_size = max(14, int(height * 0.09))
    center_font_size = max(20, int(height * 0.25))

    rank_font = get_font(rank_font_size)
    suit_font = get_font(suit_font_size)
    center_font = get_font(center_font_size)

    # Top-left corner: rank (positioned within rank extraction region y=5-55)
    rank_x = int(width * 0.03)
    rank_y = 8
    draw.text((rank_x, rank_y), rank, fill=color, font=rank_font)

    # Top-left corner: suit symbol (positioned within suit extraction region y=55-95)
    suit_x = int(width * 0.04)
    suit_y = 58
    draw.text((suit_x, suit_y), symbol, fill=color, font=suit_font)

    # Center: large suit symbol
    center_text = symbol
    bbox = draw.textbbox((0, 0), center_text, font=center_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    cx = (width - text_w) // 2
    cy = (height - text_h) // 2
    draw.text((cx, cy), center_text, fill=color, font=center_font)

    # Bottom-right corner: rank + suit (rotated 180)
    br_img = Image.new("RGBA", (60, 100), (255, 255, 255, 0))
    br_draw = ImageDraw.Draw(br_img)
    br_draw.text((8, 8), rank, fill=color, font=rank_font)
    br_draw.text((10, 58), symbol, fill=color, font=suit_font)
    br_img = br_img.rotate(180, expand=False)

    paste_x = width - br_img.width - int(width * 0.02)
    paste_y = height - br_img.height - int(height * 0.02)
    card.paste(br_img, (paste_x, paste_y), br_img)

    return card


def generate_all_cards(output_dir: str, width: int, height: int):
    """Generate all 52 playing cards."""
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for rank in RANKS:
        for suit_key in SUITS:
            card = generate_card(rank, suit_key, width, height)
            filename = f"{rank}{suit_key}.png"
            filepath = os.path.join(output_dir, filename)
            card.save(filepath)
            count += 1

    print(f"Generated {count} card images in {output_dir}/")
    print(f"Card size: {width}x{height} pixels")
    print(f"\nNext steps:")
    print(f"  1. Train reference templates:")
    print(f"     python -m card_recognition.train --samples {output_dir}/ --output reference/")
    print(f"  2. Test recognition:")
    print(f"     python -m card_recognition.test_detector --dir {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic playing card images")
    parser.add_argument(
        "--output", type=str, default="samples/",
        help="Output directory for card images (default: samples/)"
    )
    parser.add_argument(
        "--size", type=str, default=f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
        help=f"Card size in WxH pixels (default: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT})"
    )
    args = parser.parse_args()

    width, height = map(int, args.size.lower().split("x"))
    generate_all_cards(args.output, width, height)


if __name__ == "__main__":
    main()

"""
Test harness for card recognition.

Usage:
    # Test against a single image
    python -m card_recognition.test_detector --image samples/Kh.jpg

    # Test against all images in a directory (filenames must follow naming convention)
    python -m card_recognition.test_detector --dir samples/

    # Show visual debug output (extracted card, corner, rank, suit regions)
    python -m card_recognition.test_detector --image samples/Kh.jpg --debug

    # Test with a webcam or Pi camera (live mode)
    python -m card_recognition.test_detector --live
"""

import argparse
import os
import sys

import cv2
import numpy as np

from .detector import CardDetector, CARD_WIDTH, CARD_HEIGHT, CORNER_WIDTH, CORNER_HEIGHT
from .train import parse_card_filename


def test_single(detector: CardDetector, image_path: str, expected: tuple | None = None,
                debug: bool = False) -> bool:
    """
    Test card recognition on a single image.
    Returns True if the result matches expected (or if no expected value).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not read {image_path}")
        return False

    result = detector.identify(image)

    if result is None:
        status = "FAIL" if expected else "WARN"
        print(f"  [{status}] {os.path.basename(image_path)}: No card detected")
        return expected is None

    if expected:
        exp_rank, exp_suit = expected
        match = (result.rank == exp_rank and result.suit == exp_suit)
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {os.path.basename(image_path)}: "
              f"detected={result}, expected={exp_rank} of {exp_suit}")
        return match
    else:
        print(f"  [INFO] {os.path.basename(image_path)}: detected={result}")
        return True


def test_directory(detector: CardDetector, dir_path: str, debug: bool = False):
    """Test all card images in a directory. Filenames must follow naming convention."""
    files = sorted(f for f in os.listdir(dir_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))

    if not files:
        print(f"No image files found in {dir_path}")
        return

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for filename in files:
        filepath = os.path.join(dir_path, filename)
        parsed = parse_card_filename(filename)

        if parsed:
            total += 1
            if test_single(detector, filepath, expected=parsed, debug=debug):
                passed += 1
            else:
                failed += 1
        else:
            skipped += 1
            test_single(detector, filepath, expected=None, debug=debug)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    if total > 0:
        print(f"Accuracy: {passed/total:.0%}")


def test_live(detector: CardDetector):
    """Live camera test mode. Press 'q' to quit, 's' to scan current frame."""
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        camera.configure(camera.create_preview_configuration(
            main={"size": (640, 480)}
        ))
        camera.start()
        use_picamera = True
        print("Using Pi Camera")
    except (ImportError, RuntimeError):
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("No camera available")
            return
        use_picamera = False
        print("Using USB/built-in camera")

    print("Press 's' to scan, 'q' to quit")

    try:
        while True:
            if use_picamera:
                frame = camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = camera.read()
                if not ret:
                    break

            # Show live preview
            display = frame.copy()
            cv2.putText(display, "Press 's' to scan, 'q' to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Card Scanner - Live", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                result = detector.identify(frame)
                if result:
                    print(f"  Detected: {result}")
                    # Show result on frame briefly
                    cv2.putText(display, f"Detected: {result}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.imshow("Card Scanner - Live", display)
                    cv2.waitKey(1500)
                else:
                    print("  No card detected")
    finally:
        if use_picamera:
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()


def show_debug(detector: CardDetector, image_path: str):
    """Show visual debug output of the detection pipeline stages."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        return

    # Stage 1: Extract card
    card_img = detector._extract_card(image)
    if card_img is None:
        print("Could not extract card from image")
        return

    # Stage 2: Extract corner (try both orientations)
    best_rotation = 0
    best_score = -1
    for rotation in [0, 180]:
        if rotation == 180:
            rotated = cv2.rotate(card_img, cv2.ROTATE_180)
        else:
            rotated = card_img
        corner = detector._extract_corner(rotated)
        if corner is not None:
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            score = np.sum(gray < 128)
            if score > best_score:
                best_score = score
                best_rotation = rotation

    if best_rotation == 180:
        card_img = cv2.rotate(card_img, cv2.ROTATE_180)

    corner = detector._extract_corner(card_img)
    rank_img = detector._extract_rank_region(corner)
    suit_img = detector._extract_suit_region(corner)

    # Create debug display
    debug_img = np.zeros((400, 800, 3), dtype=np.uint8)

    # Original image (resized)
    orig_resized = cv2.resize(image, (200, 280))
    debug_img[10:290, 10:210] = orig_resized
    cv2.putText(debug_img, "Original", (10, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Extracted card
    card_resized = cv2.resize(card_img, (150, 210))
    debug_img[10:220, 230:380] = card_resized
    cv2.putText(debug_img, "Extracted Card", (230, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Corner
    corner_resized = cv2.resize(corner, (100, 200))
    debug_img[10:210, 400:500] = corner_resized
    cv2.putText(debug_img, "Corner", (400, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Rank region (grayscale → BGR for display)
    rank_display = cv2.cvtColor(cv2.resize(rank_img, (100, 100)), cv2.COLOR_GRAY2BGR)
    debug_img[10:110, 520:620] = rank_display
    cv2.putText(debug_img, "Rank", (520, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Suit region
    suit_display = cv2.cvtColor(cv2.resize(suit_img, (100, 100)), cv2.COLOR_GRAY2BGR)
    debug_img[140:240, 520:620] = suit_display
    cv2.putText(debug_img, "Suit", (520, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Recognition result
    result = detector.identify(image)
    result_text = str(result) if result else "Not recognized"
    cv2.putText(debug_img, result_text, (10, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Debug - Detection Pipeline", debug_img)
    print(f"Result: {result_text}")
    print("Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test card recognition")
    parser.add_argument("--image", type=str, help="Single image to test")
    parser.add_argument("--dir", type=str, help="Directory of images to test")
    parser.add_argument("--live", action="store_true", help="Live camera test")
    parser.add_argument("--debug", action="store_true", help="Show visual debug output")
    parser.add_argument(
        "--reference", type=str, default="reference/",
        help="Reference templates directory (default: reference/)"
    )
    args = parser.parse_args()

    if not any([args.image, args.dir, args.live]):
        parser.print_help()
        sys.exit(1)

    detector = CardDetector(args.reference)

    if args.image:
        if args.debug:
            show_debug(detector, args.image)
        else:
            parsed = parse_card_filename(os.path.basename(args.image))
            test_single(detector, args.image, expected=parsed)
    elif args.dir:
        test_directory(detector, args.dir, debug=args.debug)
    elif args.live:
        test_live(detector)


if __name__ == "__main__":
    main()

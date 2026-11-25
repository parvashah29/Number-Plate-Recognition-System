"""Number Plate Recognition CLI: images, folders, videos or webcam."""
import argparse
import sys
from pathlib import Path

from license_plate_detection import IMAGE_EXTS, VIDEO_EXTS, LicensePlateDetector, default_weights


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Number Plate Recognition (YOLO detection + EasyOCR/Tesseract)")
    p.add_argument("--source", default="0",
                   help="image, folder of images/videos, video file, or webcam index (default: 0)")
    p.add_argument("--weights", default=None, help=f"YOLO weights (default: {default_weights()})")
    p.add_argument("--ocr", choices=["auto", "easyocr", "tesseract"], default="auto")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    p.add_argument("--stride", type=int, default=3, help="process every Nth video frame")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames")
    p.add_argument("--no-display", action="store_true", help="do not open a preview window")
    p.add_argument("--output-dir", default="output_results")
    p.add_argument("--debug", action="store_true", help="dump OCR input crops to debug_plates/")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    src = args.source
    if not src.isdigit() and not Path(src).exists():
        print(f"Error: source not found: {src}", file=sys.stderr)
        return 2
    try:
        det = LicensePlateDetector(args.weights, ocr=args.ocr, device=args.device, conf=args.conf,
                                   debug_dir="debug_plates" if args.debug else None)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    display = not args.no_display
    files = [Path(src)] if not src.isdigit() and Path(src).is_file() else []
    if not src.isdigit() and Path(src).is_dir():
        files = sorted(f for f in Path(src).iterdir() if f.suffix.lower() in IMAGE_EXTS + VIDEO_EXTS)
        if not files:
            print(f"Error: no images or videos in {src}", file=sys.stderr)
            return 2
    try:
        if src.isdigit():
            det.process_video(src, args.output_dir, display=display, stride=args.stride, max_frames=args.max_frames)
        for f in files:
            if f.suffix.lower() in IMAGE_EXTS:
                det.process_image(str(f), args.output_dir)
            else:
                det.process_video(str(f), args.output_dir, display=display, stride=args.stride,
                                  max_frames=args.max_frames)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

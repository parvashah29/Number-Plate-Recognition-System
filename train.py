"""Fine-tune YOLOv8n on the licence-plate dataset (download it into ./archive first, see README)."""
import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the plate detector")
    ap.add_argument("--data-root", default=str(ROOT / "archive"), help="folder with images/{train,val}, labels/{train,val}")
    ap.add_argument("--base", default=str(ROOT / "yolov8n.pt"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: ultralytics auto)")
    args = ap.parse_args(argv)

    root = Path(args.data_root).resolve()
    if not (root / "images" / "train").is_dir() or not (root / "labels" / "train").is_dir():
        print(f"Error: dataset not found at {root} (expected images/train, images/val, labels/...).\n"
              "See the Dataset section of the README.", file=sys.stderr)
        return 2
    from ultralytics import YOLO
    # write a yaml with an absolute path so training works from any directory
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['license_plate']\n")
    model = YOLO(args.base)
    model.train(data=f.name, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                name="license_plate_detector", optimizer="AdamW", patience=20, exist_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Evaluate the pipeline on the validation split.

Detection: precision/recall at IoU 0.5 against the YOLO label files.
OCR: fraction of detected plates that produce a valid plate string ("read rate"). The Kaggle dataset has
box labels only; if you have plate text ground truth, pass --gt CSV (columns: filename,text) to also get
exact-match and character-level accuracy.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from license_plate_detection import IMAGE_EXTS, LicensePlateDetector, edit_distance

ROOT = Path(__file__).resolve().parent


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def load_labels(path: Path, w: int, h: int):
    boxes = []
    if path.is_file():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(ROOT / "archive"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=50, help="max images (0 = all)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--ocr", default="auto")
    ap.add_argument("--gt", default=None, help="optional CSV filename,text with plate ground truth")
    ap.add_argument("--json", default=None, help="write results to this JSON file")
    args = ap.parse_args(argv)

    root = Path(args.data_root)
    img_dir, lab_dir = root / "images" / args.split, root / "labels" / args.split
    if not img_dir.is_dir():
        print(f"Error: {img_dir} not found. See the Dataset section of the README.", file=sys.stderr)
        return 2
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        images = images[:args.limit]
    gt = {}
    if args.gt:
        with open(args.gt) as f:
            gt = {r["filename"]: r["text"].upper() for r in csv.DictReader(f)}

    det = LicensePlateDetector(args.weights, ocr=args.ocr, save_dir=None, verbose=False)
    tp = fp = fn = read = n_det = 0
    exact = chars_ok = chars_total = gt_n = 0
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        truth = load_labels(lab_dir / (p.stem + ".txt"), w, h)
        preds = det.detect(img)
        matched = set()
        for x1, y1, x2, y2, conf in preds:
            best = max(range(len(truth)), key=lambda i: iou((x1, y1, x2, y2), truth[i]), default=None)
            if best is not None and best not in matched and iou((x1, y1, x2, y2), truth[best]) >= 0.5:
                matched.add(best)
                tp += 1
            else:
                fp += 1
            crop = img[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10]
            text, _ = det.read_plate(crop)
            n_det += 1
            read += bool(text)
            if p.name in gt and best is not None and iou((x1, y1, x2, y2), truth[best]) >= 0.5:
                gt_n += 1
                exact += text == gt[p.name]
                chars_total += len(gt[p.name])
                chars_ok += max(0, len(gt[p.name]) - edit_distance(text, gt[p.name]))
        fn += len(truth) - len(matched)
    res = {"images": len(images), "precision@0.5": tp / (tp + fp) if tp + fp else 0.0,
           "recall@0.5": tp / (tp + fn) if tp + fn else 0.0, "detections": n_det,
           "ocr_read_rate": read / n_det if n_det else 0.0}
    if gt_n:
        res.update({"ocr_exact_match": exact / gt_n, "ocr_char_accuracy": chars_ok / chars_total, "gt_plates": gt_n})
    for k, v in res.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

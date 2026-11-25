"""Licence plate detection (YOLO) + OCR (EasyOCR by default, Tesseract optional)."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform

ROOT = Path(__file__).resolve().parent
BUNDLED_WEIGHTS = ROOT / "models" / "plate_best.pt"
LEGACY_WEIGHTS = ROOT / "runs" / "detects" / "license_plate_detector" / "weights" / "best.pt"
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# Indian plate layout, e.g. KA01AB1234 (used only to repair O/0, I/1 style confusions)
_INDIAN = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$")
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


def default_weights() -> Path:
    """Bundled weights first, then the legacy training output."""
    for p in (BUNDLED_WEIGHTS, LEGACY_WEIGHTS):
        if p.is_file():
            return p
    return BUNDLED_WEIGHTS


def select_device(preferred: str = "auto") -> str:
    import torch
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_plate_text(text: str) -> str:
    """Normalise OCR text; return '' when it cannot be a plate (4-10 chars, letters and digits)."""
    text = "".join(c for c in (text or "") if c.isalnum() and c.isascii()).upper()
    if not 4 <= len(text) <= 10:
        return ""
    if not (any(c.isalpha() for c in text) and any(c.isdigit() for c in text)):
        return ""
    return repair_indian_format(text)


def repair_indian_format(text: str) -> str:
    """If text is one or two character confusions away from an Indian plate layout, fix it."""
    if _INDIAN.match(text) or not 9 <= len(text) <= 10:
        return text
    n_mid = len(text) - 8  # series letters: 1-2 when total is 9-10 (pattern: 2 + 2 + n + 4)
    layout = "A" * 2 + "D" * 2 + "A" * n_mid + "D" * 4
    out = []
    for ch, kind in zip(text, layout):
        if kind == "D" and ch.isalpha():
            ch = _TO_DIGIT.get(ch, ch)
        elif kind == "A" and ch.isdigit():
            ch = _TO_ALPHA.get(ch, ch)
        out.append(ch)
    fixed = "".join(out)
    return fixed if _INDIAN.match(fixed) else text


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sharpness(img: np.ndarray) -> float:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class PlateTracker:
    """Collapses repeated reads of the same plate (exact or within 1 edit) into one entry."""

    def __init__(self, max_distance: int = 1):
        self.max_distance = max_distance
        self.plates: dict = {}  # text -> best confidence

    def update(self, text: str, conf: float) -> bool:
        """Return True if this is a plate not seen before (so it should be logged)."""
        for known in list(self.plates):
            if edit_distance(known, text) <= self.max_distance and len(known) == len(text):
                if conf > self.plates[known]:
                    self.plates[known] = conf
                return False
        self.plates[text] = conf
        return True


class LicensePlateDetector:
    """Detect plates with a YOLO model and read them with EasyOCR and/or Tesseract.

    Args:
        model_path: YOLO weights. Default: models/plate_best.pt (or the legacy runs/ path).
        ocr: 'auto' (EasyOCR, falls back to Tesseract), 'easyocr', 'tesseract'.
        device: 'auto', 'cpu', 'mps' or 'cuda'.
        conf: YOLO confidence threshold.
        save_dir: where unique plate crops are written (None disables saving).
        debug_dir: where OCR inputs are dumped (None disables).
    """

    def __init__(self, model_path: Optional[str] = None, ocr: str = "auto", device: str = "auto",
                 conf: float = 0.35, save_dir: Optional[str] = "detected_plates",
                 debug_dir: Optional[str] = None, verbose: bool = True):
        from ultralytics import YOLO

        path = Path(model_path) if model_path else default_weights()
        if not path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {path}\n"
                "Use --weights PATH, restore models/plate_best.pt (git pull), or train one with train.py.")
        self.device = select_device(device)
        self.conf = conf
        self.verbose = verbose
        self.save_dir = Path(save_dir) if save_dir else None
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.model = YOLO(str(path))
        self._log(f"Using device: {self.device}; weights: {path}")

        self.use_tesseract = False
        self.reader = None
        self._init_ocr(ocr)
        self.unique_plate_hashes: set = set()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ---- OCR engines -------------------------------------------------------------------
    def _init_ocr(self, ocr: str) -> None:
        if ocr not in ("auto", "easyocr", "tesseract"):
            raise ValueError("ocr must be auto, easyocr or tesseract")
        if ocr in ("auto", "tesseract") and shutil.which("tesseract"):
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
                self.use_tesseract = ocr == "tesseract"  # in 'auto' EasyOCR is primary
                self._pytesseract = pytesseract
                self._tesseract_available = True
            except ImportError:
                self._tesseract_available = False
        else:
            self._tesseract_available = False
        if ocr == "tesseract" and not self._tesseract_available:
            raise RuntimeError("Tesseract requested but not found. Install it (brew install tesseract) "
                               "and pip install pytesseract, or use --ocr easyocr.")
        if ocr != "tesseract" or not self._tesseract_available:
            import easyocr
            self._log("Initialising EasyOCR (models download on first use)...")
            gpu = self.device in ("cuda", "mps")
            try:
                self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            except Exception:
                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def _ocr_easyocr(self, images: List[np.ndarray]) -> Tuple[str, float]:
        best, best_score = ("", 0.0), 0.0
        for img in images:
            try:
                results = self.reader.readtext(img, allowlist=ALLOWLIST, detail=1, paragraph=False,
                                               decoder="beamsearch", beamWidth=5)
            except Exception as e:  # MPS kernels can fail; retry on CPU once
                if self.device == "mps":
                    import easyocr
                    self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                    self.device = "cpu"
                    return self._ocr_easyocr(images)
                self._log(f"EasyOCR error: {e}")
                continue
            if not results:
                continue
            # plates are often split into several boxes and may include an "IND" badge: build candidates
            # from the joined boxes (badge removed) and from each single box, prefer valid Indian layouts
            results = sorted(results, key=lambda r: r[0][0][0])
            body = [r for r in results if r[1] not in ("IND", "ND", "ID")]
            cands = []
            if body:
                cands.append(("".join(r[1] for r in body), float(np.mean([r[2] for r in body]))))
            cands += [(r[1], float(r[2])) for r in body]
            for raw, conf in cands:
                text = validate_plate_text(raw)
                if not text:
                    continue
                score = conf + (1.0 if _INDIAN.match(text) else 0.0)
                if score > best_score:
                    best, best_score = (text, conf), score
        return best

    def _ocr_tesseract(self, images: List[np.ndarray]) -> Tuple[str, float]:
        best = ("", 0.0)
        for img in images:
            for psm in (7, 8, 6):
                try:
                    data = self._pytesseract.image_to_data(
                        img, output_type=self._pytesseract.Output.DICT,
                        config=f"--psm {psm} --oem 3 -c tessedit_char_whitelist={ALLOWLIST}")
                except Exception as e:
                    self._log(f"Tesseract error: {e}")
                    return best
                pairs = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
                if not pairs:
                    continue
                text = validate_plate_text("".join(w for w, _ in pairs))
                conf = float(np.mean([c for _, c in pairs])) / 100.0
                if text and conf > best[1]:
                    best = (text, conf)
        return best

    # ---- image helpers ------------------------------------------------------------------
    @staticmethod
    def compute_image_hash(image: np.ndarray) -> str:
        gray = cv2.cvtColor(cv2.resize(image, (100, 50)), cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray.tobytes()).hexdigest()

    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Grayscale, optional perspective fix, upscale, CLAHE contrast boost, white border."""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if plate_img.ndim == 3 else plate_img
        edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)
        cnts = sorted(imutils.grab_contours(cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)),
                      key=cv2.contourArea, reverse=True)[:5]
        warped = gray
        for c in cnts:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            # only trust a quad covering most of the crop; otherwise we would crop away characters
            if len(approx) == 4 and cv2.contourArea(approx) > 0.6 * gray.shape[0] * gray.shape[1]:
                warped = four_point_transform(gray, approx.reshape(4, 2))
                break
        warped = cv2.resize(warped, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        warped = cv2.GaussianBlur(warped, (3, 3), 0)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(warped)
        return cv2.copyMakeBorder(enhanced, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    def validate_plate_text(self, text: str) -> str:
        return validate_plate_text(text)

    def read_plate(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """OCR a cropped plate. Returns ('', 0.0) if unreadable."""
        if plate_img is None or plate_img.size == 0 or min(plate_img.shape[:2]) < 8:
            return "", 0.0
        if plate_img.shape[1] < 150:
            s = 150 / plate_img.shape[1]
            plate_img = cv2.resize(plate_img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        variants = [plate_img, self.preprocess_plate(plate_img)]
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = f"{time.time_ns()}"
            cv2.imwrite(str(self.debug_dir / f"{stamp}_processed.png"), variants[1])
            cv2.imwrite(str(self.debug_dir / f"{stamp}_raw.png"), plate_img)
        if self.reader is not None and not self.use_tesseract:
            text, conf = self._ocr_easyocr(variants)
            if not text and self._tesseract_available:
                text, conf = self._ocr_tesseract(variants[1:])
            return text, conf
        text, conf = self._ocr_tesseract(variants)
        if not text and self.reader is not None:
            text, conf = self._ocr_easyocr(variants)
        return text, conf

    def save_unique_plate(self, plate_img: np.ndarray, text: str, confidence: float, idx: int = 0) -> bool:
        if self.save_dir is None:
            return False
        h = self.compute_image_hash(plate_img)
        if h in self.unique_plate_hashes:
            return False
        self.unique_plate_hashes.add(h)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns() % 10**6:06d}"
        name = f"plate_{text}_{confidence:.2f}_{stamp}.png" if text else f"plate_unknown_{stamp}_{idx}.png"
        cv2.imwrite(str(self.save_dir / name), plate_img)
        return True

    # ---- detection ----------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        res = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            out.append((x1, y1, x2, y2, float(box.conf[0])))
        return out

    def process_frame(self, frame: np.ndarray, annotate: bool = True) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """Detect and read plates in a BGR frame. Returns (annotated copy, [(text, det_confidence)])."""
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame")
        out = frame.copy() if annotate else frame
        plate_texts: List[Tuple[str, float]] = []
        H, W = frame.shape[:2]
        for idx, (x1, y1, x2, y2, conf) in enumerate(self.detect(frame)):
            pad = 10
            crop = frame[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
            text, ocr_conf = self.read_plate(crop)
            if text:
                plate_texts.append((text, conf))
            if crop.size:
                self.save_unique_plate(crop, text, conf, idx)
            if annotate:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{text} ({conf:.2f})" if text else f"plate? {conf:.2f}"
                scale = max(0.5, min(1.2, W / 1400))
                tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
                top = max(th + 8, y1)
                cv2.rectangle(out, (x1, top - th - 8), (x1 + tw, top), (0, 255, 0), -1)
                cv2.putText(out, label, (x1, top - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
        return out, plate_texts

    def process_image(self, path: str, output_dir: str = "output_results") -> List[Tuple[str, float]]:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        out, plates = self.process_frame(img)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        dest = Path(output_dir) / f"{Path(path).stem}_processed.jpg"
        cv2.imwrite(str(dest), out)
        self._log(f"{path}: {[p[0] for p in plates] or 'no plate read'} -> {dest}")
        return plates

    def process_video(self, source: str = "0", output_dir: str = "output_results", log_dir: str = "plate_logs",
                      display: Optional[bool] = None, stride: int = 3, max_frames: Optional[int] = None,
                      target_size: Tuple[int, int] = (1280, 720)) -> dict:
        """Process a video file or webcam ('0'). Returns a summary dict (unique plates, frames...)."""
        is_cam = str(source).isdigit()
        if not is_cam and not Path(source).is_file():
            raise FileNotFoundError(f"Video not found: {source}")
        cap = cv2.VideoCapture(int(source) if is_cam else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source} (unsupported codec or no camera access?)")
        if display is None:
            display = True
        tw, th = target_size

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / f"detected_plates_{time.strftime('%Y%m%d-%H%M%S')}.txt"
        writer = None
        out_path = None
        if not is_cam:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(output_dir) / f"{Path(source).stem}_processed.mp4"
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps >= 1 else 30.0
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, stride), (tw, th))

        tracker = PlateTracker()
        n_read = n_proc = frames_with_plate = 0
        with open(log_path, "w") as log:
            log.write(f"License Plate Detection Log\nSource: {source}\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                      + "-" * 50 + "\n")
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                n_read += 1
                if n_read % max(1, stride) != 0:
                    continue
                h, w = frame.shape[:2]
                s = min(tw / w, th / h)
                nw, nh = max(1, int(w * s)), max(1, int(h * s))
                canvas = np.zeros((th, tw, 3), np.uint8)
                yo, xo = (th - nh) // 2, (tw - nw) // 2
                canvas[yo:yo + nh, xo:xo + nw] = cv2.resize(frame, (nw, nh))
                shown, plates = self.process_frame(canvas)
                n_proc += 1
                frames_with_plate += bool(plates)
                if writer is not None:
                    writer.write(shown)
                for text, conf in plates:
                    if tracker.update(text, conf):
                        entry = f"Frame {n_read} | Plate: {text} | Confidence: {conf:.2f}"
                        log.write(entry + "\n")
                        log.flush()
                        self._log(entry)
                if display:
                    try:
                        cv2.imshow("License Plate Detection", shown)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        self._log("No display available (headless OpenCV): continuing without a window.")
                        display = False
                if max_frames and n_proc >= max_frames:
                    break
            log.write("\n" + "=" * 50 + f"\nUnique plates: {len(tracker.plates)}\n")
            for p, c in sorted(tracker.plates.items()):
                log.write(f"- {p} (best conf {c:.2f})\n")
        cap.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if n_read == 0:
            self._log(f"Warning: no frames could be read from {source}")
        self._log(f"Done: {n_proc} frames processed, {len(tracker.plates)} unique plates. Log: {log_path}")
        return {"frames_read": n_read, "frames_processed": n_proc, "frames_with_plate": frames_with_plate,
                "plates": dict(tracker.plates), "log": str(log_path), "video": str(out_path) if out_path else None}

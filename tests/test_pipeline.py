"""Smoke tests that need the bundled weights (skipped when absent) and EasyOCR."""
from pathlib import Path

import cv2
import numpy as np
import pytest

import license_plate_detection as lpd

pytestmark = pytest.mark.skipif(not lpd.default_weights().is_file(), reason="model weights not present")
SAMPLE = Path(__file__).resolve().parents[1] / "assets" / "sample_input.jpg"


@pytest.fixture(scope="module")
def detector():
    return lpd.LicensePlateDetector(save_dir=None, verbose=False)


def test_blank_frame_no_plates(detector):
    out, plates = detector.process_frame(np.zeros((720, 1280, 3), np.uint8))
    assert out.shape == (720, 1280, 3) and plates == []


def test_unreadable_crops_return_empty(detector):
    assert detector.read_plate(np.zeros((0, 0, 3), np.uint8)) == ("", 0.0)
    assert detector.read_plate(np.zeros((4, 4, 3), np.uint8)) == ("", 0.0)


def test_preprocess_shape(detector):
    out = detector.preprocess_plate(np.full((100, 200, 3), 128, np.uint8))
    assert out.ndim == 2 and out.shape[0] > 100


@pytest.mark.skipif(not SAMPLE.is_file(), reason="sample image absent")
def test_sample_image_detected(detector):
    img = cv2.imread(str(SAMPLE))
    assert len(detector.detect(img)) >= 1


def test_missing_weights_message():
    with pytest.raises(FileNotFoundError, match="weights"):
        lpd.LicensePlateDetector("/nonexistent/best.pt")

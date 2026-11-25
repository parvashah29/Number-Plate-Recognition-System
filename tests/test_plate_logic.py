"""Pure-logic tests: no model weights or OCR models needed."""
import numpy as np
import pytest

from license_plate_detection import (PlateTracker, edit_distance, repair_indian_format, sharpness,
                                     validate_plate_text)


@pytest.mark.parametrize("raw,expected", [
    ("ABC123", "ABC123"),
    ("ka 01-ab 1234", "KA01AB1234"),
    ("ABC", ""),              # too short
    ("ABCDEFGH", ""),         # letters only
    ("12345678", ""),         # digits only
    ("A1B2C3D4E5F6", ""),     # too long
    ("", ""),
    (None, ""),
])
def test_validate(raw, expected):
    assert validate_plate_text(raw) == expected


def test_indian_repair():
    assert repair_indian_format("KAO1AB123I") == "KA01AB1231"
    assert repair_indian_format("ABC123") == "ABC123"  # non-matching layouts untouched


def test_edit_distance():
    assert edit_distance("KA01AB1234", "KA01AB1234") == 0
    assert edit_distance("KA01AB1234", "KA01AB1284") == 1


def test_tracker_dedupes_near_duplicates():
    t = PlateTracker()
    assert t.update("KA01AB1234", 0.5) is True
    assert t.update("KA01AB1234", 0.9) is False
    assert t.update("KA01AB1284", 0.6) is False   # one-character OCR flicker
    assert t.plates["KA01AB1234"] == 0.9
    assert t.update("MH12XY9999", 0.7) is True
    assert len(t.plates) == 2


def test_sharpness_blur_lower_than_sharp():
    rng = np.random.default_rng(0)
    sharp = (rng.random((50, 100)) * 255).astype(np.uint8)
    import cv2
    assert sharpness(cv2.GaussianBlur(sharp, (9, 9), 0)) < sharpness(sharp)

import unittest
import cv2
import numpy as np
from license_plate_detection import LicensePlateDetector

class TestLicensePlateDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LicensePlateDetector()

    def test_preprocess_plate(self):
        # Create a sample plate image
        plate_img = np.zeros((100, 200, 3), dtype=np.uint8)
        # Add some noise to simulate a real plate
        plate_img[50:70, 50:150] = 255
        # Preprocess the plate
        processed_img = self.detector.preprocess_plate(plate_img)
        # Check if the processed image is not None and has the correct shape
        self.assertIsNotNone(processed_img)
        self.assertEqual(processed_img.shape, (100, 200))

    def test_validate_plate_text(self):
        # Test with valid text
        valid_text = "ABC123"
        self.assertEqual(self.detector.validate_plate_text(valid_text), "ABC123")
        # Test with invalid text
        invalid_text = "A1B2C3"
        self.assertEqual(self.detector.validate_plate_text(invalid_text), "A1B2C3")
        # Test with text that is too short
        short_text = "ABC"
        self.assertEqual(self.detector.validate_plate_text(short_text), "")

    def test_process_frame(self):
        # Create a sample frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Process the frame
        processed_frame, plates = self.detector.process_frame(frame)
        # Check if the processed frame is not None and has the correct shape
        self.assertIsNotNone(processed_frame)
        self.assertEqual(processed_frame.shape, (720, 1280, 3))
        # Check if the plates list is a list
        self.assertIsInstance(plates, list)

if __name__ == '__main__':
    unittest.main() 
import os
import cv2
import numpy as np
from license_plate_detection import LicensePlateDetector

def create_dummy_plate():
    """Create a dummy license plate image for testing"""
    # Create a white background
    img = np.zeros((200, 600, 3), dtype=np.uint8)
    img.fill(255)
    
    # Add a border
    cv2.rectangle(img, (10, 10), (590, 190), (0, 0, 0), 5)
    
    # Add text "ABC1234"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'ABC1234', (100, 130), font, 4, (0, 0, 0), 10, cv2.LINE_AA)
    
    # Save it
    os.makedirs('Testing Videos', exist_ok=True)
    cv2.imwrite('Testing Videos/test_plate.jpg', img)
    return 'Testing Videos/test_plate.jpg'

def test_system():
    print("1. Initializing Detector...")
    try:
        detector = LicensePlateDetector()
        print("   Success: Detector initialized")
    except Exception as e:
        print(f"   Failed: {str(e)}")
        return

    print("\n2. Creating Test Image...")
    img_path = create_dummy_plate()
    print(f"   Created {img_path}")

    print("\n3. Testing Detection on Image...")
    try:
        # We need to mock the video process or just call process_frame directly
        # Let's load the image and call process_frame
        frame = cv2.imread(img_path)
        processed_frame, plates = detector.process_frame(frame)
        
        print(f"   Processed frame shape: {processed_frame.shape}")
        if plates:
            print(f"   Success: Detected plates: {plates}")
        else:
            print("   Note: No plates detected (this might be expected if YOLO model doesn't recognize the drawn plate, but code ran)")
            
    except Exception as e:
        print(f"   Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()

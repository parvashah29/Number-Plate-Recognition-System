import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from typing import List, Tuple
import torch
import os
import time
import hashlib
import shutil
import easyocr
from imutils.perspective import four_point_transform
import imutils

class LicensePlateDetector:
    """
    A class to detect and recognize license plates in images and videos using YOLO and Tesseract OCR.
    
    This class handles the detection of license plates using a YOLO model, preprocessing of images,
    OCR using Tesseract, and saving/logging of detected plates.
    """
    
    def __init__(self, model_path: str = 'runs/detects/license_plate_detector/weights/best.pt'):
        """
        Initialize the license plate detector with YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model weights file.
        """
        self.model = YOLO(model_path)
        
        # Set path for Tesseract
        tesseract_cmd = shutil.which('tesseract')
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            self.use_tesseract = True
        else:
            print("Warning: Tesseract not found. Please install it (brew install tesseract).")
            self.use_tesseract = False
            
        # Check for MPS (Mac GPU) availability
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize EasyOCR reader
        print("Initializing EasyOCR...")
        # EasyOCR uses torch device detection, so gpu=True should use MPS if available
        self.reader = easyocr.Reader(['en'], gpu=True)
        
        # Create directories for logs and detected plates
        os.makedirs('plate_logs', exist_ok=True)
        os.makedirs('detected_plates', exist_ok=True)
        
        # Initialize set to track unique plates
        self.unique_plate_hashes = set()
        
    def compute_image_hash(self, image: np.ndarray) -> str:
        """
        Compute a hash of the image content for duplicate detection.
        
        Args:
            image (np.ndarray): The image to hash.
            
        Returns:
            str: The hash of the image.
        """
        # Resize image to a standard size to ensure consistent hashing
        resized = cv2.resize(image, (100, 50))
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Compute hash
        return hashlib.md5(gray.tobytes()).hexdigest()
        
    def save_unique_plate(self, plate_img: np.ndarray, text: str, confidence: float, idx: int) -> bool:
        """
        Save plate image if it's unique based on content hash and text.
        
        Args:
            plate_img (np.ndarray): The image of the detected plate.
            text (str): The recognized text from the plate.
            confidence (float): The confidence score of the detection.
            idx (int): The index of the detection.
            
        Returns:
            bool: True if the plate was saved, False otherwise.
        """
        img_hash = self.compute_image_hash(plate_img)
        
        if img_hash not in self.unique_plate_hashes:
            self.unique_plate_hashes.add(img_hash)
            
            # Create filename with text if available
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            if text:
                filename = f"detected_plates/plate_{text}_{confidence:.2f}_{timestamp}.png"
            else:
                filename = f"detected_plates/plate_unknown_{timestamp}_{idx}.png"
            
            # Save both original and preprocessed versions
            cv2.imwrite(filename, plate_img)
            
            # Save preprocessed version with _processed suffix
            thresh = self.preprocess_plate(plate_img)
            processed_filename = filename.replace('.png', '_processed.png')
            cv2.imwrite(processed_filename, thresh)
            
            return True
        return False

    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess the license plate image for better OCR.
        Includes deskewing and enhancement.
        """
        # 1. Deskewing attempt
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Find contours to detect the actual plate polygon within the crop
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        display_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                display_cnt = approx
                break
        
        # If we found a 4-point contour, warp it
        if display_cnt is not None:
            warped = four_point_transform(gray, display_cnt.reshape(4, 2))
        else:
            warped = gray
            
        # 2. Enhancement
        # Resize to make text bigger (helps OCR)
        scale = 3.0
        warped = cv2.resize(warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(warped, None, 30, 7, 21)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Thresholding can sometimes lose details, so we return the enhanced grayscale
        # Tesseract and EasyOCR handle internal binarization well
        
        # Add a white border to give OCR more breathing room
        enhanced = cv2.copyMakeBorder(enhanced, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        return enhanced

    def validate_plate_text(self, text: str) -> str:
        """
        Validate and clean the detected license plate text.
        
        Args:
            text (str): The text to validate.
            
        Returns:
            str: The validated and cleaned text.
        """
        # Remove any non-alphanumeric characters
        text = ''.join(c for c in text if c.isalnum())
        
        # Convert to uppercase
        text = text.upper()
        
        # Check minimum length and format
        if len(text) < 4 or len(text) > 10:
            return ""
            
        # Heuristic: Plates usually have a mix of letters and numbers
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        if not (has_letters and has_numbers):
            return ""
            
        return text
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Process a single frame to detect and recognize license plates.
        
        Args:
            frame (np.ndarray): The input frame/image in BGR format.
            
        Returns:
            Tuple[np.ndarray, List[str]]: The processed frame with annotations and a list of recognized plate texts.
        """
        # Detect license plates with higher confidence threshold
        results = self.model(frame, conf=0.35)[0]
        plate_texts = []
        
        # Process each detection
        for idx, box in enumerate(results.boxes):
            # Get coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Draw bounding box always
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = None
            
            # Crop the license plate region with padding
            padding = 10
            y1_pad = max(0, y1 - padding)
            y2_pad = min(frame.shape[0], y2 + padding)
            x1_pad = max(0, x1 - padding)
            x2_pad = min(frame.shape[1], x2 + padding)
            plate_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if plate_img.size > 0:
                # Resize plate image if too small
                min_width = 150
                if plate_img.shape[1] < min_width:
                    scale = min_width / plate_img.shape[1]
                    plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale)
                
                # Try multiple OCR configurations
                text = ""
                best_conf = 0.0
                
                # Preprocess plate once for both OCR engines
                thresh = self.preprocess_plate(plate_img)
                
                # 1. Try Tesseract first (faster)
                if self.use_tesseract:
                    try:
                        # Try different PSM modes
                        for psm in [7, 8, 6]:
                            tess_text = pytesseract.image_to_string(thresh, 
                                                             config=f'--psm {psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            clean_text = self.validate_plate_text(tess_text)
                            if clean_text:
                                text = clean_text
                                best_conf = 0.8 # Tesseract doesn't give easy confidence for string, assume high if validated
                                break
                    except Exception as e:
                        print(f"Tesseract Error: {str(e)}")
                
                # 2. If Tesseract failed or we want to double check, use EasyOCR
                if not text:
                    try:
                        # EasyOCR works better on raw or slightly processed images
                        # We pass the cropped plate directly
                        # Use allowlist to restrict to alphanumeric characters
                        # Pass the preprocessed image (deskewed & enhanced) to EasyOCR
                        
                        # DEBUG: Save the images being sent to OCR
                        os.makedirs('debug_plates', exist_ok=True)
                        timestamp = time.strftime("%H%M%S")
                        cv2.imwrite(f'debug_plates/plate_{timestamp}_{idx}_processed.png', thresh)
                        cv2.imwrite(f'debug_plates/plate_{timestamp}_{idx}_raw.png', plate_img)
                        
                        ocr_results = self.reader.readtext(
                            thresh, 
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                            detail=1,
                            paragraph=False,
                            decoder='beamsearch',
                            beamWidth=10
                        )
                        
                        # Fallback: If preprocessing hurt the image, try the raw color crop
                        if not ocr_results:
                            ocr_results = self.reader.readtext(
                                plate_img,
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                detail=1,
                                paragraph=False,
                                decoder='beamsearch',
                                beamWidth=5
                            )
                        
                        for (_, ocr_text, ocr_conf) in ocr_results:
                            clean_text = self.validate_plate_text(ocr_text)
                            if clean_text and ocr_conf > 0.4:
                                text = clean_text
                                best_conf = ocr_conf
                                break
                    except Exception as e:
                        print(f"EasyOCR Error: {str(e)}")
                
                if text:
                    plate_texts.append((text, confidence))
                    label = f"{text} ({confidence:.2f})"
                
                # Save plate image if it's unique
                if self.save_unique_plate(plate_img, text, confidence, idx):
                    print(f"New unique plate detected and saved!")
            
            # Draw label if available, else just confidence
            if label:
                # Draw with background for better visibility
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                cv2.rectangle(frame, (x1, y1-30), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            else:
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame, plate_texts

    def process_video(self, source='0'):
        """
        Process video from file or webcam.
        
        Args:
            source (str): '0' for webcam or path to video file.
        """
        # Create output directories if they don't exist
        os.makedirs('output_results', exist_ok=True)
        os.makedirs('plate_logs', exist_ok=True)
        
        # Create log file for detected plates
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = f'plate_logs/detected_plates_{timestamp}.txt'
        video_name = source if source == '0' else os.path.basename(source)
        
        with open(log_filename, 'w') as log_file:
            log_file.write(f"License Plate Detection Log\n")
            log_file.write(f"Video Source: {video_name}\n")
            log_file.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("-" * 50 + "\n\n")
        
        # Open video source
        cap = cv2.VideoCapture(source if source != '0' else 0)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
            
        # Set target resolution
        target_width = 1280
        target_height = 720
            
        # Create output video writer if source is a file
        out = None
        if source != '0':
            base_name = os.path.splitext(os.path.basename(source))[0]
            output_path = os.path.join('output_results', f'{base_name}_processed.mp4')
            
            # Get input video FPS
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            if not input_fps or input_fps < 1:
                input_fps = 30.0
            
            # Use x264 codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, input_fps, (target_width, target_height))
            print(f"\nProcessed video will be saved to: {output_path}")
            print(f"Detected plates will be logged to: {log_filename}")
            
        print("\nPress 'q' to quit")
        frame_count = 0
        unique_plates = set()  # Track unique plates to avoid duplicates
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 3rd frame to improve performance
            if frame_count % 3 != 0:
                continue
                
            # Resize frame while maintaining aspect ratio
            h, w = frame.shape[:2]
            scale = min(target_width/w, target_height/h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize the frame
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Create black canvas of target size
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate position to paste the resized frame
            y_offset = (target_height - new_h) // 2
            x_offset = (target_width - new_w) // 2
            
            # Paste the resized frame onto the canvas
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
                
            # Process frame
            processed_frame, plates = self.process_frame(canvas)
            
            # Show detections
            cv2.imshow("License Plate Detection", processed_frame)
            
            # Save frame if writing video
            if out is not None:
                out.write(processed_frame)
            
            # Log and display detected plates
            if plates:
                current_time = time.strftime("%H:%M:%S")
                
                # Clear terminal and show current detections
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\nDetected License Plates:")
                print("-" * 30)
                
                with open(log_filename, 'a') as log_file:
                    for text, conf in plates:
                        if text not in unique_plates:  # Log only new unique plates
                            unique_plates.add(text)
                            log_entry = f"Time: {current_time} | Plate: {text} | Confidence: {conf:.2f}\n"
                            log_file.write(log_entry)
                        
                        # Display in terminal
                        print(f"Plate: {text}")
                        print(f"Confidence: {conf:.2f}")
                        print("-" * 30)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Write summary to log file
        with open(log_filename, 'a') as log_file:
            log_file.write("\n" + "=" * 50 + "\n")
            log_file.write(f"Summary:\n")
            log_file.write(f"Total unique plates detected: {len(unique_plates)}\n")
            log_file.write("All unique plates:\n")
            for plate in sorted(unique_plates):
                log_file.write(f"- {plate}\n")
        
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nDetection completed!")
        print(f"Detected plates have been saved to: {log_filename}")

def main():
    # Initialize detector with fine-tuned model path
    detector = LicensePlateDetector()
    
    # List available test videos
    video_dir = "Testing Videos"
    if os.path.exists(video_dir):
        videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        if videos:
            print("\nAvailable test videos:")
            for i, video in enumerate(videos, 1):
                print(f"{i}. {video}")
            
            choice = input("\nEnter video number (or press Enter for webcam): ").strip()
            if choice and choice.isdigit() and 1 <= int(choice) <= len(videos):
                video_path = os.path.join(video_dir, videos[int(choice)-1])
                print(f"\nProcessing video: {videos[int(choice)-1]}")
                detector.process_video(video_path)
            else:
                print("\nStarting webcam feed...")
                detector.process_video('0')
        else:
            print("\nNo test videos found. Starting webcam feed...")
            detector.process_video('0')
    else:
        print("\nStarting webcam feed...")
        detector.process_video('0')

if __name__ == "__main__":
    main()

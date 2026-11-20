import cv2
import numpy as np
import imutils

def detect_plate(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None, None
            
        # Resize the image - maintain aspect ratio
        image = imutils.resize(image, width=600)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges sharp
        smooth = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges using Canny
        edged = cv2.Canny(smooth, 30, 200)
        
        # Find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Sort contours by area and keep top 10
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        screen_cnt = None
        
        # Loop over contours to find the license plate contour
        for c in cnts:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            
            # If our approximated contour has four points, it's likely the license plate
            if len(approx) == 4:
                screen_cnt = approx
                break
        
        if screen_cnt is None:
            print(f"No license plate contour found in {image_path}")
            return None, None
        
        # Create mask for the license plate
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [screen_cnt], 0, 255, -1)
        
        # Apply the mask to get the license plate region
        plate_img = cv2.bitwise_and(image, image, mask=mask)
        
        # Get the minimum rectangular region containing the plate
        (x, y, w, h) = cv2.boundingRect(screen_cnt)
        plate_roi = image[y:y+h, x:x+w]
        
        return plate_roi, (x, y, w, h)
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None

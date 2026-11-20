import cv2
import numpy as np
from license_plate_detection import LicensePlateDetector
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_model(test_images, ground_truth):
    detector = LicensePlateDetector()
    predictions = []
    actual = []

    for img_path, true_text in zip(test_images, ground_truth):
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Process the image
        processed_img, plates = detector.process_frame(img)
        if plates:
            # Get the predicted text (assuming the first plate is the correct one)
            predicted_text = plates[0][0]  # Assuming plates is a list of (text, confidence) tuples
            predictions.append(predicted_text)
            actual.append(true_text)
        else:
            predictions.append("")  # No plate detected
            actual.append(true_text)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predictions, average='weighted')
    accuracy = accuracy_score(actual, predictions)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    # Example usage
    test_images = ['path/to/test/image1.jpg', 'path/to/test/image2.jpg']  # Replace with actual paths
    ground_truth = ['ABC123', 'XYZ789']  # Replace with actual ground truth texts
    evaluate_model(test_images, ground_truth) 
# Number Plate Detection System

A lightweight number plate detection system that works without GPU requirements. This system uses OpenCV for image processing and Tesseract for text recognition, making it suitable for CPU-only environments.

## Features

- License plate detection using OpenCV image processing
- Text extraction using Tesseract OCR
- Support for multiple image formats (PNG, JPG, JPEG)
- Confidence scores for detected text

## Setup

1. Create a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

1. Create a folder named `test_images` in the project directory
2. Place your test images in the `test_images` folder
3. Run the main script:
```powershell
python main.py
```

The script will process all images in the `test_images` folder and display the detected license plate text along with confidence scores.

## Project Structure

- **main.py**: The entry point for the application. It uses argparse to allow users to specify a video source (webcam or video file) and initializes the LicensePlateDetector from license_plate_detection.py.
- **license_plate_detection.py**: Contains the main LicensePlateDetector class. It handles the detection of license plates using a YOLO model, preprocessing of images, OCR using Tesseract, and saving/logging of detected plates.

- **plate_detection.py**: Uses classical image processing techniques (contours, Canny, etc.) to detect license plates. It is independent of the YOLO model and can be used as an alternative detection method.

- **train.py**: Contains the logic for training or fine-tuning the YOLO model on a custom dataset. It uses the dataset configuration from dataset.yaml.

- **dataset.yaml**: Configuration file for the dataset used in training the YOLO model. It specifies the paths to the training and validation images and the class names.

## Model Performance

The model's performance is evaluated using various metrics, which are stored in the `runs/detects/license_plate_detector` directory. Below are some key metrics:

### Training and Validation Metrics

- **Results**: The `results.png` and `results.csv` files show the training and validation metrics over time, including loss, precision, recall, and F1 score.

![Training and Validation Metrics](runs/detects/license_plate_detector/results.png)

### Confusion Matrix

The confusion matrix provides insights into the model's performance in terms of true positives, false positives, true negatives, and false negatives.

![Confusion Matrix](runs/detects/license_plate_detector/confusion_matrix.png)

### Precision-Recall Curve

The Precision-Recall curve helps evaluate the model's performance across different thresholds.

![Precision-Recall Curve](runs/detects/license_plate_detector/PR_curve.png)

### F1 Score Curve

The F1 score curve represents the model's accuracy, considering both precision and recall.

![F1 Score Curve](runs/detects/license_plate_detector/F1_curve.png)

## Limitations

- Works best with clear, well-lit images
- Performance may vary depending on image quality
- Processing speed depends on CPU capabilities

## Tips for Better Results

1. Use clear, high-resolution images
2. Ensure good lighting conditions
3. Try to capture images with the license plate facing directly towards the camera
4. Avoid extreme angles or heavily skewed perspectives

from ultralytics import YOLO
from pathlib import Path

def train_yolo():
    # Initialize the model
    model = YOLO('yolov8n.pt')
    
    # Get absolute path to dataset.yaml
    dataset_path = Path('dataset.yaml').resolve()
    
    # Train the model
    results = model.train(
        data=str(dataset_path),
        epochs=100,
        imgsz=640,
        batch=8,
        name='license_plate_detector',
        optimizer='AdamW',
        patience=20,
        exist_ok=True
    )

if __name__ == '__main__':
    train_yolo()

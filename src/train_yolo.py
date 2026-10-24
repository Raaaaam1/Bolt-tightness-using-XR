import os
import shutil
from ultralytics import YOLO

def train():
    data_path = os.path.abspath("data/yolo_dataset/data.yaml")
    print(f"Starting YOLOv8n training with dataset: {data_path}")
    
    # Load base pretrained model
    model = YOLO("yolov8n.pt")
    
    # Train model
    results = model.train(
        data=data_path,
        epochs=30,
        imgsz=640,
        batch=4,
        workers=2,
        name="bolt_detector",
        project="runs/detect",
        exist_ok=True
    )
    
    best_pt = os.path.join("runs", "detect", "bolt_detector", "weights", "best.pt")
    if os.path.exists(best_pt):
        os.makedirs("models", exist_ok=True)
        dest_model = os.path.join("models", "best_bolt_yolo.pt")
        shutil.copy(best_pt, dest_model)
        print(f"\n>>> SUCCESS: Trained YOLO model saved to {dest_model} <<<")
    else:
        print("Warning: best.pt not found.")

if __name__ == "__main__":
    train()

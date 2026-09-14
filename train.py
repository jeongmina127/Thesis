from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="/mnt/c/Users/Jeongmin Cho/Desktop/Git Repository/Thesis/data.yaml",
            epochs=100, 
            imgsz=512,
            patience=5)
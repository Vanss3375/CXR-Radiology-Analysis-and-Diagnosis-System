from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='0TrainSegment\Train\data.yaml', epochs=100, imgsz=640)
model.export()
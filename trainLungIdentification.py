from ultralytics import YOLO
#load model
model = YOLO('yolov8n.pt')
#train
results = model.train(data='0TrainDataset\data.yaml', epochs=100, imgsz=640)
#save model
model.save('lung_identification_model.pt')
import os
import random
from pathlib import Path
from typing import List, Tuple
from ultralytics import YOLO
from PIL import Image, ImageDraw


def load_yolo_model(model_path: str):
    model = YOLO(model_path)
    return model


def select_random_images(folder_path: str) -> List[str]:
    extensions = ['*.png', '*.jpg', '*.jpeg']
    images = []
    for ext in extensions:
        images.extend(Path(folder_path).glob(ext))
    return random.sample(images, 5)


def get_bboxes(predictions, label: str) -> List[Tuple[int, int, int, int, float]]:
    bboxes = []
    for pred in predictions:
        if pred['name'] == label:
            bbox = (int(pred['box'][0]), int(pred['box'][1]), int(pred['box'][2]), int(pred['box'][3]), pred['confidence'], pred['name'])
            bboxes.append(bbox)
    return bboxes


def ensure_minimum_detections(image, bboxes, label, min_count=1):
    if len(bboxes) >= min_count:
        return bboxes

    retries = 0
    while len(bboxes) < min_count and retries < 10:
        random.seed(random.randint(0, 10000))
        predictions = model(image)
        predictions = predictions[0].boxes.data.cpu().numpy()
        labels = predictions[:, -1]
        boxes = predictions[:, :-1]
        scores = predictions[:, -2]

        labeled_preds = []
        for box, score, label in zip(boxes, scores, labels):
            labeled_preds.append({'box': box, 'confidence': score, 'name': model.names[label]})
        bboxes = get_bboxes(labeled_preds, label)
        retries += 1
    return bboxes


def process_images(model, image_paths: List[str]):
    results = {}
    for image_path in image_paths:
        print(f"Image: {image_path}")
        image = Image.open(image_path)
        bboxes = []

        for _ in range(3):
            predictions = model(image)
            predictions = predictions[0].boxes.data.cpu().numpy()
            labels = predictions[:, -1]
            boxes = predictions[:, :-1]
            scores = predictions[:, -2]

            labeled_preds = []
            for box, score, label in zip(boxes, scores, labels):
                labeled_preds.append({'box': box, 'confidence': score, 'name': model.names[label]})

            left_lobes = get_bboxes(labeled_preds, 'left_lobe')
            right_lobes = get_bboxes(labeled_preds, 'right_lobe')
            diaphragms = get_bboxes(labeled_preds, 'diaphragm')
            abnormalities = get_bboxes(labeled_preds, 'abnormalities')
            heart = get_bboxes(labeled_preds, 'heart')

            combined_lobes = left_lobes + right_lobes
            combined_lobes.sort(key=lambda x: x[4], reverse=True)

            left_lobes = ensure_minimum_detections(image,left_lobes, 'left_lobe')
            right_lobes = ensure_minimum_detections(image,right_lobes, 'right_lobe')
            heart = ensure_minimum_detections(image,heart, 'heart')
            diaphragms = ensure_minimum_detections(image,diaphragms, 'diaphragm')
            
            if len(heart) > 1:
                heart.sort(key=lambda x: x[4], reverse=True)
                heart = [heart[0]]
                
            if len(diaphragms) > 1:
                diaphragms.sort(key=lambda x: x[4], reverse=True)
                diaphragms = [diaphragms[0]]
                
            if len(combined_lobes) > 2:
                combined_lobes = combined_lobes[:2]

            if (len(right_lobes) == 0 and len(left_lobes) > 1) or len(left_lobes) == 0 and len(right_lobes) > 1:
                ax1,ay1,ax2,ay2,aconfidence,aname = combined_lobes[0]
                bx1,by1,bx2,by2,bconfidence,bname = combined_lobes[1]
                if ax1 < bx1:
                    aname = 'right_lobe'
                    bname = 'left_lobe'
                else:
                    bname = 'right_lobe'
                    aname = 'left_lobe'
                right_lobes = [(ax1,ay1,ax2,ay2,aconfidence,aname)]
                left_lobes = [(bx1,by1,bx2,by2,bconfidence,bname)]

            if len(left_lobes) > 1:
                left_lobes.sort(key=lambda x: x[4], reverse=True)
                left_lobes = [left_lobes[0]]

            if len(right_lobes) > 1:
                right_lobes.sort(key=lambda x: x[4], reverse=True)
                right_lobes = [right_lobes[0]]

            print("left_lobes: ", left_lobes)
            print("right_lobes: ", right_lobes)
            print("diaphragm: ", diaphragms)
            print("heart: ", heart)
            print("abnormalities: ", abnormalities)

            bboxes.extend(left_lobes)
            bboxes.extend(right_lobes)
            bboxes.extend(diaphragms)
            bboxes.extend(abnormalities)
            bboxes.extend(heart)
        results[image_path] = bboxes
    return results


def save_images_with_bboxes(image_paths: List[str], results, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    label_colors = {
        'left_lobe': "blue",
        'right_lobe': "green",
        'diaphragm': "yellow",
        'abnormalities': "purple",
        'heart': "orange"
    }

    for idx, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        for bbox in results[image_path]:
            x1, y1, x2, y2, confidence, label = bbox
            draw.rectangle([x1, y1, x2, y2], outline=label_colors[label], width=2)
            text_label = str(confidence) + " " + label
            draw.text((x1, y1), text_label, fill=label_colors[label])

        output_path = os.path.join(output_folder, f"result_{idx}.jpg")
        image.save(output_path)


model_path = 'AIs/Segmentation/weights/best.pt'
folder_path = '0TestDataset/Corona Virus Disease'
output_folder = '2TestResult'

model = load_yolo_model(model_path)
image_paths = select_random_images(folder_path)
results = process_images(model, image_paths)
save_images_with_bboxes(image_paths, results, output_folder)

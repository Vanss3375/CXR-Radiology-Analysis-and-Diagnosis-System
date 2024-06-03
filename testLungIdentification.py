import os
import random
from pathlib import Path
from typing import List, Tuple
from ultralytics import YOLO
from PIL import Image, ImageDraw
import csv
import cv2
import numpy as np
import pandas as pd
def load_yolo_model(model_path: str):
    model = YOLO(model_path)
    return model

def check_and_replace_empty(value):
    if isinstance(value, (list, np.ndarray)):
        return np.nan_to_num(value, nan=0.0)
    else:
        if pd.isna(value) or value == '':
            return 0
        else:
            return value
        
def select_all_images(folder_path: str) -> List[str]:
    extensions = ['*.png', '*.jpg', '*.jpeg']
    images = []
    for ext in extensions:
        images.extend(Path(folder_path).glob(ext))
    return images

def get_bboxes(predictions, label: str) -> List[Tuple[int, int, int, int, float]]:
    bboxes = []
    for pred in predictions:
        if pred['name'] == label:
            bbox = (int(pred['box'][0]), int(pred['box'][1]), int(pred['box'][2]), int(pred['box'][3]), pred['confidence'], pred['name'])
            bboxes.append(bbox)
    return bboxes

def get_object_class(path):
    path_parts = path.parts
    if len(path_parts) > 1:
        return path_parts[-2]
    return None
    
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

        # left_lobes = ensure_minimum_detections(image,left_lobes, 'left_lobe')
        # right_lobes = ensure_minimum_detections(image,right_lobes, 'right_lobe')
        # heart = ensure_minimum_detections(image,heart, 'heart')
        # diaphragms = ensure_minimum_detections(image,diaphragms, 'diaphragm')
        
        if len(heart) > 1:
            heart.sort(key=lambda x: x[4], reverse=True)
            heart = [heart[0]]
            
        if len(diaphragms) > 1:
            diaphragms.sort(key=lambda x: x[4], reverse=True)
            diaphragms = [diaphragms[0]]
            
        if len(combined_lobes) > 2:
            combined_lobes = combined_lobes[:2]

        if (len(right_lobes) == 0 and len(combined_lobes) == 2) or (len(left_lobes) == 0 and len(combined_lobes) == 2):
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

        if len(left_lobes) >= 2:
            left_lobes.sort(key=lambda x: x[4], reverse=True)
            left_lobes = [left_lobes[0]]

        if len(right_lobes) >= 2:
            right_lobes.sort(key=lambda x: x[4], reverse=True)
            right_lobes = [right_lobes[0]]

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
            draw.rectangle([x1, y1, x2, y2], outline=label_colors.get(label, "red"), width=2)
            text_label = f"{confidence:.2f} {label}"
            draw.text((x1, y1), text_label, fill=label_colors.get(label, "red"))

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        output_path = os.path.join(output_folder, f"result_{idx}.jpg")
        image.save(output_path, 'JPEG')

def get_lung_volume(image_paths: List[str], results) -> List[Tuple[str, float]]:
    VRX = []
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            continue  # If the image cannot be read, skip to the next image

        H1 = H2 = W1 = W2 = Wtemp1 = Wtemp2 = 0
        right_lobe_region = None
        left_lobe_region = None

        for bbox in results[image_path]:
            x1, y1, x2, y2, confidence, label = bbox
            if label == 'right_lobe':
                H1 = abs(y2 - y1)
                W1 = abs(x2 - x1)
                Wtemp1 = min(x1, x2)
                right_lobe_region = image[y1:y2, x1:x2]
            if label == 'left_lobe':
                H2 = abs(y2 - y1)
                W2 = abs(x2 - x1)
                Wtemp2 = max(x1, x2)
                left_lobe_region = image[y1:y2, x1:x2]

        Wtot = abs(Wtemp2 - Wtemp1)
        lung_volume = (Wtot / 2) * (H1 + H2) * ((W1 + W2) / 2)
        right_lobe_volume = (W1 * H1)
        left_lobe_volume = (W2 * H2)

        # Calculate average red channel value for the right lobe region if it exists
        right_lobe_color = None
        if right_lobe_region is not None and right_lobe_region.size > 0:
            right_lobe_color = np.mean(right_lobe_region[:, :, 2])  # Red channel is at index 2

        # Calculate average red channel value for the left lobe region if it exists
        left_lobe_color = None
        if left_lobe_region is not None and left_lobe_region.size > 0:
            left_lobe_color = np.mean(left_lobe_region[:, :, 2])  # Red channel is at index 2

        VRX.append((image_path, lung_volume, right_lobe_volume, left_lobe_volume, 
                    right_lobe_color, left_lobe_color))
    return VRX

def get_heart_volume(image_paths: List[str], results) -> List[Tuple[str, float]]:
    P = []
    
    for image_path in image_paths:
        A = B = C = 0
        for bbox in results[image_path]:
            x1, y1, x2, y2, confidence, label = bbox
            if label == 'diaphragm':
                C = abs(x2 - x1)
            
            if label == 'heart':
                A = abs(x2 - x1)
                B = abs(y2 - y1)
                break
        if C != 0:
            Vol = (A/C)
        else:
            Vol = 0.0
        Abnormalities = 0
        if Vol < 0.45 or Vol > 0.55:
            Abnormalities = 1
        P.append((image_path, Vol, Abnormalities))
    return P

def extract_subsequent_folder_and_filename(image_path: str) -> str:
    path_parts = Path(image_path).parts
    if len(path_parts) > 2:
        return os.path.join(path_parts[-2], path_parts[-1])
    return image_path

def save_area_to_csv(lobes_area: List[Tuple[str, float]],csv_filename: str):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Object Class", "label", "Lobe Size"])
        for areas in lobes_area:
            object_class = areas[0]
            object_label = areas[1]
            object_name = areas[2]
            writer.writerow([object_class,object_label,object_name])
            
def save_volumes_to_csv(lung_volumes: List[Tuple[str, float]], heart_volumes: List[Tuple[str, float]], csv_filename: str):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Lung Volume", "Heart Volume","Heart Cardiomegali","Left Lung Size", "Left Lung Color","Right Lung Size","Right Lung Color"])
        for lv, hv in zip(lung_volumes, heart_volumes):
            image_path_lv = extract_subsequent_folder_and_filename(lv[0])
            lung_volume = lv[1]
            llb_size = lv[3]
            llb_color = lv[5]
            rlb_size = lv[2]
            rlb_color = lv[4]
            heart_volume = hv[1]
            heart_abnormality = hv[2]
            writer.writerow([image_path_lv, lung_volume, heart_volume, heart_abnormality,llb_size,llb_color,rlb_size,rlb_color])

model_path = 'AIs/Segmentation/weights/best.pt'
folder_path = '1DataSet'
output_folder = '2TestResult'
csv_filename = 'volumes.csv'

model = load_yolo_model(model_path)
all_items = os.listdir(folder_path)
subfolders = [os.path.join(folder_path, item) for item in all_items if os.path.isdir(os.path.join(folder_path, item))]
image_paths = []
for subfolder in subfolders:
    image_paths.extend(select_all_images(subfolder))
results = process_images(model, image_paths)
lung_volumes = get_lung_volume(image_paths, results)
heart_volumes = get_heart_volume(image_paths, results)
# lobes_area = get_lobes_area(image_paths, results)
# save_area_to_csv(lobes_area, csv_filename)
save_images_with_bboxes(image_paths, results, output_folder)
save_volumes_to_csv(lung_volumes, heart_volumes, csv_filename)
print(f"Results saved to {csv_filename}")

import os
import io
import csv
import cv2
import joblib
import random
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.patches as patches

def loadYolo(model_path: str):
    model = YOLO(model_path)
    return model

def replaceEmpty(value): #?
    if isinstance(value, (list, np.ndarray)):
        return np.nan_to_num(value, nan=0.0)
    else:
        if pd.isna(value) or value == '':
            return 0
        else:
            return value

def getBBOXes(predictions, label: str) -> List[Tuple[int, int, int, int, float]]:
    bboxes = []
    for pred in predictions:
        if pred['name'] == label:
            bbox = (int(pred['box'][0]), int(pred['box'][1]), int(pred['box'][2]), int(pred['box'][3]), pred['confidence'], pred['name'])
            bboxes.append(bbox)
    return bboxes

def getObjectClass(path):
    path_parts = path.parts
    if len(path_parts) > 1:
        return path_parts[-2]
    return None
    
def ensureObjectDetection(model, image, bboxes, label, min_count=1):
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
        bboxes = getBBOXes(labeled_preds, label)
        retries += 1
    return bboxes

def initialProcessImage(model, image_path): #?
    results = {}
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
    left_lobes = getBBOXes(labeled_preds, 'left_lobe')
    right_lobes = getBBOXes(labeled_preds, 'right_lobe')
    diaphragms = getBBOXes(labeled_preds, 'diaphragm')
    abnormalities = getBBOXes(labeled_preds, 'abnormalities')
    heart = getBBOXes(labeled_preds, 'heart')
    combined_lobes = left_lobes + right_lobes
    combined_lobes.sort(key=lambda x: x[4], reverse=True)
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

def saveImage(image_path, results, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    label_colors = {
        'left_lobe': "blue",
        'right_lobe': "green",
        'diaphragm': "yellow",
        'abnormalities': "purple",
        'heart': "orange"
    }
    
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox in results[image_path]:
        x1, y1, x2, y2, confidence, label = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=label_colors.get(label, "red"), facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f"{confidence:.2f} {label}", color=label_colors.get(label, "red"), verticalalignment='top', bbox={'color': 'white', 'pad': 0})

    output_path = os.path.join(output_folder, "result.jpg")
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    plot_image = Image.open(buf)

def countLungVolume(image_path, results) -> List[Tuple[str, float]]:
    VRX = []
    image = cv2.imread(image_path)
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
    right_lobe_color = None
    if right_lobe_region is not None and right_lobe_region.size > 0:
        right_lobe_color = np.mean(right_lobe_region[:, :, 2])
    left_lobe_color = None
    if left_lobe_region is not None and left_lobe_region.size > 0:
        left_lobe_color = np.mean(left_lobe_region[:, :, 2])

    VRX.append((image_path, lung_volume, right_lobe_volume, left_lobe_volume, 
                right_lobe_color, left_lobe_color))
    return VRX

def countHeartVolume(image_path, results) -> List[Tuple[str, float]]:
    P = []
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

input_image = 'person5_bacteria_19.jpeg'
output_folder = '2TestResult'

segmentation_Path = 'AIs/Segmentation/weights/best.pt'
classification_MLp_model = 'AIs/Classification NN/weights/mlp_model.joblib'
classification_scaler_model = 'AIs/Classification NN/weights/scaler.joblib'

Segmentation_Model = loadYolo(segmentation_Path)
Classification_Mlp = joblib.load(classification_MLp_model)
Classification_Scaler = joblib.load(classification_scaler_model)

results = initialProcessImage(Segmentation_Model, input_image)
lung_volumes = countLungVolume(input_image, results)
heart_volumes = countHeartVolume(input_image, results)
plot = saveImage(input_image, results, output_folder)
new_data = pd.DataFrame({
    'Right Lung Size': [lung_volumes[0][2]],
    'Left Lung Size': [lung_volumes[0][3]],
    'Right Lung Color': [lung_volumes[0][4]],
    'Left Lung Color': [lung_volumes[0][5]],
})
left_lobe_features = new_data[['Left Lung Size', 'Left Lung Color']].values
right_lobe_features = new_data[['Right Lung Size', 'Right Lung Color']].values
left_lobe_features[:, 0] = np.log1p(left_lobe_features[:, 0])
right_lobe_features[:, 0] = np.log1p(right_lobe_features[:, 0])
features = np.vstack([left_lobe_features, right_lobe_features])
features = Classification_Scaler.transform(features)
predictions = Classification_Mlp.predict(features)
class_mapping = {0: 'Bacterial Pneumonia', 1: 'Corona Virus Disease', 2: 'Normal', 3: 'Tuberculosis', 4: 'Viral Pneumonia'} #!
decoded_predictions = [class_mapping[pred] for pred in predictions]
print(f"Predicted classes for the new data: {decoded_predictions}")
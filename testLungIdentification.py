from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def list_items_with_full_path(folder_path):
    items = [os.path.join(folder_path, item) for item in os.listdir(folder_path)]
    return items[:3]

def box_area(box):
    # Calculate the area of the bounding box given in xyxy format [x1, y1, x2, y2]
    x1, y1, x2, y2 = box.xyxy[0]
    return (x2 - x1) * (y2 - y1)

def process_results(results):
    processed_results = []
    for i, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        filtered_boxes = [box for box in boxes if box.conf >= 0.50]  # Filter boxes by confidence
        
        # Calculate area for each box and sort by area in descending order
        filtered_boxes.sort(key=box_area, reverse=True)
        
        if len(filtered_boxes) >= 2:
            processed_results.append(filtered_boxes[:2])
        else:
            processed_results.append(filtered_boxes)

    return processed_results

def main():
    model = YOLO('last.pt')  # Load a pretrained model
    image_files = list_items_with_full_path('./1TestDataset/Bacterial Pneumonia')
    print(image_files)

    best_results = []
    attempts = 0
    max_attempts = 5

    while attempts < max_attempts:
        random.seed(attempts)
        results = model(image_files)
        processed_results = process_results(results)
        
        # Check if we have at least 2 results with 2 boxes each
        valid_results = [result for result in processed_results if len(result) == 2]

        if len(valid_results) >= 2:
            best_results = valid_results
            break
        else:
            attempts += 1

    if not best_results:
        best_results = processed_results  # Fallback to whatever best we got

    # Save and display results
    for i, boxes in enumerate(best_results):
        if len(boxes) > 0:
            # Assuming result.save saves the image with bounding boxes
            save_path = f'./2TestResult/result_{i+1}.jpg'
            results[i].save(save_path)

            # Display result image in the notebook
            img = mpimg.imread(save_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis
            plt.show()

if __name__ == "__main__":
    main()

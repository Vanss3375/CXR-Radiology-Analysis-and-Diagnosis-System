from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def list_items_with_full_path(folder_path):
    items = [os.path.join(folder_path, item) for item in os.listdir(folder_path)]
    return items[:10]


model = YOLO('last.pt')  # load a pretrained model (recommended for training)
# List of items (images) in the specified folder
image_files = list_items_with_full_path('./1TestDataset/Bacterial Pneumonia')
print(image_files)
# Run batched inference on a list of images
results = model(image_files)  # Assuming `model` is predefined and compatible
# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs

    # Save result image
    save_path = f'./2TestResult/result_{i+1}.jpg'
    result.save(save_path)
    
    # Display result image in the notebook
    img = mpimg.imread(save_path)
    plt.imshow(img)
    plt.axis('off')  # Turn off axis
    plt.show()

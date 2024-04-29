import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
# Define neural network architecture
class LungDetector(nn.Module):
    def __init__(self):
        super(LungDetector, self).__init__()
        # Load a pre-trained Faster R-CNN model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # Replace the classifier with a new one
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = nn.Linear(in_features, 2)  # Output (x, y) for each vertex

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")
        return self.model(images, targets)


class LungDataset(Dataset):
    def __init__(self, image_paths, lung_coords):
        self.image_paths = image_paths
        self.lung_coords = lung_coords

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lung_coord = self.lung_coords[idx]

        image = Image.open(img_path).convert("RGB")
        image = torchvision.transforms.ToTensor()(image)

        target = {}
        target["vertices"] = torch.tensor(lung_coord, dtype=torch.float32)

        return image, target

def custom_loss(pred_vertices, target_vertices):
    # Calculate MSE loss
    batch_loss = 0.0
    for pred, target in zip(pred_vertices, target_vertices):
        # Calculate the actual number of vertices for each lung
        num_vertices = min(pred.shape[0], target.shape[0])
        # Calculate MSE loss for each lung
        lung_loss = nn.functional.mse_loss(pred[:num_vertices], target[:num_vertices], reduction='mean')
        batch_loss += lung_loss
    return batch_loss / len(pred_vertices)

def process_line(line):
    substrings = line.split(",")
    processed_integers = []
    for substring in substrings:
        if " " in substring:
            substring = substring.replace(" ", "")
        processed_integers.append(float(substring))
    return processed_integers


# Function to get coordinates from a file
def get_coord_list(image_path):
    xcoord = []
    ycoord = []
    zipimagecoord = []
    with open(image_path, "r") as file:
        lines = file.readlines()
        # Process each line
        coord_type_int = 0
        for line in lines:
            # Process the line and append the integers to the list
            if coord_type_int == 0:
                xcoord = process_line(line)
            elif coord_type_int == 1:
                ycoord = process_line(line)
            coord_type_int+=1
    zipcoord = list(zip(xcoord,ycoord))
    return zipcoord

def get_list_of_list(txt_path,segment):
    final_list = []
    txt_file_name = []
    txt_root = ""
    for root,dir,files in os.walk(txt_path):
        for file_name in files:
            txt_file_name.append(file_name)
        txt_root = root
            
    for files in txt_file_name:
        if files.startswith(segment):
            temp_file_path = os.path.join(txt_root, files)
            final_list.append(get_coord_list(temp_file_path))
    return final_list

def get_list_of_images(img_path):
    file_paths = []
    for root,dir,files in os.walk(img_path):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    return file_paths

def plot_image_with_coordinates(image_path, coordinates):
    image = plt.imread(image_path)
    fig, ax = plt.subplots()
    xlist, ylist = map(list, zip(*coordinates))
    ax.plot(xlist, ylist, color="green", linewidth=2)
    ax.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]], cmap="gray")
    ax.set_xlabel('X Width')
    ax.set_ylabel('Y Height')
    plt.show()


#!!! bikin pathnya masuk ke image path & coords path
# Hyperparameters
num_classes = 3  # Background, Left Lung, Right Lung
batch_size = 4
lr = 0.001
num_epochs = 10
input_dir_img = "2ResizedMask/img"
input_dir_txt = "2ResizedMask/txt"
# Prepare dataset
image_paths =  get_list_of_images(input_dir_img)
left_lung_coords = get_list_of_list(input_dir_txt, "L_")
right_lung_coords = get_list_of_list(input_dir_txt, "R_")

#??? FOR TESTING ONLY ???#
# print("img:",image_paths)
# print("#####################################################################################")
# print("left lung:",left_lung_coords)
# print("#####################################################################################")
# print("rightlung:",right_lung_coords)

# with open("test.txt", "w") as file:
#     i = 0
#     for sublist in left_lung_coords:
#         file.write(image_paths[i]+"\n")
#         for coordinate in sublist:
#             file.write(f'({coordinate[0]}, {coordinate[1]}) ')
#         file.write('\n\n')
#         i+=1

# Initialize left lung dataset
left_lung_dataset = LungDataset(image_paths, left_lung_coords)
left_lung_dataloader = DataLoader(left_lung_dataset, batch_size=batch_size, shuffle=True)

# Initialize right lung dataset
right_lung_dataset = LungDataset(image_paths, right_lung_coords)
right_lung_dataloader = DataLoader(right_lung_dataset, batch_size=batch_size, shuffle=True)

# Initialize left lung model
left_lung_model = LungDetector()

# Initialize right lung model
right_lung_model = LungDetector()

# Initialize optimizer for left lung model
left_lung_params = [p for p in left_lung_model.parameters() if p.requires_grad]
left_lung_optimizer = torch.optim.SGD(left_lung_params, lr=lr, momentum=0.9, weight_decay=0.0005)

# Initialize optimizer for right lung model
right_lung_params = [p for p in right_lung_model.parameters() if p.requires_grad]
right_lung_optimizer = torch.optim.SGD(right_lung_params, lr=lr, momentum=0.9, weight_decay=0.0005)

# Training loop for left lung model
for epoch in range(num_epochs):
    left_lung_model.train()
    for images, targets in left_lung_dataloader:
        left_lung_optimizer.zero_grad()
        vertices_pred = left_lung_model(images)
        loss = custom_loss(vertices_pred, targets["vertices"])  # Implement custom loss function
        loss.backward()
        left_lung_optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Left Lung Loss: {loss.item()}")

# Training loop for right lung model
for epoch in range(num_epochs):
    right_lung_model.train()
    for images, targets in right_lung_dataloader:
        right_lung_optimizer.zero_grad()
        vertices_pred = right_lung_model(images)
        loss = custom_loss(vertices_pred, targets["vertices"])  # Implement custom loss function
        loss.backward()
        right_lung_optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Right Lung Loss: {loss.item()}")

# Save models
output_dir = "3FinalSegmentAI"
os.makedirs(output_dir, exist_ok=True)
torch.save(left_lung_model.state_dict(), os.path.join(output_dir, 'left_lung_detector.pth'))
torch.save(right_lung_model.state_dict(), os.path.join(output_dir, 'right_lung_detector.pth'))
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
import os
from segmentation_models import Unet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Preprocess the polygon coordinates
# You'll need to implement this step based on your specific data format.

# Step 2: Define your neural network architecture# Step 2: Define your neural network architecture
class SegmentationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SegmentationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_apples = nn.Linear(hidden_size, output_size)
        self.fc_oranges = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out_apples = self.fc_apples(out)
        out_apples = self.sigmoid(out_apples)
        out_oranges = self.fc_oranges(out)
        out_oranges = self.sigmoid(out_oranges)
        return out_apples, out_oranges

# Step 4: Set up data loading
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



def get_sorted_files(folder_path,completeness = False):
    files = os.listdir(folder_path)
    files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
    sorted_files = sorted(files)
    full_paths = []
    if completeness == True:
        for file in sorted_files:
            full_paths.append(folder_path+"/"+file)
        return full_paths
    else:
        return sorted_files

def process_line(line):
    substrings = line.split(",")
    processed_integers = []
    for substring in substrings:
        if " " in substring:
            substring = substring.replace(" ", "")
        processed_integers.append(float(substring))
    return processed_integers

def get_coord_list(folder_path,segment):
    file_paths = get_sorted_files(folder_path)
    zipimagecoord = []
    i = 0
    for image_path in file_paths:
        xcoord = []
        ycoord = []
        if image_path.startswith(segment):
            image_name_only = image_path.split(".")[0]
            with open(folder_path+"/"+image_name_only+".txt", "r") as file:
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
        i+=1
        zipcoord = list(zip(xcoord,ycoord))
        zipimagecoord.append(zipcoord)
        filtered_list = list(filter(None, zipimagecoord))
    return filtered_list


# Define your image paths and annotations
img_path = "2LabeledMask/img"
label_path = "2LabeledMask/txt"
image_list = get_sorted_files(img_path, True)
right_lung_list = get_coord_list(label_path, "R_")
left_lung_list = get_coord_list(label_path, "L_")



# Step 3: Define a loss function
criterion = nn.BCELoss()

# Example data and labels
# Replace this with your actual data loading code
polygon_coordinates = np.random.rand(100, 10)  # Example polygon coordinates
segmentation_masks = np.random.randint(0, 2, (100, 1))  # Example segmentation masks

dataset = CustomDataset(polygon_coordinates, segmentation_masks)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Step 5: Train the model
input_size = 10  # Replace with your input size (number of coordinates)
hidden_size = 64
output_size = 1  # Assuming binary segmentation

model = SegmentationModel(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.float(), labels.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

print("Training finished")

# You can now use the trained model for inference

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define a simple convolutional neural network


class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(32*56*56, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Load the trained model
model = CNN()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict the class of an input image


def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = data_transforms(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = nn.functional.softmax(output[0], dim=0)
    return probabilities


# Class labels
class_labels = ['normal', 'pneumonia_bacteria', 'pneumonia_viral']

# Function to print prediction


def print_prediction(probabilities):
    for i, probability in enumerate(probabilities):
        print(f"{class_labels[i]}: {probability.item()*100:.2f}%")


# Input image path
input_image_path = 'dataset/test/Bacterial Pneumonia/116.jpeg'

# Predict class probabilities
probabilities = predict_image(input_image_path)

# Print prediction
print_prediction(probabilities)
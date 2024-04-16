import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

# Define a simple convolutional neural network
class CNN(nn.Module):
    def __init__(self, num_classes=5):
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

# Define data transformations with adaptive resizing
class AdaptiveResize(object):
    def __call__(self, image):
        size = min(image.size)
        resize = transforms.Resize(size)
        return resize(image)

data_transforms = transforms.Compose([
    AdaptiveResize(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data from subfolders
base_dir = 'C:\\Assignments\\Binus\\Reseach and Development in Computer Science\\ai\\train'
batch_size = 32
train_data = datasets.ImageFolder(base_dir, transform=data_transforms)

# Filter out subfolders that contain additional subfolders (need squaring)
base_classes = [c for c in train_data.classes if c == "Normal"]
squared_classes = [c for c in train_data.classes if c != "Normal"]

train_data_base = datasets.ImageFolder(os.path.join(base_dir, "Normal"), transform=data_transforms)
train_data_squared = datasets.ImageFolder(os.path.join(base_dir), transform=data_transforms,
                                          target_transform=lambda x: squared_classes.index(x) if x in squared_classes else -1)

train_loader_base = DataLoader(train_data_base, batch_size=batch_size, shuffle=True)
train_loader_squared = DataLoader(train_data_squared, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = CNN(num_classes=len(squared_classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model for base classes
num_epochs_base = 5
for epoch in range(num_epochs_base):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader_base:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_data_base)
    print(f"Base Model: Epoch [{epoch+1}/{num_epochs_base}], Loss: {epoch_loss:.4f}")

# Train the model for squared classes
num_epochs_squared = 32
for epoch in range(num_epochs_squared):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader_squared:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels[labels != -1])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_data_squared)
    print(f"Squared Model: Epoch [{epoch+1}/{num_epochs_squared}], Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

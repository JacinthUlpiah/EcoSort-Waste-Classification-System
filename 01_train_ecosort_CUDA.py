# Generated from: 01_train_ecosort_CUDA.ipynb
# Converted at: 2026-10-13T16:26:04.734Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# EcoSort training model


#Necessary libraries

# Standard Library
import os
import time

# Data & Visualization
import kagglehub #you can delete this once you installed the dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets, transforms, models

#Dont run this multiple times. If you downloaded the dataset, you don't need to download it again.

#Download latest version

#path = kagglehub.dataset_download("techsash/waste-classification-data")

#print("Path to dataset files:", path)

#Not redundant code, but just to make sure you have the dataset downloaded. If you already have it, you can comment this out.

#path = kagglehub.dataset_download("techsash/waste-classification-data")
#print(path)

cwd = os.getcwd()
print("Current working directory:", cwd)

path = r"e:\AaSchool\MAPUA\6th year\1st Qtr\CPE178P\Final Project\waste-classification-data\versions\1"
print(os.listdir(path))

print(os.listdir(os.path.join(path, "DATASET")))

print(os.listdir(os.path.join(path, "dataset", "DATASET")))

for category in ["O", "R"]:
    folder = os.path.join(path, "dataset", "TRAIN", category)
    print(category, ":", len(os.listdir(folder)), "files")
    print(os.listdir(folder)[:5])

train_path = os.path.join(path, "dataset", "TRAIN")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for row, category in enumerate(["O", "R"]):
    folder = os.path.join(train_path, category)
    files = os.listdir(folder)[:5]

    for col, filename in enumerate(files):
        img = Image.open(os.path.join(folder, filename))

        axes[row, col].imshow(img)
        axes[row, col].set_title(category)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()

print(os.listdir(os.path.join(path, "dataset", "TEST")))

for category in ["O", "R"]:
    folder = os.path.join(path, "dataset", "TEST", category)
    print(category, ":", len(os.listdir(folder)), "files")

sample_path = os.path.join(
    path, "dataset", "TRAIN", "O", "O_1.jpg"
)

img = Image.open(sample_path)

print("Image size:", img.size)
print("Image mode:", img.mode)

train_path = os.path.join(path, "dataset","TRAIN")

for category in ["O", "R"]:
    folder = os.path.join(train_path, category)
    sample = os.listdir(folder)[0]
    img = Image.open(os.path.join(folder, sample))
    print(category, sample, img.size, img.mode)

train_dir = os.path.join(path, "dataset", "TRAIN")
test_dir = os.path.join(path, "dataset", "TEST")

# 1. Training pipeline (with data augmentation and normalization)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # Resizes to 224x224 with a slight random zoom
    transforms.RandomHorizontalFlip(p=0.5),                # 50% chance to flip the image horizontally
    transforms.RandomRotation(degrees=15),                 # Randomly rotates the image up to 15 degrees
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Randomly adjusts brightness and contrast by 10%
    transforms.ToTensor(),                                 # Converts to tensor and scales pixels to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       # Standard EfficientNet/ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

# 2. Testing pipeline (strict evaluation, NO augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),                         # Standard resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       
                         std=[0.229, 0.224, 0.225])
])

class EcoSortDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# Load and split the FULL dataset
train_dir = os.path.join(path, "dataset", "TRAIN")
test_dir = os.path.join(path, "dataset", "TEST")

full_train_data = datasets.ImageFolder(train_dir)

# Initialize test_data (which was missing)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

train_size = int(0.85 * len(full_train_data))
val_size = len(full_train_data) - train_size

# Split the dataset exactly once
train_subset, val_subset = random_split(
    full_train_data, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Apply correct transforms
train_data = EcoSortDataset(train_subset, train_transform)
val_data = EcoSortDataset(val_subset, test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Testing:", len(test_data))

train_size = int(0.85 * len(train_data))
val_size = len(train_data) - train_size

train_subset, val_subset = random_split(
    train_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

print("Training:", len(train_subset))
print("Validation:", len(val_subset))
print("Testing:", len(test_data))

# Load pretrained EfficientNet-B0
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

print(model.classifier)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("Device:", device)

# Address imbalance (Weights = Total Samples / (Num Classes * Class Samples))
# O_weight = 22564 / (2 * 12565) = 0.898
# R_weight = 22564 / (2 * 9999) = 1.128
class_weights = torch.tensor([0.898, 1.128]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Freeze all feature layers first
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze the top layer (Block 8) for progressive fine-tuning
for param in model.features[8].parameters():
    param.requires_grad = True

# Ensure the new classifier is trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# Apply lower learning rate to the backbone, standard LR to the classifier
optimizer = torch.optim.Adam([
    {'params': model.features[8].parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])

print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# Optional: Add a learning rate scheduler to reduce LR if validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# Use 3,000 training images and 500 validation images for now
"""
small_train = Subset(train_subset, range(3000))
small_val = Subset(val_subset, range(500))

train_loader = DataLoader(
    small_train,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    small_val,
    batch_size=32,
    shuffle=False
)

print("Training:", len(small_train))
print("Validation:", len(small_val))
"""

num_epochs = 5 # Increased scope

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # --- 2. VALIDATION PHASE ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    # --- 3. SCHEDULER STEP ---
    # Reduces learning rate if validation loss stops improving
    scheduler.step(avg_val_loss)
    
    # --- 4. PRINT EPOCH RESULTS ---
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

model.eval()

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"Test Loss: {test_loss/len(test_loader):.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report
print(classification_report(
    all_labels,
    all_predictions,
    target_names=test_data.classes
))

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

print("Confusion Matrix:")
print(cm)

model.eval()

images, labels = next(iter(test_loader))
images = images.to(device)

# Warm-up
with torch.no_grad():
    model(images)

# Measure inference time
start_time = time.time()

with torch.no_grad():
    model(images)

end_time = time.time()

batch_time = end_time - start_time
image_time = batch_time / images.size(0)

print(f"Batch inference time: {batch_time:.4f} seconds")
print(f"Average inference time per image: {image_time:.4f} seconds")

os.makedirs("../models", exist_ok=True)

torch.save(model.state_dict(), "../models/ecosort_efficientnet_b0.pth")

print("Model saved successfully.")

# Compile the model
scripted_model = torch.jit.script(model)

# Define the export path and save
torchscript_path = "../models/ecosort_efficientnet_b0.pt"
scripted_model.save(torchscript_path)

print(f"Model exported to TorchScript at: {torchscript_path}")
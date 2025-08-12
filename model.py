import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import os
from PIL import Image
import matplotlib.pyplot as plt

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# PREP DATA
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_dir = "data/brain_mri/Training"
test_dir = "data/brain_mri/Testing"

# train data (80 train, 20 val)
full_train_dataset = ImageFolder(train_dir, transform=train_transforms)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# test data
test_dataset = ImageFolder(test_dir, transform=train_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model, loss, optimizer
model = timm.create_model('resnet18', pretrained=True, num_classes=4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model = False

# TRAINING
epochs = 5
best_val = 0.0
if train_model:

    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val:
            best_val = val_accuracy
            torch.save(model.state_dict(), "brain_mri_model_best.pth")
            print(f"Best model saved at epoch {epoch+1} with val acc {val_accuracy:.2f}%")

# TESTING
model.load_state_dict(torch.load("brain_mri_model_best.pth"))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

class_names = test_dataset.classes
misclassified = []

model.eval()
with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # For each item in the batch:
        for i in range(images.size(0)):
            if predicted[i] != labels[i]:
                # Record info: index in dataset, predicted and true label
                dataset_idx = idx * test_loader.batch_size + i
                misclassified.append((dataset_idx, predicted[i].item(), labels[i].item()))

print(f"Total misclassified samples: {len(misclassified)}")

# Optionally visualize some misclassified samples:
num_to_show = min(5, len(misclassified))

for sample in misclassified[:num_to_show]:
    idx, pred_label, true_label = sample
    img_path, _ = test_dataset.samples[idx]  # get path from ImageFolder dataset
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(f"True: {class_names[true_label]} | Predicted: {class_names[pred_label]}")
    plt.axis('off')
    plt.show()

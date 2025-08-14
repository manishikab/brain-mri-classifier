import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import timm
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dir = "data/brain_mri/Training"
test_dir = "data/brain_mri/Testing"

# Load dataset
train_dataset_raw = ImageFolder(train_dir)
test_dataset_raw = ImageFolder(test_dir)
all_samples = train_dataset_raw.samples + test_dataset_raw.samples

# Convert data for DataLoader
class MergedDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize
])

test_transforms = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# Stratified split into train/val/test
all_labels = [label for _, label in all_samples]
train_val_idx, test_idx = train_test_split(
    range(len(all_samples)),
    # 10% of data goes to the test set
    test_size=0.1,
    # test set has same class proportions as the full dataset.
    stratify=all_labels,
    random_state=42
)
train_idx, val_idx = train_test_split(
    train_val_idx,
    # 10% of total becomes validation.
    test_size=0.1111,
    # class balance again
    stratify=[all_labels[i] for i in train_val_idx],
    random_state=42
)

train_dataset = MergedDataset([all_samples[i] for i in train_idx], transform=train_transforms)
val_dataset = MergedDataset([all_samples[i] for i in val_idx], transform=test_transforms)
test_dataset = MergedDataset([all_samples[i] for i in test_idx], transform=test_transforms)

# ----------------------
# Binary dataset
# ----------------------

# 0: no tumor or 1: tumor
def to_binary_label(label):
    return 0 if label == 2 else 1

class BinaryLabelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, to_binary_label(label)

binary_train_loader = DataLoader(BinaryLabelDataset(train_dataset), batch_size=32, shuffle=True)
binary_val_loader = DataLoader(BinaryLabelDataset(val_dataset), batch_size=32, shuffle = False)
binary_test_loader = DataLoader(BinaryLabelDataset(test_dataset), batch_size=32, shuffle = False)

# ----------------------
# Multi-class tumor dataset
# ----------------------

# original: 0: glioma, 1: meningioma, 3: pituitary
tumor_classes = [0, 1, 3]

class TumorRemapDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # updated: 0: glioma, 1: meningioma, 2: pituitary
        self.label_map = {0:0, 1:1, 3:2}
        # filters out all no tumor samples
        self.indices = [i for i in range(len(dataset)) if dataset[i][1] in tumor_classes]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, self.label_map[label]

tumor_train_loader = DataLoader(TumorRemapDataset(train_dataset), batch_size=32, shuffle=True)
tumor_val_loader = DataLoader(TumorRemapDataset(val_dataset), batch_size=32)
tumor_test_loader = DataLoader(TumorRemapDataset(test_dataset), batch_size=32)

# ----------------------
# Models
# ----------------------

# 1. Binary model (ResNet34) to detect whether a tumor exists or not.
# 2. Multi-class model (ConvNeXt) to classify the type of tumor (glioma, meningioma, pituitary)
binary_model = timm.create_model('resnet34', pretrained=True, num_classes=2).to(device)
multi_model = timm.create_model('convnext_base', pretrained=True, num_classes=3).to(device)

# Class weights for multi-class
labels_multi = [label for _, label in TumorRemapDataset(train_dataset)]
counts = Counter(labels_multi)
total = len(labels_multi)
weights = [total / (3 * counts[i]) for i in range(3)]
weights[0] *= 1.3  # boost glioma
weights = torch.tensor(weights).float().to(device)

# Loss & optimizer
# weight decay to prevent overfitting
bin_criterion = nn.CrossEntropyLoss()
bin_optimizer = optim.AdamW(binary_model.parameters(), lr=.0001, weight_decay=1e-4)

multi_criterion = nn.CrossEntropyLoss(weight=weights)
multi_optimizer = optim.AdamW(multi_model.parameters(), lr=.00003, weight_decay=1e-4)

# ----------------------
# Training
# ----------------------
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, save_path):
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # x: batch of images, y: labels
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc = evaluate(model, val_loader)
        print(f"[{epoch+1}/{epochs}] Train Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved ({best_acc:.2f}%)")

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100 * correct / total

print("Training binary model")
train_model(binary_model, bin_criterion, bin_optimizer, binary_train_loader, binary_val_loader, epochs=30, save_path="binary_model_best.pth")

print("Training multi-class model")
train_model(multi_model, multi_criterion, multi_optimizer, tumor_train_loader, tumor_val_loader, epochs=30, save_path="multi_class_model_best.pth")
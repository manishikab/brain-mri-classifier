import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# PRE

#Pytorch Dataset
class BrainScanDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def classes(self):
        return self.data.classes
    

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

data_dir="data/brain_mri/Training"
dataset = BrainScanDataset(data_dir, transform=transform)

# Dataloader (batching)
dataloader = DataLoader(dataset, batch_size=32, shuffle = True)

# Pytorch Model
class SimpleScanClassifier(nn.Module):
    def __init__(self, num_classes = 4):
        super(SimpleScanClassifier, self).__init__()

        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280

        #Make Classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        output = self.classifier(x)
        return output
    
# TRAINING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleScanClassifier(num_classes=4)
model.to(device)

#Loss Function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# already in the prev code FIXME?
train_folder = data_dir
train_dataset = dataset
train_loader = dataloader

num_epochs = 5
train_losses = []


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}")

# torch.save(model.state_dict(), "brain_scan_model.pth")
# print("Model saved!")

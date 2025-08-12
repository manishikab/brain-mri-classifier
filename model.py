import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# PREPARE DATASET

# ImageNet mean and std:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_data = datasets.ImageFolder(root= 'data/brain_mri/Training', transform=train_transforms)
test_data = datasets.ImageFolder(root='data/brain_mri/Testing', transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Check dataset class names
print("Classes:", train_data.classes)
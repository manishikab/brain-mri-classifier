import torch
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from train import BinaryLabelDataset, TumorRemapDataset, test_dataset, test_transforms, device

# ----------------------
# Load models
# ----------------------
binary_model = timm.create_model('resnet34', pretrained=False, num_classes=2).to(device)
multi_model = timm.create_model('convnext_base', pretrained=False, num_classes=3).to(device)

binary_model.load_state_dict(torch.load("binary_model_best.pth", map_location=device))
multi_model.load_state_dict(torch.load("multi_class_model_best.pth", map_location=device))

binary_model.eval()
multi_model.eval()

# Data loaders
binary_test_loader = DataLoader(BinaryLabelDataset(test_dataset), batch_size=32)
tumor_test_loader = DataLoader(TumorRemapDataset(test_dataset), batch_size=32)

# ----------------------
# Testing
# ----------------------

# Binary evaluation
correct, total = 0, 0
with torch.no_grad():
    for x, y in binary_test_loader:
        x, y = x.to(device), y.to(device)
        out = binary_model(x)
        _, pred = torch.max(out, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()
print(f"Binary tumor detection accuracy: {100*correct/total:.2f}%")

# Multi-class evaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in tumor_test_loader:
        x, y = x.to(device), y.to(device)
        out = multi_model(x)
        _, pred = torch.max(out, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Multi-class tumor classification accuracy: {acc:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix (glioma, meningioma, pituitary):\n", cm)
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["glioma", "meningioma", "pituitary"]))
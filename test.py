"""Evaluate saved Brain MRI checkpoints on the leakage-safe held-out test set."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from data import RemappedDataset, make_datasets, make_label_maps

BATCH_SIZE = 32


def predict(model, loader, device):
    model.eval(); predictions, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            predictions.extend(model(images.to(device)).argmax(dim=1).cpu().numpy())
            labels.extend(targets.numpy())
    return np.array(predictions), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/brain_mri")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    _, _, test_dataset, class_to_idx, split_audit = make_datasets(args.data_root, return_audit=True)
    binary_map, tumor_map = make_label_maps(class_to_idx)
    loader_args = {"batch_size": BATCH_SIZE, "num_workers": 2, "pin_memory": device.type == "cuda"}

    binary_checkpoint = torch.load(output_dir / "binary_model_best.pt", map_location=device)
    multi_checkpoint = torch.load(output_dir / "multi_class_model_best.pt", map_location=device)
    binary_model = timm.create_model("resnet34", pretrained=False, num_classes=2).to(device)
    multi_model = timm.create_model("convnext_base", pretrained=False, num_classes=3).to(device)
    binary_model.load_state_dict(binary_checkpoint["model_state_dict"])
    multi_model.load_state_dict(multi_checkpoint["model_state_dict"])

    binary_predictions, binary_targets = predict(binary_model, DataLoader(RemappedDataset(test_dataset, binary_map), **loader_args), device)
    tumor_predictions, tumor_targets = predict(multi_model, DataLoader(RemappedDataset(test_dataset, tumor_map), **loader_args), device)
    binary_report = classification_report(binary_targets, binary_predictions, target_names=["no tumor", "tumor"], digits=3, output_dict=True, zero_division=0)
    tumor_report = classification_report(tumor_targets, tumor_predictions, target_names=["glioma", "meningioma", "pituitary"], digits=3, output_dict=True, zero_division=0)

    print("Binary best epoch:", binary_checkpoint["best_epoch"])
    print(f"Held-out binary accuracy: {(binary_predictions == binary_targets).mean():.3%} (n={len(binary_targets)})")
    print(classification_report(binary_targets, binary_predictions, target_names=["no tumor", "tumor"], digits=3, zero_division=0))
    print("Multi-class best epoch:", multi_checkpoint["best_epoch"])
    print(f"Held-out multi-class accuracy: {(tumor_predictions == tumor_targets).mean():.3%} (n={len(tumor_targets)})")
    print(confusion_matrix(tumor_targets, tumor_predictions))
    print(classification_report(tumor_targets, tumor_predictions, target_names=["glioma", "meningioma", "pituitary"], digits=3, zero_division=0))

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_split": split_audit,
        "best_validation": {"binary": binary_checkpoint["best_validation_accuracy"], "multi_class": multi_checkpoint["best_validation_accuracy"]},
        "held_out_test_accuracy": {"binary": float((binary_predictions == binary_targets).mean()), "multi_class": float((tumor_predictions == tumor_targets).mean())},
        "binary_report": binary_report,
        "multi_class_report": tumor_report,
        "multi_class_confusion_matrix": confusion_matrix(tumor_targets, tumor_predictions).tolist(),
    }
    with (output_dir / "evaluation_audit.json").open("w") as file: json.dump(audit, file, indent=2)
    print("Exact cross-split duplicate hashes:", sum(split_audit["exact_cross_split_duplicate_hashes"].values()))
    print("Saved evaluation audit to:", output_dir / "evaluation_audit.json")


if __name__ == "__main__":
    main()

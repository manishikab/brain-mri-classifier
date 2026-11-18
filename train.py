"""Train leakage-safe binary and multi-class Brain MRI baselines."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import RemappedDataset, make_datasets, make_label_maps

SEED, BATCH_SIZE, EPOCHS, PATIENCE = 42, 32, 20, 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1)
            labels = labels.to(device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(model, criterion, optimizer, train_loader, validation_loader, name, output_dir, device):
    best_validation, best_epoch, stale_epochs = -1.0, None, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        validation_accuracy = accuracy(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "validation_accuracy": validation_accuracy,
        }
        history.append(record)
        print(
            f"{name} epoch {epoch:02d}: loss={record['train_loss']:.4f}, "
            f"validation accuracy={validation_accuracy:.3%}"
        )
        if validation_accuracy > best_validation:
            best_validation, best_epoch, stale_epochs = validation_accuracy, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_validation,
                    "seed": SEED,
                },
                output_dir / f"{name}_best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"{name}: early stopping after epoch {epoch}.")
                break
    with (output_dir / f"{name}_history.json").open("w") as file:
        json.dump(history, file, indent=2)
    return {"best_epoch": best_epoch, "best_validation_accuracy": best_validation}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/brain_mri")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    train_dataset, validation_dataset, _, class_to_idx, split_audit = make_datasets(
        args.data_root, seed=SEED, return_audit=True
    )
    binary_map, tumor_map = make_label_maps(class_to_idx)
    loader_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": 2,
        "pin_memory": device.type == "cuda",
    }
    binary_train = RemappedDataset(train_dataset, binary_map)
    binary_validation = RemappedDataset(validation_dataset, binary_map)
    tumor_train = RemappedDataset(train_dataset, tumor_map)
    tumor_validation = RemappedDataset(validation_dataset, tumor_map)
    binary_train_loader = DataLoader(binary_train, shuffle=True, **loader_args)
    binary_validation_loader = DataLoader(binary_validation, **loader_args)
    tumor_train_loader = DataLoader(tumor_train, shuffle=True, **loader_args)
    tumor_validation_loader = DataLoader(tumor_validation, **loader_args)

    binary_model = timm.create_model(
        "resnet34", pretrained=True, num_classes=2
    ).to(device)
    binary_best = train_model(
        binary_model,
        nn.CrossEntropyLoss(),
        optim.AdamW(binary_model.parameters(), lr=1e-4, weight_decay=1e-4),
        binary_train_loader,
        binary_validation_loader,
        "binary_model",
        output_dir,
        device,
    )

    tumor_counts = Counter(
        tumor_train.dataset.labels[index] for index in tumor_train.indices
    )
    total_tumor_images = sum(tumor_counts.values())
    weights = torch.tensor(
        [
            total_tumor_images / (len(tumor_map) * tumor_counts[label])
            for label in tumor_map
        ],
        dtype=torch.float,
        device=device,
    )
    multi_model = timm.create_model(
        "convnext_base", pretrained=True, num_classes=3
    ).to(device)
    multi_best = train_model(
        multi_model,
        nn.CrossEntropyLoss(weight=weights),
        optim.AdamW(multi_model.parameters(), lr=3e-5, weight_decay=1e-4),
        tumor_train_loader,
        tumor_validation_loader,
        "multi_class_model",
        output_dir,
        device,
    )

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "configuration": {
            "batch_size": BATCH_SIZE,
            "maximum_epochs": EPOCHS,
            "early_stopping_patience": PATIENCE,
            "binary_architecture": "resnet34",
            "multi_class_architecture": "convnext_base",
        },
        "split": split_audit,
        "best_validation": {"binary": binary_best, "multi_class": multi_best},
    }
    with (output_dir / "training_audit.json").open("w") as file:
        json.dump(audit, file, indent=2)
    print(f"Saved checkpoints, histories, and audit to: {output_dir}")


if __name__ == "__main__":
    main()

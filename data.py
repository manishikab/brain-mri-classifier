"""Data preparation and leakage-safe splitting for the Brain MRI experiment."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)
EVAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


class ImageSampleDataset(Dataset):
    """A dataset built from ``ImageFolder`` samples and a chosen transform."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.labels = [label for _, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


class RemappedDataset(Dataset):
    """Filter and remap labels without duplicating image files."""

    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = label_map
        self.indices = [
            index for index, label in enumerate(dataset.labels) if label in label_map
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return image, self.label_map[label]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the exact-content hash for one image file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_hash_samples(samples, split_name: str):
    """Keep one image per hash and exclude hash groups with label conflicts."""
    grouped = defaultdict(list)
    for path, label in samples:
        grouped[sha256_file(path)].append((path, label))

    unique_samples, conflicting_hashes = [], 0
    for group in grouped.values():
        if len({label for _, label in group}) != 1:
            conflicting_hashes += 1
            continue
        unique_samples.append(group[0])

    return unique_samples, {
        "split": split_name,
        "files_before_deduplication": len(samples),
        "unique_images": len(unique_samples),
        "duplicate_files_excluded": len(samples) - len(unique_samples),
        "conflicting_label_hashes_excluded": conflicting_hashes,
    }


def make_datasets(
    data_root: str | Path = "data/brain_mri",
    validation_fraction: float = 0.10,
    seed: int = 42,
    return_audit: bool = False,
):
    """Create a leakage-safe train/validation/held-out-test split.

    ``Testing`` remains held out. Training images that exactly duplicate a raw
    test image are excluded; then exact duplicates are removed within each
    partition before the Training data is split.
    """
    data_root = Path(data_root)
    train_raw = ImageFolder(data_root / "Training")
    test_raw = ImageFolder(data_root / "Testing")
    if train_raw.classes != test_raw.classes:
        raise ValueError("Training and Testing class folders must match.")

    raw_test_hashes = {sha256_file(path) for path, _ in test_raw.samples}
    training_without_test_duplicates = [
        (path, label) for path, label in train_raw.samples
        if sha256_file(path) not in raw_test_hashes
    ]
    candidate_training, train_dedup = _unique_hash_samples(
        training_without_test_duplicates, "training candidates"
    )
    held_out_test, test_dedup = _unique_hash_samples(
        test_raw.samples, "held-out test"
    )

    candidate_labels = [label for _, label in candidate_training]
    train_indices, validation_indices = train_test_split(
        range(len(candidate_training)),
        test_size=validation_fraction,
        stratify=candidate_labels,
        random_state=seed,
    )
    train_samples = [candidate_training[index] for index in train_indices]
    validation_samples = [candidate_training[index] for index in validation_indices]

    train_hashes = {sha256_file(path) for path, _ in train_samples}
    validation_hashes = {sha256_file(path) for path, _ in validation_samples}
    test_hashes = {sha256_file(path) for path, _ in held_out_test}
    cross_split_duplicates = {
        "train_validation": len(train_hashes & validation_hashes),
        "train_test": len(train_hashes & test_hashes),
        "validation_test": len(validation_hashes & test_hashes),
    }
    if any(cross_split_duplicates.values()):
        raise RuntimeError(
            f"Exact duplicates remain across splits: {cross_split_duplicates}"
        )

    audit = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "raw_training_files": len(train_raw.samples),
        "raw_test_files": len(test_raw.samples),
        "raw_train_test_duplicate_hashes": len(
            {sha256_file(path) for path, _ in train_raw.samples} & raw_test_hashes
        ),
        "training_files_excluded_for_raw_test_overlap": (
            len(train_raw.samples) - len(training_without_test_duplicates)
        ),
        "deduplication": {"training": train_dedup, "held_out_test": test_dedup},
        "final_split_sizes": {
            "training_images": len(train_samples),
            "validation_images": len(validation_samples),
            "held_out_test_images": len(held_out_test),
        },
        "exact_cross_split_duplicate_hashes": cross_split_duplicates,
    }
    result = (
        ImageSampleDataset(train_samples, TRAIN_TRANSFORMS),
        ImageSampleDataset(validation_samples, EVAL_TRANSFORMS),
        ImageSampleDataset(held_out_test, EVAL_TRANSFORMS),
        train_raw.class_to_idx,
    )
    return (*result, audit) if return_audit else result


def make_label_maps(class_to_idx: dict[str, int]):
    """Build binary and tumor-only label maps from expected class folders."""
    required = {
        "glioma_tumor",
        "meningioma_tumor",
        "no_tumor",
        "pituitary_tumor",
    }
    missing = required - set(class_to_idx)
    if missing:
        raise ValueError(f"Missing expected class folders: {sorted(missing)}")
    no_tumor = class_to_idx["no_tumor"]
    binary_map = {label: int(label != no_tumor) for label in class_to_idx.values()}
    tumor_map = {
        class_to_idx["glioma_tumor"]: 0,
        class_to_idx["meningioma_tumor"]: 1,
        class_to_idx["pituitary_tumor"]: 2,
    }
    return binary_map, tumor_map

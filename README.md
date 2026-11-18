# Brain MRI Classifier

A deep-learning experiment that trains two image-classification baselines:

- A ResNet-34 model distinguishes no_tumor` images from tumor images.
- A ConvNeXt model classifies tumor images as glioma, meningioma, or pituitary tumor.

## Evaluation design

The provided `Testing` directory is preserved for final held-out evaluation. The pipeline excludes Training images that exactly duplicate a raw test image, deduplicates each partition by SHA-256 hash, and creates validation data only from the remaining Training images. It raises an error if exact hashes remain across train, validation, and held-out test partitions.

This avoids the project's earlier train/test mixing and provides a reproducible image-level split. The dataset does not include patient identifiers, so it cannot rule out patient-level overlap.

## Held-out results

The reported baseline was selected by validation accuracy and evaluated once on the leakage-safe held-out test partition (314 binary images; 233 tumor images):

| Task | Accuracy | Macro F1 | Important limitation |
| --- | ---: | ---: | --- |
| Binary tumor detection | 88.9% | 0.867 | Tumor recall: 87.1% |
| Three-class tumor classification | 73.4% | 0.726 | Glioma recall: 29.8%; most errors were glioma predicted as meningioma. |

These results are a baseline for this dataset, not evidence of clinical performance.

## Data

Download the [Brain Tumor MRI Dataset from Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and arrange it as follows. The data and generated checkpoints are intentionally excluded from Git.

```text
data/brain_mri/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## Run locally

```bash
python --version  # Python 3.10+
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py --data-root data/brain_mri --output-dir artifacts
python test.py --data-root data/brain_mri --output-dir artifacts
```

`train.py` saves validation-selected checkpoints, histories, and a split audit in `artifacts/`. `test.py` loads those checkpoints, evaluates the held-out partition, and writes a detailed evaluation audit.

## Project structure

- `data.py` — transforms, SHA-256 deduplication, leakage-safe splitting, and label maps
- `train.py` — deterministic training, validation-based early stopping, and checkpoints
- `test.py` — held-out metrics, classification reports, confusion matrix, and JSON audit

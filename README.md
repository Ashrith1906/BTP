# Breast Ultrasound Multi-Task Segmentation and Classification

A PyTorch-based multi-task computer vision project for breast ultrasound image analysis.

## Overview

This project combines lesion segmentation and 3-class classification in a staged pipeline:

```text
Ultrasound Image
       |
       v
     ASPP
       |
       v
    a-Net
       |
       +------> Lesion Segmentation Mask
       |
       v
 Image + Predicted Mask
       |
       v
 Fusion Adapter
       |
       v
    ResNet50
       |
       v
Benign / Malignant / Normal
```

## Training Strategy

1. **Stage A — Segmentation:** train the ASPP + U-Net-style a-Net lesion segmenter.
2. **Stage B — Classification:** train the ResNet50 classifier using the predicted segmentation mask through the fusion adapter.
3. **Stage C — Joint fine-tuning:** jointly optimize segmentation and classification with a weighted segmentation/classification objective.

## Dataset

The code expects the BUSI-style directory structure:

```text
data/
├── benign/
├── malignant/
└── normal/
```

Images are expected as `.png` files, with corresponding lesion masks following the image basename convention used by the notebook (for example, `image_mask.png` or `image_mask_*.png`).

**Dataset files are intentionally not included in this repository.**

## Setup

```bash
git clone <YOUR-REPOSITORY-URL>
cd breast-ultrasound-multitask

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Place the dataset under `data/` and run:

```bash
python main.py
```

## Repository Structure

```text
├── main.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── breast_cancer_multitask_original.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── train.py
│   ├── visualization.py
│   └── models/
│       ├── segmentation.py
│       ├── classification.py
│       └── __init__.py
├── checkpoints/
├── results/
└── .gitignore
```

## Evaluation

The supplied implementation reports:

- **Segmentation:** Dice Similarity Coefficient (DSC), IoU, HD95
- **Classification:** Accuracy, macro F1, macro Precision, macro Recall, macro ROC-AUC

## Reproducibility

The implementation uses a fixed random seed (`42`) and separates the raw samples into training and validation subsets before applying the training-set augmentation multiplier.

## Notes

The original Kaggle notebook is retained under `notebooks/` as a reference. The `src/` implementation reorganizes the same core pipeline into reusable modules suitable for GitHub and further development.

## Disclaimer

This repository is an academic BTP/research project and is not a clinical diagnostic system.

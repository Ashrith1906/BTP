"""Project configuration."""
from pathlib import Path
import random
import numpy as np
import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

CONFIG = {
    "DATA_ROOT": "./data",
    "CLASS_FOLDERS": ["benign", "malignant", "normal"],
    "IMG_SIZE": 256,
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 2,
    "NUM_CLASSES": 3,
    "NORMALIZATION": "imagenet",
    "AUGMENT_MULTIPLIER": 7,
    "SEG_EPOCHS": 60,
    "CLS_EPOCHS": 40,
    "JOINT_FT_EPOCHS": 15,
    "SEG_LR": 2e-4,
    "CLS_HEAD_LR": 3e-4,
    "CLS_BACKBONE_LR": 2e-5,
    "JOINT_LR": 1e-5,
    "WEIGHT_DECAY": 1e-4,
    "VAL_SPLIT": 0.20,
    "SEG_CKPT_PATH": "./checkpoints/best_anet_segmenter.pt",
    "CLS_CKPT_PATH": "./checkpoints/best_resnet50_classifier.pt",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

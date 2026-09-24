"""BUSI dataset loading, augmentation, splitting, and DataLoader construction."""
import glob
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .config import CONFIG, IMAGENET_MEAN, IMAGENET_STD, SEED

class BUSIDataset(Dataset):
    def __init__(self, samples, img_size=256, augment=False):
        self.samples = samples
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, img, mask):
        if random.random() < 0.5:
            img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
        if random.random() < 0.5:
            img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
        if random.random() < 0.6:
            angle = random.uniform(-25, 25)
            scale = random.uniform(0.85, 1.15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(
                mask, M, (w, h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT
            )
        if random.random() < 0.5:
            alpha = random.uniform(0.75, 1.3)
            beta = random.uniform(-25, 25)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.35:
            noise = np.random.normal(0, random.uniform(4, 15), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.2:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_paths, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for mp in mask_paths:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                mask = np.logical_or(mask, m > 127).astype(np.uint8)
        mask = mask.astype(np.float32)

        if self.augment:
            img, mask = self._apply_augmentation(img, mask)
            mask = (mask > 0.5).astype(np.float32)

        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (
            torch.from_numpy(img).permute(2, 0, 1).float(),
            torch.from_numpy(mask).unsqueeze(0).float(),
            torch.tensor(label, dtype=torch.long),
        )

def find_data_root(configured_root, class_folders):
    for search_dir in [configured_root, "/kaggle/input", "./", "../"]:
        if os.path.exists(search_dir):
            for dirpath, dirnames, _ in os.walk(search_dir):
                if any(c in dirnames for c in class_folders):
                    print(f"[INFO] Found dataset directory at: {dirpath}")
                    return dirpath
    raise FileNotFoundError(f"Cannot find dataset containing {class_folders}")

def build_samples(data_root, class_folders):
    data_root = find_data_root(data_root, class_folders)
    samples = []
    for label_idx, cls in enumerate(class_folders):
        cls_dir = os.path.join(data_root, cls)
        all_imgs = sorted(
            p for p in glob.glob(os.path.join(cls_dir, "*.png"))
            if "_mask" not in os.path.basename(p)
        )
        for img_path in all_imgs:
            base = os.path.splitext(img_path)[0]
            mask_paths = sorted(glob.glob(base + "_mask*.png"))
            samples.append((img_path, mask_paths, label_idx))
    return samples

def make_loaders():
    raw_samples = build_samples(CONFIG["DATA_ROOT"], CONFIG["CLASS_FOLDERS"])
    rng = random.Random(SEED)
    by_class = {}
    for sample in raw_samples:
        by_class.setdefault(sample[2], []).append(sample)

    raw_train, raw_val = [], []
    for _, items in by_class.items():
        rng.shuffle(items)
        n_val = int(len(items) * CONFIG["VAL_SPLIT"])
        raw_val.extend(items[:n_val])
        raw_train.extend(items[n_val:])

    train_samples = raw_train * CONFIG["AUGMENT_MULTIPLIER"]
    val_samples = raw_val

    train_loader = DataLoader(
        BUSIDataset(train_samples, CONFIG["IMG_SIZE"], augment=True),
        batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"], drop_last=True
    )
    val_loader = DataLoader(
        BUSIDataset(val_samples, CONFIG["IMG_SIZE"], augment=False),
        batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"]
    )
    print(f"Original BUSI images: {len(raw_samples)}")
    print(f"Training set after 7x expansion: {len(train_samples)} images")
    print(f"Validation set: {len(val_samples)} images")
    return train_loader, val_loader

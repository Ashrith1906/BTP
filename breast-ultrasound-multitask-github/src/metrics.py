"""Loss functions and evaluation metrics."""
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(1)
    union = probs.sum(1) + targets.sum(1)
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()

bce_loss_fn = nn.BCEWithLogitsLoss()
ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

def seg_loss_fn(seg_logits, masks):
    return 0.5 * bce_loss_fn(seg_logits, masks) + 0.5 * dice_loss(seg_logits, masks)

def compute_sample_metrics(pred_mask, gt_mask):
    pred_sum, gt_sum = pred_mask.sum(), gt_mask.sum()
    if gt_sum == 0 and pred_sum == 0:
        return 1.0, 1.0, 0.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0, 0.0, np.nan

    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    dice = (2.0 * inter) / (pred_sum + gt_sum + 1e-6)
    iou = inter / (union + 1e-6)

    d_pred = distance_transform_edt(1 - pred_mask)
    d_gt = distance_transform_edt(1 - gt_mask)
    surf_pred_to_gt = d_gt[pred_mask.astype(bool)]
    surf_gt_to_pred = d_pred[gt_mask.astype(bool)]
    hd95 = max(
        np.percentile(surf_pred_to_gt, 95),
        np.percentile(surf_gt_to_pred, 95)
    )
    return dice, iou, hd95

@torch.no_grad()
def evaluate(loader, seg_model, cls_model, fusion, device, threshold=0.5):
    seg_model.eval(); cls_model.eval(); fusion.eval()
    dices, ious, hds = [], [], []
    all_labels, all_preds, all_probs = [], [], []

    for imgs, masks, labels in loader:
        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
        seg_logits = seg_model(imgs)
        mask_prob = torch.sigmoid(seg_logits)
        fused = fusion(imgs, mask_prob)
        cls_logits = cls_model(fused)

        pred_masks = (mask_prob.cpu().numpy() > threshold).astype(np.uint8)[:, 0]
        gt_masks = masks.cpu().numpy().astype(np.uint8)[:, 0]
        for p, g in zip(pred_masks, gt_masks):
            d, i, h = compute_sample_metrics(p, g)
            dices.append(d); ious.append(i)
            if not np.isnan(h):
                hds.append(h)

        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs.argmax(1))
        all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "DSC": np.mean(dices) * 100,
        "IOU": np.mean(ious) * 100,
        "HD": np.mean(hds) if hds else float("nan"),
        "ACC": accuracy_score(all_labels, all_preds) * 100,
        "F1": f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100,
        "Precision": precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100,
        "Recall": recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100,
        "AUC": auc * 100,
    }

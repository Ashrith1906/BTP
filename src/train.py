"""Three-stage training pipeline."""
import os
import torch
import pandas as pd

from .config import CONFIG, DEVICE, USE_AMP, set_seed
from .data import make_loaders
from .models import ANetSegmenter, FusionAdapter, ResNet50Classifier
from .metrics import seg_loss_fn, ce_loss_fn, evaluate

def train():
    set_seed()
    os.makedirs("checkpoints", exist_ok=True)
    train_loader, val_loader = make_loaders()

    seg_model = ANetSegmenter().to(DEVICE)
    fusion = FusionAdapter().to(DEVICE)
    cls_model = ResNet50Classifier(CONFIG["NUM_CLASSES"]).to(DEVICE)

    history = {
        "seg_train_loss": [], "seg_val_dsc": [], "seg_val_iou": [],
        "cls_train_loss": [], "cls_val_acc": [], "cls_val_f1": []
    }
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Stage A: segmentation
    opt_seg = torch.optim.AdamW(
        seg_model.parameters(), lr=CONFIG["SEG_LR"],
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    sched_seg = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_seg, T_max=CONFIG["SEG_EPOCHS"], eta_min=1e-6
    )
    best_dsc = 0.0
    for epoch in range(1, CONFIG["SEG_EPOCHS"] + 1):
        seg_model.train()
        running_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt_seg.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = seg_loss_fn(seg_model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(opt_seg); scaler.update()
            running_loss += loss.item()
        sched_seg.step()
        train_loss = running_loss / len(train_loader)
        val_m = evaluate(val_loader, seg_model, cls_model, fusion, DEVICE)
        history["seg_train_loss"].append(train_loss)
        history["seg_val_dsc"].append(val_m["DSC"])
        history["seg_val_iou"].append(val_m["IOU"])
        print(f"[Seg {epoch:02d}/{CONFIG['SEG_EPOCHS']}] loss={train_loss:.4f} DSC={val_m['DSC']:.2f}")
        if val_m["DSC"] > best_dsc:
            best_dsc = val_m["DSC"]
            torch.save(seg_model.state_dict(), CONFIG["SEG_CKPT_PATH"])

    seg_model.load_state_dict(torch.load(CONFIG["SEG_CKPT_PATH"], map_location=DEVICE))

    # Stage B: differential fine-tuning classifier + fusion.
    optimizer_params = [
        {"params": cls_model.stem.parameters(), "lr": CONFIG["CLS_BACKBONE_LR"] * 0.2},
        {"params": cls_model.layer1.parameters(), "lr": CONFIG["CLS_BACKBONE_LR"] * 0.2},
        {"params": cls_model.layer2.parameters(), "lr": CONFIG["CLS_BACKBONE_LR"] * 0.5},
        {"params": cls_model.layer3.parameters(), "lr": CONFIG["CLS_BACKBONE_LR"]},
        {"params": cls_model.layer4.parameters(), "lr": CONFIG["CLS_BACKBONE_LR"]},
        {"params": cls_model.head.parameters(), "lr": CONFIG["CLS_HEAD_LR"]},
        {"params": fusion.parameters(), "lr": CONFIG["CLS_HEAD_LR"]},
    ]
    opt_cls = torch.optim.AdamW(optimizer_params, weight_decay=CONFIG["WEIGHT_DECAY"])
    sched_cls = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_cls, T_max=CONFIG["CLS_EPOCHS"], eta_min=1e-6
    )
    best_acc = 0.0
    for epoch in range(1, CONFIG["CLS_EPOCHS"] + 1):
        cls_model.train(); fusion.train(); seg_model.eval()
        running_loss = 0.0
        for imgs, _, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt_cls.zero_grad()
            with torch.no_grad():
                mask_prob = torch.sigmoid(seg_model(imgs))
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = cls_model(fusion(imgs, mask_prob))
                loss = ce_loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt_cls); scaler.update()
            running_loss += loss.item()
        sched_cls.step()
        train_loss = running_loss / len(train_loader)
        val_m = evaluate(val_loader, seg_model, cls_model, fusion, DEVICE)
        history["cls_train_loss"].append(train_loss)
        history["cls_val_acc"].append(val_m["ACC"])
        history["cls_val_f1"].append(val_m["F1"])
        print(f"[Cls {epoch:02d}/{CONFIG['CLS_EPOCHS']}] loss={train_loss:.4f} ACC={val_m['ACC']:.2f}")
        if val_m["ACC"] > best_acc:
            best_acc = val_m["ACC"]
            torch.save({"cls": cls_model.state_dict(), "fusion": fusion.state_dict()}, CONFIG["CLS_CKPT_PATH"])

    ckpt = torch.load(CONFIG["CLS_CKPT_PATH"], map_location=DEVICE)
    cls_model.load_state_dict(ckpt["cls"]); fusion.load_state_dict(ckpt["fusion"])

    # Stage C: joint fine-tuning.
    for p in seg_model.parameters():
        p.requires_grad = True
    opt_joint = torch.optim.AdamW([
        {"params": seg_model.parameters(), "lr": CONFIG["JOINT_LR"] * 0.5},
        {"params": cls_model.parameters(), "lr": CONFIG["JOINT_LR"]},
        {"params": fusion.parameters(), "lr": CONFIG["JOINT_LR"]},
    ], weight_decay=CONFIG["WEIGHT_DECAY"])

    for epoch in range(1, CONFIG["JOINT_FT_EPOCHS"] + 1):
        seg_model.train(); cls_model.train(); fusion.train()
        for imgs, masks, labels in train_loader:
            imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
            opt_joint.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                seg_logits = seg_model(imgs)
                mask_prob = torch.sigmoid(seg_logits)
                cls_logits = cls_model(fusion(imgs, mask_prob))
                total_loss = 0.3 * seg_loss_fn(seg_logits, masks) + 0.7 * ce_loss_fn(cls_logits, labels)
            scaler.scale(total_loss).backward()
            scaler.step(opt_joint); scaler.update()

        if epoch % 5 == 0 or epoch == CONFIG["JOINT_FT_EPOCHS"]:
            val_m = evaluate(val_loader, seg_model, cls_model, fusion, DEVICE)
            print(f"[Joint {epoch:02d}/{CONFIG['JOINT_FT_EPOCHS']}] DSC={val_m['DSC']:.2f} ACC={val_m['ACC']:.2f}")
            if val_m["ACC"] >= best_acc:
                best_acc = val_m["ACC"]
                torch.save(seg_model.state_dict(), CONFIG["SEG_CKPT_PATH"])
                torch.save({"cls": cls_model.state_dict(), "fusion": fusion.state_dict()}, CONFIG["CLS_CKPT_PATH"])

    final_metrics = evaluate(val_loader, seg_model, cls_model, fusion, DEVICE)
    pd.DataFrame([final_metrics]).to_csv("results/final_metrics.csv", index=False)
    print(pd.DataFrame([final_metrics]).round(2).to_string(index=False))
    return history, final_metrics

if __name__ == "__main__":
    train()

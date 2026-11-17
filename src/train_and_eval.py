import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "splits"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

CLASSES = [
    "chatpate", "chiya", "dalbhat", "jeri", "momo",
    "pakauda", "panipuri", "samosa", "selroti", "yomari"
]


class ParikarFoodDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = ROOT_DIR / row["path"]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(row["label"])
        return img, label


def get_transforms(img_size: int = 224):
    # Food-safe augmentations: geometric invariance + mild brightness/contrast jitter (no aggressive hue shift)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), value="random"),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def build_model(arch: str, num_classes: int = 10, pretrained: bool = True) -> Tuple[nn.Module, nn.Module]:
    arch = arch.lower()
    if arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        head = model.classifier
    elif arch == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        head = model.classifier
    elif arch == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        head = model.head
    elif arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        head = model.heads
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model, head


def compute_metrics(all_labels: np.ndarray, all_preds: np.ndarray, all_probs: np.ndarray, num_classes: int = 10) -> Dict:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[int(t), int(p)] += 1

    total = cm.sum()
    top1_acc = float(np.trace(cm) / max(total, 1))

    # Top-3 accuracy
    top3_Hits = 0
    for i, t in enumerate(all_labels):
        top3 = np.argsort(all_probs[i])[-3:]
        if int(t) in top3:
            top3_Hits += 1
    top3_acc = float(top3_Hits / max(len(all_labels), 1))

    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        per_class[CLASSES[c]] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "support": support
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))  # Balanced Accuracy
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / max(np.sum(supports), 1))

    return {
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "balanced_accuracy_macro_recall": round(macro_recall, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def save_confusion_matrix_plot(cm: List[List[int]], arch: str, split_name: str, out_path: Path):
    cm_arr = np.array(cm, dtype=np.float64)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, label="Recall (%)")
    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        title=f"{arch} — {split_name.upper()} Confusion Matrix (Normalized % & Counts)",
        ylabel="True Food Class",
        xlabel="Predicted Food Class"
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val_pct = cm_norm[i, j]
            val_cnt = int(cm[i][j])
            text = f"{val_cnt}\n({val_pct:.0f}%)" if val_cnt > 0 else "0"
            ax.text(
                j, i, text,
                ha="center", va="center", fontsize=8,
                color="white" if val_pct > thresh else "black"
            )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict, float]:
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    start_t = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits.float(), dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * images.size(0)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    elapsed = time.perf_counter() - start_t
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    avg_loss = running_loss / max(len(all_labels), 1)
    ms_per_img = (elapsed * 1000.0) / max(len(all_labels), 1)
    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes=len(CLASSES))
    metrics["loss"] = round(float(avg_loss), 4)
    metrics["inference_ms_per_image"] = round(float(ms_per_img), 2)
    return avg_loss, metrics, ms_per_img


def train_model(
    arch: str,
    epochs: int = 8,
    warmup_head_epochs: int = 1,
    batch_size: int = 32,
    lr_backbone: float = 1.5e-4,
    lr_head: float = 1e-3,
    num_workers: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = EXPERIMENTS_DIR / arch
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_transform, eval_transform = get_transforms(img_size=224)
    train_ds = ParikarFoodDataset(SPLITS_DIR / "train.csv", transform=train_transform)
    val_ds = ParikarFoodDataset(SPLITS_DIR / "val.csv", transform=eval_transform)
    test_ds = ParikarFoodDataset(SPLITS_DIR / "test.csv", transform=eval_transform)

    # Compute smoothed square-root weights from train split
    counts = train_ds.df["label"].value_counts().sort_index().values.astype(np.float64)
    mean_count = counts.mean()
    smoothed_weights = np.sqrt(mean_count / counts)

    # Use WeightedRandomSampler with smoothed weights for balanced exposure per epoch
    sample_weights = [smoothed_weights[int(lbl)] for lbl in train_ds.df["label"].values]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    model, head = build_model(arch, num_classes=len(CLASSES), pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n==========================================")
    print(f"Training {arch} on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Total Parameters: {total_params:,}")
    print(f"Train samples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"==========================================\n")

    # Since sampler already applies smoothed sqrt class balancing, use standard CrossEntropy with label smoothing=0.1
    # (Avoiding double-penalty of sampler + class weights simultaneously)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_criterion = nn.CrossEntropyLoss()

    history = []
    best_macro_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()

        # Phase 1: Linear head warmup (freeze backbone to protect ImageNet features from initial random gradients)
        if epoch <= warmup_head_epochs:
            for p in model.parameters():
                p.requires_grad = False
            for p in head.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(head.parameters(), lr=lr_head, weight_decay=1e-2)
            phase_name = "Phase 1 (Frozen Backbone Head Warmup)"
        else:
            if epoch == warmup_head_epochs + 1:
                for p in model.parameters():
                    p.requires_grad = True
                head_params_id = {id(p) for p in head.parameters()}
                backbone_params = [p for p in model.parameters() if id(p) not in head_params_id]
                optimizer = torch.optim.AdamW([
                    {"params": backbone_params, "lr": lr_backbone},
                    {"params": head.parameters(), "lr": lr_head * 0.5}
                ], weight_decay=1e-2)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(epochs - warmup_head_epochs, 1), eta_min=1e-6
                )
            phase_name = "Phase 2 (Full Fine-Tuning)"

        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            seen += images.size(0)

            if step % 60 == 0 or step == len(train_loader):
                print(
                    f"[{arch}] Epoch {epoch}/{epochs} ({phase_name}) | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Train Loss: {running_loss / seen:.4f} | Train Acc: {correct / seen * 100:.2f}%",
                    flush=True
                )

        if epoch > warmup_head_epochs:
            scheduler.step()

        train_loss = running_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)

        val_loss, val_metrics, _ = evaluate(model, val_loader, eval_criterion, device)
        ep_time = time.perf_counter() - ep_start

        ep_record = {
            "epoch": epoch,
            "phase": phase_name,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "val_loss": round(float(val_loss), 4),
            "val_top1_acc": val_metrics["top1_accuracy"],
            "val_top3_acc": val_metrics["top3_accuracy"],
            "val_balanced_acc": val_metrics["balanced_accuracy_macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": round(ep_time, 1)
        }
        history.append(ep_record)

        print(
            f"--> Epoch {epoch}/{epochs} Summary ({ep_time:.1f}s): "
            f"Val Loss={val_loss:.4f} | Val Top-1={val_metrics['top1_accuracy']*100:.2f}% | "
            f"Val Top-3={val_metrics['top3_accuracy']*100:.2f}% | "
            f"Val Balanced Acc={val_metrics['balanced_accuracy_macro_recall']*100:.2f}% | "
            f"Val Macro-F1={val_metrics['macro_f1']*100:.2f}%",
            flush=True
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save({
                "arch": arch,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "val_metrics": val_metrics
            }, exp_dir / "best_model.pt")
            save_confusion_matrix_plot(
                val_metrics["confusion_matrix"], arch, "val", exp_dir / "val_confusion_matrix.png"
            )

        with open(exp_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # Load best checkpoint and evaluate systematically on held-out Test set
    print(f"\nLoading best {arch} checkpoint (Epoch {best_epoch}, Val Macro-F1={best_macro_f1*100:.2f}%) for Test evaluation...")
    ckpt = torch.load(exp_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_metrics, _ = evaluate(model, test_loader, eval_criterion, device)
    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"], arch, "test", exp_dir / "test_confusion_matrix.png"
    )

    final_report = {
        "architecture": arch,
        "total_parameters": total_params,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "best_epoch": best_epoch,
        "epochs_trained": epochs,
        "history": history,
        "best_val_metrics": ckpt["val_metrics"],
        "test_metrics": test_metrics
    }

    with open(exp_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(
        f"\n[FINAL TEST RESULTS — {arch}] "
        f"Top-1 Acc: {test_metrics['top1_accuracy']*100:.2f}% | "
        f"Top-3 Acc: {test_metrics['top3_accuracy']*100:.2f}% | "
        f"Balanced Acc (Macro Recall): {test_metrics['balanced_accuracy_macro_recall']*100:.2f}% | "
        f"Macro F1: {test_metrics['macro_f1']*100:.2f}% | "
        f"Weighted F1: {test_metrics['weighted_f1']*100:.2f}% | "
        f"Latency: {test_metrics['inference_ms_per_image']:.2f} ms/img\n",
        flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="convnext_tiny", choices=["convnext_tiny", "efficientnet_v2_s", "swin_t", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_model(
        arch=args.arch,
        epochs=args.epochs,
        warmup_head_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

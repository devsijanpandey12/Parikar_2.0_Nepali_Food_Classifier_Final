import os
import json
import hashlib
import random
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
CLASSES = [
    "chatpate", "chiya", "dalbhat", "jeri", "momo",
    "pakauda", "panipuri", "samosa", "selroti", "yomari"
]
SPLITS_DIR = ROOT_DIR / "splits"
SEED = 42


def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    SPLITS_DIR.mkdir(exist_ok=True)

    valid_records = []
    corrupted_files = []
    exact_duplicates = []
    seen_hashes = {}

    print("Scanning and verifying 10 Nepali food classes...")
    for label_idx, cls_name in enumerate(CLASSES):
        cls_dir = ROOT_DIR / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")

        for img_path in sorted(cls_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
            except Exception as e:
                corrupted_files.append({"path": str(img_path.relative_to(ROOT_DIR)), "error": str(e)})
                continue

            file_hash = compute_md5(img_path)
            rel_path = str(img_path.relative_to(ROOT_DIR)).replace("\\", "/")

            if file_hash in seen_hashes:
                exact_duplicates.append({
                    "duplicate": rel_path,
                    "original": seen_hashes[file_hash],
                    "class": cls_name
                })
                continue

            seen_hashes[file_hash] = rel_path
            valid_records.append({
                "path": rel_path,
                "class_name": cls_name,
                "label": label_idx,
                "width": width,
                "height": height,
                "mode": mode,
                "md5": file_hash
            })

    df = pd.DataFrame(valid_records)
    print(f"Total verified unique images: {len(df)}")
    print(f"Corrupted images skipped: {len(corrupted_files)}")
    print(f"Exact MD5 duplicate images skipped (to prevent split leakage): {len(exact_duplicates)}")

    # Perform stratified 80 / 10 / 10 split per class
    train_rows, val_rows, test_rows = [], [], []
    for cls_name, group in df.groupby("class_name", sort=False):
        indices = list(group.index)
        random.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * 0.80))
        n_val = int(round(n * 0.10))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_rows.append(df.loc[train_idx])
        val_rows.append(df.loc[val_idx])
        test_rows.append(df.loc[test_idx])

    train_df = pd.concat(train_rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    # Calculate class weights from training set
    train_counts = train_df["class_name"].value_counts().to_dict()
    mean_train = len(train_df) / len(CLASSES)
    class_stats = {}
    for idx, cls_name in enumerate(CLASSES):
        c_train = int(train_counts.get(cls_name, 0))
        c_val = int((val_df["class_name"] == cls_name).sum())
        c_test = int((test_df["class_name"] == cls_name).sum())
        c_total = c_train + c_val + c_test
        inv_w = round(mean_train / c_train, 4)
        sqrt_w = round(float(np.sqrt(mean_train / c_train)), 4)
        class_stats[cls_name] = {
            "label": idx,
            "total_unique": c_total,
            "train": c_train,
            "val": c_val,
            "test": c_test,
            "inverse_weight": inv_w,
            "smoothed_sqrt_weight": sqrt_w
        }

    summary = {
        "total_scanned": len(valid_records) + len(corrupted_files) + len(exact_duplicates),
        "total_unique_valid": len(df),
        "corrupted_count": len(corrupted_files),
        "exact_duplicate_count": len(exact_duplicates),
        "corrupted_files": corrupted_files,
        "exact_duplicates_sample": exact_duplicates[:25],
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "classes": class_stats
    }

    with open(SPLITS_DIR / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved splits/train.csv, splits/val.csv, splits/test.csv, and splits/dataset_audit.json")


if __name__ == "__main__":
    main()

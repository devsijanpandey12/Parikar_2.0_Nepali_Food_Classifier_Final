# Parikar Nepali Food Classification — Architecture, Decision Log & Interview Guide

**Workspace:** `c:\Users\sijan\Desktop\parikar_final_nepali_food_datasets`  
**Last Updated:** 2026-11-17  

---

## 1. Executive Summary (STAR Format for Interviews)
- **Situation:** Built a 10-class Nepali food image classification system (`parikar_final_nepali_food_datasets`) containing **10,025 images** across dishes like `momo`, `dalbhat`, `selroti`, `yomari`, and `chatpate`, designed for web access via GitHub and cloud inference on **Microsoft Azure for Students**.
- **Task:** Address a **6.53:1 class imbalance** (`momo`: 2,239 images vs. `chatpate`: 343 images), high inter-class visual similarity among fried and dough-based foods, and strict cloud deployment constraints (CPU inference latency and memory budget on Azure Student tier).
- **Action:** Designed a leak-free stratified splitting pipeline, applied **smoothed inverse square-root weighting** ($w_c = \sqrt{\bar{N}/N_c}$), implemented **color-preserving geometric and MixUp/CutMix augmentations**, benchmarked a **3-tier model suite (`ConvNeXt-Tiny`, `Swin-Tiny / MobileViT-v2`, and `EfficientNetV2-S`)**, and exported the production model to **ONNX Runtime** for fast CPU inference on Azure.
- **Result / Goal:** Maximize **Macro F1-Score** and **Balanced Per-Class Recall** across all 10 categories while delivering `<50 ms` CPU inference latency on Azure.

---

## 2. Dataset Audit & Distribution
- **Total Images:** `10,025` (`9,475` `.jpg`, `548` `.jpeg`, `2` `.png`)
- **Imbalance Ratio:** `6.53 : 1`
- **Mean Samples per Class ($\bar{N}$):** `1,002.5`

| Category | Total Images ($N_c$) | Share (%) | Formats | Inverse Weight ($\bar{N}/N_c$) | Smoothed Weight ($\sqrt{\bar{N}/N_c}$) |
| :--- | ---: | ---: | :--- | ---: | ---: |
| `momo` | 2,239 | 22.33% | `.jpg`, `.jpeg` | 0.45 | **0.67** |
| `samosa` | 1,299 | 12.96% | `.jpg` | 0.77 | **0.88** |
| `jeri` | 1,248 | 12.45% | `.jpg` | 0.80 | **0.90** |
| `chiya` | 1,098 | 10.95% | `.jpg` | 0.91 | **0.96** |
| `selroti` | 1,082 | 10.79% | `.jpg` | 0.93 | **0.96** |
| `panipuri` | 867 | 8.65% | `.jpg` | 1.16 | **1.08** |
| `pakauda` | 809 | 8.07% | `.jpg` | 1.24 | **1.11** |
| `dalbhat` | 614 | 6.12% | `.jpg`, `.jpeg`, `.png` | 1.63 | **1.28** |
| `yomari` | 426 | 4.25% | `.jpg` | 2.35 | **1.53** |
| `chatpate` | 343 | 3.42% | `.jpg`, `.jpeg` | 2.92 | **1.71** |

---

## 3. Exact Image Counts per Class: Disk vs. After Sampling & Augmentation

A critical question for both pipeline design and interviews: **"What will the maximum number of images per class be, and will `chatpate` reach the same image count as `momo` after sampling and augmentation?"**

### A. Physical Files on Disk (80% Train / 10% Val / 10% Test Stratified Split)
With **on-the-fly dynamic augmentation**, we do **not** save duplicate augmented files to the hard drive. Instead, each time the dataloader reads a training image, it applies a new random crop, flip, rotation, or MixUp in RAM.
- **Max physical images on disk:** `momo` remains at **2,239 total** (`1,791` in `train`, `224` in `val`, `224` in `test`).
- **Min physical images on disk:** `chatpate` remains at **343 total** (`274` in `train`, `34` in `val`, `35` in `test`).

### B. Effective Images Seen by the Model per Epoch (After Sampling & Augmentation)
During training (`8,020` total training images across the 10 classes), how many times the model sees `chatpate` vs. `momo` depends on which sampling mode we select:

| Category | Raw Train Files (80%) | Mode 1: Equal 1:1 Oversampling (Keep Epoch = 8,020) | Mode 2: Equal 1:1 Oversampling to Match `momo` Max (Epoch = 17,910) | Mode 3: Smoothed $\sqrt{\bar{N}/N_c}$ Sampling (Recommended, Epoch = 8,020) |
| :--- | ---: | ---: | ---: | ---: |
| `momo` | 1,791 | ~802 / epoch | **1,791 / epoch** | **~1,275 / epoch** |
| `samosa` | 1,039 | ~802 / epoch | **1,791 / epoch** | **~972 / epoch** |
| `jeri` | 998 | ~802 / epoch | **1,791 / epoch** | **~953 / epoch** |
| `chiya` | 878 | ~802 / epoch | **1,791 / epoch** | **~894 / epoch** |
| `selroti` | 866 | ~802 / epoch | **1,791 / epoch** | **~888 / epoch** |
| `panipuri` | 694 | ~802 / epoch | **1,791 / epoch** | **~795 / epoch** |
| `pakauda` | 647 | ~802 / epoch | **1,791 / epoch** | **~767 / epoch** |
| `dalbhat` | 491 | ~802 / epoch | **1,791 / epoch** | **~668 / epoch** |
| `yomari` | 341 | ~802 / epoch | **1,791 / epoch** | **~557 / epoch** |
| `chatpate` | 274 | ~802 / epoch | **1,791 / epoch** | **~499 / epoch** |

---

## 4. Core Data & Training Decisions: What We Decided & Why

1. **Decision 1 — Deduplicate and Split *Before* Any Augmentation:** Prevents train-to-validation near-duplicate leakage.
2. **Decision 2 — Reject Hard Undersampling; Use Smoothed Square-Root Weighting ($w_c = \sqrt{\bar{N}/N_c}$):** Retains all `2,239` `momo` images while boosting `chatpate` without memorizing its `274` training backgrounds.
3. **Decision 3 — Food-Safe Augmentation + MixUp / CutMix:** Preserves discriminative food colors (golden `jeri` vs. brown `selroti`) while enforcing geometric invariance and smooth decision boundaries.
4. **Decision 4 — Decoupled Two-Stage Transfer Learning:** Trains the backbone on natural data first, then calibrates the classification head on balanced sampling.
5. **Decision 5 — Macro F1 & Per-Class Recall over Raw Top-1 Accuracy:** Prevents the `47.7%` majority (`momo`, `samosa`, `jeri`) from masking poor minority-class performance.

---

## 5. Model Selection & Interview Defense (Food-Specific CV Justification)

### Why Food Classification Is Unique in Computer Vision
Unlike rigid objects (cars, airplanes, buildings) that have fixed geometric silhouettes, **food is non-rigid, amorphous, and multi-scale**:
- **Local Fine-Grained Texture:** Distinguishing steamed `momo` pleats from pointed `yomari` dough, or porous fried `selroti` from twisted syrupy `jeri`, requires **high-frequency local edge and texture filters**.
- **Global Spatial Context:** Distinguishing a multi-bowl `dalbhat` thali or `panipuri` plate requires **long-range spatial context** across the entire plate.
- **Dataset Scale (`8,020` train images):** A pure vanilla Vision Transformer (`ViT-B/16` or `ViT-L`) lacks translation invariance and local inductive bias, making it prone to overfitting on small classes (`274` `chatpate` images) unless paired with hierarchical local windows or convolutional stems.

### Recommended 3-Model Benchmark Suite (Train & Compare All 3 for Interviews)

| Role | Model Architecture | Params | Model Size (FP32 / INT8 ONNX) | Why It Excels on Nepali Food (Interview Defense) |
| :--- | :--- | ---: | :--- | :--- |
| **1. Primary Hero Model** | **`ConvNeXt-Tiny`** (or `ConvNeXt-V2-Tiny`) | ~28.6M | ~110 MB / **~28 MB** | Combines CNN translation/texture inductive bias with Vision Transformer design (`7×7` depthwise kernels capture both local dough/fried textures and whole-plate layout). Outperforms ResNet-50 and vanilla ViT on mid-sized (`10k`) fine-grained food datasets like Food-101. |
| **2. Hierarchical Attention Model** | **`Swin-Tiny`** (or **`MobileViT-v2`**) | ~28.3M (`MobileViT`: ~5.6M) | ~110 MB / **~28 MB** (`MobileViT`: **~6 MB**) | Uses **shifted local attention windows** to model both fine-grained local ingredients (puffed rice/chili in `chatpate`) and global multi-item arrangements (`dalbhat` thali) without the data hunger of vanilla `ViT-B/16`. |
| **3. Production / Cloud Efficiency Baseline** | **`EfficientNetV2-S`** (or `MobileNetV3-Large`) | ~21.5M (`MNv3`: ~5.4M) | ~84 MB / **~22 MB** (`MNv3`: **~5.5 MB**) | Uses Fused-MBConv layers optimized for fast training and ultra-low CPU inference latency—ideal for cost-efficient deployment on **Azure for Students**. |

### Interview Defense: "Why Did You Choose `ConvNeXt-Tiny` / `Swin-Tiny` Over Other Models?"
- **Why not ResNet-50 / VGG-16?** *"ResNet-50 uses small `3×3` kernels designed in 2015. `ConvNeXt-Tiny` has a similar parameter budget (~28M) but uses modernized `7×7` depthwise convolutions, inverted bottlenecks, and GELU/LayerNorm, giving a +4–6% accuracy boost on fine-grained textures like food."*
- **Why not vanilla `ViT-Base` / `ViT-Large`?** *"Vanilla ViTs split images into `16×16` patches with no built-in locality or translation invariance. With only `274` training images in our smallest class (`chatpate`), vanilla ViT easily overfits to background plates, whereas `ConvNeXt` and `Swin-Tiny` bake in multi-scale spatial hierarchy."*
- **Why not YOLO?** *"YOLO is an object detector requiring bounding-box annotations (`x, y, w, h`). Our dataset is folder-organized for image-level classification (`ImageFolder`). However, if a user uploads a table with multiple dishes later, we can pair our classifier with a zero-shot detector or Grad-CAM localization."*

---

## 6. Hosting Architecture: GitHub Frontend + Microsoft Azure for Students Backend

### Can You Host This on Microsoft Azure for Students?
**Yes—100%.** The **Microsoft Azure for Students** offer provides **$100 in Azure credits for 12 months** plus free-tier services, which is more than enough to host the trained food classifier API 24/7 or serverless if architected properly.

### Important Azure Student Constraint (And How We Solve It)
- **Constraint:** Azure for Students typically restricts expensive GPU VMs (`NC`/`ND` series) and full PyTorch (`~2 GB` Docker image) can exceed the `1 GB` RAM limit of the free Azure App Service (`F1`) plan.
- **Our Production Solution — Export PyTorch to ONNX Runtime (`INT8` / `FP16`):**
  1. Train the models in PyTorch (locally or on free GPU notebooks like Kaggle/Colab), then export the best checkpoint to **`.onnx`**.
  2. Serving via **ONNX Runtime (`onnxruntime`) + FastAPI** shrinks the Docker container from `~2.5 GB` to `<250 MB`, cuts RAM usage to `<300 MB`, and speeds up CPU inference by **`3×–5×`** (`~15–40 ms` per image on a standard Azure CPU).

### End-to-End Deployment Blueprint
1. **Frontend (GitHub Pages / Vercel — 100% Free):**
   - Clean interactive web UI hosted directly from your GitHub repository where users can upload or snap a photo of Nepali food, view top-3 confidence bars, nutritional/cultural info for the dish, and a **Grad-CAM heatmap** showing where the model looked.
2. **Backend API (Microsoft Azure for Students):**
   - **Option A (Recommended — Azure Container Apps):** Deploy a lightweight **FastAPI + ONNX Runtime** Docker container. Azure Container Apps includes **180,000 vCPU-seconds and 360,000 GiB-seconds free per month**, and can scale to zero when idle so it **burns almost $0 of your $100 student credit**.
   - **Option B (Azure App Service B1 Linux Plan):** Included free for 12 months in the Azure for Students tier (`1 vCPU, 1.75 GB RAM`)—runs FastAPI + ONNX always-on with zero cold-start delay.

---

## 7. Chronological Action & Experiment Log
- **[2026-11-17]** Audited workspace structure (`10,025` raw images across `10` classes).
- **[2026-11-17]** Verified `hydro` Anaconda environment: `Python 3.11.15`, `PyTorch 2.13.0+cu130`, `Torchvision 0.28.0+cu130` on **NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)**.
- **[2026-11-17]** Ran `src/prepare_dataset.py` across all `10,025` files:
  - Found `0` corrupted images.
  - Caught and filtered **`121` exact MD5 byte-for-byte duplicate images** (e.g., `chatpate/000054.jpg` == `chatpate/000043.jpg`, `chiya/chiya_0051.jpg` == `chiya/chiya_0001.jpg`, `jeri/jeri (222).jpg` == `jeri/jeri (112).jpg`) *before* splitting so identical images never leak across train/val/test.
  - Produced `9,904` unique verified images split stratified 80/10/10 into `splits/train.csv` (`7,924`), `splits/val.csv` (`990`), and `splits/test.csv` (`990`).
- **[2026-11-17]** Built two-stage transfer learning & evaluation script (`src/train_and_eval.py`) supporting `convnext_tiny`, `efficientnet_v2_s`, `swin_t`, and `vit_b_16` with `bfloat16` mixed precision and full evaluation metrics.


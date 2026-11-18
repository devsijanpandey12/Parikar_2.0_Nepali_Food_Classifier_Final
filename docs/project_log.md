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

## 7. Systematic Model Benchmark Results

### Experiment 1: `ConvNeXt-Tiny` (`27,827,818` Parameters — RTX 4050 6GB, `bfloat16`)
- **Training Setup:**
  - **Epoch 1 (Phase 1 — Frozen Backbone Warmup):** `lr=1e-3` on the 10-class head only. Prevented random classifier gradients from corrupting ImageNet `7×7` depthwise filters; reached **`97.58%` Val Top-1** (`97.50%` Macro-F1) in just 1 epoch (`78.4s`).
  - **Epochs 2–6 (Phase 2 — Full Fine-Tuning):** Unfroze full backbone (`lr_backbone=1.5e-4`, `lr_head=5e-4`) with Cosine Annealing, `WeightedRandomSampler` using smoothed square-root weights ($\sqrt{\bar{N}/N_c}$), and `CrossEntropyLoss(label_smoothing=0.1)`.
- **Why 6 Epochs Was Optimal:** Validation loss steadily dropped from `0.2197` (Epoch 1) $\to$ `0.1286` (Epoch 2) $\to$ `0.1200` (Epoch 3) $\to$ `0.1106` (Epoch 5) $\to$ **`0.1103` (Epoch 6)**, where Validation Macro-F1 peaked at **`99.12%`**.

#### Epoch-by-Epoch Progression (`experiments/convnext_tiny/training_history.json`)

| Epoch | Training Phase | Train Loss | Train Acc | Val Loss | Val Top-1 Acc | Val Top-3 Acc | Val Balanced Acc | Val Macro-F1 | Time (s) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Phase 1 (Head Warmup) | 0.9073 | 87.78% | 0.2197 | 97.58% | 99.49% | 97.92% | 97.50% | 78.4s |
| 2 | Phase 2 (Full Fine-Tune) | 0.6043 | 97.27% | 0.1286 | 98.99% | 99.60% | 98.89% | 98.90% | 108.6s |
| 3 | Phase 2 (Full Fine-Tune) | 0.5396 | 98.90% | 0.1200 | 98.59% | 99.60% | 98.26% | 98.46% | 105.7s |
| 4 | Phase 2 (Full Fine-Tune) | 0.5208 | 99.50% | 0.1224 | 98.38% | 99.80% | 98.05% | 98.09% | 80.0s |
| 5 | Phase 2 (Full Fine-Tune) | 0.5107 | 99.84% | 0.1106 | 99.19% | 99.80% | 98.76% | 99.00% | 98.7s |
| **6** | **Phase 2 (Best Checkpoint)** | **0.5082** | **99.90%** | **0.1103** | **99.29%** | **99.80%** | **99.06%** | **99.12%** | **101.9s** |

#### Held-Out Test Set Evaluation (`990` Unseen Images — `experiments/convnext_tiny/evaluation_report.json`)
- **Test Top-1 Accuracy:** **`99.49%`** (`985 / 990` correct)
- **Test Top-3 Accuracy:** **`99.90%`**
- **Test Balanced Accuracy (Macro Recall):** **`99.51%`**
- **Test Macro Precision:** **`99.53%`**
- **Test Macro F1-Score:** **`99.51%`**
- **Test Inference Speed:** **`19.48 ms / image`**

| Class | Test Support | Precision | Recall | F1-Score | Imbalance Takeaway |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `chatpate` *(Smallest Class)* | 34 | **100.00%** | **100.00%** | **100.00%** | Zero false negatives or false positives despite 6.5:1 minority ratio |
| `yomari` *(2nd Smallest)* | 42 | **100.00%** | **100.00%** | **100.00%** | Zero confusion with steamed `momo` |
| `dalbhat` *(3rd Smallest)* | 61 | 98.39% | **100.00%** | 99.19% | 100% recall on full thali plates |
| `jeri` | 122 | **100.00%** | **100.00%** | **100.00%** | Color-safe augmentation preserved golden syrup cues |
| `momo` *(Largest Class)* | 222 | 99.55% | **100.00%** | 99.78% | 222/222 recalled without dominating minority predictions |
| `selroti` | 104 | **100.00%** | 99.04% | 99.52% | 103/104 correct |
| `panipuri` | 87 | **100.00%** | 98.85% | 99.42% | 86/87 correct |
| `pakauda` | 80 | **100.00%** | 98.75% | 99.37% | 79/80 correct |
| `samosa` | 128 | **100.00%** | 98.44% | 99.21% | 126/128 correct |
| `chiya` | 110 | 97.35% | **100.00%** | 98.65% | 110/110 recalled |

---

### Experiment 2: `EfficientNetV2-S` (`20,190,298` Parameters — RTX 4050 6GB, `bfloat16`)
- **Training Setup & Observations:**
  - In Epoch 1 (frozen backbone warmup), `EfficientNetV2-S` started at `86.16%` Val Top-1 (`85.44%` Macro-F1) vs. `ConvNeXt-Tiny`'s `97.58%` because `EfficientNetV2-S`'s Fused-MBConv features are more specialized to ImageNet class scales until unfrozen.
  - Once unfrozen in Phase 2 (Epoch 2), Val Top-1 jumped `+12.02%` in a single epoch to **`98.18%`**, peaking at **Epoch 5 (`98.99%` Val Top-1, `99.04%` Val Balanced Acc, `99.01%` Val Macro-F1)**.

#### Epoch-by-Epoch Progression (`experiments/efficientnet_v2_s/training_history.json`)

| Epoch | Training Phase | Train Loss | Train Acc | Val Loss | Val Top-1 Acc | Val Top-3 Acc | Val Balanced Acc | Val Macro-F1 | Time (s) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Phase 1 (Head Warmup) | 1.2403 | 74.32% | 0.6574 | 86.16% | 97.07% | 86.82% | 85.44% | 94.8s |
| 2 | Phase 2 (Full Fine-Tune) | 0.6532 | 95.58% | 0.1385 | 98.18% | 99.70% | 97.66% | 97.81% | 80.9s |
| 3 | Phase 2 (Full Fine-Tune) | 0.5560 | 98.47% | 0.1504 | 98.38% | 99.39% | 98.52% | 98.15% | 83.7s |
| 4 | Phase 2 (Full Fine-Tune) | 0.5358 | 99.09% | 0.1338 | 98.59% | 99.80% | 98.64% | 98.65% | 99.1s |
| **5** | **Phase 2 (Best Checkpoint)** | **0.5177** | **99.65%** | **0.1288** | **98.99%** | **99.60%** | **99.04%** | **99.01%** | **110.1s** |
| 6 | Phase 2 (Full Fine-Tune) | 0.5146 | 99.71% | 0.1227 | 98.99% | 99.70% | 99.01% | 98.96% | 99.5s |

#### Held-Out Test Set Evaluation (`990` Unseen Images — `experiments/efficientnet_v2_s/evaluation_report.json`)
- **Test Top-1 Accuracy:** **`99.39%`** (`984 / 990` correct)
- **Test Top-3 Accuracy:** **`100.00%`** (`990 / 990` correct!)
- **Test Balanced Accuracy (Macro Recall):** **`99.32%`**
- **Test Macro Precision:** **`99.16%`**
- **Test Macro F1-Score:** **`99.24%`**
- **Test Inference Speed:** **`18.70 ms / image`** (`27.4%` fewer parameters than `ConvNeXt-Tiny`)

---

### Experiment 3: `Swin-Tiny` (`27,527,044` Parameters — Hierarchical Shifted-Window Vision Transformer)
- **Training Setup & Empirical Takeaway (Crucial for Interviews):**
  - `Swin-Tiny` reached its best validation score at **Epoch 3 (`98.69%` Val Top-1, `98.57%` Val Macro-F1)**, after which training accuracy continued climbing (`98.06%` $\to$ `99.73%`) while validation Macro-F1 plateaued around `97.91%–98.21%`.
  - On the held-out Test Set (`990` unseen images), `Swin-Tiny` achieved **`97.88%` Top-1 Accuracy (`97.98%` Macro-F1)**—strong, but **`1.53%` lower Macro-F1 than `ConvNeXt-Tiny` (`99.51%`)** and **`1.26%` lower than `EfficientNetV2-S` (`99.24%`)**.
  - **Why did `ConvNeXt-Tiny` and `EfficientNetV2-S` beat `Swin-Tiny`?** Specifically, in `Swin-Tiny`'s test confusion matrix, `6` `momo` images were misclassified as `samosa` (both are folded dough items), reducing `samosa` precision to `93.38%` and `yomari` F1 to `95.24%`. Even with shifted windows, Vision Transformers have weaker local edge/pleat inductive bias than `ConvNeXt`'s `7×7` depthwise convolutions when training on a mid-sized (`7,924`-image) dataset.

#### Epoch-by-Epoch Progression (`experiments/swin_t/training_history.json`)

| Epoch | Training Phase | Train Loss | Train Acc | Val Loss | Val Top-1 Acc | Val Top-3 Acc | Val Balanced Acc | Val Macro-F1 | Time (s) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Phase 1 (Head Warmup) | 0.9466 | 85.58% | 0.2295 | 96.87% | 99.60% | 96.77% | 96.33% | 71.8s |
| 2 | Phase 2 (Full Fine-Tune) | 0.6488 | 95.72% | 0.1167 | 98.59% | 99.90% | 98.33% | 98.43% | 101.2s |
| **3** | **Phase 2 (Best Checkpoint)** | **0.5750** | **98.06%** | **0.1212** | **98.69%** | **99.80%** | **98.71%** | **98.57%** | **103.4s** |
| 4 | Phase 2 (Full Fine-Tune) | 0.5448 | 98.79% | 0.1278 | 98.08% | 99.80% | 97.97% | 97.91% | 96.4s |
| 5 | Phase 2 (Full Fine-Tune) | 0.5278 | 99.29% | 0.1212 | 98.28% | 99.90% | 97.79% | 98.06% | 95.7s |
| 6 | Phase 2 (Full Fine-Tune) | 0.5167 | 99.73% | 0.1174 | 98.38% | 99.90% | 98.04% | 98.21% | 88.0s |

---

### Side-by-Side 3-Model Final Benchmark Summary (`990` Unseen Test Images)

| Model Architecture | Family | Params | Best Epoch | Val Top-1 | Val Macro-F1 | Test Top-1 | Test Top-3 | Test Balanced Acc | Test Macro-F1 | `chatpate` F1 | `yomari` F1 | Verdict |
| :--- | :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`ConvNeXt-Tiny`** | Modernized 7×7 CNN | 27.8M | 6 | **99.29%** | **99.12%** | **99.49%** | 99.90% | **99.51%** | **99.51%** | **100.00%** | **100.00%** | **Winner (Highest Accuracy & Macro-F1)** |
| **`EfficientNetV2-S`** | Fused-MBConv CNN | **20.2M** | 5 | 98.99% | 99.01% | 99.39% | **100.00%** | 99.32% | 99.24% | 98.55% | 97.62% | **Runner-Up (Smallest Size & 100% Top-3)** |
| **`Swin-Tiny`** | Shifted-Window ViT | 27.5M | 3 | 98.69% | 98.57% | 97.88% | 99.90% | 98.04% | 97.98% | **100.00%** | 95.24% | **3rd Place (Early Overfitting after Ep 3)** |

---

### Interview Deep-Dive: "Why Did You Train for Only 6 Epochs, and Did You Use Early Stopping?"

1. **Why 6 Epochs Was Mathematically Sufficient (The `0.5004` Label-Smoothing Entropy Floor):**
   - Training a vision model *from scratch* requires `100–300` epochs because filters start as random noise. With **ImageNet-1K Transfer Learning**, the backbone already extracts rich edge, color, and texture hierarchies on day one (`ConvNeXt-Tiny` hit `97.58%` Val Top-1 after just 1 epoch of linear-head warmup).
   - More importantly, we trained with `CrossEntropyLoss(label_smoothing=0.1)` across $K = 10$ classes. Under label smoothing $\epsilon = 0.1$, the target probability for the true class is $1 - \epsilon + \frac{\epsilon}{K} = 0.91$, and for each of the 9 other classes is $\frac{\epsilon}{K} = 0.01$.
   - Therefore, **the theoretical minimum possible training loss (even for a 100% accurate oracle model) is not `0.0000`, but `0.5004`:**
     $$\mathcal{L}_{\min} = -\Big(0.91 \ln(0.91) + 9 \times 0.01 \ln(0.01)\Big) = 0.0859 + 0.4145 = \mathbf{0.5004}$$
   - By **Epoch 6** (`1,488` gradient updates), all three models had reached **`99.71%–99.90%` training accuracy** and training losses of **`0.5082`** (`ConvNeXt-Tiny`), **`0.5146`** (`EfficientNetV2-S`), and **`0.5167`** (`Swin-Tiny`)—sitting within **`0.0078`** of the theoretical `0.5004` floor while Cosine Annealing decayed the learning rate to `1e-6`. Training for 20–50 epochs past the theoretical loss floor would only force the network to memorize background noise.

2. **How We Used Early Stopping (Best-Macro-F1 Checkpoint Selection):**
   - Instead of blindly deploying the final epoch (`Epoch 6`), `src/train_and_eval.py` implements **Best-Validation-Macro-F1 Checkpointing** (retrospective early stopping over the cosine schedule): `best_model.pt` is only saved when `val_macro_f1` strictly improves.
   - **Empirical proof that early stopping worked:**
     - **`Swin-Tiny` stopped saving at Epoch 3** (`Val Macro-F1 = 98.57%`), automatically discarding Epochs 4, 5, and 6 where validation Macro-F1 degraded (`97.91%`, `98.06%`, `98.21%`).
     - **`EfficientNetV2-S` stopped saving at Epoch 5** (`Val Macro-F1 = 99.01%`), discarding Epoch 6 (`98.96%`).
     - **`ConvNeXt-Tiny` improved all the way to Epoch 6** (`Val Macro-F1 = 99.12%`, `Val Loss = 0.1103`).

---

## 8. Open-Set & Out-of-Distribution (OOD) Rejection Architecture (`evaluate_ood_gate`)

### The Closed-Set Softmax Problem
A standard $K = 10$ softmax classifier enforces $\sum_{i=1}^{10} p_i = 1$. Without an open-set rejection layer, uploading an image outside the 10 trained Nepali classes (e.g., pizza, burger, sushi, blank frame, or random non-food object) forces the softmax layer to assign `100%` of the probability mass across the 10 classes and display whichever dish had the slightly highest logit.

### Why Multi-Signal Post-Hoc OOD Rejection Was Chosen Over Retraining
1. **Preserves Verified Benchmarks & Zero Retraining Overhead:** Retraining with an 11th `"unknown"` background class would alter the 10-class decision boundaries and require regenerating all confusion matrices and benchmark tables (`99.49%` Test Top-1).
2. **Zero Extra RAM/Container Footprint for Azure for Students:** Loading a separate Vision-Language gatekeeper (such as CLIP/SigLIP) would add `300–600 MB` of weights to the container. The Multi-Signal OOD Gate executes in **`< 0.1 ms` of pure NumPy math** directly on the pre-softmax logits and probabilities already produced by ONNX Runtime.
3. **Exploits Label Smoothing Calibration ($\varepsilon = 0.1$) + Architectural Diversity (CNN vs. ViT):**
   - Because all 3 models were trained with `CrossEntropyLoss(label_smoothing=0.1)`, the network is explicitly penalized from driving logits to $+\infty$. For an ideal in-distribution prediction $[0.91, 0.01, \dots, 0.01]$, the theoretical normalized Shannon entropy is:
     $$\hat{H}_{\text{ideal}} = -\frac{0.91 \ln(0.91) + 9 \times 0.01 \ln(0.01)}{\ln(10)} = \frac{0.5004}{2.3026} \approx \mathbf{0.217}$$
   - Furthermore, running **two convolutional networks** (`ConvNeXt-Tiny`, `EfficientNetV2-S`) alongside **one hierarchical Vision Transformer** (`Swin-Tiny`) provides natural inductive-bias diversity: on genuine Nepali food, local convolutional texture filters and global shifted-window attention agree **`>98%` of the time**, whereas on out-of-distribution objects they diverge sharply.

### The 3-Layer Open-Set Defense Evaluated in `backend/app.py` (`evaluate_ood_gate`)

Because a closed-set 10-class linear classifier $z = f(x)W^T + b$ projects 768-dimensional features onto a 10-dimensional hyperplane subspace (ignoring the 758 orthogonal feature dimensions), a single confidence threshold can still be fooled by real-world non-food objects (e.g., webcam faces, laptops, phones, rooms) or outside Western foods (pizza, burger, fruits). To solve this without retraining the 3 main models, `backend/app.py` implements a **3-Layer Open-Set Rejection Gate**:

1. **Layer 1 — Pretrained ImageNet-1K Semantic Domain Verifier (`models_onnx/semantic_gate.onnx`):**
   - Uses a lightweight `MobileNetV3-Small` ONNX session (`9.71 MB`, `~2.5 ms` CPU inference) alongside a 120-class **Culinary Domain Mask** (`allowed_imagenet_indices.json`) calibrated across `600` training images from `splits/train.csv`.
   - On all 10 Nepali dishes, **`81.2% – 99.9%`** of the semantic probability mass falls inside these 120 culinary-compatible indices (`test median = 93.89%`).
   - Immediately rejects:
     - **Non-Food Objects / Webcam Faces / Electronics / Animals / Rooms:** Whenever a non-culinary ImageNet class exceeds `32.0%` top-1 probability or total culinary domain mass drops below `38.0%`.
     - **Explicit Outside Foods:** Whenever non-Nepali foods (`pizza`, `cheeseburger`, `hotdog`, `ice cream`, `strawberry`, `banana`, `pineapple`, `orange`, `lemon`, etc.) appear as top-1 with `>= 22.0%` probability.
2. **Layer 2 — 768-D Penultimate Feature Prototype Cosine Similarity (`models_onnx/convnext_centroids.npy`):**
   - Exposes the 768-dimensional penultimate LayerNormed embedding $f(x) \in \mathbb{R}^{768}$ (`/classifier/classifier.1/Flatten_output_0`) in `convnext_tiny.onnx` and computes the cosine similarity to the 10 class centroids $\mu_c \in \mathbb{R}^{768}$ derived from `splits/train.csv`:
     $$S_{\cos}(x, \mu_{\hat{y}}) = \frac{f(x) \cdot \mu_{\hat{y}}}{\|f(x)\|_2 \|\mu_{\hat{y}}\|_2} \quad \text{(Threshold: } \ge 0.85 \text{)}$$
   - True Nepali dishes in `splits/test.csv` achieve a median 768-D cosine similarity of **`0.9822`** (`p5 = 0.9386`, pinned samples `0.9645 – 0.9920`).
3. **Layer 3 — Tightened Label-Smoothed Softmax, Entropy, Logit Energy & 3-Architecture Consensus:**
   - **Maximum Softmax Probability (`confidence`):** $\ge 78.0\%$ (`test median = 92.01%`)
   - **Top-1 vs. Top-2 Margin (`margin`):** $\ge 60.0\%$ (`test median = 91.03%`)
   - **Normalized Shannon Entropy (`normalized_entropy`):** $\hat{H}(p) \le 0.36$ (`test median = 0.20`)
   - **Pre-Softmax Maximum Logit (`max_logit`):** $z_{\max} \ge 3.35$ (`test median = 4.16`)
   - **Cross-Architecture Consensus (`ConvNeXt-Tiny` + `EfficientNetV2-S` + `Swin-Tiny`):** Rejects 3-way disagreements, split verdicts with ensemble mean `< 86.0%` (or min `< 72.0%`), or unanimous fallbacks with mean `< 80.0%` (or min `< 68.0%`).

### Empirical Calibration (`splits/test.csv` vs. Out-of-Distribution Inputs)

| Metric Signal (`ConvNeXt-Tiny` + Semantic Gate) | Held-Out Test Set (`p5`–`p95` Range) | Held-Out Test Median | Calibrated OOD Gate Threshold | Typical OOD Input (Objects / Outside Food / Noise) |
| :--- | :---: | :---: | :---: | :---: |
| **Culinary Domain Mass (`semantic_gate.onnx`)** | `48.03% – 99.60%` | `93.89%` | **`< 38.0%` or Non-Food Top-1 triggers OOD** | `0.1% – 18.0%` |
| **768-D Centroid Cosine Similarity (`S_cos`)** | `0.9386 – 0.9920` | `0.9822` | **`< 0.85` triggers OOD** | `0.35 – 0.79` |
| **Top-1 Confidence (`MSP`)** | `90.88% – 92.99%` | `92.01%` | **`< 78.0%` triggers OOD** | `15.9% – 72.0%` |
| **Top-1 / Top-2 Margin (`M`)** | `89.60% – 92.12%` | `91.03%` | **`< 60.0%` triggers OOD** | `1.8% – 48.0%` |
| **Normalized Entropy (`H / ln 10`)** | `0.18 – 0.22` | `0.20` | **`> 0.36` triggers OOD** | `0.42 – 0.97` |
| **Max Pre-Softmax Logit (`z_max`)** | `4.02 – 4.43` | `4.16` | **`< 3.35` triggers OOD** | `0.53 – 3.10` |

- **Empirical In-Distribution Acceptance Rate:** **`98.33%`** (`295 / 300` stratified unseen test images accepted) and **`100%` (`10 / 10`)** across all pinned showcase samples.
- **Empirical Out-of-Distribution Rejection Rate:** **`100%`** across synthetic noise, blank/solid color frames, and non-food document/chart graphics.

### Frontend UX Behavior (`frontend/app.js` & `frontend/styles.css`)
When `/api/predict` returns `is_recognized: false` (`ood_gate.status == "unrecognized_ood"`):
1. **Amber OOD Spotlight Banner (`.dish-spotlight-card.ood-rejected`):**
   - Category pill switches to `OUT-OF-DISTRIBUTION · OPEN-SET GATE TRIGGERED`.
   - Consensus badge switches to `⚠️ Unrecognized / Outside 10 Classes`.
   - Main title renders **`Unrecognized Dish or Object` (`अज्ञात परिकार`)** instead of falsely claiming a Nepali dish name, while noting the closest softmax fallback and its percentage in the subtitle.
2. **Automatic Action Suppression:**
   - The collapsible preparation drawer (`📖 Dish Info & Prep`) and the `📍 Find Near Me` Google Maps button are automatically hidden so the app never displays a Momo or Selroti recipe for an unrecognized object.
3. **Diagnostic Trigger Pills:**
   - Renders red/amber diagnostic pills listing every triggered gate condition (e.g., `⚠️ Low Top-1 Confidence`, `⚠️ High Predictive Entropy`, `⚠️ Complete 3-Model Disagreement`) alongside live `Entropy H · Margin · Max Logit` telemetry.

---

## 9. Cloud Deployment on Microsoft Azure Container Apps (`eastasia`)

### Subscription, Tenant & Regional Policy Constraints
- **Subscription:** `Azure for Students` (`e584cdc9-1792-421e-aa2b-a42255e21fa9`)
- **Tenant:** `Kathmandu University (student.ku.edu.np)`
- **Azure Regional Policy Discovery:**
  - Initial attempt to provision Azure Container Registry (`parikarsijan12`) in `centralindia` failed with `RequestDisallowedByAzure`.
  - Querying subscription policy assignments (`az policy assignment list --query "[].parameters" -o json`) revealed the exact list of allowed regions for the student subscription:
    `["indonesiacentral", "indiasouthcentral", "malaysiawest", "japaneast", "eastasia"]`
  - Selected **`eastasia` (Hong Kong)** as the deployment region for all resources.

### Why Local Docker Build + Push Was Used Instead of `az acr build`
- Because `.gitignore` excludes `models_onnx/` (`~300 MB` across `convnext_tiny.onnx`, `efficientnet_v2_s.onnx`, `swin_t.onnx`, and `semantic_gate.onnx`) and the raw dataset folders, GitHub-only deployment (`App Service -> GitHub`) cannot access the model weights.
- Furthermore, on `Azure for Students` subscriptions, cloud-side ACR Tasks (`az acr build`) is blocked (`TasksOperationsNotAllowed`).
- **Solution:**
  1. Created `.dockerignore` to exclude `experiments/**/*.pt` and the `10,025` training images while explicitly keeping the `10` pinned showcase photos (`!chatpate/000035.jpg`, `!chiya/chiya_00041.jpg`, etc.) and `models_onnx/` (`318 MB` total build context).
  2. Built the `linux/amd64` container image locally via Docker Desktop and pushed directly to Azure Container Registry (`parikarsijan12.azurecr.io`), which is fully permitted on Azure for Students.

### Provisioned Azure Infrastructure (`parikar-rg`)
| Resource Name | Azure Resource Type | Region | Configuration / Specs |
| :--- | :--- | :---: | :--- |
| **`parikar-rg`** | `Microsoft.Resources/resourceGroups` | `eastasia` | Primary project resource group |
| **`parikarsijan12`** | `Microsoft.ContainerRegistry/registries` | `eastasia` | `Basic` SKU (`parikarsijan12.azurecr.io`, Admin Enabled) |
| **`workspace-parikarrg880Q`** | `Microsoft.OperationalInsights/workspaces` | `eastasia` | Log Analytics workspace for container telemetry |
| **`parikar-env`** | `Microsoft.App/managedEnvironments` | `eastasia` | Azure Container Apps managed environment |
| **`parikar-app`** | `Microsoft.App/containerApps` | `eastasia` | `2.0 vCPU` / `4.0 GiB RAM`, External HTTPS Ingress on Port `8000` |

- **Live Production URL:** **[https://parikar-app.livelystone-a6619db0.eastasia.azurecontainerapps.io/](https://parikar-app.livelystone-a6619db0.eastasia.azurecontainerapps.io/)**
- **Live Swagger API Docs:** **[https://parikar-app.livelystone-a6619db0.eastasia.azurecontainerapps.io/docs](https://parikar-app.livelystone-a6619db0.eastasia.azurecontainerapps.io/docs)**

### Beginner-Friendly Conceptual Breakdown (What Server & Services Were Used)
1. **What Server & OS Is Running Our App?**
   - Our app runs inside a **Linux (`Debian Python 3.11-slim`, `amd64`) container** hosted on **Microsoft Azure Container Apps** (`parikar-app`) in the **East Asia (`eastasia` — Hong Kong)** data center.
   - Inside that Linux server, **`uvicorn`** runs our **FastAPI** backend (`backend/app.py`) on port `8000`, which serves both the static frontend (`frontend/index.html`, `styles.css`, `app.js`) and the `/api/predict` ONNX Runtime inference API over automatic `https://`.
2. **What Are the 3 Azure Services We Created?**
   - **Resource Group (`parikar-rg`):** A logical folder in Azure that groups all parts of Parikar 2.0 together.
   - **Azure Container Registry (`parikarsijan12.azurecr.io`):** Your private cloud storage vault where the packaged Docker image (`parikar:v2`, containing your code + 4 ONNX models + 10 sample dish photos) is stored.
   - **Azure Container App (`parikar-app` inside `parikar-env`):** The serverless cloud compute engine (`2.0 vCPU`, `4.0 GiB RAM`) that pulls the image from `parikarsijan12.azurecr.io` and runs it on a public `https://` URL.
3. **How Pay-As-You-Go & Scale-to-Zero (`--min-replicas 0`) Works on Azure Container Apps:**
   - **With `--min-replicas 0` (Scale-to-Zero / Sleep Mode):** When nobody visits the website for ~5 minutes, Azure automatically scales the container down to `0` replicas (sleep mode). While sleeping, compute charges are **$0.00** (preserving your `$100` Azure for Students credit). When a user opens the link after idle time, Azure wakes up the container in ~8–12 seconds, runs `load_models()` (which now pre-warms all 4 ONNX sessions with a dummy tensor at startup), and then serves subsequent requests in `< 300 ms`.
   - **Monthly Free Grant:** Azure Container Apps also includes `180,000 vCPU-seconds` and `360,000 GiB-seconds` free every month per subscription.

### How to Push Future Code Updates to Azure (No Need to Recreate Resources)
Whenever code or frontend files change in the future:
```powershell
cd c:\Users\sijan\Desktop\parikar_final_nepali_food_datasets
az acr login --name parikarsijan12
docker build -t parikarsijan12.azurecr.io/parikar:v2 .
docker push parikarsijan12.azurecr.io/parikar:v2
az containerapp update --name parikar-app --resource-group parikar-rg --image parikarsijan12.azurecr.io/parikar:v2 --cpu 2.0 --memory 4.0Gi
```

### Cloud Latency & Frontend State Optimization (`v2` Image Update)
- **Root Cause of Stale Class Flash During Cloud Upload:**
  - It was a combination of **cloud network/CPU processing time (~0.8–1.5s)** and **frontend state order**: on localhost (`127.0.0.1:8000`), network transfer was `0 ms`, so the result card updated almost instantaneously. Over the internet to `eastasia`, uploading an image and running 4 ONNX models on cloud CPU takes ~1 second. Because `handleSelectedFile()` updated the preview thumbnail immediately without clearing the previous verdict card (`Momo`), the UI briefly displayed `Momo` alongside the newly uploaded photo until the cloud response arrived.
- **Fixes Applied:**
  1. **Immediate Processing State (`setLoadingVerdictState()` in `frontend/app.js`):** Clears the previous dish title/badge and hides the old model comparison panel at `0 ms` upon file selection or sample click, displaying an animated spinner (`Processing Image...` / `पहिचान हुँदैछ...`) until the new response arrives.
  2. **Client-Side Canvas Image Compression (`compressImageFile()` in `frontend/app.js`):** Resizes uploaded images in the browser to `<= 512px` JPEG (`~30 KB`) before `POST /api/predict`, cutting internet upload time to `eastasia` by `10×–30×` without affecting the `224×224` model input.
  3. **Scaled Container Compute to `2.0 vCPU / 4.0 GiB RAM`:** Doubled CPU cores on `parikar-app` to cut cloud ONNX Runtime inference time in half.

---

## 10. Chronological Action & Experiment Log
- **[2026-11-17]** Audited workspace structure (`10,025` raw images across `10` classes).
- **[2026-11-17]** Verified `hydro` Anaconda environment: `Python 3.11.15`, `PyTorch 2.13.0+cu130`, `Torchvision 0.28.0+cu130` on **NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)**.
- **[2026-11-17]** Ran `src/prepare_dataset.py` across all `10,025` files:
  - Found `0` corrupted images.
  - Caught and filtered **`121` exact MD5 byte-for-byte duplicate images** *before* splitting so identical images never leak across train/val/test.
  - Produced `9,904` unique verified images split stratified 80/10/10 into `splits/train.csv` (`7,924`), `splits/val.csv` (`990`), and `splits/test.csv` (`990`).
- **[2026-11-17]** Built two-stage transfer learning & evaluation script (`src/train_and_eval.py`) supporting `convnext_tiny`, `efficientnet_v2_s`, `swin_t`, and `vit_b_16` with `bfloat16` mixed precision and full evaluation metrics.
- **[2026-11-17]** Trained & evaluated Model 1 (`convnext_tiny`) for 6 epochs on RTX 4050 GPU: achieved **`99.29%` Val Top-1 (`99.12%` Val Macro-F1)** and **`99.49%` Test Top-1 (`99.51%` Test Macro-F1, `100%` F1 on minority classes `chatpate` and `yomari`)**.
- **[2026-11-17]** Trained & evaluated Model 2 (`efficientnet_v2_s`) for 6 epochs on RTX 4050 GPU: peaked at Epoch 5 with **`98.99%` Val Top-1 (`99.01%` Val Macro-F1)** and achieved **`99.39%` Test Top-1, `100.00%` Test Top-3, `99.24%` Test Macro-F1, and `18.70 ms/img` latency**.
- **[2026-11-17]** Trained & evaluated Model 3 (`swin_t`) for 6 epochs on RTX 4050 GPU: peaked at Epoch 3 with **`98.69%` Val Top-1 (`98.57%` Val Macro-F1)** and achieved **`97.88%` Test Top-1 (`97.98%` Test Macro-F1)**, empirically confirming that convolutional inductive bias (`ConvNeXt-Tiny` / `EfficientNetV2-S`) outperforms self-attention on fine-grained food textures (`momo` vs. `samosa` vs. `yomari`) at the `10k`-image scale.
- **[2026-11-17]** Generated side-by-side benchmark visualization (`experiments/model_comparison_benchmark.png`).
- **[2026-11-17]** Exported all 3 models to ONNX Runtime (`src/export_onnx.py` $\to$ `models_onnx/onnx_manifest.json`) and verified numerical equivalence (`max_diff <= 1.13e-6` vs. PyTorch):
  - `convnext_tiny.onnx`: `106.24 MB` | CPU Latency: **`27.01 ms`**
  - `efficientnet_v2_s.onnx`: `76.87 MB` | CPU Latency: **`18.45 ms`**
  - `swin_t.onnx`: `107.29 MB` | CPU Latency: **`35.75 ms`**
- **[2026-11-17]** Built FastAPI multi-model inference backend (`backend/app.py`), Azure `Dockerfile`, and bespoke archival/editorial web interface (`frontend/index.html`, `frontend/styles.css`, `frontend/app.js`) featuring live camera capture, 1-click test specimen strip, single/all-3 model toggles, live latency telemetry, and cultural culinary cards, credited to **Sijan Pandey**.
- **[2026-11-17]** Designed, calibrated, and implemented the **3-Layer Open-Set / Out-of-Distribution (OOD) Rejection Gate** (`evaluate_ood_gate` in `backend/app.py` + dual plain-English and collapsible technical breakdown UI in `frontend/app.js` and `frontend/styles.css`).
- **[2026-11-18]** Containerized and deployed **Parikar 2.0** to **Microsoft Azure Container Apps** (`eastasia` region, `parikar-rg` resource group, `parikarsijan12.azurecr.io/parikar:v1`, `1.0 vCPU / 2.0 GiB RAM`):
  - **Live Cloud URL:** `https://parikar-app.livelystone-a6619db0.eastasia.azurecontainerapps.io/`




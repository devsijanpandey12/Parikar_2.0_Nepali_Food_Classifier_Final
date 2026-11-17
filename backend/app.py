import io
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT_DIR / "models_onnx"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
SPLITS_DIR = ROOT_DIR / "splits"
FRONTEND_DIR = ROOT_DIR / "frontend"

CLASSES = [
    "chatpate", "chiya", "dalbhat", "jeri", "momo",
    "pakauda", "panipuri", "samosa", "selroti", "yomari"
]

# Curated, concise, non-generic culinary field notes for each of the 10 classes
DISH_CATALOG: Dict[str, Dict] = {
    "momo": {
        "name": "Momo",
        "nepali": "म:म",
        "category": "Dumpling · Staple Street & Festive Dish",
        "tagline": "Hand-pleated dumplings steamed over aromatic broth and served with fiery tomato-timur achar.",
        "preparation": "Finely minced meat, paneer, or spiced cabbage-onion filling wrapped in thin unleavened wheat dough, pleated into crescents or pouches, and steamed in a tiered muktoo.",
        "specialty": "The signature dipping sauce (achar)—roasted tomatoes, Sichuan pepper (timur), sesame, and chili—plus variations like Jhol Momo (drowned in tangy spiced broth) and Kothey (pan-seared).",
        "flavor_tags": ["Savory", "Timur Pepper", "Steamed / Pan-Seared", "Sesame Achar"]
    },
    "dalbhat": {
        "name": "Dal Bhat",
        "nepali": "दाल भात तरकारी",
        "category": "Traditional Thali · National Everyday Meal",
        "tagline": "The cornerstone of Nepali cuisine—steamed rice surrounded by spiced lentil soup, seasonal tarkari, greens, and achar.",
        "preparation": "Slow-simmered black or yellow lentils (dal) tempered with jimbu, garlic, and ghee, plated on a brass thali alongside fluffy rice (bhat), curried vegetables (tarkari), sautéed mustang greens (saag), and crispy papad.",
        "specialty": "Celebrated across the Himalayas with the phrase 'Dal Bhat Power, 24 Hour'—a complete balance of carbohydrates, plant protein, probiotic pickles, and mountain herbs.",
        "flavor_tags": ["Jimbu Tempered", "Brass Thali", "Wholesome", "Multi-Component"]
    },
    "selroti": {
        "name": "Selroti",
        "nepali": "सेलरोटी",
        "category": "Traditional Ring Bread · Tihar & Dashain Essential",
        "tagline": "Crisp-on-the-outside, soft-on-the-inside homemade rice-flour ring bread fried to a deep mahogany gold.",
        "preparation": "Coarsely ground soaked rice is stone-blended with ghee, sugar, cardamom, and ripe banana into a pourable batter, rested, and hand-poured in a continuous ring into hot oil.",
        "specialty": "Naturally gluten-free and central to Tihar and weddings; its porous crust and delicate sweet cardamom crumb pair effortlessly with spicy alu achar or hot chiya.",
        "flavor_tags": ["Rice Flour", "Cardamom & Ghee", "Festive Ring", "Crisp & Chewy"]
    },
    "yomari": {
        "name": "Yomari",
        "nepali": "योमरी",
        "category": "Newari Heritage Delicacy · Yomari Punhi Harvest Dish",
        "tagline": "Sculpted fig-shaped steamed rice-dough dumpling filled with molten chaku (caramelized molasses) and toasted sesame.",
        "preparation": "Freshly harvested new-season rice flour is kneaded with boiling water into a supple dough, hand-pinched into a hollow cone with a pointed tail, filled with chaku or khuwa (milk solids), and steamed.",
        "specialty": "Prepared by the Newar community of the Kathmandu Valley during the December full moon (Yomari Punhi) to honor Annapurna, the goddess of grains, and ward off winter cold.",
        "flavor_tags": ["Newari Heritage", "Molten Chaku", "Toasted Sesame", "Winter Harvest"]
    },
    "chatpate": {
        "name": "Chatpate",
        "nepali": "चटपटे",
        "category": "Street Snack · Tangy & Fiery Crunch",
        "tagline": "An electric, tossed-to-order street mix of puffed rice, crushed dry noodles, roasted peas, fresh herbs, and raw mustard oil.",
        "preparation": "Bhuja (puffed rice) and crunchy uncooked Wai-Wai noodles are tossed in a deep steel bowl with boiled potato cubes, red onion, cilantro, green chili, lime juice, chat masala, and pungent Tori ko Tel (mustard oil).",
        "specialty": "Served in a handmade paper cone; the sharp sinus-clearing kick of raw mustard oil combined with citrus and crunch makes it Nepal's most addictive afternoon street snack.",
        "flavor_tags": ["Mustard Oil Kick", "Crunchy Bhuja", "Lime & Chili", "Street Cone"]
    },
    "jeri": {
        "name": "Jeri",
        "nepali": "जेरी",
        "category": "Traditional Confection · Morning Sweet",
        "tagline": "Crisp, pretzel-like fermented wheat spirals fried until golden and soaked in warm saffron-cardamom sugar syrup.",
        "preparation": "Fermented maida batter is piped through a cloth nozzle in concentric coils into hot oil/ghee, fried crisp, and immediately submerged in hot sugar syrup so the hollow tubes lock in liquid sweetness.",
        "specialty": "In Nepal, warm Jeri is famously eaten for breakfast wrapped inside flaky Swari (flatbread) alongside a steaming glass of milk chiya.",
        "flavor_tags": ["Syrup-Soaked", "Crisp Spiral", "Saffron & Cardamom", "Jeri-Swari Classic"]
    },
    "samosa": {
        "name": "Samosa",
        "nepali": "समोसा",
        "category": "Savory Pastry · Teahouse Classic",
        "tagline": "Flaky tetrahedral pastry crust packed with roasted cumin, green peas, and spiced mashed potatoes.",
        "preparation": "Shortcrust dough rubbed with oil and ajwain (carom seeds) is folded into a three-dimensional cone, stuffed with spiced potato-pea masala, sealed, and slow-fried at medium heat for a blistered, flaky shell.",
        "specialty": "A staple of every Nepali mithai pasal (sweet shop), served crushed with chickpea tarkari and tangy tamarind-mint chutney.",
        "flavor_tags": ["Flaky Pastry", "Ajwain & Cumin", "Potato-Pea Masala", "Teahouse Staple"]
    },
    "panipuri": {
        "name": "Panipuri",
        "nepali": "पानीपुरी",
        "category": "Street Food · Crisp Hollow Crisps & Spiced Water",
        "tagline": "Bite-sized crisp semolina spheres cracked open, stuffed with spiced potato-chickpea mash, and brimmed with tangy tamarind-mint pani.",
        "preparation": "Paper-thin semolina-wheat rounds are flash-fried until they puff into hollow globes, then thumb-cracked and filled with mashedalu, black chickpeas, timur, and chilled spiced water.",
        "specialty": "Eaten in a single bite straight from the vendor's hand so the crisp shell, warm potato mash, and cold sour-spicy water burst together.",
        "flavor_tags": ["Single-Bite Burst", "Tamarind & Mint", "Crisp Semolina", "Street Favorite"]
    },
    "pakauda": {
        "name": "Pakauda",
        "nepali": "पकौडा",
        "category": "Spiced Fritter · Monsoon & Teatime Snack",
        "tagline": "Craggy, golden-fried gram-flour fritters loaded with sliced onions, spinach, potato, and whole spices.",
        "preparation": "Julienne onions, spinach leaves, and potato slivers are tossed with besan (chickpea flour), rice flour (for extra crunch), ajwain, turmeric, and green chilies, then dropped in loose clusters into hot oil.",
        "specialty": "The irregular, lacy edges fry up shatteringly crisp while the center stays tender—traditionally paired with afternoon masala chiya when monsoon rain hits.",
        "flavor_tags": ["Besan & Onion", "Lacy Crisp Edges", "Ajwain Spiced", "Chiya Companion"]
    },
    "chiya": {
        "name": "Chiya",
        "nepali": "चिया",
        "category": "Traditional Beverage · Daily Social Ritual",
        "tagline": "Rich, slow-boiled Nepali milk tea infused with crushed green cardamom, fresh ginger, cloves, and black pepper.",
        "preparation": "Crush fresh ginger and whole masala spices into simmering water, add strong CTC Assam/Ilam black tea leaves, pour in whole milk, and bring to three rolling boils until caramelized and velvety.",
        "specialty": "More than a beverage—Chiya is Nepal's social glue, served in small glass tumblers at street corners (chiya pasal) from dawn until dusk.",
        "flavor_tags": ["Ilam CTC Tea", "Crushed Ginger & Cardamom", "Caramelized Milk", "Daily Ritual"]
    }
}

MODEL_DISPLAY_NAMES = {
    "convnext_tiny": {
        "title": "ConvNeXt-Tiny",
        "family": "Modernized 7×7 Depthwise CNN",
        "badge": "Primary Hero · Best Overall",
        "notes": "Highest test accuracy (99.49%) & 100% F1 on minority classes (chatpate, yomari)."
    },
    "efficientnet_v2_s": {
        "title": "EfficientNetV2-S",
        "family": "Fused-MBConv Progressive CNN",
        "badge": "Compact Cloud Baseline",
        "notes": "Smallest footprint (20.2M params) with 100.00% Top-3 test accuracy."
    },
    "swin_t": {
        "title": "Swin-Tiny",
        "family": "Hierarchical Shifted-Window ViT",
        "badge": "Vision Transformer Benchmark",
        "notes": "Shifted-window self-attention benchmark (97.88% test Top-1, best at Epoch 3)."
    }
}

app = FastAPI(
    title="Parikar 2.0 — Nepali Culinary Vision Lab API",
    description="Multi-model ONNX Runtime inference engine for 10 Nepali dishes & beverages by Sijan Pandey",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX sessions at startup
ORT_SESSIONS: Dict[str, ort.InferenceSession] = {}
ONNX_MANIFEST: Dict = {}


def load_models():
    global ONNX_MANIFEST
    manifest_path = ONNX_DIR / "onnx_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            ONNX_MANIFEST = json.load(f)

    for arch in ["convnext_tiny", "efficientnet_v2_s", "swin_t"]:
        onnx_file = ONNX_DIR / f"{arch}.onnx"
        if onnx_file.exists():
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 4
            ORT_SESSIONS[arch] = ort.InferenceSession(
                str(onnx_file), sess_options=sess_opts, providers=["CPUExecutionProvider"]
            )


@app.on_event("startup")
def on_startup():
    load_models()


def preprocess_pil_image(img: Image.Image, img_size: int = 224) -> np.ndarray:
    img = img.convert("RGB")
    # Match torchvision eval_transform: Resize(255) -> CenterCrop(224) -> ToTensor -> Normalize
    target_resize = int(img_size * 1.14)
    w, h = img.size
    if w < h:
        new_w = target_resize
        new_h = int(target_resize * h / w)
    else:
        new_h = target_resize
        new_w = int(target_resize * w / h)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return np.expand_dims(arr, axis=0)  # NCHW


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def run_single_model_inference(arch: str, input_tensor: np.ndarray) -> Dict:
    if arch not in ORT_SESSIONS:
        load_models()
    if arch not in ORT_SESSIONS:
        raise HTTPException(status_code=503, detail=f"ONNX model '{arch}' is not loaded yet.")

    sess = ORT_SESSIONS[arch]
    t0 = time.perf_counter()
    logits = sess.run(["logits"], {"input": input_tensor})[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = softmax_np(logits)[0]
    top_indices = np.argsort(probs)[::-1]

    top_predictions = []
    for idx in top_indices[:3]:
        cls_key = CLASSES[int(idx)]
        top_predictions.append({
            "class_key": cls_key,
            "name": DISH_CATALOG[cls_key]["name"],
            "nepali": DISH_CATALOG[cls_key]["nepali"],
            "confidence": round(float(probs[idx]) * 100.0, 2)
        })

    all_probs = {
        CLASSES[i]: round(float(probs[i]) * 100.0, 2) for i in range(len(CLASSES))
    }

    winner_key = top_predictions[0]["class_key"]
    meta = ONNX_MANIFEST.get(arch, {})
    disp = MODEL_DISPLAY_NAMES.get(arch, {})

    return {
        "arch": arch,
        "title": disp.get("title", arch),
        "family": disp.get("family", ""),
        "badge": disp.get("badge", ""),
        "parameters": meta.get("parameters", 0),
        "size_mb": meta.get("size_mb", 0),
        "benchmark_test_top1": round(meta.get("test_metrics", {}).get("top1_accuracy", 0) * 100, 2),
        "benchmark_test_macro_f1": round(meta.get("test_metrics", {}).get("macro_f1", 0) * 100, 2),
        "latency_ms": round(latency_ms, 2),
        "predicted_class": winner_key,
        "predicted_name": DISH_CATALOG[winner_key]["name"],
        "predicted_nepali": DISH_CATALOG[winner_key]["nepali"],
        "confidence": top_predictions[0]["confidence"],
        "top3": top_predictions,
        "all_probabilities": all_probs
    }


@app.get("/api/info")
def get_lab_info():
    if not ONNX_MANIFEST:
        load_models()

    models_list = []
    for arch in ["convnext_tiny", "efficientnet_v2_s", "swin_t"]:
        meta = ONNX_MANIFEST.get(arch, {})
        disp = MODEL_DISPLAY_NAMES.get(arch, {})
        models_list.append({
            "arch": arch,
            "title": disp.get("title", arch),
            "family": disp.get("family", ""),
            "badge": disp.get("badge", ""),
            "notes": disp.get("notes", ""),
            "parameters": meta.get("parameters", 0),
            "size_mb": meta.get("size_mb", 0),
            "best_epoch": meta.get("best_epoch", 0),
            "cpu_latency_ms": meta.get("cpu_latency_ms", 0),
            "test_metrics": meta.get("test_metrics", {}),
            "confusion_matrix_url": f"/experiments/{arch}/test_confusion_matrix.png"
        })

    return {
        "project": "Parikar 2.0 — Nepali Culinary Vision Lab",
        "author": "Sijan Pandey",
        "github": "https://github.com/devsijanpandey12/Parikar_2.0_Nepali_Food_Classifier_Final",
        "dataset_summary": {
            "total_scanned": 10025,
            "exact_duplicates_removed": 121,
            "total_unique": 9904,
            "train_split": 7924,
            "val_split": 990,
            "test_split": 990,
            "classes_count": 10
        },
        "models": models_list,
        "dishes": DISH_CATALOG
    }


@app.get("/api/samples")
def get_sample_specimens():
    test_csv = SPLITS_DIR / "test.csv"
    if not test_csv.exists():
        return {"samples": []}

    import pandas as pd
    df = pd.read_csv(test_csv)
    samples = []
    for cls_key in CLASSES:
        sub = df[df["class_name"] == cls_key]
        if len(sub) > 0:
            row = sub.iloc[0]
            samples.append({
                "class_key": cls_key,
                "name": DISH_CATALOG[cls_key]["name"],
                "nepali": DISH_CATALOG[cls_key]["nepali"],
                "rel_path": row["path"],
                "image_url": f"/api/specimen-image?path={row['path']}"
            })
    return {"samples": samples}


@app.get("/api/specimen-image")
def get_specimen_image(path: str):
    clean_path = (ROOT_DIR / path).resolve()
    if not str(clean_path).startswith(str(ROOT_DIR.resolve())) or not clean_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(str(clean_path))


@app.post("/api/predict")
async def predict_image(
    model_choice: str = Form("all"),
    sample_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    if file is not None:
        raw_bytes = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    elif sample_path:
        clean_path = (ROOT_DIR / sample_path).resolve()
        if not str(clean_path).startswith(str(ROOT_DIR.resolve())) or not clean_path.exists():
            raise HTTPException(status_code=404, detail="Sample path not found")
        pil_img = Image.open(clean_path)
    else:
        raise HTTPException(status_code=400, detail="Provide either an image file or a sample_path.")

    input_tensor = preprocess_pil_image(pil_img, img_size=224)

    if model_choice == "all":
        archs_to_run = ["convnext_tiny", "efficientnet_v2_s", "swin_t"]
    else:
        if model_choice not in ["convnext_tiny", "efficientnet_v2_s", "swin_t"]:
            raise HTTPException(status_code=400, detail=f"Unknown model_choice: {model_choice}")
        archs_to_run = [model_choice]

    results = [run_single_model_inference(a, input_tensor) for a in archs_to_run]

    # Determine primary prediction (ConvNeXt-Tiny if 'all' is run, or the single selected model)
    primary = results[0]
    dish_info = DISH_CATALOG[primary["predicted_class"]]

    # Check agreement across models if multiple ran
    predicted_classes = [r["predicted_class"] for r in results]
    unanimous = len(set(predicted_classes)) == 1

    return {
        "mode": model_choice,
        "unanimous_agreement": unanimous,
        "primary_prediction": primary,
        "dish_profile": dish_info,
        "model_runs": results
    }


# Mount experiments folder for confusion matrix & benchmark charts
if EXPERIMENTS_DIR.exists():
    app.mount("/experiments", StaticFiles(directory=str(EXPERIMENTS_DIR)), name="experiments")

# Mount frontend static files at root (after /api routes) so relative paths work on both FastAPI and GitHub Pages
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


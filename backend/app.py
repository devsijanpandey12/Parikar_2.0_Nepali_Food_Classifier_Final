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

# Structured, visual, scannable culinary data (zero boring walls of text!)
DISH_CATALOG: Dict[str, Dict] = {
    "momo": {
        "name": "Momo",
        "nepali": "म:म",
        "category": "Steamed Dumpling · Street & Family Icon",
        "tagline": "Juicy hand-pleated dumplings steamed over aromatic broth and served with fiery tomato-timur achar.",
        "quick_facts": {
            "taste": "Savory, juicy & numbing timur spice",
            "method": "Steamed in tiered muktoo (12 mins)",
            "pairing": "Roasted tomato-sesame jhol achar",
            "vibe": "Nepal's #1 comfort food anywhere, anytime"
        },
        "steps": [
            "Knead thin unleavened wheat wrappers and pack with spiced minced meat, paneer, or crunchy cabbage-onion filling.",
            "Pinch into intricate circular pleats or crescent pouches to lock in the aromatic juices.",
            "Steam over boiling broth and serve piping hot with roasted tomato, chili, and Sichuan pepper (timur) dipping sauce."
        ],
        "specialty": "Whether drowned in tangy spiced soup (Jhol Momo), pan-seared crisp (Kothey), or tossed in chili sauce (C-Momo), the magic is always in the house-ground timur achar.",
        "flavor_tags": ["Hand-Pleated", "Timur Pepper Kick", "Sesame Jhol", "Steamed / Seared"]
    },
    "dalbhat": {
        "name": "Dal Bhat",
        "nepali": "दाल भात तरकारी",
        "category": "Traditional Brass Thali · National Meal",
        "tagline": "Fluffy steamed rice surrounded by jimbu-tempered lentil soup, seasonal tarkari, sautéed greens, and fiery achar.",
        "quick_facts": {
            "taste": "Earthy, garlicky & deeply comforting",
            "method": "Slow-simmered dal + sizzling ghee tadka",
            "pairing": "Mustang saag, crispy papad & timur achar",
            "vibe": "“Dal Bhat Power, 24 Hour!” — Himalayan fuel"
        },
        "steps": [
            "Simmer black or yellow lentils until velvety, then temper with sizzling ghee, garlic, cumin, and aromatic mountain herb jimbu.",
            "Steam fragrant long-grain rice and surround it on a brass thali with spiced potato-cauliflower tarkari and sautéed mustard greens.",
            "Finish with fermented radish pickle (mula ko achar), fresh lime, and roasted papad—mixed by hand for the perfect bite."
        ],
        "specialty": "A complete nutritional powerhouse combining complex carbs, lentil protein, and probiotic pickles—refilled endlessly in every Nepali home and trekking teahouse.",
        "flavor_tags": ["Jimbu Ghee Tadka", "Brass Thali", "24-Hour Trekker Fuel", "Complete Meal"]
    },
    "selroti": {
        "name": "Selroti",
        "nepali": "सेलरोटी",
        "category": "Festive Ring Bread · Tihar & Dashain Classic",
        "tagline": "Crispy-edged, pillowy-centered homemade rice-flour ring bread fried to a deep golden mahogany.",
        "quick_facts": {
            "taste": "Lightly sweet, aromatic cardamom & ghee",
            "method": "Hand-poured ring in hot oil/ghee",
            "pairing": "Spicy alu-tama curry or hot masala chiya",
            "vibe": "The unmistakable aroma of Tihar festival"
        },
        "steps": [
            "Soak coarse rice overnight and stone-grind into a thick batter with ghee, sugar, crushed green cardamom, and overripe banana.",
            "Ferment briefly, then scoop by hand and pour in one deft, continuous circular motion directly into bubbling oil.",
            "Flip with a bamboo skewer (surya) until both sides develop a lacy, blistered golden crust while the core stays soft."
        ],
        "specialty": "Naturally gluten-free and shaped like a sacred garland; its crisp honeycomb exterior soaks up hot milk chiya or spicy potato achar like nothing else.",
        "flavor_tags": ["Stone-Ground Rice", "Cardamom & Ghee", "Tihar Festival", "Crisp Honeycomb"]
    },
    "yomari": {
        "name": "Yomari",
        "nepali": "योमरी",
        "category": "Newari Heritage Sweet · Winter Harvest Delicacy",
        "tagline": "Sculpted fig-shaped rice-dough dumpling filled with molten caramelized chaku and toasted sesame seeds.",
        "quick_facts": {
            "taste": "Rich, smoky molasses & nutty sesame",
            "method": "Hand-sculpted cone steamed over clay pot",
            "pairing": "Served piping hot on December full moon",
            "vibe": "Kathmandu Valley Newari culinary art"
        },
        "steps": [
            "Knead freshly harvested new-season rice flour with boiling water into a smooth, elastic snow-white dough.",
            "Coax a dough ball between oiled fingers into a thin-walled hollow cone with a signature pointed tail.",
            "Fill with molten chaku (dark sugarcane molasses) and toasted sesame—or sweetened khuwa—seal the crown, and steam until glossy."
        ],
        "specialty": "Crafted during Yomari Punhi (December full moon) to celebrate the end of the rice harvest; biting into a warm Yomari releases a burst of liquid caramel chaku.",
        "flavor_tags": ["Newari Heritage", "Molten Chaku Core", "Toasted Sesame", "Yomari Punhi"]
    },
    "chatpate": {
        "name": "Chatpate",
        "nepali": "चटपटे",
        "category": "Street Snack · Tangy, Fiery & Crunchy",
        "tagline": "An electric tossed-to-order street mix of puffed rice, crunchy Wai-Wai noodles, fresh herbs, lime, and raw mustard oil.",
        "quick_facts": {
            "taste": "Explosive sour, spicy & pungent",
            "method": "Flash-tossed in a deep steel bowl",
            "pairing": "Served in a newspaper cone on the go",
            "vibe": "Nepal's ultimate afternoon street craving"
        },
        "steps": [
            "Combine crisp bhuja (puffed rice), crushed dry spiced noodles, and roasted peas in a deep stainless-steel mixing bowl.",
            "Toss with diced boiled potatoes, red onion, fresh cilantro, green chilies, cucumber, and black rock salt (bire nun).",
            "Finish with a squeeze of fresh lime and a generous drizzle of raw Tori ko Tel (pungent mustard oil), then funnel into a paper cone."
        ],
        "specialty": "That unmistakable, sinus-tingling punch of raw Nepali mustard oil hitting fresh lime juice and crunchy noodles—eaten fast before the puffed rice softens!",
        "flavor_tags": ["Raw Mustard Oil", "Crunchy Wai-Wai", "Bire Nun & Lime", "Paper Cone Street Icon"]
    },
    "jeri": {
        "name": "Jeri",
        "nepali": "जेरी",
        "category": "Traditional Sweet · Morning Mithai Classic",
        "tagline": "Crisp, pretzel-like fermented wheat coils fried golden and drenched in warm saffron-cardamom sugar syrup.",
        "quick_facts": {
            "taste": "Crackling crisp shell & warm liquid honeyed syrup",
            "method": "Piped spirals fried & syrup-dunked",
            "pairing": "Wrapped inside flaky Swari with hot chiya",
            "vibe": "Classic Kathmandu morning breakfast treat"
        },
        "steps": [
            "Whip and ferment a smooth wheat-flour batter until lightly tangy and aerated.",
            "Pipe concentric pretzel-like spirals through a stitched cloth nozzle directly into hot ghee or oil until crisp and golden.",
            "Submerge immediately into warm cardamom-saffron sugar syrup for 30 seconds so the hollow coils Drink up the glossy sweetness."
        ],
        "specialty": "In Nepal, warm Jeri is famously folded inside a soft, flaky fried flatbread called Swari (Jeri-Swari)—contrasting crackling sweet syrup with savory bread.",
        "flavor_tags": ["Syrup-Filled Coils", "Saffron & Cardamom", "Jeri-Swari Breakfast", "Crackling Crisp"]
    },
    "samosa": {
        "name": "Samosa",
        "nepali": "समोसा",
        "category": "Savory Pastry · Mithai Pasal Staple",
        "tagline": "Flaky, blistered triangular pastry packed with roasted cumin, green peas, and spiced mashed potatoes.",
        "quick_facts": {
            "taste": "Savory, earthy cumin & flaky ajwain crust",
            "method": "Slow-fried at medium heat for flaky layers",
            "pairing": "Tangy tamarind chutney & chickpea tarkari",
            "vibe": "Every Nepali teahouse's golden centerpiece"
        },
        "steps": [
            "Rub wheat flour with warm oil and crushed ajwain (carom seeds) until crumbly, then knead into a stiff, flaky pastry dough.",
            "Sauté cubed potatoes and green peas with roasted cumin, coriander, ginger, fenugreek, and chili.",
            "Fold into a three-cornered pyramid, crimp the edge, and slow-fry in medium-low oil for 15 minutes until deeply blistered and crisp."
        ],
        "specialty": "Unlike rushed fast-food pastries, authentic Nepali halwai samosas are slow-fried so the crust stays shattering crisp for hours, served with warm chana tarkari.",
        "flavor_tags": ["Ajwain Shortcrust", "Roasted Cumin Potato", "Tamarind Chutney", "Halwai Classic"]
    },
    "panipuri": {
        "name": "Panipuri",
        "nepali": "पानीपुरी",
        "category": "Street Food · Crisp Globes & Spiced Water",
        "tagline": "Hollow semolina crisps cracked open, loaded with spiced potato-chickpea mash, and brimmed with icy tamarind-mint pani.",
        "quick_facts": {
            "taste": "Cold sour-spicy burst + warm savory mash",
            "method": "Flash-puffed puris filled to order by hand",
            "pairing": "Extra sukha puri dusted with chat masala",
            "vibe": "Interactive street-side flavor explosion"
        },
        "steps": [
            "Roll paper-thin semolina-wheat discs and flash-fry until they puff into crisp, hollow golden spheres.",
            "Mash boiled potatoes with black chickpeas, red onion, roasted cumin, chili flakes, and fresh coriander.",
            "Crack the crown of each puri with a thumb, stuff with spiced mash, dunk into chilled mint-tamarind-timur water, and eat in one bite."
        ],
        "specialty": "Nepali street vendors spike the potato mash with crushed timur (Sichuan pepper) and roasted dry chili, giving it a distinct Himalayan tingle.",
        "flavor_tags": ["One-Bite Burst", "Chilled Mint-Tamarind Pani", "Timur Potato Mash", "Street Ritual"]
    },
    "pakauda": {
        "name": "Pakauda",
        "nepali": "पकौडा",
        "category": "Spiced Fritter · Monsoon & Teatime Favorite",
        "tagline": "Craggy, golden-fried chickpea-flour fritters loaded with caramelized onion slivers, spinach, and whole spices.",
        "quick_facts": {
            "taste": "Nutty roasted gram flour & sweet-crispy onion",
            "method": "Loose clusters fried to lacy golden crunch",
            "pairing": "Tomato-coriander achar & steaming chiya",
            "vibe": "Rainy afternoon essential across Nepal"
        },
        "steps": [
            "Slice red onions, potato matchsticks, and fresh spinach leaves, then toss with salt so their natural moisture releases.",
            "Fold in besan (chickpea flour), a spoon of rice flour for extra crunch, ajwain seeds, turmeric, and chopped green chilies.",
            "Drop loose, jagged clusters into hot oil and fry until the outer tendrils turn deep mahogany and shatteringly crisp."
        ],
        "specialty": "Adding coarse rice flour and ajwain to the chickpea batter creates irregular, lacy edges that crunch loudly while keeping the center tender.",
        "flavor_tags": ["Lacy Onion Tendrils", "Besan & Rice Flour", "Ajwain Spiced", "Monsoon Companion"]
    },
    "chiya": {
        "name": "Chiya",
        "nepali": "चिया",
        "category": "Traditional Spiced Milk Tea · Daily Social Glue",
        "tagline": "Rich, slow-boiled Nepali milk tea infused with freshly crushed green cardamom, ginger, cloves, and Ilam black tea.",
        "quick_facts": {
            "taste": "Velvety, caramelized milk & warming ginger-cardamom",
            "method": "Triple-boiled in a brass or steel saucepan",
            "pairing": "Selroti, Marie biscuits, or hot samosa",
            "vibe": "Served in glass tumblers from dawn to dusk"
        },
        "steps": [
            "Pound fresh ginger root, green cardamom pods, cloves, and black peppercorns in a mortar and simmer in water until fragrant.",
            "Add bold CTC black tea leaves from Ilam and boil until a deep amber liquor develops.",
            "Pour in rich whole milk and sugar, then ladle and aerate through three rolling boils until creamy and ligeramente caramelized."
        ],
        "specialty": "In Nepal, asking 'Chiya khanu bhayo?' (Have you had tea?) is how friendships and conversations begin—poured high into small ridged glass tumblers.",
        "flavor_tags": ["Ilam CTC Tea", "Crushed Ginger & Cardamom", "Triple-Boiled Milk", "Chiya Pasal Culture"]
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

# Hand-picked default row offsets per class in test.csv so chiya, pakauda, and selroti display distinct, appetizing photos
PREFERRED_SAMPLE_OFFSETS = {
    "chatpate": 0,
    "chiya": 4,
    "dalbhat": 1,
    "jeri": 2,
    "momo": 0,
    "pakauda": 5,
    "panipuri": 1,
    "samosa": 2,
    "selroti": 6,
    "yomari": 1
}

app = FastAPI(
    title="Parikar 2.0 — Nepali Food Classifier API",
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
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


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
        "project": "Parikar 2.0 — Nepali Food Classifier",
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
def get_sample_specimens(offset: int = 0):
    test_csv = SPLITS_DIR / "test.csv"
    if not test_csv.exists():
        return {"samples": []}

    import pandas as pd
    df = pd.read_csv(test_csv)
    samples = []
    for cls_key in CLASSES:
        sub = df[df["class_name"] == cls_key]
        if len(sub) > 0:
            base_idx = PREFERRED_SAMPLE_OFFSETS.get(cls_key, 0)
            chosen_idx = (base_idx + offset) % len(sub)
            row = sub.iloc[chosen_idx]
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

    primary = results[0]
    dish_info = DISH_CATALOG[primary["predicted_class"]]
    predicted_classes = [r["predicted_class"] for r in results]
    unanimous = len(set(predicted_classes)) == 1

    return {
        "mode": model_choice,
        "unanimous_agreement": unanimous,
        "primary_prediction": primary,
        "dish_profile": dish_info,
        "model_runs": results
    }


if EXPERIMENTS_DIR.exists():
    app.mount("/experiments", StaticFiles(directory=str(EXPERIMENTS_DIR)), name="experiments")

FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

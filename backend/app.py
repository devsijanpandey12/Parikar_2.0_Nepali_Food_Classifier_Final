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

# Exact user-selected sample files per class (overrides test.csv default when specified)
PINNED_SAMPLE_FILES = {
    "chatpate": "chatpate/000035.jpg",
    "chiya": "chiya/chiya_00041.jpg",
    "dalbhat": "dalbhat/dalbhat_00034.jpg",
    "jeri": "jeri/jeri (16).jpg",
    "momo": "momo/momo_0058.jpg",
    "pakauda": "pakauda/200.jpg",
    "panipuri": "panipuri/483.jpg",
    "samosa": "samosa/125.jpg",
    "selroti": "selroti/selroti (325).jpg",
    "yomari": "yomari/000008.jpg",
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


@app.middleware("http")
async def disable_static_cache(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/index.html", "/app.js", "/styles.css"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


ORT_SESSIONS: Dict[str, ort.InferenceSession] = {}
ONNX_MANIFEST: Dict = {}
CONVNEXT_CENTROIDS: Optional[np.ndarray] = None
SEMANTIC_GATE_SESSION: Optional[ort.InferenceSession] = None
IMAGENET_CATEGORIES: List[str] = []
ALLOWED_IMAGENET_INDICES: List[int] = []
EXPLICIT_OUTSIDE_FOODS: set = set()
CONVNEXT_FEAT_NODE = "/classifier/classifier.1/Flatten_output_0"


def load_models():
    global ONNX_MANIFEST, CONVNEXT_CENTROIDS, SEMANTIC_GATE_SESSION
    global IMAGENET_CATEGORIES, ALLOWED_IMAGENET_INDICES, EXPLICIT_OUTSIDE_FOODS

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

    centroids_path = ONNX_DIR / "convnext_centroids.npy"
    if centroids_path.exists():
        CONVNEXT_CENTROIDS = np.load(str(centroids_path))

    sem_gate_path = ONNX_DIR / "semantic_gate.onnx"
    if sem_gate_path.exists():
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        SEMANTIC_GATE_SESSION = ort.InferenceSession(
            str(sem_gate_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

    cats_path = ONNX_DIR / "imagenet_categories.json"
    if cats_path.exists():
        with open(cats_path, "r", encoding="utf-8") as f:
            IMAGENET_CATEGORIES = json.load(f)

    allowed_path = ONNX_DIR / "allowed_imagenet_indices.json"
    if allowed_path.exists():
        with open(allowed_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            ALLOWED_IMAGENET_INDICES = meta.get("allowed_indices", [])
            EXPLICIT_OUTSIDE_FOODS = set(meta.get("explicit_western_foods", []))


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


# Empirically calibrated on splits/test.csv under Label Smoothing (epsilon = 0.1, K = 10):
# - Ideal in-distribution top-1 probability is ~91.0% (test median = 92.01%, p5 = 90.88%)
# - Ideal normalized Shannon entropy H(p)/ln(10) is ~0.217 (test median = 0.20, p95 = 0.22)
# - Ideal max pre-softmax logit is ~4.16 (test p5 = 4.02)
# - 768-D Feature Centroid Cosine Similarity median = 0.9822 (p5 = 0.9386)
# - Culinary Domain Mass in ImageNet-1K semantic space median = 93.89%
OOD_THRESHOLDS = {
    "min_confidence_pct": 78.0,
    "min_margin_pct": 60.0,
    "max_normalized_entropy": 0.36,
    "min_max_logit": 3.35,
    "min_centroid_cosine_sim": 0.85,
    "min_culinary_domain_mass_pct": 38.0,
    "max_non_culinary_top1_pct": 32.0,
    "max_outside_food_top1_pct": 22.0,
    "ensemble_split_min_mean_conf_pct": 86.0,
    "ensemble_split_min_single_conf_pct": 72.0,
    "ensemble_unanimous_min_mean_conf_pct": 80.0,
    "ensemble_unanimous_min_single_conf_pct": 68.0,
}


def evaluate_semantic_domain(input_tensor: np.ndarray) -> Dict:
    """
    Runs lightweight ImageNet-1K semantic verification (MobileNetV3-Small ONNX)
    to detect non-food objects (faces, laptops, phones, rooms, animals, vehicles)
    and explicit non-Nepali foods (pizza, cheeseburger, hotdog, ice cream, fruits).
    """
    if SEMANTIC_GATE_SESSION is None or not IMAGENET_CATEGORIES or not ALLOWED_IMAGENET_INDICES:
        return {
            "enabled": False,
            "is_domain_valid": True,
            "culinary_domain_mass_pct": 100.0,
            "top_imagenet_label": "unknown",
            "top_imagenet_conf_pct": 0.0,
            "domain_rejection_reason": None,
        }

    sem_logits = SEMANTIC_GATE_SESSION.run(["logits"], {"input": input_tensor})[0][0]
    sem_probs = softmax_np(sem_logits)
    top_idx = int(np.argmax(sem_probs))
    top_label = IMAGENET_CATEGORIES[top_idx]
    top_conf_pct = round(float(sem_probs[top_idx]) * 100.0, 2)
    culinary_mass_pct = round(float(np.sum(sem_probs[ALLOWED_IMAGENET_INDICES])) * 100.0, 2)

    reason = None
    nice_label = top_label.replace("_", " ").title()
    if top_label in EXPLICIT_OUTSIDE_FOODS and top_conf_pct >= OOD_THRESHOLDS["max_outside_food_top1_pct"]:
        reason = f"Looks like a different food item ({nice_label}) rather than one of our 10 Nepali dishes"
    elif top_idx not in ALLOWED_IMAGENET_INDICES and top_conf_pct >= OOD_THRESHOLDS["max_non_culinary_top1_pct"]:
        reason = f"Doesn't look like food — appears to be an everyday object or scene ({nice_label})"
    elif culinary_mass_pct < OOD_THRESHOLDS["min_culinary_domain_mass_pct"]:
        reason = "Doesn't look like a clear plate or cup of Nepali food"

    return {
        "enabled": True,
        "is_domain_valid": reason is None,
        "culinary_domain_mass_pct": culinary_mass_pct,
        "top_imagenet_label": top_label,
        "top_imagenet_conf_pct": top_conf_pct,
        "domain_rejection_reason": reason,
    }


def run_single_model_inference(arch: str, input_tensor: np.ndarray) -> Dict:
    if arch not in ORT_SESSIONS:
        load_models()
    if arch not in ORT_SESSIONS:
        raise HTTPException(status_code=503, detail=f"ONNX model '{arch}' is not loaded yet.")

    sess = ORT_SESSIONS[arch]
    output_names = [o.name for o in sess.get_outputs()]
    has_feat = arch == "convnext_tiny" and CONVNEXT_FEAT_NODE in output_names and CONVNEXT_CENTROIDS is not None

    t0 = time.perf_counter()
    if has_feat:
        logits_batch, feat_batch = sess.run(["logits", CONVNEXT_FEAT_NODE], {"input": input_tensor})
    else:
        logits_batch = sess.run(["logits"], {"input": input_tensor})[0]
        feat_batch = None
    latency_ms = (time.perf_counter() - t0) * 1000.0

    raw_logits = logits_batch[0]
    probs = softmax_np(logits_batch)[0]
    top_indices = np.argsort(probs)[::-1]

    # Uncertainty & Open-Set OOD telemetry
    top1_prob = float(probs[top_indices[0]])
    top2_prob = float(probs[top_indices[1]])
    confidence_pct = round(top1_prob * 100.0, 2)
    margin_pct = round((top1_prob - top2_prob) * 100.0, 2)
    norm_entropy = round(
        float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))) / np.log(len(CLASSES))),
        4
    )
    max_logit = round(float(np.max(raw_logits)), 3)
    # Helmholtz free energy score E(x) = -log sum_i exp(z_i) (Liu et al., NeurIPS 2020)
    shifted_l = raw_logits - np.max(raw_logits)
    energy_score = round(float(-(np.max(raw_logits) + np.log(np.sum(np.exp(shifted_l))))), 3)

    centroid_cos_sim = None
    if feat_batch is not None and CONVNEXT_CENTROIDS is not None:
        v = feat_batch[0]
        v_norm = v / (np.linalg.norm(v) + 1e-12)
        pred_idx = int(top_indices[0])
        centroid_cos_sim = round(float(np.dot(v_norm, CONVNEXT_CENTROIDS[pred_idx])), 4)

    model_is_ood = (
        confidence_pct < OOD_THRESHOLDS["min_confidence_pct"]
        or margin_pct < OOD_THRESHOLDS["min_margin_pct"]
        or norm_entropy > OOD_THRESHOLDS["max_normalized_entropy"]
        or max_logit < OOD_THRESHOLDS["min_max_logit"]
        or (centroid_cos_sim is not None and centroid_cos_sim < OOD_THRESHOLDS["min_centroid_cosine_sim"])
    )

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
        "confidence": confidence_pct,
        "margin": margin_pct,
        "normalized_entropy": norm_entropy,
        "max_logit": max_logit,
        "energy_score": energy_score,
        "centroid_cosine_sim": centroid_cos_sim,
        "is_ood": bool(model_is_ood),
        "top3": top_predictions,
        "all_probabilities": all_probs
    }


def evaluate_ood_gate(results: List[Dict], input_tensor: Optional[np.ndarray] = None) -> Dict:
    """
    3-Layer Multi-Signal Out-of-Distribution (OOD) / Open-Set Rejection Gate:
      Layer 1: Pretrained ImageNet-1K Semantic Domain Filter (catches non-food objects & outside Western foods)
      Layer 2: 768-D Penultimate Feature Prototype Cosine Similarity (Deep Nearest-Class-Mean in ConvNeXt-Tiny)
      Layer 3: Label-Smoothed Softmax Confidence, Top-1/Top-2 Margin, Normalized Shannon Entropy,
               Pre-Softmax Logit Energy, and Cross-Architecture (CNN vs. Swin-ViT) Consensus.
    """
    primary = results[0]
    reasons: List[str] = []

    sem_info = (
        evaluate_semantic_domain(input_tensor)
        if input_tensor is not None
        else {"enabled": False, "is_domain_valid": True, "culinary_domain_mass_pct": 100.0}
    )
    if sem_info.get("domain_rejection_reason"):
        reasons.append(sem_info["domain_rejection_reason"])

    if primary.get("centroid_cosine_sim") is not None and (
        primary["centroid_cosine_sim"] < OOD_THRESHOLDS["min_centroid_cosine_sim"]
    ):
        reasons.append("The visual shape, texture, and colors don't match any of our 10 Nepali dishes")

    if (
        primary["confidence"] < OOD_THRESHOLDS["min_confidence_pct"]
        or primary["max_logit"] < OOD_THRESHOLDS["min_max_logit"]
    ):
        reasons.append("Low match confidence — the AI is not sure enough to identify a Nepali dish here")

    if (
        primary["normalized_entropy"] > OOD_THRESHOLDS["max_normalized_entropy"]
        or primary["margin"] < OOD_THRESHOLDS["min_margin_pct"]
    ):
        reasons.append("Unclear match — no single Nepali dish clearly stands out in this photo")

    pred_classes = [r["predicted_class"] for r in results]
    distinct_preds = len(set(pred_classes))
    mean_conf = float(np.mean([r["confidence"] for r in results]))
    min_conf = float(np.min([r["confidence"] for r in results]))

    if len(results) > 1:
        if distinct_preds == len(results):
            reasons.append("All 3 AI models guessed different dishes and could not agree")
        elif distinct_preds > 1 and (
            mean_conf < OOD_THRESHOLDS["ensemble_split_min_mean_conf_pct"]
            or min_conf < OOD_THRESHOLDS["ensemble_split_min_single_conf_pct"]
        ):
            reasons.append("Our 3 AI models disagreed on what this photo is")
        elif (
            mean_conf < OOD_THRESHOLDS["ensemble_unanimous_min_mean_conf_pct"]
            or min_conf < OOD_THRESHOLDS["ensemble_unanimous_min_single_conf_pct"]
        ):
            reasons.append("Overall match strength across the 3 AI models was too low")

    is_recognized = len(reasons) == 0
    supported_list = "Chatpate, Chiya, Dal Bhat, Jeri, Momo, Pakauda, Panipuri, Samosa, Selroti, or Yomari"

    if is_recognized:
        summary_msg = f"Matched with {primary['predicted_name']}."
    else:
        summary_msg = (
            f"We couldn't match this photo to any of our 10 supported Nepali dishes ({supported_list}). "
            f"Here is why it was not recognized:"
        )

    return {
        "is_recognized": is_recognized,
        "is_ood": not is_recognized,
        "status": "recognized" if is_recognized else "unrecognized_ood",
        "summary_title": primary["predicted_name"] if is_recognized else "Unrecognized Dish or Object",
        "summary_nepali": primary["predicted_nepali"] if is_recognized else "अज्ञात परिकार",
        "summary_message": summary_msg,
        "reasons": reasons,
        "semantic_domain": sem_info,
        "metrics": {
            "primary_confidence": primary["confidence"],
            "primary_margin": primary["margin"],
            "primary_normalized_entropy": primary["normalized_entropy"],
            "primary_max_logit": primary["max_logit"],
            "primary_energy_score": primary["energy_score"],
            "primary_centroid_cosine_sim": primary.get("centroid_cosine_sim"),
            "culinary_domain_mass_pct": sem_info.get("culinary_domain_mass_pct"),
            "ensemble_mean_confidence": round(mean_conf, 2),
            "ensemble_min_confidence": round(min_conf, 2),
            "distinct_predictions_count": distinct_preds,
        },
        "thresholds": OOD_THRESHOLDS,
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
        pinned_rel = PINNED_SAMPLE_FILES.get(cls_key)
        if pinned_rel and (ROOT_DIR / pinned_rel).exists():
            from urllib.parse import quote
            samples.append({
                "class_key": cls_key,
                "name": DISH_CATALOG[cls_key]["name"],
                "nepali": DISH_CATALOG[cls_key]["nepali"],
                "rel_path": pinned_rel,
                "image_url": f"/api/specimen-image?path={quote(pinned_rel)}"
            })
            continue
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
    ood_gate = evaluate_ood_gate(results, input_tensor)

    return {
        "mode": model_choice,
        "unanimous_agreement": unanimous,
        "is_recognized": ood_gate["is_recognized"],
        "ood_gate": ood_gate,
        "primary_prediction": primary,
        "dish_profile": dish_info,
        "model_runs": results
    }


if EXPERIMENTS_DIR.exists():
    app.mount("/experiments", StaticFiles(directory=str(EXPERIMENTS_DIR)), name="experiments")

FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

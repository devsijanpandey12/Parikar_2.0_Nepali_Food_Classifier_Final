const API_BASE = "";

let selectedModel = "all";
let currentUploadFile = null;
let currentSamplePath = null;
let currentPreviewSrc = "";
let sampleShuffleOffset = 0;
let mediaStream = null;

// DOM Elements
const themeToggleBtn = document.getElementById("themeToggleBtn");
const themeIcon = document.getElementById("themeIcon");
const shuffleSamplesBtn = document.getElementById("shuffleSamplesBtn");
const modelToggles = document.querySelectorAll(".model-toggle");
const selectedModelHint = document.getElementById("selectedModelHint");
const tabUploadBtn = document.getElementById("tabUploadBtn");
const tabCameraBtn = document.getElementById("tabCameraBtn");
const uploadView = document.getElementById("uploadView");
const cameraView = document.getElementById("cameraView");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewWrapper = document.getElementById("previewWrapper");
const imagePreview = document.getElementById("imagePreview");
const dropzonePrompt = document.getElementById("dropzonePrompt");
const changePhotoBtn = document.getElementById("changePhotoBtn");

const cameraVideo = document.getElementById("cameraVideo");
const cameraCanvas = document.getElementById("cameraCanvas");
const cameraIdlePrompt = document.getElementById("cameraIdlePrompt");
const cameraControlsBar = document.getElementById("cameraControlsBar");
const startCameraBtn = document.getElementById("startCameraBtn");
const capturePhotoBtn = document.getElementById("capturePhotoBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");

const runInferenceBtn = document.getElementById("runInferenceBtn");
const specimenGrid = document.getElementById("specimenGrid");

// Output Elements
const consensusBadge = document.getElementById("consensusBadge");
const dishCategory = document.getElementById("dishCategory");
const dishName = document.getElementById("dishName");
const dishNepali = document.getElementById("dishNepali");
const floatingDishImg = document.getElementById("floatingDishImg");
const primaryModelLabel = document.getElementById("primaryModelLabel");
const primaryConfidence = document.getElementById("primaryConfidence");
const primaryLatency = document.getElementById("primaryLatency");
const dishTagline = document.getElementById("dishTagline");

const factTaste = document.getElementById("factTaste");
const factMethod = document.getElementById("factMethod");
const factPairing = document.getElementById("factPairing");
const factVibe = document.getElementById("factVibe");
const dishStepsRow = document.getElementById("dishStepsRow");
const dishSpecialty = document.getElementById("dishSpecialty");
const findNearMeBtn = document.getElementById("findNearMeBtn");
const dishFlavorTags = document.getElementById("dishFlavorTags");

const modelResultsContainer = document.getElementById("modelResultsContainer");
const totalInferenceTime = document.getElementById("totalInferenceTime");

// Matrix Viewer Tabs & Collapsible Drawer
const matrixTabs = document.querySelectorAll(".matrix-tab");
const matrixViewerImg = document.getElementById("matrixViewerImg");
const toggleMatrixDrawerBtn = document.getElementById("toggleMatrixDrawerBtn");
const matrixDrawerBtnText = document.getElementById("matrixDrawerBtnText");
const matrixDrawerContent = document.getElementById("matrixDrawerContent");

const toggleDishDetailsBtn = document.getElementById("toggleDishDetailsBtn");
const toggleDishDetailsText = document.getElementById("toggleDishDetailsText");
const dishCollapsibleBody = document.getElementById("dishCollapsibleBody");

if (toggleDishDetailsBtn && dishCollapsibleBody) {
  toggleDishDetailsBtn.addEventListener("click", () => {
    const isOpen = dishCollapsibleBody.classList.toggle("open");
    toggleDishDetailsBtn.classList.toggle("open", isOpen);
    toggleDishDetailsBtn.setAttribute("aria-expanded", String(isOpen));
    toggleDishDetailsText.textContent = isOpen ? "📖 Hide Dish Info" : "📖 Dish Info & Prep";
  });
}

if (toggleMatrixDrawerBtn && matrixDrawerContent) {
  toggleMatrixDrawerBtn.addEventListener("click", () => {
    const isOpen = matrixDrawerContent.classList.toggle("open");
    toggleMatrixDrawerBtn.classList.toggle("open", isOpen);
    toggleMatrixDrawerBtn.setAttribute("aria-expanded", String(isOpen));
    matrixDrawerBtnText.textContent = isOpen
      ? "📊 Hide Confusion Matrices & Convergence Curves"
      : "📊 Show Confusion Matrices & Convergence Curves";
  });
}

const MODEL_HINTS = {
  all: "Running all 3 architectures head-to-head",
  convnext_tiny: "Single model: ConvNeXt-Tiny (27.8M · 99.49% Test Top-1)",
  efficientnet_v2_s: "Single model: EfficientNetV2-S (20.2M · 100% Test Top-3)",
  swin_t: "Single model: Swin-Tiny Vision Transformer (27.5M)"
};

// 0. Theme Toggle & Scroll Reveal Observer
if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", () => {
    const html = document.documentElement;
    const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    themeIcon.textContent = next === "dark" ? "☀" : "☾";
  });
}

function initScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
        }
      });
    },
    { threshold: 0.12 }
  );

  document.querySelectorAll(".reveal-up, .arch-card").forEach((el) => observer.observe(el));
}

function animateNumber(el, targetValue, suffix = "%", duration = 460) {
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = (targetValue * eased).toFixed(2);
    el.textContent = `${current}${suffix}`;
    if (progress < 1) {
      requestAnimationFrame(tick);
    } else {
      el.textContent = `${targetValue.toFixed(2)}${suffix}`;
    }
  }
  requestAnimationFrame(tick);
}

// 1. Model Selector Toggles
modelToggles.forEach((btn) => {
  btn.addEventListener("click", () => {
    modelToggles.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    selectedModel = btn.dataset.model;
    selectedModelHint.textContent = MODEL_HINTS[selectedModel] || "";
    if (currentUploadFile || currentSamplePath) {
      executePrediction();
    }
  });
});

// 2. Upload vs Live Camera Tabs
tabUploadBtn.addEventListener("click", () => {
  tabUploadBtn.classList.add("active");
  tabCameraBtn.classList.remove("active");
  uploadView.classList.remove("hidden");
  cameraView.classList.add("hidden");
  stopCameraStream();
});

tabCameraBtn.addEventListener("click", () => {
  tabCameraBtn.classList.add("active");
  tabUploadBtn.classList.remove("active");
  cameraView.classList.remove("hidden");
  uploadView.classList.add("hidden");
});

// 3. File Upload & Drag-and-Drop
dropzone.addEventListener("click", (e) => {
  if (e.target === changePhotoBtn) return;
  fileInput.click();
});

changePhotoBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleSelectedFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    handleSelectedFile(fileInput.files[0]);
  }
});

function handleSelectedFile(file) {
  currentUploadFile = file;
  currentSamplePath = null;
  document.querySelectorAll(".specimen-thumb").forEach((t) => t.classList.remove("active"));

  const reader = new FileReader();
  reader.onload = (ev) => {
    currentPreviewSrc = ev.target.result;
    imagePreview.src = currentPreviewSrc;
    floatingDishImg.src = currentPreviewSrc;
    previewWrapper.classList.add("visible");
    dropzonePrompt.classList.add("hidden");
  };
  reader.readAsDataURL(file);
  executePrediction();
}

// 4. Live Camera Capture (WebRTC)
startCameraBtn.addEventListener("click", async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    cameraVideo.srcObject = mediaStream;
    cameraIdlePrompt.classList.add("hidden");
    cameraControlsBar.classList.remove("hidden");
  } catch (err) {
    alert("Could not access camera: " + err.message);
  }
});

function stopCameraStream() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  cameraVideo.srcObject = null;
  cameraIdlePrompt.classList.remove("hidden");
  cameraControlsBar.classList.add("hidden");
}

stopCameraBtn.addEventListener("click", stopCameraStream);

capturePhotoBtn.addEventListener("click", () => {
  if (!mediaStream) return;
  const w = cameraVideo.videoWidth || 640;
  const h = cameraVideo.videoHeight || 480;
  cameraCanvas.width = w;
  cameraCanvas.height = h;
  const ctx = cameraCanvas.getContext("2d");
  ctx.drawImage(cameraVideo, 0, 0, w, h);

  cameraCanvas.toBlob((blob) => {
    if (!blob) return;
    const capturedFile = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
    stopCameraStream();
    tabUploadBtn.click();
    handleSelectedFile(capturedFile);
  }, "image/jpeg", 0.92);
});

// 5. Run Inference Button
runInferenceBtn.addEventListener("click", () => {
  executePrediction();
});

// 6. Execute Prediction against FastAPI Backend
async function executePrediction() {
  if (!currentUploadFile && !currentSamplePath) {
    alert("Please select a sample dish above, upload a photo, or capture a frame from your camera first.");
    return;
  }

  runInferenceBtn.textContent = "Classifying...";
  runInferenceBtn.disabled = true;

  const formData = new FormData();
  formData.append("model_choice", selectedModel);
  if (currentUploadFile) {
    formData.append("file", currentUploadFile);
  } else if (currentSamplePath) {
    formData.append("sample_path", currentSamplePath);
  }

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText);
    }
    const data = await res.json();
    renderPredictionOutput(data);
  } catch (err) {
    console.error("Prediction error:", err);
    consensusBadge.textContent = "Inference Error";
  } finally {
    runInferenceBtn.textContent = "Run Classification →";
    runInferenceBtn.disabled = false;
  }
}

function renderPredictionOutput(data) {
  const primary = data.primary_prediction;
  const dish = data.dish_profile;
  const runs = data.model_runs || [];
  const ood = data.ood_gate || { is_recognized: true, reasons: [] };
  const isRecognized = data.is_recognized !== false && ood.is_recognized !== false;
  const culinaryCardEl = document.getElementById("culinaryCard");
  const compactBarRight = document.querySelector(".compact-bar-Right");
  const telemetryPanel = document.querySelector(".telemetry-panel");
  const confidenceMeterBox = document.querySelector(".confidence-meter-box");

  // Update floating dish image beside dish title
  if (currentPreviewSrc) {
    floatingDishImg.src = currentPreviewSrc;
  }

  if (!isRecognized) {
    // CLEAN, USER-FRIENDLY UNRECOGNIZED DISH STATE BY DEFAULT + TOGGLE FOR FULL TECHNICAL METRICS
    if (culinaryCardEl) culinaryCardEl.classList.add("ood-rejected");
    if (telemetryPanel) telemetryPanel.classList.add("hidden");
    if (confidenceMeterBox) confidenceMeterBox.classList.add("hidden");
    consensusBadge.classList.add("ood-badge");
    consensusBadge.textContent = "Not Recognized";

    dishCategory.textContent = "OUTSIDE SUPPORTED MENU";
    dishName.textContent = "Dish Not Recognized";
    dishNepali.textContent = "पहिचान भएन";
    primaryModelLabel.textContent = `${primary.title} CLOSEST FALLBACK`;
    animateNumber(primaryConfidence, primary.confidence, "%", 460);
    primaryLatency.textContent = `${primary.latency_ms.toFixed(1)} ms`;
    dishTagline.textContent =
      ood.summary_message ||
      "We couldn't match this photo to any of our 10 supported Nepali dishes (Chatpate, Chiya, Dal Bhat, Jeri, Momo, Pakauda, Panipuri, Samosa, Selroti, or Yomari). Here is why:";

    // Collapse dish prep drawer and hide recipe/Google Maps buttons for unrecognized objects
    if (dishCollapsibleBody) dishCollapsibleBody.classList.remove("open");
    if (toggleDishDetailsBtn) {
      toggleDishDetailsBtn.classList.remove("open");
      toggleDishDetailsBtn.setAttribute("aria-expanded", "false");
    }
    if (toggleDishDetailsText) {
      toggleDishDetailsText.textContent = "📖 Dish Info & Prep";
    }
    if (compactBarRight) compactBarRight.classList.add("hidden");

    // 1. Simple plain-English reasons shown by default
    dishFlavorTags.innerHTML = "";
    const oodContainer = document.createElement("div");
    oodContainer.className = "ood-feedback-wrap";

    const simpleList = document.createElement("ul");
    simpleList.className = "ood-simple-reasons";
    (ood.reasons || []).forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      simpleList.appendChild(li);
    });
    oodContainer.appendChild(simpleList);

    // 2. Toggle Button to reveal all technical reasons, values, thresholds, and Model Comparison panel
    const techToggleBtn = document.createElement("button");
    techToggleBtn.type = "button";
    techToggleBtn.className = "btn btn-ghost btn-sm ood-tech-toggle-btn";
    techToggleBtn.innerHTML = `<span>🔬 Show Technical Rejection Reasons &amp; Model Metrics</span> <span class="drawer-chevron">▾</span>`;

    // 3. Technical Drawer (hidden until toggle is pressed)
    const techDrawer = document.createElement("div");
    techDrawer.className = "ood-tech-drawer hidden";

    const techReasons = ood.technical_reasons && ood.technical_reasons.length > 0 ? ood.technical_reasons : ood.reasons || [];
    const techReasonsHtml = techReasons
      .map((tr) => `<div class="ood-tech-reason-pill">⚠️ ${tr}</div>`)
      .join("");

    const m = ood.metrics || {};
    const th = ood.thresholds || {};
    const sem = ood.semantic_domain || {};
    const cosVal =
      typeof primary.centroid_cosine_sim === "number" ? primary.centroid_cosine_sim.toFixed(4) : "—";

    techDrawer.innerHTML = `
      <div class="ood-tech-header">3-LAYER OPEN-SET GATE DIAGNOSTICS &amp; THRESHOLD VALUES</div>
      <div class="ood-tech-pills-list">${techReasonsHtml}</div>
      <div class="ood-tech-metrics-grid">
        <div class="ood-tech-chip">
          <span class="ood-tech-lbl">CLOSEST SOFTMAX FALLBACK</span>
          <strong class="ood-tech-val">${primary.predicted_name} (${primary.confidence.toFixed(2)}% | min ${th.min_confidence_pct ?? 78}%)</strong>
        </div>
        <div class="ood-tech-chip">
          <span class="ood-tech-lbl">SHANNON ENTROPY H(p)/ln(10)</span>
          <strong class="ood-tech-val">${primary.normalized_entropy.toFixed(4)} (ceiling ≤ ${th.max_normalized_entropy ?? 0.36})</strong>
        </div>
        <div class="ood-tech-chip">
          <span class="ood-tech-lbl">TOP-1 VS TOP-2 MARGIN</span>
          <strong class="ood-tech-val">${primary.margin.toFixed(2)}% (min ≥ ${th.min_margin_pct ?? 60}%)</strong>
        </div>
        <div class="ood-tech-chip">
          <span class="ood-tech-lbl">MAX LOGIT &amp; HELMHOLTZ ENERGY</span>
          <strong class="ood-tech-val">z_max = ${primary.max_logit.toFixed(3)} (≥ ${th.min_max_logit ?? 3.35}) · E = ${primary.energy_score.toFixed(3)}</strong>
        </div>
        <div class="ood-tech-chip">
          <span class="ood-tech-lbl">768-D CENTROID COSINE SIM</span>
          <strong class="ood-tech-val">S_cos = ${cosVal} (min ≥ ${th.min_centroid_cosine_sim ?? 0.85})</strong>
        </div>
        <div class="ood-tech-chip">
          <span class="ood-tech-lbl">IMAGENET CULINARY DOMAIN MASS</span>
          <strong class="ood-tech-val">${m.culinary_domain_mass_pct ?? "—"}% (min ≥ ${th.min_culinary_domain_mass_pct ?? 38}% · Top: '${sem.top_imagenet_label ?? "—"}' ${sem.top_imagenet_conf_pct ?? 0}%)</strong>
        </div>
      </div>
    `;

    techToggleBtn.addEventListener("click", () => {
      const isHidden = techDrawer.classList.toggle("hidden");
      if (!isHidden) {
        techToggleBtn.innerHTML = `<span>🔬 Hide Technical Rejection Reasons &amp; Model Metrics</span> <span class="drawer-chevron">▴</span>`;
        if (telemetryPanel) telemetryPanel.classList.remove("hidden");
        if (confidenceMeterBox) confidenceMeterBox.classList.remove("hidden");
        requestAnimationFrame(() => {
          modelResultsContainer.querySelectorAll(".prob-fill").forEach((bar) => {
            bar.style.width = bar.getAttribute("data-target-width");
          });
        });
      } else {
        techToggleBtn.innerHTML = `<span>🔬 Show Technical Rejection Reasons &amp; Model Metrics</span> <span class="drawer-chevron">▾</span>`;
        if (telemetryPanel) telemetryPanel.classList.add("hidden");
        if (confidenceMeterBox) confidenceMeterBox.classList.add("hidden");
      }
    });

    oodContainer.appendChild(techToggleBtn);
    oodContainer.appendChild(techDrawer);
    dishFlavorTags.appendChild(oodContainer);
  } else {
    // IN-DISTRIBUTION / RECOGNIZED NEPALI DISH STATE
    if (culinaryCardEl) culinaryCardEl.classList.remove("ood-rejected");
    if (telemetryPanel) telemetryPanel.classList.remove("hidden");
    if (confidenceMeterBox) confidenceMeterBox.classList.remove("hidden");
    consensusBadge.classList.remove("ood-badge");
    if (compactBarRight) compactBarRight.classList.remove("hidden");

    // Consensus badge
    if (runs.length > 1) {
      if (data.unanimous_agreement) {
        consensusBadge.textContent = `${runs.length}/${runs.length} Models Agree`;
      } else {
        consensusBadge.textContent = "2/3 Majority Consensus";
      }
    } else {
      consensusBadge.textContent = `${primary.title}`;
    }

    // Culinary Spotlight Card
    dishCategory.textContent = dish.category;
    dishName.textContent = dish.name;
    dishNepali.textContent = dish.nepali;
    primaryModelLabel.textContent = `${primary.title} CONFIDENCE`;
    animateNumber(primaryConfidence, primary.confidence, "%", 460);
    primaryLatency.textContent = `${primary.latency_ms.toFixed(1)} ms`;

    dishTagline.textContent = dish.tagline;

    // Quick Facts 4-tile grid
    const qf = dish.quick_facts || {};
    factTaste.textContent = qf.taste || "—";
    factMethod.textContent = qf.method || "—";
    factPairing.textContent = qf.pairing || "—";
    factVibe.textContent = qf.vibe || "—";

    // 3 Numbered Recipe Steps
    dishStepsRow.innerHTML = "";
    (dish.steps || []).forEach((stepText, idx) => {
      const stepEl = document.createElement("div");
      stepEl.className = "step-card";
      stepEl.innerHTML = `
        <span class="step-num">STEP 0${idx + 1}</span>
        <p>${stepText}</p>
      `;
      dishStepsRow.appendChild(stepEl);
    });

    dishSpecialty.textContent = dish.specialty || "—";

    // Update Google Maps "Where Can I Get It?" link
    const mapQuery = encodeURIComponent(`Nepali ${dish.name} restaurant near me`);
    findNearMeBtn.href = `https://www.google.com/maps/search/?api=1&query=${mapQuery}`;
    findNearMeBtn.textContent = `📍 Find ${dish.name} Near Me ↗`;

    dishFlavorTags.innerHTML = "";
    (dish.flavor_tags || []).forEach((tag) => {
      const span = document.createElement("span");
      span.className = "flavor-tag";
      span.textContent = tag;
      dishFlavorTags.appendChild(span);
    });
  }

  // Total Latency
  const sumMs = runs.reduce((acc, r) => acc + r.latency_ms, 0);
  totalInferenceTime.textContent = `${runs.length} model(s) ran in ${sumMs.toFixed(1)} ms total`;

  // Render Model Head-to-Head Cards with clean 3-chip specs footer
  modelResultsContainer.innerHTML = "";
  runs.forEach((run, i) => {
    const card = document.createElement("div");
    const oodClass = !isRecognized || run.is_ood ? "ood-flagged" : "";
    card.className = `model-run-card ${run.arch === "convnext_tiny" ? "hero-model" : ""} ${oodClass}`.trim();
    card.style.animationDelay = `${i * 70}ms`;

    const top3BarsHtml = run.top3
      .map(
        (item) => `
        <div class="prob-row">
          <div class="prob-label-line">
            <span>${item.name} (${item.nepali})</span>
            <strong>${item.confidence.toFixed(2)}%</strong>
          </div>
          <div class="prob-track">
            <div class="prob-fill" data-target-width="${Math.max(item.confidence, 2)}%"></div>
          </div>
        </div>
      `
      )
      .join("");

    const paramsM = (run.parameters / 1e6).toFixed(1) + "M";
    const fallbackNote =
      !isRecognized || run.is_ood
        ? `<span class="ood-card-sub">Closest softmax fallback · Entropy H = ${
            typeof run.normalized_entropy === "number" ? run.normalized_entropy.toFixed(2) : "—"
          }</span>`
        : "";

    card.innerHTML = `
      <div class="model-card-top">
        <div>
          <div class="model-arch-title">${run.title}</div>
          <div class="model-arch-family">${run.family}</div>
        </div>
        <span class="latency-pill">${run.latency_ms.toFixed(1)} ms</span>
      </div>

      <div class="model-verdict-row">
        <div>
          <span class="verdict-dish">${run.predicted_name} <small>${run.predicted_nepali}</small></span>
          ${fallbackNote}
        </div>
        <span class="verdict-conf">${run.confidence.toFixed(2)}%</span>
      </div>

      <div class="prob-bars">
        ${top3BarsHtml}
      </div>

      <div class="model-specs-grid">
        <div class="spec-chip">
          <span class="spec-chip-lbl">PARAMS</span>
          <span class="spec-chip-val">${paramsM}</span>
        </div>
        <div class="spec-chip">
          <span class="spec-chip-lbl">ONNX SIZE</span>
          <span class="spec-chip-val">${run.size_mb} MB</span>
        </div>
        <div class="spec-chip">
          <span class="spec-chip-lbl">TEST TOP-1</span>
          <span class="spec-chip-val">${run.benchmark_test_top1}%</span>
        </div>
      </div>
    `;
    modelResultsContainer.appendChild(card);
  });

  requestAnimationFrame(() => {
    modelResultsContainer.querySelectorAll(".prob-fill").forEach((bar) => {
      bar.style.width = bar.getAttribute("data-target-width");
    });
  });
}

// 7. Load Sample Specimens & Shuffle Support
async function loadSamples(offset = 0, autoSelectFirst = false) {
  try {
    const res = await fetch(`${API_BASE}/api/samples?offset=${offset}`);
    const data = await res.json();
    const samples = data.samples || [];

    specimenGrid.innerHTML = "";
    samples.forEach((s, idx) => {
      const btn = document.createElement("div");
      btn.className = "specimen-thumb";
      btn.style.animationDelay = `${idx * 45}ms`;
      btn.innerHTML = `
        <img src="${s.image_url}" alt="${s.name}" loading="lazy" />
        <span>${s.name}</span>
      `;
      btn.addEventListener("click", () => {
        document.querySelectorAll(".specimen-thumb").forEach((t) => t.classList.remove("active"));
        btn.classList.add("active");
        currentUploadFile = null;
        currentSamplePath = s.rel_path;
        currentPreviewSrc = s.image_url;
        imagePreview.src = s.image_url;
        floatingDishImg.src = s.image_url;
        previewWrapper.classList.add("visible");
        dropzonePrompt.classList.add("hidden");
        tabUploadBtn.click();
        executePrediction();
      });
      specimenGrid.appendChild(btn);

      if (autoSelectFirst && (s.class_key === "momo" || (idx === 0 && !currentSamplePath))) {
        currentSamplePath = s.rel_path;
        currentPreviewSrc = s.image_url;
        btn.classList.add("active");
        imagePreview.src = s.image_url;
        floatingDishImg.src = s.image_url;
        previewWrapper.classList.add("visible");
        dropzonePrompt.classList.add("hidden");
      }
    });

    if (autoSelectFirst && currentSamplePath) {
      executePrediction();
    }
  } catch (err) {
    console.error("Failed to load sample specimens:", err);
  }
}

if (shuffleSamplesBtn) {
  shuffleSamplesBtn.addEventListener("click", () => {
    sampleShuffleOffset += 1;
    loadSamples(sampleShuffleOffset, false);
  });
}

// 8. Matrix Viewer Tabs
matrixTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    matrixTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    matrixViewerImg.style.opacity = "0.2";
    setTimeout(() => {
      matrixViewerImg.src = tab.dataset.img;
      matrixViewerImg.style.opacity = "1";
    }, 120);
  });
});

if ("scrollRestoration" in history) {
  history.scrollRestoration = "manual";
}

window.addEventListener("DOMContentLoaded", async () => {
  if (window.location.hash) {
    history.replaceState(null, "", window.location.pathname);
  }
  window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  initScrollAnimations();
  await loadSamples(0, true);
  window.scrollTo({ top: 0, left: 0, behavior: "instant" });
});

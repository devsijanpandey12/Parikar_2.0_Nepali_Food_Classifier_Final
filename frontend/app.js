const API_BASE = "";

let selectedModel = "all";
let currentUploadFile = null;
let currentSamplePath = null;
let mediaStream = null;

// DOM Elements
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
const primaryModelLabel = document.getElementById("primaryModelLabel");
const primaryConfidence = document.getElementById("primaryConfidence");
const primaryLatency = document.getElementById("primaryLatency");
const dishTagline = document.getElementById("dishTagline");
const dishPreparation = document.getElementById("dishPreparation");
const dishSpecialty = document.getElementById("dishSpecialty");
const dishFlavorTags = document.getElementById("dishFlavorTags");
const modelResultsContainer = document.getElementById("modelResultsContainer");
const totalInferenceTime = document.getElementById("totalInferenceTime");

// Matrix Viewer Tabs
const matrixTabs = document.querySelectorAll(".matrix-tab");
const matrixViewerImg = document.getElementById("matrixViewerImg");

const MODEL_HINTS = {
  all: "Running all 3 architectures head-to-head",
  convnext_tiny: "Single model: ConvNeXt-Tiny (27.8M · 99.49% Test Top-1)",
  efficientnet_v2_s: "Single model: EfficientNetV2-S (20.2M · 100% Test Top-3)",
  swin_t: "Single model: Swin-Tiny Vision Transformer (27.5M)"
};

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
    imagePreview.src = ev.target.result;
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
    alert("Please select a sample dish on the left, upload a photo, or capture a frame from your camera first.");
    return;
  }

  runInferenceBtn.textContent = "RUNNING ONNX TENSOR PIPELINE...";
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
    consensusBadge.textContent = "INFERENCE ERROR";
  } finally {
    runInferenceBtn.textContent = "RUN CLASSIFICATION PIPELINE →";
    runInferenceBtn.disabled = false;
  }
}

function renderPredictionOutput(data) {
  const primary = data.primary_prediction;
  const dish = data.dish_profile;
  const runs = data.model_runs || [];

  // Consensus badge
  if (runs.length > 1) {
    if (data.unanimous_agreement) {
      consensusBadge.textContent = `CONSENSUS: ${runs.length}/${runs.length} MODELS AGREE`;
      consensusBadge.style.background = "#1D6342";
    } else {
      consensusBadge.textContent = "SPLIT VERDICT ACROSS ARCHITECTURES";
      consensusBadge.style.background = "#B88014";
    }
  } else {
    consensusBadge.textContent = `SINGLE MODEL // ${primary.title.toUpperCase()}`;
    consensusBadge.style.background = "#161513";
  }

  // Culinary Card
  dishCategory.textContent = dish.category;
  dishName.textContent = dish.name;
  dishNepali.textContent = dish.nepali;
  primaryModelLabel.textContent = `${primary.title} CONFIDENCE`;
  primaryConfidence.textContent = `${primary.confidence.toFixed(2)}%`;
  primaryLatency.textContent = `${primary.latency_ms.toFixed(2)} ms on CPU`;

  dishTagline.textContent = dish.tagline;
  dishPreparation.textContent = dish.preparation;
  dishSpecialty.textContent = dish.specialty;

  dishFlavorTags.innerHTML = "";
  (dish.flavor_tags || []).forEach((tag) => {
    const span = document.createElement("span");
    span.className = "flavor-tag";
    span.textContent = tag;
    dishFlavorTags.appendChild(span);
  });

  // Total Latency
  const sumMs = runs.reduce((acc, r) => acc + r.latency_ms, 0);
  totalInferenceTime.textContent = `${runs.length} model(s) executed in ${sumMs.toFixed(2)} ms total (ONNX CPU)`;

  // Render Model Head-to-Head Cards
  modelResultsContainer.innerHTML = "";
  runs.forEach((run) => {
    const card = document.createElement("div");
    card.className = `model-run-card ${run.arch === "convnext_tiny" ? "hero-model" : ""}`;

    const top3BarsHtml = run.top3
      .map(
        (item) => `
        <div class="prob-row">
          <div class="prob-label-line">
            <span>${item.name} (${item.nepali})</span>
            <strong>${item.confidence.toFixed(2)}%</strong>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${Math.max(item.confidence, 1.5)}%"></div>
          </div>
        </div>
      `
      )
      .join("");

    const paramsM = (run.parameters / 1e6).toFixed(1) + "M";

    card.innerHTML = `
      <div class="model-card-top">
        <div>
          <div class="model-arch-title">${run.title}</div>
          <div class="model-arch-family">${run.family}</div>
        </div>
        <span class="latency-pill">${run.latency_ms.toFixed(2)} ms</span>
      </div>

      <div class="model-verdict-row">
        <span class="verdict-dish">${run.predicted_name} <small>${run.predicted_nepali}</small></span>
        <span class="verdict-conf">${run.confidence.toFixed(2)}%</span>
      </div>

      <div class="prob-bars">
        ${top3BarsHtml}
      </div>

      <div class="model-meta-footer">
        <span>PARAMS: ${paramsM} (${run.size_mb} MB)</span>
        <span>TEST TOP-1: ${run.benchmark_test_top1}%</span>
      </div>
    `;
    modelResultsContainer.appendChild(card);
  });
}

// 7. Load Sample Specimens & Run Initial Demo Specimen
async function initLab() {
  try {
    const res = await fetch(`${API_BASE}/api/samples`);
    const data = await res.json();
    const samples = data.samples || [];

    specimenGrid.innerHTML = "";
    samples.forEach((s, idx) => {
      const btn = document.createElement("div");
      btn.className = "specimen-thumb";
      btn.innerHTML = `
        <img src="${s.image_url}" alt="${s.name}" loading="lazy" />
        <span>${s.name}</span>
      `;
      btn.addEventListener("click", () => {
        document.querySelectorAll(".specimen-thumb").forEach((t) => t.classList.remove("active"));
        btn.classList.add("active");
        currentUploadFile = null;
        currentSamplePath = s.rel_path;
        imagePreview.src = s.image_url;
        previewWrapper.classList.add("visible");
        dropzonePrompt.classList.add("hidden");
        tabUploadBtn.click();
        executePrediction();
      });
      specimenGrid.appendChild(btn);

      // Automatically load 'momo' (or first specimen) on startup so the lab looks alive immediately
      if (s.class_key === "momo" || (idx === 0 && !currentSamplePath)) {
        currentSamplePath = s.rel_path;
        btn.classList.add("active");
        imagePreview.src = s.image_url;
        previewWrapper.classList.add("visible");
        dropzonePrompt.classList.add("hidden");
      }
    });

    if (currentSamplePath) {
      executePrediction();
    }
  } catch (err) {
    console.error("Failed to load sample specimens:", err);
  }
}

// 8. Matrix Viewer Tabs
matrixTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    matrixTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    matrixViewerImg.src = tab.dataset.img;
  });
});

window.addEventListener("DOMContentLoaded", initLab);

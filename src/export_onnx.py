import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_and_eval import build_model, CLASSES

ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
ONNX_DIR = ROOT_DIR / "models_onnx"

ARCHITECTURES = ["convnext_tiny", "efficientnet_v2_s", "swin_t"]


def export_all_to_onnx():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    manifest = {}

    for arch in ARCHITECTURES:
        ckpt_path = EXPERIMENTS_DIR / arch / "best_model.pt"
        eval_path = EXPERIMENTS_DIR / arch / "evaluation_report.json"
        if not ckpt_path.exists():
            print(f"Skipping {arch}: missing {ckpt_path}")
            continue

        print(f"Exporting {arch} to ONNX...")
        model, _ = build_model(arch, num_classes=len(CLASSES), pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        onnx_path = ONNX_DIR / f"{arch}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            dynamo=False,
        )

        # Verify with ONNX Runtime and measure CPU latency
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x_np = dummy_input.numpy()
        # Warmup
        for _ in range(3):
            sess.run(["logits"], {"input": x_np})

        t0 = time.perf_counter()
        n_runs = 20
        for _ in range(n_runs):
            ort_outs = sess.run(["logits"], {"input": x_np})
        cpu_ms = ((time.perf_counter() - t0) * 1000.0) / n_runs

        with torch.no_grad():
            pt_out = model(dummy_input).numpy()
        max_diff = float(np.max(np.abs(pt_out - ort_outs[0])))

        size_mb = round(onnx_path.stat().st_size / (1024 * 1024), 2)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_rep = json.load(f)

        manifest[arch] = {
            "onnx_file": f"models_onnx/{arch}.onnx",
            "size_mb": size_mb,
            "parameters": eval_rep["total_parameters"],
            "best_epoch": eval_rep["best_epoch"],
            "cpu_latency_ms": round(cpu_ms, 2),
            "max_abs_diff_vs_pytorch": round(max_diff, 6),
            "test_metrics": {
                "top1_accuracy": eval_rep["test_metrics"]["top1_accuracy"],
                "top3_accuracy": eval_rep["test_metrics"]["top3_accuracy"],
                "balanced_accuracy": eval_rep["test_metrics"]["balanced_accuracy_macro_recall"],
                "macro_f1": eval_rep["test_metrics"]["macro_f1"],
            }
        }
        print(f"  -> Saved {onnx_path.name} ({size_mb} MB) | CPU Latency: {cpu_ms:.2f} ms | Max diff: {max_diff:.2e}")

    with open(ONNX_DIR / "onnx_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Saved models_onnx/onnx_manifest.json")


if __name__ == "__main__":
    export_all_to_onnx()

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT_DIR / "experiments"

MODELS = [
    ("convnext_tiny", "ConvNeXt-Tiny\n(27.8M)"),
    ("efficientnet_v2_s", "EfficientNetV2-S\n(20.2M)"),
    ("swin_t", "Swin-Tiny\n(27.5M)")
]


def main():
    metrics_names = ["Top-1 Acc", "Top-3 Acc", "Balanced Acc", "Macro F1"]
    data = {}
    histories = {}

    for arch, label in MODELS:
        report_path = EXP_DIR / arch / "evaluation_report.json"
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        tm = rep["test_metrics"]
        data[label] = [
            tm["top1_accuracy"] * 100,
            tm["top3_accuracy"] * 100,
            tm["balanced_accuracy_macro_recall"] * 100,
            tm["macro_f1"] * 100
        ]
        histories[label] = rep["history"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Subplot 1: Grouped Bar Chart on Held-Out Test Set
    ax = axes[0]
    x = np.arange(len(metrics_names))
    width = 0.25
    colors = ["#2563eb", "#10b981", "#f59e0b"]

    for i, (_, label) in enumerate(MODELS):
        vals = data[label]
        rects = ax.bar(x + (i - 1) * width, vals, width, label=label.replace("\n", " "), color=colors[i], edgecolor="black", linewidth=0.6)
        for r in rects:
            h = r.get_height()
            ax.annotate(
                f"{h:.2f}%",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax.set_ylim(95.0, 101.0)
    ax.set_ylabel("Score (%)")
    ax.set_title("Held-Out Test Set Benchmark (990 Unseen Images)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Subplot 2: Validation Macro-F1 Curve across Epochs
    ax2 = axes[1]
    for i, (_, label) in enumerate(MODELS):
        hist = histories[label]
        epochs = [h["epoch"] for h in hist]
        val_f1 = [h["val_macro_f1"] * 100 for h in hist]
        ax2.plot(epochs, val_f1, marker="o", linewidth=2.2, color=colors[i], label=label.replace("\n", " "))

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Macro F1-Score (%)")
    ax2.set_title("Validation Macro-F1 Convergence Across 6 Epochs")
    ax2.set_xticks([1, 2, 3, 4, 5, 6])
    ax2.set_ylim(84.0, 100.5)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_file = EXP_DIR / "model_comparison_benchmark.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved benchmark comparison chart to {out_file}")


if __name__ == "__main__":
    main()

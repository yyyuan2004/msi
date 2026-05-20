"""Generate publication-quality comparison figures after ablation study.

Reads eval_results/results.json from each (config, seed) output directory
and produces:
    1. IoU box/violin plot across all configs
    2. F1 box/violin plot across all configs
    3. Radar chart of all metrics per config (mean over seeds)
    4. Params vs IoU scatter plot

Output: outputs/ablation_plots/
"""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CONFIGS = [
    "baseline",
    "spconv_se",
    "smp_unet_resnet18",
    "smp_unet_resnet34",
    "smp_unetplusplus_resnet34",
    "smp_linknet_resnet34",
    "smp_manet_resnet34",
    "smp_deeplabv3plus_mobilenetv2",
    "smp_fpn_efficientnetb0",
    "topformer_t",
    "topformer_s",
    "topformer_b",
    "seaformer_t",
    "seaformer_s",
    "seaformer_b",
    "pidnet_s",
    "pidnet_m",
]

DISPLAY = {
    "baseline": "Baseline",
    "spconv_se": "SpConv+SE",
    "smp_unet_resnet18": "UNet-R18",
    "smp_unet_resnet34": "UNet-R34",
    "smp_unetplusplus_resnet34": "UNet++-R34",
    "smp_linknet_resnet34": "LinkNet-R34",
    "smp_manet_resnet34": "MAnet-R34",
    "smp_deeplabv3plus_mobilenetv2": "DLv3+-MV2",
    "smp_fpn_efficientnetb0": "FPN-EffB0",
    "topformer_t": "TopFormer-T",
    "topformer_s": "TopFormer-S",
    "topformer_b": "TopFormer-B",
    "seaformer_t": "SeaFormer-T",
    "seaformer_s": "SeaFormer-S",
    "seaformer_b": "SeaFormer-B",
    "pidnet_s": "PIDNet-S",
    "pidnet_m": "PIDNet-M",
}

SEEDS = [42, 123, 456]

# Publication-quality style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_all_results(base_dir="outputs"):
    """Load eval results for all (config, seed) combos."""
    data = {}
    for cfg in CONFIGS:
        data[cfg] = {"mIoU": [], "F1_macro": [], "Precision_macro": [],
                      "Recall_macro": [], "IoU_class1": []}
        for seed in SEEDS:
            path = os.path.join(base_dir, f"{cfg}_seed{seed}",
                                "eval_results", "results.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                r = json.load(f)
            data[cfg]["mIoU"].append(r.get("mIoU", 0))
            data[cfg]["F1_macro"].append(r.get("F1_macro", 0))
            data[cfg]["Precision_macro"].append(r.get("Precision_macro", 0))
            data[cfg]["Recall_macro"].append(r.get("Recall_macro", 0))
            iou_pc = r.get("IoU_per_class", [0, 0])
            data[cfg]["IoU_class1"].append(float(iou_pc[1]) if len(iou_pc) > 1 else 0)
    return data


def count_params(cfg_name):
    """Count model params (returns M)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.config import load_config
        from model.model import build_model
        cfg = load_config(os.path.join("configs", f"{cfg_name}.yaml"))
        model = build_model(cfg)
        return sum(p.numel() for p in model.parameters()) / 1e6
    except Exception:
        return None


def plot_box_violin(data, metric_key, metric_label, output_path):
    """Box + strip plot for a given metric across all configs."""
    labels = []
    values = []
    for cfg in CONFIGS:
        vals = data[cfg].get(metric_key, [])
        if vals:
            labels.append(DISPLAY.get(cfg, cfg))
            values.append([v * 100 for v in vals])

    if not values:
        print(f"  No data for {metric_key}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 6))

    bp = ax.boxplot(values, labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="red",
                                                    markeredgecolor="red", markersize=5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, vals in enumerate(values):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter([i + 1 + j for j in jitter], vals, s=30, alpha=0.8,
                   zorder=5, edgecolors="black", linewidths=0.5)

    ax.set_ylabel(f"{metric_label} (%)")
    ax.set_title(f"{metric_label} Distribution Across Models (3 seeds)")
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Annotate mean +/- std
    for i, vals in enumerate(values):
        mean = np.mean(vals)
        std = np.std(vals)
        ax.text(i + 1, ax.get_ylim()[1] - 0.5, f"{mean:.1f}",
                ha="center", va="top", fontsize=7, color="red", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_grouped_bar(data, output_path):
    """Grouped bar chart: multiple metrics side-by-side per config."""
    metrics = ["IoU_class1", "mIoU", "F1_macro", "Precision_macro", "Recall_macro"]
    labels_m = ["IoU(defect)", "mIoU", "F1", "Precision", "Recall"]

    configs_with_data = [c for c in CONFIGS if data[c]["IoU_class1"]]
    if not configs_with_data:
        return

    x = np.arange(len(configs_with_data))
    n_metrics = len(metrics)
    bar_w = 0.8 / n_metrics
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4"]

    fig, ax = plt.subplots(figsize=(max(14, len(configs_with_data) * 1.2), 6))

    for mi, (mkey, mlabel) in enumerate(zip(metrics, labels_m)):
        means = [np.mean(data[c][mkey]) * 100 for c in configs_with_data]
        stds = [np.std(data[c][mkey]) * 100 for c in configs_with_data]
        ax.bar(x + mi * bar_w - 0.4 + bar_w / 2, means, bar_w, yerr=stds,
               capsize=2, color=colors[mi], label=mlabel, edgecolor="black",
               linewidth=0.3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(c, c) for c in configs_with_data],
                       rotation=45, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Multi-Metric Comparison Across All Models")
    ax.legend(loc="lower right", ncol=n_metrics, fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_params_vs_iou(data, output_path):
    """Scatter: model params (M) vs IoU(defect) with error bars."""
    configs_with_data = [c for c in CONFIGS if data[c]["IoU_class1"]]
    if not configs_with_data:
        return

    params = []
    iou_means = []
    iou_stds = []
    names = []
    for c in configs_with_data:
        p = count_params(c)
        if p is None:
            continue
        params.append(p)
        vals = [v * 100 for v in data[c]["IoU_class1"]]
        iou_means.append(np.mean(vals))
        iou_stds.append(np.std(vals))
        names.append(DISPLAY.get(c, c))

    if not params:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(params, iou_means, yerr=iou_stds, fmt="o", capsize=4,
                markersize=8, color="#2196F3", ecolor="#90CAF9", elinewidth=1.5)

    for x, y, name in zip(params, iou_means, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 8),
                    fontsize=8, alpha=0.85)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("IoU Defect (%)")
    ax.set_title("Model Efficiency: Params vs. IoU(defect)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    out_dir = "outputs/ablation_plots"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading ablation results...")
    data = load_all_results()

    n_with_data = sum(1 for c in CONFIGS if data[c]["IoU_class1"])
    print(f"Found results for {n_with_data}/{len(CONFIGS)} configs.\n")

    if n_with_data == 0:
        print("No results found. Run ablation experiments first.")
        return

    print("Generating plots:")

    plot_box_violin(data, "IoU_class1", "IoU (Defect Class)",
                    os.path.join(out_dir, "iou_boxplot.png"))

    plot_box_violin(data, "F1_macro", "F1 (Macro)",
                    os.path.join(out_dir, "f1_boxplot.png"))

    plot_box_violin(data, "mIoU", "mIoU",
                    os.path.join(out_dir, "miou_boxplot.png"))

    plot_grouped_bar(data, os.path.join(out_dir, "grouped_bar.png"))

    plot_params_vs_iou(data, os.path.join(out_dir, "params_vs_iou.png"))

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()

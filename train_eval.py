"""自动化训练→评估→可视化工作流。

功能:
    1. 运行训练 (train.py 的 train 函数)
    2. 自动在 val set 上运行评估 (eval.py 的逻辑)
    3. 生成科研级别的逐epoch指标曲线图 (PNG)
    4. 可选: 生成数据增强可视化对比图

用法:
    python train_eval.py --config configs/baseline.yaml --seed 42
    python train_eval.py --config configs/se.yaml --seed 42 --vis_augment

注意:
    - 本脚本不影响原有 train.py 和 eval.py 的独立使用
    - 默认 split 比例: train:val = 7:3 (test=0)
    - eval 在 val set 上执行
"""

import argparse
import json
import os
import sys
import time
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from train import train, set_seed
from data.dataset import MSIDataset, get_dataset_kwargs
from data.augment import (
    get_train_transforms, get_val_transforms,
    Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90,
    RandomCrop, ElasticTransform, Cutout, GaussianBlur, IntensityJitter,
    GaussianNoise, Resize,
)
from data.split import get_data_splits, get_kfold_splits
from model.model import build_model
from utils.metrics import SegmentationMetrics
from eval import evaluate, plot_confusion_matrix, visualize_predictions, visualize_error_analysis, print_results, _normalize_band


# ---------------------------------------------------------------------------
# 科研绘图: 逐epoch指标曲线
# ---------------------------------------------------------------------------

def plot_metric_curves(log_path, output_dir, experiment_name):
    """从 training_log.json 生成科研级别的逐epoch指标曲线图。

    输出图片:
        - loss_curve.png: 训练/验证损失曲线
        - iou_f1_curve.png: Class1 IoU 与 F1 曲线 (主图)
        - precision_recall_curve.png: Precision 与 Recall 曲线
        - lr_curve.png: 学习率变化曲线
        - metrics_summary.png: 所有关键指标汇总图
    """
    with open(log_path, "r") as f:
        logs = json.load(f)

    if not logs:
        print("WARNING: training_log.json is empty, skipping curve plotting.")
        return

    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    epochs = [e["epoch"] for e in logs]
    train_loss = [e["train_loss"] for e in logs]
    val_loss = [e["val_loss"] for e in logs]
    iou_c1 = [e["IoU_class1"] for e in logs]
    f1 = [e["F1_macro"] for e in logs]
    precision = [e["Precision_macro"] for e in logs]
    recall = [e["Recall_macro"] for e in logs]
    lr = [e["lr"] for e in logs]

    # 全局绘图风格
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
    })

    def _annotate_max(ax, x, y, label, color):
        """在曲线最大值处标注。"""
        idx = int(np.argmax(y))
        xmax, ymax = x[idx], y[idx]
        ax.annotate(
            f"max={ymax:.4f}\n(epoch {xmax})",
            xy=(xmax, ymax), fontsize=8,
            xytext=(15, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
        )

    def _annotate_min(ax, x, y, label, color):
        """在曲线最小值处标注。"""
        idx = int(np.argmin(y))
        xmin, ymin = x[idx], y[idx]
        ax.annotate(
            f"min={ymin:.4f}\n(epoch {xmin})",
            xy=(xmin, ymin), fontsize=8,
            xytext=(15, -25), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
        )

    def _add_mean_line(ax, y, color, label):
        """添加均值水平线。"""
        mean_val = np.mean(y)
        ax.axhline(y=mean_val, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(
            0.98, mean_val, f"mean={mean_val:.4f}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color=color, alpha=0.7,
        )

    # --- 1. Loss 曲线 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, color="#2196F3", label="Train Loss")
    ax.plot(epochs, val_loss, color="#F44336", label="Val Loss")
    _annotate_min(ax, epochs, val_loss, "Val Loss", "#F44336")
    _add_mean_line(ax, val_loss, "#F44336", "val mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training & Validation Loss — {experiment_name}")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "loss_curve.png"), dpi=200)
    plt.close(fig)

    # --- 2. IoU(class1) + F1 曲线 (主图) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, iou_c1, color="#4CAF50", label="IoU (defect, class 1)")
    ax.plot(epochs, f1, color="#FF9800", label="F1 (macro)")
    _annotate_max(ax, epochs, iou_c1, "IoU", "#4CAF50")
    _annotate_max(ax, epochs, f1, "F1", "#FF9800")
    _add_mean_line(ax, iou_c1, "#4CAF50", "IoU mean")
    _add_mean_line(ax, f1, "#FF9800", "F1 mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"IoU (Defect) & F1 Score — {experiment_name}")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "iou_f1_curve.png"), dpi=200)
    plt.close(fig)

    # --- 3. Precision + Recall 曲线 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, precision, color="#9C27B0", label="Precision (macro)")
    ax.plot(epochs, recall, color="#00BCD4", label="Recall (macro)")
    _annotate_max(ax, epochs, precision, "Precision", "#9C27B0")
    _annotate_max(ax, epochs, recall, "Recall", "#00BCD4")
    _add_mean_line(ax, precision, "#9C27B0", "Prec mean")
    _add_mean_line(ax, recall, "#00BCD4", "Rec mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision & Recall — {experiment_name}")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "precision_recall_curve.png"), dpi=200)
    plt.close(fig)

    # --- 4. 汇总图 (1x3 子图：Loss | IoU+F1 | Precision+Recall) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (0) Loss
    axes[0].plot(epochs, train_loss, color="#2196F3", label="Train Loss")
    axes[0].plot(epochs, val_loss, color="#F44336", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=9)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    # (1) IoU + F1
    axes[1].plot(epochs, iou_c1, color="#4CAF50", label="IoU(defect)")
    axes[1].plot(epochs, f1, color="#FF9800", label="F1(macro)")
    axes[1].set_title("IoU & F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")

    # (2) Precision + Recall
    axes[2].plot(epochs, precision, color="#9C27B0", label="Precision")
    axes[2].plot(epochs, recall, color="#00BCD4", label="Recall")
    axes[2].set_title("Precision & Recall")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=9)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")

    # 在汇总图里标注最优值
    best_iou_idx = int(np.argmax(iou_c1))
    best_f1_idx = int(np.argmax(f1))
    summary_text = (
        f"Best IoU(defect): {iou_c1[best_iou_idx]:.4f} @ epoch {epochs[best_iou_idx]}  |  "
        f"Best F1(macro): {f1[best_f1_idx]:.4f} @ epoch {epochs[best_f1_idx]}  |  "
        f"Mean IoU(defect): {np.mean(iou_c1):.4f}  |  "
        f"Mean F1(macro): {np.mean(f1):.4f}"
    )
    fig.text(0.5, 0.01, summary_text, ha="center", fontsize=10,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#E3F2FD", ec="#90CAF9"))

    fig.suptitle(f"Training Metrics Summary — {experiment_name}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(os.path.join(vis_dir, "metrics_summary.png"), dpi=200)
    plt.close(fig)

    print(f"Metric curves saved to {vis_dir}")


# ---------------------------------------------------------------------------
# 数据增强可视化
# ---------------------------------------------------------------------------

def visualize_augmentations(cfg, output_dir, num_samples=3):
    """可视化各种数据增强方法的效果，原图与增强后对比。

    对每种增强方法分别生成一张对比图 (原图 | 增强后)，保存到
    output_dir/visualization/augmentations/ 目录。
    """
    data_dir = cfg["data"]["data_dir"]
    image_dir = cfg["data"]["image_dir"]
    mask_dir = cfg["data"]["mask_dir"]

    image_root = os.path.join(data_dir, image_dir)
    if not os.path.isdir(image_root):
        print(f"WARNING: image directory not found: {image_root}, skipping augment visualization.")
        return

    stems = sorted([f[:-4] for f in os.listdir(image_root) if f.endswith(".npy")])
    if not stems:
        print("WARNING: no .npy files found, skipping augment visualization.")
        return

    vis_dir = os.path.join(output_dir, "visualization", "augmentations")
    os.makedirs(vis_dir, exist_ok=True)

    vis_bands = tuple(cfg.get("eval", {}).get("vis_bands", [0, 4, 8]))
    num_samples = min(num_samples, len(stems))
    selected_stems = stems[:num_samples]

    # 定义要可视化的增强方法
    augment_methods = {
        "HorizontalFlip": RandomHorizontalFlip(p=1.0),
        "VerticalFlip": RandomVerticalFlip(p=1.0),
        "Rotation90": RandomRotation90(),
        "ElasticTransform": ElasticTransform(alpha=50, sigma=7, p=1.0),
        "Cutout": Cutout(num_holes=2, max_h_frac=0.3, max_w_frac=0.3, p=1.0),
        "GaussianBlur": GaussianBlur(kernel_range=(5, 5), sigma_range=(1.5, 1.5), p=1.0),
        "IntensityJitter": IntensityJitter(scale_range=(0.7, 1.3), p=1.0),
        "GaussianNoise": GaussianNoise(std=0.02, p=1.0),
    }

    def to_rgb(img, bands):
        """将多光谱图像的指定波段转为伪彩色 RGB。"""
        rgb = np.stack([img[b] for b in bands], axis=-1)
        for c in range(3):
            vmin, vmax = np.percentile(rgb[:, :, c], [2, 98])
            if vmax - vmin > 1e-6:
                rgb[:, :, c] = np.clip((rgb[:, :, c] - vmin) / (vmax - vmin), 0, 1)
            else:
                rgb[:, :, c] = 0.0
        return rgb

    for stem in selected_stems:
        image = np.load(os.path.join(image_root, stem + ".npy")).astype(np.float32)
        image = image.transpose(2, 0, 1)  # (C, H, W)

        mask_npy = os.path.join(data_dir, mask_dir, stem + ".npy")
        mask_png = os.path.join(data_dir, mask_dir, stem + ".png")
        if os.path.exists(mask_npy):
            mask = np.load(mask_npy).astype(np.int64)
        elif os.path.exists(mask_png):
            from PIL import Image
            mask = np.array(Image.open(mask_png)).astype(np.int64)
        else:
            mask = np.zeros(image.shape[1:], dtype=np.int64)

        n_methods = len(augment_methods)
        fig, axes = plt.subplots(n_methods, 4, figsize=(16, 3.5 * n_methods))

        for row, (name, aug) in enumerate(augment_methods.items()):
            np.random.seed(42)
            aug_image, aug_mask = aug(image.copy(), mask.copy())

            orig_rgb = to_rgb(image, vis_bands)
            aug_rgb = to_rgb(aug_image, vis_bands)

            # 原图
            axes[row, 0].imshow(orig_rgb)
            axes[row, 0].set_title("Original Image", fontsize=9)
            axes[row, 0].axis("off")

            # 原始mask
            axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("Original Mask", fontsize=9)
            axes[row, 1].axis("off")

            # 增强后图像
            axes[row, 2].imshow(aug_rgb)
            axes[row, 2].set_title(f"After {name}", fontsize=9)
            axes[row, 2].axis("off")

            # 增强后mask
            axes[row, 3].imshow(aug_mask, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].set_title(f"Mask after {name}", fontsize=9)
            axes[row, 3].axis("off")

            # 行标签
            axes[row, 0].set_ylabel(name, fontsize=10, rotation=0,
                                     labelpad=80, va="center")

        fig.suptitle(f"Data Augmentation Visualization — {stem}", fontsize=14)
        fig.tight_layout(rect=[0.08, 0, 1, 0.97])
        fig.savefig(os.path.join(vis_dir, f"augment_{stem}.png"), dpi=150)
        plt.close(fig)

    print(f"Augmentation visualizations saved to {vis_dir}")


# ---------------------------------------------------------------------------
# 评估阶段 (复用 eval.py 的逻辑)
# ---------------------------------------------------------------------------

def run_eval(cfg, seed, output_dir, splits=None):
    """Evaluate the best checkpoint and generate all evaluation outputs.

    Reports on the held-out TEST split when available (single 0.7/0.15/0.15
    split); falls back to the val split in k-fold mode where test is empty.

    Args:
        splits: Optional pre-computed splits dict (for k-fold mode).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"WARNING: best checkpoint not found at {ckpt_path}, skipping eval.")
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Data
    data_dir = cfg["data"]["data_dir"]
    if splits is None:
        splits = get_data_splits(
            data_dir=data_dir,
            image_dir=cfg["data"]["image_dir"],
            seed=seed,
        )

    eval_split = "test" if splits.get("test") else "val"
    print(f"  [eval] reporting on '{eval_split}' split ({len(splits[eval_split])} samples)")

    val_transform = get_val_transforms(cfg)
    ds_kwargs = get_dataset_kwargs(cfg)
    val_dataset = MSIDataset(
        splits[eval_split], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=val_transform,
        num_classes=cfg["data"]["num_classes"],
        **ds_kwargs,
    )
    _nw = cfg["train"].get("num_workers", 4)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=_nw,
        pin_memory=True,
        persistent_workers=_nw > 0,
        prefetch_factor=2 if _nw > 0 else None,
    )

    # Model
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_dir = os.path.join(output_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    # Metrics
    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    # Evaluate
    results, all_preds, all_masks, all_images, all_images_raw, all_stems = evaluate(
        model, val_loader, metrics, device, num_classes
    )

    # Print
    print_results(results, cfg["experiment_name"])

    # Save results JSON
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            results_serializable[k] = v.tolist()
        else:
            results_serializable[k] = v

    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(results_serializable, f, indent=2)

    # Confusion matrix
    plot_confusion_matrix(results, eval_dir, num_classes)

    # Visualization
    vis_bands = tuple(cfg.get("eval", {}).get("vis_bands", [0, 4, 8]))
    num_vis = cfg.get("eval", {}).get("num_vis_samples", 10)
    visualize_predictions(
        all_images, all_preds, all_masks, all_stems,
        eval_dir, vis_bands=vis_bands, num_samples=num_vis,
        images_raw=all_images_raw,
    )

    # TP/FP/FN error analysis overlay (use processed images — same spatial
    # dimensions as preds/masks, not the pre-transform raw images)
    visualize_error_analysis(
        all_images, all_preds, all_masks, all_stems,
        eval_dir, vis_bands=vis_bands, num_samples=num_vis,
    )

    print(f"Evaluation results saved to {eval_dir}")
    return results


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_single(cfg, seed, output_dir, experiment_name, args, splits=None, fold_tag=""):
    """运行单次 (config, seed[, fold]) 的完整 train→eval→viz 流程。

    支持断点续跑:
        - 若 output_dir/done.flag 已存在 → 跳过整个流程，从 eval_results.json
          读回上次的指标。
        - 训练阶段由 train.py 内部基于 last_checkpoint.pth 自动续跑。
    """
    os.makedirs(output_dir, exist_ok=True)
    done_flag = os.path.join(output_dir, "done.flag")

    # Run-level skip: full pipeline already completed before
    if os.path.exists(done_flag):
        print("=" * 70)
        print(f" [skip] Run already completed: {experiment_name}{fold_tag}")
        print(f"        Found {done_flag}")
        print("=" * 70)
        cached_eval_path = os.path.join(output_dir, "eval_results.json")
        if os.path.exists(cached_eval_path):
            with open(cached_eval_path, "r") as _f:
                return json.load(_f)
        return None

    # Save config copy
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("=" * 70)
    print(f" Automated Train-Eval Workflow")
    print(f" Experiment: {experiment_name}{fold_tag}")
    print(f" Seed: {seed}")
    print(f" Output: {output_dir}")
    print("=" * 70)

    # Step 1: Training (train.py handles per-epoch resume internally)
    if not args.skip_train:
        print("\n[Step 1/4] Training...")
        train_result = train(cfg, seed, output_dir, splits=splits)
        print(f"Training result: {train_result}")
    else:
        print("\n[Step 1/4] Training skipped (--skip_train)")

    # Step 2: Metric curves
    log_path = os.path.join(output_dir, "training_log.json")
    if os.path.exists(log_path):
        print("\n[Step 2/4] Generating metric curves...")
        plot_metric_curves(log_path, output_dir, experiment_name + fold_tag)
    else:
        print("\n[Step 2/4] No training_log.json found, skipping curve plotting.")

    # Step 3: Evaluation on the held-out split (test for single split, val for k-fold)
    eval_results = None
    if not args.skip_eval:
        print("\n[Step 3/4] Evaluating on held-out split...")
        eval_results = run_eval(cfg, seed, output_dir, splits=splits)
    else:
        print("\n[Step 3/4] Evaluation skipped (--skip_eval)")

    # Step 4: Augmentation visualization
    if args.vis_augment:
        print("\n[Step 4/4] Generating augmentation visualizations...")
        visualize_augmentations(cfg, output_dir, num_samples=3)
    else:
        print("\n[Step 4/4] Augmentation visualization skipped (use --vis_augment to enable)")

    # Cache eval results + mark run as done so reruns are skipped.
    if eval_results is not None:
        try:
            cached_path = os.path.join(output_dir, "eval_results.json")
            with open(cached_path, "w") as _f:
                json.dump(eval_results, _f, indent=2, default=str)
        except Exception as _e:
            print(f"  (skipped eval_results.json: {_e})")

    with open(done_flag, "w") as _f:
        _f.write(f"completed_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        _f.write(f"experiment={experiment_name}{fold_tag}\n")
        _f.write(f"seed={seed}\n")

    print("\n" + "=" * 70)
    print(f" Run complete: {experiment_name}{fold_tag}")
    print(f" Outputs: {output_dir}")
    print("=" * 70)
    return eval_results


def aggregate_kfold_results(fold_results, output_dir, experiment_name, n_splits):
    """聚合 k-fold 结果：清晰美观的 per-fold 表格 + mean ± std 汇总 + JSON + PNG。

    控制台输出示例：
        ╔══════════════════════════════════════════════════════════════════╗
        ║   K-Fold Summary — baseline   (5 folds)                          ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║ Fold |  IoU(c1) | mIoU   | F1     | Prec   | Recall              ║
        ║   1  |  0.7240  | 0.8330 | 0.8821 | 0.8915 | 0.8732              ║
        ║   2  |  0.7185  | 0.8295 | 0.8784 | 0.8870 | 0.8702              ║
        ...
        ║ mean |  0.7196  | 0.8307 | 0.8800 | 0.8893 | 0.8718              ║
        ║ std  |  0.0042  | 0.0017 | 0.0023 | 0.0030 | 0.0021              ║
        ╚══════════════════════════════════════════════════════════════════╝
    """
    primary = ["IoU_class1", "mIoU", "F1_macro", "Precision_macro", "Recall_macro"]
    headers = ["IoU(c1)", "mIoU", "F1", "Prec", "Recall"]

    # 解构每个 fold 的指标
    per_fold_rows = []
    for k, r in enumerate(fold_results):
        if r is None:
            per_fold_rows.append(None)
            continue
        iou_c1 = float(r["IoU_per_class"][1]) if "IoU_per_class" in r and len(r["IoU_per_class"]) > 1 else float("nan")
        per_fold_rows.append({
            "IoU_class1": iou_c1,
            "mIoU": float(r.get("mIoU", float("nan"))),
            "F1_macro": float(r.get("F1_macro", float("nan"))),
            "Precision_macro": float(r.get("Precision_macro", float("nan"))),
            "Recall_macro": float(r.get("Recall_macro", float("nan"))),
        })

    aggregated = {}
    for key in primary:
        vals = [row[key] for row in per_fold_rows if row is not None and not np.isnan(row[key])]
        if vals:
            aggregated[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": [float(v) for v in vals],
            }

    summary_path = os.path.join(output_dir, "kfold_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "n_splits": n_splits,
            "metrics": aggregated,
            "per_fold": per_fold_rows,
        }, f, indent=2)

    # ---- Pretty 控制台表格 ----
    col_w = 9
    label_w = 4
    title = f" K-Fold Summary — {experiment_name}   ({n_splits} folds) "

    def _format_row(label, cells):
        # label: 4 chars right-justified; cells: list of formatted col_w-char strings
        return (" " + label.rjust(label_w) + " │ "
                + " │ ".join(cells) + " ")

    head_cells = [f"{h:>{col_w}}" for h in headers]
    head = _format_row("Fold", head_cells)
    inner_w = max(len(head), len(title))

    bar_top = "╔" + "═" * inner_w + "╗"
    bar_mid = "╠" + "═" * inner_w + "╣"
    bar_bot = "╚" + "═" * inner_w + "╝"

    def _value_row(label, vals):
        cells = [
            f"{v:>{col_w}.4f}" if v is not None and not np.isnan(v) else f"{'NaN':>{col_w}}"
            for v in vals
        ]
        return "║" + _format_row(label, cells).ljust(inner_w) + "║"

    print()
    print(bar_top)
    print("║" + title.center(inner_w) + "║")
    print(bar_mid)
    print("║" + head.ljust(inner_w) + "║")
    print(bar_mid)

    best_idx = None
    if "IoU_class1" in aggregated and aggregated["IoU_class1"]["values"]:
        ious = [row["IoU_class1"] if row is not None else float("-inf") for row in per_fold_rows]
        best_idx = int(np.argmax(ious))

    for fi, row in enumerate(per_fold_rows):
        marker = "*" if fi == best_idx else ""
        label = f"{fi + 1}{marker}"
        if row is None:
            print("║" + (" " + label.rjust(label_w) + " │ (no result)").ljust(inner_w) + "║")
            continue
        print(_value_row(label, [row[m] for m in primary]))

    print(bar_mid)
    print(_value_row("mean", [aggregated[m]["mean"] if m in aggregated else float("nan") for m in primary]))
    print(_value_row("std", [aggregated[m]["std"] if m in aggregated else float("nan") for m in primary]))
    print(bar_bot)

    if best_idx is not None:
        print(f"  * = best IoU(c1) fold (fold {best_idx+1})")
    print(f"  Detail JSON  -> {summary_path}")

    # ---- 同时输出 markdown 表格文件，方便复制到论文/报告 ----
    md_path = os.path.join(output_dir, "kfold_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# K-Fold Summary — {experiment_name} ({n_splits} folds)\n\n")
        f.write("| Fold | " + " | ".join(headers) + " |\n")
        f.write("|------|" + "|".join(["------"] * len(headers)) + "|\n")
        for fi, row in enumerate(per_fold_rows):
            if row is None:
                f.write(f"| {fi+1} | " + " | ".join(["NaN"] * len(headers)) + " |\n")
                continue
            cells = [f"{row[m]:.4f}" for m in primary]
            f.write(f"| {fi+1} | " + " | ".join(cells) + " |\n")
        f.write("| **mean** | " + " | ".join(
            f"**{aggregated[m]['mean']:.4f}**" if m in aggregated else "NaN" for m in primary
        ) + " |\n")
        f.write("| std  | " + " | ".join(
            f"{aggregated[m]['std']:.4f}" if m in aggregated else "NaN" for m in primary
        ) + " |\n")
    print(f"  Markdown     -> {md_path}")

    # ---- 可视化：per-fold 柱状图 + mean±std ----
    try:
        png_path = os.path.join(output_dir, "kfold_summary.png")
        fig, ax = plt.subplots(figsize=(max(8, n_splits * 1.0), 5))
        x = np.arange(n_splits)
        n_metrics = len(primary)
        bar_w = 0.8 / n_metrics
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4"]
        for mi, (mkey, mlabel) in enumerate(zip(primary, headers)):
            vals = [row[mkey] if row is not None else 0.0 for row in per_fold_rows]
            ax.bar(x + mi * bar_w - 0.4 + bar_w / 2, vals, bar_w,
                   color=colors[mi % len(colors)], label=mlabel, edgecolor="black", linewidth=0.4)
            if mkey in aggregated:
                ax.axhline(aggregated[mkey]["mean"], color=colors[mi % len(colors)],
                           linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(n_splits)])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"K-Fold Cross-Validation — {experiment_name} ({n_splits} folds)")
        ax.legend(loc="lower right", ncol=n_metrics, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # bottom annotation: mean±std
        summary_line = "  ".join(
            f"{lbl}: {aggregated[mkey]['mean']:.4f}±{aggregated[mkey]['std']:.4f}"
            for mkey, lbl in zip(primary, headers) if mkey in aggregated
        )
        fig.text(0.5, 0.01, summary_line, ha="center", fontsize=9,
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#90CAF9"))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        print(f"  Bar chart    -> {png_path}")
    except Exception as e:
        print(f"  (skipped bar chart: {e})")


def main():
    parser = argparse.ArgumentParser(
        description="自动化训练→评估→可视化工作流（支持 k-fold 交叉验证）"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/<experiment>_seed<seed>)")
    parser.add_argument("--vis_augment", action="store_true",
                        help="Generate augmentation visualization")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, only run eval + plotting (requires existing checkpoint)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, only run training + plotting")
    parser.add_argument("--kfold", type=int, default=0,
                        help="K-fold cross validation (0 = disabled, default 7:3 split). "
                             "If >0, runs k folds and aggregates results.")
    args = parser.parse_args()

    # Load config (with _base inheritance)
    from utils.config import load_config
    cfg = load_config(args.config)

    experiment_name = cfg["experiment_name"]

    # ===== K-fold mode =====
    if args.kfold > 0:
        n_splits = args.kfold
        if args.output_dir is None:
            args.output_dir = os.path.join(
                "outputs", f"{experiment_name}_seed{args.seed}_kfold{n_splits}"
            )
        os.makedirs(args.output_dir, exist_ok=True)

        # Generate fold splits
        folds = get_kfold_splits(
            data_dir=cfg["data"]["data_dir"],
            image_dir=cfg["data"]["image_dir"],
            n_splits=n_splits,
            seed=args.seed,
        )

        print("=" * 70)
        print(f" K-Fold Cross-Validation Mode")
        print(f" Experiment: {experiment_name}")
        print(f" Seed: {args.seed}")
        print(f" Folds: {n_splits}")
        print(f" Output: {args.output_dir}")
        print("=" * 70)

        fold_results = []
        for k, splits in enumerate(folds):
            fold_dir = os.path.join(args.output_dir, f"fold{k}")
            print(f"\n>>> Fold {k+1}/{n_splits}: train={len(splits['train'])}, "
                  f"val={len(splits['val'])}")
            result = run_single(
                cfg, args.seed, fold_dir, experiment_name, args,
                splits=splits, fold_tag=f"_fold{k}",
            )
            fold_results.append(result)

        # Aggregate
        aggregate_kfold_results(fold_results, args.output_dir, experiment_name, n_splits)
        return

    # ===== Single-split mode (default 7:3) =====
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs", f"{experiment_name}_seed{args.seed}"
        )
    run_single(cfg, args.seed, args.output_dir, experiment_name, args)


if __name__ == "__main__":
    main()

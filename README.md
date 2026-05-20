[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/yyyuan2004/main-msibruisenet)

# MSI-Bruise-Seg: Apple Multispectral Image Bruise Segmentation

Pixel-level semantic segmentation of apple bruise defects from 9-band near-infrared multispectral images (MSI).
基于近红外多光谱图像（MSI）的苹果瘀伤（defect）像素级语义分割项目。

---

## 1. Dataset

```
/root/autodl-tmp/datasets/185_9bands/
├── images/    # .npy, shape (H, W, 9), float32 reflectance
└── masks/     # .npy or .png, shape (H, W), 0=background / 1=bruise
```

- 185 annotated samples, 9 NIR bands (565-730 nm)
- `images/` and `masks/` filenames correspond one-to-one

> Change `data.data_dir` in `configs/_defaults.yaml` when switching datasets.

---

## 2. Config Inheritance System

All experiment configs inherit shared defaults via `_base`:

```yaml
# configs/baseline.yaml
_base: _defaults.yaml
experiment_name: "baseline"
model:
  encoder_name: "mobilenetv2"
  skip_module: "none"
  ...
```

**`configs/_defaults.yaml`** contains the shared data/train/augment/eval settings.
Each config only overrides what is unique (model architecture, special preprocessing).

To change a global hyperparameter (e.g., epochs, num_workers), edit `_defaults.yaml` once.

---

## 3. Unified Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, weight_decay=1e-4 |
| Learning Rate | 5e-4, CosineAnnealing to 1e-6 |
| Warmup | Optional (disabled by default) |
| Epochs | 300 |
| Batch Size | 32 |
| Loss | 0.5*CE + 0.5*Dice |
| Input Size | 512x512, random crop 384x384 |
| Augmentation | h-flip / v-flip / {90,180,270} rotation |
| Early Stopping | class-1 IoU, patience=70 |
| num_workers | 8 |
| Band selection | `band_indices=[0,2,4,5]`, `num_channels=4` (default) |

---

## 4. Model Zoo

**Custom MobileNetV2-UNet family**
| Config | Modification |
|--------|-------------|
| `baseline` | Pure MobileNetV2 + UNet decoder |
| `spconv_se` | + 1D SpectralConv after S1 + SE at every skip |

**SMP (segmentation_models_pytorch)**
| Config | Architecture |
|--------|-------------|
| `smp_unet_resnet18` | U-Net + ResNet18 |
| `smp_unet_resnet34` | U-Net + ResNet34 |
| `smp_unetplusplus_resnet34` | UNet++ + ResNet34 |
| `smp_linknet_resnet34` | Linknet + ResNet34 |
| `smp_manet_resnet34` | MAnet + ResNet34 |
| `smp_deeplabv3plus_mobilenetv2` | DeepLabV3+ + MobileNetV2 |
| `smp_fpn_efficientnetb0` | FPN + EfficientNet-B0 |

**Lightweight Transformers & Real-time**
| Config | Architecture | Params |
|--------|-------------|--------|
| `topformer_t/s/b` | TopFormer (CVPR 2022) | 1.4/3.1/5.1M |
| `seaformer_t/s/b` | SeaFormer (ICLR 2023) | 1.7/4/8M |
| `pidnet_s/m` | PIDNet (CVPR 2023) | 7.6/23M |

---

## 5. in_channels Adaptation Strategy

Different encoder families handle non-3-channel input differently:

| Encoder type | Strategy | Details |
|---|---|---|
| Custom (MobileNetV2/V3/EfficientNet-B0) | Copy first 3 pretrained channels + Kaiming init for extra channels | `encoder.py`: copies ImageNet conv1 weights for ch 0-2, applies `kaiming_normal_` for ch 3+ |
| SMP models | SMP built-in `in_channels` adaptation | Averages or repeats pretrained conv1 weights automatically (see `smp` library source) |
| TopFormer / SeaFormer / PIDNet | Same as custom | First 3 channels copied from pretrained, remainder Kaiming-initialized |

---

## 6. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# or for exact tested versions:
pip install -r requirements.lock.txt

# 2. Single config train+eval+plotting
python train_eval.py --config configs/baseline.yaml --seed 42

# 3. 5-fold cross-validation
python train_eval.py --config configs/baseline.yaml --seed 42 --kfold 5

# 4. Full ablation (16 configs x 3 seeds)
bash run_ablation.sh

# 5. Full ablation + k-fold
bash run_ablation.sh --kfold 5

# 6. Generate comparison plots only (after ablation)
python scripts/plot_ablation.py

# 7. HSI band range search (204-channel)
bash run_band_search.sh
```

---

## 7. Project Structure

```
.
├── configs/
│   ├── _defaults.yaml              # Shared base config (all others inherit)
│   ├── baseline.yaml               # MobileNetV2-UNet
│   ├── spconv_se.yaml              # + SpConv + SE
│   ├── smp_*.yaml                  # 6 SMP configs
│   ├── topformer_*.yaml            # 3 TopFormer configs
│   ├── seaformer_*.yaml            # 3 SeaFormer configs
│   └── pidnet_*.yaml               # 2 PIDNet configs
├── data/
│   ├── dataset.py                  # MSIDataset: .npy loading + band selection
│   ├── augment.py                  # Spatial augmentations (image/mask sync)
│   └── split.py                    # 7:3 split / k-fold
├── model/
│   ├── encoder.py                  # MobileNetV2/V3/EfficientNet-B0 (in_ch adapt)
│   ├── decoder.py                  # UNet decoder (skip: none/se)
│   ├── modules.py                  # SEBlock, SpectralConv1D
│   ├── smp_models.py              # SMP wrapper
│   ├── topformer.py / seaformer.py / pidnet.py
│   ├── model.py                    # build_model factory
│   └── loss.py                     # CE+Dice / Focal
├── utils/
│   ├── config.py                   # Config loader with _base inheritance
│   ├── metrics.py                  # IoU / F1 / Precision / Recall
│   └── spectral_analysis.py       # Spectral pre-analysis
├── scripts/
│   ├── band_range_search.py        # C(9,k) band exhaustive search
│   ├── plot_ablation.py            # Publication-quality comparison plots
│   └── ...
├── train.py                        # Training loop (resume, early stopping)
├── eval.py                         # Evaluation + TP/FP/FN error analysis
├── train_eval.py                   # One-click: train -> eval -> plots (+ k-fold)
├── run_ablation.sh                 # Full ablation automation
├── run_band_search.sh              # HSI band search automation
├── aggregate_results.py            # Aggregate multi-seed results
├── requirements.txt                # Min version dependencies
└── requirements.lock.txt           # Pinned tested versions
```

---

## 8. Evaluation & Visualization

Each (config, seed) run produces:
- **Metrics**: class-1 IoU (primary), mIoU, F1, Precision, Recall
- **Confusion matrix** PNG
- **Prediction visualization**: per-sample raw bands + pseudo-color + pred vs GT
- **Error analysis overlay**: per-sample TP (green) / FP (red) / FN (blue) overlay

After full ablation, `scripts/plot_ablation.py` generates:
- IoU/F1/mIoU box plots across all models
- Grouped bar chart (all metrics side-by-side)
- Params vs IoU scatter (efficiency frontier)

---

## 9. Resume & Checkpointing

- **Training resume**: `last_checkpoint.pth` saved every 10 epochs; on restart, training continues from last saved state (model + optimizer + scheduler + RNG).
- **Run-level skip**: `done.flag` marks completed runs; `run_ablation.sh` automatically skips them.
- **Band search resume**: per-combo partial JSON with atomic writes.
- **`training_done.flag`**: written after training completes; skips the entire training loop on re-run.

---

## 10. Output Structure

**Single split (7:3):**
```
outputs/<config>_seed<seed>/
├── checkpoints/best_model.pth
├── training_log.json
├── visualization/
│   ├── loss_curve.png
│   ├── iou_f1_curve.png
│   └── metrics_summary.png
└── eval_results/
    ├── results.json
    ├── confusion_matrix.png
    ├── visualizations/           # prediction comparison
    └── error_analysis/           # TP/FP/FN overlays
```

**After ablation:**
```
outputs/
├── ablation_table.txt
└── ablation_plots/
    ├── iou_boxplot.png
    ├── f1_boxplot.png
    ├── grouped_bar.png
    └── params_vs_iou.png
```

---

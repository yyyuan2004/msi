"""Spectral pre-analysis: correlation matrices, spectral curves, PCA, and linear regression.

Run this script FIRST after receiving data to understand spectral characteristics.
Outputs diagnostic plots and tables that inform experiment design.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from data.dataset import get_file_stems


# NIR wavelength labels (approximate, 23nm spacing)
BAND_LABELS = [f"Band {i+1}" for i in range(9)]


def load_all_data(data_dir, image_dir="images", mask_dir="masks"):
    """Load all spectral images and masks."""
    stems = get_file_stems(data_dir, image_dir)
    print(f"Found {len(stems)} samples")

    images = []
    masks = []
    for stem in stems:
        img = np.load(os.path.join(data_dir, image_dir, stem + ".npy")).astype(np.float32)
        # Try .npy first, then .png
        mask_npy = os.path.join(data_dir, mask_dir, stem + ".npy")
        mask_png = os.path.join(data_dir, mask_dir, stem + ".png")
        if os.path.exists(mask_npy):
            mask = np.load(mask_npy).astype(np.int64)
        elif os.path.exists(mask_png):
            from PIL import Image
            mask = np.array(Image.open(mask_png)).astype(np.int64)
        else:
            raise FileNotFoundError(f"Mask not found for {stem}")
        images.append(img)
        masks.append(mask)

    return images, masks, stems


def analyze_correlation(images, masks, stems, data_dir, output_dir):
    """Compute and plot 9x9 band correlation matrices for normal and defect regions."""
    print("Computing band correlation matrices...")

    # Collect pixels by class
    normal_pixels = []
    defect_pixels = []

    for img, mask, stem in zip(images, masks, stems):
        # Load whole apple mask
        apple_npy = os.path.join(data_dir, "whole", stem + ".npy")
        if os.path.exists(apple_npy):
            apple_mask = np.load(apple_npy).astype(np.int64)
        else:
            apple_mask = (img.mean(axis=2) > 0.05).astype(np.int64)
    
        healthy_mask = (apple_mask > 0) & (mask == 0)
        defect_mask = mask > 0

        if healthy_mask.any():
            normal_pixels.append(img[healthy_mask])
        if defect_mask.any():
            defect_pixels.append(img[defect_mask])

    normal_pixels = np.concatenate(normal_pixels, axis=0) if normal_pixels else np.empty((0, 9))
    defect_pixels = np.concatenate(defect_pixels, axis=0) if defect_pixels else np.empty((0, 9))

    print(f"  Normal pixels: {len(normal_pixels)}, Defect pixels: {len(defect_pixels)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pixels, title in [
        (axes[0], normal_pixels, "Normal Region"),
        (axes[1], defect_pixels, "Defect Region"),
    ]:
        if len(pixels) > 0:
            corr = np.corrcoef(pixels.T)
            sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, center=0, cmap="RdBu_r",
                        annot=True, fmt=".2f", xticklabels=BAND_LABELS,
                        yticklabels=BAND_LABELS, square=True)
        ax.set_title(f"Band Correlation — {title}")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "band_correlation.png"), dpi=200)
    plt.close(fig)
    print("  Saved band_correlation.png")

    return normal_pixels, defect_pixels


def analyze_spectral_curves(normal_pixels, defect_pixels, output_dir):
    """Plot mean spectral curves with std for normal and defect regions."""
    print("Computing spectral curves...")

    fig, ax = plt.subplots(figsize=(10, 6))
    bands = np.arange(1, 10)

    if len(normal_pixels) > 0:
        mean_n = normal_pixels.mean(axis=0)
        std_n = normal_pixels.std(axis=0)
        ax.plot(bands, mean_n, "b-o", label="Normal (mean)")
        ax.fill_between(bands, mean_n - std_n, mean_n + std_n, alpha=0.2, color="blue")

    if len(defect_pixels) > 0:
        mean_d = defect_pixels.mean(axis=0)
        std_d = defect_pixels.std(axis=0)
        ax.plot(bands, mean_d, "r-s", label="Defect (mean)")
        ax.fill_between(bands, mean_d - std_d, mean_d + std_d, alpha=0.2, color="red")

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Reflectance")
    ax.set_title("Mean Spectral Curves: Normal vs Defect")
    ax.set_xticks(bands)
    ax.set_xticklabels(BAND_LABELS, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectral_curves.png"), dpi=200)
    plt.close(fig)
    print("  Saved spectral_curves.png")


def analyze_pca(normal_pixels, defect_pixels, output_dir):
    """PCA cumulative variance explained ratio for all 9 components."""
    print("Computing PCA...")

    all_pixels = np.concatenate(
        [p for p in [normal_pixels, defect_pixels] if len(p) > 0], axis=0
    )

    # Subsample if too many pixels to speed up PCA
    if len(all_pixels) > 500000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_pixels), 500000, replace=False)
        all_pixels = all_pixels[idx]

    pca = PCA(n_components=9)
    pca.fit(all_pixels)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    print("  PCA Cumulative Variance Explained:")
    for i in range(9):
        print(f"    PC{i+1}: {cumvar[i]:.4f} ({pca.explained_variance_ratio_[i]:.4f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, 10), pca.explained_variance_ratio_, alpha=0.6, label="Individual")
    ax.plot(range(1, 10), cumvar, "r-o", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained Ratio")
    ax.set_title("PCA on 9-Band Spectral Data")
    ax.set_xticks(range(1, 10))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_variance.png"), dpi=200)
    plt.close(fig)
    print("  Saved pca_variance.png")

    # Save table
    with open(os.path.join(output_dir, "pca_table.txt"), "w") as f:
        f.write("Component | Individual Var | Cumulative Var\n")
        f.write("-" * 50 + "\n")
        for i in range(9):
            f.write(f"PC{i+1:d}       | {pca.explained_variance_ratio_[i]:.6f}       "
                    f"| {cumvar[i]:.6f}\n")

    return pca


def analyze_3band_regression(normal_pixels, defect_pixels, output_dir):
    """3-band linear regression R^2: use bands 1/5/9 to predict the remaining 6."""
    print("Computing 3-band linear regression R²...")

    source_bands = [0, 4, 8]  # Band 1, 5, 9 (0-indexed)
    target_bands = [i for i in range(9) if i not in source_bands]

    results = {}
    for region_name, pixels in [("Normal", normal_pixels), ("Defect", defect_pixels)]:
        if len(pixels) == 0:
            continue

        # Subsample for speed
        if len(pixels) > 500000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(pixels), 500000, replace=False)
            pixels = pixels[idx]

        X = pixels[:, source_bands]
        r2_scores = {}
        for t in target_bands:
            y = pixels[:, t]
            reg = LinearRegression()
            reg.fit(X, y)
            r2 = reg.score(X, y)
            r2_scores[f"Band {t+1}"] = r2

        results[region_name] = r2_scores

    # Print and save
    with open(os.path.join(output_dir, "regression_r2.txt"), "w") as f:
        header = f"{'Target Band':<15}"
        for region in results:
            header += f"| {region:>10} "
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for t in target_bands:
            band_name = f"Band {t+1}"
            line = f"{band_name:<15}"
            for region in results:
                r2 = results[region].get(band_name, float("nan"))
                line += f"| {r2:>10.4f} "
            f.write(line + "\n")
            print(f"  {line}")

    print("  Saved regression_r2.txt")
    return results


def main():
    parser = argparse.ArgumentParser(description="Spectral pre-analysis for MSI data")
    parser.add_argument("--data_dir", type=str, default="/home/yy/datasets/153",
                        help="Root data directory")
    parser.add_argument("--image_dir", type=str, default="images",
                        help="Subdirectory for spectral images")
    parser.add_argument("--mask_dir", type=str, default="masks",
                        help="Subdirectory for masks")
    parser.add_argument("--output_dir", type=str, default="outputs/spectral_analysis",
                        help="Output directory for plots and tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    images, masks, stems = load_all_data(args.data_dir, args.image_dir, args.mask_dir)

    # 1. Band correlation matrices
    normal_pixels, defect_pixels = analyze_correlation(images, masks, stems, args.data_dir, args.output_dir)

    # 2. Mean spectral curves
    analyze_spectral_curves(normal_pixels, defect_pixels, args.output_dir)

    # 3. PCA cumulative variance
    analyze_pca(normal_pixels, defect_pixels, args.output_dir)

    # 4. 3-band linear regression R²
    analyze_3band_regression(normal_pixels, defect_pixels, args.output_dir)

    print(f"\nAll analysis outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

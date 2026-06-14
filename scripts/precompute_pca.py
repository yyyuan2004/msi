"""Precompute a PCA projection matrix for MSI data.

Prefer the integrated, leak-free driver instead of this script:
    python scripts/run_pca_baseline.py --data_dir <root> --seed 42
which fits PCA on the train split only, writes the results package, and trains.

This standalone script is kept for ad-hoc use. By default it fits on the TRAIN
split (seed-matched to training) to avoid val/test leakage; pass --all_images to
fit on every image (NOT recommended — leaks val/test spectra).

Usage:
    python scripts/precompute_pca.py --data_dir /root/autodl-tmp/datasets/185_9bands \
        --seed 42 --output pca_matrix.npz

The output .npz contains:
    - components: (n_components, 9) PCA projection matrix
    - mean: (9,) per-channel mean for centering
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pca_transform import compute_pca_matrix
from data.split import get_data_splits


def main():
    parser = argparse.ArgumentParser(
        description="Precompute PCA projection matrix for MSI data")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory (contains images/ and whole/)")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--whole_dir", type=str, default="whole")
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--max_pixels", type=int, default=500000)
    parser.add_argument("--output", type=str, default="pca_matrix.npz")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the train split and pixel sampling")
    parser.add_argument("--all_images", action="store_true",
                        help="Fit on ALL images instead of the train split "
                             "(leaks val/test spectra — not recommended)")
    args = parser.parse_args()

    file_list = None
    if not args.all_images:
        splits = get_data_splits(data_dir=args.data_dir, image_dir=args.image_dir,
                                 seed=args.seed)
        file_list = sorted(splits["train"])
        print(f"Fitting PCA on the TRAIN split only: {len(file_list)} images (seed={args.seed}).")
    else:
        print("WARNING: --all_images fits PCA on every image (val/test leakage).")

    components, mean = compute_pca_matrix(
        data_dir=args.data_dir, image_dir=args.image_dir, whole_dir=args.whole_dir,
        n_components=args.n_components, max_pixels=args.max_pixels, seed=args.seed,
        file_list=file_list)

    np.savez(args.output, components=components, mean=mean)
    print(f"\nPCA matrix saved to: {args.output}")
    print(f"  components shape: {components.shape}")
    print(f"  mean shape: {mean.shape}")


if __name__ == "__main__":
    main()

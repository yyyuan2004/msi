"""Compute a per-band HSI prior (defect-vs-normal separability) on the TRAIN split.

The output .npz feeds DiagonalBandGate via ``model.band_gate_prior_path``. The
prior is computed on the training split ONLY (matching the gate's split seed),
so no val/test label information leaks into the gate.

Usage:
    python scripts/compute_band_prior.py \
        --config configs/gate.yaml \
        --data_dir /root/autodl-tmp/datasets/185_9bands \
        --seed 42 --metric fisher \
        --output outputs/band_prior/band_prior_seed42_fisher.npz

    # per-fold prior (k-fold experiments)
    python scripts/compute_band_prior.py --config configs/gate.yaml \
        --kfold 5 --fold 0 --seed 42
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.split import get_data_splits, get_kfold_splits
from utils.config import load_config
from utils.spectral_analysis import gather_class_pixels, compute_band_prior


def _load_mask(mask_root, stem):
    npy = os.path.join(mask_root, stem + ".npy")
    png = os.path.join(mask_root, stem + ".png")
    if os.path.exists(npy):
        return np.load(npy).astype(np.int64)
    if os.path.exists(png):
        return np.array(Image.open(png)).astype(np.int64)
    raise FileNotFoundError(f"Mask not found for '{stem}'")


def load_split(data_dir, image_dir, mask_dir, stems):
    image_root = os.path.join(data_dir, image_dir)
    mask_root = os.path.join(data_dir, mask_dir)
    images, masks = [], []
    for stem in stems:
        images.append(np.load(os.path.join(image_root, stem + ".npy")).astype(np.float32))
        masks.append(_load_mask(mask_root, stem))
    return images, masks


def main():
    parser = argparse.ArgumentParser(description="Train-only per-band HSI prior")
    parser.add_argument("--config", type=str, default="configs/gate.yaml")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data.data_dir from config")
    parser.add_argument("--seed", type=int, default=42,
                        help="Split seed (MUST match the gate training seed)")
    parser.add_argument("--kfold", type=int, default=0,
                        help="If >0, use k-fold splits; pick --fold")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--metric", type=str, default="fisher",
                        choices=["fisher", "cohens_d", "auc_sep", "refl_gap", "auc"])
    parser.add_argument("--band_indices", type=str, default=None,
                        help="Comma list of candidate band indices (default: all)")
    parser.add_argument("--max_pixels", type=int, default=300000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data"]["data_dir"]
    image_dir = cfg["data"]["image_dir"]
    mask_dir = cfg["data"]["mask_dir"]

    band_indices = None
    if args.band_indices:
        band_indices = [int(x) for x in args.band_indices.split(",")]

    if args.kfold > 0:
        folds = get_kfold_splits(data_dir, image_dir, n_splits=args.kfold, seed=args.seed)
        train_stems = folds[args.fold]["train"]
        split_tag = f"kfold{args.kfold}_f{args.fold}_seed{args.seed}"
    else:
        splits = get_data_splits(data_dir, image_dir, seed=args.seed)
        train_stems = splits["train"]
        split_tag = f"seed{args.seed}"

    print(f"Computing prior on {len(train_stems)} TRAIN samples ({split_tag})")
    images, masks = load_split(data_dir, image_dir, mask_dir, train_stems)
    normal, defect = gather_class_pixels(images, masks, train_stems, data_dir, band_indices)
    print(f"  normal pixels: {len(normal)}, defect pixels: {len(defect)}")

    result = compute_band_prior(normal, defect, metric=args.metric,
                                max_pixels=args.max_pixels, seed=args.seed)

    n_bands = len(result["prior"])
    bands = band_indices if band_indices is not None else list(range(n_bands))

    print(f"\nPer-band separability (metric for prior = {args.metric}):")
    print(f"  {'band':>6} {'fisher':>10} {'cohens_d':>10} {'auc':>8} {'prior(z)':>10}")
    for i in range(n_bands):
        print(f"  {bands[i]:>6} {result['fisher'][i]:>10.4f} "
              f"{result['cohens_d'][i]:>10.4f} {result['auc'][i]:>8.4f} "
              f"{result['prior'][i]:>10.4f}")
    ranked = np.argsort(result["prior"])[::-1]
    print(f"\n  Top bands by prior: {[bands[i] for i in ranked]}")

    output = args.output or os.path.join(
        "outputs", "band_prior", f"band_prior_{split_tag}_{args.metric}.npz")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez(
        output,
        prior=result["prior"].astype(np.float32),
        fisher=result["fisher"],
        cohens_d=result["cohens_d"],
        refl_gap=result["refl_gap"],
        auc=result["auc"],
        auc_sep=result["auc_sep"],
        bands=np.array(bands),
        metric=args.metric,
        seed=args.seed,
    )
    print(f"\nSaved prior to {output}")
    print(f"Set this in your gate config:\n  model.band_gate_prior_path: \"{output}\"")


if __name__ == "__main__":
    main()

"""Leak-free PCA fitting + results package for the PCA baseline.

The PCA projection used by ``configs/pca_baseline.yaml`` must be fit on the
TRAINING split only — fitting on all 185 images leaks val/test spectra into the
9->3 projection. This module fits PCA on the apple-region pixels of a given list
of training stems, writes the model's ``pca_matrix.npz`` (components + mean), and
emits a self-contained results package describing the projection and the
underlying spectra.

Package CSVs (written to ``out_dir``):
    pca_loadings_and_band_contributions.csv  loadings V[pc,band] and squared contributions
    pca_explained_variance.csv               per-component variance + cumulative ratio
    band_statistics_by_region.csv            apple / healthy / defect per-band stats
    defect_healthy_feature_separability.csv  per-band Cohen's d + descriptive pixel AUC
    pca_reconstruction_error.csv             per-band 9->3 information loss
    apple_region_band_correlation.csv        9x9 source-band correlation (apple pixels)
    dataset_per_image_audit.csv              per-image apple/defect pixel counts + split

Everything except the per-image audit is computed from TRAIN pixels only. The
audit covers every image but labels each with its split, so it doubles as the
record of exactly which images were used to fit the PCA.

Pure numpy + scikit-learn; importable and testable without torch/GPU.
"""

import csv
import os

import numpy as np


# --------------------------------------------------------------------------- #
# IO helpers (mirror data/dataset.py mask conventions)
# --------------------------------------------------------------------------- #
def load_image(data_dir, image_dir, stem):
    """Load a spectral image as (H, W, C) float32."""
    return np.load(os.path.join(data_dir, image_dir, stem + ".npy")).astype(np.float32)


def load_apple_mask(data_dir, whole_dir, stem, image, threshold=0.05):
    """Apple-region boolean mask (H, W): whole/<stem>.npy (>0) or threshold fallback."""
    whole_path = os.path.join(data_dir, whole_dir, stem + ".npy")
    if os.path.exists(whole_path):
        m = np.load(whole_path)
        if m.ndim == 3:
            m = m[..., 0]
        return m > 0, "whole_mask_npy"
    return image.mean(axis=-1) > threshold, f"threshold_fallback(thr={threshold})"


def load_defect_mask(data_dir, mask_dir, stem):
    """Defect boolean mask (H, W) from masks/<stem>.npy or .png (>0)."""
    npy_path = os.path.join(data_dir, mask_dir, stem + ".npy")
    png_path = os.path.join(data_dir, mask_dir, stem + ".png")
    if os.path.exists(npy_path):
        m = np.load(npy_path)
    elif os.path.exists(png_path):
        from PIL import Image
        m = np.array(Image.open(png_path))
    else:
        raise FileNotFoundError(f"Mask not found for '{stem}' ({npy_path} / {png_path})")
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0


def band_wavelengths(num_bands, wl_start, wl_end):
    if num_bands <= 1:
        return [float(wl_start)]
    step = (wl_end - wl_start) / (num_bands - 1)
    return [round(wl_start + step * i, 1) for i in range(num_bands)]


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _f(x, nd=6):
    """Format a float for CSV, NaN-safe."""
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# --------------------------------------------------------------------------- #
# Pixel collection
# --------------------------------------------------------------------------- #
def _subsample(arr, cap, rng):
    if cap is not None and len(arr) > cap:
        idx = rng.choice(len(arr), cap, replace=False)
        return arr[idx]
    return arr


def collect_region_pixels(data_dir, image_dir, mask_dir, whole_dir, train_stems,
                          apple_cap=500_000, defect_cap=200_000, healthy_cap=500_000,
                          threshold=0.05, seed=42, band_indices=None):
    """Gather apple / healthy / defect pixel samples (N, C) over the training stems.

    Each image is subsampled so the per-region totals stay bounded; the caps are
    applied again at the end so the final arrays never exceed them. ``band_indices``
    mirrors the dataset's pre-PCA band selection (None = use all bands).
    """
    rng = np.random.RandomState(seed)
    n = max(1, len(train_stems))
    apple_per = max(1, apple_cap // n)
    healthy_per = max(1, healthy_cap // n)
    apple_parts, healthy_parts, defect_parts = [], [], []

    for stem in train_stems:
        img = load_image(data_dir, image_dir, stem)              # (H, W, C_raw)
        apple, _ = load_apple_mask(data_dir, whole_dir, stem, img, threshold)
        defect = load_defect_mask(data_dir, mask_dir, stem)
        if defect.shape != apple.shape:
            raise ValueError(f"mask/apple shape mismatch for '{stem}': "
                             f"{defect.shape} vs {apple.shape}")
        if band_indices is not None:
            img = img[..., band_indices]                         # pre-PCA band selection
        healthy = apple & ~defect
        defect_in_apple = apple & defect

        apple_parts.append(_subsample(img[apple], apple_per, rng))
        healthy_parts.append(_subsample(img[healthy], healthy_per, rng))
        if defect_in_apple.any():
            defect_parts.append(img[defect_in_apple])

    apple_px = _subsample(np.concatenate(apple_parts, 0), apple_cap, rng)
    healthy_px = _subsample(np.concatenate(healthy_parts, 0), healthy_cap, rng)
    defect_px = (_subsample(np.concatenate(defect_parts, 0), defect_cap, rng)
                 if defect_parts else np.empty((0, apple_px.shape[1]), np.float32))
    return apple_px, healthy_px, defect_px


# --------------------------------------------------------------------------- #
# Per-CSV writers
# --------------------------------------------------------------------------- #
def _region_stats_rows(region, px, wl):
    rows = []
    C = px.shape[1] if px.size else len(wl)
    for b in range(C):
        col = px[:, b] if px.size else np.array([])
        if col.size:
            q = np.percentile(col, [25, 50, 75])
            rows.append([region, b, wl[b], col.size, _f(col.mean()), _f(col.std()),
                         _f(col.min()), _f(q[0]), _f(q[1]), _f(q[2]), _f(col.max())])
        else:
            rows.append([region, b, wl[b], 0, "", "", "", "", "", "", ""])
    return rows


def _auc(pos, neg):
    """Rank-based AUC (Mann-Whitney) of `pos` scoring higher than `neg`."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = (csum - (counts - 1) / 2.0)
    ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def write_package(out_dir, n_components, apple_px, healthy_px, defect_px,
                  audit_rows, wl, seed=42):
    """Fit PCA on apple pixels and write the npz + all package CSVs.

    Returns (components[n_components, C], mean[C]) for the deployment npz.
    """
    from sklearn.decomposition import PCA

    os.makedirs(out_dir, exist_ok=True)
    C = apple_px.shape[1]

    # Full-rank PCA on apple pixels (drives every projection-related CSV).
    pca = PCA(n_components=C)
    pca.fit(apple_px)
    V = pca.components_.astype(np.float64)               # (C, C) rows = PCs
    mean = pca.mean_.astype(np.float64)                  # (C,)
    evr = pca.explained_variance_ratio_
    ev = pca.explained_variance_

    # ---- 1. loadings + squared contributions ----
    rows = []
    for pc in range(C):
        for b in range(C):
            load = V[pc, b]
            rows.append([pc + 1, b, wl[b], _f(load), _f(load ** 2),
                         _f(evr[pc])])
    _write_csv(os.path.join(out_dir, "pca_loadings_and_band_contributions.csv"),
               ["pc", "band_index", "wavelength_nm", "loading",
                "squared_contribution", "explained_variance_ratio"], rows)

    # ---- 2. explained variance ----
    cum = np.cumsum(evr)
    rows = [[pc + 1, _f(ev[pc]), _f(evr[pc]), _f(cum[pc])] for pc in range(C)]
    _write_csv(os.path.join(out_dir, "pca_explained_variance.csv"),
               ["pc", "explained_variance", "explained_variance_ratio",
                "cumulative_explained_variance_ratio"], rows)

    # ---- 3. band statistics by region ----
    rows = (_region_stats_rows("apple", apple_px, wl)
            + _region_stats_rows("healthy", healthy_px, wl)
            + _region_stats_rows("defect", defect_px, wl))
    _write_csv(os.path.join(out_dir, "band_statistics_by_region.csv"),
               ["region", "band_index", "wavelength_nm", "n_pixels", "mean", "std",
                "min", "p25", "median", "p75", "max"], rows)

    # ---- 4. defect vs healthy separability ----
    rows = []
    for b in range(C):
        d = defect_px[:, b] if defect_px.size else np.array([])
        h = healthy_px[:, b] if healthy_px.size else np.array([])
        if d.size and h.size:
            md, mh = d.mean(), h.mean()
            nd, nh = len(d), len(h)
            pooled = np.sqrt(((nd - 1) * d.var(ddof=1) + (nh - 1) * h.var(ddof=1))
                             / max(nd + nh - 2, 1))
            cohens_d = (md - mh) / pooled if pooled > 0 else float("nan")
            rows.append([b, wl[b], _f(md), _f(mh), _f(abs(md - mh)),
                         _f(pooled), _f(cohens_d), _f(_auc(d, h), 4)])
        else:
            rows.append([b, wl[b], "", "", "", "", "", ""])
    _write_csv(os.path.join(out_dir, "defect_healthy_feature_separability.csv"),
               ["band_index", "wavelength_nm", "mean_defect", "mean_healthy",
                "abs_diff", "pooled_std", "cohens_d", "pixel_auc"], rows)

    # ---- 5. reconstruction error (9 -> n_components -> 9) ----
    Vk = V[:n_components]                                 # (k, C)
    centered = apple_px - mean
    recon = centered @ Vk.T @ Vk + mean                  # (N, C)
    resid = apple_px - recon
    var = apple_px.var(axis=0)
    res_var = resid.var(axis=0)
    rmse = np.sqrt((resid ** 2).mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - res_var / var
        retained = 1.0 - res_var / var
    rows = [[b, wl[b], _f(rmse[b]), _f(r2[b]), _f(res_var[b] / var[b] if var[b] else np.nan),
             _f(retained[b])] for b in range(C)]
    _write_csv(os.path.join(out_dir, "pca_reconstruction_error.csv"),
               ["band_index", "wavelength_nm", "rmse", "r2",
                "residual_variance_fraction", "retained_variance_fraction"], rows)

    # ---- 6. apple-region band correlation ----
    corr = np.corrcoef(apple_px, rowvar=False)
    header = ["band"] + [f"b{b}_{wl[b]:.0f}nm" for b in range(C)]
    rows = [[f"b{i}_{wl[i]:.0f}nm"] + [_f(corr[i, j], 4) for j in range(C)]
            for i in range(C)]
    _write_csv(os.path.join(out_dir, "apple_region_band_correlation.csv"), header, rows)

    # ---- 7. per-image audit ----
    _write_csv(os.path.join(out_dir, "dataset_per_image_audit.csv"),
               ["stem", "split", "height", "width", "total_pixels", "apple_pixels",
                "defect_pixels", "defect_in_apple_pixels", "defect_frac_of_apple",
                "apple_mask_source"], audit_rows)

    components = V[:n_components].astype(np.float32)
    return components, mean.astype(np.float32)


def audit_images(data_dir, image_dir, mask_dir, whole_dir, splits, threshold=0.05):
    """Per-image apple/defect counts for every image, labelled by split."""
    split_of = {}
    for name in ("train", "val", "test"):
        for s in splits.get(name, []):
            split_of[s] = name
    rows = []
    for stem in sorted(split_of):
        img = load_image(data_dir, image_dir, stem)
        apple, src = load_apple_mask(data_dir, whole_dir, stem, img, threshold)
        defect = load_defect_mask(data_dir, mask_dir, stem)
        H, W = apple.shape
        ap = int(apple.sum())
        df = int(defect.sum())
        dia = int((apple & defect).sum())
        rows.append([stem, split_of[stem], H, W, H * W, ap, df, dia,
                     _f(dia / ap if ap else np.nan, 6), src])
    return rows

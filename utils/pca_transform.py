"""PCA dimensionality reduction for 9-channel MSI data.

Provides functions to compute PCA projection matrix from training data
and apply it to reduce 9-channel spectral images to n_components channels.
"""

import os
import numpy as np


def compute_pca_matrix(data_dir, image_dir="images", whole_dir="whole",
                       n_components=3, max_pixels=500000, seed=42, file_list=None):
    """Compute PCA projection matrix on apple-region pixels.

    Steps:
        1. Load images/*.npy files (H, W, 9).
        2. Use whole/*.npy masks to select apple-region pixels (non-background).
        3. Collect 9-dim pixel vectors (random sample if too many).
        4. Fit PCA to get components (n_components, 9) and mean (9,).

    Args:
        data_dir: Root data directory containing image_dir and whole_dir.
        image_dir: Subdirectory name for spectral images.
        whole_dir: Subdirectory name for whole-apple masks.
        n_components: Number of PCA components to keep.
        max_pixels: Maximum number of pixels to sample for PCA fitting.
        seed: Random seed for reproducibility.
        file_list: Optional list of stems (without extension) to restrict fitting
            to — pass the TRAIN split here to avoid val/test leakage. None = all
            images in image_dir.

    Returns:
        components: (n_components, 9) ndarray — PCA projection matrix.
        mean: (9,) ndarray — per-channel mean for centering.
    """
    from sklearn.decomposition import PCA

    image_root = os.path.join(data_dir, image_dir)
    whole_root = os.path.join(data_dir, whole_dir)

    rng = np.random.RandomState(seed)
    all_pixels = []

    if file_list is not None:
        fnames = [s if s.endswith(".npy") else s + ".npy" for s in file_list]
    else:
        fnames = sorted(f for f in os.listdir(image_root) if f.endswith(".npy"))
    print(f"Computing PCA from {len(fnames)} images...")

    for fname in fnames:
        image = np.load(os.path.join(image_root, fname)).astype(np.float32)  # (H, W, 9)

        # Load whole-apple mask to filter background
        mask_path = os.path.join(whole_root, fname)
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(bool)  # (H, W)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            pixels = image[mask]  # (N, 9)
        else:
            # If no whole mask, use all pixels
            pixels = image.reshape(-1, 9)

        all_pixels.append(pixels)

    all_pixels = np.concatenate(all_pixels, axis=0)  # (total_N, 9)
    print(f"Total apple pixels: {len(all_pixels)}")

    # Random sample if too many pixels
    if len(all_pixels) > max_pixels:
        indices = rng.choice(len(all_pixels), max_pixels, replace=False)
        all_pixels = all_pixels[indices]
        print(f"Sampled {max_pixels} pixels for PCA fitting")

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(all_pixels)

    components = pca.components_.astype(np.float32)  # (n_components, 9)
    mean = pca.mean_.astype(np.float32)              # (9,)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return components, mean


def apply_pca(image_9ch, components, mean):
    """Apply precomputed PCA projection to a 9-channel image.

    Args:
        image_9ch: (9, H, W) float32 array.
        components: (n_components, 9) PCA projection matrix.
        mean: (9,) per-channel mean for centering.

    Returns:
        (n_components, H, W) float32 array.
    """
    C, H, W = image_9ch.shape
    centered = image_9ch - mean[:, None, None]  # (9, H, W)
    projected = components @ centered.reshape(C, -1)  # (n_components, H*W)
    return projected.reshape(components.shape[0], H, W)

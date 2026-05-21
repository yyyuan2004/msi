"""Physical interpretability of gate-selected bands.

Aggregates the bands chosen by the learnable gate across seeds/folds (from a
gate-batch manifest or explicit selections) and cross-references the selection
frequency against the spectral diagnostics of utils/spectral_analysis.py,
generalized to C bands:
    - defect-vs-sound separability (Cohen's d),
    - inter-band redundancy (mean |correlation| to other bands),
    - per-band linear predictability (regression R^2 from the other bands),
    - PCA loading contribution.

The question answered: does the gate *reproducibly* select bands that are
physically reasonable, i.e. highly discriminative (defect vs sound) and
non-redundant? A positive Spearman correlation between selection frequency and
separability, and a negative one with redundancy / regression-R^2, supports a
"yes".

Usage:
    python scripts/gate_band_interpretability.py \
        --manifest outputs/gate_batch/batch_manifest.json \
        --dataset b15_valid_span51_060-110 \
        --data_dir /root/autodl-fs/15/b15_valid_span51_060-110

    python scripts/gate_band_interpretability.py \
        --data_dir <ds> --selected 2,7,9   # analyze an explicit selection
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

try:
    from PIL import Image
except ImportError:
    Image = None


def parse_range(name):
    m = re.findall(r"(\d+)-(\d+)", name or "")
    return (int(m[-1][0]), int(m[-1][1])) if m else None


def band_labels(C, rng):
    if rng:
        return np.linspace(rng[0], rng[1], C).round().astype(int).tolist()
    return list(range(C))


def _load_mask(mask_root, stem):
    npy, png = os.path.join(mask_root, stem + ".npy"), os.path.join(mask_root, stem + ".png")
    if os.path.exists(npy):
        return np.load(npy).astype(np.int64)
    if os.path.exists(png) and Image is not None:
        return np.array(Image.open(png)).astype(np.int64)
    raise FileNotFoundError(f"Mask not found for {stem}")


def load_class_pixels(data_dir, image_dir, mask_dir, max_pixels=300000, seed=42):
    """Return (sound_pixels, defect_pixels, C) over all samples in a dataset."""
    img_root = os.path.join(data_dir, image_dir)
    mask_root = os.path.join(data_dir, mask_dir)
    whole_root = os.path.join(data_dir, "whole")
    stems = sorted(f[:-4] for f in os.listdir(img_root) if f.endswith(".npy"))
    sound, defect = [], []
    for stem in stems:
        img = np.load(os.path.join(img_root, stem + ".npy")).astype(np.float32)  # (H,W,C)
        mask = _load_mask(mask_root, stem)
        whole = os.path.join(whole_root, stem + ".npy")
        if os.path.isdir(whole_root) and os.path.exists(whole):
            apple = np.load(whole).astype(np.int64)
        else:
            apple = (img.mean(axis=2) > 0.05).astype(np.int64)
        h, d = (apple > 0) & (mask == 0), mask > 0
        if h.any():
            sound.append(img[h])
        if d.any():
            defect.append(img[d])
    C = int(np.load(os.path.join(img_root, stems[0] + ".npy")).shape[-1])
    sound = np.concatenate(sound, axis=0) if sound else np.empty((0, C))
    defect = np.concatenate(defect, axis=0) if defect else np.empty((0, C))
    rng = np.random.RandomState(seed)
    sub = lambda a: a[rng.choice(len(a), max_pixels, replace=False)] if len(a) > max_pixels else a
    return sub(sound), sub(defect), C


def spectral_diagnostics(sound, defect, seed=42):
    """Per-band separability / redundancy / predictability / PCA contribution."""
    C = sound.shape[1]
    mu_s, mu_d = sound.mean(0), defect.mean(0)
    var_s, var_d = sound.var(0), defect.var(0)
    n0, n1 = len(sound), len(defect)
    pooled = np.sqrt(((n0 - 1) * var_s + (n1 - 1) * var_d) / max(n0 + n1 - 2, 1))
    separability = np.abs(mu_d - mu_s) / (pooled + 1e-8)          # Cohen's d

    allpx = np.concatenate([sound, defect], axis=0)
    if len(allpx) > 100000:
        allpx = allpx[np.random.RandomState(seed).choice(len(allpx), 100000, replace=False)]
    corr = np.corrcoef(allpx.T)
    redundancy = (np.abs(corr).sum(1) - 1.0) / max(C - 1, 1)       # mean |corr| to others

    reg_r2 = np.zeros(C)
    for b in range(C):
        others = [j for j in range(C) if j != b]
        reg = LinearRegression().fit(allpx[:, others], allpx[:, b])
        reg_r2[b] = max(0.0, reg.score(allpx[:, others], allpx[:, b]))

    pca = PCA(n_components=min(C, 6)).fit(allpx)
    contrib = (np.abs(pca.components_) * pca.explained_variance_ratio_[:, None]).sum(0)

    return {
        "mean_sound": mu_s, "mean_defect": mu_d,
        "separability": separability, "redundancy": redundancy,
        "reg_r2": reg_r2, "pca_contrib": contrib, "corr": corr,
    }


def collect_frequency(manifest, dataset, C, ks=None):
    counts, n = np.zeros(C), 0
    for rec in manifest.get("runs", {}).values():
        if rec.get("dataset") != dataset or rec.get("mode") != "gate":
            continue
        if rec.get("status") != "done":
            continue
        if ks and rec.get("k") not in ks:
            continue
        for b in (rec.get("selected_local") or []):
            if 0 <= b < C:
                counts[b] += 1
        n += 1
    return counts / max(n, 1), n


def _spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def plot(freq, diag, labels, dataset, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure.")
        return
    C = len(freq)
    x = np.arange(C)
    fig, axes = plt.subplots(3, 1, figsize=(max(8, C * 0.5), 11), sharex=True)

    ax = axes[0]
    ax.plot(x, diag["mean_sound"], "b-o", ms=3, label="sound (mean)")
    ax.plot(x, diag["mean_defect"], "r-s", ms=3, label="defect (mean)")
    for b in np.where(freq > 0)[0]:
        ax.axvspan(b - 0.4, b + 0.4, color="orange", alpha=0.10 + 0.30 * freq[b])
    ax.set_ylabel("reflectance"); ax.legend(fontsize=8)
    ax.set_title(f"Gate band interpretability — {dataset}")

    ax = axes[1]
    ax.bar(x, freq, color="#ff7f0e", alpha=0.6, label="selection frequency")
    ax.set_ylabel("selection freq"); ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(x, diag["separability"], "k-^", ms=3, label="separability (Cohen's d)")
    ax2.set_ylabel("Cohen's d")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(x, diag["redundancy"], "g-o", ms=3, label="redundancy (mean |corr|)")
    ax.plot(x, diag["reg_r2"], "m-s", ms=3, label="predictability (reg R²)")
    for b in np.where(freq > 0)[0]:
        ax.axvspan(b - 0.4, b + 0.4, color="orange", alpha=0.10 + 0.30 * freq[b])
    ax.set_ylabel("redundancy / R²"); ax.set_xlabel("approx. HSI band index")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output}")


def main():
    parser = argparse.ArgumentParser(description="Physical interpretability of gate-selected bands")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None,
                        help="gate_batch batch_manifest.json (for selection frequency)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset name in the manifest (default: basename of data_dir)")
    parser.add_argument("--selected", type=str, default=None,
                        help="explicit local band indices (comma) instead of a manifest")
    parser.add_argument("--ks", type=str, default=None, help="restrict to these k values, e.g. 2,3")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--mask_dir", type=str, default="masks")
    parser.add_argument("--range", type=str, default=None, help="band range 'AAA-BBB' for labels")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset or os.path.basename(os.path.normpath(args.data_dir))
    sound, defect, C = load_class_pixels(args.data_dir, args.image_dir, args.mask_dir)
    print(f"{dataset}: C={C} bands | sound px={len(sound)} defect px={len(defect)}")
    diag = spectral_diagnostics(sound, defect)

    if args.selected:
        sel = [int(x) for x in args.selected.split(",")]
        freq = np.zeros(C)
        for b in sel:
            freq[b] = 1.0
        n_runs = 1
    elif args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        ks = [int(x) for x in args.ks.split(",")] if args.ks else None
        freq, n_runs = collect_frequency(manifest, dataset, C, ks)
    else:
        parser.error("Provide --manifest or --selected")
    print(f"Aggregated over {n_runs} gate run(s)")

    rng = parse_range(args.range or dataset)
    labels = band_labels(C, rng)

    # Cross-reference: does selection track physically reasonable bands?
    sp_sep = _spearman(freq, diag["separability"])
    sp_red = _spearman(freq, diag["redundancy"])
    sp_r2 = _spearman(freq, diag["reg_r2"])
    print("\nSpearman(selection frequency, ·):")
    print(f"  separability (Cohen's d) : {sp_sep:+.3f}  (expect > 0)")
    print(f"  redundancy (mean |corr|) : {sp_red:+.3f}  (expect < 0)")
    print(f"  predictability (reg R²)  : {sp_r2:+.3f}  (expect < 0)")
    order = np.argsort(freq)[::-1]
    top = [(labels[b], round(float(freq[b]), 2), round(float(diag["separability"][b]), 2))
           for b in order if freq[b] > 0]
    print(f"\nMost-selected bands (HSI idx, freq, Cohen's d): {top}")
    verdict = (sp_sep > 0.2 and sp_red < -0.1)
    print(f"\nVerdict: gate selections are {'consistent with' if verdict else 'NOT clearly aligned with'}"
          f" physically discriminative, non-redundant bands.")

    out = args.output or os.path.join("outputs", "interpretability", f"{dataset}_interpretability.png")
    plot(freq, diag, labels, dataset, out)

    summary = {
        "dataset": dataset, "C": C, "n_runs": int(n_runs),
        "selection_frequency": freq.tolist(), "band_labels": labels,
        "spearman_separability": sp_sep, "spearman_redundancy": sp_red,
        "spearman_reg_r2": sp_r2, "verdict_physically_aligned": bool(verdict),
    }
    sjson = os.path.join(os.path.dirname(out) or ".", f"{dataset}_interpretability.json")
    os.makedirs(os.path.dirname(sjson) or ".", exist_ok=True)
    with open(sjson, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {sjson}")


if __name__ == "__main__":
    main()

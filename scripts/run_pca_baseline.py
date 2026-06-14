"""One-command, leak-free PCA baseline: fit PCA on the train split, then train.

Old workflow (two manual, leakage-prone steps):
    1. python scripts/precompute_pca.py ...   # fits PCA on ALL 185 images
    2. python train_eval.py --config configs/pca_baseline.yaml

That leaks val/test spectra into the 9->3 projection. This driver does both in
one shot, leak-free:

    1. Resolve the SAME train/val/test split the model will use (get_data_splits,
       deterministic in --seed) and record which images are in train.
    2. Fit PCA on the apple-region pixels of the TRAIN images only -> pca_matrix.npz.
    3. Emit the PCA results package (7 CSVs, see utils/pca_package.py) + a zip.
    4. Launch train_eval.py with the same seed and a config pointing at the
       train-only npz, so the network and its evaluation use the leak-free
       projection. Because the split is seed-deterministic, the split used for
       PCA and for training are identical.

Usage:
    python scripts/run_pca_baseline.py --data_dir /root/autodl-tmp/datasets/185_9bands --seed 42
    python scripts/run_pca_baseline.py --data_dir ... --seed 42 --package_only   # no training
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _import_by_path(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PROJECT_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="Leak-free integrated PCA baseline")
    ap.add_argument("--config", default="configs/pca_baseline.yaml")
    ap.add_argument("--data_dir", default=None, help="Override cfg.data.data_dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default=None,
                    help="Default: outputs/<experiment_name>_seed<seed>")
    ap.add_argument("--apple_cap", type=int, default=500_000)
    ap.add_argument("--defect_cap", type=int, default=200_000)
    ap.add_argument("--healthy_cap", type=int, default=500_000)
    ap.add_argument("--wl_start", type=float, default=577.0,
                    help="Wavelength (nm) of band 0 (same 9 candidate bands as the gate finder)")
    ap.add_argument("--wl_end", type=float, default=725.0,
                    help="Wavelength (nm) of the last band")
    ap.add_argument("--package_only", action="store_true",
                    help="Fit PCA + build the package, but do NOT train")
    ap.add_argument("--skip_train", action="store_true",
                    help="Pass --skip_train to train_eval (eval/plots only)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    # config loader + split + package module, imported by path (no torch needed here).
    cfgmod = _import_by_path("_pca_cfg", "utils/config.py")
    splitmod = _import_by_path("_pca_split", "data/split.py")
    pkg = _import_by_path("_pca_pkg", "utils/pca_package.py")

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    cfg = cfgmod.load_config(cfg_path)
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir

    data_dir = cfg["data"]["data_dir"]
    image_dir = cfg["data"]["image_dir"]
    mask_dir = cfg["data"]["mask_dir"]
    whole_dir = "whole"
    band_indices = cfg["data"].get("band_indices")
    n_components = cfg["data"]["num_channels"]
    experiment_name = cfg.get("experiment_name", "pca_baseline")

    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "outputs", f"{experiment_name}_seed{args.seed}")
    out_dir = os.path.abspath(out_dir)
    pkg_dir = os.path.join(out_dir, "pca_package")
    npz_path = os.path.join(out_dir, "pca_matrix.npz")
    os.makedirs(pkg_dir, exist_ok=True)

    # 1. Resolve the same split the model will use.
    splits = splitmod.get_data_splits(data_dir=data_dir, image_dir=image_dir, seed=args.seed)
    train_stems = sorted(splits["train"])
    print(f"[split] seed={args.seed}  train={len(splits['train'])}  "
          f"val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"[split] PCA is fit on the {len(train_stems)} TRAIN images only "
          f"(see pca_package/dataset_per_image_audit.csv for the full list).")

    # detect raw band count (respecting any pre-PCA band selection)
    sample_img = pkg.load_image(data_dir, image_dir, train_stems[0])
    raw_bands = sample_img.shape[-1] if band_indices is None else len(band_indices)
    wl = pkg.band_wavelengths(raw_bands, args.wl_start, args.wl_end)
    if not (1 <= n_components <= raw_bands):
        sys.exit(f"ERROR: num_channels (n_components={n_components}) must be in "
                 f"[1, {raw_bands}] (raw bands).")

    # 2. Collect TRAIN pixels and audit ALL images.
    apple_px, healthy_px, defect_px = pkg.collect_region_pixels(
        data_dir, image_dir, mask_dir, whole_dir, train_stems,
        apple_cap=args.apple_cap, defect_cap=args.defect_cap,
        healthy_cap=args.healthy_cap, seed=args.seed, band_indices=band_indices)
    print(f"[pixels] apple={len(apple_px)}  healthy={len(healthy_px)}  defect={len(defect_px)}")
    audit_rows = pkg.audit_images(data_dir, image_dir, mask_dir, whole_dir, splits)

    # 3. Fit PCA + write package + the deployment npz.
    components, mean = pkg.write_package(
        pkg_dir, n_components, apple_px, healthy_px, defect_px, audit_rows, wl, seed=args.seed)
    np.savez(npz_path, components=components, mean=mean)
    print(f"[pca] components {components.shape}, mean {mean.shape} -> {npz_path}")
    print(f"[pca] package (7 CSVs) -> {pkg_dir}")

    zip_base = os.path.join(out_dir, "pca_package")
    shutil.make_archive(zip_base, "zip", pkg_dir)
    print(f"[pca] package zip -> {zip_base}.zip")

    # 4. Train via train_eval.py with the train-only npz (same seed = same split).
    run_cfg = {**cfg}
    run_cfg["data"] = {**cfg["data"], "use_pca": True, "pca_matrix_path": npz_path}
    run_cfg_path = os.path.join(out_dir, "pca_run_config.yaml")
    with open(run_cfg_path, "w") as f:
        yaml.safe_dump(run_cfg, f, default_flow_style=False, sort_keys=False)

    if args.package_only:
        print("\n[done] --package_only: PCA matrix + package written; training skipped.")
        print(f"       To train later:\n       {args.python} train_eval.py "
              f"--config {run_cfg_path} --seed {args.seed} --output_dir {out_dir}")
        return

    cmd = [args.python, os.path.join(PROJECT_ROOT, "train_eval.py"),
           "--config", run_cfg_path, "--seed", str(args.seed), "--output_dir", out_dir]
    if args.skip_train:
        cmd.append("--skip_train")
    print(f"\n[train] launching: {' '.join(cmd)}\n")
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    if ret != 0:
        sys.exit(f"train_eval.py exited with code {ret}")
    print(f"\n[done] PCA baseline complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

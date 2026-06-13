"""Run ONLY the band gate, many times, to read off the finally-selected bands.

This is a stripped-down driver for one question: *which physical bands does the
learnable gate keep?* It does not evaluate segmentation quality — no val/test
metrics are computed — it just trains the gate end-to-end under the usual
segmentation loss and records the top-k bands it converges to.

For every ``k`` in ``--ks`` (default 1,2,3) it trains ``--runs`` times (default
30) with different seeds (different gate init + data split), and for each run
writes the retained band indices. Afterwards it aggregates, across runs, how
often each band was kept (the Table-5 "selection frequency" quantity) and draws
a per-band frequency-distribution figure.

Early stopping (gate-selection-stability, *not* IoU):
    The temperature tau is cosine-annealed from tau_start to tau_end over the
    first ``--anneal_epochs`` epochs; after that the gate is in its committed
    (near-hard) regime. A run stops once the top-k selection has been identical
    for ``--stable_patience`` consecutive committed epochs, or at ``--max_epochs``.
    This targets exactly what we want — the band set has stopped moving — instead
    of a metric the user does not need.

Outputs (under ``--out_root``, default outputs/gate_band_finder/):
    k{K}/run{I:02d}_seed{S}/selected_bands.json   # one per run (the result file)
    k{K}/run{I:02d}_seed{S}/gate_history.json      # per-epoch selection trace
    all_runs.json                                  # flat list of every run
    band_selection_frequency.json / .csv           # aggregated frequencies
    band_frequency.png                             # the requested figure
    band_frequency_by_k.png                        # per-k panels (supplementary)
    gate_band_finder.log

Resumable: a run whose selected_bands.json exists with "final": true is skipped,
and any single-run exception is logged and the batch continues.

Examples:
    # Full job on the default dataset (configs/gate.yaml data path)
    python scripts/gate_band_finder.py --ks 1,2,3 --runs 30

    # Point at a dataset explicitly, smaller budget
    python scripts/gate_band_finder.py --data_dir /root/autodl-tmp/datasets/185_9bands \
        --ks 1,2,3 --runs 30 --max_epochs 150 --anneal_epochs 80 --stable_patience 20

    # Only (re)build the aggregation + figure from existing run files (no torch/GPU)
    python scripts/gate_band_finder.py --plots_only
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys
import time
import traceback

# Make the repo root importable when run as `python scripts/gate_band_finder.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------------------------------- #
# Config loading (import utils/config.py by path so --plots_only needs no torch)
# --------------------------------------------------------------------------- #
def load_config(path):
    spec = importlib.util.spec_from_file_location(
        "_gbf_cfgloader", os.path.join(PROJECT_ROOT, "utils", "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_config(path)


def band_wavelengths(num_bands, wl_start, wl_end):
    """Linear local-band-index -> wavelength (nm) map (matches the paper's Table 5)."""
    if num_bands <= 1:
        return [float(wl_start)]
    step = (wl_end - wl_start) / (num_bands - 1)
    return [round(wl_start + step * i, 1) for i in range(num_bands)]


def run_dir_for(out_root, k, run_idx, seed):
    return os.path.join(out_root, f"k{k}", f"run{run_idx:02d}_seed{seed}")


def log(log_path, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Single gate run (imports torch lazily so the plotting path stays torch-free)
# --------------------------------------------------------------------------- #
def train_gate_once(cfg, k, seed, out_dir, args, device, log_path):
    """Train the gate once and return the sorted list of retained local band indices."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from data.dataset import MSIDataset, get_dataset_kwargs
    from data.augment import get_train_transforms
    from data.split import get_data_splits
    from model.model import build_model
    from model.loss import SegmentationLoss

    # ---- reproducibility (same convention as train.py) ----
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    data_dir = cfg["data"]["data_dir"]
    splits = get_data_splits(
        data_dir=data_dir, image_dir=cfg["data"]["image_dir"], seed=seed)
    # The gate only needs data + masks to learn band scores; no eval is done.
    stems = list(splits["train"])
    if args.use_all_data:
        stems = stems + list(splits["val"]) + list(splits["test"])

    ds_kwargs = get_dataset_kwargs(cfg)
    dataset = MSIDataset(
        stems, data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"], mask_dir=cfg["data"]["mask_dir"],
        transform=get_train_transforms(cfg),
        num_classes=cfg["data"]["num_classes"], **ds_kwargs)

    num_workers = (args.num_workers if args.num_workers is not None
                   else cfg["train"].get("num_workers", 4))
    loader = DataLoader(
        dataset, batch_size=args.batch_size or cfg["train"]["batch_size"],
        shuffle=True, num_workers=num_workers,
        pin_memory=cfg["train"].get("pin_memory", True),
        drop_last=len(stems) >= (args.batch_size or cfg["train"]["batch_size"]),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None)

    model = build_model(cfg).to(device)
    if model.get_selected_bands() is None:
        raise RuntimeError("No band gate in the built model; check use_band_gate in the config.")

    criterion = SegmentationLoss(
        loss_type=cfg["train"].get("loss", "ce_dice"),
        ce_weight=cfg["train"].get("ce_weight", 0.5),
        dice_weight=cfg["train"].get("dice_weight", 0.5))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr or cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4))

    anneal = max(1, args.anneal_epochs)
    history = []          # [{epoch, selection, train_loss, tau}]
    committed_prev = None
    stable_count = 0
    stopped_epoch = args.max_epochs

    for epoch in range(1, args.max_epochs + 1):
        frac = min((epoch - 1) / max(1, anneal - 1), 1.0)
        model.set_gate_progress(frac)

        model.train()
        running = 0.0
        nb = 0
        for images, masks, _raw, _apple, _stems in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            nb += 1
        train_loss = running / max(nb, 1)

        sel = tuple(model.get_selected_bands())
        tau = float(model.band_gate._tau.item())
        history.append({"epoch": epoch, "selection": list(sel),
                        "train_loss": round(train_loss, 6), "tau": round(tau, 5)})

        # Selection-stability early stopping, only once tau has fully annealed.
        committed = epoch >= anneal
        if committed:
            if sel == committed_prev:
                stable_count += 1
            else:
                stable_count = 1
                committed_prev = sel
            if stable_count >= args.stable_patience:
                stopped_epoch = epoch
                break

        if epoch % 10 == 0 or epoch == 1:
            log(log_path, f"    k={k} seed={seed} ep{epoch:03d} "
                          f"loss={train_loss:.4f} tau={tau:.3f} sel={list(sel)} "
                          f"stable={stable_count}")

    final_sel = list(model.get_selected_bands())
    candidate = cfg["data"].get("band_indices")
    global_sel = [candidate[i] for i in final_sel] if candidate else list(final_sel)
    wl = band_wavelengths(cfg["data"]["num_channels"], args.wl_start, args.wl_end)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gate_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    result = {
        "k": k, "seed": seed, "num_bands": cfg["data"]["num_channels"],
        "selected_bands_local": final_sel,
        "selected_bands_global": global_sel,
        "selected_wavelengths_nm": [wl[i] for i in final_sel],
        "candidate_band_indices": candidate,
        "stopped_epoch": stopped_epoch,
        "max_epochs": args.max_epochs,
        "anneal_epochs": anneal,
        "stable_patience": args.stable_patience,
        "stable_for_epochs": stable_count,
        "final_tau": round(float(model.band_gate._tau.item()), 5),
        "final": True,
    }
    with open(os.path.join(out_dir, "selected_bands.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def is_done(out_dir):
    p = os.path.join(out_dir, "selected_bands.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            return bool(json.load(f).get("final"))
    except (ValueError, OSError):
        return False


# --------------------------------------------------------------------------- #
# Aggregation + plotting (torch-free; works under --plots_only)
# --------------------------------------------------------------------------- #
def collect_runs(out_root):
    """Read every k*/run*/selected_bands.json under out_root."""
    runs = []
    for p in sorted(glob.glob(os.path.join(out_root, "k*", "run*", "selected_bands.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
            if d.get("final") and d.get("selected_bands_local") is not None:
                runs.append(d)
        except (ValueError, OSError):
            continue
    return runs


def aggregate(runs, num_bands, wl_start, wl_end):
    wl = band_wavelengths(num_bands, wl_start, wl_end)
    ks = sorted({r["k"] for r in runs})
    counts = {k: [0] * num_bands for k in ks}
    n_runs = {k: 0 for k in ks}
    for r in runs:
        k = r["k"]
        n_runs[k] += 1
        for b in r["selected_bands_local"]:
            if 0 <= b < num_bands:
                counts[k][b] += 1
    freq = {k: [counts[k][b] / n_runs[k] if n_runs[k] else 0.0 for b in range(num_bands)]
            for k in ks}
    return {
        "ks": ks,
        "num_bands": num_bands,
        "runs_per_k": n_runs,
        "wavelengths_nm": wl,
        "counts": counts,
        "frequency": freq,
    }


def write_tables(agg, out_root):
    with open(os.path.join(out_root, "band_selection_frequency.json"), "w") as f:
        json.dump(agg, f, indent=2)
    csv_path = os.path.join(out_root, "band_selection_frequency.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "band_index", "wavelength_nm", "count", "n_runs", "frequency"])
        for k in agg["ks"]:
            for b in range(agg["num_bands"]):
                w.writerow([k, b, agg["wavelengths_nm"][b], agg["counts"][k][b],
                            agg["runs_per_k"][k], f"{agg['frequency'][k][b]:.4f}"])
    return csv_path


def _get_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("WARNING: matplotlib unavailable, skipping figures.")
        return None


def plot_frequency(agg, out_root):
    """One grouped-bar figure: per-band selection frequency, grouped by k."""
    plt = _get_mpl()
    if plt is None or not agg["ks"]:
        return None
    import numpy as np

    ks = agg["ks"]
    C = agg["num_bands"]
    wl = agg["wavelengths_nm"]
    x = np.arange(C)
    width = 0.8 / len(ks)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(max(7, C * 0.9), 5))
    for i, k in enumerate(ks):
        offset = (i - (len(ks) - 1) / 2) * width
        bars = ax.bar(x + offset, agg["frequency"][k], width,
                      label=f"k={k} (n={agg['runs_per_k'][k]})",
                      color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)
        for bar, fv in zip(bars, agg["frequency"][k]):
            if fv > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, fv + 0.012, f"{fv:.2f}",
                        ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"b{b}\n{wl[b]:.0f} nm" for b in range(C)], fontsize=8)
    ax.set_xlabel("Candidate band (local index / approx. wavelength)")
    ax.set_ylabel("Selection frequency across runs")
    ax.set_ylim(0, 1.08)
    ax.set_title("Gate band-selection frequency (gate-only runs)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_root, "band_frequency.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def plot_frequency_by_k(agg, out_root):
    """Supplementary: one bar panel per k."""
    plt = _get_mpl()
    if plt is None or not agg["ks"]:
        return None
    import numpy as np

    ks = agg["ks"]
    C = agg["num_bands"]
    wl = agg["wavelengths_nm"]
    x = np.arange(C)
    fig, axes = plt.subplots(len(ks), 1, figsize=(max(7, C * 0.8), 2.4 * len(ks)),
                             sharex=True)
    if len(ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        bars = ax.bar(x, agg["frequency"][k], color="#1f77b4",
                      edgecolor="white", linewidth=0.5)
        for bar, fv in zip(bars, agg["frequency"][k]):
            if fv > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, fv + 0.01, f"{fv:.2f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(f"k={k}\n(n={agg['runs_per_k'][k]})")
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"b{b}\n{wl[b]:.0f} nm" for b in range(C)], fontsize=8)
    axes[-1].set_xlabel("Candidate band (local index / approx. wavelength)")
    axes[0].set_title("Gate band-selection frequency by k", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_root, "band_frequency_by_k.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def build_outputs(out_root, num_bands, wl_start, wl_end):
    runs = collect_runs(out_root)
    if not runs:
        print(f"No completed runs found under {out_root}.")
        return
    with open(os.path.join(out_root, "all_runs.json"), "w") as f:
        json.dump(runs, f, indent=2)
    agg = aggregate(runs, num_bands, wl_start, wl_end)
    write_tables(agg, out_root)
    plot_frequency(agg, out_root)
    plot_frequency_by_k(agg, out_root)
    # Console summary.
    print("\n=== Gate band-selection frequency ===")
    for k in agg["ks"]:
        order = sorted(range(num_bands), key=lambda b: -agg["frequency"][k][b])
        top = ", ".join(f"b{b}({agg['wavelengths_nm'][b]:.0f}nm)={agg['frequency'][k][b]:.2f}"
                        for b in order if agg["frequency"][k][b] > 0)
        print(f"  k={k} [{agg['runs_per_k'][k]} runs]: {top}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Run only the band gate to find the finally-selected bands.")
    ap.add_argument("--config", default="configs/gate.yaml")
    ap.add_argument("--data_dir", default=None, help="Override cfg.data.data_dir")
    ap.add_argument("--out_root", default="outputs/gate_band_finder")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--runs", type=int, default=30, help="Repeats per k")
    ap.add_argument("--seed_base", type=int, default=0,
                    help="Seeds are seed_base + run_index (shared across k)")
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--anneal_epochs", type=int, default=80,
                    help="tau cosine-annealed to tau_end over these epochs")
    ap.add_argument("--stable_patience", type=int, default=20,
                    help="Stop after the selection is unchanged for this many committed epochs")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None,
                    help="DataLoader workers (override cfg). Use 0 on WSL if workers crash.")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wl_start", type=float, default=577.0,
                    help="Wavelength (nm) of local band 0 (paper Table 5: 577)")
    ap.add_argument("--wl_end", type=float, default=725.0,
                    help="Wavelength (nm) of the last local band (paper Table 5: 725)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--use_all_data", action="store_true",
                    help="Train the gate on train+val+test (more data; no eval is done anyway)")
    ap.add_argument("--plots_only", action="store_true",
                    help="Skip training; only aggregate existing runs + redraw the figure")
    args = ap.parse_args()

    cfg = load_config(os.path.join(PROJECT_ROOT, args.config)
                      if not os.path.isabs(args.config) else args.config)
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    # Gate-only: never the full/random arms.
    cfg["model"]["use_band_gate"] = True
    cfg["model"]["band_gate_random_select"] = False

    num_bands = cfg["data"]["num_channels"]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    bad = [k for k in ks if not (1 <= k <= num_bands)]
    if bad:
        print(f"ERROR: k must be in [1, {num_bands}] (num_channels); offending: {bad}")
        sys.exit(1)

    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(args.out_root, "gate_band_finder.log")

    if args.plots_only:
        build_outputs(args.out_root, num_bands, args.wl_start, args.wl_end)
        return

    seeds = [args.seed_base + i for i in range(args.runs)]
    total = len(ks) * len(seeds)
    log(log_path, f"=== gate band finder: ks={ks}, runs={args.runs}, "
                  f"total={total}, num_bands={num_bands}, "
                  f"max_epochs={args.max_epochs}, anneal={args.anneal_epochs}, "
                  f"stable_patience={args.stable_patience} ===")

    # Resolve device once (lazy torch import so --plots_only needs no torch).
    import torch
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    log(log_path, f"device={device}")

    done = 0
    for k in ks:
        run_cfg = json.loads(json.dumps(cfg))  # deep copy
        run_cfg["model"]["band_gate_k"] = k
        run_cfg["experiment_name"] = f"gate_finder_k{k}"
        for idx, seed in enumerate(seeds):
            out_dir = run_dir_for(args.out_root, k, idx, seed)
            done += 1
            if is_done(out_dir):
                log(log_path, f"[skip {done}/{total}] k={k} run{idx:02d} seed={seed} (already done)")
                continue
            log(log_path, f"[run  {done}/{total}] k={k} run{idx:02d} seed={seed}")
            t0 = time.time()
            try:
                res = train_gate_once(run_cfg, k, seed, out_dir, args, device, log_path)
                log(log_path, f"[done {done}/{total}] k={k} seed={seed} "
                              f"-> bands {res['selected_bands_local']} "
                              f"({res['selected_wavelengths_nm']} nm) "
                              f"stop@{res['stopped_epoch']} ({time.time() - t0:.0f}s)")
            except Exception as e:  # noqa: BLE001 - keep the batch alive
                tb = traceback.format_exc()
                log(log_path, f"[FAIL {done}/{total}] k={k} seed={seed}: {e!r}")
                log(log_path, tb)
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    with open(os.path.join(out_dir, "error.txt"), "w") as ef:
                        ef.write(tb)
                except OSError:
                    pass
            # Refresh aggregation + figure after every run so partial progress is usable.
            build_outputs(args.out_root, num_bands, args.wl_start, args.wl_end)

    log(log_path, "=== finished ===")


if __name__ == "__main__":
    main()

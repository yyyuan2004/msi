"""Export a deployable k-channel model from a trained gated (B-channel) model.

A gated model M_k = F ∘ G_k is trained on the full B-band candidate input; at
eval the gate applies a hard k-hot mask, so the backbone effectively consumes
only the k selected bands. This script materializes that into a *strict
k-channel* SegmentationModel (no gate) by slicing the first-conv weights for the
selected bands and copying all other weights verbatim. The exported model is
mathematically equivalent to the gated model at inference (verified numerically),
but takes a k-band input directly — i.e. a real reduced-band MSI sensor.

All deployment/inference reporting should use the exported model, e.g.:
    python eval.py --checkpoint <out>/deploy_kch.pth --split test --seed <seed>

Usage:
    python scripts/export_deploy_model.py --run_dir outputs/<exp>_seed42
    python scripts/export_deploy_model.py \
        --checkpoint outputs/<exp>_seed42/checkpoints/best_model.pth \
        --selected_bands outputs/<exp>_seed42/selected_bands.json
"""

import argparse
import copy
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import build_model


def _first_encoder_conv(model):
    """Return (name, module) of the first Conv2d in the encoder (the input conv)."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name.startswith("encoder"):
            return name, m
    raise RuntimeError("No encoder Conv2d found; export supports the default SegmentationModel.")


def export_deploy_model(checkpoint_path, selected_bands_path=None, output_path=None,
                        verify_size=64):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise ValueError("Checkpoint has no embedded config; cannot rebuild the model.")
    cfg9 = ckpt["config"]

    if cfg9["model"].get("architecture", "default") != "default":
        raise ValueError("Deploy export only supports the default SegmentationModel.")
    if not cfg9["model"].get("use_band_gate", False):
        raise ValueError("This checkpoint has no band gate; nothing to export.")

    # Resolve the selected bands.
    selected_local, selected_global, candidate = None, None, cfg9["data"].get("band_indices")
    if selected_bands_path and os.path.exists(selected_bands_path):
        with open(selected_bands_path) as f:
            sb = json.load(f)
        selected_local = sb.get("selected_bands_local")
        selected_global = sb.get("selected_bands_global")

    # Build the gated (B-channel) model and load weights.
    model9 = build_model(cfg9)
    model9.load_state_dict(ckpt["model_state_dict"])
    model9.eval()

    if selected_local is None:
        selected_local = model9.get_selected_bands()
    if selected_global is None:
        selected_global = ([candidate[i] for i in selected_local]
                           if candidate is not None else list(selected_local))

    # Cross-check: the gate's deterministic selection must match the json.
    gate_sel = model9.get_selected_bands()
    if sorted(gate_sel) != sorted(selected_local):
        print(f"WARNING: gate selects {gate_sel} but selected_bands.json says "
              f"{selected_local}; using the gate's selection.")
        selected_local = gate_sel
        selected_global = ([candidate[i] for i in selected_local]
                           if candidate is not None else list(selected_local))
    selected_local = sorted(selected_local)
    k = len(selected_local)
    B = cfg9["data"]["num_channels"]
    print(f"Exporting {B}-ch gated model -> {k}-ch deploy model")
    print(f"  selected (local idx into the {B}-band input): {selected_local}")
    print(f"  selected (global band idx for the dataset):   {selected_global}")

    # Build the strict k-channel deploy config + model (no gate).
    cfg3 = copy.deepcopy(cfg9)
    cfg3["data"]["num_channels"] = k
    cfg3["data"]["band_indices"] = selected_global  # dataset reads exactly these bands
    cfg3["model"]["use_band_gate"] = False
    cfg3["experiment_name"] = cfg9.get("experiment_name", "model") + f"_deploy{k}ch"
    model3 = build_model(cfg3)
    model3.eval()

    # Transfer weights: everything except the first conv (shape differs) and the gate.
    fname9, fconv9 = _first_encoder_conv(model9)
    fname3, fconv3 = _first_encoder_conv(model3)
    fkey9, fkey3 = fname9 + ".weight", fname3 + ".weight"
    sd9 = model9.state_dict()
    transfer = {kk: vv for kk, vv in sd9.items()
                if kk != fkey9 and not kk.startswith("band_gate")}
    missing, unexpected = model3.load_state_dict(transfer, strict=False)
    # The only missing key should be the first conv we set manually below.
    leftover = [m for m in missing if m != fkey3]
    if leftover or unexpected:
        raise RuntimeError(f"Unexpected state_dict mismatch: missing={leftover}, "
                           f"unexpected={list(unexpected)}")
    with torch.no_grad():
        fconv3.weight.copy_(fconv9.weight[:, selected_local, :, :])

    # Numerical equivalence check: gated(full B-band) == deploy(k-band slice).
    with torch.no_grad():
        x = torch.randn(2, B, verify_size, verify_size)
        y9 = model9(x)
        y3 = model3(x[:, selected_local, :, :])
        max_abs_err = float((y9 - y3).abs().max())
    print(f"  equivalence check: max|gated - deploy| = {max_abs_err:.3e} "
          f"({'OK' if max_abs_err < 1e-4 else 'FAIL'})")
    if max_abs_err >= 1e-4:
        raise RuntimeError("Deploy model is not equivalent to the gated model; aborting.")

    if output_path is None:
        base = os.path.dirname(os.path.dirname(checkpoint_path))  # run_dir
        output_path = os.path.join(base, "deploy", f"deploy_{k}ch.pth")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "model_state_dict": model3.state_dict(),
        "config": cfg3,
        "selected_bands_local": selected_local,
        "selected_bands_global": selected_global,
        "num_channels": k,
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "equivalence_max_abs_err": max_abs_err,
    }, output_path)
    print(f"Saved deploy model to {output_path}")
    print(f"Evaluate on the test split with:\n"
          f"  python eval.py --checkpoint {output_path} --split test --seed <seed>")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export a deployable k-channel model")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run dir with checkpoints/best_model.pth and selected_bands.json")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the gated best_model.pth (overrides --run_dir)")
    parser.add_argument("--selected_bands", type=str, default=None,
                        help="Path to selected_bands.json (default: <run_dir>/selected_bands.json)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    selected = args.selected_bands
    if checkpoint is None:
        if args.run_dir is None:
            parser.error("Provide --run_dir or --checkpoint")
        checkpoint = os.path.join(args.run_dir, "checkpoints", "best_model.pth")
        if selected is None:
            selected = os.path.join(args.run_dir, "selected_bands.json")

    export_deploy_model(checkpoint, selected, args.output)


if __name__ == "__main__":
    main()

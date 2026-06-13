#!/bin/bash
# ==============================================================================
# Gate-only band finder.
#
# Trains ONLY the learnable band gate (no metrics, no full/random baselines),
# repeated RUNS times for each k in KS, and reports which physical bands the
# gate keeps. Early stopping is on gate-selection stability: tau is annealed,
# then a run stops once the chosen bands stop changing (see scripts/gate_band_finder.py).
#
# Results go to OUT_ROOT (gitignored): per-run selected_bands.json, an aggregated
# band_selection_frequency.{json,csv}, and the per-band frequency figure
# band_frequency.png (+ band_frequency_by_k.png).
#
# Usage:
#   bash run_gate_band_finder.sh
#   bash run_gate_band_finder.sh --data_dir /root/autodl-tmp/datasets/185_9bands
#   bash run_gate_band_finder.sh --plots_only          # rebuild figure from existing runs
# ==============================================================================

set -e

# ===== user config =====
DATA_DIR="/root/autodl-tmp/datasets/185_9bands"   # dataset root with images/ + masks/
OUT_ROOT="outputs/gate_band_finder"
KS="1,2,3"
RUNS=30
MAX_EPOCHS=150
ANNEAL_EPOCHS=80
STABLE_PATIENCE=20
# =======================

python scripts/gate_band_finder.py \
    --config configs/gate.yaml \
    --data_dir "$DATA_DIR" \
    --out_root "$OUT_ROOT" \
    --ks "$KS" \
    --runs "$RUNS" \
    --max_epochs "$MAX_EPOCHS" \
    --anneal_epochs "$ANNEAL_EPOCHS" \
    --stable_patience "$STABLE_PATIENCE" \
    "$@"

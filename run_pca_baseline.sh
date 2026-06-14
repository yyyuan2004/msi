#!/bin/bash
# ==============================================================================
# Leak-free, integrated PCA baseline (replaces the old two-step workflow).
#
# One command: fit PCA on the TRAIN split only -> write pca_matrix.npz + a 7-CSV
# results package (zipped) -> train + eval via train_eval.py with the same seed
# (identical split, no val/test leakage into the 9->3 projection).
#
# Usage:
#   bash run_pca_baseline.sh
#   bash run_pca_baseline.sh --data_dir /root/autodl-tmp/datasets/185_9bands --seed 42
#   bash run_pca_baseline.sh --package_only       # PCA matrix + package, no training
# ==============================================================================

set -e

# ===== user config =====
DATA_DIR="/root/autodl-tmp/datasets/185_9bands"
SEED=42
# =======================

python scripts/run_pca_baseline.py \
    --config configs/pca_baseline.yaml \
    --data_dir "$DATA_DIR" \
    --seed "$SEED" \
    "$@"

#!/bin/bash
# ==============================================================================
# Ablation Study Runner
#
# Runs all config x seed experiments sequentially:
#   Each (config, seed) does: train -> eval -> metric curves -> error analysis
#
# Usage:
#   bash run_ablation.sh                          # 7:3 split (default)
#   bash run_ablation.sh --kfold 5                # 5-fold cross-validation
#   bash run_ablation.sh --vis_augment            # 7:3 + augmentation viz
#   bash run_ablation.sh --kfold 5 --vis_augment  # 5-fold + augment viz
# ==============================================================================

set -e

# 17 configs: 2 custom + 7 SMP + 3 TopFormer + 3 SeaFormer + 2 PIDNet
CONFIGS=(
    "baseline"
    "gate_k3"
    "gate_k4"
    "gate_k5"
    # "spconv_se"
    "smp_unet_resnet18"
    "smp_unet_resnet34"
    # "smp_unetplusplus_resnet34"
    # "smp_linknet_resnet34"
    # "smp_manet_resnet34"
    "smp_deeplabv3plus_mobilenetv2"
    "smp_fpn_efficientnetb0"
    # "topformer_t"
    # "topformer_s"
    # "topformer_b"
    # "seaformer_t"
    # "seaformer_s"
    # "seaformer_b"
    # "pidnet_s"
    # "pidnet_m"
)

SEEDS=(42 456 91 1026 1 23 45 67 8889 1010)

# Parse CLI arguments
KFOLD=0
VIS_AUGMENT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kfold)
            KFOLD="$2"
            shift 2
            ;;
        --vis_augment)
            VIS_AUGMENT="--vis_augment"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

KFOLD_FLAG=""
MODE_DESC="default 7:3 split"
if [ "$KFOLD" -gt 0 ]; then
    KFOLD_FLAG="--kfold ${KFOLD}"
    MODE_DESC="${KFOLD}-fold cross-validation"
fi

echo "=============================================="
echo " MSI Bruise Ablation Study (Automated)"
echo " Configs: ${#CONFIGS[@]}"
echo " Seeds:   ${SEEDS[*]}"
echo " Mode:    ${MODE_DESC}"
echo " Strategy: all configs per seed, then next seed"
echo "=============================================="

# Step 1: Spectral pre-analysis (one-time)
DATA_DIR=$(python -c "
from utils.config import load_config
cfg = load_config('configs/baseline.yaml')
print(cfg['data']['data_dir'])
" 2>/dev/null || echo "")

if [ -n "$DATA_DIR" ] && [ -d "${DATA_DIR}/images" ]; then
    echo ""
    echo "[Pre-analysis] Running spectral analysis..."
    python utils/spectral_analysis.py \
        --data_dir "${DATA_DIR}" \
        --output_dir outputs/spectral_analysis || true
    echo "[Pre-analysis] Done."
fi

# Step 2: Train & eval for each (seed, config)
FIRST_RUN=true
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# SEED = ${seed}"
    echo "############################################################"

    for config in "${CONFIGS[@]}"; do
        echo ""
        echo "=============================================="
        echo " Train-Eval: ${config} | Seed: ${seed} | Mode: ${MODE_DESC}"
        echo "=============================================="

        EXTRA_FLAGS=""
        if [ "$FIRST_RUN" = true ] && [ -n "$VIS_AUGMENT" ]; then
            EXTRA_FLAGS="--vis_augment"
            FIRST_RUN=false
        fi

        if [ "$KFOLD" -gt 0 ]; then
            OUTPUT_DIR="outputs/${config}_seed${seed}_kfold${KFOLD}"
        else
            OUTPUT_DIR="outputs/${config}_seed${seed}"
        fi

        # Skip if already completed
        if [ "$KFOLD" -eq 0 ] && [ -f "${OUTPUT_DIR}/done.flag" ]; then
            echo "[skip] ${OUTPUT_DIR}/done.flag exists, run already completed."
            continue
        fi
        if [ "$KFOLD" -gt 0 ] && [ -f "${OUTPUT_DIR}/kfold_summary.json" ]; then
            echo "[skip] ${OUTPUT_DIR}/kfold_summary.json exists, k-fold already aggregated."
            continue
        fi

        python -u train_eval.py \
            --config "configs/${config}.yaml" \
            --seed "${seed}" \
            --output_dir "${OUTPUT_DIR}" \
            ${KFOLD_FLAG} \
            ${EXTRA_FLAGS}
    done
done

# Step 3: Aggregate results + generate comparison plots
if [ "$KFOLD" -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo " Aggregating results..."
    echo "=============================================="
    python aggregate_results.py || echo "(aggregate_results.py skipped/failed)"

    echo ""
    echo "=============================================="
    echo " Generating comparison plots..."
    echo "=============================================="
    python scripts/plot_ablation.py || echo "(plot_ablation.py skipped/failed)"
fi

echo ""
echo "=============================================="
echo " All ablation experiments completed!"
echo "=============================================="

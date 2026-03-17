#!/usr/bin/env bash
# visualize.sh – launch the online SIBR visualizer for a trained 3DGEER model.
#
# Usage:
#   bash visualize.sh
#
# Edit the variables below to match your scene and output paths.

set -e

# ------------------------------------------------------------------
# User-configurable settings
# ------------------------------------------------------------------

SCENE_IDS="1d003b07bd/dslr 1f7cbbdde1/dslr"
DATA_ROOT="data/scnt/datasets/"

# Must match the values used during training
STEP=0.002
FOVMOD_TRAIN=1.3

# Iteration to load (-1 = latest checkpoint)
ITER=-1

# Network GUI settings (must match what the SIBR viewer connects to)
IP="127.0.0.1"
PORT=6009

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

for SCENE_ID in $SCENE_IDS; do
    echo "Visualizing scene: $SCENE_ID"

    DATASET_DIR="${DATA_ROOT}${SCENE_ID}/"
    OUTPUT_DIR="./output/scnt/${SCENE_ID}"

    BEAP_DIR_TRAIN="beap_fov_${FOVMOD_TRAIN}_step_${STEP}/"
    TRAIN_MASK_FN="fov_${FOVMOD_TRAIN}_step_${STEP}_mask.png"

    python visualizer.py \
        -s "$DATASET_DIR" \
        -m "$OUTPUT_DIR" \
        --iteration "$ITER" \
        --ip "$IP" \
        --port "$PORT" \
        --sample_step "$STEP" \
        --fov_mod "$FOVMOD_TRAIN" \
        --mask_path "${DATASET_DIR}${BEAP_DIR_TRAIN}${TRAIN_MASK_FN}" \
        --sibr_mask_refcam "${DATASET_DIR}colmap/cameras_fish.txt"
done

#!/usr/bin/env bash
# visualize.sh – launch the online SIBR visualizer for a trained 3DGEER model.
#
# Usage:
#   bash visualize.sh SCENE_ID DATA_ROOT CKPT_DIR MODE [NEAR_THRESHOLD]
#
#   SCENE_ID   Scene identifier, e.g. 1d003b07bd/dslr
#   DATA_ROOT  Root directory of the dataset, e.g. data/scnt/datasets
#   CKPT_DIR   Root directory of saved checkpoints, e.g. ckpt/scnt
#   MODE       Render mode: BEAP (default), KB, or PH
#   NEAR_THRESHOLD  Near-plane culling threshold passed to the rasterizer
#                   (default: 0.01 for KB when FOCAL_SCALING < 1, otherwise 0.2)

set -e

SCENE_ID=$1
DATA_ROOT=$2
CKPT_DIR=$3
MODE=${4:-BEAP}  # KB, BEAP, or PH

if [ -z "$SCENE_ID" ] || [ -z "$DATA_ROOT" ] || [ -z "$CKPT_DIR" ]; then
    echo "Usage: bash visualize.sh SCENE_ID DATA_ROOT CKPT_DIR [MODE] [NEAR_THRESHOLD]"
    echo "  MODE defaults to BEAP. Valid modes: BEAP, KB, PH"
    echo "  NEAR_THRESHOLD defaults to 0.01 for KB when FOCAL_SCALING < 1, otherwise 0.2"
    exit 1
fi

DATASET_DIR=$DATA_ROOT/$SCENE_ID
OUTPUT_DIR=$CKPT_DIR/$SCENE_ID

# Must match the values used during training
STEP=0.002
FOVMOD_VIS=1.3

# Iteration to load (-1 = latest checkpoint)
ITER=30000

# Network GUI settings (must match what the SIBR viewer connects to)
IP="127.0.0.1"
PORT=6009

# KB / PH mode scalings
DIST_SCALING=1.0
FOCAL_SCALING=1.0
MIRR_SHIFT=0.0

if [ -n "${5+x}" ]; then
    NEAR_THRESHOLD=$5
elif [ "$MODE" = "KB" ] && awk "BEGIN { exit !($FOCAL_SCALING < 1.0) }"; then
    NEAR_THRESHOLD=0.01
else
    NEAR_THRESHOLD=0.2
fi

echo "Scene: $SCENE_ID"
echo "Mode: $MODE"
echo "Near threshold: $NEAR_THRESHOLD"

if [ "$MODE" = "BEAP" ]; then

    BEAP_DIR_VIS="beap_fov_${FOVMOD_VIS}_step_${STEP}"
    TRAIN_MASK_FN="fov_${FOVMOD_VIS}_step_${STEP}_mask.png"

    if [ ! -d "${DATASET_DIR}/${BEAP_DIR_VIS}" ] || [ ! -f "${DATASET_DIR}/${BEAP_DIR_VIS}/${TRAIN_MASK_FN}" ]; then
        echo "== Generating BEAP images =="
        python data/scnt/scnt_kb2beap.py \
            --path "$DATASET_DIR" \
            --dst "$BEAP_DIR_VIS" \
            --step "$STEP" \
            --fov_mod "$FOVMOD_VIS" \
            --mask_dst "$TRAIN_MASK_FN"
    fi

    python visualizer.py \
        -s "$DATASET_DIR" \
        -m "$OUTPUT_DIR" \
        --iteration "$ITER" \
        --ip "$IP" \
        --port "$PORT" \
        --render_model "$MODE" \
        --sample_step "$STEP" \
        --fov_mod "$FOVMOD_VIS" \
        --near_threshold "$NEAR_THRESHOLD" \
        --sibr_mask_refcam "${DATASET_DIR}/colmap/cameras_fish.txt"
        # --mask_path "${DATASET_DIR}/${BEAP_DIR_VIS}/${TRAIN_MASK_FN}"

elif [ "$MODE" = "KB" ]; then

    echo "== Generating ray map =="
    python data/scnt/scnt_raymap.py \
        --path "$DATA_ROOT" \
        --scenes "$SCENE_ID" \
        --distortion_scaling "$DIST_SCALING" \
        --focal_scaling "$FOCAL_SCALING" \
        --mirror_shift "$MIRR_SHIFT"

    python visualizer.py \
        -s "$DATASET_DIR" \
        -m "$OUTPUT_DIR" \
        --iteration "$ITER" \
        --ip "$IP" \
        --port "$PORT" \
        --render_model "$MODE" \
        --sample_step "$STEP" \
        --fov_mod "$FOVMOD_VIS" \
        --raymap_path "${DATASET_DIR}/raymap_fisheye.npy" \
        --distortion_scaling "$DIST_SCALING" \
        --focal_scaling "$FOCAL_SCALING" \
        --mirror_shift "$MIRR_SHIFT" \
        --near_threshold "$NEAR_THRESHOLD"


elif [ "$MODE" = "PH" ]; then

    python visualizer.py \
        -s "$DATASET_DIR" \
        -m "$OUTPUT_DIR" \
        --iteration "$ITER" \
        --ip "$IP" \
        --port "$PORT" \
        --render_model "$MODE" \
        --sample_step "$STEP" \
        --fov_mod "$FOVMOD_VIS" \
        --focal_scaling "$FOCAL_SCALING" \
        --near_threshold "$NEAR_THRESHOLD"

else
    echo "Unknown MODE: $MODE"
    echo "Valid modes are: BEAP, KB, PH"
    exit 1
fi

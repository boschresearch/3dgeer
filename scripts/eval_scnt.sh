#!/usr/bin/env bash
set -e

SCENE_ID=$1
DATA_ROOT=$2
CKPT_DIR=$3
MODE=$4 # KB, BEAP, or PH
ITERS_NUM=30000
STEP_EVAL=0.002
FOVMOD_EVAL=2.0

DATASET_DIR=$2/$1
OUTPUT_DIR=$3/$1

echo "Scene: $SCENE_ID"
echo "Mode: $MODE"

if [ "$MODE" = "BEAP" ]; then

BEAP_DIR_EVAL="beap_fov_${FOVMOD_EVAL}_step_${STEP_EVAL}/"
EVAL_MASK_FN="fov_${FOVMOD_EVAL}_step_${STEP_EVAL}_mask.png"
python data/scnt/scnt_eq2beap.py --path "$DATASET_DIR" --dst "$BEAP_DIR_EVAL" --step "$STEP_EVAL" --fov_mod "$FOVMOD_EVAL" --mask_dst "$EVAL_MASK_FN"

echo "Rendering BEAP space"

python render.py \
    -m ${OUTPUT_DIR} \
    -s ${DATASET_DIR} \
    --iteration ${ITERS_NUM} \
    --camera_model FISHEYE \
    --render_model ${MODE} \
    --skip_train \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp

elif [ "$MODE" = "KB" ]; then

echo "Rendering KB fisheye"

DIST_SCALING=1.0 # set DIST_SCALING as 0 to render EQ
FOCAL_SCALING=1.0
MIRR_SHIFT=0.0

echo "== Generating ray map =="
python data/scnt/scnt_raymap.py \
    --path ${DATA_ROOT} \
    --scenes ${SCENE_ID} \
    --distortion_scaling ${DIST_SCALING} \
    --focal_scaling ${FOCAL_SCALING} \
    --mirror_shift ${MIRR_SHIFT}

echo "== Rendering =="
python render.py \
    -m ${OUTPUT_DIR} \
    -s ${DATASET_DIR} \
    --iteration ${ITERS_NUM} \
    --camera_model FISHEYE \
    --render_model ${MODE} \
    --skip_train \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp \
    --raymap_path ${DATASET_DIR}/raymap_fisheye.npy \
    --distortion_scaling ${DIST_SCALING} \
    --focal_scaling ${FOCAL_SCALING} \
    --mirror_shift ${MIRR_SHIFT}

echo "== Evaluating metrics =="
python metrics.py \
    -m ${OUTPUT_DIR} \
    --block_mask \
    --iters ${ITERS_NUM} \
    --custom_gt ${CKPT_DIR}/${SCENE_ID}/test/ours_${ITERS_NUM}/gt \

elif [ "$MODE" = "PH" ]; then

echo "Rendering Pinhole"

FOCAL_SCALING=1.0

echo "== Rendering =="
python render.py \
    -m ${OUTPUT_DIR} \
    -s ${DATASET_DIR} \
    --iteration ${ITERS_NUM} \
    --camera_model FISHEYE \
    --render_model ${MODE} \
    --skip_train \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp \
    --focal_scaling ${FOCAL_SCALING}

echo "== Evaluating metrics =="
python metrics.py \
    -m ${OUTPUT_DIR} \
    --block_mask \
    --iters ${ITERS_NUM} \
    --custom_gt ${CKPT_DIR}/${SCENE_ID}/test/ours_${ITERS_NUM}/gt \

else
echo "Unknown MODE: $MODE"
exit 1
fi


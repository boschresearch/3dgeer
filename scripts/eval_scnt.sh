#!/usr/bin/env bash
set -e

SCENE_ID=$1
DATA_ROOT=$2
CKPT_DIR=$3

DATASET_DIR=$2/$1
OUTPUT_DIR=$3/$1

STEP_EVAL=0.002
FOVMOD_EVAL=2.0
FOVMAP_DIR_EVAL=beap_fov_${FOVMOD_EVAL}_step_${STEP_EVAL}
TEST_MASK_FN=fov_${FOVMOD_EVAL}_step_${STEP_EVAL}_mask.png

DIST_SCALING=1.0
FOCAL_SCALING=1.0
MIRR_SHIFT=0.0
RENDER_MODEL=KB

ITERS_NUM=30000

echo "Scene: $SCENE_ID"
echo "Dataset: $DATASET_DIR"

echo "== Generating FoV map =="
python data/scnt/scnt_eq2beap.py \
    --path ${DATASET_DIR} \
    --dst ${FOVMAP_DIR_EVAL} \
    --step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --mask_dst ${TEST_MASK_FN}

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
    --render_model ${RENDER_MODEL} \
    --distortion_scaling ${DIST_SCALING} \
    --skip_train \
    --raymap_path ${DATASET_DIR}/raymap_fisheye.npy \
    --sample_step ${STEP_EVAL} \
    --fov_mod ${FOVMOD_EVAL} \
    --train_test_exp

echo "== Evaluating metrics =="
python metrics.py \
    -m ${OUTPUT_DIR} \
    --block_mask \
    --iters ${ITERS_NUM} \
    --custom_gt ${CKPT_DIR}/${SCENE_ID}/test/ours_${ITERS_NUM}/gt \
    --use_remap
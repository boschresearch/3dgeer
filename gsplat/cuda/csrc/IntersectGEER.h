#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "Cameras.h"
#include "Common.h"

namespace gsplat {

void preprocess_gaussians(
    int P,
    const float* means3D,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* Ks,
    const float* opacities,
    const float* viewmatrix,
    const int W, const int H,
    const float tan_fovx, const float tan_fovy,
    const CameraModelType camera_model,
    const float* radial_coeffs,
    const float* tangential_coeffs,
    const float* thin_prism_coeffs,
    const FThetaCameraDistortionParameters ftheta_coeffs,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const int tile_size, const int tile_width, const int tile_height,
    int* radii,
    int* pbf_id,
    float* depths,
    int* tiles_touched
);

void duplicate_with_keys(
    int P,
    const float* depths,
    const int64_t* offsets,
    int64_t* isect_ids,
    int32_t* flatten_ids,
    const int* radii,
    const int4* aabb,
    const int* tiles_touched,
    const int tile_size, const int tile_width, const int tile_height
);

} // namespace gsplat

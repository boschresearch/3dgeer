#pragma once

#include <cstdint>
#include <cuda_runtime.h>


namespace gsplat {

void preprocess_gaussians(
    int P,
    // int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
    const float* Ks,
	const float* opacities,
	// const float* shs,
	// bool* clamped,
	// const float* colors_precomp,
	const float* viewmatrix,
	const float* mirror_transformed_tan_theta,
	const float* mirror_transformed_tan_phi, 
	// const glm::vec3* cam_pos,
	const int W, int H,
    const float tan_fovx, float tan_fovy,

    const CameraModelType camera_model,
	// const float focal_x, float focal_y,
	// const float principal_x, float principal_y,
	const float* radial_coeffs, // [C, 4] or [C, 6]
	const float near_plane,
	const float far_plane,
	const float radius_clip,

    const int tile_size, const int tile_width, const int tile_height,

    // // Outputs (except xmap, ymap, h_opacity, prefilted, and antialiasing)
	int* radii,
	int* aabb,
	float4* beap_xxyy,
	const float* xmap, // Set to nullptr for now until KB is reintegrated
	const float* ymap, // Set to nullptr for now until KB is reintegrated
	float3* means3D_view,
	float* depths,
	// // float* rgb,
	// // float2* h_opacity,
	float3* w2o,
	// const dim3 grid,
	int* tiles_touched
	// bool prefiltered
	// // bool antialiasing
);

void duplicate_with_keys(
	int P,
	const float3* points_xyz,
	const float3* w2o,
	// const float2* h_opacity,
	const float* depths,
	const int64_t* offsets,

	// uint64_t* gaussian_keys_unsorted,
	// uint32_t* gaussian_values_unsorted,
    int64_t* isect_ids,       // [n_isects]
    int32_t* flatten_ids,      // [n_isects]

	int* radii,
	const int4* aabb,
	const float4* beap_xxyy,
	const float* xmap,
	const float* ymap,
	const int W, const int H,
	int* tiles_touched,
    const int tile_size, const int tile_width, const int tile_height, const uint32_t tile_n_bits
	// dim3 grid
);

// void identify_tile_ranges(int L, int64_t* point_list_keys, int64_t* ranges);

}
/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* mirror_transformed_tan_theta, 
		const float* mirror_transformed_tan_phi,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float principal_x, float principal_y,
		const float* distortion_coeffs,
		const float tan_fovx, float tan_fovy,
		int* radii,
		int* pbf_id,
		float4* pbf_tan,
		const float* xmap, 
		const float* ymap,
		float3* points_xyz_view, 
		float* depths,
		float* colors,
		float2* h_opacity,
		float3* w2o,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing,
		int mode);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		int mode,
		float focal_x, float focal_y,
		const float* tan_theta,
		const float* tan_phi,
		const float* raymap,
		const float4* pbf_tan,
		const float3* points_xyz_view,
		const float* features,
		const float2* h_opacity,
		const float3* w2o,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* depths,
		float* depth);
}

#endif

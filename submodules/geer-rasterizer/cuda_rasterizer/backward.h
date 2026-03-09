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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* tan_theta,
		const float* tan_phi,
		const float* bg_color,
		const float3* means3D_view,
		const float2* h_opacity,
		const float3* w2o,
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_invdepths,
		float3* dL_dmean2D,
		glm::vec3* dL_dmeans,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_dinvdepths,
		glm::vec3* dL_dsigmaInv);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const float3* means3D_view,
		const float* depths,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const float* opacities,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float2* h_opacity,
		const float3* w2o,
		const float* view,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		float3* dL_dmean2D,
		const float* dL_dinvdepth,
		float* dL_dopacity,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		const glm::vec3* dL_dsigmaInv,
		bool antialiasing);
}

#endif

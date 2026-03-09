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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;          // Euclidean distance from camera to each Gaussian centre (used for depth-sorted blending)
		char* scanning_space;
		bool* clamped;          // Per-channel SH clamping flags (for backward pass)
		int* pbf;               // Pixel-space PBF of each Gaussian: [x_min, x_max, y_min, y_max]
		float4* pbf_tan;        // PBF (Particle Bounding Frustum) in ray-direction tangent space {tan_theta_min, tan_theta_max, tan_phi_min, tan_phi_max}
		int* internal_radii;
		float3* means3D_view;   // Gaussian centre positions in view/camera space
		float2* h_opacity;      // Packed {x=antialiasing variance h_var, y=antialiasing-scaled opacity}
		float3* w2o;            // World-to-object (canonical) matrix rows: 3 float3 per Gaussian = Σ^{-1/2}R_view^T
		float* rgb;             // Pre-computed RGB colors (from SH evaluation)
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;          // Per-tile [start, end) range in the sorted Gaussian list
		uint32_t* n_contrib;    // Per-pixel: index of the last contributing Gaussian (for backward)
		float* accum_alpha;     // Per-pixel accumulated transmittance T (= product of (1-alpha) terms)

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;  // Unsorted keys: upper 32 bits = tile ID, lower 32 bits = depth bits
		uint64_t* point_list_keys;           // Sorted keys
		uint32_t* point_list_unsorted;       // Unsorted Gaussian indices
		uint32_t* point_list;               // Sorted Gaussian indices (tile-then-depth order)
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};
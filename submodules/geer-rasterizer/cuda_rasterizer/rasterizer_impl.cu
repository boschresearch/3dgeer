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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, false, p_view);
}

__global__ void extractRaymapChannel(const float* raymap,
	int W, int H,
	float* channelmap, int c, 
	int N) 
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N || raymap == nullptr)
		return;
	int ray_idx = idx;
	if (c == 1) {
		int x = int(idx / H);
		int y = idx - H * x;
		ray_idx = y * W + x;
	}
	
	channelmap[idx] = raymap[3 * ray_idx + c] / raymap[3 * ray_idx + 2];
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float3* points_xyz,
	const float3* w2o,
	const float2* h_opacity,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	const int4* pbf,
	const float4* pbf_tan,
	const float* xmap,
	const float* ymap,
	const int W, int H,
	uint32_t* tiles_touched,
	dim3 grid,
	int mode)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if ((radii[idx] > 0) && (tiles_touched[idx] > 0))
	{
		const bool use_kb_sorted_tile_culling = false;

		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

		// Update unsorted arrays with Gaussian idx for every tile that Gaussian touches

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is | tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		if ((xmap == nullptr) || (mode != 1) || !use_kb_sorted_tile_culling) {
			uint2 rect_min, rect_max;
			getRect2(pbf[idx], rect_min, rect_max, grid);
	
			for (int y = rect_min.y; y < rect_max.y; y++)
			{
				for (int x = rect_min.x; x < rect_max.x; x++)
				{
					uint64_t key = y * grid.x + x;
					key <<= 32;
					key |= *((uint32_t*)&depths[idx]);
					gaussian_keys_unsorted[off] = key;
					gaussian_values_unsorted[off] = idx;
					off++;
				}
			}
		} else {
			tiles_touched[idx] = duplicateToTilesTouched(
				points_xyz[idx], w2o + 3 * idx, h_opacity[idx].y,
				pbf[idx], pbf_tan[idx], grid,
				W, H,
				idx, off, depths[idx],
				gaussian_keys_unsorted,
				gaussian_values_unsorted,
				xmap, ymap);
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1) ranges[currtile].y = L;
}

// ranges for each tile: Compute the length of each tile's range in the full sorted list
__global__ void computeRangeLengths(const uint2* ranges, int* range_len, int N)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
		return;
	range_len[idx] = ranges[idx].y - ranges[idx].x;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means3D_view, P, 128);
	obtain(chunk, geom.h_opacity, P, 128);
	obtain(chunk, geom.w2o, P * 3, 128);
	obtain(chunk, geom.pbf, P * 4, 128); 
	obtain(chunk, geom.pbf_tan, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* viewmatrix,
	const float* mirror_transformed_tan_theta, const float* mirror_transformed_tan_phi, 
	const float* tan_theta, const float* tan_phi,
	const float* cam_pos,
	const float focal_x, float focal_y, 
	const float principal_x, float principal_y,
	const float* distortion_coeffs,
	const float* raymap, 
	float* xmap, float* ymap,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* kernel_times,
	float* out_color,
	float* depth,
	bool antialiasing,
	const int mode,
	int* radii,
	int* range_len,
	float near_threshold,
	bool debug,
	int asso_mode)
{
	cudaEvent_t overallStart, overallStop;
	cudaEvent_t preprocessStart, preprocessStop;
	cudaEvent_t duplicateStart, duplicateStop;
	cudaEvent_t sortStart, sortStop;
	cudaEvent_t renderStart, renderStop;
	cudaEventCreate(&overallStart);
	cudaEventCreate(&overallStop);
	cudaEventCreate(&preprocessStart);
	cudaEventCreate(&preprocessStop);
	cudaEventCreate(&duplicateStart);
	cudaEventCreate(&duplicateStop);
	cudaEventCreate(&sortStart);
	cudaEventCreate(&sortStop);
	cudaEventCreate(&renderStart);
	cudaEventCreate(&renderStop);
	float milliseconds_overall, milliseconds_preprocess, milliseconds_duplicate, milliseconds_sort, milliseconds_render;

	cudaEventRecord(overallStart, 0);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	if ((mode == 1) && (raymap != nullptr)) {
		extractRaymapChannel << <(width * height + 255) / 256, 256 >> > (
			raymap,
			width, height,
			xmap, 0,
			width * height);
		CHECK_CUDA(, debug);
	
		extractRaymapChannel << <(width * height + 255) / 256, 256 >> > (
			raymap,
			width, height,
			ymap, 1,
			width * height);
		CHECK_CUDA(, debug);
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	cudaEventRecord(preprocessStart, 0);
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		colors_precomp,
		viewmatrix, 
		mirror_transformed_tan_theta, 
		mirror_transformed_tan_phi,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
	    principal_x, principal_y,
	    distortion_coeffs,
		tan_fovx, tan_fovy,
		radii,
		geomState.pbf,
		geomState.pbf_tan,
		xmap, ymap,
		geomState.means3D_view,
		geomState.depths,
		geomState.rgb,
		geomState.h_opacity,
		geomState.w2o,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing,
		mode,
		near_threshold,
		asso_mode
	), debug)

	cudaEventRecord(preprocessStop, 0);
	cudaEventSynchronize(preprocessStop);
	cudaEventElapsedTime(&milliseconds_preprocess, preprocessStart, preprocessStop);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	cudaEventRecord(duplicateStart, 0);
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means3D_view,
		geomState.w2o,
		geomState.h_opacity,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		(int4*) geomState.pbf,
		geomState.pbf_tan,
		xmap, ymap,
		width, height,
		geomState.tiles_touched,
		tile_grid,
		mode)
	CHECK_CUDA(, debug)
	cudaEventRecord(duplicateStop, 0);
	cudaEventSynchronize(duplicateStop);
	cudaEventElapsedTime(&milliseconds_duplicate, duplicateStart, duplicateStop);
	
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cudaEventRecord(sortStart, 0);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)
	cudaEventRecord(sortStop, 0);
	cudaEventSynchronize(sortStop);
	cudaEventElapsedTime(&milliseconds_sort, sortStart, sortStop);

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// ranges for each tile
	computeRangeLengths << <(tile_grid.x * tile_grid.y + 255) / 256, 256 >> > (
		imgState.ranges,
		range_len, // ranges for each tile
		tile_grid.x * tile_grid.y);
	CHECK_CUDA(, debug);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	cudaEventRecord(renderStart, 0);
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		mode,
		focal_x, focal_y,
		tan_theta, tan_phi, 
		raymap, 
		geomState.pbf_tan, 
		geomState.means3D_view, 
		feature_ptr,
		geomState.h_opacity,
		geomState.w2o,  
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)

	cudaEventRecord(overallStop, 0);
	cudaEventSynchronize(overallStop);

	cudaEventElapsedTime(&milliseconds_render, renderStart, overallStop);
	cudaEventElapsedTime(&milliseconds_overall, overallStart, overallStop);
	kernel_times[0] = milliseconds_overall;
	kernel_times[1] = milliseconds_preprocess;
	kernel_times[2] = milliseconds_duplicate;
	kernel_times[3] = milliseconds_sort;
	kernel_times[4] = milliseconds_render;

	cudaEventDestroy(overallStart);
	cudaEventDestroy(overallStop);

	cudaEventDestroy(preprocessStart);
	cudaEventDestroy(preprocessStop);

	cudaEventDestroy(duplicateStart);
	cudaEventDestroy(duplicateStop);

	cudaEventDestroy(sortStart);
	cudaEventDestroy(sortStop);

	cudaEventDestroy(renderStart);
	cudaEventDestroy(renderStop);
	
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* tan_theta, 
	const float* tan_phi, 
	const float* raymap,
	const int mode,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* viewmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dsigmaInv,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		tan_theta, tan_phi,
		raymap,
		focal_x, focal_y,
		mode,
		background,
		geomState.means3D_view, 
		geomState.h_opacity,
		geomState.w2o,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth,
		(glm::vec3*)dL_dsigmaInv), debug);

	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		geomState.means3D_view,
		geomState.depths,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		geomState.h_opacity,
		geomState.w2o,
		viewmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		(glm::vec3*)dL_dsigmaInv,
		antialiasing), debug);
}

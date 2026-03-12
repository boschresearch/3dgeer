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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#include <thrust/sort.h>
#include <thrust/binary_search.h>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__device__ const float ut_alpha = 1.0f;
__device__ const float ut_beta = 2.0f;
__device__ const float ut_kappa = 0.0f;
__device__ const float ut_lambda = ut_alpha * ut_alpha * (3.f + ut_kappa) - 3.f;

__device__ __forceinline__ float sq(float x) { return x * x; }
__device__ __forceinline__ float cube(float x) { return x * x * x; }


__device__ __forceinline__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(const float3 &a, const float3 &b) {
    return make_float3(a.y * b.z - a.z * b.y, 
                       a.z * b.x - a.x * b.z, 
                       a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float2 divide_z(const glm::vec3& v) {
    return make_float2(v.x / v.z, v.y / v.z);
}

__device__ __forceinline__ float2 atan(const float2& v) {
	return make_float2(atan(v.x), atan(v.y));
}

__device__ __forceinline__ float3 toFloat3(const glm::vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float2 invinterpolated_uv(
	const float focal_x, const float focal_y, 
	const float principal_x, const float principal_y, 
	const float4 dist_coeff, 
	const float tan_x, const float tan_y) {
	// Compute the inverse interpolation for the UV coordinates
	float2 uv_indices;
	float radius = sqrtf(sq(tan_x) + sq(tan_y));
	float angle = atanf(radius);
	float angle_sq = sq(angle);
	float angle_sq_sq = sq(angle_sq);

	float r = angle * (1.0 + dist_coeff.x * angle_sq + dist_coeff.y * angle_sq_sq + dist_coeff.z * angle_sq * angle_sq_sq + dist_coeff.w * angle_sq_sq * angle_sq_sq);
	uv_indices.x = (tan_x * r * focal_x) / radius + principal_x;
	uv_indices.y = (tan_y * r * focal_y) / radius + principal_y;
	return uv_indices;
}

__forceinline__ __device__ void getRect2(const int4 pbf, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((pbf.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((pbf.z) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((pbf.y + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((pbf.w + BLOCK_Y - 1) / BLOCK_Y)))
	};
}


__forceinline__ __device__ void searchsorted_intersect(
	const float* ref_start, int span,
	const float* values,
	int* indices
) {
	thrust::lower_bound(thrust::device, ref_start, ref_start + span, values, values + 2, indices);
}

// Check whether the ray direction for pixel `idx` falls within the
// Particle Bounding Frustum (PBF) defined in tan-space as pbf_tan.
// pbf_tan: (.x=tan_theta_min, .y=tan_theta_max, .z=tan_phi_min, .w=tan_phi_max)
__forceinline__ __device__ bool checkValid(
	const float* raymap,
	const int idx,
	const float4 pbf_tan
) {
	float2 rayf = {(float)raymap[idx * 3] / (float)raymap[idx * 3 + 2], (float)raymap[idx * 3 + 1] / (float)raymap[idx * 3 + 2]};
	if (rayf.x < pbf_tan.x || rayf.x > pbf_tan.y) return false;
	if (rayf.y < pbf_tan.z || rayf.y > pbf_tan.w) return false;
	return true;
}

// Emit (tile_id | depth, gaussian_id) key/value pairs for every tile that the
// Particle Bounding Frustum (PBF) of this Gaussian overlaps.  When xmap/ymap
// are provided the PBF is projected onto the KB image grid via sorted intersection;
// otherwise a simple tile-rectangle loop is used (BEAP mode).
__forceinline__ __device__ uint32_t duplicateToTilesTouched(    
	const float3 points_xyz,
	const float3* w2o,
	const float opac,
	int4 pbf,
	float4 pbf_tan,
	const dim3 grid,
	const int W, int H,
    uint32_t idx, uint32_t off, float depth,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const float* xmap,
	const float* ymap
	)
{
	uint2 rect_min, rect_max;

	getRect2(pbf, rect_min, rect_max, grid);

	int y_span = rect_max.y - rect_min.y;
	int x_span = rect_max.x - rect_min.x;

	// If no tiles are touched, return 0
	if (y_span * x_span == 0) {
		return 0;
	}

	bool isY = y_span > x_span;
	const uint2 rect_max_ = isY ? rect_max : make_uint2(rect_max.y, rect_max.x);
	const uint2 rect_min_ = isY ? rect_min : make_uint2(rect_min.y, rect_min.x);
	const int4 pbf_ = isY ? pbf : make_int4(pbf.z, pbf.w, pbf.x, pbf.y);
	const float2 pbf_tan_ = isY ? make_float2(pbf_tan.x, pbf_tan.y) : make_float2(pbf_tan.z, pbf_tan.w);
	const float* cmap = isY ? xmap : ymap;
	const int W_ = isY ? W : H;
	// const int H_ = isY ? H : W;

	uint32_t tiles_count = 0;
    int2 slice_intersect_top, slice_intersect_bottom;
	int slice_lefttop, slice_leftbottom;

	// For each tile that the bounding rect overlaps, emit a 
	// key/value pair. The key is |  tile ID  |      depth      |,
	// and the value is the ID of the Gaussian. Sorting the values 
	// with this key yields Gaussian IDs in a list, such that they
	// are first sorted by tile and then by depth. 
	for (int y = rect_min_.y; y < rect_max_.y; y++)
	{
		// Get original BEAP ranged slice;
		slice_leftbottom = min(max(pbf_.z, y * BLOCK_Y), pbf_.w) * W_ + pbf_.x;
		searchsorted_intersect(cmap + slice_leftbottom, pbf_.y - pbf_.x + 1, (float*)(&pbf_tan_), (int*)(&slice_intersect_bottom));

		slice_lefttop = min(max(pbf_.z, (y * BLOCK_Y + BLOCK_Y - 1)), pbf_.w) * W_ + pbf_.x;
		searchsorted_intersect(cmap + slice_lefttop, pbf_.y - pbf_.x + 1, (float*)(&pbf_tan_), (int*)(&slice_intersect_top));

		// Cull out useless tiles;
		int tmp_left = min(max(0, min(slice_intersect_top.x, slice_intersect_bottom.x)), pbf_.y - pbf_.x);
		int tmp_right = min(max(0, max(slice_intersect_top.y, slice_intersect_bottom.y)), pbf_.y - pbf_.x);
		if (tmp_left >= tmp_right) {
			continue;
		}
		int min_tile_x = max(rect_min_.x,
            min(rect_max_.x, (int)((pbf_.x + tmp_left) / BLOCK_X))
        );
        int max_tile_x = max(rect_min_.x,
            min(rect_max_.x, (int)((pbf_.x + tmp_right + BLOCK_X - 1) / BLOCK_X))
        );
		tiles_count += (max_tile_x - min_tile_x);
		for (int x = min_tile_x; x < max_tile_x; x++)
		{

			if (gaussian_keys_unsorted != nullptr) {
				uint64_t key = isY ? y * grid.x + x : x * grid.x + y;
				key <<= 32;
				key |= *((uint32_t*)&depth);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
	return tiles_count;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

// ── Packed-type conventions used throughout the rasterizer ──────────────────
// float2 h_opacity  : { x = antialiasing variance h_var,
//                       y = antialiasing-scaled Gaussian opacity }
// float4 pbf_tan    : PBF (Particle Bounding Frustum) in ray-direction
//                     tangent space { x = tan_theta_min,  y = tan_theta_max,
//                                     z = tan_phi_min,    w = tan_phi_max }
// float3 w2o[3]     : rows of the world-to-object (canonical) matrix Σ^{-1/2}R_view^T
//                     (one float3 per row → 3 rows total per Gaussian)
// ────────────────────────────────────────────────────────────────────────────

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	bool prefiltered,
	float3& p_view,
	float near_threshold = 0.2f)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space 
	p_view = transformPoint4x3(p_orig, viewmatrix);

 	if (p_view.z <= near_threshold)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif

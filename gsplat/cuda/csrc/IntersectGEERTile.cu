#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

// for CUB_WRAPPER
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "IntersectGEER.h"
// #include "Utils.cuh"

#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include <cstdio>

namespace gsplat {

namespace cg = cooperative_groups;

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = { // Matrix is row major in gsplat
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
	};
	return transformed;
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float near_plane,
	const float far_plane,
	// bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space 
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= near_plane || p_view.z >= far_plane)
	{
		// if (prefiltered)
		// {
		// 	printf("Point is filtered although prefiltered is set. This shouldn't happen!");
		// 	__trap();
		// }
		return false;
	}
	return true;
}

__device__ glm::mat3 computeRotationMatrix(const glm::vec4 rot, const float* viewmatrix)
{
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
		2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
		2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y)
	);

	// viewmatrix float* has been the column-major, 0,1,2 is the column; 
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 R_view = W * R;
	return R_view;
}

__device__ __forceinline__ float3 toFloat3(const glm::vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ float sq(float x) { return x * x; }

__device__ bool computeCov3D(const glm::vec3 scale, const float mod, const glm::mat3 R_view, float* cov3D, const float h_var)
{
	glm::mat3 R_scaled = glm::mat3(
        R_view[0] * (sq(scale.x * mod) + h_var),
        R_view[1] * (sq(scale.y * mod) + h_var),
        R_view[2] * (sq(scale.z * mod) + h_var)
	);

	glm::mat3 Cov3D_mat = R_scaled * glm::transpose(R_view);

	// Covariance is symmetric, only store upper right
	cov3D[0] = Cov3D_mat[0][0];
	cov3D[1] = Cov3D_mat[0][1];
	cov3D[2] = Cov3D_mat[0][2];
	cov3D[3] = Cov3D_mat[1][1];
	cov3D[4] = Cov3D_mat[1][2];
	cov3D[5] = Cov3D_mat[2][2];

	const float det_cov_plus_h_cov = cov3D[0] * cov3D[3] * cov3D[5] + 2.f * cov3D[1] * cov3D[2] * cov3D[4] - cov3D[0] * cov3D[4] * cov3D[4] - cov3D[3] * cov3D[2] * cov3D[2] - cov3D[5] * cov3D[1] * cov3D[1];

	if (det_cov_plus_h_cov == 0.0f)
		return false;

	return true;
}

__device__ void omni_map_xy(const float4& m, const float xi, float* result) {
	float _m0 = xi * sqrtf(1 + m.x * m.x);
	float _m1 = xi * sqrtf(1 + m.y * m.y);
	float _m2 = xi * sqrtf(1 + m.z * m.z);
	float _m3 = xi * sqrtf(1 + m.w * m.w);
	result[0] = m.x / (1 + _m0);
	result[1] = m.x / (1 - _m0);
	result[2] = m.y / (1 + _m1);
	result[3] = m.y / (1 - _m1);
	result[4] = m.z / (1 + _m2);
	result[5] = m.z / (1 - _m2);
	result[6] = m.w / (1 + _m3);
	result[7] = m.w / (1 - _m3);
}

__device__ void omni_map_fov(const float tan_fovx, const float tan_fovy, const float xi, float* result) {
	float _tan_fovx = xi * sqrtf(1 + tan_fovx * tan_fovx);
	float _tan_fovy = xi * sqrtf(1 + tan_fovy * tan_fovy);
	result[0] = tan_fovx / (1 + _tan_fovx);
	result[1] = -result[0];
	result[2] = tan_fovy / (1 + _tan_fovy);
	result[3] = -result[2];
}

__forceinline__ __device__ float omni_map_float(const float m, const float z, const float xi) {
    if (xi == 0.0f) {
        return m;
    }
	return m / (1 + xi * (z / fabsf(z)) * sqrtf(1 + m * m));
}

__device__ bool computePBF(
    const glm::vec3 scale, const float mod, const glm::mat3 R_view, const float3 p_view, const float lambda, float4& aabb, const float tan_fovx, const float tan_fovy, float h_var)
{
    float lambda_sq = sq(lambda);
	float cov3d[6];
	if (!computeCov3D(scale, mod, R_view, cov3d, h_var))
		return false;
	
	float Tc_22 = lambda_sq * cov3d[5] - p_view.z * p_view.z;
	if (Tc_22 == 0.0f)
		return false;

	float Tc_00 = lambda_sq * cov3d[0] - p_view.x * p_view.x;
    float Tc_02 = lambda_sq * cov3d[2] - p_view.x * p_view.z;
    float Tc_11 = lambda_sq * cov3d[3] - p_view.y * p_view.y;
    float Tc_12 = lambda_sq * cov3d[4] - p_view.y * p_view.z;

    float center[2];
    center[0] = Tc_02 / Tc_22;
    center[1]= Tc_12 / Tc_22;

    float half_extend[2];
    half_extend[0] = sqrtf(Tc_02 * Tc_02 - Tc_22 * Tc_00) / fabsf(Tc_22);
    half_extend[1] = sqrtf(Tc_12 * Tc_12 - Tc_22 * Tc_11) / fabsf(Tc_22);

	float neg = false;
	if (isnan(half_extend[0]))
	{ 
		half_extend[0] = fmaxf(fabsf(center[0] - tan_fovx), fabsf(center[0] + tan_fovx));
		neg = true; 
	}
	if (isnan(half_extend[1]))
	{ 
		half_extend[1] = fmaxf(fabsf(center[1] - tan_fovy), fabsf(center[1] + tan_fovy));
		neg = true;
	}
	float _left = center[0] - half_extend[0];
	float _right = center[0] + half_extend[0];
	float _bottom = center[1] - half_extend[1];
	float _upper = center[1] + half_extend[1];

    aabb.x = _left;
    aabb.y = _right;
	aabb.z = _bottom;
    aabb.w = _upper;

	// If half-extend is negative, return and do not compute the omni
	if (neg) return;

	// Omni mapping for AABB
	float xi = 1.0;
    float aabb_omni[8];
	omni_map_xy(aabb, xi, aabb_omni);

    const float eps = 1e-6f;
    float depth = p_view.z;
    depth = (fabsf(depth) < eps) ? eps : depth; // Prevent division by zero
    float gaus_center_omni[2] = {
        omni_map_float(p_view.x / depth, depth, xi),
        omni_map_float(p_view.y / depth, depth, xi)
    };

    float fov_omni[4];
    omni_map_fov(tan_fovx, tan_fovy, xi, fov_omni);

    float aa_omni[4] = { aabb_omni[0], aabb_omni[1], aabb_omni[2], aabb_omni[3] };
	float bb_omni[4] = { aabb_omni[4], aabb_omni[5], aabb_omni[6], aabb_omni[7] };
	float a_min = -INFINITY;
	float a_max = INFINITY;
	float b_min = -INFINITY;
	float b_max = INFINITY;

    int a_min_idx = -1;
	int a_max_idx = -1;
	int b_min_idx = -1;
	int b_max_idx = -1;

	for (int i = 0; i < 4; i++) {
        if (aa_omni[i] < gaus_center_omni[0] && aa_omni[i] >= a_min){
            a_min = aa_omni[i];
            a_min_idx = i;
        }
        if (aa_omni[i] > gaus_center_omni[0] && aa_omni[i] <= a_max){ 
            a_max = aa_omni[i];
            a_max_idx = i;
        }
		if (bb_omni[i] < gaus_center_omni[1] && bb_omni[i] >= b_min){
            b_min = bb_omni[i];
            b_min_idx = i;
        }
        if (bb_omni[i] > gaus_center_omni[1] && bb_omni[i] <= b_max){
            b_max = bb_omni[i];
            b_max_idx = i;
        }
    }
    if (a_min < fov_omni[1]) a_min_idx = 4;
    if (a_min > fov_omni[0]) a_min_idx = 5;

    if (a_max < fov_omni[1]) a_max_idx = 4;
    if (a_max > fov_omni[0]) a_max_idx = 5;

    if (b_min < fov_omni[3]) b_min_idx = 4;
    if (b_min > fov_omni[2]) b_min_idx = 5;

    if (b_max < fov_omni[3]) b_max_idx = 4;
    if (b_max > fov_omni[2]) b_max_idx = 5;

    if (a_min_idx == 4) aabb.x = -tan_fovx;
    else if (a_min_idx == 5) aabb.x = tan_fovx;
    else if (a_min_idx == 0) aabb.x = _left;
    else if (a_min_idx == 1) aabb.x = _left;
    else if (a_min_idx == 2) aabb.x = _right;
    else if (a_min_idx == 3) aabb.x = _right;
    
    if (a_max_idx == 5) aabb.y = tan_fovx;
    else if (a_max_idx == 4) aabb.y = -tan_fovx;
    else if (a_max_idx == 0) aabb.y = _left;
    else if (a_max_idx == 1) aabb.y = _left;
    else if (a_max_idx == 2) aabb.y = _right;
    else if (a_max_idx == 3) aabb.y = _right;

    if (b_min_idx == 4) aabb.z = -tan_fovy;
    else if (b_min_idx == 5) aabb.z = tan_fovy;
    else if (b_min_idx == 0) aabb.z = _bottom;
    else if (b_min_idx == 1) aabb.z = _bottom;
    else if (b_min_idx == 2) aabb.z = _upper;
    else if (b_min_idx == 3) aabb.z = _upper;

    if (b_max_idx == 5) aabb.w = tan_fovy;
    else if (b_max_idx == 4) aabb.w = -tan_fovy;
    else if (b_max_idx == 0) aabb.w = _bottom;
    else if (b_max_idx == 1) aabb.w = _bottom;
    else if (b_max_idx == 2) aabb.w = _upper;
    else if (b_max_idx == 3) aabb.w = _upper;

    return true;
}

__forceinline__ __device__ void getRect2(const int4 aabb, const int tile_size, const int tile_width, const int tile_height, uint2& rect_min, uint2& rect_max)
{
	rect_min = {
		static_cast<unsigned int>(min(tile_width, max((int)0, (int)((aabb.x) / tile_size)))),
		static_cast<unsigned int>(min(tile_height, max((int)0, (int)((aabb.z) / tile_size))))
	};
	rect_max = {
		static_cast<unsigned int>(min(tile_width, max((int)0, (int)((aabb.y + tile_size - 1) / tile_size)))),
		static_cast<unsigned int>(min(tile_height, max((int)0, (int)((aabb.w + tile_size - 1) / tile_size))))
	};
}

__forceinline__ __device__ int lower_bound(const float* first, const float* last, float val) {
    const float* it = first;
    int count = last - first;
    while (count > 0) {
        int step = count / 2;
        if (it[step] < val) {
            it += step + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return it - first;
}

__forceinline__ __device__ void searchsorted_pbf(
    const float* ref_u, int u_span,
    const float* ref_v, int v_span,
    const float* uv_values,
    int* u_indices, int* v_indices) {
    u_indices[0] = lower_bound(ref_u, ref_u + u_span, uv_values[0]);
    u_indices[1] = lower_bound(ref_u, ref_u + u_span, uv_values[1]);
    v_indices[0] = lower_bound(ref_v, ref_v + v_span, uv_values[2]);
    v_indices[1] = lower_bound(ref_v, ref_v + v_span, uv_values[3]);
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

__forceinline__ __device__ void invinterpolated_aabb(
	const int W, int H,
	const float focal_x, float focal_y, 
	const float principal_x, float principal_y, 
	const float4 dist_coeff, 
	const float4 tan_xxyy,
    int* u_indices, int* v_indices) {
	if ((tan_xxyy.y < 0.0f && tan_xxyy.z > 0.0f) || (tan_xxyy.x > 0.0f && tan_xxyy.w < 0.0f)) {
		float2 _left_bottom = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.x, tan_xxyy.z);
		float2 _right_top = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.y, tan_xxyy.w);
		u_indices[0] = (int)floor(_left_bottom.x);
		u_indices[1] = (int)floor(_right_top.x);
		v_indices[0] = (int)floor(_left_bottom.y);
		v_indices[1] = (int)floor(_right_top.y);
	} else if ((tan_xxyy.y < 0.0f && tan_xxyy.w < 0.0f) || (tan_xxyy.x > 0.0f && tan_xxyy.z > 0.0f)) {
		float2 _left_top = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.x, tan_xxyy.w);
		float2 _right_bottom = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.y, tan_xxyy.z);
		u_indices[0] = (int)floor(_left_top.x);
		u_indices[1] = (int)floor(_right_bottom.x);
		v_indices[0] = (int)floor(_right_bottom.y);
		v_indices[1] = (int)floor(_left_top.y);
	} else if ((tan_xxyy.x < 0.0f && tan_xxyy.y > 0.0f) && tan_xxyy.z > 0.0f) {
		float2 _left_bottom = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.x, tan_xxyy.z);
		float2 _right_bottom = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.y, tan_xxyy.z);
		float2 _mid_top = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, 0.0f, tan_xxyy.w);
		u_indices[0] = (int)floor(_left_bottom.x);
		u_indices[1] = (int)floor(_right_bottom.x);
		v_indices[0] = (int)floor(fminf(_left_bottom.y, _right_bottom.y));
		v_indices[1] = (int)floor(_mid_top.y);
	} else if ((tan_xxyy.x < 0.0f && tan_xxyy.y > 0.0f) && tan_xxyy.w < 0.0f) {
		float2 _right_top = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.y, tan_xxyy.w);
		float2 _left_top = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.x, tan_xxyy.w);
		float2 _mid_bottom = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, 0.0f, tan_xxyy.z);
		u_indices[0] = (int)floor(_left_top.x);
		u_indices[1] = (int)floor(_right_top.x);
		v_indices[0] = (int)floor(_mid_bottom.y);
		v_indices[1] = (int)floor(fmaxf(_left_top.y, _right_top.y));
	} else if ((tan_xxyy.z < 0.0f && tan_xxyy.w > 0.0f) && tan_xxyy.y < 0.0f) {
		float2 _right_top = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.y, tan_xxyy.w);
		float2 _right_bottom = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.y, tan_xxyy.z);
		float2 _left_mid = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.x, 0.0f);
		u_indices[0] = (int)floor(_left_mid.x);
		u_indices[1] = (int)floor(fmaxf(_right_bottom.x, _right_top.x));
		v_indices[0] = (int)floor(_right_bottom.y);
		v_indices[1] = (int)floor(_right_top.y);
	} else if ((tan_xxyy.z < 0.0f && tan_xxyy.w > 0.0f) && tan_xxyy.x > 0.0f) {
		float2 _left_bottom = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.x, tan_xxyy.z);
		float2 _left_top = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.x, tan_xxyy.w);
		float2 _right_mid = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.y, 0.0f);
		u_indices[0] = (int)floor(fminf(_left_bottom.x, _left_top.x));
		u_indices[1] = (int)floor(_right_mid.x);
		v_indices[0] = (int)floor(_left_bottom.y);
		v_indices[1] = (int)floor(_left_top.y);
	} else {
		float2 _mid_top = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, 0.0f, tan_xxyy.w);
		float2 _mid_bottom = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, 0.0f, tan_xxyy.z);
		float2 _left_mid = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.x, 0.0f);
		float2 _right_mid = invinterpolated_uv(focal_x, focal_y, principal_x, principal_y, dist_coeff, tan_xxyy.y, 0.0f);
		u_indices[0] = (int)floor(_left_mid.x);
		u_indices[1] = (int)floor(_right_mid.x);
		v_indices[0] = (int)floor(_mid_bottom.y);
		v_indices[1] = (int)floor(_mid_top.y);
	}
	u_indices[0] = fminf(fmaxf((int)0, u_indices[0]), (int)(W-1));
	u_indices[1] = fminf(fmaxf((int)0, u_indices[1]), (int)(W-1));
	v_indices[0] = fminf(fmaxf((int)0, v_indices[0]), (int)(H-1));
	v_indices[1] = fminf(fmaxf((int)0, v_indices[1]), (int)(H-1));
}

// // __device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
// // {
// // 	// The implementation is loosely based on code for 
// // 	// "Differentiable Point-Based Radiance Fields for 
// // 	// Efficient View Synthesis" by Zhang et al. (2022)
// // 	glm::vec3 pos = means[idx];
// // 	glm::vec3 dir = pos - campos;
// // 	dir = dir / glm::length(dir);

// // 	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
// // 	glm::vec3 result = SH_C0 * sh[0];

// // 	if (deg > 0)
// // 	{
// // 		float x = dir.x;
// // 		float y = dir.y;
// // 		float z = dir.z;
// // 		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

// // 		if (deg > 1)
// // 		{
// // 			float xx = x * x, yy = y * y, zz = z * z;
// // 			float xy = x * y, yz = y * z, xz = x * z;
// // 			result = result +
// // 				SH_C2[0] * xy * sh[4] +
// // 				SH_C2[1] * yz * sh[5] +
// // 				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
// // 				SH_C2[3] * xz * sh[7] +
// // 				SH_C2[4] * (xx - yy) * sh[8];

// // 			if (deg > 2)
// // 			{
// // 				result = result +
// // 					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
// // 					SH_C3[1] * xy * z * sh[10] +
// // 					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
// // 					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
// // 					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
// // 					SH_C3[5] * z * (xx - yy) * sh[14] +
// // 					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
// // 			}
// // 		}
// // 	}
// // 	result += 0.5f;

// // 	// RGB colors are clamped to positive values. If values are
// // 	// clamped, we need to keep track of this for the backward pass.
// // 	clamped[3 * idx + 0] = (result.x < 0);
// // 	clamped[3 * idx + 1] = (result.y < 0);
// // 	clamped[3 * idx + 2] = (result.z < 0);
// // 	return glm::max(result, 0.0f);
// // }

// template<int C>
__global__ void preprocess_gaussians_kernel(
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
    const float* mirror_transformed_tan_theta, // tan_theta of mirror transformed PBF 
    const float* mirror_transformed_tan_phi, // tan_phi of mirror transformed PBF 
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

    // // Outputs (except xmap, ymap, h_opacity, prefiltered, and antialiasing)
    int* radii,
    int* pbf_id,
    float4* beap_xxyy,
    const float* xmap, // Set to nullptr for now until KB is reintegrated
    const float* ymap, // Set to nullptr for now until KB is reintegrated
    float3* points_xyz_view,
    float* depths,
    // // float* rgb,
    // // float2* h_opacity, // Input
    float3* w2o,
    // const dim3 grid,
    int* tiles_touched
    // bool prefiltered, // Flag
    // // bool antialiasing
) {
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
    
    // if (idx == 5) {
    //     printf("means3D: (%f, %f, %f)\n",
    //         means3D[3 * idx + 0],
    //         means3D[3 * idx + 1],
    //         means3D[3 * idx + 2]);

    //     glm::vec3 s = scales[idx];
    //     printf("scales: (%f, %f, %f)\n", s.x, s.y, s.z);

    //     glm::vec4 q = rotations[idx];
    //     printf("rotations: (%f, %f, %f, %f)\n",
    //         q.x, q.y, q.z, q.w);

    //     printf("opacity: %f\n", opacities[idx]);
    // }

	// // Initialize radius and touched tiles to 0. If this isn't changed,
	// // this Gaussian will not be processed further.
    radii[idx] = 0;
    pbf_id[idx * 4] = 0; 
    pbf_id[idx * 4 + 1] = 0;
    pbf_id[idx * 4 + 2] = 0; 
    pbf_id[idx * 4 + 3] = 0;

	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, means3D, viewmatrix, near_plane, far_plane,
    // prefiltered,
        p_view
    ))
		return;

	glm::mat3 R_view = computeRotationMatrix(rotations[idx], viewmatrix);
	float cutoff = 3.0f;

	if (opacities[idx] < 1.0f / 255.0f) return;
	// float3 p_view_identity = {means3D[3 * idx] + viewmatrix[12], means3D[3 * idx + 1] + viewmatrix[13], means3D[3 * idx + 2] + viewmatrix[14]};
	// if (!omni_hvar(scales[idx], scale_modifier, opacities[idx], h_opacity + idx, true)) return;

	// Prepare world-to-canonical transformation maxtrix for exact ray-Gaussian integral
	// see details in 3DGEER https://openreview.net/pdf?id=4voMNlRWI7, Eq.3
    // float3 w2o1 = toFloat3(R_view[0] / (sqrtf(sq(scales[idx].x)) * scale_modifier));
    // float3 w2o2 = toFloat3(R_view[1] / (sqrtf(sq(scales[idx].y)) * scale_modifier));
    // float3 w2o3 = toFloat3(R_view[2] / (sqrtf(sq(scales[idx].z)) * scale_modifier));
    // w2o[idx * 3 + 0] = w2o1;
	// w2o[idx * 3 + 1] = w2o2;
	// w2o[idx * 3 + 2] = w2o3;

	// w2o[idx * 3 + 0] = toFloat3(R_view[0] / (sqrtf(sq(scales[idx].x) + h_opacity[idx].x) * scale_modifier));
	// w2o[idx * 3 + 1] = toFloat3(R_view[1] / (sqrtf(sq(scales[idx].y) + h_opacity[idx].x) * scale_modifier));
	// w2o[idx * 3 + 2] = toFloat3(R_view[2] / (sqrtf(sq(scales[idx].z) + h_opacity[idx].x) * scale_modifier));

	points_xyz_view[idx] = p_view;

	// // Compute exact and tight Particle Bounding Frustum (PBF);
	// // see details in 3DGEER paper: https://openreview.net/pdf?id=4voMNlRWI7, Eq.10 (mathmatical proof in Sec.D.1)
	float4 tan_xxyy; // clamped tan value in x / y dir, i.e., tan_theta, tan_phi
    if (!computePBF(scales[idx], scale_modifier, R_view, p_view, cutoff, tan_xxyy, tan_fovx, tan_fovy, 0)) return;
	// if (!computePBF(scales[idx], scale_modifier, R_view, p_view, cutoff, tan_xxyy, tan_fovx, tan_fovy, h_opacity[idx].x)) return;
	if ((tan_xxyy.y - tan_xxyy.x) * (tan_xxyy.w - tan_xxyy.z) == 0)
		return;

    // _aa[0] = (int) (Ks[0] * tan_xxyy.x + K[2]);
    // _aa[1] = (int) (Ks[0] * tan_xxyy.y + K[2] + 1);
    // _bb[0] = (int) (Ks[4] * tan_xxyy.z + K[5]);
    // _bb[1] = (int) (Ks[4] * tan_xxyy.w + K[5] + 1);

	// if (xmap == nullptr)
	// {
	// 	// Convert PBF into BEAP space;
	// 	searchsorted_aabb(mirror_transformed_tan_theta, W, mirror_transformed_tan_phi, H, (float*)(&tan_xxyy), _aa, _bb);
	// } else {
    //     // TODO: Add KB map
	// 	// // Bound PBF into KB imaging space;
	// 	// const float4* kb_params4 = reinterpret_cast<const float4*>(kb_coeff);
	// 	// const float4 kb_params = kb_params4[0];
	// 	// invinterpolated_aabb(W, H, focal_x, focal_y, principal_x, principal_y, kb_params, tan_xxyy, _aa, _bb);
	// }
	// int4 _aabb = {_aa[0], _aa[1], _bb[0], _bb[1]};
    // float4 _aabb = {
    //     Ks[0] * tan_xxyy.x + Ks[2],
    //     Ks[0] * tan_xxyy.y + Ks[2],
    //     Ks[4] * tan_xxyy.z + Ks[5],
    //     Ks[4] * tan_xxyy.w + Ks[5]
    // };
	// if ((_aabb.y - _aabb.x) * (_aabb.w - _aabb.z) == 0)
	// 	return;

	int mode; // 0: BEAP, 1: Fisheye/KB, 2: Pinhole
	if (camera_model == CameraModelType::FISHEYE) mode = 1;
	else if (camera_model == CameraModelType::PINHOLE) mode = 2;
	else mode = 0;

	int _aa[2], _bb[2];

	if (mode == 0) { // BEAP
		// Convert PBF into BEAP space;
		searchsorted_pbf(mirror_transformed_tan_theta, W, mirror_transformed_tan_phi, H, (float*)(&tan_xxyy), _aa, _bb);
	} else if (mode == 1) { // Fisheye/KB
        const float4* kb_params4 = reinterpret_cast<const float4*>(radial_coeffs);
        const float4 kb_params = kb_params4[0];
        invinterpolated_aabb(W, H, Ks[0], Ks[4], Ks[2], Ks[5], kb_params, tan_xxyy, _aa, _bb);

		// if (idx == 560) {
		// 	printf("CUDA %d: W %d, H %d \nKs [%f, %f, %f, %f]\nKB [%f, %f, %f, %f]\ntan_xxyy [%f, %f, %f, %f]\naabb [%d, %d, %d, %d]\n",
		// 		idx, W, H, Ks[0], Ks[4], Ks[2], Ks[5],
		// 		kb_params.x, kb_params.y, kb_params.z, kb_params.w,
		// 		tan_xxyy.x, tan_xxyy.y, tan_xxyy.z, tan_xxyy.w,
		// 		_aa[0], _aa[1], _bb[0], _bb[1]
		// 	);
		// }
	} else if (mode == 2) { // Pinhole
		// searchsorted_pbf(mirror_transformed_tan_theta, W, mirror_transformed_tan_phi, H, (float*)(&tan_xxyy), _aa, _bb);
		float4 _aabb = {
            Ks[0] * tan_xxyy.x + Ks[2],
            Ks[0] * tan_xxyy.y + Ks[2],
            Ks[4] * tan_xxyy.z + Ks[5],
            Ks[4] * tan_xxyy.w + Ks[5]
        };

		_aa[0] = min(max(0, (int) (_aabb.x)), W);
		_aa[1] = min(max(0, (int) (_aabb.y + 1)), W);
		_bb[0] = min(max(0, (int) (_aabb.z)), H);
		_bb[1] = min(max(0, (int) (_aabb.w + 1)), H);
	} else {
		printf("Error: Mode does not exist.");
	}

	int4 _pbf = {_aa[0], _aa[1], _bb[0], _bb[1]};
	if ((_pbf.y - _pbf.x) * (_pbf.w - _pbf.z) == 0)
		return;

	// int4 my_aabb = {(int) (_aabb.x), (int) (_aabb.y + 1), (int) (_aabb.z), (int) (_aabb.w + 1)};
	// float2 point_image = { (my_aabb.y + my_aabb.x)/2.f, (my_aabb.w + my_aabb.z)/2.f };

	uint2 rect_min, rect_max;
	getRect2(_pbf, tile_size, tile_width, tile_height, rect_min, rect_max);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;
	int my_radius = max(rect_max.x - rect_min.x, rect_max.y - rect_min.y);
	if (my_radius <= radius_clip)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// if (colors_precomp == nullptr)
	// {
	// 	glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *cam_pos, shs, clamped);
	// 	rgb[idx * C + 0] = result.x;
	// 	rgb[idx * C + 1] = result.y;
	// 	rgb[idx * C + 2] = result.z;
	// }

	// Store some useful helper data for the next steps.
	depths[idx] = sqrtf((p_view.z * p_view.z) + (p_view.x * p_view.x) + (p_view.y * p_view.y));
	radii[idx] = my_radius;
	
	pbf_id[idx * 4] = _pbf.x;
	pbf_id[idx * 4 + 1] = _pbf.y;
	pbf_id[idx * 4 + 2] = _pbf.z;
	pbf_id[idx * 4 + 3] = _pbf.w;

	beap_xxyy[idx] = tan_xxyy;

    if (xmap == nullptr)
	{
		tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	} else {
		// tiles_touched[idx] = duplicateToTilesTouched(
		// 	p_view, w2o + 3*idx, h_opacity[idx].y,
		// 	_pbf, tan_xxyy, grid,
		// 	W, H,
		// 	0, 0, 0, nullptr, nullptr,
		// 	xmap,
		// 	ymap
		// );
	}
    if (idx < 10) { // (idx >= 39755 && idx < 39760) {
    //     // float3 w2o1 = toFloat3(R_view[0] / (sqrtf(sq(scales[idx].x)) * scale_modifier));
    //     // float3 w2o2 = toFloat3(R_view[1] / (sqrtf(sq(scales[idx].y)) * scale_modifier));
    //     // float3 w2o3 = toFloat3(R_view[2] / (sqrtf(sq(scales[idx].z)) * scale_modifier));
    //     printf(
    //         "CUDA %d: depth %f, radii %d, aabb %d %d %d %d, beap %f %f %f %f, touched %d, w2o [%f %f %f] [%f %f %f] [%f %f %f], mean [%f %f %f]\n",
    //         idx, sqrtf((p_view.z * p_view.z) + (p_view.x * p_view.x) + (p_view.y * p_view.y)),
    //         my_radius,
    //         my_aabb.x, my_aabb.y, my_aabb.z, my_aabb.w,
    //         tan_xxyy.x, tan_xxyy.y, tan_xxyy.z, tan_xxyy.w,
    //         (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x),
    //         w2o1.x, w2o1.y, w2o1.z, w2o2.x, w2o2.y, w2o2.z, w2o3.x, w2o3.y, w2o3.z,
    //         p_view.x, p_view.y, p_view.z
    //     );
    //     printf("sizeof(float4)=%u alignof(float4)=%u\n",
    //        (unsigned) sizeof(float4), (unsigned) alignof(float4));
    //     printf("sizeof(float3)=%u alignof(float3)=%u\n",
    //        (unsigned) sizeof(float3), (unsigned) alignof(float3));
            // printf(
            //     "CUDA %llu: viewmatrix [%f %f %f %f] [%f %f %f %f] [%f %f %f %f] [%f %f %f %f]",
            //     idx, viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
            //     viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
            //     viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
            //     viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]
            // );
            // printf(
            //     "CUDA %llu: K [%f %f %f] [%f %f %f] [%f %f %f]",
            //     idx, Ks[0], Ks[1], Ks[2], Ks[3], Ks[4], Ks[5], Ks[6], Ks[7], Ks[8]
            // );
    }
}

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
	// bool prefiltered,
	// // bool antialiasing
) {
    preprocess_gaussians_kernel << <(P + 255) / 256, 256 >> > ( // preprocess_gaussians_kernel<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
        // D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
        Ks,
		opacities,
		// shs,
		// clamped,
		// colors_precomp,
		viewmatrix, 
		mirror_transformed_tan_theta, mirror_transformed_tan_phi,
		// cam_pos,
		W, H,
		tan_fovx, tan_fovy,

        camera_model,
		// focal_x, focal_y,
		// principal_x, principal_y,
		radial_coeffs,
		near_plane,
		far_plane,
		radius_clip,

        tile_size, tile_width, tile_height,

		radii,
		aabb,
        beap_xxyy,
		xmap, ymap,
		means3D_view,
		depths,
		// // rgb,
        // // h_opacity,
		w2o,
		// grid,
		tiles_touched
		// prefiltered,
		// // antialiasing
    );
    cudaDeviceSynchronize();
}

// // Duplication methods

__global__ void duplicate_with_keys_kernel(
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
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if ((radii[idx] > 0) && (tiles_touched[idx] > 0))
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

        int64_t iid = 0; // idx / P; TODO: multiple images
        const int64_t iid_enc = iid << (32 + tile_n_bits);
        // tolerance for negative depth
        int32_t depth_i32 = *(int32_t *)&(depths[idx]);  // Bit-level reinterpret
        int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);  // Zero-extend to 64-bit

		// Update unsorted arrays with Gaussian idx for every tile that Gaussian touches

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is | camera ID | tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		if (xmap == nullptr) {
			uint2 rect_min, rect_max;
			getRect2(aabb[idx], tile_size, tile_width, tile_height, rect_min, rect_max);
	
			for (int32_t y = rect_min.y; y < rect_max.y; y++)
			{
				for (int32_t x = rect_min.x; x < rect_max.x; x++)
				{
                    int64_t tile_id = y * tile_width + x;
                    isect_ids[off] = iid_enc | (tile_id << 32) | depth_id_enc;
                    flatten_ids[off] = static_cast<int32_t>(idx);
					// uint64_t key = y * grid.x + x;
					// key <<= 32;
					// key |= *((uint32_t*)&depths[idx]);
					// gaussian_keys_unsorted[off] = key;
					// gaussian_values_unsorted[off] = idx;
					off++;
				}
			}
            // printf("AAAAAAAA: %d\n", (rect_max.y-1) * tile_width + (rect_max.x - 1));
		} else {
            // TODO
			// tiles_touched[idx] = duplicateToTilesTouched(
			// 	points_xyz[idx], w2o + 3 * idx, h_opacity[idx].y,
			// 	aabb[idx], beap_xxyy[idx], grid,
			// 	W, H,
			// 	idx, off, depths[idx],
			// 	gaussian_keys_unsorted,
			// 	gaussian_values_unsorted,
			// 	xmap, ymap);
		}
	}
}

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
) {
    duplicate_with_keys_kernel << <(P + 255) / 256, 256 >> > (
        P,
        points_xyz,
        w2o,
        // const float2* h_opacity,
        depths,
        offsets,

        // uint64_t* gaussian_keys_unsorted,
        // uint32_t* gaussian_values_unsorted,
        isect_ids,       // [n_isects]
        flatten_ids,      // [n_isects]

        radii,
        aabb,
        beap_xxyy,
        xmap,
        ymap,
        W, H,
        tiles_touched,
        tile_size, tile_width, tile_height, tile_n_bits
        // dim3 grid
    );
    cudaDeviceSynchronize();
}

// // Check keys to see if it is at the start/end of one tile's range in 
// // the full sorted list. If yes, write start/end of this tile. 
// // Run once per instanced (duplicated) Gaussian ID.
// __global__ void identify_tile_ranges_kernel(int L, int64_t* point_list_keys, int64_t* ranges)
// {
// 	auto idx = cg::this_grid().thread_rank();
// 	if (idx >= L)
// 		return;

// 	// Read tile ID from key. Update start/end of tile range if at limit.
// 	int64_t key = point_list_keys[idx];
// 	int64_t currtile = key >> 32;
// 	if (idx == 0)
// 		ranges[currtile*2] = 0;
// 	else
// 	{
// 		int64_t prevtile = point_list_keys[idx - 1] >> 32;
// 		if (currtile != prevtile)
// 		{
// 			ranges[prevtile*2+1] = idx;
// 			ranges[currtile*2] = idx;
// 		}
// 	}
// 	if (idx == L - 1) ranges[currtile*2+1] = L;
// }

// void identify_tile_ranges(int L, int64_t* point_list_keys, int64_t* ranges) {
//     identify_tile_ranges_kernel << <(L + 255) / 256, 256 >> > (
//         L,
//         point_list_keys,
//         ranges
//     );
//     cudaDeviceSynchronize();
// }

// __forceinline__ __device__ void searchsorted_intersect(
// 	const float* ref_start, int span,
// 	const float* values,
// 	int* indices
// ) {
// 	thrust::lower_bound(thrust::device, ref_start, ref_start + span, values, values + 2, indices);
// }

// __forceinline__ __device__ uint32_t duplicateToTilesTouched(
// 	const float3 points_xyz,
// 	const float3* w2o,
// 	const float opac,
// 	int4 aabb,
// 	float4 beap_xxyy,
// 	const dim3 grid,
// 	const int W, int H,
//     uint32_t idx, uint32_t off, float depth,
// 	uint64_t* gaussian_keys_unsorted,
// 	uint32_t* gaussian_values_unsorted,
// 	const float* xmap,
// 	const float* ymap
// )
// {
// 	uint2 rect_min, rect_max;

// 	getRect2(aabb, rect_min, rect_max, grid);

// 	int y_span = rect_max.y - rect_min.y;
// 	int x_span = rect_max.x - rect_min.x;

// 	// If no tiles are touched, return 0
// 	if (y_span * x_span == 0) {
// 		return 0;
// 	}

// 	bool isY = y_span > x_span;
// 	const uint2 rect_max_ = isY ? rect_max : make_uint2(rect_max.y, rect_max.x);
// 	const uint2 rect_min_ = isY ? rect_min : make_uint2(rect_min.y, rect_min.x);
// 	const int4 aabb_ = isY ? aabb : make_int4(aabb.z, aabb.w, aabb.x, aabb.y);
// 	const float2 beap_xxyy_ = isY ? make_float2(beap_xxyy.x, beap_xxyy.y) : make_float2(beap_xxyy.z, beap_xxyy.w);
// 	const float* cmap = isY ? xmap : ymap;
// 	const int W_ = isY ? W : H;
// 	const int H_ = isY ? H : W;

// 	uint32_t tiles_count = 0;
//     int2 slice_intersect_top, slice_intersect_bottom;
// 	int slice_lefttop, slice_leftbottom;

// 	// For each tile that the bounding rect overlaps, emit a 
// 	// key/value pair. The key is |  tile ID  |      depth      |,
// 	// and the value is the ID of the Gaussian. Sorting the values 
// 	// with this key yields Gaussian IDs in a list, such that they
// 	// are first sorted by tile and then by depth. 
// 	for (int y = rect_min_.y; y < rect_max_.y; y++)
// 	{
// 		// Get original BEAP ranged slice;
// 		slice_leftbottom = min(max(aabb_.z, y * BLOCK_Y), aabb_.w) * W_ + aabb_.x;
// 		searchsorted_intersect(cmap + slice_leftbottom, aabb_.y - aabb_.x + 1, (float*)(&beap_xxyy_), (int*)(&slice_intersect_bottom));

// 		slice_lefttop = min(max(aabb_.z, (y * BLOCK_Y + BLOCK_Y - 1)), aabb_.w) * W_ + aabb_.x;
// 		searchsorted_intersect(cmap + slice_lefttop, aabb_.y - aabb_.x + 1, (float*)(&beap_xxyy_), (int*)(&slice_intersect_top));

// 		// Cull out useless tiles;
// 		int tmp_left = min(max(0, min(slice_intersect_top.x, slice_intersect_bottom.x)), aabb_.y - aabb_.x);
// 		int tmp_right = min(max(0, max(slice_intersect_top.y, slice_intersect_bottom.y)), aabb_.y - aabb_.x);
// 		if (tmp_left >= tmp_right) {
// 			continue;
// 		}
// 		int min_tile_x = max(rect_min_.x,
//             min(rect_max_.x, (int)((aabb_.x + tmp_left) / BLOCK_X))
//         );
//         int max_tile_x = max(rect_min_.x,
//             min(rect_max_.x, (int)((aabb_.x + tmp_right + BLOCK_X - 1) / BLOCK_X))
//         );
// 		tiles_count += (max_tile_x - min_tile_x);
// 		for (int x = min_tile_x; x < max_tile_x; x++)
// 		{

// 			if (gaussian_keys_unsorted != nullptr) {
// 				uint64_t key = isY ? y * grid.x + x : x * grid.x + y;
// 				key <<= 32;
// 				key |= *((uint32_t*)&depth);
// 				gaussian_keys_unsorted[off] = key;
// 				gaussian_values_unsorted[off] = idx;
// 				off++;
// 			}
// 		}
// 	}
// 	return tiles_count;
// }

}
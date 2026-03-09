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

#include "forward.h"
#include "auxiliary.h"
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

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
		viewmatrix[0], viewmatrix[1], viewmatrix[2],
		viewmatrix[4], viewmatrix[5], viewmatrix[6],
		viewmatrix[8], viewmatrix[9], viewmatrix[10]);

	glm::mat3 R_view = W * R;
	return R_view;
}

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

__forceinline__ __device__ void searchsorted_pbf(
    const float* ref_u, int u_span,
    const float* ref_v, int v_span,
    const float* uv_values,
    int* u_indices, int* v_indices) {
    thrust::lower_bound(thrust::device, ref_u, ref_u + u_span, uv_values, uv_values + 2, u_indices);
    thrust::lower_bound(thrust::device, ref_v, ref_v + v_span, uv_values + 2, uv_values + 4, v_indices);
}

__forceinline__ __device__ void invinterpolated_pbf(
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

// Compute antialiasing variance (h_var) and pack into h_opacity:
//   h_opacity->x = h_var  (small isotropic variance added to each axis for anti-aliasing)
//   h_opacity->y = antialiasing-scaled Gaussian opacity
// Returns false if the resulting opacity falls below the visibility threshold.
__device__ bool omni_hvar(const glm::vec3 scale, const float mod, const float opacity, float2* h_opacity, bool antialiasing)
{
	float h_var = 1e-7f;
	
	float h_cov_scaling = 1.0f;
	if (antialiasing) {
		h_cov_scaling = (scale.x * mod) * (scale.y * mod) * (scale.z * mod) / sqrtf((sq(scale.x * mod) + h_var) * (sq(scale.y * mod) + h_var) * (sq(scale.z * mod) + h_var));
	}

	h_opacity->x = h_var;
	h_opacity->y = h_cov_scaling * opacity;

	if (h_opacity->y < 1.0f / 255.0f)
		return false;

	return true;
}

__device__ void mirror_transform_pbf(const float4& m, const float xi, float* result) {
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

__device__ void mirror_transform_fov(const float tan_fovx, const float tan_fovy, const float xi, float* result) {
	float _tan_fovx = xi * sqrtf(1 + tan_fovx * tan_fovx);
	float _tan_fovy = xi * sqrtf(1 + tan_fovy * tan_fovy);
	result[0] = tan_fovx / (1 + _tan_fovx);
	result[1] = -result[0];
	result[2] = tan_fovy / (1 + _tan_fovy);
	result[3] = -result[2];
}

__forceinline__ __device__ float mirror_transform_tan(const float m, const float z, const float xi) {
    if (xi == 0.0f) {
        return m;
    }
	return m / (1 + xi * (z / fabsf(z)) * sqrtf(1 + m * m));
}

// Compute the Particle Bounding Frustum (PBF) for a 3D Gaussian in ray-direction
// tangent space.  The PBF is a tight axis-aligned bounding box (AABB) on the set
// of rays that intersect the Gaussian's λ-sigma ellipsoid.  It is defined as
//   pbf = {tan_theta_min, tan_theta_max, tan_phi_min, tan_phi_max}
// and maps directly to pixel ranges via the BEAP reference arrays or the KB grid.
// See 3DGEER paper (https://openreview.net/pdf?id=4voMNlRWI7), Sec. D / Eq. 10.
// Returns false if the Gaussian is degenerate or entirely outside the frustum.
__device__ bool computePBF(
    const glm::vec3 scale, const float mod, const glm::mat3 R_view, const float3 p_view, const float lambda, float4& pbf, const float tan_fovx, const float tan_fovy, float h_var)
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

	bool neg = false;
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

    pbf.x = _left;
    pbf.y = _right;
	pbf.z = _bottom;
    pbf.w = _upper;

	// If half-extend is negative, return and do not compute the omni
	if (neg) return;

	// Omni mapping for AABB
	float xi = 1.0;
    float mirror_transformed_pbf[8];
	mirror_transform_pbf(pbf, xi, mirror_transformed_pbf);

    const float eps = 1e-6f;
    float depth = p_view.z;
    depth = (fabsf(depth) < eps) ? eps : depth; // Prevent division by zero
    float gaus_center_omni[2] = {
        mirror_transform_tan(p_view.x / depth, depth, xi),
        mirror_transform_tan(p_view.y / depth, depth, xi)
    };

    float fov_omni[4];
    mirror_transform_fov(tan_fovx, tan_fovy, xi, fov_omni);

    float aa_omni[4] = { mirror_transformed_pbf[0], mirror_transformed_pbf[1], mirror_transformed_pbf[2], mirror_transformed_pbf[3] };
	float bb_omni[4] = { mirror_transformed_pbf[4], mirror_transformed_pbf[5], mirror_transformed_pbf[6], mirror_transformed_pbf[7] };
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

    if (a_min_idx == 4) pbf.x = -tan_fovx;
    else if (a_min_idx == 5) pbf.x = tan_fovx;
    else if (a_min_idx == 0) pbf.x = _left;
    else if (a_min_idx == 1) pbf.x = _left;
    else if (a_min_idx == 2) pbf.x = _right;
    else if (a_min_idx == 3) pbf.x = _right;
    
    if (a_max_idx == 5) pbf.y = tan_fovx;
    else if (a_max_idx == 4) pbf.y = -tan_fovx;
    else if (a_max_idx == 0) pbf.y = _left;
    else if (a_max_idx == 1) pbf.y = _left;
    else if (a_max_idx == 2) pbf.y = _right;
    else if (a_max_idx == 3) pbf.y = _right;

    if (b_min_idx == 4) pbf.z = -tan_fovy;
    else if (b_min_idx == 5) pbf.z = tan_fovy;
    else if (b_min_idx == 0) pbf.z = _bottom;
    else if (b_min_idx == 1) pbf.z = _bottom;
    else if (b_min_idx == 2) pbf.z = _upper;
    else if (b_min_idx == 3) pbf.z = _upper;

    if (b_max_idx == 5) pbf.w = tan_fovy;
    else if (b_max_idx == 4) pbf.w = -tan_fovy;
    else if (b_max_idx == 0) pbf.w = _bottom;
    else if (b_max_idx == 1) pbf.w = _bottom;
    else if (b_max_idx == 2) pbf.w = _upper;
    else if (b_max_idx == 3) pbf.w = _upper;

    return true;
}

//// END OF THE IMPLEMENTATION OF THE (RAY-SPLATTING) FORWARD FUNCTION

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* mirror_transformed_tan_x, // tan_theta of mirror transformed PBF 
	const float* mirror_transformed_tan_y, // tan_phi of mirror transformed PBF 
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	const float principal_x, float principal_y,
	const float* distortion_coeffs, // KB fisheye distortion coefficients (k1,k2,k3,k4 in polynomial angle model)
	int* radii,
	int* pbf_id,
	float4* pbf_tan,
	const float* xmap, 
	const float* ymap, 
	float3* points_xyz_view,
	float* depths,
	float* rgb,
	float2* h_opacity,
	float3* w2o, 
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing,
	int mode)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	pbf_id[idx * 4] = 0; 
	pbf_id[idx * 4 + 1] = 0;
	pbf_id[idx * 4 + 2] = 0; 
	pbf_id[idx * 4 + 3] = 0; 

	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, prefiltered, p_view))
		return;

	glm::mat3 R_view = computeRotationMatrix(rotations[idx], viewmatrix);
	float cutoff = 3.0f;
	float3 p_view_identity = {orig_points[3 * idx] + viewmatrix[12], orig_points[3 * idx + 1] + viewmatrix[13], orig_points[3 * idx + 2] + viewmatrix[14]};
	if (!omni_hvar(scales[idx], scale_modifier, opacities[idx], h_opacity + idx, true)) return;

	// Prepare world-to-canonical transformation maxtrix for exact ray-Gaussian integral
	// see details in 3DGEER https://openreview.net/pdf?id=4voMNlRWI7, Eq.3 
	w2o[idx * 3 + 0] = toFloat3(R_view[0] / (sqrtf(sq(scales[idx].x) + h_opacity[idx].x) * scale_modifier));
	w2o[idx * 3 + 1] = toFloat3(R_view[1] / (sqrtf(sq(scales[idx].y) + h_opacity[idx].x) * scale_modifier));
	w2o[idx * 3 + 2] = toFloat3(R_view[2] / (sqrtf(sq(scales[idx].z) + h_opacity[idx].x) * scale_modifier));

	points_xyz_view[idx] = p_view;

	// Compute exact and tight Particle Bounding Frustum (PBF);
	// see details in 3DGEER paper: https://openreview.net/pdf?id=4voMNlRWI7, Eq.10 (mathmatical proof in Sec.D.1)
	float4 tan_xxyy; // clamped tan value in x / y dir, i.e., tan_theta, tan_phi
	if (!computePBF(scales[idx], scale_modifier, R_view, p_view, cutoff, tan_xxyy, tan_fovx, tan_fovy, h_opacity[idx].x)) return;
	if ((tan_xxyy.y - tan_xxyy.x) * (tan_xxyy.w - tan_xxyy.z) == 0)
		return;

	int _aa[2];
	int _bb[2];
	if (mode == 0)
	{
		// Convert PBF into BEAP space;
		searchsorted_pbf(mirror_transformed_tan_x, W, mirror_transformed_tan_y, H, (float*)(&tan_xxyy), _aa, _bb);
	} else if (mode == 1) {
		// Bound PBF into KB imaging space;
		const float4* kb_params4 = reinterpret_cast<const float4*>(distortion_coeffs);
		const float4 kb_params = kb_params4[0];
		invinterpolated_pbf(W, H, focal_x, focal_y, principal_x, principal_y, kb_params, tan_xxyy, _aa, _bb);
	} else {
		_aa[0] = (int)fmaxf(-(float)W, fminf(2.0f * W, floorf(focal_x * tan_xxyy.x + W / 2.0f - 1.0f)));
		_aa[1] = (int)fmaxf(-(float)W, fminf(2.0f * W, ceilf( focal_x * tan_xxyy.y + W / 2.0f + 1.0f)));
		_bb[0] = (int)fmaxf(-(float)H, fminf(2.0f * H, floorf(focal_y * tan_xxyy.z + H / 2.0f - 1.0f)));
		_bb[1] = (int)fmaxf(-(float)H, fminf(2.0f * H, ceilf( focal_y * tan_xxyy.w + H / 2.0f + 1.0f)));
	}
	int4 _pbf = {_aa[0], _aa[1], _bb[0], _bb[1]};
	if ((_pbf.y - _pbf.x) * (_pbf.w - _pbf.z) == 0)
		return;

	int4 screen_grid_aligned_pbf = _pbf;
	float2 point_image = { (screen_grid_aligned_pbf.y + screen_grid_aligned_pbf.x)/2.f, (screen_grid_aligned_pbf.w + screen_grid_aligned_pbf.z)/2.f };

	uint2 rect_min, rect_max;
	getRect2(screen_grid_aligned_pbf, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;
	int my_radius = max(rect_max.x - rect_min.x, rect_max.y - rect_min.y);

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = sqrtf((p_view.z * p_view.z) + (p_view.x * p_view.x) + (p_view.y * p_view.y));
	radii[idx] = my_radius;
	
	pbf_id[idx * 4] = screen_grid_aligned_pbf.x; 
	pbf_id[idx * 4 + 1] = screen_grid_aligned_pbf.y;
	pbf_id[idx * 4 + 2] = screen_grid_aligned_pbf.z;
	pbf_id[idx * 4 + 3] = screen_grid_aligned_pbf.w;

	pbf_tan[idx] = tan_xxyy;
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	if ((xmap != nullptr) && (mode == 1)) { // optimize tilestouched only for kb (mode == 1)
		tiles_touched[idx] = duplicateToTilesTouched(
			p_view, w2o + 3*idx, h_opacity[idx].y,
			screen_grid_aligned_pbf, tan_xxyy, grid,
			W, H,
			0, 0, 0, nullptr, nullptr,
			xmap,
			ymap
		);
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const int mode,
	const float focal_x, float focal_y,
	const float* tan_theta, 
	const float* tan_phi, 
	const float* raymap, 
	const float4* __restrict__ pbf_tan,
	const float3* __restrict__ points_xyz_view,
	const float* __restrict__ features,
	const float2* __restrict__ h_opacity,
	const float3* __restrict__ w2o_mat,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth
)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float3 rayf;
	// if (raymap == nullptr) {
	// 	rayf = make_float3((float)tan_theta[min(pix.x, W-1)], (float)tan_phi[min(pix.y, H-1)], 1.f);
	// } else {
	// 	rayf = make_float3((float)raymap[pix_id * 3], (float)raymap[pix_id * 3 + 1],(float)raymap[pix_id * 3 + 2]);
	// }
	if (mode == 0) {
		rayf = make_float3((float)tan_theta[min(pix.x, W-1)], (float)tan_phi[min(pix.y, H-1)], 1.f);
	} else if (mode == 1) {
		rayf = make_float3((float)raymap[pix_id * 3], (float)raymap[pix_id * 3 + 1],(float)raymap[pix_id * 3 + 2]);
	} else {
		rayf = { ((float)pix.x + 0.5f) / focal_x - W / (2.0f * focal_x), ((float)pix.y + 0.5f) / focal_y - H / (2.0f * focal_y), 1.0f };
	}

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;

	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xyz[BLOCK_SIZE];
	__shared__ float2 collected_h_opacity[BLOCK_SIZE];
	__shared__ float3 collected_w2o[BLOCK_SIZE * 3]; 
	__shared__ float4 collected_pbf_tan[BLOCK_SIZE * 4];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			int thread_idx = block.thread_rank();

			collected_id[thread_idx] = coll_id;
			collected_xyz[thread_idx] = points_xyz_view[coll_id];
			collected_pbf_tan[thread_idx] = pbf_tan[coll_id];

			for (int j = 0; j < 3; j++) {
				collected_w2o[thread_idx * 3 + j] = w2o_mat[coll_id * 3 + j];
			}
			collected_h_opacity[thread_idx] = h_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// if (rayf not in pbf_tan) continue;
			float4 b_xxyy = collected_pbf_tan[j];

			// acceleration in RenderCUDA; still has GPU bubble
			if (((rayf.x / rayf.z) < b_xxyy.x) || ((rayf.x / rayf.z) > b_xxyy.y))
				continue;
			if (((rayf.y / rayf.z) < b_xxyy.z) || ((rayf.y / rayf.z) > b_xxyy.w))
				continue;

			float3 xyz = collected_xyz[j];
			float2 h_o = collected_h_opacity[j];
			float3* w2o = collected_w2o + j * 3; 
			
			// see 3DGEER paper: https://openreview.net/pdf?id=4voMNlRWI7 (Eq. 5, mathmatical proof in Sec.B)
			float3 p_obj = { dot(xyz, w2o[0]), dot(xyz, w2o[1]), dot(xyz, w2o[2]) }; 
			float3 d_obj = { dot(rayf, w2o[0]), dot(rayf, w2o[1]), dot(rayf, w2o[2]) }; 
			float3 normal = cross(d_obj, p_obj); // check Sec.C.7
			float power_mah = -0.5f * dot(normal, normal) / dot(d_obj, d_obj);

			if (power_mah > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, h_o.y * exp(power_mah)); // For ray-splatting
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
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
	const float3* means3D_view, 
	const float* colors,
	const float2* h_opacity,
	const float3* w2o,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		mode,
		focal_x, focal_y,
		tan_theta, tan_phi,
		raymap,
		pbf_tan,
		means3D_view,
		colors,
		h_opacity,
		w2o,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* mirror_transformed_tan_x,
	const float* mirror_transformed_tan_y, 
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float principal_x, float principal_y,
	const float* distortion_coeffs,
	const float tan_fovx, float tan_fovy,
	int* radii,
	int* pbf,
	float4* pbf_tan,
	const float* xmap,
	const float* ymap,
	float3* means3D_view,
	float* depths,
	float* rgb,
	float2* h_opacity,
	float3* w2o, 
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing,
	int mode)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		colors_precomp,
		viewmatrix, 
		mirror_transformed_tan_x, mirror_transformed_tan_y,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		principal_x, principal_y,
		distortion_coeffs,
		radii,
		pbf, pbf_tan,
		xmap, ymap,
		means3D_view, 
		depths,
		rgb, h_opacity,
		w2o,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing,
		mode
		);
}

#include <array>
#include <cfloat>
#include <cmath>
#include <cooperative_groups.h>

#include "Common.h"
#include "Cameras.cuh"
#include "IntersectGEER.h"

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
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= near_plane || p_view.z >= far_plane)
	{
		return false;
	}
	return true;
}

__device__ glm::mat3 computeRotationMatrix(const glm::vec4 rot, const float* viewmatrix)
{
	// Quaternions are normalized by the caller.
	glm::vec4 q = rot;
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

	// Convert the row-major view matrix to GLM column-major storage.
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 R_view = W * R;
	return R_view;
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
	if (neg) return false;

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

// Conservative interval arithmetic for mapping a tangent-space PBF through a
// nonlinear camera model.  Tile association must over-estimate the projected
// footprint: an under-estimate drops a Gaussian before exact ray evaluation.
struct GEERInterval {
    float lo;
    float hi;
};

__forceinline__ __device__ GEERInterval geer_interval(float lo, float hi) {
    return {fminf(lo, hi), fmaxf(lo, hi)};
}

__forceinline__ __device__ GEERInterval geer_point(float value) {
    return {value, value};
}

__forceinline__ __device__ GEERInterval geer_add(GEERInterval a, GEERInterval b) {
    return {a.lo + b.lo, a.hi + b.hi};
}

__forceinline__ __device__ GEERInterval geer_mul(GEERInterval a, GEERInterval b) {
    float p0 = a.lo * b.lo;
    float p1 = a.lo * b.hi;
    float p2 = a.hi * b.lo;
    float p3 = a.hi * b.hi;
    return {
        fminf(fminf(p0, p1), fminf(p2, p3)),
        fmaxf(fmaxf(p0, p1), fmaxf(p2, p3))
    };
}

__forceinline__ __device__ GEERInterval geer_scale(GEERInterval a, float scale) {
    return geer_mul(a, geer_point(scale));
}

__forceinline__ __device__ GEERInterval geer_square(GEERInterval a) {
    float hi = fmaxf(a.lo * a.lo, a.hi * a.hi);
    float lo = a.lo <= 0.f && a.hi >= 0.f
        ? 0.f
        : fminf(a.lo * a.lo, a.hi * a.hi);
    return {lo, hi};
}

__forceinline__ __device__ GEERInterval geer_div(
    GEERInterval numerator, GEERInterval denominator, bool &valid
) {
    if (denominator.lo <= 0.f && denominator.hi >= 0.f) {
        valid = false;
        return {-INFINITY, INFINITY};
    }
    return geer_mul(
        numerator,
        geer_interval(1.f / denominator.lo, 1.f / denominator.hi)
    );
}

template <size_t N>
__forceinline__ __device__ GEERInterval geer_eval_poly_interval(
    const std::array<float, N> &coeffs, GEERInterval x
) {
    GEERInterval value = geer_point(0.f);
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        value = geer_add(geer_mul(value, x), geer_point(coeffs[i]));
    }
    return value;
}

__forceinline__ __device__ bool geer_store_pixel_aabb(
    const int W,
    const int H,
    GEERInterval u,
    GEERInterval v,
    int *u_indices,
    int *v_indices
) {
    if (!isfinite(u.lo) || !isfinite(u.hi) ||
        !isfinite(v.lo) || !isfinite(v.hi)) {
        u_indices[0] = 0;
        u_indices[1] = W;
        v_indices[0] = 0;
        v_indices[1] = H;
        return true;
    }

    // One-pixel padding absorbs floating-point and half-open-boundary effects.
    // Check the image limits before converting to int: high-order distortion
    // polynomials can produce finite floats outside the integer range.
    auto lower_index = [](float value, int limit) {
        if (value <= 1.f) return 0;
        if (value >= static_cast<float>(limit) + 1.f) return limit;
        return static_cast<int>(floorf(value)) - 1;
    };
    auto upper_index = [](float value, int limit) {
        if (value <= -1.f) return 0;
        if (value >= static_cast<float>(limit) - 1.f) return limit;
        return static_cast<int>(ceilf(value)) + 1;
    };
    u_indices[0] = lower_index(u.lo, W);
    u_indices[1] = upper_index(u.hi, W);
    v_indices[0] = lower_index(v.lo, H);
    v_indices[1] = upper_index(v.hi, H);
    return u_indices[0] < u_indices[1] && v_indices[0] < v_indices[1];
}

__forceinline__ __device__ bool opencv_pinhole_aabb(
    const int W,
    const int H,
    const float *K,
    const float *radial_coeffs,
    const float *tangential_coeffs,
    const float *thin_prism_coeffs,
    const float4 tan_xxyy,
    int *u_indices,
    int *v_indices
) {
    GEERInterval x = geer_interval(tan_xxyy.x, tan_xxyy.y);
    GEERInterval y = geer_interval(tan_xxyy.z, tan_xxyy.w);
    GEERInterval x2 = geer_square(x);
    GEERInterval y2 = geer_square(y);
    GEERInterval r2 = geer_add(x2, y2);

    float k[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float p[2] = {0.f, 0.f};
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    if (radial_coeffs != nullptr) {
#pragma unroll
        for (int i = 0; i < 6; ++i) k[i] = radial_coeffs[i];
    }
    if (tangential_coeffs != nullptr) {
        p[0] = tangential_coeffs[0];
        p[1] = tangential_coeffs[1];
    }
    if (thin_prism_coeffs != nullptr) {
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = thin_prism_coeffs[i];
    }

    GEERInterval numerator = geer_add(
        geer_point(1.f),
        geer_mul(r2, geer_add(geer_point(k[0]), geer_mul(
            r2, geer_add(geer_point(k[1]), geer_scale(r2, k[2]))
        )))
    );
    GEERInterval denominator = geer_add(
        geer_point(1.f),
        geer_mul(r2, geer_add(geer_point(k[3]), geer_mul(
            r2, geer_add(geer_point(k[4]), geer_scale(r2, k[5]))
        )))
    );
    bool valid = true;
    GEERInterval radial = geer_div(numerator, denominator, valid);
    if (!valid) {
        return geer_store_pixel_aabb(
            W, H, {-INFINITY, INFINITY}, {-INFINITY, INFINITY},
            u_indices, v_indices
        );
    }

    GEERInterval xy2 = geer_scale(geer_mul(x, y), 2.f);
    GEERInterval a2 = geer_add(r2, geer_scale(x2, 2.f));
    GEERInterval a3 = geer_add(r2, geer_scale(y2, 2.f));
    GEERInterval r4 = geer_square(r2);
    GEERInterval xd = geer_add(
        geer_mul(x, radial),
        geer_add(
            geer_add(geer_scale(xy2, p[0]), geer_scale(a2, p[1])),
            geer_add(geer_scale(r2, s[0]), geer_scale(r4, s[1]))
        )
    );
    GEERInterval yd = geer_add(
        geer_mul(y, radial),
        geer_add(
            geer_add(geer_scale(a3, p[0]), geer_scale(xy2, p[1])),
            geer_add(geer_scale(r2, s[2]), geer_scale(r4, s[3]))
        )
    );
    GEERInterval u = geer_add(geer_scale(xd, K[0]), geer_point(K[2]));
    GEERInterval v = geer_add(geer_scale(yd, K[4]), geer_point(K[5]));
    return geer_store_pixel_aabb(W, H, u, v, u_indices, v_indices);
}

// Validate approximate inverse-polynomial endpoints by enclosing the roots.
// Newton's absolute 1e-6 pixel update criterion is below FP32 resolution for
// ordinary image radii. A false convergence flag alone must not trigger a
// whole-image PBF. Keep that fallback when a finite, monotone bracket cannot
// be established.
static __forceinline__ __device__ bool geer_ftheta_inverse_bounds(
    const std::array<float, 6> &poly,
    float theta_lo, float theta_hi,
    float estimate_lo, float estimate_hi,
    float &radius_lo, float &radius_hi
) {
    if (!std::isfinite(theta_lo) || !std::isfinite(theta_hi) ||
        !std::isfinite(estimate_lo) || !std::isfinite(estimate_hi) ||
        theta_lo < 0.f || theta_hi < theta_lo ||
        estimate_lo < 0.f || estimate_hi < estimate_lo) return false;
    for (float coefficient : poly)
        if (!std::isfinite(coefficient)) return false;

    // Pad in radius space and propagate the padded interval through the
    // existing camera mapping. This is separate from the final pixel padding.
    const float padding = fmaxf(1e-3f, 8.f * FLT_EPSILON * estimate_hi);
    radius_lo = fmaxf(0.f, estimate_lo - padding);
    radius_hi = estimate_hi + padding;
    if (!std::isfinite(radius_hi)) return false;

    // Evaluate both residuals and a derivative interval in double precision:
    // FP32 cancellation is exactly the failure we are checking here.
    double value_lo = 0., value_hi = 0.;
    double deriv_lo = 0., deriv_hi = 0.;
    for (int i = 5; i >= 0; --i) {
        value_lo = value_lo * radius_lo + poly[i];
        value_hi = value_hi * radius_hi + poly[i];
        if (i > 0) {
            const double a = deriv_lo * radius_lo;
            const double b = deriv_lo * radius_hi;
            const double c = deriv_hi * radius_lo;
            const double d = deriv_hi * radius_hi;
            deriv_lo = fmin(fmin(a, b), fmin(c, d)) + i * double(poly[i]);
            deriv_hi = fmax(fmax(a, b), fmax(c, d)) + i * double(poly[i]);
        }
    }
    return std::isfinite(value_lo) && std::isfinite(value_hi) &&
           std::isfinite(deriv_lo) && std::isfinite(deriv_hi) &&
           deriv_lo > 0. && value_lo <= double(theta_lo) &&
           value_hi >= double(theta_hi);
}

__forceinline__ __device__ bool ftheta_delta_interval(
    GEERInterval theta,
    const FThetaCameraDistortionParameters &dist,
    GEERInterval &delta
) {
    if (dist.reference_poly ==
        FThetaCameraDistortionParameters::PolynomialType::ANGLE_TO_PIXELDIST) {
        delta = geer_eval_poly_interval(dist.angle_to_pixeldist_poly, theta);
    } else {
        std::array<float, 5> derivative = {
            dist.pixeldist_to_angle_poly[1],
            2.f * dist.pixeldist_to_angle_poly[2],
            3.f * dist.pixeldist_to_angle_poly[3],
            4.f * dist.pixeldist_to_angle_poly[4],
            5.f * dist.pixeldist_to_angle_poly[5]
        };
        auto inverse_at = [&](float angle, bool &converged) {
            converged = false;
            return eval_poly_inverse_horner_newton<3>(
                PolynomialProxy<PolynomialType::FULL, 6>{dist.pixeldist_to_angle_poly},
                PolynomialProxy<PolynomialType::FULL, 5>{derivative},
                PolynomialProxy<PolynomialType::FULL, 6>{dist.angle_to_pixeldist_poly},
                angle,
                converged
            );
        };
        bool lo_converged;
        bool hi_converged;
        float lo = inverse_at(theta.lo, lo_converged);
        float hi = inverse_at(theta.hi, hi_converged);
        // The convergence flag can be false for an accurate FP32 inverse.
        // Require a monotone root bracket instead, including its padding in
        // the projected bounds. Failed validation retains the full-image path.
        if (!geer_ftheta_inverse_bounds(
            dist.pixeldist_to_angle_poly, theta.lo, theta.hi,
            lo, hi, delta.lo, delta.hi
        )) return false;
    }
    if (!isfinite(delta.lo) || !isfinite(delta.hi) || delta.hi < 0.f)
        return false;
    delta.lo = fmaxf(0.f, delta.lo);
    return true;
}

__forceinline__ __device__ bool ftheta_aabb(
    const int W,
    const int H,
    const float *K,
    const FThetaCameraDistortionParameters &dist,
    const float4 tan_xxyy,
    int *u_indices,
    int *v_indices
) {
    GEERInterval x = geer_interval(tan_xxyy.x, tan_xxyy.y);
    GEERInterval y = geer_interval(tan_xxyy.z, tan_xxyy.w);
    GEERInterval r2 = geer_add(geer_square(x), geer_square(y));
    float r_min = sqrtf(fmaxf(0.f, r2.lo));
    float r_max = sqrtf(fmaxf(0.f, r2.hi));
    float theta_min = atanf(r_min);
    if (theta_min > dist.max_angle) return false;
    GEERInterval theta = {
        theta_min,
        fminf(atanf(r_max), dist.max_angle)
    };
    GEERInterval delta;
    if (!ftheta_delta_interval(theta, dist, delta)) {
        return geer_store_pixel_aabb(
            W, H, {-INFINITY, INFINITY}, {-INFINITY, INFINITY},
            u_indices, v_indices
        );
    }

    GEERInterval unit_x;
    GEERInterval unit_y;
    if (r_min <= 1e-7f) {
        unit_x = {-1.f, 1.f};
        unit_y = {-1.f, 1.f};
    } else {
        GEERInterval inv_r = {1.f / r_max, 1.f / r_min};
        unit_x = geer_mul(x, inv_r);
        unit_y = geer_mul(y, inv_r);
        unit_x = {fmaxf(-1.f, unit_x.lo), fminf(1.f, unit_x.hi)};
        unit_y = {fmaxf(-1.f, unit_y.lo), fminf(1.f, unit_y.hi)};
    }

    GEERInterval mapped_x = geer_mul(delta, unit_x);
    GEERInterval mapped_y = geer_mul(delta, unit_y);
    float c = dist.linear_cde[0];
    float d = dist.linear_cde[1];
    float e = dist.linear_cde[2];
    GEERInterval u = geer_add(
        geer_add(geer_scale(mapped_x, c), geer_scale(mapped_y, d)),
        geer_point(K[2] + .5f)
    );
    GEERInterval v = geer_add(
        geer_add(geer_scale(mapped_x, e), mapped_y),
        geer_point(K[5] + .5f)
    );
    return geer_store_pixel_aabb(W, H, u, v, u_indices, v_indices);
}

__global__ void preprocess_gaussians_kernel(
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
) {
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    radii[idx] = 0;
    pbf_id[idx * 4] = 0;
    pbf_id[idx * 4 + 1] = 0;
    pbf_id[idx * 4 + 2] = 0;
    pbf_id[idx * 4 + 3] = 0;

	tiles_touched[idx] = 0;

	float3 p_view;
	if (!in_frustum(idx, means3D, viewmatrix, near_plane, far_plane,
        p_view
    ))
		return;

	glm::mat3 R_view = computeRotationMatrix(rotations[idx], viewmatrix);
	float cutoff = 3.0f;

	if (opacities[idx] < 1.0f / 255.0f) return;

    // Compute the Particle Bounding Frustum, then map it to pixel bounds.
	float4 tan_xxyy; // clamped tan value in x / y dir, i.e., tan_theta, tan_phi
    if (!computePBF(scales[idx], scale_modifier, R_view, p_view, cutoff, tan_xxyy, tan_fovx, tan_fovy, 0)) return;
	if ((tan_xxyy.y - tan_xxyy.x) * (tan_xxyy.w - tan_xxyy.z) == 0)
		return;

    int _aa[2], _bb[2];
    if (camera_model == CameraModelType::FISHEYE) {
        const float4* kb_params4 = reinterpret_cast<const float4*>(radial_coeffs);
        const float4 kb_params = kb_params4[0];
        invinterpolated_aabb(W, H, Ks[0], Ks[4], Ks[2], Ks[5], kb_params, tan_xxyy, _aa, _bb);

	} else if (camera_model == CameraModelType::PINHOLE) {
		if (radial_coeffs == nullptr && tangential_coeffs == nullptr &&
			thin_prism_coeffs == nullptr) {
			_aa[0] = min(max(0, static_cast<int>(Ks[0] * tan_xxyy.x + Ks[2])), W);
			_aa[1] = min(max(0, static_cast<int>(Ks[0] * tan_xxyy.y + Ks[2] + 1)), W);
			_bb[0] = min(max(0, static_cast<int>(Ks[4] * tan_xxyy.z + Ks[5])), H);
			_bb[1] = min(max(0, static_cast<int>(Ks[4] * tan_xxyy.w + Ks[5] + 1)), H);
		} else if (!opencv_pinhole_aabb(
			W, H, Ks, radial_coeffs, tangential_coeffs,
			thin_prism_coeffs, tan_xxyy, _aa, _bb
		)) return;
	} else if (camera_model == CameraModelType::FTHETA) {
		if (!ftheta_aabb(W, H, Ks, ftheta_coeffs, tan_xxyy, _aa, _bb))
			return;
	} else {
		return; // Unsupported models are rejected by the host wrapper.
	}

	int4 _pbf = {_aa[0], _aa[1], _bb[0], _bb[1]};
	if ((_pbf.y - _pbf.x) * (_pbf.w - _pbf.z) == 0)
		return;

	uint2 rect_min, rect_max;
	getRect2(_pbf, tile_size, tile_width, tile_height, rect_min, rect_max);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;
	int my_radius = max(rect_max.x - rect_min.x, rect_max.y - rect_min.y);
	if (my_radius <= radius_clip)
		return;

	depths[idx] = sqrtf((p_view.z * p_view.z) + (p_view.x * p_view.x) + (p_view.y * p_view.y));
	radii[idx] = my_radius;

	pbf_id[idx * 4] = _pbf.x;
	pbf_id[idx * 4 + 1] = _pbf.y;
	pbf_id[idx * 4 + 2] = _pbf.z;
	pbf_id[idx * 4 + 3] = _pbf.w;

    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

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
) {
    preprocess_gaussians_kernel<<<(P + 255) / 256, 256>>>(
        P, means3D, scales, scale_modifier, rotations, Ks, opacities, viewmatrix,
        W, H, tan_fovx, tan_fovy, camera_model, radial_coeffs, tangential_coeffs,
        thin_prism_coeffs, ftheta_coeffs, near_plane, far_plane, radius_clip,
        tile_size, tile_width, tile_height, radii, pbf_id, depths, tiles_touched
    );
    cudaDeviceSynchronize();
}

__global__ void duplicate_with_keys_kernel(
    int P,
    const float* depths,
    const int64_t* offsets,
    int64_t* isect_ids,
    int32_t* flatten_ids,
    const int* radii,
    const int4* aabb,
    const int* tiles_touched,
    const int tile_size, const int tile_width, const int tile_height
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || radii[idx] <= 0 || tiles_touched[idx] <= 0) return;

    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    const int32_t depth_i32 = *(const int32_t*)&depths[idx];
    const int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);
    uint2 rect_min, rect_max;
    getRect2(aabb[idx], tile_size, tile_width, tile_height, rect_min, rect_max);

    // GEER currently handles one camera. Sort each tile's Gaussians by range.
    for (int32_t y = rect_min.y; y < rect_max.y; y++) {
        for (int32_t x = rect_min.x; x < rect_max.x; x++) {
            const int64_t tile_id = y * tile_width + x;
            isect_ids[off] = (tile_id << 32) | depth_id_enc;
            flatten_ids[off] = static_cast<int32_t>(idx);
            off++;
        }
    }
}

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
) {
    duplicate_with_keys_kernel<<<(P + 255) / 256, 256>>>(
        P, depths, offsets, isect_ids, flatten_ids, radii, aabb, tiles_touched,
        tile_size, tile_width, tile_height
    );
    cudaDeviceSynchronize();
}

} // namespace gsplat

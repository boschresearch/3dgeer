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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

//// IMPLEMENTATION OF THE 3DGEER BACKWARD FUNCTION
// Backward pass for the conversion of scale and rotation to the inversed Cov3Ds. 
__device__ void computeWorldToObject(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, glm::mat3 dL_dMt_inv, glm::vec3* dL_dscales, glm::vec4* dL_drots, float& dL_dhvar, float h_var) {
	// see details in 3DGEER paper (Eq.C.11-12)
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x; //r
	float x = q.y; //i
	float y = q.z; //j
	float z = q.w; //k

	// Compute inv_rotation matrix from quaternion (the transpose of the rotation matrix); column-major
	glm::mat3 R_inv = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y), // column 0
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x), // column 1
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)  // column 2
	);
	glm::mat3 Rt_inv = glm::transpose(R_inv); // to get the row vectors

	// Compute inverse scaling matrix
	glm::mat3 S_inv = glm::mat3(1.0f);
	float scaling_inv[3] = { sqrtf(1.f / (sq(scale.x * mod) + h_var)), sqrtf(1.f / (sq(scale.y * mod) + h_var)), sqrtf(1.f / (sq(scale.z * mod) + h_var)) };
	S_inv[0][0] *= scaling_inv[0];
	S_inv[1][1] *= scaling_inv[1];
	S_inv[2][2] *= scaling_inv[2];
	
	// Compute gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	float dL_dhscale_x = glm::dot(Rt_inv[0], dL_dMt_inv[0]) * (-S_inv[0][0] * S_inv[0][0]);
	dL_dscale->x = dL_dhscale_x * (S_inv[0][0] * (scale.x * mod));
	float dL_dhscale_y = glm::dot(Rt_inv[1], dL_dMt_inv[1]) * (-S_inv[1][1] * S_inv[1][1]);
	dL_dscale->y = dL_dhscale_y * (S_inv[1][1] * (scale.y * mod));
	float dL_dhscale_z = glm::dot(Rt_inv[2], dL_dMt_inv[2]) * (-S_inv[2][2] * S_inv[2][2]);
	dL_dscale->z = dL_dhscale_z * (S_inv[2][2] * (scale.z * mod));

	// Scale the loss gradient w.r.t. inv(RS)
	dL_dMt_inv[0] *= S_inv[0][0];
	dL_dMt_inv[1] *= S_inv[1][1];
	dL_dMt_inv[2] *= S_inv[2][2];
	
	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt_inv[0][1] - dL_dMt_inv[1][0]) + 2 * y * (dL_dMt_inv[2][0] - dL_dMt_inv[0][2]) + 2 * x * (dL_dMt_inv[1][2] - dL_dMt_inv[2][1]); //wrt r,r
	dL_dq.y = 2 * y * (dL_dMt_inv[1][0] + dL_dMt_inv[0][1]) + 2 * z * (dL_dMt_inv[2][0] + dL_dMt_inv[0][2]) + 2 * r * (dL_dMt_inv[1][2] - dL_dMt_inv[2][1]) - 4 * x * (dL_dMt_inv[2][2] + dL_dMt_inv[1][1]); //wrt i,x
	dL_dq.z = 2 * x * (dL_dMt_inv[1][0] + dL_dMt_inv[0][1]) + 2 * r * (dL_dMt_inv[2][0] - dL_dMt_inv[0][2]) + 2 * z * (dL_dMt_inv[1][2] + dL_dMt_inv[2][1]) - 4 * y * (dL_dMt_inv[2][2] + dL_dMt_inv[0][0]); //wrt j,y
	dL_dq.w = 2 * r * (dL_dMt_inv[0][1] - dL_dMt_inv[1][0]) + 2 * x * (dL_dMt_inv[2][0] + dL_dMt_inv[0][2]) + 2 * y * (dL_dMt_inv[1][2] + dL_dMt_inv[2][1]) - 4 * z * (dL_dMt_inv[1][1] + dL_dMt_inv[0][0]); //wrt k,z

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = reinterpret_cast<float4*>(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


template<int C>
__global__ void preprocessCUDA_mah(
	int P, int D, int M,
	const float3* means,
	const float3* means3D_view, 
	const float* depths,
	const int* radii,
	const float2* h_opacity,
	const float3* w2o,
	const float* viewmatrix, 
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const glm::vec3* campos,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	glm::vec3* dL_dmeans,
	glm::vec3* dL_dmean2D,
	float* dL_dcolor,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const glm::vec3* dL_dsigmaInv,
	float* dL_dopacity)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const glm::mat3 Wt = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	float3 p_view = means3D_view[idx];
	float h_var = h_opacity[idx].x;

	// Compute gradient updates due to computing covariance from scale/rotation
	float dL_dhvar = 0.0f;
	const glm::vec3* dL_dsigma_inv = dL_dsigmaInv + 3 * idx;
	// see 3DGEER paper: https://openreview.net/pdf?id=4voMNlRWI7 (Eq.9)
	glm::vec3 dL_dw2o[3];
	dL_dw2o[0] = glm::vec3( dL_dmeans[idx].x * p_view.x + dL_dsigma_inv[0].x, 
							dL_dmeans[idx].x * p_view.y + dL_dsigma_inv[0].y,
							dL_dmeans[idx].x * p_view.z + dL_dsigma_inv[0].z );

	dL_dw2o[1] = glm::vec3( dL_dmeans[idx].y * p_view.x + dL_dsigma_inv[1].x, 
							dL_dmeans[idx].y * p_view.y + dL_dsigma_inv[1].y,
							dL_dmeans[idx].y * p_view.z + dL_dsigma_inv[1].z );

	dL_dw2o[2] = glm::vec3( dL_dmeans[idx].z * p_view.x + dL_dsigma_inv[2].x, 
							dL_dmeans[idx].z * p_view.y + dL_dsigma_inv[2].y,
							dL_dmeans[idx].z * p_view.z + dL_dsigma_inv[2].z );

	glm::mat3 dL_dMt_inv = {
		Wt * dL_dw2o[0],
		Wt * dL_dw2o[1],
		Wt * dL_dw2o[2]
	};

	if (scales)
		// see 3DGEER paper: https://openreview.net/pdf?id=4voMNlRWI7 (Eq.10-12)
		computeWorldToObject(idx, scales[idx], scale_modifier, rotations[idx], dL_dMt_inv, dL_dscale, dL_drot, dL_dhvar, h_var);

	// Compute gradient updates to means in the view space
	glm::vec3 dL_dmean_view = glm::vec3( 
		dL_dmeans[idx].x * w2o[idx * 3].x + dL_dmeans[idx].y * w2o[idx * 3 + 1].x + dL_dmeans[idx].z * w2o[idx * 3 + 2].x,
		dL_dmeans[idx].x * w2o[idx * 3].y + dL_dmeans[idx].y * w2o[idx * 3 + 1].y + dL_dmeans[idx].z * w2o[idx * 3 + 2].y,
		dL_dmeans[idx].x * w2o[idx * 3].z + dL_dmeans[idx].y * w2o[idx * 3 + 1].z + dL_dmeans[idx].z * w2o[idx * 3 + 2].z );
	
	float h_cov_scaling = (scales[idx].x * scale_modifier) / sqrtf((sq(scales[idx].x * scale_modifier) + h_var)) * (scales[idx].y * scale_modifier) / sqrtf((sq(scales[idx].y * scale_modifier) + h_var)) * (scales[idx].z * scale_modifier) / sqrtf((sq(scales[idx].z * scale_modifier) + h_var));
	dL_dopacity[idx] *= fmaxf(h_cov_scaling, 5e-3f);

	// Compute gradient updates to means in the object space
	// see 3DGEER paper: https://openreview.net/pdf?id=4voMNlRWI7 (Eq.13)
	dL_dmeans[idx] = Wt * dL_dmean_view;

	// For densification
	dL_dmean2D[idx] = glm::vec3( 
		dL_dmean2D[idx].x * w2o[idx * 3].x + dL_dmean2D[idx].y * w2o[idx * 3 + 1].x + dL_dmean2D[idx].z * w2o[idx * 3 + 2].x,
		dL_dmean2D[idx].x * w2o[idx * 3].y + dL_dmean2D[idx].y * w2o[idx * 3 + 1].y + dL_dmean2D[idx].z * w2o[idx * 3 + 2].y,
		dL_dmean2D[idx].x * w2o[idx * 3].z + dL_dmean2D[idx].y * w2o[idx * 3 + 1].z + dL_dmean2D[idx].z * w2o[idx * 3 + 2].z ) * (sq(depths[idx]) + 1.f) * 3e-1f;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* tan_theta,
	const float* tan_phi,
	const float* __restrict__ bg_color,
	const float3* __restrict__ points_xyz_view,
	const float2* __restrict__ h_opacity,
	const float3* __restrict__ w2o, 
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_invdepths,
	float3* __restrict__ dL_dmean2D,
	glm::vec3* __restrict__ dL_dmeans,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dinvdepths,
	glm::vec3* __restrict__ dL_dsigmaInv
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	float3 rayf = {(float)tan_theta[min(pix.x, W-1)], (float)tan_phi[min(pix.y, H-1)], 1.f};

	const bool inside = pix.x < W&& pix.y < H;

	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xyz[BLOCK_SIZE];
	__shared__ float2 collected_h_opacity[BLOCK_SIZE];
	__shared__ float3 collected_w2o[BLOCK_SIZE * 3];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float dL_invdepth;
	float accum_invdepth_rec = 0;
	if (inside)
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		if(dL_invdepths)
		dL_invdepth = dL_invdepths[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_invdepth = 0;

	// // Gradient of pixel coordinate w.r.t. normalized 
	// // screen-space viewport corrdinates (-1 to 1)
	// const float ddelx_dx = 0.5 * W;
	// const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			int thread_idx = block.thread_rank();

			collected_id[thread_idx] = coll_id;
			collected_xyz[thread_idx] = points_xyz_view[coll_id];  

			collected_h_opacity[thread_idx] = h_opacity[coll_id];
			for (int j = 0; j < 3; j++) {
				collected_w2o[thread_idx * 3 + j] = w2o[coll_id * 3 + j];  
			}

			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + thread_idx] = colors[coll_id * C + i];

			if(dL_invdepths)
			collected_depths[thread_idx] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float3 xyz = collected_xyz[j];
			const float2 h_o = collected_h_opacity[j];
			float3* w2o_rows = collected_w2o + j * 3; 
			
			// see 3DGEER paper: https://openreview.net/pdf?id=4voMNlRWI7 (Eq. 5, Sec.C)
			// Transform Gaussian centre and ray direction into the canonical/object frame
			// using the packed world-to-object matrix rows w2o_rows[0..2].
			const float3 p_obj = { dot(xyz, w2o_rows[0]), dot(xyz, w2o_rows[1]), dot(xyz, w2o_rows[2]) };
			const float3 d_obj = { dot(rayf, w2o_rows[0]), dot(rayf, w2o_rows[1]), dot(rayf, w2o_rows[2]) };  
			const float3 normal = cross(d_obj, p_obj);
			const float dobj_norm_sq = dot(d_obj, d_obj);
			const float D2 = dot(normal, normal) / dobj_norm_sq;
			const float power_mah = -0.5f * D2;

			if (power_mah > 0.0f)
				continue;

			const float G = exp(power_mah);
			const float alpha = min(0.99f, h_o.y * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			// Propagate gradients from inverse depth to alphaas and
			// per Gaussian inverse depths
			if (dL_dinvdepths)
			{
				const float invd = 1.f / collected_depths[j];
				accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
				last_invdepth = invd;
				dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth;
				atomicAdd(&(dL_dinvdepths[global_id]), dchannel_dcolor * dL_invdepth);
			}

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = h_o.y * dL_dalpha;

			const float dL_dD2 = -0.5f * dL_dG * G;  
			const float3 dL_dnormal = { 2.f * dL_dD2 * normal.x / dobj_norm_sq, 2.f * dL_dD2 * normal.y / dobj_norm_sq, 2.f * dL_dD2 * normal.z / dobj_norm_sq };  
			const float dL_ddenom = dL_dG * G * (-power_mah / dobj_norm_sq);  
			float3 dL_dpobj;
			dL_dpobj.x = dL_dnormal.y * d_obj.z - dL_dnormal.z * d_obj.y;
			dL_dpobj.y = dL_dnormal.z * d_obj.x - dL_dnormal.x * d_obj.z;
			dL_dpobj.z = dL_dnormal.x * d_obj.y - dL_dnormal.y * d_obj.x;

			atomicAdd(&dL_dmeans[global_id].x, dL_dpobj.x);
			atomicAdd(&dL_dmeans[global_id].y, dL_dpobj.y);
			atomicAdd(&dL_dmeans[global_id].z, dL_dpobj.z);

			atomicAdd(&dL_dmean2D[global_id].x, fabsf(dL_dpobj.x));
			atomicAdd(&dL_dmean2D[global_id].y, fabsf(dL_dpobj.y));
			atomicAdd(&dL_dmean2D[global_id].z, fabsf(dL_dpobj.z));

			// see 3DGEER: https://openreview.net/pdf?id=4voMNlRWI7 (Eq.C.6-8)
			float dL_ddobj[3];
			dL_ddobj[0] = 2.f * dL_ddenom * d_obj.x - dL_dnormal.y * p_obj.z + dL_dnormal.z * p_obj.y;
			dL_ddobj[1] = 2.f * dL_ddenom * d_obj.y - dL_dnormal.z * p_obj.x + dL_dnormal.x * p_obj.z;
			dL_ddobj[2] = 2.f * dL_ddenom * d_obj.z - dL_dnormal.x * p_obj.y + dL_dnormal.y * p_obj.x;
			// Atomic addition component-wise
			#pragma unroll
			for (int i = 0; i < 3; i++) {
				atomicAdd(&dL_dsigmaInv[global_id * 3 + i].x, dL_ddobj[i] * rayf.x);
				atomicAdd(&dL_dsigmaInv[global_id * 3 + i].y, dL_ddobj[i] * rayf.y);
				atomicAdd(&dL_dsigmaInv[global_id * 3 + i].z, dL_ddobj[i] * rayf.z);
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
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
	const float* viewmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	float3* dL_dmean2D,
	const float* dL_dinvdepth,
	float* dL_dopacity,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const glm::vec3* dL_dsigmaInv,
	bool antialiasing)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	preprocessCUDA_mah<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		(float3*)means3D_view,  
		depths,
		radii,
		h_opacity,
		w2o,
		viewmatrix,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		campos,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		(glm::vec3*)dL_dmean3D,
		(glm::vec3*)dL_dmean2D,
		dL_dcolor,
		dL_dsh,
		dL_dscale,
		dL_drot,
		(glm::vec3*)dL_dsigmaInv,
		dL_dopacity);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
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
	glm::vec3* dL_dmean3D,  
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dinvdepths,
	glm::vec3* dL_dsigmaInv)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		tan_theta, tan_phi,  
		bg_color,
		means3D_view,  
		h_opacity,
		w2o,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_invdepths,
		dL_dmean2D,
		dL_dmean3D, 
		dL_dopacity,
		dL_dcolors,
		dL_dinvdepths,
		dL_dsigmaInv);
}

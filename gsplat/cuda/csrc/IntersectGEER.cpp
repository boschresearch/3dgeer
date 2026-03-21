#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"    // where all the macros are defined
#include "IntersectGEER.h" // where the launch function is declared
#include "Intersect.h"
#include "Ops.h"       // a collection of all gsplat operators

#include <cstdio>

namespace gsplat {

// TODO: Integrate camera parallelization
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> intersect_tile_geer(
    const int P, // N, num_gaussians

    const at::Tensor means,                // [N, 3]
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const float scale_modifier, // set to 1
    const at::Tensor opacities, // [..., N]
    const at::Tensor viewmats0, // [C, 4, 4]
    const CameraModelType camera_model,
    const at::Tensor Ks, // [C, 3, 3]
    const at::optional<at::Tensor> radial_coeffs, // [C, 4] or [C, 6]
    const float near_plane,
	const float far_plane,

    const at::Tensor ref_tan_x, // tan_theta of mirror transformed PBF
    const at::Tensor ref_tan_y, // tan_phi of mirror transformed PBF
    const int W,
    const int H,
    const float tan_fovx, float tan_fovy, // tan of fovx and fovy

    const int tile_size, const int tile_width, const int tile_height,
    // const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    // const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    // const at::Tensor depths,                     // [..., N] or [nnz]
    // const at::optional<at::Tensor> image_ids,    // [nnz]
    // const at::optional<at::Tensor> gaussian_ids, // [nnz]
    // const uint32_t I,
    // const uint32_t tile_size,
    // const uint32_t tile_width,
    // const uint32_t tile_height,
    const bool sort
    // const bool segmented
) {
    auto opt = means.options();

    at::Tensor default_radial_coeffs;

    if (radial_coeffs.has_value()) {
        auto coeffs = radial_coeffs.value();

        int expected = (camera_model == CameraModelType::PINHOLE) ? 6 : 4;

        TORCH_CHECK(
            coeffs.numel() == expected,
            "Expected ", expected, " radial coeffs but got ", coeffs.numel()
        );

        default_radial_coeffs = coeffs
            .to(opt.device())
            .to(at::kFloat)
            .view({-1})              // force 1D
            .contiguous();

    } else if (camera_model == CameraModelType::PINHOLE) {
        default_radial_coeffs = at::zeros({6}, opt.dtype(at::kFloat));

    } else if (camera_model == CameraModelType::FISHEYE) {
        default_radial_coeffs = at::zeros({4}, opt.dtype(at::kFloat));
    } else {
        TORCH_CHECK(false, "Camera model not supported yet");
    }

    // CUDA FN 1
    // for all gaussians computePBF --> AABB
    // for all gaussians convert AABBs into BEAP/KB space

    at::Tensor radii = at::empty({P}, opt.dtype(at::kInt));
    at::Tensor aabb_id = at::empty({P*4}, opt.dtype(at::kInt));
    at::Tensor beap_xxyy = at::empty({P*4}, opt.dtype(at::kFloat));
    at::Tensor means3D_view = at::empty({P*3}, opt.dtype(at::kFloat));
    at::Tensor depths = at::empty({P}, opt.dtype(at::kFloat));
    at::Tensor w2o = at::empty({P*9}, opt.dtype(at::kFloat));
    at::Tensor tiles_per_gauss = at::empty({P}, opt.dtype(at::kInt));
    
    preprocess_gaussians( // 3DGEER: FORWARD::preprocess
        P, // int P, // aka N
        // int D, int M,
        means.contiguous().data_ptr<float>(), // const float* means3D,
        (glm::vec3*) scales.contiguous().data_ptr<float>(), // const glm::vec3* scales,
        scale_modifier, // const float scale_modifier,
        (glm::vec4*) quats.contiguous().data_ptr<float>(), // const glm::vec4* rotations,
        Ks.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(), // const float* opacities,
        // const float* shs,
        // bool* clamped,
        // const float* colors_precomp,
        viewmats0.contiguous().data_ptr<float>(), // const float* viewmatrix,
        ref_tan_x.contiguous().data_ptr<float>(), // const float* ref_tan_x, // tan_theta of mirror transformed PBF 
        ref_tan_y.contiguous().data_ptr<float>(), // const float* ref_tan_y, // tan_phi of mirror transformed PBF 
        // const glm::vec3* cam_pos,
        W, H, // const int W, int H,
        tan_fovx, tan_fovy, // const float tan_fovx, float tan_fovy,
        // const float focal_x, float focal_y,
        // const float principal_x, float principal_y,
        camera_model, // const CameraModelType camera_model,
        default_radial_coeffs.contiguous().data_ptr<float>(), // const float* kb_coeff,
        near_plane, far_plane,
        tile_size, tile_width, tile_height,

        // Outputs (except xmap, ymap, h_opacity, prefiltered, and antialiasing)
        radii.contiguous().data_ptr<int>(), // int* radii,
        aabb_id.contiguous().data_ptr<int>(), // int* aabb_id,
        (float4*) beap_xxyy.contiguous().data_ptr<float>(), // float4* beap_xxyy,
        nullptr, // const float* xmap, // Set to nullptr for now until KB is reintegrated
        nullptr, // const float* ymap, // Set to nullptr for now until KB is reintegrated
        (float3*) means3D_view.contiguous().data_ptr<float>(), // float3* points_xyz_view,
        depths.contiguous().data_ptr<float>(), // float* depths,
        // // float* rgb,
        // // float2* h_opacity, // Input
        (float3*) w2o.contiguous().data_ptr<float>(), // float3* w2o,
        // tile_grid, // const dim3 grid,
        tiles_per_gauss.contiguous().data_ptr<int>() // uint32_t* tiles_touched
        // bool prefiltered, // Flag
        // // bool antialiasing
    );

    // auto radii_cpu = radii.cpu();
    // auto aabb_id_cpu = aabb_id.cpu();
    // auto beap_xxyy_cpu = beap_xxyy.cpu();
    // auto depths_cpu = depths.cpu();
    // auto tiles_touched_cpu = tiles_per_gauss.cpu();
    // auto w2o_cpu = w2o.cpu();
    // auto means3D_view_cpu = means3D_view.cpu();

    // auto radii_ptr = radii_cpu.data_ptr<int>();
    // auto aabb_id_ptr = aabb_id_cpu.data_ptr<int>();
    // auto beap_xxyy_ptr = (float4*) beap_xxyy_cpu.data_ptr<float>();
    // auto depths_ptr = depths_cpu.data_ptr<float>();
    // auto tiles_touched_ptr = tiles_touched_cpu.data_ptr<int>();
    // auto w2o_ptr = (float3*) w2o_cpu.data_ptr<float>();
    // auto means3D_view_ptr = (float3*) means3D_view_cpu.data_ptr<float>();


    // for (int idx=39755; idx<39760; idx++) {
    //     // if (depths_ptr[idx] != 0.0) {
    //     printf(
    //         "%d: depth %f, radii %d, aabb %d %d %d %d, beap %f %f %f %f, touched %d, w2o [%f %f %f] [%f %f %f] [%f %f %f], mean [%f %f %f]\n",
    //         idx, depths_ptr[idx], radii_ptr[idx],
    //         aabb_id_ptr[idx * 4], aabb_id_ptr[idx * 4 + 1], aabb_id_ptr[idx * 4 + 2], aabb_id_ptr[idx * 4 + 3],
    //         beap_xxyy_ptr[idx].x, beap_xxyy_ptr[idx].y, beap_xxyy_ptr[idx].z, beap_xxyy_ptr[idx].w, tiles_touched_ptr[idx],
    //         w2o_ptr[3*idx].x, w2o_ptr[3*idx].y, w2o_ptr[3*idx].z,
    //         w2o_ptr[3*idx+1].x, w2o_ptr[3*idx+1].y, w2o_ptr[3*idx+1].z,
    //         w2o_ptr[3*idx+2].x, w2o_ptr[3*idx+2].y, w2o_ptr[3*idx+2].z,
    //         means3D_view_ptr[idx].x, means3D_view_ptr[idx].y, means3D_view_ptr[idx].z
    //     );
    //     // }
    // }

    // auto test_cpu = tiles_touched.cpu();
    // uint32_t* test_ptr = test_cpu.data_ptr<uint32_t>();
    // printf(
    //     "test: (%d, %d, %d, %d)\n",
    //     test_ptr[980558],
    //     test_ptr[963868],
    //     test_ptr[981941],
    //     test_ptr[981331]
    // );

    // CUDA FN 2
    // duplicate to keys in gsplat format

    // auto tiles_flat = tiles_per_gauss.view({-1});
    // at::Tensor neg_mask = tiles_flat.lt(0);

    // if (neg_mask.any().item<bool>()) {
    //     printf("tiles_per_gauss contains negative values!\n");

    //     // get indices of negative entries
    //     auto neg_indices = neg_mask.nonzero().squeeze();

    //     if (neg_indices.dim() == 0) {
    //         neg_indices = neg_indices.unsqueeze(0);
    //     }

    //     // gather the negative values
    //     auto neg_values = tiles_flat.index_select(0, neg_indices);

    //     auto aabb_id_cpu = aabb_id.cpu();
    //     auto aabb_id_ptr = aabb_id_cpu.data_ptr<int>();

    //     auto beap_xxyy_cpu = beap_xxyy.cpu();
    //     auto beap_xxyy_ptr = (float4*) beap_xxyy_cpu.data_ptr<float>();


    //     int n_print = std::min<int>(neg_values.numel(), 10);
    //     printf("First %d negative entries:\n", n_print);
    //     for (int i = 0; i < n_print; ++i) {
    //         int idx = neg_indices[i].item<int>();
    //         int val = neg_values[i].item<int>();
    //         printf("idx %d: %d (%f, %f, %f, %f) -> (%d, %d, %d, %d) \n", idx, val,
    //         beap_xxyy_ptr[idx].x, beap_xxyy_ptr[idx].y, beap_xxyy_ptr[idx].z, beap_xxyy_ptr[idx].w,
    //         aabb_id_ptr[4*idx], aabb_id_ptr[4*idx+1], aabb_id_ptr[4*idx+2], aabb_id_ptr[4*idx+3]);
    //     }
    // } else {
    //     printf("tiles_per_gauss has no negative values.\n");
    // }



    at::Tensor cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0);
    int64_t n_isects = cum_tiles_per_gauss[cum_tiles_per_gauss.size(0) - 1].item<int64_t>();

    at::Tensor isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));

    uint32_t n_tiles = tile_width * tile_height;

    int I = 1; // TODO
    uint32_t image_n_bits = (uint32_t)floor(log2(I)) + 1;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    assert(image_n_bits + tile_n_bits <= 32);

    duplicate_with_keys(
		P,
		(float3*) means3D_view.contiguous().data_ptr<float>(), // geomState.means3D_view,
		(float3*) w2o.contiguous().data_ptr<float>(), // geomState.w2o,
		// geomState.h_opacity,
		depths.contiguous().data_ptr<float>(), // geomState.depths,
		cum_tiles_per_gauss.contiguous().data_ptr<int64_t>(), // geomState.point_offsets, // TODO

		// binningState.point_list_keys_unsorted,
		// binningState.point_list_unsorted,
        isect_ids.data_ptr<int64_t>(),
        flatten_ids.data_ptr<int32_t>(),

		radii.contiguous().data_ptr<int>(), // radii,
		(int4*) aabb_id.contiguous().data_ptr<int>(), // (int4*) geomState.aabb,
		(float4*) beap_xxyy.contiguous().data_ptr<float>(), // geomState.beap_xxyy,
		nullptr, nullptr, // xmap, ymap,
		W, H, // width, height,
		tiles_per_gauss.contiguous().data_ptr<int>(), // geomState.tiles_touched,
        tile_size, tile_width, tile_height, tile_n_bits
		// tile_grid
    );

    // CUDA FN 3
    // radix sort
    // optionally sort the Gaussians by isect_ids
    at::Tensor ranges = at::zeros({n_tiles*2}, opt.dtype(at::kLong));
    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        // if (segmented) {
        //     segmented_radix_sort_double_buffer(
        //         n_isects,
        //         I,
        //         image_n_bits,
        //         tile_n_bits,
        //         offsets,
        //         isect_ids,
        //         flatten_ids,
        //         isect_ids_sorted,
        //         flatten_ids_sorted
        //     );
        // } else {
        // printf("Sorting...");
        radix_sort_double_buffer(
            n_isects,
            image_n_bits,
            tile_n_bits,
            isect_ids,
            flatten_ids,
            isect_ids_sorted, 
            flatten_ids_sorted
        );
        // }

        if (n_isects > 0) {
            // printf("Getting ranges...");
            identify_tile_ranges(
                n_isects,
                isect_ids_sorted.contiguous().data_ptr<int64_t>(),
                ranges.contiguous().data_ptr<int64_t>()
            );
        }

        return std::make_tuple(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted, beap_xxyy);
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids, beap_xxyy);
    }
    // return std::make_tuple(
    //     at::Tensor(), at::Tensor(), at::Tensor(),
    //     at::Tensor()
    // );
}

}
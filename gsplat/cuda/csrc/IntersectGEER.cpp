#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <tuple>

#include "Common.h"
#include "IntersectGEER.h"
#include "Intersect.h"
#include "Ops.h"

namespace gsplat {

// TODO: Integrate camera parallelization.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
intersect_tile_geer(
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
    const at::optional<at::Tensor> tangential_coeffs, // [C, 2]
    const at::optional<at::Tensor> thin_prism_coeffs, // [C, 4]
    const FThetaCameraDistortionParameters ftheta_coeffs,
    const float near_plane,
	const float far_plane,
    const float radius_clip,

    const int W,
    const int H,
    const float tan_fovx, float tan_fovy, // tan of fovx and fovy

    const int tile_size, const int tile_width, const int tile_height,
    const bool sort
) {
    auto opt = means.options();

    at::Tensor normalized_radial_coeffs;
    at::Tensor normalized_tangential_coeffs;
    at::Tensor normalized_thin_prism_coeffs;

    if (radial_coeffs.has_value()) {
        auto coeffs = radial_coeffs.value();
        TORCH_CHECK(
            camera_model == CameraModelType::PINHOLE ||
                camera_model == CameraModelType::FISHEYE,
            "Radial coefficients are only valid for pinhole and fisheye cameras"
        );
        int expected = camera_model == CameraModelType::PINHOLE ? 6 : 4;

        TORCH_CHECK(
            coeffs.numel() == expected,
            "Expected ", expected, " radial coeffs but got ", coeffs.numel()
        );

        normalized_radial_coeffs = coeffs
            .to(opt.device())
            .to(at::kFloat)
            .view({-1})              // force 1D
            .contiguous();

    } else if (camera_model == CameraModelType::FISHEYE) {
        normalized_radial_coeffs = at::zeros({4}, opt.dtype(at::kFloat));
    }

    if (tangential_coeffs.has_value()) {
        TORCH_CHECK(
            camera_model == CameraModelType::PINHOLE,
            "Tangential coefficients are only valid for pinhole cameras"
        );
        TORCH_CHECK(
            tangential_coeffs.value().numel() == 2,
            "Expected 2 tangential coeffs but got ",
            tangential_coeffs.value().numel()
        );
        normalized_tangential_coeffs = tangential_coeffs.value()
            .to(opt.device()).to(at::kFloat).view({-1}).contiguous();
    }

    if (thin_prism_coeffs.has_value()) {
        TORCH_CHECK(
            camera_model == CameraModelType::PINHOLE,
            "Thin-prism coefficients are only valid for pinhole cameras"
        );
        TORCH_CHECK(
            thin_prism_coeffs.value().numel() == 4,
            "Expected 4 thin-prism coeffs but got ",
            thin_prism_coeffs.value().numel()
        );
        normalized_thin_prism_coeffs = thin_prism_coeffs.value()
            .to(opt.device()).to(at::kFloat).view({-1}).contiguous();
    }

    TORCH_CHECK(
        camera_model == CameraModelType::PINHOLE ||
            camera_model == CameraModelType::FISHEYE ||
            camera_model == CameraModelType::FTHETA,
        "Camera model not supported by GEER tile intersection"
    );
    if (camera_model == CameraModelType::FTHETA) {
        TORCH_CHECK(
            ftheta_coeffs.max_angle > 0.f,
            "F-theta GEER rendering requires ftheta_coeffs with max_angle > 0"
        );
    }

    // Compute each Gaussian's PBF, pixel bounds, and tile count.
    at::Tensor radii = at::empty({P}, opt.dtype(at::kInt));
    at::Tensor pbf_id = at::empty({P * 4}, opt.dtype(at::kInt));
    at::Tensor depths = at::empty({P}, opt.dtype(at::kFloat));
    at::Tensor tiles_per_gauss = at::empty({P}, opt.dtype(at::kInt));

    preprocess_gaussians(
        P,
        means.contiguous().data_ptr<float>(),
        (glm::vec3*) scales.contiguous().data_ptr<float>(),
        scale_modifier,
        (glm::vec4*) quats.contiguous().data_ptr<float>(),
        Ks.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        viewmats0.contiguous().data_ptr<float>(),
        W, H, tan_fovx, tan_fovy, camera_model,
        normalized_radial_coeffs.defined() ? normalized_radial_coeffs.data_ptr<float>() : nullptr,
        normalized_tangential_coeffs.defined() ? normalized_tangential_coeffs.data_ptr<float>() : nullptr,
        normalized_thin_prism_coeffs.defined() ? normalized_thin_prism_coeffs.data_ptr<float>() : nullptr,
        ftheta_coeffs,
        near_plane, far_plane, radius_clip,
        tile_size, tile_width, tile_height,
        radii.data_ptr<int>(),
        pbf_id.data_ptr<int>(),
        depths.data_ptr<float>(),
        tiles_per_gauss.data_ptr<int>()
    );

    at::Tensor cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}).to(at::kLong), 0);
    int64_t n_isects = cum_tiles_per_gauss[cum_tiles_per_gauss.size(0) - 1].item<int64_t>();
    at::Tensor isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));

    uint32_t n_tiles = tile_width * tile_height;
    uint32_t image_n_bits = 1; // One camera.
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    assert(image_n_bits + tile_n_bits <= 32);

    duplicate_with_keys(
        P,
        depths.data_ptr<float>(),
        cum_tiles_per_gauss.data_ptr<int64_t>(),
        isect_ids.data_ptr<int64_t>(),
        flatten_ids.data_ptr<int32_t>(),
        radii.data_ptr<int>(),
        (int4*) pbf_id.data_ptr<int>(),
        tiles_per_gauss.data_ptr<int>(),
        tile_size, tile_width, tile_height
    );

    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        radix_sort_double_buffer(
            n_isects, image_n_bits, tile_n_bits,
            isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted
        );
        isect_ids = isect_ids_sorted;
        flatten_ids = flatten_ids_sorted;
    }
    return std::make_tuple(
        tiles_per_gauss, isect_ids, flatten_ids, pbf_id.view({P, 4})
    );
}

} // namespace gsplat

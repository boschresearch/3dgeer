#include <ATen/Functions.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <cassert>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Cameras.cuh"
#include "Ops.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

__global__ void prepare_opencv_fisheye_max_angles_kernel(
    const uint32_t B,
    const uint32_t image_width,
    const uint32_t image_height,
    const float *__restrict__ Ks,           // [B, 3, 3]
    const float *__restrict__ radial_coeffs, // [B, 4]
    float *__restrict__ max_angles           // [B]
) {
    const uint32_t camera_index =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (camera_index >= B) {
        return;
    }

    OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
    cm_params.resolution = {image_width, image_height};
    cm_params.shutter_type = ShutterType::GLOBAL;
    cm_params.principal_point = {
        Ks[camera_index * 9 + 2], Ks[camera_index * 9 + 5]
    };
    cm_params.focal_length = {
        Ks[camera_index * 9 + 0], Ks[camera_index * 9 + 4]
    };
    if (radial_coeffs != nullptr) {
        cm_params.radial_coeffs =
            make_array<float, 4>(radial_coeffs + camera_index * 4);
    }

    // This is the only constructor invocation per camera that performs the
    // maximum-angle root solve.
    OpenCVFisheyeCameraModel camera_model(cm_params);
    max_angles[camera_index] = camera_model.max_angle;
}

at::Tensor prepare_opencv_fisheye_max_angles(
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const at::optional<at::Tensor> radial_coeffs
) {
    DEVICE_GUARD(Ks);
    CHECK_INPUT(Ks);
    if (radial_coeffs.has_value()) {
        CHECK_INPUT(radial_coeffs.value());
    }

    at::DimVector batch_dims(Ks.sizes().slice(0, Ks.dim() - 2));
    const uint32_t B = Ks.numel() / 9;
    at::Tensor max_angles = at::empty(batch_dims, Ks.options());
    if (B == 0) {
        return max_angles;
    }

    constexpr uint32_t block_size = 256;
    const uint32_t grid_size = (B + block_size - 1) / block_size;
    prepare_opencv_fisheye_max_angles_kernel<<<
        grid_size,
        block_size,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        B,
        image_width,
        image_height,
        Ks.data_ptr<float>(),
        radial_coeffs.has_value()
            ? radial_coeffs.value().data_ptr<float>()
            : nullptr,
        max_angles.data_ptr<float>()
    );
    return max_angles;
}

__device__ inline vec3 compute_ray_direction_from_camera_model(
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t camera_index,
    const float px,
    const float py,
    const CameraModelType camera_model_type,
    const float *__restrict__ Ks,
    const float *__restrict__ radial_coeffs,
    const float *__restrict__ tangential_coeffs,
    const float *__restrict__ thin_prism_coeffs,
    const float *__restrict__ fisheye_max_angles,
    const FThetaCameraDistortionParameters ftheta_coeffs
) {
    const vec2 focal_length = {
        Ks[camera_index * 9 + 0],
        Ks[camera_index * 9 + 4]
    };
    const vec2 principal_point = {
        Ks[camera_index * 9 + 2],
        Ks[camera_index * 9 + 5]
    };

    vec3 ray_dir = {0.f, 0.f, 0.f};

    if (camera_model_type == CameraModelType::ORTHO) {
        ray_dir = {0.f, 0.f, 1.f};
        return ray_dir;
    }

    if (camera_model_type == CameraModelType::PINHOLE) {
        if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
            PerfectPinholeCameraModel::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = ShutterType::GLOBAL;
            cm_params.principal_point = {principal_point.x, principal_point.y};
            cm_params.focal_length = {focal_length.x, focal_length.y};
            PerfectPinholeCameraModel camera_model(cm_params);
            auto const camera_ray = camera_model.image_point_to_camera_ray(vec2(px, py));
            if (camera_ray.valid_flag) {
                ray_dir = camera_ray.ray_dir;
            }
        } else {
            OpenCVPinholeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = ShutterType::GLOBAL;
            cm_params.principal_point = {principal_point.x, principal_point.y};
            cm_params.focal_length = {focal_length.x, focal_length.y};
            if (radial_coeffs != nullptr) {
                cm_params.radial_coeffs = make_array<float, 6>(radial_coeffs + camera_index * 6);
            }
            if (tangential_coeffs != nullptr) {
                cm_params.tangential_coeffs = make_array<float, 2>(tangential_coeffs + camera_index * 2);
            }
            if (thin_prism_coeffs != nullptr) {
                cm_params.thin_prism_coeffs = make_array<float, 4>(thin_prism_coeffs + camera_index * 4);
            }
            OpenCVPinholeCameraModel camera_model(cm_params);
            auto const camera_ray = camera_model.image_point_to_camera_ray(vec2(px, py));
            if (camera_ray.valid_flag) {
                ray_dir = camera_ray.ray_dir;
            }
        }
        return ray_dir;
    }

    if (camera_model_type == CameraModelType::FISHEYE) {
        OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = ShutterType::GLOBAL;
        cm_params.principal_point = {principal_point.x, principal_point.y};
        cm_params.focal_length = {focal_length.x, focal_length.y};
        if (radial_coeffs != nullptr) {
            cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + camera_index * 4);
        }
        assert(fisheye_max_angles != nullptr);
        OpenCVFisheyeCameraModel camera_model(
            cm_params, 1e-6f, fisheye_max_angles[camera_index]
        );
        auto const camera_ray = camera_model.image_point_to_camera_ray(vec2(px, py));
        if (camera_ray.valid_flag) {
            ray_dir = camera_ray.ray_dir;
        }
        return ray_dir;
    }

    if (camera_model_type == CameraModelType::FTHETA) {
        FThetaCameraModel<>::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = ShutterType::GLOBAL;
        cm_params.principal_point = {principal_point.x, principal_point.y};
        cm_params.dist = ftheta_coeffs;
        FThetaCameraModel camera_model(cm_params);
        auto const camera_ray = camera_model.image_point_to_camera_ray(vec2(px, py));
        if (camera_ray.valid_flag) {
            ray_dir = camera_ray.ray_dir;
        }
        return ray_dir;
    }

    assert(false);
    return ray_dir;
}

__global__ void compute_raymap_kernel(
    const uint32_t B,
    const uint32_t image_width,
    const uint32_t image_height,
    const CameraModelType camera_model_type,
    const float *__restrict__ Ks,                // [B, 3, 3]
    const float *__restrict__ radial_coeffs,      // [B, 6] or [B, 4]
    const float *__restrict__ tangential_coeffs,  // [B, 2]
    const float *__restrict__ thin_prism_coeffs,  // [B, 4]
    const float *__restrict__ fisheye_max_angles, // [B]
    const FThetaCameraDistortionParameters ftheta_coeffs,
    float *__restrict__ raymap                    // [B, image_height, image_width, 3]
) {
    const uint32_t camera_index = blockIdx.x;
    const uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t j = blockIdx.z * blockDim.x + threadIdx.x;

    if (camera_index >= B || i >= image_height || j >= image_width) {
        return;
    }

    const float px = static_cast<float>(j) + 0.5f;
    const float py = static_cast<float>(i) + 0.5f;

    const vec3 ray_dir = compute_ray_direction_from_camera_model(
        image_width,
        image_height,
        camera_index,
        px,
        py,
        camera_model_type,
        Ks,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        fisheye_max_angles,
        ftheta_coeffs
    );

    const uint32_t pixel_offset =
        camera_index * image_height * image_width * 3 +
        i * image_width * 3 +
        j * 3;
    raymap[pixel_offset + 0] = ray_dir.x;
    raymap[pixel_offset + 1] = ray_dir.y;
    raymap[pixel_offset + 2] = ray_dir.z;
}

at::Tensor compute_raymap(
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const CameraModelType camera_model,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    const FThetaCameraDistortionParameters ftheta_coeffs
) {
    DEVICE_GUARD(Ks);
    CHECK_INPUT(Ks);
    if (radial_coeffs.has_value()) {
        CHECK_INPUT(radial_coeffs.value());
    }
    if (tangential_coeffs.has_value()) {
        CHECK_INPUT(tangential_coeffs.value());
    }
    if (thin_prism_coeffs.has_value()) {
        CHECK_INPUT(thin_prism_coeffs.value());
    }

    auto opt = Ks.options();
    at::DimVector batch_dims(Ks.sizes().slice(0, Ks.dim() - 2));
    uint32_t B = Ks.numel() / 9;

    at::DimVector raymap_shape(batch_dims);
    raymap_shape.append({image_height, image_width, 3});
    at::Tensor raymaps = at::empty(raymap_shape, opt);
    at::Tensor fisheye_max_angles;
    if (camera_model == CameraModelType::FISHEYE) {
        fisheye_max_angles = prepare_opencv_fisheye_max_angles(
            Ks, image_width, image_height, radial_coeffs
        );
    }

    constexpr uint32_t tile_size = 16;
    const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;
    const uint32_t tile_width = (image_width + tile_size - 1) / tile_size;

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {B, tile_height, tile_width};

    compute_raymap_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        B,
        image_width,
        image_height,
        camera_model,
        Ks.data_ptr<float>(),
        radial_coeffs.has_value() ? radial_coeffs.value().data_ptr<float>() : nullptr,
        tangential_coeffs.has_value() ? tangential_coeffs.value().data_ptr<float>() : nullptr,
        thin_prism_coeffs.has_value() ? thin_prism_coeffs.value().data_ptr<float>() : nullptr,
        fisheye_max_angles.defined()
            ? fisheye_max_angles.data_ptr<float>()
            : nullptr,
        ftheta_coeffs,
        raymaps.data_ptr<float>()
    );

    return raymaps;
}

} // namespace gsplat

import os
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# ---------------- Polynomial Evaluation ----------------
def _eval_poly_horner(poly_coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluates a polynomial y=f(x) using Horner's scheme"""
    y = torch.zeros_like(x)
    for fi in torch.flip(poly_coefficients, dims=(0,)):
        y = y * x + fi
    return y

def _eval_poly_inverse_horner_newton(
    poly_coefficients: torch.Tensor,
    poly_derivative_coefficients: torch.Tensor,
    inverse_poly_approximation_coefficients: torch.Tensor,
    newton_iterations: int,
    y: torch.Tensor,
) -> torch.Tensor:
    """Evaluates inverse x=f^{-1}(y) using Horner + Newton iterations"""
    x = _eval_poly_horner(inverse_poly_approximation_coefficients, y)
    assert newton_iterations >= 0
    x_iter = [torch.zeros_like(x) for _ in range(newton_iterations + 1)]
    x_iter[0] = x
    for i in range(newton_iterations):
        dfdx = _eval_poly_horner(poly_derivative_coefficients, x_iter[i])
        residuals = _eval_poly_horner(poly_coefficients, x_iter[i]) - y
        x_iter[i + 1] = x_iter[i] - residuals / dfdx
    return x_iter[newton_iterations]

# ---------------- Camera Rays ----------------
def image_points_to_camera_rays(
    camera_model_parameters,
    image_points,
    newton_iterations: int = 3,
    min_2d_norm: float = 1e-6,
    device: str = "cpu",
):
    dtype: torch.dtype = torch.float32

    principal_point = torch.tensor(camera_model_parameters.principal_point, dtype=dtype, device=device)
    focal_length = torch.tensor(camera_model_parameters.focal_length, dtype=dtype, device=device)
    resolution = torch.tensor(camera_model_parameters.resolution.astype(np.int32), device=device)
    max_angle = float(camera_model_parameters.max_angle)

    min_2d_norm = torch.tensor(min_2d_norm, dtype=dtype, device=device)

    # Radial polynomial
    k1, k2, k3, k4 = camera_model_parameters.radial_coeffs
    forward_poly = torch.tensor([0, 1, 0, k1, 0, k2, 0, k3, 0, k4], dtype=dtype, device=device)
    dforward_poly = torch.tensor([1, 0, 3*k1, 0, 5*k2, 0, 7*k3, 0, 9*k4], dtype=dtype, device=device)

    # Approx backward polynomial (linear approx)
    max_normalized_dist = np.max(camera_model_parameters.resolution / 2 / camera_model_parameters.focal_length)
    approx_backward_poly = torch.tensor([0, max_angle / max_normalized_dist], dtype=dtype, device=device)

    image_points = image_points.to(dtype)
    normalized_image_points = (image_points - principal_point) / focal_length
    deltas = torch.linalg.norm(normalized_image_points, axis=1, keepdims=True)

    thetas = _eval_poly_inverse_horner_newton(
        forward_poly, dforward_poly, approx_backward_poly, newton_iterations, deltas
    )

    cam_rays = torch.cat([
        torch.sin(thetas) * normalized_image_points / torch.clamp(deltas, min=min_2d_norm),
        torch.cos(thetas)
    ], dim=1)
    mask = deltas.flatten() < min_2d_norm
    cam_rays[mask, :] = normalized_image_points.new_tensor([0, 0, 1])
    return cam_rays

# ---------------- Error Map ----------------
def compute_error_map(rays, cam_model, grid, H, W, device="cpu"):
    fx, fy = cam_model.focal_length
    cx, cy = cam_model.principal_point
    k1, k2, k3, k4 = cam_model.radial_coeffs

    xys = rays[:, :2] / rays[:, 2:3]
    norm = torch.norm(xys, dim=1)
    theta = torch.atan(norm)
    theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    x_proj = theta_d * xys[:, 0] / torch.clamp(norm, min=1e-9)
    y_proj = theta_d * xys[:, 1] / torch.clamp(norm, min=1e-9)

    x_img = x_proj * fx + cx
    y_img = y_proj * fy + cy

    u = grid[0, :].to(device)
    v = grid[1, :].to(device)
    error = (x_img - u) ** 2 + (y_img - v) ** 2
    error_map = error.reshape(H, W).cpu().numpy()
    error_map = np.clip(error_map, 0, 30)
    return error_map


def compute_max_distance_to_border(image_size_component: float, principal_point_component: float) -> float:
    """Given an image size component (x or y) and corresponding principal point component (x or y),
    returns the maximum distance (in image domain units) from the principal point to either image boundary."""
    center = 0.5 * image_size_component
    if principal_point_component > center:
        return principal_point_component
    else:
        return image_size_component - principal_point_component


def compute_max_radius(image_size: np.ndarray, principal_point: np.ndarray) -> float:
    """Compute the maximum radius from the principal point to the image boundaries."""
    max_diag = np.array(
        [
            compute_max_distance_to_border(image_size[0], principal_point[0]),
            compute_max_distance_to_border(image_size[1], principal_point[1]),
        ]
    )
    return np.linalg.norm(max_diag).item()

# ---------------- Camera Model Wrapper ----------------
class CameraModel:
    def __init__(self, focal_length, principal_point, resolution, radial_coeffs, max_angle):
        self.focal_length = np.array(focal_length, dtype=np.float32)
        self.principal_point = np.array(principal_point, dtype=np.float32)
        self.resolution = np.array(resolution, dtype=np.float32)
        self.radial_coeffs = radial_coeffs
        self.max_angle = max_angle

def run(args):
    scannetpp_data_path = args.path
    target_scene_names = args.scenes.split(",")

    # H = int(1168)
    # W = int(1752)
    # u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    # grid = torch.from_numpy(np.stack([u_grid, v_grid], axis=0)).float()  # [2,H,W]
    # grid_flat = grid.reshape(2, -1)  # [2,N]
    # points = grid_flat.T  # [N,2]
    # image_points = points.float()

    for scene_name in target_scene_names:
        scene_dir = os.path.join(scannetpp_data_path, scene_name)

        print(f"Processing {scene_name}")
        # scene_transform_file = os.path.join(scene_dir, 'dslr/nerfstudio/transforms.json')
        scene_transform_file = os.path.join(scene_dir, 'nerfstudio/transforms.json')
        scene_info = json.load(open(scene_transform_file))

        # Build camera model
        focal_length = (scene_info['fl_x'] * args.focal_scaling, scene_info['fl_y'] * args.focal_scaling)
        principal_point = (scene_info['cx'], scene_info['cy'])
        W = scene_info['w']
        H = scene_info['h']
        resolution = (W, H)
        radial_coeffs = (scene_info['k1'] * args.distortion_scaling, scene_info['k2'] * args.distortion_scaling, \
                         scene_info['k3'] * args.distortion_scaling, scene_info['k4'] * args.distortion_scaling)
        
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        grid = torch.from_numpy(np.stack([u_grid, v_grid], axis=0)).float()  # [2,H,W]
        grid_flat = grid.reshape(2, -1)  # [2,N]
        points = grid_flat.T  # [N,2]
        image_points = points.float()

        max_radius_pixels = compute_max_radius(np.array(resolution).astype(np.float64), np.array(principal_point))
        fov_angle_x = 2.0 * max_radius_pixels / focal_length[0]
        fov_angle_y = 2.0 * max_radius_pixels / focal_length[1]
        max_angle = np.max([fov_angle_x, fov_angle_y]) / 2.0
        
        cam_model = CameraModel(focal_length, principal_point, resolution, radial_coeffs, max_angle)

        # Compute camera rays
        rays = image_points_to_camera_rays(cam_model, image_points, newton_iterations=3, device="cpu")

        # ---- Ray Map Visualization ----
        # output_dir = os.path.join(scene_dir, 'dslr')
        output_dir = scene_dir
        os.makedirs(output_dir, exist_ok=True)

        # ---- Error Map Visualization ----
        error_map = compute_error_map(rays, cam_model, grid_flat, H, W, device="cpu")
        plt.imshow(error_map, cmap="hot")
        plt.colorbar()
        plt.title(f"Error Map: {scene_name}")
        plt.savefig(os.path.join(output_dir, 'error_map.png'), dpi=300)
        plt.close()
        print(f"Saved error_map to {os.path.join(output_dir, 'error_map.png')}")
        print(f'max error: {error_map.max()}')

        print("saving grid")
        # np.save(os.path.join(scene_dir, 'dslr/raymap_fisheye.npy'), rays.detach().cpu().numpy().reshape(H, W, 3))
        np.save(os.path.join(scene_dir, 'raymap_fisheye.npy'), rays.detach().cpu().numpy().reshape(H, W, 3))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default="/media/scannetpp/demo/")
    parser.add_argument('--scenes', type=str, default="2a1a3afad9,4ef75031e3,1f7cbbdde1,4ef75031e3,1d003b07bd,0a5c013435")
    parser.add_argument('--focal_scaling', type=float, default=1.0)
    parser.add_argument('--distortion_scaling', type=float, default=1.0)
    parser.add_argument('--mirror_shift', type=float, default=0.0)
    args = parser.parse_args()
    run(args)
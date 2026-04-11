#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.general_utils import PILtoTorch
import cv2
import math


def _tanfov_from_raymap(raymap, min_rz: float = 1e-3, max_tan: float = 1e4):
    """Derive (tanfovx, tanfovy) from the actual ray-direction extents of a raymap.

    Used in KB/EQ mode to replace the ``fov_mod``-based heuristic, which
    systematically under-estimates the FOV and causes the PBF frustum-clipping
    in the CUDA kernel (``computePBF``/``computeAABB_*``) to cull valid
    edge-of-image Gaussians during training.

    Parameters
    ----------
    raymap : numpy.ndarray or torch.Tensor, shape (H, W, 3)
        Per-pixel camera-space ray directions (rx, ry, rz).  Rays are assumed
        to point forward (rz > 0 for in-image pixels).
    min_rz : float
        Pixels whose rz is at or below this threshold are ignored to avoid
        division by zero / near-infinite tangent values (e.g. rays at ≥90°).
    max_tan : float
        Hard upper cap on the returned tangent values (prevents infinities
        from slipping through; ``atan(1e4) ≈ 89.99°``).

    Returns
    -------
    (tanfovx, tanfovy) : (float, float) or (None, None)
        Maximum absolute tangent values in x and y.  Returns ``(None, None)``
        when no valid pixels are found so the caller can keep its default.
    """
    if isinstance(raymap, torch.Tensor):
        arr = raymap.detach().cpu().numpy() if raymap.is_cuda else raymap.numpy()
    else:
        arr = np.asarray(raymap, dtype=np.float32)

    rz = arr[:, :, 2]
    valid = rz > min_rz
    if not valid.any():
        return None, None

    safe_rz = np.where(valid, rz, 1.0)
    tanx = np.abs(arr[:, :, 0]) / safe_rz
    tany = np.abs(arr[:, :, 1]) / safe_rz
    tanfovx = float(np.clip(tanx[valid].max(), 0.0, max_tan))
    tanfovy = float(np.clip(tany[valid].max(), 0.0, max_tan))
    return tanfovx, tanfovy

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(np.float64, elems[4:])))
    return camera_id, model, width, height, params

def fov2tan(fovx, fovy, interval):
    theta_arr = np.arange(interval / 2, fovx, interval)
    theta_arr = np.sort(np.concatenate((-theta_arr, theta_arr)))
    phi_arr = np.arange(interval / 2, fovy, interval)
    phi_arr = np.sort(np.concatenate((-phi_arr, phi_arr)))

    sin_t = np.sin(theta_arr)
    cos_t = np.cos(theta_arr)
    sin_p = np.sin(phi_arr)[:, None]
    cos_p = np.cos(phi_arr)[:, None]
    tan_t = sin_t / cos_t
    tan_p = sin_p / cos_p
    return tan_t, tan_p

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, 
                 focal_x, focal_y, principal_x, principal_y, distortion_coeffs,
                 depth_params, image, invdepthmap,
                 image_name, uid, step,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 render_model = "BEAP", focal_scaling = 1.0, distortion_scaling = 1.0, mirror_shift = 0.0,
                 raymap = None,
                 intr_width = None, intr_height = None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # ── GEER ray-direction grids ──────────────────────────────────────────
        arr_theta, arr_phi = self.fov_sample2ray(FoVx/2, FoVy/2, step)
        cos_theta = torch.cos(arr_theta)
        cos_phi = torch.cos(arr_phi)
        
        cos_theta = torch.where(torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta).to(self.data_device)
        cos_phi = torch.where(torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi).to(self.data_device)
        self.tan_theta = torch.tan(arr_theta).to(self.data_device)
        self.tan_phi = torch.tan(arr_phi).to(self.data_device)
        self.mirror_transformed_tan_theta = self.mirror_transform(self.tan_theta, cos_theta).float()
        self.mirror_transformed_tan_phi = self.mirror_transform(self.tan_phi, cos_phi).float()
        self.sampled_image = self.original_image

        self.render_model = 0 if render_model == "BEAP" else (1 if render_model in ("KB", "EQ") else 2)

        if render_model == "BEAP":
            # BEAP mode: use the sorted tan_theta / tan_phi grids for rasterization.
            # No fisheye intrinsics are needed; the image dimensions come from the
            # ray-grid sizes.
            self.focal_x = None
            self.focal_y = None
            self.principal_x = None
            self.principal_y = None
            self.distortion_coeffs = None
            self.mirror_shift = None
            self.raymap = None
            self.image_width = self.tan_theta.shape[0]
            self.image_height = self.tan_phi.shape[0]
        elif render_model == "KB" or render_model == "EQ":
            # KB mode: Kannala-Brandt fisheye camera.  Uses a per-pixel raymap and
            # polynomial distortion coefficients for the PBF→pixel projection.
            # EQ mode: equidistant fisheye (distortion_scaling=0 zeroes the KB coeffs).
            distortion_scaling = 0.0 if render_model == "EQ" else distortion_scaling
            self.focal_x = focal_x.item() * focal_scaling
            self.focal_y = focal_y.item() * focal_scaling
            self.principal_x = principal_x.item()
            self.principal_y = principal_y.item()
            self.distortion_coeffs = torch.from_numpy(distortion_coeffs.astype(np.float32)) * distortion_scaling
            self.mirror_shift = mirror_shift
            assert raymap is not None
            self.raymap = torch.from_numpy(raymap.astype(np.float32)) #self.scannetpp_raymap(raymap, resolution, focal_x, focal_y, FoVx, FoVy, step)
            self.image_width = self.raymap.shape[1]
            self.image_height = self.raymap.shape[0]
            # Override FoVx/FoVy using the actual ray extents of the raymap
            _tanfovx, _tanfovy = _tanfov_from_raymap(raymap)
            if _tanfovx is not None:
                self.FoVx = 2.0 * math.atan(_tanfovx)
                self.FoVy = 2.0 * math.atan(_tanfovy)
        elif render_model == "PH":
            # Pinhole camera: standard perspective projection using focal lengths and
            # principal point.  No distortion correction or per-pixel raymap required;
            # the CUDA else-branch computes ray directions directly from focal lengths.
            if focal_x is None or focal_y is None or principal_x is None or principal_y is None:
                raise ValueError(
                    "PH render mode requires focal_x, focal_y, principal_x, and "
                    "principal_y to be set (got None for one or more of these values)"
                )
            ref_w = intr_width if intr_width is not None else image.size[0]
            ref_h = intr_height if intr_height is not None else image.size[1]
            scale_x = resolution[0] / ref_w
            scale_y = resolution[1] / ref_h
            self.focal_x = focal_x.item() * focal_scaling * scale_x
            self.focal_y = focal_y.item() * focal_scaling * scale_y
            self.principal_x = principal_x.item() * scale_x
            self.principal_y = principal_y.item() * scale_y
            self.distortion_coeffs = None
            self.mirror_shift = None
            self.raymap = None
            self.image_width = resolution[0]
            self.image_height = resolution[1]
            # Recompute FoVx/FoVy from the actual (post-scaling) focal lengths
            self.FoVx = 2.0 * math.atan(self.image_width / (2.0 * self.focal_x))
            self.FoVy = 2.0 * math.atan(self.image_height / (2.0 * self.focal_y))
            # Update the projection matrix to match the corrected FOV.
            self.projection_matrix = getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            ).transpose(0, 1).cuda()
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
            ).squeeze(0)

    @staticmethod
    def fov_sample2ray(fovx, fovy, interval):
        """Build symmetric 1-D arrays of ray-direction half-angles in [-fov, +fov].

        Each element is spaced `interval` radians apart, starting at interval/2.
        Returns (theta_arr, phi_arr) as sorted float tensors.
        """
        theta_arr = torch.arange(interval / 2, fovx, interval)
        theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
        phi_arr = torch.arange(interval / 2, fovy, interval)
        phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))

        return theta_arr.float(), phi_arr.float()

    @staticmethod
    def mirror_transform(m, z, xi=0.0): #1.1
        """Apply the omnidirectional mapping to a tangent array m.

        Mirror transform tan(θ); reference: Appendix D.2.
        """
        return m / (1+xi*(z/(torch.abs(z)))*(1+m**2)**0.5)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, sample_step,
                 render_model=0, focal_x=None, focal_y=None, principal_x=None, principal_y=None,
                 distortion_coeffs=None, raymap=None):
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.sample_step = sample_step
        self.render_model = render_model

        arr_theta, arr_phi = self.fov_sample2ray(self.FoVx/2, self.FoVy/2, sample_step)

        cos_theta = torch.cos(arr_theta)
        cos_phi = torch.cos(arr_phi)

        cos_theta = torch.where(torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta)
        cos_phi = torch.where(torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi)
        self.tan_theta = torch.tan(arr_theta)
        self.tan_phi = torch.tan(arr_phi)
        self.mirror_transformed_tan_theta = self.mirror_transform(self.tan_theta, cos_theta)
        self.mirror_transformed_tan_phi = self.mirror_transform(self.tan_phi, cos_phi)

        if render_model == 0:  # BEAP
            self.focal_x = None
            self.focal_y = None
            self.principal_x = None
            self.principal_y = None
            self.distortion_coeffs = None
            self.raymap = None
            # Derive image dimensions from the ray-grid sizes, consistent with Camera BEAP mode.
            self.image_width = self.tan_theta.shape[0]
            self.image_height = self.tan_phi.shape[0]
        elif render_model == 1:  # KB / EQ
            if raymap is None:
                raise ValueError("KB/EQ render mode requires a raymap")
            self.focal_x = focal_x
            self.focal_y = focal_y
            self.principal_x = principal_x
            self.principal_y = principal_y
            self.distortion_coeffs = distortion_coeffs
            self.raymap = raymap
            self.image_width = raymap.shape[1]
            self.image_height = raymap.shape[0]
            # Mirror the same fov_mod-independent FOV derivation used in Camera.
            _tanfovx, _tanfovy = _tanfov_from_raymap(raymap)
            if _tanfovx is not None:
                self.FoVx = 2.0 * math.atan(_tanfovx)
                self.FoVy = 2.0 * math.atan(_tanfovy)
        elif render_model == 2:  # PH
            self.focal_x = focal_x
            self.focal_y = focal_y
            self.principal_x = principal_x
            self.principal_y = principal_y
            self.distortion_coeffs = None
            self.raymap = None
            self.image_width = width
            self.image_height = height
            # Mirror the FoVx/FoVy correction from Camera PH mode: if focal
            # lengths are provided, recompute the FOV from them so that tanfovx/
            # tanfovy in the CUDA rasterizer are consistent with focal_x/focal_y.
            if focal_x is not None and focal_y is not None and focal_x > 0 and focal_y > 0:
                self.FoVx = 2.0 * math.atan(self.image_width / (2.0 * self.focal_x))
                self.FoVy = 2.0 * math.atan(self.image_height / (2.0 * self.focal_y))

    @staticmethod
    def fov_sample2ray(fovx, fovy, interval):
        theta_arr = torch.arange(interval / 2, fovx, interval)
        theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
        phi_arr = torch.arange(interval / 2, fovy, interval)
        phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))
        
        return theta_arr.float(), phi_arr.float()

    @staticmethod
    def mirror_transform(m, z, xi=0.0): #1.1
        return m / (1+xi*(z/(torch.abs(z)))*(1+m**2)**0.5)
    
    def get_viewpoint_mask(self, ref_camera_dir):
        _, _, width, height, params = read_intrinsics_text(ref_camera_dir)
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        kk = np.asarray(params[4:], dtype=np.float64)
        if kk.size < 4:
            kk = np.pad(kk, (0, 4 - kk.size), mode="constant")
        kk = kk[:4]
        tan_theta, tan_phi = fov2tan(self.FoVx/2, self.FoVy/2, self.sample_step)
        radius = np.sqrt(tan_theta ** 2 + tan_phi ** 2)
        theta = np.arctan(radius)
        r = theta * (1.0 + kk[0] * theta**2 + kk[1] * theta**4 + kk[2] * theta**6 + kk[3] * theta**8)
        u = tan_theta * r * fx / radius + cx
        v = tan_phi * r * fy / radius + cy
        u_mask = np.logical_and(u >= 0, u < width)
        v_mask =  np.logical_and(v >= 0, v < height) 
        valid_mask = u_mask & v_mask
        self.valid_mask = (valid_mask).astype(np.uint8)
        return self.valid_mask


        

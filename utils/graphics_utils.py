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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def focal2fov2(focal, pixels):
    return pixels / focal

def clamp_projection(t, tan_fovx, tan_fovy):
    """
    Given a transformed point t, clamps its x and y components.
    
    t: tensor with [t.x, t.y, t.z]
    tan_fovx, tan_fovy: tangent values of half the field of view in x and y.
    """
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    # Clamp the normalized x and y
    clamped_x = torch.clamp(txtz, -limx, limx)
    clamped_y = torch.clamp(tytz, -limy, limy)
    # Multiply back by t.z to get the clamped coordinates
    t_new = t.clone()
    t_new[0] = clamped_x * t[2]
    t_new[1] = clamped_y * t[2]
    return t_new

def project_to_screen(mean, view_matrix, tan_fovx, tan_fovy, W, H):
    view_matrix = view_matrix.t()
    # 1. Transform the point 'mean' using the view matrix.
    P, _ = mean.shape
    ones = torch.ones(P, 1, dtype=mean.dtype, device=mean.device)
    mean_hom = torch.cat([mean, ones], dim=1)
    t = torch.matmul(view_matrix.unsqueeze(0), mean_hom.unsqueeze(2))
    #t = torch.matmul(mean_hom, view_matrix.unsqueeze(0))
    t = t[:, :3, 0] # (P, 3)
    
    # --- 2. Clamp the x and y components ---
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy

    # Compute normalized x and y (i.e. x/z and y/z)
    t_x = t[:, 0]
    t_y = t[:, 1]
    t_z = t[:, 2]

    txtz = t_x / t_z
    tytz = t_y / t_z

    # Clamp these normalized coordinates between [-lim, lim]
    txtz_clamped = torch.clamp(txtz, min=-limx, max=limx)
    tytz_clamped = torch.clamp(tytz, min=-limy, max=limy)

    # Multiply back by t_z to get the new x and y components.
    t[:, 0] = txtz_clamped * t_z
    t[:, 1] = tytz_clamped * t_z

    # --- 3. Construct the matrix J for each point ---
    # J is defined per point as:
    # [ [ focal_x/t_z,        0,  -(focal_x*t_x)/(t_z^2) ],
    #   [        0,   focal_y/t_z, -(focal_y*t_y)/(t_z^2) ],
    #   [        0,           0,                      0 ] ]
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    J_row0 = torch.stack([focal_x / t_z, torch.zeros_like(t_z), -(focal_x * t_x) / (t_z * t_z)], dim=1)
    J_row1 = torch.stack([torch.zeros_like(t_z), focal_y / t_z, -(focal_y * t_y) / (t_z * t_z)], dim=1)
    J = torch.stack([J_row0, J_row1], dim=2)  # shape (P, 3, 2)

    # 4. Construct matrix V from the view matrix.
    V = view_matrix.t()[:3, :3]
    
    # 5. Compute T = V * J (matrix multiplication).
    T = torch.matmul(V.unsqueeze(0), J)  # shape (P, 3, 2)

    grad = torch.matmul(T.permute(0,2,1), mean.grad.unsqueeze(2))[..., 0]

    return grad / 75.0
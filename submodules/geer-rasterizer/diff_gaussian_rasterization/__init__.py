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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings,
    ):
        """Forward pass: rasterize 3D Gaussians into an image using the GEER algorithm.

        Returns (color [C,H,W], radii [P], invdepth [1,H,W], kernel_times [5], tile_ranges [T]).
        """

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.mirror_transformed_tan_theta,
            raster_settings.mirror_transformed_tan_phi,
            raster_settings.tan_theta,
            raster_settings.tan_phi,
            raster_settings.focal_x, raster_settings.focal_y, 
            raster_settings.principal_x, raster_settings.principal_y,
            raster_settings.distortion_coeffs,
            raster_settings.raymap,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.render_mode,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, kernel_times, ranges, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args) # ranges for each tile

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths, kernel_times, ranges

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):
        """Backward pass: propagate gradients from pixel loss to Gaussian parameters.

        Inputs grad_out_color [C,H,W] and grad_out_depth [1,H,W] (ignored middle output).
        Returns gradients for (means3D, means2D, sh, colors_precomp, opacities, scales, rotations, None).
        """

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                raster_settings.viewmatrix, 
                raster_settings.tan_theta,
                raster_settings.tan_phi, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    """
        Settings passed to the CUDA rasterizer for each forward/backward call.
    """
    image_height: int
    image_width: int
    tanfovx: float        
    tanfovy: float       
    bg: torch.Tensor       
    scale_modifier: float   
    viewmatrix: torch.Tensor  
    # BEAP ray-direction reference arrays (sorted, used for PBF→pixel mapping in BEAP mode)
    mirror_transformed_tan_theta: torch.Tensor  # 1-D tensor of tan(θ) values along the horizontal axis
    mirror_transformed_tan_phi: torch.Tensor    # 1-D tensor of tan(φ) values along the vertical axis
    # Per-pixel ray direction tangents (BEAP mode: 1-D sorted grids; KB mode: unused)
    tan_theta: torch.Tensor  # Horizontal ray tangents
    tan_phi: torch.Tensor    # Vertical ray tangents
    # KB / EQ fisheye intrinsics (only used when render_model==1)
    focal_x: float          # Horizontal focal length (pixels)
    focal_y: float          # Vertical focal length (pixels)
    principal_x: float      # Horizontal principal point (pixels)
    principal_y: float      # Vertical principal point (pixels)
    distortion_coeffs: torch.Tensor  # KB polynomial distortion coefficients [k1, k2, k3, k4]
    raymap: torch.Tensor    # [H, W, 3] per-pixel ray direction map (used in KB/EQ mode)
    sh_degree: int          
    campos: torch.Tensor    # Camera centre in world space [3]
    prefiltered: bool       
    debug: bool            
    antialiasing: bool   
    render_mode: int   

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            raster_settings, 
        )


from typing import Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from ..cuda._wrapper import (
    FThetaCameraDistortionParameters,
    isect_tiles_geer,
)
from .camera import get_camera_tanfov


def _prepare_geer(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"],
    rasterize_mode: Literal["classic", "antialiased"],
    packed: bool,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
) -> Tuple[
    Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]],
    Tuple[Tensor, Tensor, Tensor, Tensor],
    Tensor,
]:
    """Prepare GEER metadata and tile intersections without a 2D projection."""
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]

    if rasterize_mode == "antialiased":
        # 3DGEER antialiasing inflates the 3D covariance rather than a
        # projected 2D covariance. Use the inflated scales for both PBF
        # construction and exact ray evaluation, then preserve Gaussian mass
        # with the corresponding determinant ratio.
        h_var = 1e-7
        antialiased_scales = torch.sqrt(scales.square() + h_var)
        compensations = torch.prod(scales / antialiased_scales, dim=-1)
        geer_opacities = opacities * compensations
        compensations = compensations[..., None, :]
        scales = antialiased_scales
    else:
        compensations = None
        geer_opacities = opacities

    tanfovx, tanfovy = get_camera_tanfov(
        camera_model,
        Ks,
        width,
        height,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
    )

    tiles_per_gauss, isect_ids, flatten_ids, pbf_bounds = isect_tiles_geer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=geer_opacities,
        viewmats=viewmats,
        camera_model=camera_model,
        Ks=Ks,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=packed,
    )

    pbf_bounds = pbf_bounds.reshape(batch_dims + (C, N, 4))
    pbf_spans = torch.stack(
        (
            pbf_bounds[..., 1] - pbf_bounds[..., 0],
            pbf_bounds[..., 3] - pbf_bounds[..., 2],
        ),
        dim=-1,
    ).clamp_min_(0)
    radii = (pbf_spans + 1) // 2

    pbf_bounds_f = pbf_bounds.to(means)
    means2d = torch.stack(
        (
            0.5 * (pbf_bounds_f[..., 0] + pbf_bounds_f[..., 1]),
            0.5 * (pbf_bounds_f[..., 2] + pbf_bounds_f[..., 3]),
        ),
        dim=-1,
    )

    world_to_camera_z = viewmats[..., 2, :3]
    camera_z_offset = viewmats[..., 2, 3]
    depths = (
        world_to_camera_z @ means.transpose(-1, -2)
        + camera_z_offset[..., None]
    )
    conics = means.new_zeros(batch_dims + (C, N, 3))

    proj_results = radii, means2d, depths, conics, compensations
    intersections = tiles_per_gauss, isect_ids, flatten_ids, pbf_bounds
    return proj_results, intersections, scales

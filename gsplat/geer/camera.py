import math
from collections import OrderedDict

import numpy as np
import torch

from ..cuda._wrapper import compute_raymap


_TANFOV_CACHE_MAX_SIZE = 256
_TANFOV_CACHE = OrderedDict()


def _tensor_cache_key(value):
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach()
    values = tensor.to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
    return tuple(tensor.shape), str(tensor.dtype), tuple(values)


def _ftheta_cache_key(ftheta_coeffs):
    if ftheta_coeffs is None:
        return None
    return (
        ftheta_coeffs.reference_poly.value,
        tuple(ftheta_coeffs.pixeldist_to_angle_poly),
        tuple(ftheta_coeffs.angle_to_pixeldist_poly),
        float(ftheta_coeffs.max_angle),
        tuple(ftheta_coeffs.linear_cde),
    )


def _camera_tanfov_cache_key(
    camera_model,
    Ks,
    width,
    height,
    radial_coeffs,
    tangential_coeffs,
    thin_prism_coeffs,
    ftheta_coeffs,
):
    return (
        camera_model,
        width,
        height,
        str(Ks.device),
        _tensor_cache_key(Ks),
        _tensor_cache_key(radial_coeffs),
        _tensor_cache_key(tangential_coeffs),
        _tensor_cache_key(thin_prism_coeffs),
        _ftheta_cache_key(ftheta_coeffs),
    )


def _clear_camera_tanfov_cache():
    _TANFOV_CACHE.clear()


def _tanfov_from_raymap(raymap, min_rz: float = 1e-3, max_tan: float = 1e4):
    """Derive (tanfovx, tanfovy) from the actual ray-direction extents of a raymap.

    Used for nonlinear cameras to replace the ``fov_mod``-based heuristic, which
    systematically under-estimates the FOV and causes the PBF frustum-clipping
    in the CUDA kernel (``computePBF``) to cull valid
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
        rays = raymap.detach()
        rz = rays[..., 2]
        valid = rz > min_rz
        safe_rz = torch.where(valid, rz, torch.ones_like(rz))
        invalid = torch.full((), -torch.inf, dtype=rays.dtype, device=rays.device)
        tanfov = torch.stack(
            (
                torch.where(valid, rays[..., 0].abs() / safe_rz, invalid).amax(),
                torch.where(valid, rays[..., 1].abs() / safe_rz, invalid).amax(),
            )
        ).cpu()
        tanfovx, tanfovy = (float(value) for value in tanfov)
        if not math.isfinite(tanfovx) or not math.isfinite(tanfovy):
            return None, None
        return min(max(tanfovx, 0.0), max_tan), min(
            max(tanfovy, 0.0), max_tan
        )

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

def get_camera_tanfov(
    camera_model,
    Ks,
    width,
    height,
    fov_mod=1,
    radial_coeffs=None,
    tangential_coeffs=None,
    thin_prism_coeffs=None,
    ftheta_coeffs=None,
):
    """Return tangent FOV extents for one supported GEER camera."""
    # Ks [..., C, 3, 3]
    K = Ks.to("cpu").squeeze() # one image

    focal_length = (K[0, 0] * fov_mod, K[1, 1] * fov_mod)

    has_pinhole_distortion = (
        radial_coeffs is not None
        or tangential_coeffs is not None
        or thin_prism_coeffs is not None
    )

    if camera_model == "pinhole" and not has_pinhole_distortion:
        return width / (2 * focal_length[0]), height / (2 * focal_length[1])
    elif camera_model in ("pinhole", "fisheye", "ftheta"):
        cache_key = _camera_tanfov_cache_key(
            camera_model,
            Ks,
            width,
            height,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_coeffs,
        )
        cached = _TANFOV_CACHE.get(cache_key)
        if cached is not None:
            _TANFOV_CACHE.move_to_end(cache_key)
            return cached

        raymap = compute_raymap(
            Ks,
            width,
            height,
            camera_model=camera_model,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
        ).squeeze() # [H,W,3] assume one image

        tanfovx, tanfovy = _tanfov_from_raymap(raymap)
        if tanfovx is None or tanfovy is None:
            raise ValueError(
                f"No forward-facing image rays found for GEER {camera_model} camera"
            )
        _TANFOV_CACHE[cache_key] = (tanfovx, tanfovy)
        _TANFOV_CACHE.move_to_end(cache_key)
        if len(_TANFOV_CACHE) > _TANFOV_CACHE_MAX_SIZE:
            _TANFOV_CACHE.popitem(last=False)
        return tanfovx, tanfovy

    else:
        raise ValueError(f"Camera model not supported by GEER: {camera_model}")

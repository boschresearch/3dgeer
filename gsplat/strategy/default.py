from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import warnings
from typing_extensions import Literal

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class DefaultStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    # Safety knobs to prevent unbounded GS growth (helps avoid OOM).
    # 0 means "no limit".
    max_gaussians: int = 0
    max_grow_per_refine: int = 0

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {
            "grad2d": None,
            "count": None,
            "scene_scale": scene_scale,
            "_warned_no_grad_key_for_gradient": False,
            "_warned_missing_grad_key_for_gradient": False,
            "_warned_growth_clipped": False,
        }
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def _maybe_initialize_running_stats(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
    ) -> None:
        """Ensure the running stats tensors exist on the correct device.

        This is normally done in `_update_state()`, but some rendering modes (e.g. UT /
        eval3d) do not produce a differentiable tensor for `key_for_gradient`, in which
        case `_update_state()` will early-return and we still want the strategy to be
        a no-op instead of crashing later when refinement triggers.
        """
        # Use a stable reference parameter for sizing/device.
        try:
            ref_param = params["means"]
        except Exception:
            ref_param = next(iter(params.values()))
        n_gaussian = len(ref_param)
        device = ref_param.device

        if state.get("grad2d", None) is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=device)
        if state.get("count", None) is None:
            state["count"] = torch.zeros(n_gaussian, device=device)
        if self.refine_scale2d_stop_iter > 0 and state.get("radii", None) is None:
            state["radii"] = torch.zeros(n_gaussian, device=device)

    def _fallback_image_plane_grads_from_means_grad(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        info: Dict[str, Any],
        packed: bool,
    ) -> Optional[torch.Tensor]:
        """Approximate per-Gaussian image-plane grads from `params["means"].grad`.

        This is used when `info[self.key_for_gradient]` is not differentiable or its
        `.grad` is unavailable (common with `with_ut=True` and/or `with_eval3d=True`).

        Limitations:
        - Only supports non-packed mode.
        - Assumes `info["depths"]` stores camera-space depth (Z).
        - Uses a pinhole-style approximation even for distorted camera models.
        """
        if packed:
            return None
        if "viewmats" not in info or "Ks" not in info or "depths" not in info:
            return None
        try:
            means_param = params["means"]
        except Exception:
            return None
        means_grad = info.get("_strategy_means_grad", None)
        if means_grad is None:
            means_grad = means_param.grad
        if means_grad is None or means_grad.ndim != 2 or means_grad.shape[-1] != 3:
            return None

        viewmats = info["viewmats"]
        Ks = info["Ks"]
        depths = info["depths"]
        if viewmats.ndim != 3 or Ks.ndim != 3:
            return None
        if viewmats.shape[-2:] != (4, 4) or Ks.shape[-2:] != (3, 3):
            return None

        # depths: [C, N] or [C, N, 1] (some backends may keep a trailing dim)
        if depths.ndim == 3 and depths.shape[-1] == 1:
            depths = depths.squeeze(-1)
        if depths.ndim != 2:
            return None

        # Transform world-space mean gradients into camera space: dL/dXc = R * dL/dXw
        R = viewmats[:, :3, :3]  # [C, 3, 3]
        if R.shape[0] != Ks.shape[0] or R.shape[0] != depths.shape[0]:
            return None
        if depths.shape[1] != means_grad.shape[0]:
            return None

        g_cam = torch.einsum("cij,nj->cni", R, means_grad)  # [C, N, 3]

        fx = Ks[:, 0, 0].clamp_min(1e-6).unsqueeze(-1)  # [C, 1]
        fy = Ks[:, 1, 1].clamp_min(1e-6).unsqueeze(-1)  # [C, 1]
        Z = depths  # [C, N]

        # From x = fx * X/Z + cx, y = fy * Y/Z + cy:
        # dL/dx = (dL/dX) * (Z/fx), dL/dy = (dL/dY) * (Z/fy)
        dL_dx = g_cam[..., 0] * Z / fx  # [C, N]
        dL_dy = g_cam[..., 1] * Z / fy  # [C, N]
        return torch.stack([dL_dx, dL_dy], dim=-1)  # [C, N, 2]

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        geer_gradient = info.get("geer_gradient")
        key_tensor = (
            geer_gradient if geer_gradient is not None else info[self.key_for_gradient]
        )
        if not getattr(key_tensor, "requires_grad", False):
            # Some projection / rendering modes intentionally return non-differentiable
            # metadata tensors (e.g. UT projection). In that case, densification based
            # on `key_for_gradient` cannot run; we keep training running and make the
            # strategy a no-op instead of crashing.
            if not state.get("_warned_no_grad_key_for_gradient", False):
                warnings.warn(
                    f"DefaultStrategy: info['{self.key_for_gradient}'] has "
                    "requires_grad=False, so 2D-gradient-based refinement "
                    "(duplicate/split) will be disabled. This can happen when the "
                    "renderer/projection returns non-differentiable meta tensors "
                    "(e.g. `with_ut=True` or `with_eval3d=True`).",
                    category=UserWarning,
                    stacklevel=2,
                )
                state["_warned_no_grad_key_for_gradient"] = True
            return
        key_tensor.retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._maybe_initialize_running_stats(params, state)
        updated = self._update_state(params, state, info, packed=packed)
        if not updated and not state.get("_warned_missing_grad_key_for_gradient", False):
            warnings.warn(
                f"DefaultStrategy: gradient for info['{self.key_for_gradient}'] was "
                "not available after backward, so 2D-gradient-based refinement "
                "(duplicate/split) will be disabled.",
                category=UserWarning,
                stacklevel=2,
            )
            state["_warned_missing_grad_key_for_gradient"] = True

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ) -> bool:
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        geer_gradient = info.get("geer_gradient")
        using_geer_gradient = geer_gradient is not None
        key_tensor = (
            geer_gradient if using_geer_gradient else info[self.key_for_gradient]
        )
        if self.absgrad and not using_geer_gradient:
            grads_src = getattr(key_tensor, "absgrad", None)
        else:
            grads_src = getattr(key_tensor, "grad", None)
        if grads_src is None:
            grads_src = self._fallback_image_plane_grads_from_means_grad(
                params=params, info=info, packed=packed
            )
        if grads_src is None:
            return False

        # Standard projected-mean gradients are normalized to [-1, 1] screen
        # space. The GEER proxy already includes the depth-dependent scaling
        # from the original rasterizer and must not be scaled a second time.
        grads = grads_src.clone()
        if not using_geer_gradient:
            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"].max(dim=-1).values  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )
        return True

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_dupli = is_dupli.sum().item()
        n_split = is_split.sum().item()

        # Clip growth to avoid unbounded explosions in #Gaussians (which can lead to OOM).
        max_new = None
        if self.max_gaussians > 0:
            max_new = max(0, int(self.max_gaussians) - int(len(params["means"])))
        if self.max_grow_per_refine > 0:
            max_new = (
                int(self.max_grow_per_refine)
                if max_new is None
                else min(max_new, int(self.max_grow_per_refine))
            )
        if max_new is not None:
            if max_new <= 0:
                return 0, 0
            n_new = n_dupli + n_split
            if n_new > max_new:
                # Select the top-k candidates by gradient magnitude.
                cand_dupli = torch.where(is_dupli)[0]
                cand_split = torch.where(is_split)[0]
                cand_ids = torch.cat([cand_dupli, cand_split], dim=0)
                if len(cand_ids) > 0:
                    scores = grads[cand_ids]
                    k = min(int(max_new), int(scores.numel()))
                    topk = torch.topk(scores, k=k, sorted=False).indices
                    sel_ids = cand_ids[topk]
                    selected = torch.zeros_like(is_grad_high, dtype=torch.bool)
                    selected[sel_ids] = True
                    is_dupli = is_dupli & selected
                    is_split = is_split & selected
                    n_dupli = is_dupli.sum().item()
                    n_split = is_split.sum().item()
                else:
                    is_dupli = torch.zeros_like(is_grad_high, dtype=torch.bool)
                    is_split = torch.zeros_like(is_grad_high, dtype=torch.bool)
                    n_dupli, n_split = 0, 0

                if not state.get("_warned_growth_clipped", False):
                    warnings.warn(
                        "DefaultStrategy: clipped densification (duplicate/split) to "
                        f"{max_new} new Gaussians this refine step to avoid OOM. "
                        "Tune with `max_gaussians` / `max_grow_per_refine`.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                    state["_warned_growth_clipped"] = True

        # first duplicate
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune

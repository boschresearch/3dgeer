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

import os
import torch
import numpy as np
import cv2
import sys
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render, network_gui, GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import match_mask_to_image


def visualize(dataset, opt, pipe, iteration, sample_step, fov_mod, mask_path,
              sibr_mask_refcam=None, render_model='BEAP', focal_scaling=1.0,
              distortion_scaling=1.0, mirror_shift=0.0, raymap_path=None):
    """
    Load a trained checkpoint and serve the SIBR online viewer via network_gui.
    This mirrors the network_gui loop from train.py but runs indefinitely after
    training is complete, without performing any gradient updates.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        dataset.fov_mod = fov_mod
        dataset.sample_step = sample_step
        dataset.raymap = None
        if raymap_path is not None and os.path.exists(raymap_path):
            try:
                dataset.raymap = np.load(raymap_path)
            except (IOError, ValueError) as e:
                print(f"Warning: could not load raymap from '{raymap_path}': {e}")
        dataset.render_model = render_model
        dataset.focal_scaling = focal_scaling
        dataset.distortion_scaling = distortion_scaling
        dataset.mirror_shift = mirror_shift

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        valid_mask = None
        if mask_path is not None:
            valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if valid_mask is not None:
                valid_mask = np.repeat(valid_mask[None, ...], 3, axis=0)
                valid_mask = torch.tensor(valid_mask)

        print("Loaded model at iteration", scene.loaded_iter)
        print("Waiting for SIBR viewer connection…")

        # Map render_model string to the integer code used by the rasterizer
        # (0=BEAP, 1=KB/EQ, 2=PH) and pre-extract any camera intrinsics that
        # need to be forwarded to every MiniCam created inside the receive() loop.
        _render_model_map = {"BEAP": 0, "KB": 1, "EQ": 1, "PH": 2}
        render_model_int = _render_model_map.get(render_model, 0)
        cam_extra_params: dict = {}
        if render_model in ("KB", "EQ", "PH"):
            train_cams = scene.getTrainCameras()
            if train_cams:
                ref_cam = train_cams[0]
                cam_extra_params["focal_x"] = ref_cam.focal_x
                cam_extra_params["focal_y"] = ref_cam.focal_y
                cam_extra_params["principal_x"] = ref_cam.principal_x
                cam_extra_params["principal_y"] = ref_cam.principal_y
                if render_model in ("KB", "EQ"):
                    cam_extra_params["distortion_coeffs"] = ref_cam.distortion_coeffs
                    cam_extra_params["raymap"] = ref_cam.raymap

        while True:
            if network_gui.conn is None:
                network_gui.try_connect()
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    # sample_step is forwarded to network_gui.receive() so it can
                    # construct the MiniCam with the correct ray-sampling step size.
                    # render_model_int and cam_extra_params propagate KB/PH intrinsics.
                    extra_params = {
                        "sample_step": sample_step,
                        "render_model_int": render_model_int,
                        **cam_extra_params,
                    }
                    (custom_cam, do_training,
                     pipe.convert_SHs_python, pipe.compute_cov3D_python,
                     keep_alive, scaling_modifier,
                     width, height) = network_gui.receive(extra_params)

                    if custom_cam is not None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier)["render"]

                        if sibr_mask_refcam is not None:
                            net_mask = custom_cam.get_viewpoint_mask(sibr_mask_refcam)
                            net_mask = torch.tensor(np.repeat(net_mask[None, ...], 3, axis=0))
                            net_image[net_mask == 0] = 0.0

                        if valid_mask is not None:
                            net_image[match_mask_to_image(valid_mask, net_image) == 0] = 0.0

                        net_image = torch.nn.functional.interpolate(
                            net_image[None, ...], (height, width), mode='bilinear')[0]
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255)
                            .byte().permute(1, 2, 0).contiguous().cpu().numpy())

                    network_gui.send(net_image_bytes, dataset.source_path)

                    # Stop serving only when the viewer explicitly disconnects
                    if not keep_alive:
                        break
                except Exception as e:
                    print("Network GUI error:", e)
                    network_gui.conn = None


if __name__ == "__main__":
    parser = ArgumentParser(description="Online SIBR visualizer (post-training)")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--iteration', type=int, default=-1,
                        help="Checkpoint iteration to load (-1 = latest saved)")
    parser.add_argument('--sample_step', type=float, default=None)
    parser.add_argument('--fov_mod', type=float, default=None)
    parser.add_argument('--mask_path', type=str, default=None)
    parser.add_argument('--sibr_mask_refcam', type=str, default=None)
    parser.add_argument('--render_model', type=str, default='BEAP',
                        choices=['BEAP', 'KB', 'PH'],
                        help="Render mode: BEAP (default), KB, or PH")
    parser.add_argument('--focal_scaling', type=float, default=1.0)
    parser.add_argument('--distortion_scaling', type=float, default=1.0)
    parser.add_argument('--mirror_shift', type=float, default=0.0)
    parser.add_argument('--raymap_path', type=str, default=None,
                        help="Path to pre-generated raymap .npy file (required for KB mode)")

    args = get_combined_args(parser)

    # Ensure optional attributes are set on args with sensible defaults.
    # get_combined_args() skips None-valued command-line arguments during the
    # cfg_args merge, so attributes absent from cfg_args may be missing entirely
    # unless explicitly guarded here.
    for attr, default in [
        ('fov_mod', getattr(args, 'fov_mod', None)),
        ('sample_step', getattr(args, 'sample_step', None)),
        ('render_model', 'BEAP'),
        ('focal_scaling', 1.0),
        ('distortion_scaling', 1.0),
        ('mirror_shift', 0.0),
        ('mask_path', getattr(args, 'mask_path', None)),
        ('sibr_mask_refcam', getattr(args, 'sibr_mask_refcam', None)),
        ('raymap_path', getattr(args, 'raymap_path', None)),
    ]:
        if not hasattr(args, attr) or getattr(args, attr) is None:
            setattr(args, attr, default)

    print("Visualizing", args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)

    visualize(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.iteration,
        args.sample_step,
        args.fov_mod,
        args.mask_path,
        args.sibr_mask_refcam,
        args.render_model,
        args.focal_scaling,
        args.distortion_scaling,
        args.mirror_shift,
        args.raymap_path,
    )

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


def visualize(dataset, opt, pipe, iteration, sample_step, mask_path, sibr_mask_refcam=None):
    """
    Load a trained checkpoint and serve the SIBR online viewer via network_gui.
    This mirrors the network_gui loop from train.py but runs indefinitely after
    training is complete, without performing any gradient updates.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        dataset.fov_mod = getattr(dataset, 'fov_mod', None)
        dataset.sample_step = sample_step
        dataset.raymap = None
        dataset.render_model = getattr(dataset, 'render_model', 'BEAP')
        dataset.focal_scaling = getattr(dataset, 'focal_scaling', 1.0)
        dataset.distortion_scaling = getattr(dataset, 'distortion_scaling', 1.0)
        dataset.mirror_shift = getattr(dataset, 'mirror_shift', 0.0)

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

        while True:
            if network_gui.conn is None:
                network_gui.try_connect()
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    # sample_step is forwarded to network_gui.receive() so it can
                    # construct the MiniCam with the correct ray-sampling step size.
                    extra_params = {"sample_step": sample_step}
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
                            net_image[valid_mask == 0] = 0.0

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

    args = get_combined_args(parser)

    # Forward optional render-time attributes that may not exist in the cfg_args
    for attr, default in [
        ('fov_mod', args.fov_mod),
        ('sample_step', args.sample_step),
        ('render_model', 'BEAP'),
        ('focal_scaling', 1.0),
        ('distortion_scaling', 1.0),
        ('mirror_shift', 0.0),
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
        args.mask_path,
        args.sibr_mask_refcam,
    )

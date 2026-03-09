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

import time
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
import numpy as np

def render_set(model_path,name, iteration, views, gaussians, pipeline, background, train_test_exp):
    max_allocated_memory_before = torch.cuda.max_memory_allocated()
    print(f"Max Allocated Memory Before Rendering: {max_allocated_memory_before} bytes")
    torch.cuda.empty_cache()

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    asso_path = os.path.join(model_path, name, "ours_{}".format(iteration), "asso")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(asso_path, exist_ok=True)

    render_times_overall = []
    render_times_prep = []
    render_times_dup = []
    render_times_sort = []
    render_times_render = []

    render_times = []
    image_save_times = []

    range_lens = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_start = time.time()
        rendering_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)
        render_end = time.time()

        rendering = rendering_pkg["render"]
        runtime = rendering_pkg["time"]
        range_len = rendering_pkg["range_len"]  # ranges for each tile

        range_lens.append(range_len)

        render_times.append((render_end - render_start)*1000)

        render_times_overall.append(runtime[0])
        render_times_prep.append(runtime[1])
        render_times_dup.append(runtime[2])
        render_times_sort.append(runtime[3])
        render_times_render.append(runtime[4])

        image_save_start = time.time()
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        grid_size = 16
        grid_w = int((rendering.shape[2] + grid_size - 1) / grid_size)
        grid_h = int((rendering.shape[1] + grid_size - 1) / grid_size)

        torchvision.utils.save_image((range_len.reshape(grid_h, grid_w)[:, :, None].float() / range_len.max()).permute(2,0,1), \
                                      os.path.join(asso_path, '{0:05d}'.format(idx) + ".png"))

        image_save_end = time.time()
        image_save_times.append((image_save_end - image_save_start)*1000)
        
    
    means = torch.tensor(render_times).mean()
    maxs = torch.tensor(render_times).max()
    FPS = 1.0 / (means / 1000.0)
    print(f"  AVG_Render_Time : {means} ms")
    print(f"  MAX_Render_Time : {maxs} ms")
    print(f"  FPS: {FPS}")   
    max_allocated_memory_after = torch.cuda.max_memory_allocated()
    print(f"Max Allocated Memory After Rendering: {max_allocated_memory_after} bytes")

    # Print memory usage statistics
    print(f"Memory Usage: {max_allocated_memory_after - max_allocated_memory_before} bytes")

    range_lens = torch.cat(range_lens, dim=0)
    print(f"Associated Gaus num of each tile\n: mean {range_lens.float().mean().item()} Gaussians, std {range_lens.float().std().item()} Gaussians, min {range_lens.float().min().item()} Gaussians, max {range_lens.float().max().item()} Gaussians")

    means = torch.tensor(render_times_overall).mean()
    FPS = 1.0 / (means / 1000.0)
    print(f"  AVG_OVERALL_Time : {means} ms")
    print(f" AVG_OVERALL_Time FPS: {FPS}")

    means = torch.tensor(render_times_prep).mean()
    print(f"  AVG_PREP_Time : {means} ms")

    means = torch.tensor(render_times_dup).mean()
    print(f"  AVG_DUP_Time : {means} ms")

    means = torch.tensor(render_times_sort).mean()
    print(f"  AVG_SORT_Time : {means} ms")

    means = torch.tensor(render_times_render).mean()
    print(f"  AVG_RenFunc_Time : {means} ms")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, \
                fov_mod: float, sample_step: float, render_model: str, \
                focal_scaling: float, distortion_scaling: float, mirror_shift: float, raymap_path=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        dataset.fov_mod = fov_mod
        dataset.sample_step = sample_step

        dataset.raymap = None
        if raymap_path is not None and os.path.exists(raymap_path):
            dataset.raymap = np.load(raymap_path)
        dataset.render_model = render_model
        dataset.focal_scaling = focal_scaling
        dataset.distortion_scaling = distortion_scaling
        dataset.mirror_shift = mirror_shift

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_train_cameras=skip_train, skip_test_cameras=skip_test)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp)

        if not skip_test:
             dataset.train_test_exp = False
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_model", type=str, default = 'BEAP')
    parser.add_argument("--focal_scaling", type=float, default = 1.0)
    parser.add_argument("--distortion_scaling", type=float, default = 1.0)
    parser.add_argument("--mirror_shift", type=float, default = 0.0)
    parser.add_argument("--raymap_path", type=str, default = None)
    parser.add_argument("--sample_step", type=float, default = None)
    parser.add_argument("--fov_mod", type=float, default = None)
    args = get_combined_args(parser)
    for k in ["fov_mod", "sample_step", "distortion_scaling", "focal_scaling", "mirror_shift", "raymap_path"]:
        if not hasattr(args, k):
            setattr(args, k, None)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, \
                args.fov_mod, args.sample_step, args.render_model, \
                args.focal_scaling, args.distortion_scaling, args.mirror_shift, args.raymap_path)
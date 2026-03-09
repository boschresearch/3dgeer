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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import glob
import math

def readImages(renders_dir, gt_dir, renders_list, start, end):
    renders = []
    gts = []
    image_names = []
    for load_num in range(start, end):
        fname = renders_list[load_num].rsplit("/")[-1]
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, use_remap=False, iters=None, custom_gt=None, custom_mask=None, reverse_mask=False, block_mask=False):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            if iters is not None:
                if not method.endswith("ours_"+str(iters)):
                    continue
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            if use_remap:
                print("Remapped back to original space.")
                gt_dir = gt_dir.with_name(gt_dir.name + "_remap")
                renders_dir = renders_dir.with_name(renders_dir.name + "_remap")
            if custom_gt is not None:
                gt_dir = Path(custom_gt)
                print("Custom GT loaded from", gt_dir)
            renders_list = sorted(glob.glob(str(renders_dir / "*.png")))
            
            mask = None

            if not block_mask:
                try: mask_path = [f for f in renders_list if "mask" in f][0]
                except: mask_path = None
                if custom_mask is not None:
                    mask_path = custom_mask
                if mask_path is not None and len(mask_path) > 0:
                    mask = Image.open(mask_path)
                    mask = tf.to_tensor(mask).unsqueeze(0)[:, :3, :, :].cuda()
                if reverse_mask:
                    mask = 1 - mask
                    print(mask[mask == 1].sum())

            renders_list = [f for f in renders_list if "mask" not in f]
            num_rendered = len(renders_list)
            # Split into every N image to prevent one-time load in too many image that may cause OOM.
            N = 20
            ssims = []
            psnrs = []
            lpipss = []
            image_namess = []
            for i in range(math.ceil(num_rendered / N)):
                renders, gts, image_names = readImages(renders_dir, gt_dir, renders_list, i*N, min(num_rendered, (i+1)*N))
                image_namess.extend(image_names)

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx], mask=mask))
                    psnrs.append(psnr(renders[idx], gts[idx], mask=mask))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_namess)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_namess)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_namess)}})

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--use_remap', action='store_true')
    parser.add_argument('--reverse_mask', action='store_true')
    parser.add_argument('--iters', type=int, default = None)
    parser.add_argument('--custom_gt', type=str, default=None)
    parser.add_argument('--custom_mask', type=str, default=None)
    parser.add_argument('--block_mask', action='store_true')
    args = parser.parse_args()
    evaluate(args.model_paths, args.use_remap, args.iters, args.custom_gt, args.custom_mask, args.reverse_mask, args.block_mask)

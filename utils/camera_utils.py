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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
import psutil
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    distortion_coeffs = getattr(cam_info, 'distortion_coeffs', None)
    if distortion_coeffs is None and getattr(args, "render_model", None) in ("KB", "EQ"):
        distortion_coeffs = np.zeros(4, dtype=np.float32)

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, focal_x=cam_info.focal_x, focal_y=cam_info.focal_y,
                  principal_x=cam_info.principal_x, principal_y=cam_info.principal_y, distortion_coeffs=distortion_coeffs,
                  depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, step=args.sample_step, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,
                  render_model=args.render_model, 
                  focal_scaling=args.focal_scaling, distortion_scaling=args.distortion_scaling, mirror_shift=args.mirror_shift,
                  raymap=args.raymap,
                  intr_width=cam_info.width, intr_height=cam_info.height)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def cameraList_from_camInfos_fisheye(cam_infos, resolution_scale, is_nerf_synthetic, is_test_dataset, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        if id%100 == 0:
            print(f"[ INFO ] Loading camera {id}/{len(cam_infos)}")
            print_memory_usage()
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def print_memory_usage():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"RAM: {mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB")
    print(f"SWAP: {swap.used / 1e9:.2f} GB / {swap.total / 1e9:.2f} GB")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_vram = torch.cuda.get_device_properties(device).total_memory
        allocated_vram = torch.cuda.memory_allocated(device)
        cached_vram = torch.cuda.memory_reserved(device)
        
        print(f"GPU VRAM: {allocated_vram / 1e9:.2f} GB / {total_vram / 1e9:.2f} GB allocated")
        print(f"GPU VRAM cached: {cached_vram / 1e9:.2f} GB / {total_vram / 1e9:.2f} GB cached")
    else:
        print("No GPU detected.")

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

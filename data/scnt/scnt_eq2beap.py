# This is the code to prepare the FOV images from the scannet fisheye images.

import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
# from utils.graphics_utils import focal2fov
# import torch
import shutil

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_elliptical_mask_bool(height, width, scale=1.0):
    Y, X = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    a, b = width / 2, height / 2  # semi-major and semi-minor axes

    a*= scale  # scale the axes to create an elliptical mask
    b*= scale  # scale the axes to create an elliptical mask

    mask = ((X - center_x)**2) / (a**2) + ((Y - center_y)**2) / (b**2) <= 1
    return mask  # dtype=bool, shape=(H,W)


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(np.float64, elems[4:])))
    return camera_id, model, width, height, params

def fov2tan(fovx, fovy, interval):
    theta_arr = np.arange(interval / 2, fovx, interval)
    theta_arr = np.sort(np.concatenate((-theta_arr, theta_arr)))
    phi_arr = np.arange(interval / 2, fovy, interval)
    phi_arr = np.sort(np.concatenate((-phi_arr, phi_arr)))

    sin_t = np.sin(theta_arr)
    cos_t = np.cos(theta_arr)
    sin_p = np.sin(phi_arr)[:, None]
    cos_p = np.cos(phi_arr)[:, None]
    tan_t = sin_t / cos_t
    tan_p = sin_p / cos_p
    return tan_t, tan_p

    # r = ((sin_t**2)*(cos_p**2)+(cos_t**2)*(sin_p**2)+(cos_t**2)*(cos_p**2))**0.5
    # x = (sin_t * cos_p) / r
    # y = (cos_t * sin_p) / r
    # z = (cos_t * cos_p) / r
    # ray = torch.cat((x[...,None], y[...,None], z[...,None]), dim=-1).flatten(0,-2)

    # return ray, theta_arr, phi_arr

def focal2halffov2(focal, pixels):
    return pixels / 2 / focal

def prepare_sibr_cfg(args):
    root_dir = args.path
    cam_root_dir = args.cam_path
    if cam_root_dir is None:
        cam_root_dir = root_dir
    sibr_cfg_dir = Path(cam_root_dir) / "colmap" / "stereo" / "sparse"

    cameras_txt = os.path.join(cam_root_dir, "colmap", "cameras.txt")
    images_txt = os.path.join(cam_root_dir, "colmap", "images.txt")
    points_txt = os.path.join(cam_root_dir, "colmap", "points3D.txt")
    cameras_fish_txt = os.path.join(cam_root_dir, "colmap", "cameras_fish.txt")

    if not os.path.exists(cameras_txt):
        raise FileNotFoundError(f"Source file not found: {cameras_txt}")
    if not os.path.exists(cameras_fish_txt):
        shutil.copy2(cameras_txt, cameras_fish_txt)

    # Replace "OPENCV_FISHEYE" with "OPENCV" in cameras.txt
    with open(cameras_txt, "r") as f:
        content = f.read().replace("OPENCV_FISHEYE", "OPENCV")
    # Save the modified version in the same directory
    with open(cameras_txt, "w") as f:
        f.write(content)
    print("Replace 'OPENCV_FISHEYE' with 'OPENCV' in cameras.txt")
    
    # Ensure sibr_cfg_dir exists
    sibr_cfg_dir.mkdir(parents=True, exist_ok=True)

    # Copy cameras.txt to sibr_cfg_dir
    shutil.copy2(cameras_txt, sibr_cfg_dir / "cameras.txt")
    shutil.copy2(images_txt, sibr_cfg_dir / "images.txt")
    shutil.copy2(points_txt, sibr_cfg_dir / "points3D.txt")
    #shutil.copy2(points_ply, sibr_cfg_dir / "points3D.ply")
    print(f"Prepare directory: {sibr_cfg_dir} for sibr online rendering.\n")

def colmap_main(args):
    root_dir = args.path
    cam_root_dir = args.cam_path
    if cam_root_dir is None:
        cam_root_dir = root_dir
    camera_dir = Path(cam_root_dir) / "colmap" / "cameras_fish.txt"
    input_image_dir = Path(root_dir) / args.src
    out_image_dir = Path(root_dir) / args.dst
    
    _, _, width, height, params = read_intrinsics_text(camera_dir)

    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]

    FoVx = min(focal2halffov2(fx, width) * args.fov_mod, np.pi / 2)
    FoVy = min(focal2halffov2(fy, height) * args.fov_mod, np.pi / 2)
    print("FOVx in deg: ", 2 * FoVx * 180 / np.pi)
    print("FOVy in deg: ", 2 * FoVy * 180 / np.pi)
    tan_theta, tan_phi = fov2tan(FoVx, FoVy, args.step)
    
    distortion_params = params[4:]
    kk = distortion_params
    
    frames = sorted(os.listdir(input_image_dir))
    radius = np.sqrt(tan_theta ** 2 + tan_phi ** 2)
    theta = np.arctan(radius)
    r = theta * (1.0 + kk[0] * theta**2 + kk[1] * theta**4 + kk[2] * theta**6 + kk[3] * theta**8)
    u = tan_theta * r * fx / radius + cx
    v = tan_phi * r * fy / radius + cy
    u_mask = np.logical_and(u >= 0, u < width)
    v_mask =  np.logical_and(v >= 0, v < height) 
    valid_mask = u_mask & v_mask
    u, v = u.astype(np.float32), v.astype(np.float32)

    #remap ego mask
    if args.add_ego_mask:
        ego_mask = generate_elliptical_mask_bool(height, width).astype(np.float32)
        ego_mask = cv2.remap(ego_mask, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        valid_mask = valid_mask & (ego_mask > 0.5)

    print("mask covered percentage: ", valid_mask.sum() / (np.ones_like(valid_mask)).sum())

    if valid_mask is not None:
        mask_output_path = Path(out_image_dir) / args.mask_dst
        mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_output_path), valid_mask.astype(np.uint8) * 255)
        print("Save mask to:", mask_output_path, "with shape: ", valid_mask.shape)
    else:
        print("Warning: valid_mask is None")

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame
        image = cv2.imread(str(image_path))

        FOV_image = cv2.remap(
            image,
            u,
            v,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        out_image_path = Path(out_image_dir) / frame
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        FOV_image = FOV_image * (valid_mask[:,:,None])
        FOV_image = FOV_image.astype(np.uint8)
        cv2.imwrite(str(out_image_path), FOV_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default="/media/scannetpp/0a5c013435/dslr/")
    parser.add_argument('--cam_path', type=str, default=None)
    parser.add_argument('--src', type=str, default="resized_images")
    parser.add_argument('--dst', type=str, default="undistorted_fovmaps")
    parser.add_argument('--mask_dst', type=str, default=None)
    parser.add_argument('--add_ego_mask', action='store_true', help="Whether to add an ego mask")
    parser.add_argument('--step', type=float, default=None)
    parser.add_argument('--fov_mod', type=float, default=None)
    args = parser.parse_args()
    prepare_sibr_cfg(args)
    colmap_main(args)
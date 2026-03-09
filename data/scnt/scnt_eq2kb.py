import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from scene.colmap_loader import read_intrinsics_binary


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
                params = np.array(tuple(map(float, elems[4:])))
    return camera_id, model, width, height, params

def focal2halffov2(focal, pixels):
    return pixels / 2 / focal

def colmap_main(args):
    root_dir = args.path
    camera_dir = Path(root_dir) / "colmap" / "cameras_fish.txt"
    input_image_dir = args.src
    out_image_dir = args.dst
    
    _, _, width, height, params = read_intrinsics_text(camera_dir)
    print(params)
    
    # adjust fx, fy, cx, cy by the actual image size
    if args.r == -1:
        ratio = 1.0
    else:
        ratio = 1 / args.r
    
    fx = params[0] * ratio
    fy = params[1] * ratio

    FoVx = min(focal2halffov2(fx, width) * args.fov_mod, np.pi / 2)
    FoVy = min(focal2halffov2(fy, height) * args.fov_mod, np.pi / 2)
    print("FOVx in deg: ", 2 * FoVx * 180 / np.pi)
    print("FOVy in deg: ", 2 * FoVy * 180 / np.pi)

    width = int(width * ratio)
    height = int(height * ratio)
    
    # Use prepared fisheye grid map by DAC https://github.com/yuliangguo/depth_any_camera
    try:
        grid_map_file = Path(args.path) / "grid_fisheye.npy"
        grid_fisheye = np.load(grid_map_file)
        print("grid map file: ", grid_map_file)
    except:
        if args.gridmap_restrict:
            raise ValueError("Grid map restrict is not supported")
        else:
            grid_fisheye = np.load("./gridmap/scannetpp/grid_fisheye.npy")
            print("WARNING: Grid map file may not match with the camera intrinsic;", grid_map_file)

    grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (width, height), interpolation=cv2.INTER_NEAREST)
    grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (width, height))
    grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
    
    # Reverse warping
    reverse_mapx = np.zeros((width, height), dtype=np.float32)
    reverse_mapy = np.zeros((width, height), dtype=np.float32)
    # More exact reverse warping using grid_fisheye
    for i in tqdm(range(0, width), desc="calculate_reverse_maps"):
        for j in range(0, height):
            X_c = grid_fisheye[j, i, 0]
            Y_c = grid_fisheye[j, i, 1]
            Z_c = grid_fisheye[j, i, 2]
            tan_theta = X_c / (Z_c + 1e-9)
            tan_phi = Y_c / (Z_c + 1e-9)
            
            theta = np.arctan(tan_theta)
            phi = np.arctan(tan_phi)
            
            x2 = theta / args.step + (FoVx // args.step)
            y2 = phi / args.step + (FoVy // args.step)
            reverse_mapx[i, j] = x2
            reverse_mapy[i, j] = y2
    frames = os.listdir(input_image_dir)

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame
        undistorted_image = cv2.imread(str(image_path))
        
        reversed_image = cv2.remap(
            undistorted_image,
            reverse_mapx.T,
            reverse_mapy.T,
            interpolation=cv2.INTER_LINEAR,
            # interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            # borderMode=cv2.BORDER_REFLECT_101,
            borderValue=(0, 0, 0)
        )
        reversed_image_path = Path(out_image_dir) / frame
        reversed_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(reversed_image_path), reversed_image)

    dummy_image = np.ones_like(undistorted_image)
    mask = cv2.remap(
        dummy_image,
        reverse_mapx.T,
        reverse_mapy.T,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    print("mask covered percentage: ", mask.sum() / (np.ones_like(mask)).sum())
    cv2.imwrite(str(Path(out_image_dir) / "mask.png"), mask * 255)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-r', type=int, default=-1)

    parser.add_argument('--path', type=str, default="/media/scannetpp/demo/0a5c013435/dslr/")
    parser.add_argument('--src', type=str, default="undistorted_fovmaps")
    parser.add_argument('--dst', type=str, default="remapped_fisheye")
    parser.add_argument('--step', type=float, default=2e-3)
    parser.add_argument('--fov_mod', type=float, default=1.3)
    parser.add_argument('--gridmap_restrict', action='store_true', default=False)
    args = parser.parse_args()
    colmap_main(args)
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from scene.colmap_loader import read_next_bytes, CAMERA_MODEL_IDS


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


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        camera_properties = read_next_bytes(
            fid, num_bytes=24, format_char_sequence="iiQQ")
        camera_id = camera_properties[0]
        model_id = camera_properties[1]
        width = camera_properties[2]
        height = camera_properties[3]
        num_params = CAMERA_MODEL_IDS[model_id].num_params
        params = read_next_bytes(fid, num_bytes=8*num_params,
                                    format_char_sequence="d"*num_params)
    return camera_id, model_id, width, height, params

def focal2halffov2(focal, pixels):
    return pixels / 2 / focal

def colmap_main(args):
    root_dir = args.path
    camera_dir = Path(root_dir) / args.camera_config

    if os.path.exists(camera_dir) and args.camera_config.endswith(".txt"):
        _, _, width, height, params = read_intrinsics_text(camera_dir)
        print(params)
    elif os.path.exists(camera_dir) and args.camera_config.endswith(".bin"):
        _, _, width, height, params = read_intrinsics_binary(camera_dir)
        print(params)
    else:
        raise ValueError("Camera intrinsics file not found")

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

    # grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (width, height), interpolation=cv2.INTER_NEAREST)
    grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (width, height))
    np.save(Path(args.path) / 'raymap_fisheye.npy', grid_fisheye) # only kb ray

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-r', type=int, default=-1)

    parser.add_argument('--path', type=str, default="/media/scannetpp/demo/0a5c013435/dslr/")
    parser.add_argument('--camera_config', type=str, default="colmap/cameras_fish.txt")
    parser.add_argument('--step', type=float, default=2e-3)
    parser.add_argument('--fov_mod', type=float, default=1.3)
    parser.add_argument('--gridmap_restrict', action='store_true', default=False)
    args = parser.parse_args()
    colmap_main(args)
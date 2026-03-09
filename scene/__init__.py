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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_camInfos_fisheye

def check_colmap(args):
    return os.path.exists(os.path.join(args.source_path, 'sparse/0'))

def check_blender(args):
    return os.path.exists(os.path.join(args.source_path, "transforms.json")) or os.path.exists(os.path.join(args.source_path, "transforms_train.json"))

def check_mvl(args):
    return os.path.exists(os.path.join(args.source_path, "img"))

def check_scannetpp(args):
    return os.path.exists(os.path.join(args.source_path, 'resized_images')) or os.path.exists(os.path.join(args.source_path, 'images'))

def dataset_selector(args):
    dataset = args.dataset
    if check_scannetpp(args) and (dataset == "AUTO" or dataset == "SCANNETPP"):
        return "Scannetpp"
    if check_colmap(args) and (dataset == "AUTO" or dataset == "COLMAP"):
        return "Colmap"
    if check_blender(args) and (dataset == "AUTO" or dataset == "BLENDER"):
        return "Blender"
    if check_scannetpp(args) and (dataset == "AUTO" or dataset == "MVL"):
        return "Mvl"
    assert False, "Could not recognize scene type!"


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], skip_train_cameras=False, skip_test_cameras=False):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        dataset = dataset_selector(args)
        # for ray-splatting
        scene_info = sceneLoadTypeCallbacks[dataset](args)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            if not skip_train_cameras:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos_fisheye(scene_info.train_cameras, resolution_scale, False, False, args)
            if not skip_test_cameras:
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos_fisheye(scene_info.test_cameras, resolution_scale, False, True, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

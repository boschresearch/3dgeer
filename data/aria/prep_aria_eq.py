"""
Extract frames from vrs, read camera intrinsics and extrinsics from mps (Meta internal SLAM system),
and read out 3D points from mps point cloud. The processing logic is adapted from nerfstudio & project aria demo

In this way, we can use COLMAP GUI to visualize the results
"""
# watch out for potentiailly different world frame conventions
# ref: https://github.com/facebookresearch/projectaria_tools/issues/157 (no it is not applicable)

from pathlib import Path
import shutil
import numpy as np
from projectaria_tools.core.data_provider import create_vrs_data_provider
# from projectaria_tools.core.image import InterpolationMethod
# from projectaria_tools.core import calibration
import projectaria_tools.core as aria_core
from projectaria_tools.core.mps.utils import get_nearest_pose, filter_points_from_confidence
import pycolmap
from PIL import Image
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from tqdm import tqdm
# To establish the mapping between mps pose and vrs recording,
# from mps to vrs: provider.get_image_data_by_time_ns
# from vrs to mps: projectaria_tools.core.mps.utils.get_nearest_pose (binary search)


def get_pcd_from_mps(mps_data_dir):
    points_path = mps_data_dir / "global_points.csv.gz"
    if not points_path.exists():
        # MPS point cloud output was renamed in Aria's December 4th, 2023 update.
        # https://facebookresearch.github.io/projectaria_tools/docs/ARK/sw_release_notes#project-aria-updates-aria-mobile-app-v140-and-changes-to-mps
        points_path = mps_data_dir / "semidense_points.csv.gz"

    # read point cloud
    points_data = aria_core.mps.read_global_point_cloud(str(points_path))
    points_data = filter_points_from_confidence(
        points_data)

    return np.array([point.position_world for point in points_data])


def get_posed_images(provider, mps_traj, stream_id):
    """
    Generator that yields (image_data, extrinsics) tuples for each image in the provider,
    where extrinsics is a 4x4 matrix from get_nearest_pose.
    """
    # vrs to mps
    start_idx = 0  # 250
    num_data = provider.get_num_data(stream_id)

    cnt = 0
    for idx in range(start_idx, num_data):

        image_data = provider.get_image_data_by_index(
            stream_id, idx)
        pose = get_nearest_pose(
            mps_traj, image_data[1].capture_timestamp_ns)
        if pose is None:
            continue
        else:
            cnt += 1
            if cnt > 300:
                break
        # this is Pose of the device coordinate frame in world frame
        pose_mat = pose.transform_world_device.to_matrix()
        # there is extra device to rgb camera transformation

        # extrinsics is the inverse of pose, but we want to avoid the matrix inversion
        # as it has special close-form solution

        # FIXME: currently, there seems to be a constant offset. I suppose it is caused by the difference between device coordinate and the rgb camera, which we don't account for
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = pose_mat[:3, :3].T
        extrinsics[:3, 3] = -pose_mat[:3, :3].T @ pose_mat[:3, 3]

        # R_z = np.array([
        #     [0, 1, 0],
        #     [-1, 0, 0],
        #     [0, 0, 1]
        # ], dtype=np.float32)

        # # if I rotate the image, should I apply the rotation to the extrinsics?
        # extrinsics[:3, :3] = R_z @ extrinsics[:3, :3]
        # extrinsics[:3, 3] = R_z @ extrinsics[:3, 3]

        yield image_data, extrinsics

    # mps to vrs
    # mps pose is at IMU rate, we need to subsample
    # for timed_pose in mps_traj:
    #     timestamp_ns = int(timed_pose.tracking_timestamp.total_seconds() * 1e9)
    #     extrinsics = timed_pose.transform_world_device.to_matrix()
    #     image_data = provider.get_image_data_by_time_ns(
    #         stream_id, timestamp_ns,
    #         TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
    #     yield image_data, extrinsics


def main(vrs_file: Path, mps_data_dir: Path, output_dir: Path, force_delete=True, start_end_idx=None):
    if force_delete and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = create_vrs_data_provider(str(vrs_file.absolute()))
    assert provider is not None, "Cannot open file"

    mps_traj_file = mps_data_dir / "closed_loop_trajectory.csv"
    mps_traj = aria_core.mps.read_closed_loop_trajectory(str(mps_traj_file))

    # turn on color correction to fix overly blue bug
    provider.set_color_correction(True)
    provider.set_devignetting(False)

    # focus on RGB camera images in the VRS
    sensor_name = "camera-rgb"
    stream_id = provider.get_stream_id_from_label(sensor_name)

    # get camera calibration for intrinsics
    rgb_calib = provider.get_device_calibration().get_camera_calib(sensor_name)

    # set our target undistorted intrinsics
    tgt_img_size = 1440
    tgt_focal = 750.0  # focal length in pixels
    dst_calib = aria_core.calibration.get_linear_camera_calibration(
        tgt_img_size, tgt_img_size, tgt_focal, sensor_name + "-undistorted")
    
    dst_calib2 = aria_core.calibration.get_spherical_camera_calibration(
        tgt_img_size, tgt_img_size, tgt_focal, sensor_name + "-undistorted")

    rec = pycolmap.Reconstruction()
    # TODO: add as COLMAP camera
    colmap_cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=tgt_img_size, height=tgt_img_size,
        params=np.array([tgt_focal, tgt_img_size / 2, tgt_img_size / 2]),
        camera_id=0,
    )
    rec.add_camera(colmap_cam)

    # Directly adding 3D points from MPS to COLMAP is not ideal:
    # First, we have too many points, as it is for the whole scene, but only a subset are visible in the images.
    # Second, there is no color on points, which is a problem for 3DGS initialization
    # TODO: after camera pos is added, we add 3D points, and figure out the visibility by projection,
    # and take the color as the mean color of visible pixels
    pcd = get_pcd_from_mps(mps_data_dir)
    pcd_is_visible = np.zeros(len(pcd), dtype=bool)
    pcd_colors = np.zeros((len(pcd), 3), dtype=np.uint8)
    # for point in pcd:
    #     rec.add_point3D(point.position_world,
    #                     pycolmap.Track())

    # if start_end_idx is not None:
    #     image_range = range(start_end_idx[0], start_end_idx[1])
    # else:
    #     first_pose_time_ns = mps_traj[0].tracking_timestamp.total_seconds() * 1e9
    #     options = provider.get_default_deliver_queued_options()
    #     options.set_truncate_first_device_time_ns(first_pose_time_ns)
    #     options.deactivate_stream_all()
    #     provider.
    #     num_data = provider.get_num_data(stream_id)
    #     image_range = range(num_data)

    img_save_dir = output_dir / "resized_images"
    img_save_dir.mkdir(exist_ok=True)
    img_names = []
    # for idx in image_range:
    #     image_data, image_data_record = provider.get_image_data_by_index(
    #         stream_id, idx)
    #   # get camera extrinsics
    # extrinsics = get_nearest_pose(
    # mps_traj, image_data_record.capture_timestamp_ns).transform_world_device.to_matrix()

    for idx, (image_data, extrinsics_device) in tqdm(enumerate(get_posed_images(provider, mps_traj, stream_id))):
        # undistort the image
        rectified_array = aria_core.calibration.distort_by_calibration(
            image_data[0].to_numpy_array(),
            dst_calib, rgb_calib,
            aria_core.image.InterpolationMethod.BILINEAR)

        undistorted_array = aria_core.calibration.distort_by_calibration(
            image_data[0].to_numpy_array(),
            dst_calib2, rgb_calib,
            aria_core.image.InterpolationMethod.BILINEAR)
        
        raw_array = image_data[0].to_numpy_array()


        # extrinsics w.r.t device, need to further transform to current camera
        extrinsics = np.linalg.inv(
            rgb_calib.get_transform_device_camera().to_matrix()) @ extrinsics_device

        # save the image with patterned name, so it can match the entry in database
        # img_name = f"{image_data[1].capture_timestamp_ns}.jpg"
        img_name = f"{idx:06d}.jpg"
        img_path = img_save_dir / img_name
        # Image.fromarray(rectified_array).save(img_path)
        # Image.fromarray(np.rot90(raw_array, k=3)).save(img_path)
        # Image.fromarray(raw_array).save(img_path)
        Image.fromarray(undistorted_array).save(img_path)
        img_names.append(img_name)

        # TODO: add as COLMAP image
        # notice: COLMAP image focuses on the "camera pose + 2D keypoints", not raw image data,
        # as it assumes the image has been processed by feature extractor
        colmap_im = pycolmap.Image(
            id=idx,
            name=img_name,
            camera_id=colmap_cam.camera_id,
            cam_from_world=pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d(extrinsics[:3, :3]),
                translation=extrinsics[:3, 3]
            )
        )
        colmap_im.registered = True
        # p2d = colmap_cam.img_from_cam(
        #     colmap_im.cam_from_world * [p.xyz for p in rec.points3D.values()]
        # )
        # p2d_obs = np.array(p2d)
        # colmap_im.points2D = pycolmap.ListPoint2D(
        #     [pycolmap.Point2D(p, id_) for p, id_ in zip(p2d_obs, rec.points3D)]
        # )
        rec.add_image(colmap_im)

        # evaluate pcd visibility at this frame
        # pcd_2d = colmap_im.project_point(pcd)
        rot = colmap_im.cam_from_world.rotation.matrix()
        trans = colmap_im.cam_from_world.translation
        pcd_cam = np.einsum('ij, kj->ki', rot, pcd) + trans
        pcd_2d = colmap_cam.img_from_cam(pcd_cam)

        pcd_is_visible_this_frame = (pcd_2d[:, 0] >= 0) & (pcd_2d[:, 0] < tgt_img_size) & \
            (pcd_2d[:, 1] >= 0) & (pcd_2d[:, 1] < tgt_img_size)
        # only consider points in front of the camera
        pcd_is_visible_this_frame &= pcd_cam[:, 2] > 0

        pcd_is_visible |= pcd_is_visible_this_frame

        pcd_2d_xy = pcd_2d[pcd_is_visible_this_frame].astype(np.int32)
        pcd_colors[pcd_is_visible_this_frame] = rectified_array[pcd_2d_xy[:, 1], pcd_2d_xy[:, 0]]
        # if pcd_colors[pcd_is_visible_this_frame].sum() == 0:
        #     pcd_colors[pcd_is_visible_this_frame] = rectified_array[pcd_2d_xy[:,
        #                                                                       1], pcd_2d_xy[:, 0]]
        # else:
        #     # accumulate color
        #     pcd_colors[pcd_is_visible_this_frame] += rectified_array[pcd_2d_xy[:, 1], pcd_2d_xy[:, 0]]
        #     # average color
        #     pcd_colors[pcd_is_visible_this_frame] //= 2
    # collect visible points, and add them to COLMAP
    print('adding visible points to COLMAP...')
    pcd_visible = pcd[pcd_is_visible]
    pcd_colors_visible = pcd_colors[pcd_is_visible].clip(
        0, 255).astype(np.uint8)
    print(f'Found {len(pcd_visible)} visible points in total.')
    for point, color in zip(pcd_visible, pcd_colors_visible):
        rec.add_point3D(point, pycolmap.Track(), color)

    # create visibility relation for all visible points
    # register point to each image
    print('registering points to images...')
    for image in rec.images.values():
        # p2d = colmap_cam.img_from_cam(
        #     image.cam_from_world * [p.xyz for p in rec.points3D.values()]
        # )
        # p2d_obs = np.array(p2d)
        rot = image.cam_from_world.rotation.matrix()
        trans = image.cam_from_world.translation
        pcd_cam = np.einsum('ij, kj->ki', rot, pcd_visible) + trans
        pcd_2d = colmap_cam.img_from_cam(pcd_cam)
        pcd_is_visible_this_frame = (pcd_2d[:, 0] >= 0) & (pcd_2d[:, 0] < tgt_img_size) & \
            (pcd_2d[:, 1] >= 0) & (pcd_2d[:, 1] < tgt_img_size)
        p2d_obs = pcd_2d[pcd_is_visible_this_frame]
        image.points2D = pycolmap.ListPoint2D(
            [pycolmap.Point2D(p, id_) for p, id_ in zip(
                p2d_obs, range(len(pcd_visible)))]
        )

    # output_model_dir = output_dir / "sparse/0"
    output_model_dir = output_dir / "colmap"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_model_dir / "database.db"
    if database_path.exists():
        database_path.unlink()
    database_path.touch()
    # create COLMAP database
    database = pycolmap.Database(str(database_path))
    database.write_camera(colmap_cam, use_camera_id=True)
    database.close()
    pycolmap.import_images(
        database_path=str(database_path),
        image_path=str(img_save_dir),
        image_list=img_names,
        options=pycolmap.ImageReaderOptions(
            existing_camera_id=0
        )
    )

    # optional: we can add 2D observations of these points as well

    # save to COLMAP results
    rec.write(str(output_model_dir))


if __name__ == "__main__":
    import tyro
    tyro.cli(main, description="Convert MPS data to COLMAP format for visualization.")
# Adapted from Lyft Dataset SDK dev-kit
# Licensed under the Creative Commons

#TODO: Probably need to adapt to torch version (latter part)
#TODO: Need to compare execution speed
#TODO: Change the function comment

import time
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from lyft_dataset_sdk.utils.data_classes import PointCloud, LidarPointCloud, RadarPointCloud  # NOQA
from lyft_dataset_sdk.utils.geometry_utils import view_points  # NOQA


def map_pc_to_image(lyft_data,
                    pointsensor_token: str, 
                    camera_token: str = None,
                    get_ego=False,
                    get_world=False) -> LidarPointCloud:
    """Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to
    the image plane.

    Args:
        pointsensor_token: Lidar/radar sample_data token.
        camera_token: Camera sample_data token.

    Returns: tuple of
        pointcloud <np.float: 2, n)>
        coloring <np.float: n>, image <Image>
        
    """
    
    pointsensor = lyft_data.get("sample_data", pointsensor_token)
    pcl_path = lyft_data.data_path / pointsensor["filename"]
    if pointsensor["sensor_modality"] == "lidar":
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = lyft_data.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))
    pc_ego = pc.points.copy()
    if get_ego: return pc

    # Second step: transform to the global frame.
    poserecord = lyft_data.get("ego_pose", pointsensor["ego_pose_token"])
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc.translate(np.array(poserecord["translation"]))
    if get_world: return pc

    # Obtain image
    assert camera_token is not None, "Must specify a camera token"
    cam = lyft_data.get("sample_data", camera_token)
    im = Image.open(str(lyft_data.data_path / cam["filename"]))
    
    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = lyft_data.get("ego_pose", cam["ego_pose_token"])
    pc.translate(-np.array(poserecord["translation"]))
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = lyft_data.get("calibrated_sensor", cam["calibrated_sensor_token"])
    pc.translate(-np.array(cs_record["translation"]))
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
   
    # Map the image points to the lidar points.
    idx = points.round().astype(np.int)
    col_idx = idx[0,:]
    row_idx = idx[1,:]
    image = np.asarray(im)
    points_rgb = image[row_idx, col_idx].T
    return np.vstack((pc_ego[:, mask], points_rgb))


if __name__ == "__main__":
    from lyft_dataset_sdk.lyftdataset import LyftDataset
    from config import cfg
    cfg = cfg.data
    lyft_data = LyftDataset(data_path=cfg.lyft,
                            json_path=cfg.train_path,
                            verbose=False)
    one_scene = lyft_data.scene[0]
    first_sample_token = one_scene["first_sample_token"]
    sample = lyft_data.get('sample', first_sample_token)
    sample_data = sample['data']
    lidar_top_channel = 'LIDAR_FRONT_RIGHT'
    lidar_token = sample_data[lidar_top_channel]
    lidar_data = lyft_data.get('sample_data', lidar_token)
    cs_lidar = lyft_data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    cam_front_channel = 'CAM_FRONT_ZOOMED'
    cam_token = sample_data[cam_front_channel]
    cam_data = lyft_data.get("sample_data", cam_token)
    cs_cam = lyft_data.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    points, coloring, im, mask = map_pc_to_image(lyft_data, lidar_token, cam_token)
    pass
__author__ = 'Eralien'
# Copyright 2019 Siyuan Feng
# Date: Oct 19

# TODO: the eval-enabled dataloader

from config import cfg
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import spconv
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from data.data_process import map_pc_to_image
from pyquaternion import Quaternion
from spconv.utils import VoxelGeneratorV2

from utils import timeit


@timeit
def get_datasets(**kwargs):
    """ Wrapper to get train, valid datasets.

    Notice for test datasets, it is in a different path which hasn't been implemented
    in this function yet.
    """
    fullset = CarlaokDataset()
    dataset_length = CarlaokDataset.get_dataset_len()
    split_length = [int(dataset_length*ratio) for ratio in cfg.data.split_ratio]
    split_length[-1] += dataset_length - sum(split_length)
    return random_split(fullset, split_length)


class CarlaokDataset(Dataset):
    """ Lyft Dataset
    Using lyft dataset sdk to generate the trainable data

    Attributes:
        __init__: ...
        __getitem__: ...

    """
    # The lyft data sdk
    lyft_data = LyftDataset(data_path=cfg.data.lyft,
                            json_path=cfg.data.train_path,
                            verbose=False)

    classes = {}
    for i, one_class in enumerate(lyft_data.category, 1):
        classes[one_class['name']] = i

    # The voxel generator
    voxel_generator = VoxelGeneratorV2(cfg.voxel.voxel_size,
                                       cfg.voxel.range,
                                       cfg.voxel.max_num)
    grid_size = voxel_generator.grid_size

    # Split the data into train, test and validation when initialized
    datacfg = cfg.data
    type_name = ["train", "valid", "test"]


    def __init__(self, device: str = None):
        """ Initalize the dataset list using lyft samples

        Args:
            device: cpu or gpu
            validation: true if to validate or test

        Returns:
            CarlaokDataset: torch Dataset instance
        """
        datacfg_ = CarlaokDataset.datacfg
        self.dft_lidar_channel = datacfg_.default_lidar_channel
        self.aux_lidar_channel = datacfg_.auxiliary_lidar_channel
        self.use_all_lidar = datacfg_.all_lidar
        self.use_cam = datacfg_.use_cam
        if self.use_cam: self.cam_channel = datacfg_.cam_channel
        self.device = cfg.device if device is None else device


    def __getitem__(self, idx) -> torch.Tensor:
        """ Return the selected data from the lyft dataset

        Args:
            bla

        Returns:
            bla

        """
        sample = CarlaokDataset.lyft_data.sample[idx]
        sensor_token = sample['data']
        dft_lidar_token = sensor_token[self.dft_lidar_channel]
        dft_pc = self.get_lidar_ego(dft_lidar_token)    # Calibrate the lidar
        bboxes = CarlaokDataset.lyft_data.get_sample_data(dft_lidar_token, flat_vehicle_coordinates=True)[1]


        # Calibrate and append other lidars to the TOP lidar
        if self.use_all_lidar:
            aux_lidar_token = []
            for channel in self.aux_lidar_channel:
                if sensor_token.get(channel) is not None:
                    aux_lidar_token.append(sensor_token[channel])
            for token in aux_lidar_token:
                dft_lidar_data = CarlaokDataset.lyft_data.get('sample_data', dft_lidar_token)
                aux_pc = self.map_pc_to_default(token, dft_lidar_data)
                dft_pc.points = np.hstack([dft_pc.points, aux_pc.points])
                # print("aux pc: ", aux_pc.points.shape)
        assert dft_pc.points.shape[0] == 4, "The raw lidar data has 4 channels"

        # Use camera information
        if self.use_cam:
            pc_rgb = np.array([]).reshape(7,-1)     # should be xyzirgb, 7 dims
            cam_token = [sensor_token[channel] for channel in self.cam_channel]
            for token in cam_token:
                one_pc_rgb = self.map_pc_to_image(dft_lidar_token, token)
                pc_rgb = np.hstack([pc_rgb, one_pc_rgb])
                # print("rgb: ", one_pc_rgb.shape)
            gt_pc = pc_rgb.T
            del pc_rgb
        else:
            gt_pc = dft_pc.points.T
            del dft_pc

        if cfg.multi_GPU is True:
            voxels = CarlaokDataset.voxel_generator.generate_multi_gpu(gt_pc)
        else:
            voxels = CarlaokDataset.voxel_generator.generate(gt_pc)

        # Make the np raw voxels to torch
        voxels = self.voxel_process(voxels)
        voxels = self.builtin_simplevoxel(voxels)
        voxels["bboxes"] = self.bboxes_generator(bboxes)
        return voxels


    def __len__(self):
        return len(CarlaokDataset.lyft_data.sample)


    def map_pc_to_image(self, pointsensor_token: str, cam_token: str) -> np.ndarray:
        """Map point cloud to the camera coorditions.

        The lidar token and cam token all refer to the sensor coordination.
        Lidar pose -> Ego pose (at the lidar moment) -> World pose ->
        Ego pose (at the camera moment) -> Camera pose

        Return:
            np.ndarray (N, 3) of the information of cam placed as
        """
        return map_pc_to_image(CarlaokDataset.lyft_data, pointsensor_token, cam_token)


    def map_pc_to_default(self, lidar_token: str, default_lidar_data: str) -> LidarPointCloud:
        pc = self.get_lidar_world(lidar_token)
        poserecord = CarlaokDataset.lyft_data.get("ego_pose", default_lidar_data["ego_pose_token"])
        pc.translate(-np.array(poserecord["translation"]))
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)
        return pc


    def get_lidar_ego(self, pointsensor_token: str) -> LidarPointCloud:
        return map_pc_to_image(CarlaokDataset.lyft_data, pointsensor_token, get_ego=True)


    def get_lidar_world(self, pointsensor_token: str) -> LidarPointCloud:
        return map_pc_to_image(CarlaokDataset.lyft_data, pointsensor_token, get_world=True)


    def voxel_process(self, voxels):
        """This function is to provide a quick realization for our task.

        It corporates self.data_transform and self.voxel_convert_to_torch
        (deprecated)

        Arg:
            voxels -> dict. see the "Survey"

        Return:
            voxels: dict
            voxels['voxels']: torch.FloatTensor (N, 15, 7) or (N, 15, 4) if cam not used
            voxels['num_points']: scalar
            voxels['coordinates']: torch.IntTensor (N, 3)
            voxels['voxel_point_mask']: torch.BoolTensor (N, 15, 1)
        """
        voxels['voxels'] = torch.tensor(voxels["voxels"],
                                        dtype=torch.float32,
                                        device=self.device)
        # by using dict.pop, it returns the value and delete the key val pair
        voxels['num_points'] = torch.tensor(voxels.pop("num_points_per_voxel").sum(),
                                            dtype=torch.int32,
                                            device=self.device)
        voxels['coordinates'] = torch.tensor(voxels["coordinates"],
                                             dtype=torch.int32,
                                             device=self.device)
        voxels['voxel_point_mask'] = torch.tensor(voxels["voxel_point_mask"],
                                                  dtype=torch.bool,
                                                  device=self.device)
        return voxels


    def builtin_simplevoxel(self, voxels):
        """ This function realize the simple voxel in VFE

        This funtion avoids burden on computation graph and complexity.
        It get the mean value for each voxel among all the points in it.

        Args:
            voxels: dict
            voxels['voxels']: torch.FloatTensor (N, 15, 7) or (N, 15, 4) if cam not used
            voxels['num_points']: scalar
            voxels['coordinates']: torch.IntTensor (N, 3)
            voxels['voxel_point_mask']: torch.LongTensor (N, 15, 1)

        Return:
            voxels: dict
            voxels['voxels']: torch.FloatTensor (N, 7) or (N, 4) if cam not used
        """
        raw_voxels = voxels['voxels']
        mask = voxels['voxel_point_mask']
        pt_per_voxel = mask.sum(dim=1) # (N,)
        raw_voxels = raw_voxels.masked_fill(mask.logical_not(), 0.).sum(1) / pt_per_voxel
        voxels['voxels'] = raw_voxels
        return voxels


    def bboxes_generator(self, bboxes):
        """ From original lyft-dev-kit Box class to torch.Tensor

        Arg:
            bboxes: Box

        Return:
            torch.FloatTensor (N, 8), N is the number of instances in current scene
            where 8 stands for class, center_xyz, wlh, and angle (yaw, based on flat ground calibrated)
        """
        output = torch.FloatTensor(len(bboxes), 8)
        for i, bbox in enumerate(bboxes):
            class_num = CarlaokDataset.classes[bbox.name]
            output[i, 0] = class_num
            output[i, 1:4] = torch.from_numpy(bbox.center)
            output[i, 4:7] = torch.from_numpy(bbox.wlh)
            output[i, -1] = bbox.orientation.radians
        return output.to(self.device)


    @classmethod
    def get_dataset_len(cls):
        """ Return the length of the dataset """
        return len(cls.lyft_data.sample)


def collate_fn(items):
    """ Make batch tensor for sparse voxel tensor and bbox

    Args:
        items: list[dict]

    Returns:
        spTensor: spconv.SparseConvTensor shaped (B, C, D, H, W)
        bboxes: torch.FloatTensor shaped (B, N, 8), where N is the max number of instances
        in each batch member
    """
    voxels_num = np.array([i['voxel_num'] for i in items])
    voxels_num_sum = voxels_num.sum()
    batch_size = len(items)
    features_raw = [i['voxels'][:,-3:] for i in items]
    features = torch.cat(features_raw)
    assert features.shape[0] == voxels_num_sum, "Concatenation error occurs"
    batch_idx = torch.cat([torch.IntTensor([i]).repeat(k) \
        for i, k in zip(range(batch_size), voxels_num)]).to(features.device)
    idx_raw = torch.cat([i['coordinates'] for i in items])
    indices = torch.cat((batch_idx.view(-1,1), idx_raw), dim=1)
    spatial_shape = cfg.voxel.spatial_shape
    spTensor = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)

    instance_list = [i['bboxes'] for i in items]
    max_instance = max([inst.shape[0] for inst in instance_list])
    bboxes_list = []
    for b in instance_list:
        bboxes_list.append(F.pad(b.view(1,1,*list(b.shape)),
                                 (0,0,0,max_instance-b.shape[0]),
                                 "constant", -1.).squeeze(0))
    bboxes = torch.cat(bboxes_list, dim=0)
    return (spTensor, bboxes)


if __name__ == '__main__':
    import time
    from sys import getsizeof
    train_set, valid_set = get_datasets()
    print(getsizeof(train_set), getsizeof(valid_set))
    del valid_set

    for i in range(len(train_set)):
        start = time.time()
        sample = train_set[i]
        print("time: {:3.2f}".format(time.time() - start))
        print("sample size: ", getsizeof(sample))
    pass

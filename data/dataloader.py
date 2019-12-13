__author__ = 'Eralien'
# Copyright 2019 Siyuan Feng
# Date: Oct 19

# TODO: adapt for eval-enabled dataloader

import os, math
import shutil
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from config import cfg
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from data.data_process import map_pc_to_image
from pyquaternion import Quaternion
from spconv.utils import VoxelGeneratorV2

# from utils.voxel_gen import pointcloud_gen, voxel_gen
# from utils.sp2d import sparse2dense


def get_datasets():
    """ Wrapper to get train, valid, and test datasets """
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
        self.cfg = CarlaokDataset.datacfg
        self.dft_lidar_channel = self.cfg.default_lidar_channel
        self.aux_lidar_channel = self.cfg.auxiliary_lidar_channel
        self.use_all_lidar = self.cfg.all_lidar
        self.use_cam = self.cfg.use_cam
        if self.use_cam: self.cam_channel = self.cfg.cam_channel
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
        boxes = CarlaokDataset.lyft_data.get_boxes(dft_lidar_token) 
        # print("raw pc: ", dft_pc.points.shape)
        
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
        assert dft_pc.points.shape[0] == 4
                
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
        
        return voxels

    
    def __len__(self):
        return len(CarlaokDataset.lyft_data.sample)
        
    
    def map_pc_to_image(self, pointsensor_token: str, cam_token: str) -> np.ndarray:
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
        """This function is to provide quick realization for our task.
        It corporates self.data_transform and self.voxel_convert_to_torch
        (deprecated)

        Input: data -> dict. see the "Survey"
        Output: voxels in torch tensor
        """
        voxels['voxels'] = torch.tensor(voxels["voxels"],
                                        dtype=torch.float32,
                                        device=self.device)
        # by using dict.pop, it returns the value and delete the key val pair
        voxels['num_points'] = torch.tensor(voxels.pop("num_points_per_voxel").sum(),
                                            dtype=torch.int32,
                                            device=self.device)
        voxels['coordinates'] = torch.tensor(voxels["coordinates"],
                                             dtype=torch.int64,
                                             device=self.device)
        voxels['voxel_point_mask'] = torch.tensor(voxels["voxel_point_mask"],
                                                  dtype=torch.long,
                                                  device=self.device)
        return voxels
    
    
    @classmethod
    def get_dataset_len(cls):
        """ Return the length of the dataset """
        return len(cls.lyft_data.sample)
        

def collate_fn(items):
    pass

    
if __name__ == '__main__':
    import os, time
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.CUDA_VISIBLE_DEVICES
    from sys import getsizeof

    train_set, valid_set, test_set = get_datasets()
    print(getsizeof(train_set), getsizeof(valid_set), getsizeof(test_set))
    del valid_set, test_set
    
    for i in range(len(train_set)):
        start = time.time()
        sample = train_set[i]
        print("time: ", time.time() - start)
        print("sample size: ", getsizeof(sample))
    pass
    
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
from torch.utils.data import Dataset, DataLoader
import pickle
from config import cfg
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from utils.data_process import map_pc_to_image
from pyquaternion import Quaternion
# from utils.voxel_gen import pointcloud_gen, voxel_gen
# from utils.sp2d import sparse2dense


class CarlaokDataset(Dataset):
    """ Lyft Dataset
    Using lyft dataset sdk to generate the trainable data
    
    Attributes:
        __init__: ...
        __getitem__: ...
        
    """

    def __init__(self,
                 device: str = None, 
                 validation: bool = False):
        """ Initalize the dataset list using lyft samples
        
        Args: 
            device: cpu or gpu
            validation: true if to validate or test
        
        Returns:
            CarlaokDataset: torch Dataset instance
        """
        self.cfg = cfg.data
        self.lyft_data = LyftDataset(data_path=self.cfg.lyft,
                                     json_path=self.cfg.train_path,
                                     verbose=False)
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
        sample = self.lyft_data.sample[idx]
        sensor_token = sample['data']
        dft_lidar_token = sensor_token[self.dft_lidar_channel]
        dft_pc = self.get_lidar_ego(dft_lidar_token)    # Calibrate the lidar
        print("raw pc: ", dft_pc.points.shape)
        
        # Calibrate and append other lidars to the TOP lidar
        if self.use_all_lidar: 
            aux_lidar_token = []
            for channel in self.aux_lidar_channel:
                if sensor_token.get(channel) is not None:
                    aux_lidar_token.append(sensor_token[channel])
            for token in aux_lidar_token:
                dft_lidar_data = self.lyft_data.get('sample_data', dft_lidar_token)
                aux_pc = self.map_pc_to_default(token, dft_lidar_data)
                dft_pc.points = np.hstack([dft_pc.points, aux_pc.points])
                print("aux pc: ", aux_pc.points.shape)
        assert dft_pc.points.shape[0] == 4
                
        # Use camera information
        if self.use_cam:
            pc_rgb = np.array([]).reshape(7,-1)     # should be xyzirgb, 7 dims
            cam_token = [sensor_token[channel] for channel in self.cam_channel]
            for token in cam_token:
                one_pc_rgb = self.map_pc_to_image(dft_lidar_token, token)
                pc_rgb = np.hstack([pc_rgb, one_pc_rgb])
                print("rgb: ", one_pc_rgb.shape)
            return pc_rgb
        # Otherwise
        return dft_pc.points
    
    
    def __len__(self):
        return len(self.lyft_data.sample)
        
    
    def map_pc_to_image(self, pointsensor_token: str, cam_token: str) -> np.ndarray:
        return map_pc_to_image(self.lyft_data, pointsensor_token, cam_token)
    
    
    def map_pc_to_default(self, lidar_token: str, default_lidar_data: str) -> LidarPointCloud:
        pc = self.get_lidar_world(lidar_token)
        poserecord = self.lyft_data.get("ego_pose", default_lidar_data["ego_pose_token"])
        pc.translate(-np.array(poserecord["translation"]))
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)
        return pc
    
    
    def get_lidar_ego(self, pointsensor_token: str) -> LidarPointCloud:
        return map_pc_to_image(self.lyft_data, pointsensor_token, get_ego=True)
    
    
    def get_lidar_world(self, pointsensor_token: str) -> LidarPointCloud:
        return map_pc_to_image(self.lyft_data, pointsensor_token, get_world=True)
        
    
    # @classmethod
    # def get_dataset_len(cls):
    #     """ Return the length of the dataset """
    #     lyft_data = LyftDataset(data_path=cfg.data.lyft,
    #                             json_path=cfg.data.train_path,
    #                             verbose=False)
    #     return len(lyft_data.sample)
    
    
if __name__ == '__main__':
    import os, time
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.CUDA_VISIBLE_DEVICES

    # print(CarlaokDataset.get_dataset_len())
    train_set = CarlaokDataset()
    
    for i in range(len(train_set)):
        start = time.time()
        sample = train_set[i]
        print("time: ", time.time() - start)
        print("out sample: ", sample.shape)
        pass
    

    pass
    
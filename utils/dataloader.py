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
from utils.data_process import map_pc_to_image
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
                 device: str = 'cpu', 
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
        
        
    def __getitem__(self, idx):
        """ Return the selected data from the lyft dataset

        Args:
            bla
        
        Returns:
            bla

        """
        sample_data = self.lyft_data[idx]['data']
        lidar_token = []
        for channel in self.cfg.lidar_channel:
            lidar_token.append(lyft_data.get('sample_data', sample_data[channel]))

        if self.cfg.render_cam:
            cam_token = []
            for channel in self.cfg.cam_channel:
                cam_token.append(lyft_data)

    
    def __len__(self):
        return len(self.lyft_data.sample)
        
    
    def __map_pc_to_image__(self, pointsensor_token: str, cam_token: str) -> np.ndarray:
        return map_pc_to_image(self.lyft_data, pointsensor_token, cam_token)
    
    
    def __get_lidar_ego__(self, pointsensor_token: str) -> np.ndarray:
        return map_pc_to_image(self.lyft_data, pointsensor_token, get_ego=True)
    
    
    def __get_lidar_world__(self, pointsensor_token: str) -> np.ndarray:
        return map_pc_to_image(self.lyft_data, pointsensor_token, get_world=True)
    
    
    
    
    @staticmethod
    def get_dataset_len():
        """ Return the length of the dataset """
        lyft_data = LyftDataset(data_path=cfg.data.lyft,
                                json_path=cfg.data.train_path,
                                verbose=False)
        return len(lyft_data.sample)
    
    
if __name__ == '__main__':
    print(CarlaokDataset.get_dataset_len())
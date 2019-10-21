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
# from utils.voxel_gen import pointcloud_gen, voxel_gen
# from utils.sp2d import sparse2dense


class VoxelDataset(Dataset):
    """ Point Cloud Dataset
    """

    def __init__(self, device, transform=None, evaluate=False):
        """ Generate the point   
        """
        # Make clear how the evalution goes
        evalcfg = cfg.eval
        self.eval_enable = evalcfg.enable
        if self.eval_enable:
            pass
        
        
    def __getitem__(self, idx):
        """返回输入的数据

        输入：点云原始数据
        输出：包含体素信息的字典

        注意：体素信息有
            1. 体素格子大小
            2. 体素本身的数据信息（详情参见pointnet-segmentation-survey）

        包含方法：
            1. 从点云生成体素
            2. 体素可视化
        """
        pass
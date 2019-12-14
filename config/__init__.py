# ---------------------------------------------------------------------------- #
#          This file stores most of the variables used in this project         #
# ---------------------------------------------------------------------------- #

__author__ = "Siyuan Feng"

#TODO: add dataloader config

import os, glob
from easydict import EasyDict as edict
import numpy as np
from datetime import datetime
import torch

__C = edict()
cfg = __C


# ------------------------------------ GPU ----------------------------------- #

__C.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
__C.CUDA_VISIBLE_DEVICES = "0"                   # Type your available GPUs!!
__C.multi_GPU = False if len(__C.CUDA_VISIBLE_DEVICES) == 1 else True


# ---------------------------------- Dataset --------------------------------- #

__C.data = edict()
__C.data.base = os.path.join(".", "data")
__C.data.lyft = os.path.join(__C.data.base, "lyft")
__C.data.train_path = os.path.join(__C.data.lyft,
                                     "train_data")
__C.data.train_sample = os.path.join(__C.data.lyft,
                                     "train_data",
                                     "calibrated_sensor.json")
__C.data.train_csv = os.path.join(__C.data.lyft, "train.csv")
__C.data.image = os.path.join(__C.data.base, "image/")

# some controlling buttons
__C.data.button = edict()
__C.data.button.LIST_SCENE = False
__C.data.button.LIST_CATEG = False
__C.data.button.REND_SAMPLE = False
__C.data.button.LIST_SAMPLE = False
__C.data.button.REND_PC_IMG = False
__C.data.button.REND_LIDAR_3D = False

# data channel
__C.data.all_lidar = False
__C.data.default_lidar_channel = 'LIDAR_TOP'
__C.data.auxiliary_lidar_channel = ['LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT']
__C.data.use_cam = True
__C.data.cam_channel = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT',
                        'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
                        'CAM_BACK_RIGHT', 'CAM_FRONT_ZOOMED']

# split into train, validate and test
__C.data.split_ratio = [0.8, 0.2]


# --------------------------- Generation of Voxels --------------------------- #

__C.voxel = edict()
__C.voxel.voxel_size: list = [0.25, 0.25, 0.1] # format X, Y, Z (H, W, D)
__C.voxel.range: list = [0, -32, -1, 64, 32, 3]  # format: xyzxyz, minmax
__C.voxel.max_num = 15          # Max number of points contained in a voxel
__C.voxel.render_plot = False
__C.voxel.render_gt = False     # May be very slow if True


# ----------------------------------- Model ---------------------------------- #

__C.model = edict()
__C.model.batch_size = 1
__author__ = "Siyuan Feng"

#TODO: add dataloader config
#TODO: add 

import os, glob
from easydict import EasyDict as edict
import numpy as np
from datetime import datetime

__C = edict()
cfg = __C


""" GPU """
__C.CUDA_VISIBLE_DEVICES = "1"                   # Type your available GPUs!!
__C.multi_GPU = False if len(__C.CUDA_VISIBLE_DEVICES) == 1 else True


""" Data path init """
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
__C.data.lidar = True
__C.data.lidar_channel = ['LIDAR_TOP']
__C.data.render_cam = False
__C.data.cam_channel = ['CAM_FRONT']
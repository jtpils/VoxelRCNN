# The configuration file
import os, glob
from easydict import EasyDict as edict
import numpy as np
from datetime import datetime

__C = edict()
cfg = __C


# Data path init
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
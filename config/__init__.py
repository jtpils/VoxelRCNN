# The configuration file
import os, glob
from easydict import EasyDict as edict
import numpy as np
from datetime import datetime

__C = edict()
cfg = __C

# Data path hardcode
__C.data = edict()
__C.data.base = os.path.join(".", "data", "lyft")
# __C.data.train_data =
__C.data.train_path = os.path.join(__C.data.base, 
                                     "train_data")
__C.data.train_sample = os.path.join(__C.data.base, 
                                     "train_data", 
                                     "calibrated_sensor.json")
__C.data.train_csv = os.path.join(__C.data.base, "train.csv")

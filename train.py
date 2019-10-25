__author__ = "Siyuan Feng"
__email__ = "siyuanf@umich.edu"

# TODO: add BatchNorm3d variation
# TODO: add featrue map distribution map
# TODO: attach BatchNorm3d hook

from config import cfg
import os, time, sys, threading, math
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.CUDA_VISIBLE_DEVICES

import scipy as sp
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from log import setup_logger, GlobalLogger

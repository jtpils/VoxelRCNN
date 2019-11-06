__author__ = "Siyuan Feng"
__email__ = "siyuanf@umich.edu"

# TODO: add BatchNorm3d variation
# TODO: add featrue map distribution map
# TODO: attach BatchNorm3d hook

from config import cfg
import os, time, sys, threading, math
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

import scipy as sp
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from log import setup_logger, GlobalLogger
from utils.dataloader import get_datasets
from 


device = cfg.device


def train():
    # load data
    train_set, _, _ = get_datasets()
    train_loader = DataLoader(train_set,
                              batch_size=cfg.model.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_custom)
    train_loader = DataLoader(train_set,
                              batch_size=cfg.model.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_custom)
    
if __name__ == "__main__":
    pass
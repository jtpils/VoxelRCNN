__author__ = "Siyuan Feng"
__email__ = "siyuanf@umich.edu"

# TODO: add BatchNorm3d variation
# TODO: add featrue map distribution map
# TODO: attach BatchNorm3d hook

from config import cfg

import math
import scipy as sp
import numpy as np
import pandas as pd
import torch
from torch import optim, nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from log import setup_logger, GlobalLogger
from data.dataloader import get_datasets, collate_fn
from src import get_VoxelMaskRCNN

torch.autograd.set_detect_anomaly(True)
device = cfg.device


def main():
    # load data as train_set and validation set
    train_set, _, = get_datasets()
    train_loader = DataLoader(train_set,
                              batch_size=cfg.model.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_fn)

    # Prepare the net
    net = get_VoxelMaskRCNN()
    net_cuda = net.to(device)

    optimizer = optim.SGD(net_cuda.parameters(),
                          lr=cfg.model.SGD_lr,
                          momentum=cfg.model.SGD_momentum)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
        lambda epoch: cfg.model.SGD_lr_decay ** epoch)

    starting_epoch = 0
    for epoch in range(starting_epoch, cfg.model.num_epoches):
        train(epoch, train_loader, net_cuda)


def train(epoch, train_loader, model):
    pbar = tqdm(enumerate(train_loader, 1),
                ascii=True,
                total=math.ceil(len(train_loader.dataset)/train_loader.batch_size))
    for i, data in pbar:
        pbar.set_description("Epoch {}".format(epoch + 1))
        spTensor, bboxes = data
        output = model(spTensor)
        pass

    # Cleanup tqdm
    pbar.close()


if __name__ == "__main__":
    main()
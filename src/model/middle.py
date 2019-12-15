__author__ = 'Siyuan Feng'
# see http://10.10.51.40:3000/shiming.li/pointcloud_segmentation_survey
# for more info!!!

# TODO: Make the network to be easily regulated in cfg files.
# TODO: Make Conv3d

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import cfg
from easydict import EasyDict as edict
from models.convmod import Conv3dMod, Conv3dMod_GCN, Interpolate


bncfg = cfg.model.bn


def middle_module_gen(name="v0", criterion=None):
    if name == "v0":
        return MiddleSiyuan(
            num_input_features=cfg.model.num_input_features,
            name=cfg.model.middle_name,
            criterion=criterion)
    elif name == "v1":
        return MiddleV1(
            num_input_features=cfg.model.num_input_features,
            name=cfg.model.middle_name,
            criterion=criterion)
    elif name == "v2":
        return MiddleV2(
            num_input_features=cfg.model.num_input_features,
            name=cfg.model.middle_name,
            criterion=criterion)
    elif name == "v3":
        return MiddleV3(
            num_input_features=cfg.model.num_input_features,
            name=cfg.model.middle_name,
            criterion=criterion)
    elif name == "sparse":
        return MiddleSparse(
            num_input_features=cfg.model.num_input_features,
            name=cfg.model.middle_name,
            criterion=criterion)


class Loss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, x, target):
        # Compute the loss; Hardcode the permute axis sequence
        loss_output = x.permute(0, 2, 3, 4, 1).contiguous().view(-1,cfg.demo_dataset.class_num)
        loss_target = target.view(-1)

        # loss
        return self.criterion(loss_output, loss_target)


class MiddleSiyuan(nn.Module):
    def __init__(self,
        num_input_features=4,
        name='Middle_v0',
        criterion=None):
        super(MiddleSiyuan, self).__init__()
        self._name = name
        self.num_input_features = num_input_features
        self.loss = Loss(criterion)
        self.middle = nn.Sequential(
            Conv3dMod(16, 32, num_input_features=self.num_input_features),      # conv_mod:0
            Conv3dMod(32, 64, stride=(2,2,1)),                                  # conv_mod:1
            Conv3dMod(64, 64, hidden_layer=4),                                  # conv_mod:2
            Conv3dMod(64, 64, stride=1, hidden_layer=4))                        # conv_mod:3
        self.upsample = nn.Sequential(
            Interpolate(size=(64,64,20), mode='trilinear'),
            Interpolate(size=(128,128,40), mode='trilinear'),
            Interpolate(size=(256,256,40), mode='trilinear'),
            Conv3dMod(64, 5, stride=1, padding=1, hidden_layer=1))


    def forward(self, x, target):
        x = self.middle(x)
        x = self.upsample(x)
        loss = self.loss(x, target)
        return x, loss


class MiddleV1(nn.Module):
    def __init__(self,
        num_input_features=4,
        name='Middle_v1',
        criterion=None):
        super().__init__()

        self._name = name
        self.num_input_features = num_input_features
        self.loss = Loss(criterion)

        self.middle_backbone = nn.Sequential(
            Conv3dMod(16, 32, hidden_layer=2,
                      num_input_features=self.num_input_features),              # conv_mod:0
            Conv3dMod(32, 64, hidden_layer=2),                                  # conv_mod:1
            Conv3dMod(64, 128, hidden_layer=2, stride=(2,2,1)),                 # conv_mod:2
            Conv3dMod(128, 128))                                                # conv_mod:3

        self.middle_branch = nn.Sequential(
            Conv3dMod(128, 256, stride=(2,2,1)),                                # conv_mod:4
            Conv3dMod(256, 256, stride=1),                                      # conv_mod:5
            Interpolate(size=(16,16,5), mode='trilinear'),                      # interp:x
            # Notice that when kernel size is 1, padding should be 0
            Conv3dMod(256, 128, kernel=1, stride=1, padding=0, hidden_layer=1)) # conv_mod:6

        self.upsample = nn.Sequential(
            Conv3dMod(128, 64, stride=1, hidden_layer=1),                       # conv_mod:7
            Interpolate(size=(32,32,10), mode='trilinear'),                     # interp:0
            Conv3dMod(64, 32, stride=1, hidden_layer=1),                        # conv_mod:8
            Interpolate(size=(32,32,10), mode='trilinear'),                     # interp:1
            Conv3dMod(32, 32, stride=1, hidden_layer=1),                        # conv_mod:9
            Interpolate(size=(64,64,20), mode='trilinear'),                     # interp:2
            Interpolate(size=(128,128,40), mode='trilinear'),                   # interp:3
            Interpolate(size=(256,256,40), mode='trilinear'),                   # interp:4
            Conv3dMod(32, 5, stride=1, padding=1, hidden_layer=1))              # conv_mod:10

    def forward(self, x, target):
        x_backbone = self.middle_backbone(x)
        x_branch = self.middle_branch(x_backbone)
        x = x_backbone + x_branch
        x = self.upsample(x)

        loss = self.loss(x, target)
        return x, loss


class MiddleV2(nn.Module):
    def __init__(self,
        num_input_features=4,
        name='Middle_v2',
        criterion=None):
        super().__init__()
        self._name = name
        self.num_input_features = num_input_features
        self.middle = nn.Sequential(
            Conv3dMod_GCN(16, 32, num_input_features=self.num_input_features),      # conv_mod:0
            Conv3dMod_GCN(32, 64, stride=(2,2,1)),                                  # conv_mod:1
            Conv3dMod_GCN(64, 64, hidden_layer=4),                                  # conv_mod:2
            Conv3dMod_GCN(64, 64, stride=1, hidden_layer=4))                        # conv_mod:3
        self.upsample = nn.Sequential(
            Interpolate(size=(64,64,20), mode='trilinear'),
            Interpolate(size=(128,128,40), mode='trilinear'),
            Interpolate(size=(256,256,40), mode='trilinear'),
            Conv3dMod_GCN(64, 5, stride=1, padding=1, hidden_layer=1))
        self.loss = Loss(criterion)

    def forward(self, x, target):
        x = self.middle(x)
        x = self.upsample(x)
        loss = self.loss(x, target)
        return x, loss


class MiddleV3(nn.Module):
    def __init__(self,
        num_input_features=4,
        name='Middle_v2',
        criterion=None):
        super().__init__()

        self._name = name
        self.num_input_features = num_input_features
        self.loss = Loss(criterion)

        self.middle_backbone = nn.Sequential(
            Conv3dMod_GCN(16, 32, hidden_layer=2,
                        num_input_features=self.num_input_features),              # conv_mod:0
            Conv3dMod_GCN(32, 64, hidden_layer=2),                                  # conv_mod:1
            Conv3dMod_GCN(64, 128, hidden_layer=2, stride=(2,2,1)),                 # conv_mod:2
            Conv3dMod_GCN(128, 128))                                                # conv_mod:3

        self.middle_branch = nn.Sequential(
            Conv3dMod_GCN(128, 256, stride=(2,2,1)),                                # conv_mod:4
            Conv3dMod_GCN(256, 256, stride=1),                                      # conv_mod:5
            Interpolate(size=(16,16,5), mode='trilinear'),                      # interp:x
            # Notice that when kernel size is 1, padding should be 0
            Conv3dMod_GCN(256, 128, kernel=1, stride=1, padding=0, hidden_layer=1)) # conv_mod:6

        self.upsample = nn.Sequential(
            Conv3dMod_GCN(128, 64, stride=1, hidden_layer=1),                       # conv_mod:7
            Interpolate(size=(32,32,10), mode='trilinear'),                     # interp:0
            Conv3dMod_GCN(64, 32, stride=1, hidden_layer=1),                        # conv_mod:8
            Interpolate(size=(32,32,10), mode='trilinear'),                     # interp:1
            Conv3dMod_GCN(32, 32, stride=1, hidden_layer=1),                        # conv_mod:9
            Interpolate(size=(64,64,20), mode='trilinear'),                     # interp:2
            Interpolate(size=(128,128,40), mode='trilinear'),                   # interp:3
            Interpolate(size=(256,256,40), mode='trilinear'),                   # interp:4
            Conv3dMod_GCN(32, 5, stride=1, padding=1, hidden_layer=1))              # conv_mod:10


    def forward(self, x, target):
        x_backbone = self.middle_backbone(x)
        x_branch = self.middle_branch(x_backbone)
        x = x_backbone + x_branch
        x = self.upsample(x)
        loss = self.loss(x, target)
        return x, loss


class MiddleSparse(nn.Module):

# if __name__ == "__main__":
#     a = torch.randint(0, 5, (1,3,32,32,20)).cuda().float()
#     netv1 = Conv3dGCN(3,
#                       6,
#                       5,
#                       stride=2,
#                       padding=2,
#                       dilation=1,
#                       groups=1,
#                       bias=True,
#                       padding_mode='zeros',
#                       conv_mode='normal').cuda()
#     netv2 = Conv3dGCN(3,
#                       6,
#                       5,
#                       stride=2,
#                       padding=2,
#                       dilation=1,
#                       groups=1,
#                       bias=True,
#                       padding_mode='zeros',
#                       conv_mode='gcn').cuda()
#     output1 = netv1(a)
#     output2 = netv2(a)
#     pass




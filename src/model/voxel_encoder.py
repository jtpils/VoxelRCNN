# Voxel_encoder
# this is for building Voxel Feature Encoder network module
# stores VFE vital part in here

# TODO: SimpleVoxelRaduis realization implementation
# TODO: PointNet implementation
# TODO: Target acceleration
# TODO: Simple Voxel support batch

import torch
from torch import nn, optim
import torch.nn.functional as F
from config import cfg


def vfe_module_gen():
    return SimpleVoxel(
        num_input_features=cfg.model.num_input_features,
        name=cfg.model.vfe_name)


class SimpleVoxel(nn.Module):
    """SimpleVoxel

    SimpleVoxel is one of the realizations of VFE
    
    It only returns the mean of the points"""
    
    
    def __init__(self,
                 num_input_features=4,
                 name='VoxelFeatureExtractor'):
        super(SimpleVoxel, self).__init__()
        self.name = name
        self.num_input_features = num_input_features


    def forward(self, features, num_voxels):
        """
        The proposed features:
            features: [concated_num_points, num_voxel_size, 3(4)]
            num_voxels: [concated_num_points]
            
        Check README.md for further information
        """
        with torch.no_grad():
            points_mean = features[:, :, :self.num_input_features].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()
        # Using torch.view requires method Tensor.contiguous afterwards
        # 


def target_gen():
    return Target(name=cfg.model.vfe_name)

   
class Target(nn.Module):
    def __init__(self,
                 name='Target'):
        super(Target, self).__init__()
        self.name = name
    
    def forward(self, features):
        """
        Generates the sparse target
        By calculating which class has the max number of points in one voxel,
        to assign class number for each voxel.
        """
        target = []
        for i in range(features.shape[0]):
            class_num = features[i,:,-1]
            # Count which class has the maxinum number
            occurance_count = [0]       # set the None class occurance as 0
            occurance_count.extend(
                [(class_num == i).sum() for i in range(1, cfg.demo_dataset.class_num)])
            max_class = occurance_count.index(max(occurance_count))
            target.append(max_class)
        return torch.LongTensor(target).reshape(-1,1)
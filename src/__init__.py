from torch import nn
from torchsummary import summary
from spconv import SparseSequential, SubMConv3d, SparseConv3d
from src.model.voxel_encoder import vfe_module_gen

def get_VoxelMaskRCNN():
    net = VoxelMaskRCNN()
    # net.apply(init_weights)
    return net


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class VoxelMaskRCNN(nn.Module):
    """ Gather all the network """

    def __init__(self, verbose=False):
        super().__init__()
        if verbose:
            raise NotImplementedError("summary not implemented yet")
            # ?shape summary(self.voxel_encoder, (Shape))
        self.whatever = SparseSequential(
            SubMConv3d(3, 16, 3, indice_key="subm0"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            SparseConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            nn.BatchNorm1d(32),
            nn.ReLU()
        )


    def forward(self, features):
        output = self.whatever(features)
        return output
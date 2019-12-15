from torch import nn
from torchsummary import summary
from src.model.voxel_encoder import vfe_module_gen


def get_VoxelMaskRCNN():
    net = VoxelMaskRCNN()
    net.apply(init_weights)
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
        self.whatever = nn.Sequential(
            nn.Linear(3,5),
            nn.ReLU()
        )


    def forward(self, features):
        output = self.whatever(features)
        return output
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import cfg
from easydict import EasyDict as edict
import spconv


# -------------------------------- GLOBAL CONV ------------------------------- #

class Conv3dGCN(nn.Module):
    """ Implement Global Convolution Network (GCN) in the Conv3d

    Math: For a GCN conv3d operation, if we want the (D_out, H_out, W_out)
    the same as a normal conv3d, then certain parameters must be changed.

    Spefically, to adopt a more robust GCN, we need to decouple a (n, m, k)
    sized kernel conv as a (n, 1, 1), (1, m, 1), (1, 1, k) conv.

    The question of the core problem, is to find a formula that transforms
    the n/m/k dim to 1 dim. This is how we construct the problem.

    Restrictions:
    1. the input dim and the output dim should remain the same with GCN
    2. the dilation and the stride will not change with GCN
    3. that being said, the *padding* will be the only thing that will change.

    Deduction:
    conv3d formula is:
    Dim_out = floor((Dim_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    After calculation, the changed padding should be:

    ###################################################
    new_padding = old_padding - dilation * (kernel - 1)
    ###################################################

    @param: duplicate Conv3d params, with a name.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 conv_mode='gcn'):
        super().__init__()
        self._in = in_channels
        self._out = out_channels
        self._kernel = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups
        self._bias = bias
        self._padding_mode = padding_mode
        self._conv_mode = conv_mode
        self._conv_mode_list = ['normal', 'gcn']
        if self._conv_mode not in self._conv_mode_list:
            raise Exception("Convolution mode error")

        # Distinguish between normal conv3d and gcn-based conv3d
        if self._conv_mode == 'normal':
            self._conv3d_layer = nn.Conv3d(
                self._in,
                self._out,
                self._kernel,
                stride=self._stride,
                padding=self._padding,
                dilation=self._dilation,
                groups=self._groups,
                bias=self._bias,
                padding_mode=self._padding_mode)
        elif self._conv_mode == 'gcn':
            self._gcn_list = self._padding_align()
            self._conv3d_gcn = nn.ModuleList([])
            for i, branch in enumerate(self._gcn_list):
                self._conv3d_gcn.append = nn.Conv3d(
                    branch.in_channels,
                    branch.out_channels,
                    tuple(branch.kernel),
                    stride=branch.stride,
                    padding=tuple(branch.padding),
                    dilation=branch.dilation,
                    groups=branch.groups,
                    bias=branch.bias,
                    padding_mode=branch.padding_mode)


    def _padding_align(self):
        gcn = edict()

        # Assign those params that won't change with GCN
        gcn.in_channels = self._in
        gcn.out_channels = self._out
        gcn.stride = self._stride
        gcn.dilation = self._dilation
        gcn.groups = self._groups
        gcn.bias = self._bias
        gcn.padding_mode = self._padding_mode
        if not isinstance(self._padding, tuple):
            gcn.padding = np.asarray([self._padding] * 3)   # To change

        # create the gcn_list
        gcn_list = [edict(gcn.copy()), edict(gcn.copy()), edict(gcn.copy())]
        del gcn

        # Calculate new padding after
        original_kernel = self._expand2three(self._kernel)
        original_dilation = self._expand2three(self._dilation)
        original_padding = self._expand2three(self._padding)
        gcn_padding = original_padding - np.multiply(original_dilation,
                                                     (original_kernel - 1) / 2)

        # Assign the changed kernel and padding to each gcn_list item
        for i, gcn in enumerate(gcn_list):
            gcn.kernel = np.array([1, 1, 1])                    # To change
            gcn.kernel[i] = original_kernel[i]
            gcn.kernel = tuple(gcn.kernel)
            old_padding = gcn.padding.copy()
            gcn.padding = gcn_padding.copy()
            gcn.padding[i] = old_padding[i]
            del old_padding
            gcn.padding = tuple(gcn.padding.astype(np.int32))
        return gcn_list


    def _expand2three(self, x):
        """if the param is int, tile it to 3 times"""
        np_x = np.array(x, dtype=int)
        return np.tile(np_x, 3) if np_x.size == 1 else np_x


    def forward(self, x):
        if self._conv_mode == 'normal':
            output = self._conv3d_layer(x)
        elif self._conv_mode == 'gcn':
            output = 0
            for layers in self._conv3d_gcn:
                output += layers(x)
        return output


# -------------------------------- REGULAR MOD ------------------------------- #

class Conv3dMod(nn.Module):
    """Conv3dModule

    Conv3d Module follow the basic structure of SECOND.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=3,
                 stride=2,
                 padding=1,
                 hidden_layer=3,
                 num_input_features=None,
                 name='Conv3d'):
        super(Conv3dMod, self).__init__()
        self._name = name
        self._hidden = hidden_layer
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._stride = stride
        self._padding = padding
        self._num_input_features = num_input_features
        self.conv3d_module = nn.ModuleList([])

        for i in range(self._hidden-1):
            # Convolution
            if i == 0 and self._num_input_features is not None:
                conv3d_layer = nn.Conv3d(
                    self._num_input_features,
                    self._in_channels,
                    self._kernel,
                    padding=1) # To keep the dim not shrinking
            else:
                conv3d_layer = nn.Conv3d(
                    self._in_channels,
                    self._in_channels,
                    self._kernel,
                    padding=1) # To keep the dim not shrinking
            self.conv3d_module.append(conv3d_layer)
            # Batch Normalization
            bn_layer = nn.batchnorm1d(self._in_channels)
            self.conv3d_module.append(bn_layer)
            # ReLU
            relu_layer = nn.ReLU(True)
            self.conv3d_module.append(relu_layer)

        # Convlution downsampling
        conv3d_layer_down = nn.Conv3d(
            self._in_channels,
            self._out_channels,
            self._kernel,
            stride=self._stride,
            padding=self._padding)
        self.conv3d_module.append(conv3d_layer_down)
        # Batch Normalization
        bn_layer = nn.BatchNorm1d(self._out_channels)
        self.conv3d_module.append(bn_layer)
        # ReLU
        relu_layer = nn.ReLU(True)
        self.conv3d_module.append(relu_layer)


    def forward(self, x):
        for layer in self.conv3d_module:
            # i: index 0-n; l: the layer itself in the module list.
            x = layer(x)
        return x


class Conv3dMod_GCN(nn.Module):
    """Conv3dModule_GCN"""
    def __init__(self,
                in_channels,
                out_channels,
                kernel=3,
                stride=2,
                padding=1,
                hidden_layer=3,
                num_input_features=None,
                name='Conv3d_GCN'):
        super().__init__()
        self._name = name
        self._hidden = hidden_layer
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._stride = stride
        self._padding = padding
        self._num_input_features = num_input_features
        self.conv3d_module = nn.ModuleList([])

        for i in range(self._hidden - 1):
            # Convolution
            if i == 0 and self._num_input_features is not None:
                conv3d_layer = Conv3dGCN(
                    self._num_input_features,
                    self._in_channels,
                    self._kernel,
                    padding=1) # To keep the dim not shrinking
            else:
                conv3d_layer = Conv3dGCN(
                    self._in_channels,
                    self._in_channels,
                    self._kernel,
                    padding=1) # To keep the dim not shrinking
            self.conv3d_module.append(conv3d_layer)
            # Batch Normalization
            bn_layer = nn.batchnorm1d(self._in_channels)
            self.conv3d_module.append(bn_layer)
            # ReLU
            relu_layer = nn.ReLU(True)
            self.conv3d_module.append(relu_layer)

        # Convlution downsampling
        conv3d_layer_down = Conv3dGCN(
            self._in_channels,
            self._out_channels,
            self._kernel,
            stride=self._stride,
            padding=self._padding)
        self.conv3d_module.append(conv3d_layer_down)
        # Batch Normalization
        bn_layer = nn.batchnorm1d(self._out_channels)
        self.conv3d_module.append(bn_layer)
        # ReLU
        relu_layer = nn.ReLU(True)
        self.conv3d_module.append(relu_layer)


    def forward(self, x):
        for layer in self.conv3d_module:
            # i: index 0-n; l: the layer itself in the module list.
            x = layer(x)
        return x


# -------------------------------- INTERPOLATE ------------------------------- #

class Interpolate(nn.Module):
    """A workaround for using F.interpolate as an nn Module
    """
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode


    def __str__(self):
        print("Interpolate(size={}, mode={})".format(self.size, self.mode))


    def forward(self, x):
        return self.interp(
            x,
            size=self.size,
            mode=self.mode,
            align_corners=False)


# ---------------------------------- SPARSE ---------------------------------- #

class SparseConv3dMod(nn.Module):
    """ SparseConv3dMod

    Sparse Conv3d Module follow the basic structure of SECOND Middle.
    The detailed structure is:

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=3,
                 stride=2,
                 padding=1,
                 hidden_layer=3,
                 num_input_features=None,
                 name='Conv3d',
                 ReLU_name="ReLU"):
        super(Conv3dMod, self).__init__()
        self._name = name
        self._hidden = hidden_layer
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._stride = stride
        self._padding = padding
        self._num_input_features = num_input_features
        self.conv3d_module = spconv.SparseModule([])
        assert ReLU_name in ["ReLU", "Leaky"], "Not Implemented"
        ReLU = nn.ReLU if ReLU_name == "ReLU" else nn.LeakyReLU

        for i in range(self._hidden-1):
            # Convolution
            if i == 0 and self._num_input_features is not None:
                conv3d_layer = spconv.SubMConv3d(
                    self._num_input_features,
                    self._in_channels,
                    self._kernel,
                    padding=1) # To keep the dim not shrinking
            else:
                conv3d_layer = spconv.SubMConv3d(
                    self._in_channels,
                    self._in_channels,
                    self._kernel,
                    padding=1) # To keep the dim not shrinking
            self.conv3d_module.append(conv3d_layer)
            # Batch Normalization
            bn_layer = nn.BatchNorm1d(self._in_channels)
            self.conv3d_module.append(bn_layer)
            # ReLU
            relu_layer = ReLU()
            self.conv3d_module.append(relu_layer)

        # Convlution downsampling
        conv3d_layer_down = spconv.SparseConv3d(
            self._in_channels,
            self._out_channels,
            self._kernel,
            stride=self._stride,
            padding=self._padding)
        self.conv3d_module.append(conv3d_layer_down)
        # Batch Normalization
        bn_layer = nn.BatchNorm1d(self._out_channels)
        self.conv3d_module.append(bn_layer)
        # ReLU
        relu_layer = ReLU()
        self.conv3d_module.append(relu_layer)

        # Make it a Sequential
        self.conv3d_module = spconv.SparseSequential(*self.conv3d_module)


    def forward(self, x):
        x = self.conv3d_module(x)
        return x
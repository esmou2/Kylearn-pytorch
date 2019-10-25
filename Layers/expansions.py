import torch
import torch.nn as nn
import numpy as np
from utils.shuffle import feature_shuffle


# ------------------------------- 6 expansion layers ------------------------------- #
class LinearExpansion(nn.Module):
    '''expansion 1D -> 2D'''
    def __init__(self, d_features, n_channel, n_depth):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        self.expansion = nn.Linear(d_features, d_features * n_channel * n_depth)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, d_out]} -- output
        '''
        x = self.expansion(x)
        x = x.view([-1, self.d_features, self.n_channel * self.n_depth])
        return x

    def initialize_param(self, init, *args):
        init(self.expansion.weight, *args)


class ReduceParamLinearExpansion(nn.Module):
    '''expansion 1D -> 2D'''
    def __init__(self, d_features, n_channel, n_depth):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        d_hid = int(np.round(np.sqrt(d_features)))
        self.layer1 = nn.Linear(d_features, d_hid)
        self.layer2 = nn.Linear(d_hid, d_features * n_channel * n_depth)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, d_out]} -- output
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def initialize_param(self, init, *args):
        init(self.layer1.weight, *args)
        init(self.layer2.weight, *args)


class ConvExpansion(nn.Module):
    '''expansion 1D -> 2D'''
    def __init__(self, d_features, n_channel, n_depth):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        self.conv = nn.Conv1d(1, n_channel * n_depth, kernel_size=3, padding=1)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, d_features, n_channel * n_depth]} -- output
        '''
        assert x.dim() <= 3
        if x.dim() == 2:
            x = x.view(-1, 1, self.d_features)
        x = self.conv(x)
        x = x.transpose(2, 1)
        return x

    def initialize_param(self, init, *args):
        init(self.conv.weight, *args)


class LinearConvExpansion(nn.Module):
    '''expansion 1D -> 2D'''
    def __init__(self, d_features, n_channel, n_depth):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        self.d_hid = int(np.round(np.sqrt(d_features)))
        self.linear = nn.Linear(d_features, self.d_hid * d_features)
        self.conv = nn.Conv1d(self.d_hid, n_channel * n_depth, kernel_size=1)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, d_features, n_channel * n_depth]} -- output
        '''
        x = self.linear(x).view(-1, self.d_hid, self.d_features)
        x = self.conv(x)
        x = x.transpose(2, 1)
        return x

    def initialize_param(self, init, *args):
        init(self.linear.weight, *args)
        init(self.conv.weight, *args)


class ShuffleConvExpansion(nn.Module):
    '''expansion 1D -> 2D'''
    def __init__(self, d_features, n_channel, n_depth):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        self.index = feature_shuffle(d_features, depth=self.dim)
        self.index = torch.tensor(self.index)
        self.d_features = d_features
        self.conv = nn.Conv1d(self.n_channel * self.n_depth, self.n_channel * self.n_depth, kernel_size=3, padding=1,
                              groups=self.n_channel * self.n_depth)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, d_features, n_channel * n_depth]} -- output
        '''
        x = x[:, self.index]  # [batch, d_out]
        x = x.view(-1, self.n_channel * self.n_depth, self.d_features)  # [batch, n_channel, d_features]
        x = self.conv(x)  # [batch, n_channel, d_features]
        x = x.transpose(2, 1)
        return x

    def initialize_param(self, init, *args):
        init(self.conv.weight, *args)


class ChannelWiseConv(nn.Module):
    def __init__(self, d_features, n_channel, n_depth):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        self.conv = nn.Conv1d(n_depth, n_depth * n_channel, kernel_size=3, padding=1, groups=n_depth, bias=False)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, n_depth, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, n_depth, n_channel, d_features]} -- output
        '''
        x = self.conv(x).view(
            [-1, self.n_depth, self.n_channel, self.d_features])  # [batch, n_depth, n_channel, d_features]
        return x

    def initialize_param(self, init, *args):
        init(self.conv.weight, *args)


class ChannelWiseConvExpansion(nn.Module):
    '''expansion 1D -> 3D'''

    def __init__(self, d_features, n_channel, n_depth):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        self.channel_conv = ChannelWiseConv(d_features, n_channel, n_depth)


    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, n_depth, n_channel * d_features]} -- output
        '''
        x = self.channel_conv(x)  # [batch, n_depth, n_channel, d_features]
        x = x.view([-1, self.n_depth, self.n_channel * self.d_features])  # [batch, n_depth, n_channel * d_features]
        return x

    def initialize_param(self, init, *args):
        self.channel_conv.initialize_param(init, *args)

import torch
import torch.nn as nn
import numpy as np
from utils.shuffle import feature_shuffle_index


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
                x {Tensor, shape [batch, n_channel * n_depth, d_features]} -- output
        '''
        x = self.expansion(x)
        x = x.view([-1, self.n_channel * self.n_depth, self.d_features])
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
                x {Tensor, shape [batch, n_channel * n_depth, d_features]} -- output
        '''
        x = self.layer1(x)
        x = self.layer2(x).view([-1, self.n_channel * self.n_depth, self.d_features])
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
                x {Tensor, shape [batch, n_channel * n_depth, d_features]} -- output
        '''
        assert x.dim() <= 3
        if x.dim() == 2:
            x = x.view(-1, 1, self.d_features)
        x = self.conv(x)
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
                x {Tensor, shape [batch, n_channel * n_depth, d_features]} -- output
        '''
        x = self.linear(x).view(-1, self.d_hid, self.d_features)
        x = self.conv(x)
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

        self.index = feature_shuffle_index(d_features, depth=self.dim)
        self.index = torch.tensor(self.index)
        self.d_features = d_features
        self.conv = nn.Conv1d(self.n_channel * self.n_depth, self.n_channel * self.n_depth, kernel_size=3, padding=1,
                              groups=self.n_channel * self.n_depth)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, n_channel * n_depth, d_features]} -- output
        '''
        x = x[:, self.index]  # [batch, d_out]
        x = x.view(-1, self.n_channel * self.n_depth, self.d_features)  # [batch, n_channel * n_depth, d_features]
        x = self.conv(x)
        return x

    def initialize_param(self, init, *args):
        init(self.conv.weight, *args)


class ChannelWiseConv1d(nn.Module):
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
    '''expansion 2D -> 3D -> flatten'''

    def __init__(self, d_features, n_channel, n_depth):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        self.channel_conv = ChannelWiseConv1d(d_features, n_channel, n_depth)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, n_depth, d_features]

            Returns:
                x {Tensor, shape [batch, n_depth, n_channel * d_features]} -- output
        '''
        x = self.channel_conv(x)  # [batch, n_depth, n_channel, d_features]
        x = x.view([-1, self.n_depth, self.n_channel * self.d_features])  # [batch, n_depth, n_channel * d_features]
        return x

    def initialize_param(self, init, *args):
        self.channel_conv.initialize_param(init, *args)


class ChannelWiseConv2d(nn.Module):
    def __init__(self, d_features, n_channel, n_depth, c_total=3):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        self.conv = nn.Conv2d(n_depth, n_depth * n_channel, kernel_size=(c_total, 3), padding=(0, 1), groups=n_depth, bias=False)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, c_total, n_depth, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, n_depth, n_channel, d_features]} -- output
        '''
        x = x.permute([0, 2, 1, 3]).contiguous()  # [batch, n_depth, c_total, d_features]
        x = self.conv(x)  # [batch, n_depth * n_channel, 1, d_features]
        x = x.view(
            [-1, self.n_depth, self.n_channel, self.d_features])  # [batch, n_depth, n_channel, d_features]
        return x

    def initialize_param(self, init, *args):
        init(self.conv.weight, *args)

class ChannelWiseConvExpansionV2(nn.Module):
    '''expansion 3D -> 3D -> flatten'''

    def __init__(self, d_features, n_channel, n_depth, c_total=3):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth

        self.channel_conv = ChannelWiseConv2d(d_features, n_channel, n_depth, c_total=c_total)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, c_total, n_depth, d_features]

            Returns:
                x {Tensor, shape [batch, n_depth, n_channel * d_features]} -- output
        '''
        x = self.channel_conv(x)  # [batch, n_depth, n_channel, d_features]
        x = x.view([-1, self.n_depth, self.n_channel * self.d_features])  # [batch, n_depth, n_channel * d_features]
        return x

    def initialize_param(self, init, *args):
        self.channel_conv.initialize_param(init, *args)

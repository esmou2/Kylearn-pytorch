import torch
import torch.nn as nn
import numpy as np
from Layers.expansions import feature_shuffle
from Layers.bottlenecks import LinearBottleneckLayer, ShuffleBottleneckLayer

class ScaledDotProduction(nn.Module):
    '''Scaled Dot Production'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value):
        '''
            Arguments:
                query {Tensor, shape [n_head * batch, n_depth, n_channel * d_features]} -- query
                key {Tensor, shape [n_head * batch, n_depth, n_channel * d_features]} -- key
                value {Tensor, shape [n_head * batch, n_depth, n_vchannel * d_features]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, n_depth, n_vchannel * d_features] -- output
                attn {Tensor, shape [n_head * batch, n_depth, n_depth] -- reaction attention
        '''
        attn = torch.bmm(query, key.transpose(2, 1))  # [n_head * batch, n_depth, n_depth]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.bmm(attn, value)

        return output, attn


class SelfAttentionLayer(nn.Module):
    '''Self Attention'''

    def __init__(self, expansion_layer, n_depth, d_features, n_head, d_hid=256, dropout=0.1, use_bottleneck=True):
        super().__init__()
        self.d_features = d_features
        self.n_head = n_head
        self.n_depth = n_depth
        self.use_bottleneck = use_bottleneck

        self.query = expansion_layer(d_features=d_features, n_channel=n_head, n_depth=n_depth)
        self.key = expansion_layer(d_features=d_features, n_channel=n_head, n_depth=n_depth)
        self.value = expansion_layer(d_features=d_features, n_channel=n_head, n_depth=1)

        self.query.initialize_param(nn.init.xavier_normal_)
        self.key.initialize_param(nn.init.xavier_normal_)
        self.value.initialize_param(nn.init.xavier_normal_)

        self.attention = ScaledDotProduction(temperature=np.power(n_depth, 0.5))

        self.layer_norm = nn.LayerNorm(d_features)

        self.fc = nn.Linear(n_head * d_features, d_features)
        nn.init.xavier_normal(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        if use_bottleneck:
            self.bottleneck = LinearBottleneckLayer(d_features, d_hid)

    def forward(self, feature_1, feature_2=None):
        '''
            Arguments:
                feature_1 {Tensor, shape [batch, d_features]} -- feature part 1

            Returns:
                output {Tensor, shape [batch, d_features]} -- output
                attn {Tensor, shape [n_head * batch, d_features, d_features]} -- self attention
        '''
        d_features, n_head, n_depth, n_vchannel = self.d_features, self.n_head, self.n_depth, self.n_v

        batch_size, _ = feature_1.size()

        residual = feature_1

        query = self.query(feature_1).view(batch_size, d_features, n_head, n_depth)
        key = self.key(feature_1).view(batch_size, d_features, n_head, n_depth)
        value = self.value(feature_1).view(batch_size, 1, n_head, d_features)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, d_features, n_depth)
        key = key.premute(2, 0, 1, 3).contiguous().view(-1, d_features, n_depth)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_features)

        output, attn = self.attention(query, key, value)

        output = output.view(n_head, batch_size, d_features, 1)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        if self.use_bottleneck:
            output = self.bottleneck(output)

        return output, attn


class ShuffleSelfAttention(nn.Module):
    '''Self Attention'''

    def __init__(self, expansion_layer, n_depth, d_features, n_head, n_channel, n_vchannel):
        super().__init__()
        self.n_depth = n_depth
        self.d_features = d_features
        self.n_head = n_head
        self.n_channel = n_channel
        self.n_vchannel = n_vchannel
        total_channels = n_head * n_channel
        total_vchannels = n_head * n_vchannel

        self.query = expansion_layer(d_features=d_features, n_channel=total_channels, n_depth=n_depth)
        self.key = expansion_layer(d_features=d_features, n_channel=total_channels, n_depth=n_depth)
        self.value = expansion_layer(d_features=d_features, n_channel=total_vchannels, n_depth=n_depth)

        self.query.initialize_param(nn.init.xavier_normal_)
        self.key.initialize_param(nn.init.xavier_normal_)
        self.value.initialize_param(nn.init.xavier_normal_)

        self.attention = ScaledDotProduction(temperature=np.power(n_depth, 0.5))

    def forward(self, feature_map):
        '''
            Arguments:
                feature_map {Tensor, shape [batch, n_depth, d_features]} -- feature part 1

            Returns:
                output {Tensor, shape [batch, n_head * n_vchannel, n_depth, d_features]} -- output
                attn {Tensor, shape [n_head * batch, n_depth, n_depth]} -- self attention
        '''
        d_f1, n_head, n_channel, n_vchannel, n_depth = self.d_features, self.n_head, self.n_channel, self.n_vchannel, self.n_depth

        batch_size, _, _ = feature_map.size()

        query = self.query(
            feature_map)  # shape [batch, n_depth, n_head * n_channel * d_features] for using ChannelWiseConvExpansion, otherwise [batch, d_features, n_head * n_channel * n_depth]
        dim_2nd = query.shape[1]  # n_depth or d_features
        query = query.view(batch_size, dim_2nd, n_head,
                           -1)  # [batch, n_depth, n_head, n_channel * d_features] or [batch, d_features, n_head, n_channel * n_depth]
        key = self.key(feature_map)
        key = key.view(batch_size, dim_2nd, n_head, -1)
        value = self.value(feature_map)
        value = value.view(batch_size, dim_2nd, n_head, -1)

        query = query.permute(2, 0, 1, 3)  # [n_head, batch, n_depth, n_channel * d_features]
        query = query.contiguous().view(n_head * batch_size, dim_2nd,
                                        -1)  # [n_head * batch, n_depth, n_channel * d_features] or [n_head * batch, d_features, n_channel * n_depth]
        key = key.permute(2, 0, 1, 3).contiguous().view(n_head * batch_size, dim_2nd, -1)
        value = value.permute(2, 0, 1, 3).contiguous().view(n_head * batch_size, dim_2nd, -1)

        output, attn = self.attention(query, key,
                                      value)  # [n_head * batch, n_depth, n_vchannel * d_features]

        output = output.view(n_head, batch_size, dim_2nd,
                             -1)  # [n_head, batch, n_depth, n_vchannel * d_features]
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, dim_2nd, n_head * n_vchannel,
                                                              -1)  # [batch, n_depth, n_head * n_vchannel, d_features]
        output = output.transpose(1, 2)  # [batch, n_head * n_vchannel, n_depth, d_features]

        return output, attn




class ShuffleSelfAttentionLayer(nn.Module):
    def __init__(self, expansion_layer, n_depth, d_features, n_head, n_channel, n_vchannel, dropout,
                 mode='1d', use_bottleneck=True, d_bottleneck=None):
        super().__init__()

        assert n_depth > 1

        self.d_features = d_features
        self.n_depth = n_depth
        self.mode = mode
        self.use_bottleneck = use_bottleneck

        self.index = feature_shuffle(d_features, depth=n_depth)
        self.index = torch.tensor(self.index)

        self.shuffle_slf_attn = ShuffleSelfAttention(expansion_layer, n_depth, d_features, n_head, n_channel, n_vchannel)

        self.layer_norm = nn.LayerNorm(d_features)
        self.dropout = nn.Dropout(dropout)

        if mode == '1d':
            self.conv = nn.Conv2d(n_head * n_vchannel, 1, kernel_size=(n_depth, 1), bias=False)
            nn.init.xavier_normal(self.conv.weight)

        elif mode == '2d':
            self.conv = nn.Conv2d(n_head * n_vchannel, 1, kernel_size=(1, 1), bias=False)  # or use fc
            nn.init.xavier_normal(self.conv.weight)
        else:
            pass

        if use_bottleneck:
            self.bottleneck = ShuffleBottleneckLayer(n_depth, d_features, mode, d_hid=d_bottleneck)

    def forward(self, features):
        if features.dim() == 2:
            feature_map = features[:, self.index] # [batch, n_depth * d_features]
            feature_map = feature_map.view([-1, self.n_depth, self.d_features])  # [batch, n_depth, d_features]
        else:
            feature_map = features

        residual = features

        output, attn = self.shuffle_slf_attn(feature_map)  # output shape [batch, n_head * n_vchannel, n_depth, d_features]
        output = self.conv(output)

        if self.mode == '1d':
            if features.dim() == 3:
                residual = 0
                print('[WARNING]: Got features with dimension=3 while output mode is 1d, cannot apply residual')
            output = output.view([-1, self.d_features])

        elif self.mode == '2d':
            residual = feature_map
            output = output.view([-1, self.n_depth, self.d_features])  # [batch, n_depth, d_features]

        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        if self.use_bottleneck:
            output = self.bottleneck(output)

        return output, attn

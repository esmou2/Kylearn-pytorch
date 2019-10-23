import torch
import torch.nn as nn
import numpy as np
from Layers.expansion import feature_shuffle


class LinearBottleneckLayer(nn.Module):
    ''' Bottleneck Layer '''

    def __init__(self, d_features, d_hid, d_out=None, dropout=0.1):
        super().__init__()
        if d_out == None:
            d_out = d_features

        self.encode = nn.Linear(d_features, d_hid)
        self.decode = nn.Linear(d_hid, d_out)
        nn.init.xavier_normal_(self.encode.weight)
        nn.init.xavier_normal_(self.decode.weight)
        self.layer_norm = nn.LayerNorm(d_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch_size, d_features]}
            Returns:
                x {Tensor, shape [batch_size, d_features]}
        '''
        residual = x
        encode = nn.functional.relu(self.encode(x))
        decode = self.decode(encode)
        output = self.dropout(decode)
        output = self.layer_norm(output + residual)
        output = output + residual
        return output


class ReactionDotProduction(nn.Module):
    ''' Scaled Dot Productionss '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, expansion, depth, value):
        '''
            Arguments:
                expansion {Tensor, shape [n_head * batch, d_f1, n_depth]} -- expansion
                depth {Tensor, shape [n_head * batch, 1, n_depth]} -- depth
                value {Tensor, shape [n_head * batch, 1, d_f1]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, 1, d_f1]} -- output
                attn {Tensor, shape [n_head * batch, 1, d_f1]} -- reaction attention
        '''
        attn = torch.bmm(depth, expansion.transpose(1, 2))  # [n_head * batch, 1, d_f1]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.mul(attn, value)

        return output, attn


class ReactionAttentionLayerV1(nn.Module):
    '''Reaction Attention'''

    def __init__(self, expansion_layer, n_head, n_depth, d_f1, d_f2, d_hid=256, dropout=0.1, use_bottleneck=True):
        super().__init__()

        self.d_f1 = d_f1
        self.d_f2 = d_f2
        self.n_head = n_head
        self.n_depth = np.floor(n_depth / n_head).astype(int)
        self.use_bottleneck = use_bottleneck
        self.expansion = expansion_layer(d_features=d_f1, n_channel=n_head, n_depth=n_depth)
        self.expansion.initialize_param(nn.init.normal_)
        # self.expansion.initialize_param(nn.init.normal_, mean=0, std=np.sqrt(2.0 / (d_f1 + d_f1 * n_depth)))

        self.depth = nn.Linear(d_f2, n_head * self.n_depth)
        self.value = nn.Linear(d_f1, n_head * d_f1)

        nn.init.normal_(self.depth.weight, mean=0, std=np.sqrt(2.0 / (d_f2 + self.n_depth)))
        nn.init.normal_(self.value.weight, mean=0, std=np.sqrt(1.0 / (d_f1)))
        self.attention = ReactionDotProduction(temperature=np.power(self.n_depth, 0.5))

        self.layer_norm = nn.LayerNorm(d_f1)

        self.fc = nn.Linear(n_head * d_f1, d_f1)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        if use_bottleneck:
            self.bottleneck = LinearBottleneckLayer(d_f1, d_hid)

    def forward(self, feature_1, feature_2):
        '''
            Arguments:
                feature_1 {Tensor, shape [batch, d_f1]} -- feature part 1
                feature_2 {Tensor, shape [batch, d_f2]} -- feature part 2, can be categorical data

            Returns:
                output {Tensor, shape [batch, d_f1]} -- output
                attn {Tensor, shape [n_head * batch, 1, d_f1]} -- self attention
        '''
        d_f1, d_f2, n_head, n_depth = self.d_f1, self.d_f2, self.n_head, self.n_depth

        batch_size, _ = feature_1.size()

        residual = feature_1

        expansion = self.expansion(feature_1).view(batch_size, d_f1, n_head, n_depth)  # [batch, d_f1, n_head * n_depth]
        depth = self.depth(feature_2).view(batch_size, 1, n_head, n_depth)
        value = self.value(feature_1).view(batch_size, 1, n_head, d_f1)
        # value = feature_1.repeat(1, n_head).view(batch_size, 1, n_head, d_f1)

        expansion = expansion.permute(2, 0, 1, 3).contiguous().view(-1, d_f1, n_depth)
        depth = depth.permute(2, 0, 1, 3).contiguous().view(-1, 1, n_depth)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_f1)

        output, attn = self.attention(expansion, depth, value)

        output = output.view(n_head, batch_size, 1, d_f1)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        if self.use_bottleneck:
            output = self.bottleneck(output)

        return output, attn


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
                query {Tensor, shape [n_head * batch, n_depth, (n_channel / n_head) * d_features]} -- query
                key {Tensor, shape [n_head * batch, n_depth, (n_channel / n_head) * d_features]} -- key
                value {Tensor, shape [n_head * batch, n_depth, (n_vchannel / n_head) * d_features]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, n_depth, (n_vchannel / n_head) * d_features] -- output
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

    def __init__(self, expansion_layer, n_head, n_depth, d_f1, d_hid=256, dropout=0.1, use_bottleneck=True):
        super().__init__()
        self.d_f1 = d_f1
        self.n_head = n_head
        self.n_depth = n_depth
        self.use_bottleneck = use_bottleneck

        self.query = expansion_layer(d_features=d_f1, n_channel=n_head, n_depth=n_depth)
        self.key = expansion_layer(d_features=d_f1, n_channel=n_head, n_depth=n_depth)
        self.value = expansion_layer(d_features=d_f1, n_channel=n_head, n_depth=1)

        self.query.initialize_param(nn.init.xavier_normal_)
        self.key.initialize_param(nn.init.xavier_normal_)
        self.value.initialize_param(nn.init.xavier_normal_)

        self.attention = ScaledDotProduction(temperature=np.power(n_depth, 0.5))

        self.layer_norm = nn.LayerNorm(d_f1)

        self.fc = nn.Linear(n_head * d_f1, d_f1)
        nn.init.xavier_normal(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        if use_bottleneck:
            self.bottleneck = LinearBottleneckLayer(d_f1, d_hid)

    def forward(self, feature_1, feature_2=None):
        '''
            Arguments:
                feature_1 {Tensor, shape [batch, d_f1]} -- feature part 1

            Returns:
                output {Tensor, shape [batch, d_f1]} -- output
                attn {Tensor, shape [n_head * batch, d_f1, d_f1]} -- self attention
        '''
        d_f1, n_head, n_depth, n_vchannel = self.d_f1, self.n_head, self.n_depth, self.n_v

        batch_size, _ = feature_1.size()

        residual = feature_1

        query = self.query(feature_1).view(batch_size, d_f1, n_head, n_depth)
        key = self.key(feature_1).view(batch_size, d_f1, n_head, n_depth)
        value = self.value(feature_1).view(batch_size, 1, n_head, d_f1)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, d_f1, n_depth)
        key = key.premute(2, 0, 1, 3).contiguous().view(-1, d_f1, n_depth)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_f1)

        output, attn = self.attention(query, key, value)

        output = output.view(n_head, batch_size, d_f1, 1)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        if self.use_bottleneck:
            output = self.bottleneck(output)

        return output, attn


class ShuffleSelfAttention(nn.Module):
    '''Self Attention'''

    def __init__(self, expansion_layer, n_head, n_channel, n_vchannel, n_depth, d_features):
        super().__init__()
        self.d_features = d_features
        self.n_head = n_head
        self.n_channel = n_channel
        self.n_vchannel = n_vchannel
        self.n_depth = n_depth

        self.query = expansion_layer(d_features=d_features, n_channel=n_channel, n_depth=n_depth)
        self.key = expansion_layer(d_features=d_features, n_channel=n_channel, n_depth=n_depth)
        self.value = expansion_layer(d_features=d_features, n_channel=n_vchannel, n_depth=n_depth)

        self.query.initialize_param(nn.init.xavier_normal_)
        self.key.initialize_param(nn.init.xavier_normal_)
        self.value.initialize_param(nn.init.xavier_normal_)

        self.attention = ScaledDotProduction(temperature=np.power(n_depth, 0.5))

    def forward(self, feature_map):
        '''
            Arguments:
                feature_map {Tensor, shape [batch, n_depth, d_features]} -- feature part 1

            Returns:
                output {Tensor, shape [batch, n_vchannel, n_depth, d_features]} -- output
                attn {Tensor, shape [n_head * batch, n_depth, n_depth]} -- self attention
        '''
        d_f1, n_head, n_channel, n_vchannel, n_depth = self.d_features, self.n_head, self.n_channel, self.n_vchannel, self.n_depth

        batch_size, _, _ = feature_map.size()

        query = self.query(
            feature_map)  # shape [batch, n_depth, n_channel * d_features] for using ChannelWiseConvExpansion, otherwise [batch, d_features, n_channel * n_depth]
        dim_2nd = query.shape[1]  # n_depth or d_features
        query = query.view(batch_size, dim_2nd, n_head,
                           -1)  # [batch, n_depth, n_head, (n_channel / n_head) * d_features] or [batch, d_features, n_head, (n_channel / n_head) * n_depth]
        key = self.key(feature_map)
        key = key.view(batch_size, dim_2nd, n_head, -1)
        value = self.value(feature_map)
        value = value.view(batch_size, dim_2nd, n_head, -1)

        query = query.permute(2, 0, 1, 3)  # [n_head, batch, n_depth, (n_channel / n_head) * d_features]
        query = query.contiguous().view(n_head * batch_size, dim_2nd,
                                        -1)  # [n_head * batch, n_depth, (n_channel / n_head) * d_features] or [n_head * batch, d_features, (n_channel / n_head) * n_depth]
        key = key.permute(2, 0, 1, 3).contiguous().view(n_head * batch_size, dim_2nd, -1)
        value = value.permute(2, 0, 1, 3).contiguous().view(n_head * batch_size, dim_2nd, -1)

        output, attn = self.attention(query, key,
                                      value)  # [n_head * batch, n_depth, (n_vchannel / n_head) * d_features]

        output = output.view(n_head, batch_size, dim_2nd,
                             -1)  # [n_head, batch, n_depth, (n_vchannel / n_head) * d_features]
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, dim_2nd, n_vchannel,
                                                              -1)  # [batch, n_depth, n_vchannel, d_features]
        output = output.transpose(1, 2)  # [batch, n_vchannel, n_depth, d_features]

        return output, attn


class ShuffleBottleneckLayer(nn.Module):
    ''' Bottleneck Layer '''

    def __init__(self, n_depth, d_features, mode, d_hid=None, dropout=0.1):
        super().__init__()
        self.n_depth = n_depth
        self.d_features = d_features
        self.mode = mode
        if d_hid == None:
            d_hid = d_features

        if mode == '1d':
            self.bottle_neck_1 = nn.Linear(d_features, d_hid)
            self.bottle_neck_2 = nn.Linear(d_hid, d_features)

        elif mode == '2d':
            # self.bottle_neck_1 = nn.Conv1d(n_depth, d_hid, kernel_size=1, bias=False)
            # self.bottle_neck_2 = nn.Conv1d(d_hid, n_depth, kernel_size=1, bias=False)
            self.bottle_neck_1 = nn.Conv1d(d_features, d_hid, kernel_size=1)
            self.bottle_neck_2 = nn.Conv1d(d_hid, d_features, kernel_size=1)
        else:
            pass

        nn.init.xavier_normal_(self.bottle_neck_1.weight)
        nn.init.xavier_normal_(self.bottle_neck_2.weight)
        self.layer_norm = nn.LayerNorm([d_features])

        self.activation = nn.functional.relu

        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        '''
            Arguments:
                features {Tensor, shape [batch, d_features] or [batch, n_depth, d_features]} -- features

            Returns:
                x {Tensor, shape [batch_size, d_features]}
        '''
        residual = features

        if self.mode == '1d':
            output = self.bottle_neck_1(features)
            output = self.activation(output)
            output = self.bottle_neck_2(output)


        elif self.mode == '2d':
            output = features.transpose(1, 2)
            output = self.bottle_neck_1(output)
            output = self.activation(output)
            output = self.bottle_neck_2(output)
            output = output.transpose(2, 1)
        else:
            residual = 0
            output = features

        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class ShuffleSelfAttentionLayer(nn.Module):
    def __init__(self, expansion_layer, n_head, n_channel, n_vchannel, n_depth, d_features, d_hid, dropout=0.1, mode='1d',
                 use_bottleneck=True):
        super().__init__()
        self.d_features = d_features
        self.n_depth = n_depth
        self.mode = mode
        self.use_bottleneck = use_bottleneck

        self.index = feature_shuffle(d_features, depth=n_depth)
        self.index = torch.tensor(self.index)

        self.shuffle_slf_attn = ShuffleSelfAttention(expansion_layer, n_head, n_channel, n_vchannel, n_depth,
                                                     d_features)

        self.layer_norm = nn.LayerNorm(d_features)
        self.dropout = nn.Dropout(dropout)

        if mode == '1d':
            self.conv = nn.Conv2d(n_vchannel, 1, kernel_size=(n_depth, 1), bias=False)
            nn.init.xavier_normal(self.conv.weight)

        elif mode == '2d':
            self.conv = nn.Conv2d(n_vchannel, 1, kernel_size=(1, 1), bias=False)  # or use fc
            nn.init.xavier_normal(self.conv.weight)
        else:
            pass

        if use_bottleneck:
            self.bottleneck = ShuffleBottleneckLayer(n_depth, d_features, mode, d_hid)

    def forward(self, features):
        if features.dim() == 2:
            feature_map = features[:, self.index].contiguous()  # [batch, n_depth * d_features]
            feature_map = feature_map.view([-1, self.n_depth, self.d_features])  # [batch, n_depth, d_features]
        else:
            feature_map = features

        residual = features

        output, attn = self.shuffle_slf_attn(feature_map)  # output shape [batch, n_vchannel, n_depth, d_features]
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

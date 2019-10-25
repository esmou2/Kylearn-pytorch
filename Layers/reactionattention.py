import torch
import torch.nn as nn
import numpy as np
from Layers.bottlenecks import LinearBottleneckLayer

class ReactionDotProduction(nn.Module):
    ''' Scaled Dot Productionss '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value):
        '''
            Arguments:
                key {Tensor, shape [n_head * batch, d_features, n_depth_per_head]} -- expansion
                query {Tensor, shape [n_head * batch, 1, n_depth_per_head]} -- depth
                value {Tensor, shape [n_head * batch, 1, d_features]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, 1, d_features]} -- output
                attn {Tensor, shape [n_head * batch, 1, d_features]} -- reaction attention
        '''
        attn = torch.bmm(query, key.transpose(1, 2))  # [n_head * batch, 1, d_features]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.mul(attn, value)

        return output, attn


class ReactionAttentionLayerV1(nn.Module):
    '''Reaction Attention'''

    def __init__(self,expansion_layer, n_depth, d_features, d_meta, n_head, dropout,
                 use_bottleneck=True, d_bottleneck=None):
        super().__init__()

        self.d_features = d_features
        self.d_meta = d_meta
        self.n_head = n_head
        self.n_depth = n_depth
        self.use_bottleneck = use_bottleneck
        self.expansion = expansion_layer(d_features=d_features, n_channel=n_head, n_depth=n_depth) # output [batch, d_features, n_channel * n_depth]
        self.expansion.initialize_param(nn.init.xavier_normal_)

        # query, value map
        self.query = nn.Linear(d_meta, n_head * self.n_depth)
        self.value = nn.Linear(d_features, n_head * d_features)

        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.value.weight)

        self.attention = ReactionDotProduction(temperature=np.power(self.n_depth, 0.5))

        self.layer_norm = nn.LayerNorm(d_features)

        self.fc = nn.Linear(n_head * d_features, d_features)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        if use_bottleneck:
            self.bottleneck = LinearBottleneckLayer(d_features, d_bottleneck)

    def forward(self, features, meta):
        '''
            Arguments:
                feature_1 {Tensor, shape [batch, d_features]} -- feature part 1
                feature_2 {Tensor, shape [batch, d_meta]} -- feature part 2, can be categorical data

            Returns:
                output {Tensor, shape [batch, d_features]} -- output
                attn {Tensor, shape [n_head * batch, 1, d_features]} -- self attention
        '''
        d_features, d_meta, n_head, n_depth_per_head = self.d_features, self.d_meta, self.n_head, self.n_depth

        batch_size, _ = features.size()

        residual = features

        query = self.query(meta).view(batch_size, 1, n_head, n_depth_per_head)
        key = self.expansion(features).view(batch_size, n_depth_per_head, d_features, n_head)  # [batch, n_depth, n_head, d_features]
        value = self.value(features).view(batch_size, 1, n_head, d_features)
        # value = feature_1.repeat(1, n_head).view(batch_size, 1, n_head, d_features)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, 1, n_depth_per_head)
        key = key.permute(2, 0, 3, 1).contiguous().view(-1, d_features, n_depth_per_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_features)

        output, attn = self.attention(query, key, value)

        output = output.view(n_head, batch_size, 1, d_features)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        if self.use_bottleneck:
            output = self.bottleneck(output)

        return output, attn



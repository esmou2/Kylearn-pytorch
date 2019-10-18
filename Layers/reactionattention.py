import torch
import torch.nn as nn
import numpy as np
from utils.shuffle import feature_shuffle

# ------------------------------- 5 expansion layers ------------------------------- #
class LinearExpansion(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.expansion = nn.Linear(d_in, d_out)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_in]} -- input

            Returns:
                x {Tensor, shape [batch, d_out]} -- output
        '''
        x = self.expansion(x)
        return x

    def initialize_param(self, init, *args):
        init(self.expansion.weight, *args)

class ReduceParamLinearExpansion(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        d_hid = int(np.round(np.sqrt(d_in)))
        self.layer1 = nn.Linear(d_in, d_hid)
        self.layer2 = nn.Linear(d_hid, d_out)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_in]} -- input

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

    def __init__(self, d_in, d_out):
        super().__init__()
        n_channel = d_out // d_in
        self.d_in = d_in
        self.conv = nn.Conv1d(1, n_channel, kernel_size=3, padding=1)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_in]} -- input

            Returns:
                x {Tensor, shape [batch, d_in, n_channel]} -- output
        '''
        assert x.dim() <= 3
        if x.dim() == 2:
            x = x.view(-1, 1, self.d_in)
        x = self.conv(x)
        x = x.transpose(2,1)
        return x

    def initialize_param(self, init, *args):
        init(self.conv.weight, *args)

class LinearConvExpansion(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_hid = int(np.round(np.sqrt(d_in)))
        n_channel = d_out // d_in
        self.d_in = d_in
        self.linear = nn.Linear(d_in, self.d_hid * d_in)
        self.conv = nn.Conv1d(self.d_hid, n_channel, kernel_size=1)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_in]} -- input

            Returns:
                x {Tensor, shape [batch, d_in, n_channel]} -- output
        '''
        x = self.linear(x).view(-1, self.d_hid, self. d_in)
        x = self.conv(x)
        x = x.transpose(2,1)
        return x

    def initialize_param(self, init, *args):
        init(self.linear.weight, *args)
        init(self.conv.weight, *args)


class ShuffleConvExpansion(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.n_channel = int(d_out // d_in)
        self.index = feature_shuffle(d_in, depth=self.n_channel)
        self.index = torch.tensor(self.index)
        self.d_in = d_in
        self.conv = nn.Conv1d(self.n_channel, self.n_channel, kernel_size=3, padding=1, groups=self.n_channel)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, d_in]} -- input

            Returns:
                x {Tensor, shape [batch, d_in, n_channel]} -- output
        '''
        x = x[:, self.index] # [batch, d_out]
        x = x.view(-1, self.n_channel, self.d_in) # [batch, n_channel, d_in]
        x = self.conv(x) # [batch, n_channel, d_in]
        x = x.transpose(2,1)
        return x

    def initialize_param(self, init, *args):
        init(self.conv.weight, *args)

# ------------------------------- 5 expansion layers ------------------------------- #


class BottleneckLayer(nn.Module):
    ''' Bottleneck Layer '''

    def __init__(self, d_in, d_hid, d_out=None, dropout=0.1):
        super().__init__()
        if d_out == None:
            d_out = d_in

        self.encode = nn.Linear(d_in, d_hid)
        self.decode = nn.Linear(d_hid, d_out)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch_size, d_in]}
            Returns:
                x {Tensor, shape [batch_size, d_in]}
        '''
        residual = x
        encode = nn.functional.relu(self.encode(x))
        decode = self.decode(encode)
        output = self.dropout(decode)
        # output = self.layer_norm(output + residual)
        output = output + residual
        return output


class ReactionDotProduction(nn.Module):
    ''' Scaled Dot Productionss '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, expansion, reactant, value):
        '''
            Arguments:
                expansion {Tensor, shape [n_head * batch, d_f1, d_reactant]} -- expansion
                reactant {Tensor, shape [n_head * batch, 1, d_reactant]} -- reactant
                value {Tensor, shape [n_head * batch, 1, d_f1]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, 1, d_f1]} -- output
                attn {Tensor, shape [n_head * batch, 1, d_f1]} -- reaction attention
        '''
        attn = torch.bmm(reactant, expansion.transpose(1, 2))  # [n_head * batch, 1, d_f1]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.mul(attn, value)

        return output, attn


class ReactionAttentionLayerV1(nn.Module):
    '''Reaction Attention'''

    def __init__(self, expansion_layer, n_head, d_reactant, d_f1, d_f2, d_hid=256, dropout=0.1, use_bottleneck=True):
        super().__init__()
        self.d_f1 = d_f1
        self.d_f2 = d_f2
        self.n_head = n_head
        self.d_reactant = np.floor(d_reactant / n_head).astype(int)
        self.use_bottleneck = use_bottleneck
        self.expansion = expansion_layer(d_f1, n_head * d_f1 * self.d_reactant)
        self.expansion.initialize_param(nn.init.normal_)
        # self.expansion.initialize_param(nn.init.normal_, mean=0, std=np.sqrt(2.0 / (d_f1 + d_f1 * d_reactant)))


        self.reactant = nn.Linear(d_f2, n_head * self.d_reactant)
        self.value = nn.Linear(d_f1, n_head * d_f1)

        nn.init.normal_(self.reactant.weight, mean=0, std=np.sqrt(2.0 / (d_f2 + self.d_reactant)))
        nn.init.normal_(self.value.weight, mean=0, std=np.sqrt(1.0 / (d_f1)))
        self.attention = ReactionDotProduction(temperature=np.power(self.d_reactant, 0.5))

        self.layer_norm = nn.LayerNorm(d_f1)

        self.fc = nn.Linear(n_head * d_f1, d_f1)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        if use_bottleneck:
            self.bottleneck = BottleneckLayer(d_f1, d_hid)

    def forward(self, feature_1, feature_2):
        '''
            Arguments:
                feature_1 {Tensor, shape [batch, d_f1]} -- feature part 1
                feature_2 {Tensor, shape [batch, d_f2]} -- feature part 2, can be categorical data

            Returns:
                output {Tensor, shape [batch, d_f1]} -- output
                attn {Tensor, shape [n_head * batch, 1, d_f1]} -- self attention
        '''
        d_f1, d_f2, n_head, d_reactant = self.d_f1, self.d_f2, self.n_head, self.d_reactant

        batch_size, _ = feature_1.size()

        residual = feature_1

        expansion = self.expansion(feature_1).view(batch_size, d_f1, n_head, d_reactant)
        reactant = self.reactant(feature_2).view(batch_size, 1, n_head, d_reactant)
        value = self.value(feature_1).view(batch_size, 1, n_head, d_f1)
        # value = feature_1.repeat(1, n_head).view(batch_size, 1, n_head, d_f1)

        expansion = expansion.permute(2, 0, 1, 3).contiguous().view(-1, d_f1, d_reactant)
        reactant = reactant.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_reactant)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_f1)

        output, attn = self.attention(expansion, reactant, value)

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
                query {Tensor, shape [n_head * batch, d_f1, d_reactant]} -- query
                key {Tensor, shape [n_head * batch, d_f1, d_reactant]} -- key
                value {Tensor, shape [n_head * batch, 1, d_f1]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, 1, d_f1] -- output
                attn {Tensor, shape [n_head * batch, d_f1, d_f1] -- reaction attention
        '''
        attn = torch.bmm(query, key.transpose(2, 1))  # [n_head * batch, d_f1, d_f1]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.bmm(value, attn)

        return output, attn


class SelfAttentionLayer(nn.Module):
    '''Self Attention'''

    def __init__(self, n_head, d_reactant, d_f1, d_hid=256, dropout=0.1, use_bottleneck=True):
        super().__init__()
        self.d_f1 = d_f1
        self.n_head = n_head
        self.d_reactant = d_reactant
        self.use_bottleneck = use_bottleneck

        self.query = ReduceParamLinearExpansion(d_f1, n_head * d_f1 * d_reactant)
        self.key = ReduceParamLinearExpansion(d_f1, n_head * d_f1 * d_reactant)
        self.value = ReduceParamLinearExpansion(d_f1, n_head * d_f1)

        self.query.initialize_param(nn.init.normal_, mean=0, std=np.sqrt(2.0 / (d_f1 + d_f1 * d_reactant)))
        self.key.initialize_param(nn.init.normal_, mean=0, std=np.sqrt(2.0 / (d_f1 + d_f1 * d_reactant)))
        self.value.initialize_param(nn.init.normal_, mean=0, std=np.sqrt(2.0 / (d_f1 + d_f1 * d_reactant)))

        self.attention = ScaledDotProduction(temperature=np.power(d_reactant, 0.5))

        self.layer_norm = nn.LayerNorm(d_f1)

        self.fc = nn.Linear(n_head * d_f1, d_f1)
        nn.init.xavier_normal(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        if use_bottleneck:
            self.bottleneck = BottleneckLayer(d_f1, d_hid)

    def forward(self, feature_1, feature_2=None):
        '''
            Arguments:
                feature_1 {Tensor, shape [batch, d_f1]} -- feature part 1

            Returns:
                output {Tensor, shape [batch, d_f1]} -- output
                attn {Tensor, shape [n_head * batch, d_f1, d_f1]} -- self attention
        '''
        d_f1, n_head, d_reactant = self.d_f1, self.n_head, self.d_reactant

        batch_size, _ = feature_1.size()

        residual = feature_1

        query = self.key(feature_1).view(batch_size, d_f1, n_head, d_reactant)
        key = self.reactant(feature_1).view(batch_size, d_f1, n_head, d_reactant)
        value = self.value(feature_1).view(batch_size, 1, n_head, d_f1)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, d_f1, d_reactant)
        key = key.premute(2, 0, 1, 3).contiguous().view(-1, d_f1, d_reactant)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_f1)

        output, attn = self.attention(query, key, value)

        output = output.view(n_head, batch_size, 1, d_f1)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        if self.use_bottleneck:
            output = self.bottleneck(output)

        return output, attn

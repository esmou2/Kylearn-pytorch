import torch
import torch.nn as nn
import numpy as np


class BottleneckLayer(nn.Module):
    ''' Bottleneck Layer '''

    def __init__(self, d_in, d_hid, d_out=None, dropout=0.1):
        super().__init__()
        if d_out == None:
            d_out=d_in

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
        decode = self.w_2(encode)
        output = self.dropout(decode)
        output = self.layer_norm(output + residual)
        return output


class ReactionDotProduction(nn.Module):
    ''' Scaled Dot Productionss '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, expansion, reactant, value):
        '''
            Arguments:
                expansion {Tensor, shape [n_head * batch, d_f1, d_reactant]} -- expansion
                reactant {Tensor, shape [n_head * batch, d_reactant, 1]} -- reactant
                value {Tensor, shape [n_head * batch, d_f1, 1]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, d_f1, 1]} -- output
                attn {Tensor, shape [n_head * batch, d_f1, 1]} -- reaction attention
        '''
        attn = torch.bmm(expansion, reactant)  # [n_head * batch, d_f1, 1]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.mul(attn, value)

        return output, attn


class ReactionAttentionLayer(nn.Module):
    '''Reaction Attention'''

    def __init__(self, n_head, d_reactant, d_f1, d_f2, d_hid=256, dropout=0.1, use_bottleneck=True):
        super().__init__()
        self.d_f1 = d_f1
        self.d_f2 = d_f2
        self.n_head = n_head
        self.d_reactant = d_reactant
        self.use_bottleneck = use_bottleneck
        self.expansion = nn.Linear(d_f1, n_head * d_f1 * d_reactant)
        self.reactant = nn.Linear(d_f2, n_head * d_reactant)
        self.value = nn.Linear(d_f1, n_head * d_f1)

        nn.init.kaiming_normal(self.expansion.weight)
        nn.init.kaiming_normal(self.reactant.weight)
        nn.init.kaiming_normal(self.value.weight)

        self.attention = ReactionDotProduction(temperature=np.power(d_reactant, 0.5))

        self.layer_norm = nn.LayerNorm(d_f1)

        self.fc = nn.Linear(n_head * d_f1, d_f1)
        nn.init.kaiming_normal(self.fc.weight)

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
                attn {Tensor, shape [n_head * batch, d_f1, 1]} -- self attention
        '''
        d_f1, d_f2, n_head, d_reactant = self.d_f1, self.d_f2, self.n_head, self.d_reactant

        batch_size, _ = feature_1.size()

        residual = feature_1

        expansion = self.expansion(feature_1).view(batch_size, d_f1, n_head, d_reactant)
        reactant = self.reactant(feature_2).view(batch_size, 1, n_head, d_reactant)
        value = self.value(feature_1).view(batch_size, 1, n_head, d_f1)

        expansion = expansion.permute(2, 0, 1, 3).contiguous().view(-1, d_f1, d_reactant)
        reactant = reactant.premute(2, 0, 3, 1).contiguous().view(-1, d_reactant, 1)
        value = value.permute(2, 0, 3, 1).contiguous().view(-1, d_f1, 1)

        output, attn = self.attention(expansion, reactant, value)

        output = output.view(n_head, batch_size, d_f1, 1)
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
                value {Tensor, shape [n_head * batch, d_f1, 1]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, d_f1, 1] -- output
                attn {Tensor, shape [n_head * batch, d_f1, d_f1] -- reaction attention
        '''
        attn = torch.bmm(query, key.transpose(2, 1))  # [n_head * batch, d_f1, d_f1]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.bmm(attn, value)

        return output, attn


class SelfAttentionLayer(nn.Module):
    '''Self Attention'''

    def __init__(self, n_head, d_reactant, d_f1, d_hid=256, dropout=0.1, use_bottleneck=True):
        super().__init__()
        self.d_f1 = d_f1
        self.n_head = n_head
        self.d_reactant = d_reactant
        self.use_bottleneck = use_bottleneck

        self.query = nn.Linear(d_f1, n_head * d_f1 * d_reactant)
        self.key = nn.Linear(d_f1, n_head * d_f1 * d_reactant)
        self.value = nn.Linear(d_f1, n_head * d_f1)

        nn.init.kaiming_normal(self.query.weight)
        nn.init.kaiming_normal(self.key.weight)
        nn.init.kaiming_normal(self.value.weight)

        self.attention = ScaledDotProduction(temperature=np.power(d_reactant, 0.5))

        self.layer_norm = nn.LayerNorm(d_f1)

        self.fc = nn.Linear(n_head * d_f1, d_f1)
        nn.init.kaiming_normal(self.fc.weight)

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
        value = value.permute(2, 0, 3, 1).contiguous().view(-1, d_f1, 1)

        output, attn = self.attention(query, key, value)

        output = output.view(n_head, batch_size, d_f1, 1)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        if self.use_bottleneck:
            output = self.bottleneck(output)

        return output, attn

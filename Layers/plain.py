import torch
import torch.nn as nn
import numpy as np

class PlainConvLayer(nn.Module):
    def __init__(self, n_channels, d_features, dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(1, n_channels, kernel_size=3, bias=False)
        self.conv2 = nn.Conv1d(n_channels, 1, kernel_size=3, bias=False)
        nn.init.xavier_normal(self.conv1.weight)
        nn.init.xavier_normal(self.conv2.weight)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_features)


    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch, 1, d_features]}
            Returns:
                output {Tensor, shape [batch, 1, d_features]}
        '''
        residual = x
        output = self.conv1(x)  # [batch, n_channels, d_features]
        output = self.activation(output)
        output = self.layer_norm(output)
        output = self.conv2(output)
        output = self.activation(output)
        output = self.layer_norm(output + residual)

        return output





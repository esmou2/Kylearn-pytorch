import torch
import torch.nn as nn
import numpy as np

class ResNetLayer(nn.Module):
    # def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
    def __init__(self, in_channel, out_channel, stride,
                 use_bottleneck=False, d_bottleneck=128):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        '''
            Arguments:
                enc_input {Tensor, shape [batch, length, d_features]} -- input
                non_pad_mask {Tensor, shape [batch, length, 1]} -- index of which position in a sequence is a padding
                slf_attn_mask {Tensor, shape [batch, length, length]} -- self attention mask

            Returns:
                enc_output {Tensor, shape [batch, length, d_features]} -- output
                encoder_self_attn {n_head * batch, length, length} -- n_head self attention matrices
        '''

        enc_output, encoder_self_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.bottleneck(enc_output)
        enc_output *= non_pad_mask

        return enc_output, encoder_self_attn
import torch.nn as nn
from Layers.resnet import ResNetLayer
import numpy as np

class ResNet(nn.Module):
    ''' A encoder models with self attention mechanism. '''

    def __init__(
            self, position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, d_k=None, d_v=None, dropout=0.1, use_bottleneck=True, d_bottleneck=256):
        super().__init__()

        if d_k == None or d_v == None:
            if d_k == d_v:
                d_reduce_param = np.floor(d_features / n_head).astype(int)
                d_k, d_v = d_reduce_param, d_reduce_param
            elif d_k == None:
                d_k = d_v
            else:
                d_v = d_k

        self.position_enc = position_encoding_layer(d_features=d_features, max_length=max_seq_length, d_meta=d_meta)

        self.layer_stack = nn.ModuleList([
            ResNet(d_features, n_head, d_k, d_v, dropout, use_bottleneck=use_bottleneck, d_bottleneck=d_bottleneck)
            for _ in range(n_layers)])

    def forward(self, feature_sequence, position, non_pad_mask=None, slf_attn_mask=None):
        '''
            Arguments:
                input_feature_sequence {Tensor, shape [batch, max_sequence_length, d_features]} -- input feature sequence
                position {Tensor, shape [batch, max_sequence_length (, d_meta)]} -- input feature position sequence
                non_pad_mask {Tensor, shape [batch, length, 1]} -- index of which position in a sequence is a padding
                slf_attn_mask {Tensor, shape [batch, length, length]} -- self attention mask

            Returns:
                enc_output {Tensor, shape [batch, max_sequence_length, d_features]} -- encoder output (representation)
                encoder_self_attn_list {List, length: n_layers} -- encoder self attention list,
                each element is a Tensor with shape [n_head * batch, max_sequence_length, max_sequence_length]
        '''

        encoder_self_attn_list = []
        enc_output = feature_sequence

        # Add position information at the beginning
        pos_enc = self.position_enc(position)
        enc_output = feature_sequence + pos_enc

        for enc_layer in self.layer_stack:
            enc_output, encoder_self_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            encoder_self_attn_list += [encoder_self_attn]

        return enc_output, encoder_self_attn_list
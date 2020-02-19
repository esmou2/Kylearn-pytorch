import torch.nn as nn
from Layers.transformer import EncoderLayer, DecoderLayer
import numpy as np

class Plain(nn.Module):
    ''' A Plain module. '''

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
            EncoderLayer(d_features, n_head, d_k, d_v, dropout, use_bottleneck=use_bottleneck, d_bottleneck=d_bottleneck)
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

        # Add position information at the beginning
        pos_enc = self.position_enc(position)
        enc_output = feature_sequence + pos_enc

        enc_output = enc_output.flatten(start_dim=1)

        return enc_output, encoder_self_attn_list

class Encoder(nn.Module):
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
            EncoderLayer(d_features, n_head, d_k, d_v, dropout, use_bottleneck=use_bottleneck, d_bottleneck=d_bottleneck)
            for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_features)

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

        # Add position information at the beginning
        pos_enc = self.position_enc(position)
        enc_output = feature_sequence + pos_enc
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, encoder_self_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            encoder_self_attn_list += [encoder_self_attn]

        enc_output = self.layer_norm(enc_output)

        return enc_output, encoder_self_attn_list


class Decoder(nn.Module):
    ''' A decoder models with self attention mechanism. '''

    def __init__(
            self, position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, dropout=0.1, use_bottleneck=True, d_bottleneck=256):

        super().__init__()

        self.position_enc = position_encoding_layer(d_features=d_features, max_length=max_seq_length, d_meta=d_meta)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_features, n_head, dropout, use_bottleneck=use_bottleneck, d_bottleneck=d_bottleneck)
            for _ in range(n_layers)])

    def forward(self, target_feature_sequence, tg_position, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        '''
            Arguments:
                target_feature_sequence {Tensor, shape [batch, max_sequence_length, d_features]} -- input feature sequence
                tg_position {Tensor, shape [batch, max_sequence_length (, d_meta)]} -- input feature position sequence
                enc_output {Tensor, shape [batch, max_sequence_length, d_features]} -- encoder output

            Returns:
                dec_output {Tensor, shape [batch, max_sequence_length, d_features]} -- decoder output (representation)
                dec_slf_attn_list {List, length: n_layers} -- decoder self attention list,
                each element is a Tensor with shape [n_head * batch, max_sequence_length, max_sequence_length]
                dec_enc_attn_list {List, length: n_layers} -- decoder-encoder attention list,
                each element is a Tensor with shape [n_head * batch, max_sequence_length, max_sequence_length]
        '''

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = target_feature_sequence + self.position_enc(tg_position)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            dec_slf_attn_list += [dec_slf_attn]
            dec_enc_attn_list += [dec_enc_attn]

        return dec_output, dec_slf_attn_list, dec_enc_attn_list


class Transformer(nn.Module):
    ''' A sequence to sequence models with attention mechanism. '''

    def __init__(
            self, position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, dropout=0.1, use_bottleneck=True,
            d_bottleneck=256):

        super().__init__()

        self.encoder = Encoder(position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, dropout, use_bottleneck, d_bottleneck)

        self.decoder = Decoder(position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, dropout, use_bottleneck, d_bottleneck)

        # Logit scaler
        # self.x_logit_scale = (d_features ** -0.5)
        # self.x_logit_scale = 1.

    def forward(self, input_feature_sequence, in_position, target_feature_sequence, tg_position):
        '''
            Arguments:
                input_feature_sequence {Tensor, shape [batch, max_sequence_length, d_features]} -- input feature sequence
                in_position {Tensor, shape [batch, max_sequence_length, d_meta]} -- input feature position sequence

            Returns:
                dec_output {Tensor, shape [batch, ?]} -- decoder output (representation)
        '''

        # target_feature_sequence, tg_position = target_feature_sequence[:, :-1], tg_position[:, :-1]
        enc_output, encoder_self_attn_list = self.encoder(input_feature_sequence, in_position, non_pad_mask=None, slf_attn_mask=None)
        dec_output, dec_slf_attn_list, dec_enc_attn_list = self.decoder(target_feature_sequence, tg_position, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None)

        # dec_output = self.fc(dec_output) * self.x_logit_scale
        return dec_output


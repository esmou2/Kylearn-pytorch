import torch
import torch.nn as nn
import numpy as np


def get_non_pad_mask(seq, padding_idx=0):
    '''
        Arguments:
            seq {batch, length} -- the word index of the sequence
            padding_idx -- padding symbol

        Returns:
            mask {Tensor, shape: [batch, length, 1]} -- non padding mask, 1 means not padding position while 0 means padding position
    '''
    assert seq.dim() == 2
    mask = seq.ne(padding_idx).type(torch.float)
    return mask.unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, padding_idx=0):
    '''
        For masking out the padding part of key sequence.
        Arguments:
            seq_k {Tensor, shape [batch, len_k]} -- key sequence
            seq_q {Tensor, shape [batch, len_q]} -- query sequence

        Returns:
            padding_mask {Tensor, shape [batch, len_q, len_k]} -- mask matrix, if the corresponding position is padding then True
            key mask [batch, k] -> expand q times [batch, len_q, len_k]

    '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(padding_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class EncoderLayer(nn.Module):
    # def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
    def __init__(self, d_features, n_head, d_k, d_v, dropout,
                 use_bottleneck=False, d_bottleneck=128):
        super().__init__()

        # To reduce computation
        # d_reduce_param = np.floor(d_features / n_head).astype(int)

        self.self_attn = MultiHeadAttention(n_head, d_features, d_k, d_v, dropout)
        if use_bottleneck:
            self.bottleneck = PositionwiseFeedForward(d_features, d_bottleneck, dropout=dropout)

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
        ## enc_output *= non_pad_mask

        enc_output = self.bottleneck(enc_output)
        ## enc_output *= non_pad_mask

        return enc_output, encoder_self_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    # def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
    def __init__(self, d_features, n_head, dropout,
                 use_bottleneck=False, d_bottleneck=128):
        super().__init__()

        # To reduce computation
        d_reduce_param = np.floor(d_features / n_head).astype(int)

        self.slf_attn = MultiHeadAttention(n_head, d_features, d_reduce_param, d_reduce_param, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_features, d_reduce_param, d_reduce_param, dropout=dropout)
        if use_bottleneck:
            self.bottleneck = PositionwiseFeedForward(d_features, d_bottleneck, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        '''
            Arguments:
                dec_input {Tensor, shape [batch, in_length, d_features]} -- decoder input
                enc_output {Tensor, shape [batch, tg_length, d_features]} -- encoder output
                non_pad_mask {Tensor, shape [batch, tg_length, 1]} -- index of which position in a sequence is a padding
                slf_attn_mask {Tensor, shape [batch, tg_length, tg_length]} -- self attention mask
                dec_enc_attn_mask {Tensor, shape [batch, tg_length, in_length]}
            Returns:
                dec_output {Tensor, shape [batch_size, seq_length, d_features]} -- output
                dec_slf_attn {Tensor, shape [n_head, seq_length, 1]} -- decoder self attention
                dec_enc_attn {Tensor, shape [batch_size, seq_length, seq_length]} -- decoder-encoder attention
        '''

        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output) # wider the network
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''
#
#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(attn_dropout)
#         self.softmax = nn.Softmax(dim=2)
#
#     def forward(self, query, key, value, mask=None):
#         '''
#             Arguments:
#                 query {Tensor, shape [n_head * batch, q_length, dk]} -- query
#                 key {Tensor, shape [n_head * batch, k_length, dk]} -- key
#                 value {Tensor, shape [n_head * batch, v_length, dv]} -- value
#                 mask {Tensor, shape [n_head * batch, q_length, k_length]} --self attn mask
#
#             Returns:
#                 output {Tensor, shape [n_head * batch, q_length, dv] -- output
#                 attn {Tensor, shape [n_head * batch, q_length, k_length] -- self attention
#
#         '''
#         attn = torch.bmm(query, key.transpose(1, 2)) # [n_head * batch, q_length, k_length]
#         attn = attn / self.temperature
#
#         if mask is not None:
#             attn = attn.masked_fill(mask, -np.inf)
#
#         attn = self.softmax(attn) # softmax over k_length
#         attn = self.dropout(attn)
#         output = torch.bmm(attn, value)
#
#         return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.functional.softmax

    def forward(self, query, key, value, mask=None):
        '''
            Arguments:
                query {Tensor, shape [ batch, n_head, q_length, dk]} -- query
                key {Tensor, shape [ batch, n_head, k_length, dk]} -- key
                value {Tensor, shape [ batch, n_head, k_length, dv]} -- value
                mask {Tensor, shape [batch, 1, q_length, k_length]} --self attn mask

            Returns:
                output {Tensor, shape [n_head * batch, q_length, dv] -- output
                attn {Tensor, shape [n_head * batch, q_length, k_length] -- self attention

        '''
        attn = torch.matmul(query / self.temperature, key.transpose(2, 3)) # [batch, n_head, q_length, k_length]

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn, dim=-1) # softmax over k_length
        attn = self.dropout(attn)  # [batch, n_head, q_length, k_length]
        output = torch.matmul(attn, value)  # [batch, n_head, q_length, dv]
        # output = value

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_features, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_features, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_features, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_features, n_head * d_v, bias=False)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_features + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_features + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_features + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_features)

        self.fc = nn.Linear(n_head * d_v, d_features, bias=False)
        # nn.init.xavier_normal_(self.fc.weight)


    def forward(self, query, key, value, mask=None):
        '''
            Arguments:
                query {Tensor, shape [batch, q_length, d_features]} -- query
                key {Tensor, shape [batch, k_length, d_features]} -- key
                value {Tensor, shape [batch, v_length, d_features]} -- value
                mask {Tensor, shape [batch, q_length, k_length]} --self attn mask

            Returns:
                output {Tensor, shape [batch, q_length, d_features]} -- output
                attn {Tensor, shape [n_head * batch, q_length, k_length] -- self attention
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = query.size()
        sz_b, len_k, _ = key.size()
        sz_b, len_v, _ = value.size()

        assert len_k == len_v

        residual = query

        query = self.layer_norm(query)

        query = self.w_qs(query).view(sz_b, len_q, n_head, d_k)  # target
        # [batch_size, seq_length, w2v_length] -> [batch_size, seq_length, n_head * dk] -> [batch_size, seq_length, n_head, dk]
        key = self.w_ks(key).view(sz_b, len_k, n_head, d_k)
        value = self.w_vs(value).view(sz_b, len_v, n_head, d_v)

        # query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [n_head * batch_size, seq_length, dk]
        # key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        # value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        query = query.transpose(1, 2).contiguous()  # [batch_size, n_head, seq_length, dq]
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        if len(mask) != 0:
            # mask = mask.repeat(n_head, 1, 1)  # [n_head * batch_size, seq_length, seq_length]
            mask = mask.unsqueeze(1)
        output, attn = self.attention(query, key, value, mask=mask)  # [batch, n_head, q_length, dv]

        # output = output.view(sz_b, n_head, len_q, d_v)
        # output = output.permute(0, 2, 1, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_v, -1)
        output = self.fc(output)
        output = self.dropout(output)
        output = output + residual

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch_size, length, d_features]}

            Returns:
                x {Tensor, shape [batch_size, length, d_features]}

        '''

        residual = x
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = nn.functional.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x += residual

        return x

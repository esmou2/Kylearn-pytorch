import torch
import torch.nn as nn
import numpy as np


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        '''
            Arguments:
                enc_input {Tensor, shape [batch, length, embedding_length]} -- input
                non_pad_mask {Tensor, shape [batch, length, 1]} -- index of which position in a sequence is a padding
                slf_attn_mask {Tensor, shape [batch, q_length, k_length]} -- self attention mask
            Returns:
                enc_output {Tensor, shape [batch, q_length, embedding_length]} -- output
                encoder_self_attn {n_head * batch, q_length, k_length}

        '''
        enc_output, encoder_self_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output) # wider the network
        enc_output *= non_pad_mask

        # enc_output:       [batch_size, seq_length, w2v_length]
        # non_pad_mask:     [n_head, seq_length, 1]
        # slf_attn_mask:    [batch_size, seq_length, seq_length]
        return enc_output, encoder_self_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        # dec_input:        [batch_size, tgt_length, w2v_length]
        # non_pad_mask:     [batch_size, tgt_length, 1]
        # slf_attn_mask:    [batch_size, tgt_length, seq_length]
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output) # wider the network
        dec_output *= non_pad_mask

        # dec_output:       [batch_size, seq_length, w2v_length]
        # non_pad_mask:     [n_head, seq_length, 1]
        # slf_attn_mask:    [batch_size, seq_length, seq_length]
        return dec_output, dec_slf_attn, dec_enc_attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, mask=None):
        '''
            Arguments:
                query {Tensor, shape [n_head * batch, q_length, dk]} -- query
                key {Tensor, shape [n_head * batch, k_length, dk]} -- key
                value {Tensor, shape [n_head * batch, v_length, dv]} -- value
                mask {Tensor, shape [n_head * batch, q_length, k_length]} --self attn mask

            Returns:
                output {Tensor, shape [n_head * batch, q_length, dv] -- output
                attn {Tensor, shape [n_head * batch, q_length, k_length] -- self attention

        '''
        attn = torch.bmm(query, key.transpose(1, 2)) # [n_head * batch, q_length, k_length]
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn) # softmax over k_length
        attn = self.dropout(attn)
        output = torch.bmm(attn, value)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        '''
            Arguments:
                query {Tensor, shape [batch, q_length, embedding_length]} -- query
                key {Tensor, shape [batch, k_length, embedding_length]} -- key
                value {Tensor, shape [batch, v_length, embedding_length]} -- value
                mask {Tensor, shape [batch, q_length, k_length]} --self attn mask

            Returns:
                output {Tensor, shape [batch, q_length, embedding_length]} -- output
                attn {Tensor, shape [n_head * batch, q_length, k_length] -- self attention
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = query.size()
        sz_b, len_k, _ = key.size()
        sz_b, len_v, _ = value.size()

        assert len_k == len_v

        residual = query

        query = self.w_qs(query).view(sz_b, len_q, n_head, d_k)  # target
        # [batch_size, seq_length, w2v_length] -> [batch_size, seq_length, n_head * dk] -> [batch_size, seq_length, n_head, dk]
        key = self.w_ks(key).view(sz_b, len_k, n_head, d_k)
        value = self.w_vs(value).view(sz_b, len_v, n_head, d_v)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [n_head * batch_size, seq_length, dk]
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # [n_head * batch_size, seq_length, seq_length]
        output, attn = self.attention(query, key, value, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            Arguments:
            x {Tensor, shape [batch_size, q_length, embedding_length]}

        '''
        residual = x
        output = x.transpose(1, 2)
        output = nn.functional.relu(self.w_1(output))
        output = self.w_2(output)
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

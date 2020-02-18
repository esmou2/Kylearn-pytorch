import torch
import torch.nn as nn
import numpy as np

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''
        Arguments:
            n_position {Int} -- the maximum position
            d_hid {Int} -- the dimension of the embedding
            padding_idx -- padding symbol

        Returns:
            sinusoid_table {Tensor, shape: [n_position+1, d_hid]} -- sinusoid encoding table
    '''
    n_position += 1

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class SinusoidPositionEncoding(nn.Module):

    def __init__(self, d_features, max_length=None, d_meta = None):
        super().__init__()
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_length, d_features, padding_idx=0),
            freeze=True)

    def forward(self, x):
        '''
            Argument:
                x {Tensor, shape: [batch, length]} -- sequence position index masked
            Returns:
                x {Tensor, shape: [batch, length, d_features]} -- positional encoding
        '''
        x = self.position_enc(x)
        pass
        return x

class LinearPositionEncoding(nn.Module):

    def __init__(self, d_features, max_length=None, d_meta = None):
        super().__init__()
        self.position_enc = nn.Linear(d_meta, d_features, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.position_enc(x)
        x = self.tanh(x)
        return x


class TimeFacilityEncoding(nn.Module):

    def __init__(self, d_features, max_length=None, d_meta = None):
        super().__init__()
        self.time_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_length, d_features, padding_idx=0),
            freeze=True)
        torch.manual_seed(1)
        self.facility_enc = torch.nn.Embedding(d_meta+1, d_features, padding_idx=0)
        self.facility_enc.weight.requires_grad = False


    def forward(self, x):
        '''
            Argument:
                x {Tensor, shape: [batch, length, 2]} -- sequence position index masked
            Returns:
                x {Tensor, shape: [batch, length, d_features]} -- positional encoding
        '''
        facility_index = x[:, :, 1].long()
        facility_mask = facility_index == 0
        facility = self.facility_enc(facility_index)
        time = x[:, :, 0].masked_fill(facility_mask, 0)
        time = self.time_enc(time)

        x = time + facility
        return x
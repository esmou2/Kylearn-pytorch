import torch.nn as nn

class LinearBottleneckLayer(nn.Module):
    ''' Bottleneck Layer '''

    def __init__(self, d_features, d_hid, d_out=None, dropout=0.1):
        super().__init__()
        if d_out == None:
            d_out = d_features

        self.encode = nn.Linear(d_features, d_hid)
        self.decode = nn.Linear(d_hid, d_out)
        nn.init.xavier_normal_(self.encode.weight)
        nn.init.xavier_normal_(self.decode.weight)
        self.layer_norm = nn.LayerNorm(d_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch_size, d_features]}
            Returns:
                x {Tensor, shape [batch_size, d_features]}
        '''
        residual = x
        encode = nn.functional.relu(self.encode(x))
        decode = self.decode(encode)
        output = self.dropout(decode)
        output = self.layer_norm(output + residual)
        output = output + residual
        return output

class ShuffleBottleneckLayer(nn.Module):
    ''' Bottleneck Layer '''

    def __init__(self, n_depth, d_features, mode, d_hid=None, dropout=0.1):
        super().__init__()
        self.n_depth = n_depth
        self.d_features = d_features
        self.mode = mode
        if d_hid == None:
            d_hid = d_features

        if mode == '1d':
            self.bottle_neck_1 = nn.Linear(d_features, d_hid)
            self.bottle_neck_2 = nn.Linear(d_hid, d_features)

        elif mode == '2d':
            # self.bottle_neck_1 = nn.Conv1d(n_depth, d_hid, kernel_size=1, bias=False)
            # self.bottle_neck_2 = nn.Conv1d(d_hid, n_depth, kernel_size=1, bias=False)
            self.bottle_neck_1 = nn.Conv1d(d_features, d_hid, kernel_size=1)
            self.bottle_neck_2 = nn.Conv1d(d_hid, d_features, kernel_size=1)
        else:
            pass

        nn.init.xavier_normal_(self.bottle_neck_1.weight)
        nn.init.xavier_normal_(self.bottle_neck_2.weight)
        self.layer_norm = nn.LayerNorm([d_features])

        self.activation = nn.functional.relu

        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        '''
            Arguments:
                features {Tensor, shape [batch, d_features] or [batch, n_depth, d_features]} -- features

            Returns:
                x {Tensor, shape [batch_size, d_features]}
        '''
        residual = features

        if self.mode == '1d':
            output = self.bottle_neck_1(features)
            output = self.activation(output)
            output = self.bottle_neck_2(output)


        elif self.mode == '2d':
            output = features.transpose(1, 2)
            output = self.bottle_neck_1(output)
            output = self.activation(output)
            output = self.bottle_neck_2(output)
            output = output.transpose(2, 1)
        else:
            residual = 0
            output = features

        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

class ShuffleBottleneckLayerV2(nn.Module):
    ''' Bottleneck Layer '''

    def __init__(self, n_depth, d_features, mode, d_hid=None, dropout=0.1, c_total=3):
        super().__init__()
        self.n_depth = n_depth
        self.d_features = d_features
        self.mode = mode
        if d_hid == None:
            d_hid = d_features

        if mode == 'residual':
            self.bottle_neck_1 = nn.Conv1d(d_features, d_hid, kernel_size=1)
            self.bottle_neck_2 = nn.Conv1d(d_hid, d_features, kernel_size=1)

        elif mode == 'dense':
            self.bottle_neck_1 = nn.Conv1d(c_total, d_hid, kernel_size=1)
            self.bottle_neck_2 = nn.Conv1d(d_hid, c_total, kernel_size=1)
        else:
            pass

        nn.init.xavier_normal_(self.bottle_neck_1.weight)
        nn.init.xavier_normal_(self.bottle_neck_2.weight)
        self.layer_norm = nn.LayerNorm([d_features])

        self.activation = nn.functional.relu

        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        '''
            Arguments:
                features {Tensor, shape [batch, ?, d_features]} -- features

            Returns:
                x {Tensor, shape [batch_size, d_features]}
        '''
        residual = features

        if self.mode == 'residual':
            # features {Tensor, shape [batch, 3, d_features]}
            output = features.transpose(1, 2)
            output = self.bottle_neck_1(output)
            output = self.activation(output)
            output = self.bottle_neck_2(output)
            output = output.transpose(2, 1)

        elif self.mode == 'dense':
            # features {Tensor, shape [batch, c_total, d_features]}
            output = self.bottle_neck_1(features)  # [batch, d_hid, d_features]
            output = self.activation(output)
            output = self.bottle_neck_2(output)  # [batch, c_total, d_features]
            output = output.transpose(2, 1)
        else:
            residual = 0
            output = features

        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output
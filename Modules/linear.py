import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(d_in, d_hid)
        self.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(d_hid, d_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
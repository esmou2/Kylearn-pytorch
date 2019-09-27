import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, input_shape, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(input_shape))
        self.beta = nn.Parameter(torch.zeros(input_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
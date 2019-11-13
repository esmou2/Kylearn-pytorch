import torch.nn as nn
from Layers.plain import PlainConvLayer
class PlainConvStack(nn.Module):
    ''' Self Attention Stack Module '''

    def __init__(
            self, expansion, d_features, n_layers, n_channels, dropout=0.1, use_bottleneck=None, d_bottleneck=None):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            PlainConvLayer(n_channels, d_features, dropout)
            for _ in range(n_layers)])

    def forward(self, x):
        for layers in self.layer_stack:
            x = layers(
                x)

        return x
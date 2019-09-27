import torch
import torch.nn as nn
from Layers.normalization import LayerNorm
from Layers.transformer import EncoderLayer

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, n_layers, input_shape, output_shape, dropout):
        super().__init__()
        self.layer_stack = nn.ModuleList([EncoderLayer(input_shape, output_shape, dropout=dropout)
                                          for _ in range(n_layers)])
        self.layer_norm = LayerNorm(input_shape)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layer_stack:
            x = layer(x, mask)
        return self.layer_norm(x)

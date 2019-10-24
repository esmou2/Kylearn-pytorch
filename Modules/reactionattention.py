import torch
import torch.nn as nn
import numpy as np
from Layers.reactionattention import *


class ReactionAttentionStack(nn.Module):
    ''' Reaction Attention Stack Module '''

    def __init__(
            self, expansion_layer, n_layers, n_head, d_features, d_meta, n_depth, d_hid=256, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            ReactionAttentionLayerV1(expansion_layer, n_head, n_depth, d_features, d_meta, d_bottleneck=d_hid, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, features, meta_features, return_attns=False):

        reaction_attn_list = []

        for ra_layer in self.layer_stack:
            features, reaction_attn = ra_layer(
                features, meta_features)
            if return_attns:
                reaction_attn_list += [reaction_attn]

        if return_attns:
            return features, reaction_attn_list
        return features,


class SelfAttentionStack(nn.Module):
    ''' Self Attention Stack Module '''

    def __init__(
            self, expansion_layer, n_layers, n_head, d_f1, n_depth, d_hid=256, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            SelfAttentionLayer(expansion_layer, n_head, n_depth, d_f1, d_hid=d_hid, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, feature_1, feature_2=None, return_attns=False):

        self_attn_list = []

        for sa_layer in self.layer_stack:
            feature_1, self_attn = sa_layer(
                feature_1, feature_2)
            if return_attns:
                self_attn_list += [self_attn]

        if return_attns:
            return feature_1, self_attn_list
        return feature_1,


class AlternateStack(nn.Module):
    ''' Alternately stack the 2 attention blocks '''

    def __init__(
            self, expansion_layer, n_layers, n_head, d_f1, d_f2, n_depth, d_hid=256, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            ReactionAttentionLayerV1(expansion_layer, n_head, n_depth, d_f1, d_f2, d_bottleneck=d_hid,
                                     dropout=dropout) if i % 2 == 0
            else SelfAttentionLayer(expansion_layer, n_head, n_depth, d_f1, d_hid=d_hid, dropout=dropout)
            for i in range(n_layers)])

    def forward(self, feature_1, feature_2, return_attns=False):

        alternate_attn_list = []

        for attn_layer in self.layer_stack:
            feature_1, alternate_attn = attn_layer(
                feature_1, feature_2)
            if return_attns:
                alternate_attn_list += [alternate_attn]

        if return_attns:
            return feature_1, alternate_attn_list
        return feature_1,


class ParallelStack(nn.Module):
    ''' Stack the 2 attention blocks in parallel '''

    def __init__(
            self, expansion_layer, n_layers, n_head, d_f1, d_f2, n_depth, d_hid=256, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers

        self.ra_stack = nn.ModuleList([
            ReactionAttentionLayerV1(expansion_layer, n_head, n_depth, d_f1, d_f2, dropout=dropout,
                                     use_bottleneck=False)
            for _ in range(n_layers)])

        self.sa_stack = nn.ModuleList([
            SelfAttentionLayer(expansion_layer, n_head, n_depth, d_f1, dropout=dropout, use_bottleneck=False)
            for _ in range(n_layers)])

        self.bottleneck_stack = nn.ModuleList([
            LinearBottleneckLayer(2 * d_f1, d_hid, d_out=d_f1)
            for _ in range(n_layers)])

    def forward(self, feature_1, feature_2, return_attns=False):

        ensemble_attn_list = []

        for layer_index in range(self.n_layers):
            feature_1_ra, attn_ra = self.ra_stack[layer_index](feature_1, feature_2)
            feature_1_sa, attn_sa = self.sa_stack[layer_index](feature_1)
            feature_1 = torch.cat((feature_1_ra, feature_1_sa), dim=-1)
            feature_1 = self.bottleneck_stack(feature_1)
            ensemble_attn = torch.cat((attn_ra, attn_sa), dim=-2)
            if return_attns:
                ensemble_attn_list += [ensemble_attn]

        if return_attns:
            return feature_1, ensemble_attn_list
        return feature_1,


class ShuffleSelfAttentionStack(nn.Module):
    ''' Self Attention Stack Module '''

    def __init__(
            self,expansion_layer, n_depth, d_features, n_layers, n_head, n_channel, n_vchannel, d_bottleneck=256, dropout=0.1, mode='1d', use_bottleneck=True):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            ShuffleSelfAttentionLayer(expansion_layer, n_depth, d_features, n_head, n_channel, n_vchannel, dropout=dropout,
                                      mode=mode, use_bottleneck=use_bottleneck, d_bottleneck=d_bottleneck)
            for _ in range(n_layers)])

    def forward(self, feature_1, feature_2=None):

        self_attn_list = []

        for sa_layer in self.layer_stack:
            feature_1, self_attn = sa_layer(
                feature_1)
            self_attn_list += [self_attn]

        return feature_1, self_attn_list

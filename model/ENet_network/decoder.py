from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import InitialBlock, RegularBottleneck, DownsamplingBottleneck, UpsamplingBottleneck, Feature_Fuse
class ENet_Decoder(nn.Module):

    def __init__(self, classes = 19, decoder_relu = True):
        super(ENet_Decoder,self).__init__()
        #########
        #self.fuse_block = Feature_Fuse(128, 128, 128)

        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

        self.project_layer = nn.Conv2d(128, classes, 1, bias=False)

    def forward(self, x, x_o):
        #features_tofuse = [x[0], x_o]
        #x[0] = self.fuse_block(features_tofuse) 
        x, max_indices1_0, max_indices2_0 = x[0], x[1], x[2]

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)
        
        return x

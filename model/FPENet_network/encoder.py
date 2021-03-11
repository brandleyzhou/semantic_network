from __future__ import absolute_import
import torch.nn as nn
from  .layers import SEModule, FPEBlock, MEUModule


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class FPENet_Encoder(nn.Module):
    def __init__(self, zero_init_residual = False, width = 16, scales=4,se =False,norm_layer = None):
        super(FPENet_Encoder,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        outplanes = [int(width * 2 ** i) for i in range(3)] # planes=[16,32,64]
        self.block_num = [1,3,9]
        self.dilation = [1,2,4,8]

        self.inplanes = outplanes[0]
        self.conv1 = nn.Conv2d(3, outplanes[0], kernel_size=3, stride=2, padding=1,bias=False)
        self.bn1 = norm_layer(outplanes[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FPEBlock, outplanes[0], self.block_num[0], dilation=self.dilation,
                                       stride=1, t=1, scales=scales, se=se, norm_layer=norm_layer)
        self.layer2 = self._make_layer(FPEBlock, outplanes[1], self.block_num[1], dilation=self.dilation,
                                       stride=2, t=4, scales=scales, se=se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(FPEBlock, outplanes[2], self.block_num[2], dilation=self.dilation,
                                       stride=2, t=4, scales=scales, se=se, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, FPENet_Encoder):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, dilation, stride=1, t=1, scales=4, se=False, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, dilat=dilation, downsample=downsample, stride=stride, t=t, scales=scales, se=se,
                            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilat=dilation, scales=scales, se=se, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        ## stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_1 = self.layer1(x)
        features.append(x_1)
        
        ## stage 2
        x_2_0 = self.layer2[0](x_1)
        x_2_1 = self.layer2[1](x_2_0)
        x_2_2 = self.layer2[2](x_2_1)
        x_2 = x_2_0 + x_2_2
        features.append(x_2)

        ## stage 3
        x_3_0 = self.layer3[0](x_2)
        x_3_1 = self.layer3[1](x_3_0)
        x_3_2 = self.layer3[2](x_3_1)
        x_3_3 = self.layer3[3](x_3_2)
        x_3_4 = self.layer3[4](x_3_3)
        x_3_5 = self.layer3[5](x_3_4)
        x_3_6 = self.layer3[6](x_3_5)
        x_3_7 = self.layer3[7](x_3_6)
        x_3_8 = self.layer3[8](x_3_7)
        x_3 = x_3_0 + x_3_8
        features.append(x_3)
        
        return features


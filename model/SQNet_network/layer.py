from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1,stride=1)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        # self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ELU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        # out1 = self.bn2(out1)
        out2 = self.conv3(x)
        # out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class ParallelDilatedConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(ParallelDilatedConv, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=4, dilation=4)
        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        self.relu3 = nn.ELU(inplace=True)
        self.relu4 = nn.ELU(inplace=True)

    def forward(self, x):
        out1 = self.dilated_conv_1(x)
        out2 = self.dilated_conv_2(x)
        out3 = self.dilated_conv_3(x)
        out4 = self.dilated_conv_4(x)
        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        out3 = self.relu3(out3)
        out4 = self.relu4(out4)
        out = out1 + out2 + out3 + out4
        return out

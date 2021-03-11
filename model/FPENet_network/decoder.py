from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F
from  .layers import SEModule, FPEBlock, MEUModule

class FPENet_Decoder(nn.Module):

    def __init__(self,classes = 19, zero_init_residual=False):
        super(FPENet_Decoder,self).__init__()
        self.meu1= MEUModule(64,32,64)
        self.meu2= MEUModule(64,16,32)
        # Projection layer
        self.project_layer = nn.Conv2d(32, classes, kernel_size = 1)
        print('sss')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, FPENet_Decoder):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self,features):
        x2 = self.meu1(features[2], features[1])
        x1 = self.meu2(x2, features[0])
        output = self.project_layer(x1)

        # Bilinear interpolation x2
        output = F.interpolate(output,scale_factor=2, mode = 'bilinear', align_corners=True)
        return output


from __future__ import absolute_import, division, print_function

from layer import Fire,ParallelDilatedConv


class SQNet_Encoder(nn.Module):
    def __init__(self,classes):
        self.num_classes = classes
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1)

        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fire1_1 = Fire(96, 16, 64)
        self.fire1_2 = Fire(128, 16, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, tride=2)

        self.fire2_1 = Fire(128, 32, 128)
        self.fire2_2 = Fire(256, 32, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fire3_1 = Fire(256, 64, 256)
        self.fire3_2 = Fire(512, 64, 256)
        self.fire3_3 = Fire(512, 64, 256)
        self.parallel = ParallelDilatedConv(512, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self,x):
        features = {}

        x = self.conv1(x)
        x_1 = self.relu1(x)
        features['x_1'] = x_1
        # x_1
        x = self.maxpool1(x_1)
        x = self.fire1_1(x)
        x_2 = self.fire1_2(x)
        features['x_2'] = x_2
        
        # x_2
        x = self.maxpool2(x_2)
        x = self.fire2_1(x)
        x_3 = self.fire2_2(x)
        features['x_3'] = x_3

        # x_3
        x = self.maxpool3(x_3)
        x = self.fire3_1(x)
        x = self.fire3_2(x)
        x = self.fire3_3(x)
        x = self.parallel(x)
        # x
        features['x'] = x

        return features


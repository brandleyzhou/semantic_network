import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Encoder_Se_block(nn.Module):
    def __init__(self, in_scalar = 2, out_scalar = 2, increase = 16):
        super(Encoder_Se_block,self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(in_scalar, increase, bias = False),
                    nn.ReLU(inplace = True),
                    nn.Linear(increase, out_scalar, bias = True)
                    )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, feature, x_o):
        feature_max = torch.max(feature)
        x_max = torch.max(x_o)
        f_w = feature_max / (feature_max + x_max) 
        x_w = x_max / (feature_max + x_max)
        x = torch.tensor([f_w,x_w]).cuda()
        x = self.sigmoid(self.fc(x))
        
        return x

class InitialBlock(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, padding=0, bias=False,relu=True):
        super(InitialBlock, self).__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels-3,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias,
        )
        # MP need padding too
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_prelu = activation


    def forward(self, input):
        main = self.main_branch(input)
        ext = self.ext_branch(input)

        out = torch.cat((main, ext), dim=1)

        out = self.batch_norm(out)
        return self.out_prelu(out)

class RegularBottleneck(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0,
                 dilation=1, asymmetric=False, dropout_prob=0., bias=False, relu=True):
        super(RegularBottleneck, self).__init__()

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # 1x1 projection conv
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation,
        )
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size,1),
                          stride=1, padding=(padding,0), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation,
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1,kernel_size),
                          stride=1, padding=(0, padding), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation,
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation,
            )

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(channels),
            activation,
        )
        self.ext_regu1 = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation
    
    def forward(self, input):
         main = input

         ext = self.ext_conv1(input)
         ext = self.ext_conv2(ext)
         ext = self.ext_conv3(ext)
         ext = self.ext_regu1(ext)

         out = main + ext
         return self.out_prelu(out)

class DownsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 return_indices=False,
                 dropout_prob=0.,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            kernel_size,
            stride=2,
            padding=padding,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2, no padding
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels,internal_channels,kernel_size=2,stride=2,bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation
        )

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        # calculate for padding ch_ext - ch_main
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate, padding for less channels of main branch
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out), max_indices

class UpsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dropout_prob=0.,
                 bias=False,
                 relu=True):
        super().__init__()

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation)

        # Transposed convolution
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x, max_indices):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)

class Feature_Fuse(nn.Module):
    def __init__(self,in_channel_1, in_channel_2,out_channel):
        super(Feature_Fuse, self).__init__()
        self.in_channel = in_channel_1 + in_channel_2
        self.out_channel = out_channel

        self.conv = nn.Conv2d(self.in_channel,self.out_channel,kernel_size = 1, stride = 1)
        self.relu = nn.ReLU(inplace =True)

    def forward(self, features):
        x = torch.cat(features,1)
        x = self.relu(self.conv(x))
        return x

    



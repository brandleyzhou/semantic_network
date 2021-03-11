from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from .layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, classes = 19, use_skips=True, mobile_encoder=False):
        super(DepthDecoder, self).__init__()
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.num_ch_enc = num_ch_enc
        if mobile_encoder:
            self.num_ch_dec = np.array([4, 12, 20, 40, 80])
        else:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            if i == 0:
            # upconv_1
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            else:
                self.convs[("attentionConv", i)] = fSEModule(self.num_ch_dec[i], self.num_ch_enc[i - 1])

        #for s in self.scales:
        #    self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.output_conv = nn.Conv2d(16, classes, kernel_size = 1, stride = 1, bias =False)
    def forward(self, input_features):
        for i in input_features:
            print(i.size())

        input()
        middle_features = []
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)

            if  i != 0:
                x = self.convs[("attentionConv", i)](x, [input_features[i - 1]])
            else:
                middle_features.append(x)
                x = [upsample(x)]
                if self.use_skips and i > 0:
                    x += [input_features[i - 1]]
                x = torch.cat(x, 1)
                x = self.convs[("upconv", i, 1)](x)
            if i == 0:
                outputs = self.sigmoid(self.output_conv(x))

        return outputs

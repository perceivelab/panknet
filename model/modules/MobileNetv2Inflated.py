import torch.nn as nn
import torchvision.models as models
import collections

from model.modules.blocks import ConvBNReLU, InvertedResidual

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class MobileNetV2_3D_featExt(nn.Module):
    def __init__(self, input_size=224, width_mult=1., pretrained = True):
        super(MobileNetV2_3D_featExt, self).__init__()
        if pretrained:
            mobilenet_v2 = models.mobilenet_v2(pretrained=True)
            
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, (1,1,1)],#b1
            [6, 24, 2, (2,2,2)],#b2
            [6, 32, 3, (2,2,2)],#b3
            [6, 64, 4, (2,2,2)],#b4
            [6, 96, 3, (1,1,1)],#b5
            [6, 160, 3, (2,2,2)],#b6
            [6, 320, 1, (1,1,1)],#b7
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = collections.OrderedDict()
        idx=0
        self.features[str(idx)] = ConvBNReLU(1, input_channel, kernel = (3,3,3), stride = (1,2,2), padding = 1, pretrained = pretrained, block2d = mobilenet_v2.features[idx] if pretrained else [])
        idx += 1
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features[str(idx)]=InvertedResidual(input_channel, output_channel, s, expand_ratio=t, pretrained = pretrained, block2d = mobilenet_v2.features[idx] if pretrained else [])
                    idx += 1
                else:
                    self.features[str(idx)]=InvertedResidual(input_channel, output_channel, (1,1,1), expand_ratio=t, pretrained=pretrained, block2d = mobilenet_v2.features[idx] if pretrained else [])
                    idx += 1
                input_channel = output_channel
        # building last several layers
        self.features[str(idx)]= ConvBNReLU(input_channel, self.last_channel, kernel = 1, stride = 1, padding = 0, pretrained = pretrained, block2d = mobilenet_v2.features[idx] if pretrained else [])
        # make it nn.Sequential
        self.features = nn.Sequential(self.features)

    def forward(self, x):
        #x = self.features(x)
        b0 = self.features[0](x)
        b1 = self.features[1](b0)
        b2 = self.features[3](self.features[2](b1))
        b3 = self.features[6](self.features[5](self.features[4](b2)))
        b4 = self.features[10](self.features[9](self.features[8](self.features[7](b3))))
        b5 = self.features[13](self.features[12](self.features[11](b4)))
        b6 = self.features[16](self.features[15](self.features[14](b5)))
        #b7 = self.features[17](b6)
        #b8 = self.features[18](b7)
        return b2, b3, b4, b6 

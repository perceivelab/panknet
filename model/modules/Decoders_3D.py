import torch.nn as nn
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate

from model.modules.blocks import ConvBNReLU

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

    
class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.depthConv = ConvBNReLU(inp=in_planes, oup= out_planes, kernel=kernel_size, stride= stride, padding = padding, groups = in_planes)
        self.pointConv = ConvBNReLU(inp=out_planes, oup= out_planes, kernel=1, stride= 1, padding = 0, groups = 1)

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        return x 

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv_bnr = ConvBNReLU(inp=in_planes, oup= out_planes, kernel=kernel_size, stride= stride, padding = padding)

    def forward(self, x):
        x = self.conv_bnr(x)
        return x

class Deconv(nn.Sequential):
    def __init__(self, in_channel, out_channel, num_conv=2, kernel_size=3, stride=1, padding=1, upsample_depth = 1):
        super(Deconv, self).__init__()
        b_conv = BasicConv3d(in_channel, out_channel ,kernel_size=kernel_size, stride=stride, padding=padding)
        self.add_module('0', b_conv)
        for i in range(1,num_conv):
            conv = SepConv3d(out_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding)
            self.add_module(f'{i}', conv)
        upsample= Upsample(scale_factor=(upsample_depth,2,2), mode='trilinear')
        self.add_module('upsample', upsample)

class Decoder_3D(nn.Module):
    def __init__(self, in_channel, out_channel=[], list_num_conv=[], out_sigmoid=False, upsample_depth_list = []):
        super(Decoder_3D, self).__init__()
        self.out_sigmoid = out_sigmoid
        deconvBlock = []
        deconvBlock.append(Deconv(in_channel, out_channel[0], num_conv=list_num_conv[0], upsample_depth= upsample_depth_list[0] if upsample_depth_list else 1))
        for i in range(1, len(list_num_conv)):
            deconvBlock.append(Deconv(out_channel[i-1], out_channel[i], num_conv=list_num_conv[i], upsample_depth= upsample_depth_list[i] if upsample_depth_list else 1))
        self.deconvBlock = nn.Sequential(*deconvBlock)
        
        self.last_conv=nn.Conv3d(in_channels=out_channel[-2], out_channels = out_channel[-1], kernel_size=1, stride=1, bias=True)
        if self.out_sigmoid:
            self.sigmoid= nn.Sigmoid()
            
    def forward(self, x):
        x = self.deconvBlock(x)
        x = self.last_conv(x)
        
        if self.out_sigmoid:
            x=self.sigmoid(x)
        
        return x
        
        
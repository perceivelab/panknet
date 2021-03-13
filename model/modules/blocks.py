from torch import nn
from model.modules.inflate import inflate_conv, inflate_batch_norm

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size), stride=(1,stride,stride), padding=(0,padding,padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self, inp, oup, kernel = 3, stride = 1, padding= 1, groups = 1, pretrained = False, block2d = []):
        super(ConvBNReLU, self).__init__()
        self.groups = groups
        if isinstance(kernel, int):
            kernel = (kernel, kernel, kernel)
        elif isinstance(kernel, tuple) and len(kernel)==3 and all(isinstance(k, int) for k in kernel):
            kernel = kernel
        else:
            raise 'Invalid kernel param'
            
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        elif isinstance(stride, tuple) and len(stride)==3 and all(isinstance(s, int) for s in stride):
            stride = stride
        else:
            raise 'Invalid stride param'
            
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        elif isinstance(padding, tuple) and len(padding)==3 and all(isinstance(p, int) for p in padding):
            padding = padding
        else:
            raise 'Invalid stride param'
            
        if pretrained:
            conv = inflate_conv(inp, oup, block2d[0], kernel, stride, padding, groups=groups)
            batch = inflate_batch_norm(block2d[1])
        else:
            conv = nn.Conv3d(inp, oup, kernel_size = kernel, stride = stride, padding = padding, bias=False, groups=groups)
            batch = nn.BatchNorm3d(oup)
            
        relu = nn.ReLU6(inplace=True)
            
        #self.seq = nn.Sequential(*[conv, batch, relu])
        self.add_module('0', conv)
        self.add_module('1', batch)
        self.add_module('2', relu)
        
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, pretrained, block2d = []):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        #assert stride in [1, 2]
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if expand_ratio == 1:
            if pretrained:
                conv = inflate_conv(hidden_dim, oup, block2d.conv[1], kernel=1, stride=1, padding=0)
                batch = inflate_batch_norm(block2d.conv[2])
                block2d0= block2d.conv[0]
            else:
                conv = nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False)
                batch= nn.BatchNorm3d(oup)
                block2d0= []
            self.conv = nn.Sequential(*[
                # dw
                ConvBNReLU(hidden_dim,hidden_dim, kernel = 3, stride = stride, padding = 1 ,groups = hidden_dim, pretrained = pretrained, block2d=block2d0), 
                # pw-linear
                conv,
                batch
                ]
            )
        else:
            if pretrained:
                conv = inflate_conv(hidden_dim, oup, block2d.conv[2], kernel=1, stride=1, padding=0)
                batch = inflate_batch_norm(block2d.conv[3])
                block2d0= block2d.conv[0]
                block2d1= block2d.conv[1]
            else:
                conv = nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False)
                batch= nn.BatchNorm3d(oup)
                block2d0 = []
                block2d1 = []
            self.conv = nn.Sequential(*[
                # pw
                ConvBNReLU(inp,hidden_dim, kernel = 1, stride = 1, padding = 0, pretrained = pretrained, block2d=block2d0), 
                # dw
                ConvBNReLU(hidden_dim,hidden_dim, kernel = 3, stride = stride, padding = 1 ,groups = hidden_dim, pretrained = pretrained, block2d=block2d1), 
                # pw-linear
                conv,
                batch
            ])

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


#PointWise Convolution that only divide Depth dimension by 2
class DepthConv(nn.Sequential):
    def __init__(self, num_conv, ch):
        super(DepthConv, self).__init__()
        for i in range(num_conv):
            conv = ConvBNReLU(ch, ch, kernel=(3,1,1), stride=(2,1,1), padding=(1,0,0))
            self.add_module(f'{i}', conv)
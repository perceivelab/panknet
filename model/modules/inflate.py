import torch
from torch import nn

def inflate_conv(inp, oup, conv2d, kernel, stride, padding, groups=1):
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
            
    if conv2d.bias is None:
        bias = False
    else:
        bias = True
    conv3d = nn.Conv3d(inp, oup, kernel_size = kernel, stride = stride, padding = padding, bias = bias, groups=groups)
    if(conv3d.in_channels!=conv2d.in_channels and conv3d.in_channels==1):
        weight2d = torch.mean(conv2d.weight.data, dim = 1, keepdim=True )
    else:
        weight2d = conv2d.weight.data
    time_dim = kernel[2]
    weight3d = weight2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
    weight3d = weight3d / time_dim
    conv3d.weight =nn.Parameter(weight3d)
    if bias:
        conv3d.bias = conv2d.bias
    return conv3d


def inflate_batch_norm(batch2d):
    batch3d = nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d
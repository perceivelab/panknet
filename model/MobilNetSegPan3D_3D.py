# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

from model.modules.MobileNetv2Inflated import MobileNetV2_3D_featExt
from model.modules.Decoders_3D import Decoder_3D

class MobilNetSegPan3D_3D(nn.Module):
    def __init__(self, pretrained = True, out_sigmoid = False, out_ch = 1, multi_output = False, num_slices = 8):
        super(MobilNetSegPan3D_3D, self).__init__()
        self.multi_output= multi_output
        self.decoderInChannel = [24,32,64,160]
        self.out_sigmoid = out_sigmoid
        self.encoder = MobileNetV2_3D_featExt(pretrained = pretrained)
        #Decoder
        list_up = [1,2]
        list_up.append(2) if num_slices//2**1 >0 else list_up.append(1)
        #list_up = [1,2,2]
        self.decoder1 = Decoder_3D(in_channel=self.decoderInChannel[0], out_channel=[8,3, out_ch], list_num_conv =[2,2], out_sigmoid= out_sigmoid, upsample_depth_list = list_up)
        list_up.append(2) if num_slices//2**2 >0 else list_up.append(1)
        #list_up = [1,2,2,2]
        self.decoder2 = Decoder_3D(in_channel=self.decoderInChannel[1], out_channel=[16, 8, 3, out_ch], list_num_conv =[2,2,2], out_sigmoid= out_sigmoid, upsample_depth_list = list_up)
        list_up.append(2) if num_slices//2**3 >0 else list_up.append(1)
        #list_up = [1,2,2,2,2]
        self.decoder3 = Decoder_3D(in_channel=self.decoderInChannel[2], out_channel=[32, 16, 8, 3, out_ch], list_num_conv =[3,2,2,2], out_sigmoid= out_sigmoid, upsample_depth_list = list_up)
        #list_up = [1,2,2,2,2]
        self.decoder4 = Decoder_3D(in_channel=self.decoderInChannel[3], out_channel=[128, 64, 32, 8, 3, out_ch], list_num_conv =[3,3,2,2,2], out_sigmoid= out_sigmoid, upsample_depth_list = list_up)
        
        self.last_conv = nn.Conv3d(out_ch*4, out_ch, kernel_size=1, stride=1)
        
        if self.out_sigmoid:
            self.sigmoid = nn.Sigmoid()
    
    def forward(self,inp):
        #inp should be BNHWD, the model works with input shape BNDHW
        inp = inp.permute(0,1,4,2,3)
        enc1, enc2, enc3, enc4 = self.encoder(inp)
        #Decoder1
        out1= self.decoder1(enc1)
        #Decoder2
        out2= self.decoder2(enc2)
        #Decoder3
        out3= self.decoder3(enc3)
        #Decoder4
        out4= self.decoder4(enc4)
        #concatenate saliency map
        x = torch.cat((out4, out3, out2, out1), 1)
        x = self.last_conv(x)
        if self.out_sigmoid:
            x = self.sigmoid(x)
        
        if self.training:
            if self.multi_output:
                output = {'map1':out1.permute(0,1,3,4,2), 'map2':out2.permute(0,1,3,4,2), 'map3':out3.permute(0,1,3,4,2), 'map4':out4.permute(0,1,3,4,2), 'final':x.permute(0,1,3,4,2)}
            else:
                output = {'final': x.permute(0,1,3,4,2)}
        
            return output
        
        return x.permute(0,1,3,4,2)

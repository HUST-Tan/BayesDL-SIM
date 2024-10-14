# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:26:45 2021

@author: ChefLT
"""
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import numpy as np

class Patch_Discriminator(nn.Module):
    
    def __init__(self, input_c=1, basic_ndf=64, n_layers=3, kernel_size=4, 
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        
        super(Patch_Discriminator, self).__init__()
        
        # use_bias
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        # padding size
        padding = int(np.ceil((kernel_size - 1) / 2))
        
        # head
        body = [nn.Conv2d(in_channels=input_c, out_channels=basic_ndf, kernel_size=kernel_size, 
                          stride=2, padding=padding),
                     nn.LeakyReLU(0.2, True)]
        
        # (Conv-BN-LeakyReLu) * (n_layer-1)
        ndf_expand = 1    # channel增大倍数
        for n in range(1, n_layers):
            ndf_expand_last = ndf_expand
            ndf_expand = min(8, 2**n)
            body += [nn.Conv2d(ndf_expand_last*basic_ndf, ndf_expand*basic_ndf, kernel_size, 
                                    stride=2, padding=padding, bias=use_bias), 
                          norm_layer(ndf_expand*basic_ndf),
                          nn.LeakyReLU(0.2, True)
                          ] 
        
        # (conv-BN-LeakyReLu)
        ndf_expand_last = ndf_expand
        ndf_expand = min(8, 2**n_layers)
        body += [nn.Conv2d(ndf_expand_last*basic_ndf, ndf_expand*basic_ndf, kernel_size,
                                stride=1, padding=padding, bias=use_bias),
                      norm_layer(ndf_expand*basic_ndf),
                          nn.LeakyReLU(0.2, True)                      
                      ]
        body += [nn.Conv2d(basic_ndf * ndf_expand, 1, kernel_size=kernel_size, stride=1, padding=padding)]
        
        if use_sigmoid:
            body += [nn.Sigmoid()]
            
        self.body = nn.Sequential(*body)
        
    
    def forward(self, x):
        return self.body(x)




# =====================================================================
#                            UNet Discriminator
# =====================================================================
### U-Net Discriminator ###
# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, wide=True,
                preactivation=True, activation=nn.LeakyReLU(0.1, inplace=False), downsample=nn.AvgPool2d(2, stride=2)):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
            
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)

        self.bn1 = self.which_bn(self.hidden_channels)
        self.bn2 = self.which_bn(out_channels)

    # def shortcut(self, x):
    #     if self.preactivation:
    #         if self.learnable_sc:
    #             x = self.conv_sc(x)
    #         if self.downsample:
    #             x = self.downsample(x)
    #     else:
    #         if self.downsample:
    #             x = self.downsample(x)
    #         if self.learnable_sc:
    #             x = self.conv_sc(x)
    #     return x
        
    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it 
            #              will negatively affect the shortcut connection.
            h = self.activation(x)
        else:
            h = x    
        h = self.bn1(self.conv1(h))
        # h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)     
            
        return h #+ self.shortcut(x)
    


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, activation=nn.LeakyReLU(0.1, inplace=False), 
                upsample=nn.Upsample(scale_factor=2, mode='nearest')):
        super(GBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(out_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            # x = self.upsample(x)
        h = self.bn1(self.conv1(h))
        # h = self.activation(self.bn2(h))
        # h = self.conv2(h)
        # if self.learnable_sc:       
        #     x = self.conv_sc(x)
        return h #+ x


class UnetD(torch.nn.Module):
    def __init__(self, in_c=1, base_channels=64):
        super(UnetD, self).__init__()
        
        self.in_c = in_c
        self.base_channels = base_channels

        self.enc_b1 = DBlock(1, 64, preactivation=False)
        self.enc_b2 = DBlock(64, 128)
        self.enc_b3 = DBlock(128, 192)
        self.enc_b4 = DBlock(192, 256)
#        self.enc_b5 = DBlock(256, 320)
#        self.enc_b6 = DBlock(320, 384)

        self.enc_out = nn.Conv2d(256, 1, kernel_size=1, padding=0)

        self.dec_b1 = GBlock(256, 192)
        self.dec_b2 = GBlock(192*2, 128)
        self.dec_b3 = GBlock(128*2, 64)
        self.dec_b4 = GBlock(64*2, 32)


        self.dec_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # print(classname)
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):                            # [b,1,h,w]
        e1 = self.enc_b1(x)                          # conv-BN-AvgPool       ---> [b,64,h/2,w/2]
        e2 = self.enc_b2(e1)                         # LReLu-conv-BN-AvgPool ---> [b,128,h/4,w/4]
        e3 = self.enc_b3(e2)                         # LReLu-conv-BN-AvgPool ---> [b,192,h/8,w/8]
        e4 = self.enc_b4(e3)                         # LReLu-conv-BN-AvgPool ---> [b,256,h/16,w/16]
#        e5 = self.enc_b5(e4)
#        e6 = self.enc_b6(e5)

        e_out = self.enc_out(F.leaky_relu(e4, 0.1))  # LReLu-conv ---> [b,1,h/16,w/16]
        # print(e1.size())
        # print(e2.size())
        # print(e3.size())
        # print(e4.size())
        # print(e5.size())
        # print(e6.size())

        d1 = self.dec_b1(e4)                         # LReLu-Upsample-conv-BN ---> [b,192,h/8,w/8]
        d2 = self.dec_b2(torch.cat([d1, e3], 1))     # LReLu-Upsample-conv-BN ---> [b,128,h/4,w/4]
        d3 = self.dec_b3(torch.cat([d2, e2], 1))     # LReLu-Upsample-conv-BN ---> [b,64,h/2,w/2]
        d4 = self.dec_b4(torch.cat([d3, e1], 1))     # LReLu-Upsample-conv-BN ---> [b,32,h,w]
#        d5 = self.dec_b5(torch.cat([d4, e1], 1))
#        d6 = self.dec_b6(torch.cat([d5, e1], 1))

        d_out = self.dec_out(F.leaky_relu(d4, 0.1))  # LReLu-conv ---> [b,1,h,w]

        return e_out, d_out, [e1,e2,e3,e4], [d1,d2,d3,d4]

if __name__ == '__main__':
    D = UnetD(in_c=1, base_channels=64)
    
    # number of params
    n_p = 0
    for p in D.parameters():
        n_p += p.numel()
    print('# Params: %s'%n_p)
    
    # output shape of enc and dec
    a = torch.ones((4,1,128,128))
    e_out, d_out, _, _ = D(a)
    print('Shape of Enc Output: ',e_out.shape)
    print('Shape of Dec Output: ',d_out.shape)
    
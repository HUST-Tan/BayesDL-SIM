# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:02:58 2022

@author: Administrator
"""

from .common import conv, FFT_layer1, Upsampler,SiLU
from .RCAN import ResidualGroup
import torch
import torch.nn as nn


# ==============================
# Complex Group
# ==============================
class ComplexGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, n_blocks, short_skip=True, long_skip=True):
        super(ComplexGroup, self).__init__()
        self.long_skip = long_skip
        
        body = []
        for _ in range(n_blocks):
            body.append(FFT_layer1(n_feat, kernel_size, skip=short_skip))
        body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*body)
    
    def forward(self, x):
        res = self.body(x)
        if self.long_skip:
            res += x
        return res
    


# ==============================
# Pure complex NN
# ==============================
class FreqComplexNN(nn.Module):
    def __init__(self, 
                 in_channels=9,
                 n_feat=64,
                 out_channels=1, 
                 kernel_size=3,
                 n_blocks=5,
                 n_groups=5,
                 sr_scale =2
                 ):
        super(FreqComplexNN, self).__init__()
        
        # (1) head --> one single conv
        self.head = conv(in_channels, n_feat, kernel_size)
        
        # (2) body --> cascade ComplexGroup + conv
        body = [ComplexGroup(n_feat, kernel_size, n_blocks, short_skip=True, long_skip=True) for _ in range(n_groups)]
        body.append(conv(n_feat, n_feat, kernel_size))
        #body = [FFT_layer1(n_feat, kernel_size=3, skip=True) for _ in range(10)]
        self.body = nn.Sequential(*body)
        
        # (3) tail --> Upsampler + conv
        tail = [Upsampler(sr_scale, n_feat, act=False), conv(n_feat, out_channels, kernel_size)]
        self.tail = nn.Sequential(*tail)
    
    def forward(self, x):
        y = self.head(x)
        res = self.body(y) + y
        res = self.tail(res)
        return res


# ==============================
# local FFT branch
# ==============================
class FreqComplexNN_v1(nn.Module):
    def __init__(self, 
                 in_channels=9, 
                 out_channels=1, 
                 n_feat=64, 
                 kernel_size=3,
                 n_blocks=4,
                 n_groups=4,
                 sr_scale=2,
                 CA_type='CA',
                 act = 'relu'
                 ):
        super(FreqComplexNN_v1, self).__init__()
        if act.lower() == 'gelu':
            act = nn.GELU()
        elif act.lower() == 'relu':
            act = nn.ReLU(True)
        elif act.lower() == 'silu':
            act = SiLU()
        elif act.lower() == 'lrelu':
            act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
#        act = nn.GELU() if act.lower() == 'gelu' else nn.ReLU(True)
        
        # (1) head
        self.head = conv(in_channels, n_feat, kernel_size=3)
        
        # (2) body
        body = [ResidualGroup(n_feat, kernel_size, n_resblocks=n_blocks, fft_branch='FFT_Layer1', CA_type=CA_type,
                                  reduction=16, act=act, res_scale=1) for _ in range(n_groups-1)]
        body.append(ResidualGroup(n_feat, kernel_size, n_resblocks=n_blocks, fft_branch='FFT_Layer1', CA_type=CA_type,
                                  reduction=16, act=act, res_scale=1))
        
        body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*body)
        
        
        # (3) tail
        tail = [Upsampler(sr_scale, n_feat, act=False), conv(n_feat, out_channels, kernel_size)]
        self.tail = nn.Sequential(*tail)
    
    def forward(self, x):
        y = self.head(x)
        res = self.body(y)
        res += y
        res = self.tail(res)
        return res


# ==============================
# global FFT branch
# ==============================
class FreqComplexNN_v2(nn.Module):
    def __init__(self, 
                 in_channels=9, 
                 out_channels=1, 
                 n_feat=64, 
                 kernel_size=3,
                 n_blocks=4,
                 n_groups=4,
                 n_fft_blocks=5,
                 sr_scale=2,
                 CA_type='CA',
                 act='relu'
                 ):
        super(FreqComplexNN_v2, self).__init__()
        act = nn.GELU() if act.lower() == 'gelu' else nn.ReLU(True)
        
        # (1) head
        self.head = conv(in_channels, n_feat, kernel_size=3)
        
        # (2) body
        body = [ResidualGroup(n_feat, kernel_size, n_resblocks=n_blocks, fft_branch='', CA_type=CA_type,
                                  reduction=16, act=act, res_scale=1) for _ in range(n_groups)]
        body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*body)
        
        body_fft = [FFT_layer1(n_feat, kernel_size=3, skip=True, in_spatial=True, out_spatial=False)]
        body_fft += [FFT_layer1(n_feat, kernel_size=3, skip=True, in_spatial=False, out_spatial=False) for _ in range(n_fft_blocks-2)]
        body_fft += [FFT_layer1(n_feat, kernel_size=3, skip=True, in_spatial=False, out_spatial=True)]
        self.body_fft = nn.Sequential(*body_fft)
        
        # (3) tail
        tail = [Upsampler(sr_scale, n_feat, act=False), conv(n_feat, out_channels, kernel_size)]
        self.tail = nn.Sequential(*tail)
    
    def forward(self, x):
        y = self.head(x)
        res = self.body(y) + self.body_fft(y) + y
        res = self.tail(res)
        return res

# ==================================
# Global Group // FFT branch
# ==================================
class FreqComplexNN_v3(nn.Module):
    def __init__(self, 
                 in_channels=9,
                 out_channels=1,
                 n_feat=64,
                 kernel_size=3,
                 n_blocks=4,
#                 n_groups=4,
                 sr_scale=2,
                 CA_type='CA',
                 act='ReLU'
                 ):
        super(FreqComplexNN_v3, self).__init__()
        act = nn.GELU() if act.lower() == 'gelu' else nn.ReLU(True)
        
        # (1) head
        self.head = conv(in_channels, n_feat, kernel_size=3)
        
        # (2) RG1
        self.RG1 = ResidualGroup(n_feat, kernel_size, n_resblocks=n_blocks, fft_branch='', CA_type=CA_type,
                                  reduction=16, act=act, res_scale=1)
        self.fft1 = nn.Sequential(*[FFT_layer1(n_feat, kernel_size=3, skip=True) for _ in range(2)])
        
        # (3) RG2
        self.RG2 = ResidualGroup(n_feat, kernel_size, n_resblocks=n_blocks, fft_branch='', CA_type=CA_type,
                                  reduction=16, act=act, res_scale=1)
        self.fft2 = nn.Sequential(*[FFT_layer1(n_feat, kernel_size=3, skip=True) for _ in range(2)])
        
        # (4) RG3
        self.RG3 = ResidualGroup(n_feat, kernel_size, n_resblocks=n_blocks, fft_branch='', CA_type=CA_type,
                                  reduction=16, act=act, res_scale=1)
        self.fft3 = nn.Sequential(*[FFT_layer1(n_feat, kernel_size=3, skip=True) for _ in range(2)])
        
        # (5) RG4
        self.RG4 = ResidualGroup(n_feat, kernel_size, n_resblocks=n_blocks, fft_branch='', CA_type=CA_type,
                                  reduction=16, act=act, res_scale=1)
        self.fft4 = nn.Sequential(*[FFT_layer1(n_feat, kernel_size=3, skip=True) for _ in range(2)])
        
        # (6) RG5
        self.RG5 = ResidualGroup(n_feat, kernel_size, n_resblocks=n_blocks, fft_branch='', CA_type=CA_type,
                                  reduction=16, act=act, res_scale=1)
        self.fft5 = nn.Sequential(*[FFT_layer1(n_feat, kernel_size=3, skip=True) for _ in range(2)])
        
        # (7) CONV
        self.conv = conv(n_feat, n_feat, kernel_size)
        
        # (8) Tail
        tail = [Upsampler(sr_scale, n_feat, act=False), conv(n_feat, out_channels, kernel_size)]
        self.tail = nn.Sequential(*tail)
        
    def forward(self, x):
        x = self.head(x)
        
        y_spa = self.RG1(x)
        y_freq = self.fft1(x)
        y_fusion = y_spa + y_freq
        
        y_spa = self.RG2(y_fusion)
        y_freq = self.fft2(y_fusion)
        y_fusion = y_spa + y_freq
        
        y_spa = self.RG3(y_fusion)
        y_freq = self.fft3(y_fusion)
        y_fusion = y_spa + y_freq
        
        y_spa = self.RG4(y_fusion)
        y_freq = self.fft4(y_fusion)
        y_fusion = y_spa + y_freq
        
        y_spa = self.RG5(y_fusion)
        y_freq = self.fft5(y_fusion)
        y_fusion = y_spa + y_freq
        
        y = self.conv(y_fusion)
        y += x
        
        y = self.tail(y)
        return y
        
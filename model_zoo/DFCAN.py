# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 22:33:14 2021

@author: Administrator
"""

import torch
import torch.nn as nn

def fft2d(x, gamma=0.1):
    '''
    x: torch tensor of shape [B,C,H,W]
    '''
    fft = torch.fft.fft2(torch.complex(x, torch.zeros_like(x)), dim=[-2,-1])
    absfft = torch.pow(torch.abs(fft)+1e-8, gamma)
    return torch.fft.fftshift(absfft,dim=[-2,-1])


class FCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FCALayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                  nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel//reduction, kernel_size=1, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(channel//reduction, channel, kernel_size=1, padding=0),
                                   nn.Sigmoid())
    
    def forward(self, x):
        absfft1 = fft2d(x, gamma=0.8)
        absfft2 = self.conv1(absfft1)
        W = self.avg_pool(absfft2)
        W = self.conv2(W)
        return x*W


class FCAB(nn.Module):   # Conv-Gelu-Conv-Gelu-FCA
    def __init__(self, channel, reduction=16):
        super(FCAB, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(channel, channel, 3, 1, 1))
            m.append(nn.GELU())
        m.append(FCALayer(channel, reduction=reduction))
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        y = self.body(x)
        return x+y


class FCA_RG(nn.Module):
    def __init__(self, channel, reduction=16, n_FCAB=4):
        super(FCA_RG, self).__init__()
        m = []
        for _ in range(n_FCAB):
            m.append(FCAB(channel, reduction))
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        y = self.body(x)
        return x+y

class DFCAN(nn.Module):
    def __init__(self, in_c, channel=64, out_c=1, n_RG=4, n_FCAB=4, reduction=16):
        super(DFCAN, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_c, channel, 3, 1, 1),
                                       nn.GELU())
        m = []
        for _ in range(n_RG):
            m.append(FCA_RG(channel, reduction=reduction, n_FCAB=n_FCAB))
        self.body = nn.Sequential(*m)
        
        self.tail = nn.Sequential(nn.Conv2d(channel, channel*4, 3, 1, 1),
                                  nn.GELU(),
                                  nn.PixelShuffle(2),
                                  nn.Conv2d(channel, out_c, 3, 1, 1),
#                                  nn.Sigmoid()
                                  )
    
    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return y
    
    def my_load_state_dict(self, state_dict, strict=True):
        own_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state_dict:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state_dict[name].copy_(param)
                except:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state_dict[name].size(), param.size()))
                
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
    
    

if __name__ == '__main__':
    net = DFCAN(in_c=9, channel=64, out_c=1, n_RG=10, n_FCAB=10, reduction=16)
    
    n_p = 0
    for p in net.parameters():
        n_p += p.numel()
    print('# Params: ',n_p)
    
    a = torch.ones((4,9,64,64))
    print(net(a).shape)
    
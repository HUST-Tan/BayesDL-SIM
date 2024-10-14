# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 20:08:17 2021

@author: ChefLT
"""
import torch
import torch.nn as nn
from .common import conv, RCAB, Upsampler, ComplexConv2d


# ===============================
#        Residual Group (RG)
# ===============================
class ResidualGroup(nn.Module):   # n_resblocks个RCAB + conv
    def __init__(self, n_feat, kernel_size, n_resblocks, reduction=16, bias=True, bn=False, ln=False, act=nn.ReLU(True), 
                  res_scale=1, CA_type='CA', fft_branch='FFT_Layer', group_skip=True, block_skip=True):
        super(ResidualGroup, self).__init__()
        self.skip = group_skip
        
        modules_body = []
            
        # RCAB x N
        modules_body = [ RCAB(n_feat, kernel_size, reduction=reduction, bias=bias, bn=bn, 
                              ln=ln, act=act, res_scale=res_scale, CA_type=CA_type, 
                              skip=block_skip, fft_branch=fft_branch) 
                        for _ in range(n_resblocks)]
        

        modules_body.append(conv(n_feat, n_feat, kernel_size))
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        if self.skip:
            res += x
        return res

        
# ===========================================
#  Bayesian Deep learning (BayesDL)
# ===========================================
class BayesDL(nn.Module):
    def __init__(self, in_channels,           
                 sr_scale=2,                  
                 out_channels=1,               
                 n_resgroups=10,              
                 n_resblocks=20,              
                 n_feat=64,                  
                 kernel_size=3,               
                 reduction=16,               
                 act = 'relu',           
                 CA_type='CA',                
                 learn_std=False,           
                 drop = 0,                    
                 drop2d = 0,                  
                 fft_branch = '',             
                 ln = False,                  
                 fft_type = 'rfft',           # fft or rfft
                 likelihood = 'gauss',        # likehihood when outputing std
                 ):             
        super(BayesDL, self).__init__()
                
        self.in_channels = in_channels
        self.learn_std = learn_std
        self.fft_type = fft_type
        self.likelihood = likelihood
        
        if act.lower() == 'gelu':
            act = nn.GELU()
        else:
            act = nn.ReLU(True)

        
        # ====================
        # (1) Head module
        # ====================
        modules_head = [conv(in_channels, n_feat, kernel_size)]


        # ====================
        # (2) Body module
        # ====================
        modules_body = []
        for ii in range(n_resgroups):
            modules_body.append(ResidualGroup(n_feat, kernel_size, n_resblocks=n_resblocks, reduction=reduction, 
                                              ln=ln, act=act, CA_type=CA_type, fft_branch=fft_branch))
            if drop > 0:
                modules_body.append(nn.Dropout(drop))
            
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        # ====================
        # (3) Tail module
        # ====================

        modules_tail = [ Upsampler(sr_scale, n_feat, act=False),
                        nn.Dropout2d(drop2d) if drop2d > 0 else nn.Identity()]
        modules_tail += [conv(n_feat, out_channels, kernel_size)]
        
               
        # =====================================
        # (4) Tail module for learning std
        # =====================================
        if self.learn_std:
            
                
            # 空域：高斯/拉普拉斯
            if self.likelihood in ['gauss', 'laplace']:
                uncer_tail = [#Upsampler(sr_scale, n_feat, act=False),
                              conv(n_feat, n_feat, kernel_size), nn.ELU(),
                              conv(n_feat, n_feat, kernel_size), nn.ELU(),
                              conv(n_feat, n_feat, kernel_size), nn.ELU(),
                              conv(n_feat, n_feat, kernel_size), nn.ELU(),
                              conv(n_feat, 1, kernel_size)]
                #    uncer_tail = [conv(n_feat, 1, kernel_size)]
                self.uncer_tail = nn.Sequential(*uncer_tail)
                
            # 空域：student-T
            elif self.likelihood == 'student':
                uncer_tail_precision = [conv(n_feat, n_feat, kernel_size), nn.ELU(),
                                      conv(n_feat, n_feat, kernel_size), nn.ELU(),
                                      conv(n_feat, n_feat, kernel_size), nn.ELU(),
                                      conv(n_feat, n_feat, kernel_size), nn.ELU(),
                                      conv(n_feat, 1, kernel_size)]
                uncer_tail_df = [conv(n_feat, n_feat, kernel_size), nn.ELU(),
                               conv(n_feat, n_feat, kernel_size), nn.ELU(),
                               conv(n_feat, n_feat, kernel_size), nn.ELU(),
                               conv(n_feat, n_feat, kernel_size), nn.ELU(),
                               conv(n_feat, 1, kernel_size)]
                self.uncer_tail_precision = nn.Sequential(*uncer_tail_precision)
                self.uncer_tail_df = nn.Sequential(*uncer_tail_df)
                    
           
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        y = self.head(x)
        res = self.body(y)
        res += y
        y_up = self.tail[0](res)
        
        y = self.tail[1:](y_up)  
        
        if self.learn_std:
            
            if self.likelihood in ['gauss','laplace']:
                log_std = self.uncer_tail(nn.functional.interpolate(res, scale_factor=2, mode='nearest'))   # predicted log-variance
                return [y, log_std]
            elif self.likelihood == 'student':
                log_precision = self.uncer_tail_precision(nn.functional.interpolate(res, scale_factor=2, mode='nearest'))  # log-precision
                log_df = self.uncer_tail_df(nn.functional.interpolate(res, scale_factor=2, mode='nearest'))        # log-df
                return [y, log_precision, log_df]
                
        else:
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
                

    
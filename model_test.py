# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:51:08 2023

@author: Administrator
"""

import torch.nn as nn
import torch
import os
import collections

from model_zoo.BayesDL import BayesDL
from model_zoo.DFCAN import DFCAN
from utils import prctile_norm



class Model_test():
    
    def __init__(self, args, INPUT_C, device=None):
        
        self.args = args
        self.learn_std = args.likelihood != ''
        self.INPUT_C = INPUT_C
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.is_parallel = True if len(args.GPU) > 1 else False
        self.GPU = args.GPU
        self.device = torch.device('cuda:{}'.format(self.GPU[0]))
        
        
        # =======================
        # (1) Define netG
        # =======================
        if args.model == 'BayesDL':
            self.netG = BayesDL(INPUT_C, args.sr_scale, n_resgroups=args.nGroups, n_resblocks=args.nBlocks, 
                             CA_type=args.CA_type, ln=False, learn_std=self.learn_std, 
                             n_feat=args.n_feat, act=args.act, drop2d=args.drop2d, drop=args.drop, 
                             likelihood=self.args.likelihood).cuda(self.GPU[0])
                
        elif args.model == 'DFCAN':
            self.netG = DFCAN(INPUT_C).cuda(self.GPU[0])
            
        
        
        
        # =============================
        # (2) loading pre-trained weights
        # =============================
        if os.path.isfile(self.args.pretrained_G):
            self.load_model(self.args.pretrained_G)
        
        for p in self.netG.parameters():
            p.requires_grad = False
        
        
        self.N_infer = 1


    def load_model(self, path):
        if self.is_parallel:  # load model trained with multi GPU
            state_dict = torch.load(path, map_location='cuda:%d'%int(self.GPU[0]))
            new_state_dict = collections.OrderedDict()
            for item, value in state_dict.items():
                new_item = '.'.join(item.split('.')[1:])
                new_state_dict[new_item] = value
            self.netG.load_state_dict(new_state_dict)
        else:
            self.netG.load_state_dict(torch.load(path, map_location='cuda:%d'%int(self.GPU[0])))
            
        print('****** Load Pretrained Model from %s ******'%path)
            
    
    
    def print_net(self):
        def print_network(net):
            num_params = 0
            for param in net.parameters():
                if param.requires_grad: num_params += param.numel()
            #print(net)
            print('Total number of parameters of %s: %d' % ('model', num_params))
        print_network(self.netG)
        
    
    def chop_inference(self, x, shave=20): # x ~ [B,frames,H,W]
        
        h,w = x.shape[-2:]
        h_half, w_half = h//2, w//2
        h_size = h_half + shave   
        w_size = w_half + shave
        
        top = slice(0, h_size)
        bottom = slice(h - h_size, h)
        left = slice(0, w_size)
        right = slice(w - w_size, w)
        
        x_chops = [ x[...,top,left],
                    x[...,top,right],
                    x[...,bottom,left],
                    x[...,bottom,right] ]
        
        y_chops = []
        for x_chop in x_chops:
            x_chop = prctile_norm(x_chop)
            y_chops.append(self.netG(x_chop.to(self.device)).detach().cpu())
            
        
        h,w = h*2, w*2
        h_half, w_half = h_half*2, w_half*2
        h_size, w_size = h_size*2, w_size*2
        shave *= 2
        
        b,c = y_chops[0].shape[:2]
        sr = y_chops[0].new(b,c,h,w)
        sr[..., 0:h_half, 0:w_half] = y_chops[0][..., 0:h_half, 0:w_half]
        sr[..., 0:h_half, w_half:w] = y_chops[1][..., 0:h_half, (w_size - w + w_half):w_size]
        sr[..., h_half:h, 0:w_half] = y_chops[2][..., (h_size - h + h_half):h_size, 0:w_half]
        sr[..., h_half:h, w_half:w] = y_chops[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        
        return sr
    
    
    
    
    
    
    
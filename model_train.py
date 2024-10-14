# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:22:54 2021

@author: ChefLT
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import matplotlib.pyplot as plt
import math

from model_zoo.BayesDL import BayesDL
from model_zoo.D_net import Patch_Discriminator,UnetD
from model_zoo.DFCAN import DFCAN
import losses
from utils import psnr,ssim,tensor2im,prctile_norm,Freq_metric,fft_torch,apodImRect
from augment import apply_augment
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout, Bayes_ensemble
from scalablebdl.prior_reg import PriorRegularizor
from scalablebdl.mean_field import PsiSGD, to_bayesian, to_deterministic


#from models.networks.grl import GRL

class Model_train():
    
    def __init__(self, args, INPUT_C, device=None):
        
        self.args = args
        self.adv = (args.train_type == 'adv')                         
        self.learn_std = ('uncertainty' in args.train_type) and (args.likelihood != '')
        self.INPUT_C = INPUT_C
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.is_parallel = True if len(args.GPU) > 1 else False
        self.GPU = args.GPU
        self.device = torch.device('cuda:{}'.format(self.GPU[0]))
        

        # =======================
        # Define netG
        # =======================
        if args.model == 'BayesDL':
            self.netG = BayesDL(INPUT_C, args.sr_scale, n_resgroups=args.nGroups, n_resblocks=args.nBlocks, 
                             CA_type=args.CA_type, ln=False, learn_std=self.learn_std, 
                             n_feat=args.n_feat, act=args.act, drop2d=args.drop2d, drop=args.drop, 
                             likelihood=self.args.likelihood).cuda(self.GPU[0])
                
        elif args.model == 'DFCAN':
            self.netG = DFCAN(INPUT_C).cuda(self.GPU[0])
            
        
        # Load pretrain
        if os.path.exists(self.args.pretrained_G):
            if not args.is_train and self.is_parallel:  # load model trained with multi GPU
                state_dict = torch.load(args.pretrained_G,map_location='cuda:%d'%int(self.GPU[0]))
                new_state_dict = collections.OrderedDict()
                for item,value in state_dict.items():
                    new_item = '.'.join(item.split('.')[1:])
                    new_state_dict[new_item] = value
                self.netG.load_state_dict(new_state_dict)
            
            else:
                #self.netG.load_state_dict(torch.load(args.pretrained_G))
                self.netG.my_load_state_dict(torch.load(args.pretrained_G,map_location='cuda:%d'%int(self.GPU[0])))
                
            print('****** Load Pretrained Model from %s ******'%self.args.pretrained_G)

        

        # Multi GPU
        if self.is_parallel:
            self.netG = torch.nn.DataParallel(self.netG, device_ids=self.GPU)
        
        
        # =========================
        # netD
        # =========================
        if self.args.is_train:
            if self.adv:
                if self.args.model_D == 'patchD':
                    self.netD = Patch_Discriminator(use_sigmoid=True).cuda(self.GPU[0])
                elif self.args.model_D == 'unetD':
                    self.netD = UnetD(in_c=1, base_channels=64).cuda(self.GPU[0])
            
                if os.path.exists(self.args.pretrained_D):
                    checkpoint = torch.load(args.pretrained_D)
                    self.netD.load_state_dict(checkpoint)
                    print('****** Load Pretrained Model from %s ******'%self.args.pretrained_D)
                else:
                    self.netD.apply(self.weights_init)
                
                # multi GPU
                if self.is_parallel:
                    self.netD = torch.nn.DataParallel(self.netD, device_ids=self.GPU)
        
        
        # =========================
        # Loss fn
        # =========================
        if self.args.is_train:
            self.mse_loss_fn = nn.MSELoss()
            self.l1_loss_fn = nn.L1Loss()
            self.ssim_loss_fn = losses.SSIM()
            self.fft_loss_fn = losses.FFT_loss(loss_type='L1', alpha=0, mask='', device=torch.device('cuda:%d'%self.GPU[0]))
         #   self.fft_loss_fn = losses.FFT_loss(loss_type='L1', alpha=0, device=torch.device('cuda:%d'%self.GPU[0]))
         #   self.ffl_loss_fn = FocalFrequencyLoss(alpha=0)
            self.discLoss = losses.DiscLoss(use_l1=False, tensor=self.Tensor, device=torch.device('cuda:%d'%self.GPU[0])) # get_g_loss and get_d_loss
            self.discLoss_unet = losses.DiscLoss_unetD(use_l1=False, tensor=self.Tensor)
        
        
        # =========================
        # optimizer and scheduler
        # =========================
        if self.args.train_type == 'uncertainty_tail':
            for name, p in self.netG.named_parameters():
                p.requires_grad = True if 'uncer_tail' in name else False
                
        if self.args.is_train:
            self.optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), 
                                          lr=self.args.lr, weight_decay=args.weight_decay)
            #self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=400, gamma = 0.5)
            self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=self.args.epoch,
                                                                    eta_min=1e-6)
           # if self.is_parallel:
           #     self.optimizer_G = torch.nn.DataParallel(self.optimizer_G, device_ids=self.GPU)
            
            if self.adv:
                self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.args.lr)
               # self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=args.lr_decay_epoch, 
               #                                      gamma = args.lr_decay_rate)
                self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=self.args.epoch,
                                                                       eta_min=1e-6)
                #if self.is_parallel:
                #    self.optimizer_D = torch.nn.DataParallel(self.optimizer_D, device_ids=self.GPU)
        
        self.UpdateD = 5
        
        
        # ================
        # MFVI
        # ================
        if self.args.mfvi:
            #self.netG.head = to_bayesian(self.netG.head)
            self.netG.body = to_bayesian(self.netG.body)
            unfreeze(self.netG)
            print('Convert to Bayesian NN ......')
            
            if self.args.is_train:
                mus, psis = [], []
                for name, param in self.netG.named_parameters():
                    if 'psi' in name: 
                        psis.append(param)
                    else: 
                        mus.append(param)
                        
                #self.optimizer_G = optim.Adam(psis, lr=self.args.lr)
                self.optimizer_G = optim.Adam([{"params": psis, "lr": 0.00001, "weight_decay": 0},
                                               {"params": mus, "lr": self.args.lr, "weight_decay": 5e-4}])
                self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=self.args.epoch,
                                                                        eta_min=1e-6)
                
                self.regularizer = PriorRegularizor(self.netG.body, 
                                                    decay=1/552, num_data=552, num_mc_samples=20)
        
        
        
        
        #print('================================')
        #self.print_net()
        #print('================================')
        
    
    def set_input(self, data, is_eval=False):
        x = data[0]
        y = data[1]
        if (not is_eval) and len(self.args.augs) > 0:   # cutmix / mixup / cutmixup
            y, x, mask, aug = apply_augment(y, x, augs=self.args.augs)

        self.x = x.cuda(self.GPU[0])
        self.y = y.cuda(self.GPU[0])
        
        
    
    def forward(self):
        if self.learn_std:          
            if self.args.likelihood in ['gauss', 'laplace']:
                self.y_,self.var = self.netG(self.x)
            elif self.args.likelihood == 'student':
                self.y_, self.log_precision, self.log_df = self.netG(self.x)
            
        else:
            self.y_ = self.netG(self.x)
    
    
    def forward_slice_test(self, x, minpatch):
        B,C,H,W = x.shape
        y = torch.zeros((B, 1, 2*H , 2*W), dtype=x.dtype, device=x.device)
        uncertainty = torch.zeros((B, 1, 2*H, 2*W), dtype=x.dtype, device=x.device)
        
        for h in list(range(0,x.shape[-1],minpatch)):
            for w in list(range(0,x.shape[-1],minpatch)):
        
                uppad = 32 if h > 0 else 0
                downpad = 32 if h < x.shape[-1] else 0
                leftpad = 32 if w > 0 else 0
                rightpad = 32 if w < x.shape[-1] else 0
                
                y[..., 2 * h:2 * h + minpatch * 2, 2 * w:2 * w + minpatch * 2] = self.netG.forward(
                    x[..., h - uppad:h + downpad + minpatch, w - leftpad:w + rightpad + minpatch],
                )[0][..., 2 * uppad:2 * uppad + minpatch * 2, 2 * leftpad:2 * leftpad + minpatch * 2]
                
                uncertainty[..., 2 * h:2 * h + minpatch * 2, 2 * w:2 * w + minpatch * 2] = self.netG.forward(
                    x[..., h - uppad:h + downpad + minpatch, w - leftpad:w + rightpad + minpatch],
                )[1][..., 2 * uppad:2 * uppad + minpatch * 2, 2 * leftpad:2 * leftpad + minpatch * 2]
        return y, uncertainty
        
        
        
    
    def backward_G(self):
        if not self.learn_std:
                        
            # Pixel loss
            self.mae_loss = self.args.mae_lambda * self.l1_loss_fn(self.y, self.y_)
            self.mse_loss = self.args.mse_lambda * self.mse_loss_fn(self.y, self.y_)
            
            # FFT loss
            self.fft_loss = self.args.fft_lambda * self.fft_loss_fn(apodImRect(self.y,20,device=self.device), apodImRect(self.y_,20,device=self.device))
                            
            # ssim loss
            self.ssim_loss = self.args.ssim_lambda * (1-self.ssim_loss_fn(self.y, self.y_))
            
            # adv loss
            if self.adv:
                if self.args.model_D == 'patchD':
                    self.adv_loss = self.args.adv_lambda * self.discLoss.get_g_loss(self.netD, self.y_)
                elif self.args.model_D == 'unetD':
                    self.adv_loss = self.args.adv_lambda * self.discLoss_unet.get_g_loss(self.netD, self.y_, self.y)
            else:
                self.adv_loss = torch.tensor([0]).cuda(self.GPU[0])
            
            # total g loss
            self.g_loss = self.mae_loss + self.mse_loss + self.ssim_loss + self.adv_loss + self.fft_loss
            
     
        
        elif self.learn_std:
                           
            if self.args.likelihood == 'laplace':
                s_ = torch.exp(-self.var)
                self.resi_loss = self.l1_loss_fn(torch.mul(self.y_, s_), torch.mul(self.y, s_))
                self.uncer_reg = torch.mean(self.var)                   
                self.g_loss = self.resi_loss + self.uncer_reg
                
            elif self.args.likelihood == 'gauss':
                s_ = torch.exp(-self.var)
                self.resi_loss = torch.mean(torch.mul((self.y - self.y_)**2, s_))
                self.uncer_reg = torch.mean(self.var)
                self.g_loss = self.resi_loss + self.uncer_reg
                
            elif self.args.likelihood == 'student':
                precision = torch.exp(self.log_precision)
                df = torch.exp(self.log_df)
                error = (self.y - self.y_)**2
                Z = -0.5*self.log_precision + 0.5*self.log_df + 0.5*math.log(math.pi) + torch.lgamma(0.5 * df) - torch.lgamma(0.5 * (df + 1.))
                self.g_loss = torch.mean(0.5 * (df + 1.) * torch.log1p(error * precision / df) + Z)
                               
        
        self.g_loss.backward()
        
    def backward_G_pretrain(self):
        ''' impose Pixel loss and SSIM loss '''
        self.mae_loss = self.args.mae_lambda * self.l1_loss_fn(self.y, self.y_)
        self.ssim_loss = self.args.ssim_lambda * (1-self.ssim_loss_fn(self.y, self.y_))
        self.mse_loss = 0
        self.fft_loss = self.args.fft_lambda * self.fft_loss_fn(self.y, self.y_)

        self.g_loss = self.mae_loss + self.ssim_loss + self.fft_loss
        self.g_loss.backward()
        
        self.adv_loss = torch.tensor([0]).cuda(self.GPU[0])
        self.mse_loss = torch.tensor([0]).cuda(self.GPU[0])
#        self.fft_loss = torch.tensor([0]).cuda(self.GPU[0])
        self.d_loss = torch.tensor([0]).cuda(self.GPU[0])
        
    
    def backward_D(self):      
        if self.args.model_D == 'unetD':
            self.d_loss = self.discLoss_unet.get_d_loss(self.netD, self.y_, self.y)
            self.d_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), 0.1)
        elif self.args.model_D == 'patchD':
            self.d_loss = self.discLoss.get_d_loss(self.netD, self.y_, self.y)
            self.d_loss.backward(retain_graph=True)  
    
    
    def optimize_paras(self, epoch):
        
        # pretrained
        if self.adv and (epoch < 500):
        #    if self.is_parallel:
        #        self.optimizer_G.module.zero_grad()
        #    else:
        #        self.optimizer_G.zero_grad()
            self.optimizer_G.zero_grad()
            self.backward_G_pretrain()
            self.optimizer_G.step()
        #    if self.is_parallel:
        #        self.optimizer_G.module.step()
        #    else:
        #        self.optimizer_G.step()
        
        # UNetD
        else:
            if self.adv:
                for i in range(self.UpdateD):
                    #if self.is_parallel:
                    #    self.optimizer_D.module.zero_grad()
                    #else:
                    #    self.optimizer_D.zero_grad()
                    self.optimizer_D.zero_grad()
                    self.backward_D()
                    self.optimizer_D.step()
                    #if self.is_parallel:
                    #    self.optimizer_D.module.step()
                    #else:
                    #    self.optimizer_D.step()
            
            #self.optimizer_G.zero_grad() if not self.is_parallel else self.optimizer_G.module.zero_grad()
            self.optimizer_G.zero_grad()
            self.backward_G()
            if self.args.mfvi:
                self.regularizer.step()
            self.optimizer_G.step()
            #if self.is_parallel:
            #    self.optimizer_G.module.step()
            #else:
            #    self.optimizer_G.step()
            if self.args.sgld and (epoch-1) % 50 >= 30:
                for n in [x for x in self.netG.parameters() if len(x.size()) == 4]:
                    noise = torch.randn(n.size()) * self.args.param_noise_sigma *self.optimizer_G.param_groups[0]['lr']
                    noise = noise.float().cuda(self.GPU[0])
                    n.data = n.data + noise
                    
        
    
    def update_lr(self, epoch):
        #old_lr_G = self.optimizer_G.param_groups[0]['lr'] if not self.is_parallel else self.optimizer_G.module.param_groups[0]['lr']
        #self.scheduler_G.step(epoch)
        #new_lr_G = self.optimizer_G.param_groups[0]['lr'] if not self.is_parallel else self.optimizer_G.module.param_groups[0]['lr']
        old_lr_G = self.optimizer_G.param_groups[0]['lr']
        self.scheduler_G.step(epoch)
        new_lr_G = self.optimizer_G.param_groups[0]['lr']
        print('****** Update lr of Generator  %.6f ---> %.6f ********' %(old_lr_G, new_lr_G))
        
        if self.adv:
            #old_lr_D = self.optimizer_D.param_groups[0]['lr'] if not self.is_parallel else self.optimizer_D.module.param_groups[0]['lr']
            #self.scheduler_D.step(epoch)
            #new_lr_D = self.optimizer_D.param_groups[0]['lr'] if not self.is_parallel else self.optimizer_D.module.param_groups[0]['lr']
            old_lr_D = self.optimizer_D.param_groups[0]['lr']
            self.scheduler_D.step(epoch)
            new_lr_D = self.optimizer_D.param_groups[0]['lr']
            print('****** Update lr of Discrimitor  %.6f ---> %.6f ********' %(old_lr_D, new_lr_D))
    
    
    def model_eval_fn(self, val_data_loader):
       # self.netG.eval()
        
        val_psnr = []
        val_ssim = []
        
        
        with torch.no_grad():
            for j,val_data in enumerate(val_data_loader):
                self.set_input(val_data, is_eval=True)
                self.forward()
                
                val_psnr.append(psnr(tensor2im(self.y, True), tensor2im(self.y_, True)))
                val_ssim.append(ssim(tensor2im(self.y, True), tensor2im(self.y_, True)))
               
        
        val_psnr = np.mean(val_psnr)
        val_ssim = np.mean(val_ssim)
                        
   #     self.netG.train()
        return val_psnr,val_ssim
    
    
    def model_eval(self, val_data_loader):
        self.netG.eval()
        
        if (self.args.drop+self.args.drop2d) > 0 or self.args.mfvi:   
            # open dropout
            for m in self.netG.modules():
                if m.__class__.__name__.find('Dropout') >= 0:
                    m.train()      
            num_mc = 5
        else:
            num_mc = 1
                  
        val_psnrs,val_ssims = [], []
        with torch.no_grad():
            for j, val_data in enumerate(val_data_loader):
                self.set_input(val_data, is_eval=True)
                self.y_ = 0
                for mc in range(num_mc):
                    if not self.learn_std:
                        self.y_ += self.netG(self.x)
                    else:
                        self.y_ += self.netG(self.x)[0]
                        
                self.y_ /= num_mc
                val_psnrs.append(psnr(tensor2im(self.y, True), tensor2im(self.y_, True)))
                #val_ssims.append(ssim(tensor2im(self.y, True), tensor2im(self.y_, True)))
        
        self.val_psnr = np.mean(val_psnrs)
        self.val_ssim = 0#np.mean(val_ssims)
                    
        self.netG.train()
            
        return self.val_psnr, self.val_ssim
            
                
    
    
    def print_net(self):
        def print_network(net):
            num_params = 0
            for param in net.parameters():
                if param.requires_grad: num_params += param.numel()
            #print(net)
            print('Total number of parameters of %s: %d' % ('model', num_params))
        print_network(self.netG)
        if self.adv:
            print_network(self.netD)
        
        
    def save_model(self, save_path, model_name):
        save_G_path = os.path.join(save_path, str(model_name)+'_G.pth')
        torch.save(self.netG.state_dict(), save_G_path)
        
        if self.adv:
            save_D_path = os.path.join(save_path, str(model_name)+'_D.pth')
            torch.save(self.netD.state_dict(), save_D_path)
    
    
    def weights_init(self,m):
        classname = m.__class__.__name__
        
        # 卷积层初始化，kernel ~ N(0,0.02)，bias置0
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        
        # BN层初始化
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    
    def logger(self, writer, iteration):
        if not self.learn_uncer:
            writer.add_scalar(tag='G_mae_loss', scalar_value = self.mae_loss, global_step=iteration)
            writer.add_scalar(tag='G_mse_loss', scalar_value = self.mse_loss, global_step=iteration)
            writer.add_scalar(tag='G_ssim_loss', scalar_value = self.ssim_loss, global_step=iteration)
            writer.add_scalar(tag='G_fft_loss', scalar_value = self.fft_loss, global_step=iteration)
            if self.adv:
                writer.add_scalar(tag='G_adv_loss', scalar_value = self.adv_loss, global_step=iteration)
                writer.add_scalar(tag='D_adv_loss', scalar_value = self.d_loss, global_step=iteration)
        
        # Learning rate
        #writer.add_scalar(tag='Learning_rate_G', scalar_value=self.optimizer_G.param_groups[0]['lr'] if not self.is_parallel else self.optimizer_G.module.param_groups[0]['lr'], global_step=iteration)
        writer.add_scalar(tag='Learning_rate_G', scalar_value=self.optimizer_G.param_groups[0]['lr'] , global_step=iteration)
        if self.adv:
            #writer.add_scalar(tag='Learning_rate_D', scalar_value=self.optimizer_G.param_groups[0]['lr'] if not self.is_parallel else self.optimizer_G.module.param_groups[0]['lr'], global_step=iteration)
            writer.add_scalar(tag='Learning_rate_D', scalar_value=self.optimizer_G.param_groups[0]['lr'] , global_step=iteration)
        
        # validation metrics
        writer.add_scalar(tag='Validation PSNR', scalar_value = self.val_psnr, global_step=iteration)
        writer.add_scalar(tag='Validation SSIM', scalar_value = self.val_ssim, global_step=iteration)
        #writer.add_scalar(tag='Validation LFM', scalar_value = self.val_LFM, global_step=iteration)
        #writer.add_scalar(tag='Validation HFM', scalar_value = self.val_HFM, global_step=iteration)
        
        # Image
        writer.add_image(tag='Input Image', img_tensor=tensor2im(self.x, True), global_step=iteration, dataformats='HW')
        writer.add_image(tag='Output Image', img_tensor=tensor2im(self.y_, True), global_step=iteration, dataformats='HW')
        writer.add_image(tag='Ground Truth', img_tensor=tensor2im(self.y, True), global_step=iteration, dataformats='HW')
        
        # uncertainty map
        if self.learn_uncer:
            figure = plt.figure()
            plt.imshow(torch.exp(self.var[0,0,:,:]).cpu().detach(),'jet')
            writer.add_figure(tag='Aleatoric', figure=figure, global_step=iteration)
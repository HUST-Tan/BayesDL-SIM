# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:58:18 2021

@author: ChefLT
"""
import os
import imageio
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from utils import prctile_norm
from natsort import natsorted
import random
import torch.nn.functional as F
from PIL import Image
import tifffile as tf


# ============================================
# SIM Raw data for training (Online version)
# ============================================
class BioSR_SIM_train_online(Dataset):
    def __init__(self, root_dir='D:/code/fluorescence/dataset/BioSR/train_online/F-actin',
                 wf_upscale=0,      
                 sr_scale = 2,      
                 patch_size = 128,  
                 input_frames = [i for i in range(1,10)],
                 cell_index = [i for i in range(6,36)],    
                 snr_index = [i for i in range(1,13)],     
                 
                 ):   
        self.is_cos7_lifeact = 'Lifeact' in root_dir
        self.is_beads = 'beads' in root_dir
        self.is_er = 'ER' in root_dir
        self.is_mt = 'MT' in root_dir
        if self.is_mt:
            gt_file_name = 'gt'
        else:
            gt_file_name = 'gt'
        
        
        self.root_dir = root_dir
        self.wf_upscale = wf_upscale
        self.sr_scale = sr_scale
        self.patch_size = patch_size
        #self.input_frames = natsorted(input_frames)
        self.input_frames = natsorted([i-1 for i in input_frames])
        self.cell_index = natsorted(cell_index)
        self.snr_index = natsorted(snr_index)
        
        
        self.input_path = os.path.join(self.root_dir,'Raw')
        self.gt_path = os.path.join(self.root_dir,'GT')
        
        
        cells = ['cell_'+str(i) for i in self.cell_index]
        if self.is_beads or self.is_cos7_lifeact:
            levels = ['seq_'+str(i) for i in self.snr_index]
        else:
            levels = ['level_'+str(i) for i in self.snr_index]
        
        input_folders = []
        gt_files = []
        for cell in cells:
            cell_input_path = os.path.join(self.input_path, cell)
            cell_gt_file = os.path.join(self.gt_path, cell, '{}.tif'.format(gt_file_name))           
            
            for level in levels:
                cell_level_input_path = os.path.join(cell_input_path, level)
                input_folders.append(cell_level_input_path)
                if self.is_er:
                    cell_gt_file = os.path.join(self.gt_path, cell, level, '{}.tif'.format(gt_file_name))
                gt_files.append(cell_gt_file)
                
                
        self.input_folders = input_folders
        self.gt_files = gt_files
        assert len(self.input_folders) == len(self.gt_files) == len(self.cell_index)*len(self.snr_index)
        
        
            
    
    def __getitem__(self, index):
        input_folder = self.input_folders[index]
        input_files = natsorted(os.listdir(input_folder))
        
        
        # 1. read raw and gt
        input_imgs = []
        for frame in [0,1,2,3,4,5,6,7,8]:
            input_file = input_files[frame]
            input_im = imageio.imread(os.path.join(input_folder, input_file))   # [502,502]
            
            if self.wf_upscale:
                input_im = cv2.resize(input_im, (input_im.shape[0]*self.sr_scale, input_im.shape[1]*self.sr_scale),
                                  interpolation = cv2.INTER_CUBIC) # [1004,1004] 
            input_imgs.append(input_im)
        input_imgs = np.array(input_imgs)   # [9,502,502]
        
        if len(self.input_frames) < 9:
            wf = np.mean(input_imgs, axis=0, keepdims=True)
            input_imgs = input_imgs[self.input_frames, :, :]
            input_imgs = np.concatenate([wf,input_imgs], axis=0)
        
        
        
        gt_file = self.gt_files[index]
        gt_im = imageio.imread(gt_file)    # [1004,1004]
        
    
        
        # 2. norm
        input_imgs = prctile_norm(input_imgs)
        gt_im = prctile_norm(gt_im)
        
        
        
        
        # 3. crop
        #img_size = 1004 if self.wf_upscale else 502
        img_size = gt_im.shape[-1] if self.wf_upscale else gt_im.shape[-1]//2
        threshold = 100 if self.is_beads else 1000
        
        count = 0
        density = 0
        while True:
            count += 1
            if not self.wf_upscale:
                start_w = random.randint(0, img_size-self.patch_size - 1)
                start_h = random.randint(0, img_size-self.patch_size - 1)
                input_imgs_crop = input_imgs[:, start_h:(start_h+self.patch_size), start_w:(start_w+self.patch_size)]
                gt_im_crop = gt_im[(start_h*2):(start_h*2+2*self.patch_size), 
                          (start_w*2):(start_w*2+2*self.patch_size)]
                
            else:
                start_w = random.randint(0, img_size-2*self.patch_size - 1)
                start_h = random.randint(0, img_size-2*self.patch_size - 1)
                input_imgs_crop = input_imgs[:, start_h:(start_h+2*self.patch_size),
                                        start_w:(start_w+2*self.patch_size)]
                gt_im_crop = gt_im[(start_h):(start_h+2*self.patch_size), 
                          (start_w):(start_w+2*self.patch_size)]
                
                                    
            
            if self.is_beads:
                # remove background patches and seive beads density
                if np.sum(np.where(gt_im_crop, 1, 0)) > 50 and calc_beads_density(gt_im_crop, 0.0312) > density:
                    break
                elif count > 20:
                    density *= 0.9
                    count = 0
#                print(calc_beads_density(gt_im_crop, 0.0312))
            
            else:
                # remove background patches
                if np.sum(np.where(gt_im_crop, 1, 0)) > threshold:
                    break
            
            
            
            
        '''
        
        # 3. (rotate) and crop
        if np.random.rand(1) < 0.5:
            while True:
                img_size = 1004 if self.wf_upscale else 502     
                if not self.wf_upscale:
                    start_w = random.randint(0, img_size-self.patch_size - 1)
                    start_h = random.randint(0, img_size-self.patch_size - 1)
                    input_imgs_crop = input_imgs[:, start_h:(start_h+self.patch_size), start_w:(start_w+self.patch_size)]
                    gt_im_crop = gt_im[(start_h*2):(start_h*2+2*self.patch_size), 
                      (start_w*2):(start_w*2+2*self.patch_size)]
                else:
                    start_w = random.randint(0, img_size-2*self.patch_size - 1)
                    start_h = random.randint(0, img_size-2*self.patch_size - 1)
                    input_imgs_crop = input_imgs[:, start_h:(start_h+2*self.patch_size),
                                    start_w:(start_w+2*self.patch_size)]
                    gt_im_crop = gt_im[(start_h):(start_h+2*self.patch_size), 
                      (start_w):(start_w+2*self.patch_size)]
                if np.sum(np.where(gt_im_crop,1,0)) > 0:
                    break
        else:
            while True:
                # rotate
                angle = np.random.randint(0,90)
                input_imgs_rotate = np.ones(input_imgs.shape)
                for i in range(input_imgs.shape[0]):
                    input_imgs_rotate[i,...] = np.array(Image.fromarray(input_imgs[i,...]).rotate(angle, fillcolor=100))
                gt_im_rotate = np.array(Image.fromarray(gt_im).rotate(angle, fillcolor=100))
        
                # crop
                img_size = 1004 if self.wf_upscale else 502 
                if not self.wf_upscale:
                    start_w = random.randint(0, img_size-self.patch_size - 1)
                    start_h = random.randint(0, img_size-self.patch_size - 1)
                    input_imgs_crop = input_imgs_rotate[:, start_h:(start_h+self.patch_size), start_w:(start_w+self.patch_size)]
                    gt_im_crop = gt_im_rotate[(start_h*2):(start_h*2+2*self.patch_size), 
                      (start_w*2):(start_w*2+2*self.patch_size)]
                else:
                    start_w = random.randint(0, img_size-2*self.patch_size - 1)
                    start_h = random.randint(0, img_size-2*self.patch_size - 1)
                    input_imgs_crop = input_imgs_rotate[:, start_h:(start_h+2*self.patch_size),
                                    start_w:(start_w+2*self.patch_size)]
                    gt_im_crop = gt_im_rotate[(start_h):(start_h+2*self.patch_size), 
                      (start_w):(start_w+2*self.patch_size)]
            
                # discard if two many 100 exist
                if np.sum(np.where(gt_im_crop==100,1,0)) == 0 and np.sum(np.where(gt_im_crop,1,0)) > 0:
                    #print('********** Rotate training samples **********')
                    break
        '''
        
        # 4. ToTensor
        input_imgs_tensor = torch.Tensor(input_imgs_crop)               # [9,128,128]          
        gt_im_tensor = torch.Tensor(gt_im_crop[np.newaxis,:,:])         # [1,256,256]
        
        
        
        # 5. aug
        aug = random.randint(0, 8)
        if aug == 1:                 # h方向翻转
            input_imgs_tensor = input_imgs_tensor.flip(1)
            gt_im_tensor = gt_im_tensor.flip(1)
            
            
        elif aug == 2:               # w方向翻转
            input_imgs_tensor = input_imgs_tensor.flip(2)
            gt_im_tensor = gt_im_tensor.flip(2)
            
            
        elif aug == 3:               # 逆时针旋转90
            input_imgs_tensor = torch.rot90(input_imgs_tensor, dims=(1,2))
            gt_im_tensor = torch.rot90(gt_im_tensor, dims=(1,2))
            
            
        elif aug == 4:               # 逆时针旋转90 x 2
            input_imgs_tensor = torch.rot90(input_imgs_tensor, dims=(1,2), k=2)
            gt_im_tensor = torch.rot90(gt_im_tensor, dims=(1,2), k=2)
            
            
        elif aug == 5:               # 逆时针旋转90 x 3
            input_imgs_tensor = torch.rot90(input_imgs_tensor, dims=(1,2), k=3)
            gt_im_tensor = torch.rot90(gt_im_tensor, dims=(1,2), k=3)
            
            
        elif aug == 6:
            input_imgs_tensor = torch.rot90(input_imgs_tensor.flip(1), dims=(1,2))
            gt_im_tensor = torch.rot90(gt_im_tensor.flip(1), dims=(1,2))
           
            
        elif aug == 7:
            input_imgs_tensor = torch.rot90(input_imgs_tensor.flip(2), dims=(1,2))
            gt_im_tensor = torch.rot90(gt_im_tensor.flip(2), dims=(1,2))
           
        
        
        return input_imgs_tensor,gt_im_tensor
    
    def __len__(self):
        return len(self.input_folders)


# =========================================
# SIM Raw data for validating
# =========================================
class BioSR_SIM_val(Dataset):
    def __init__(self, root_dir='D:/code/fluorescence/dataset/BioSR/train_online/F-actin',
                 wf_upscale=0,      
                 sr_scale = 2,      
                 padding = 0,       
                 input_frames = [i for i in range(1,9+1)],   
                 cell_index = [i for i in range(1,5+1)],     
                 snr_index = [1,12],                         
                 ):                       
        
        self.is_cos7_lifeact = 'Lifeact' in root_dir
        self.is_beads = 'beads' in root_dir
        self.is_er = 'ER' in root_dir
        self.is_mt = 'MT' in root_dir 
        gt_file_name = 'gt'

        self.root_dir = root_dir
        self.wf_upscale = wf_upscale
        self.sr_scale = sr_scale
        self.padding = padding
        #self.input_frames = natsorted(input_frames)
        self.input_frames = natsorted([i-1 for i in input_frames])
        self.cell_index = natsorted(cell_index)
        self.snr_index = natsorted(snr_index)
               

        self.input_path = os.path.join(self.root_dir,'Raw')
        self.gt_path = os.path.join(self.root_dir,'GT')
        
        
        cells = ['cell_'+str(i) for i in self.cell_index]
        if self.is_beads or self.is_cos7_lifeact:
            levels = ['seq_'+str(i) for i in self.snr_index]
        else:
            levels = ['level_'+str(i) for i in self.snr_index]
        
        input_folders, gt_files, = [], []
        for cell in cells:
            cell_input_path = os.path.join(self.input_path, cell)
            cell_gt_file = os.path.join(self.gt_path, cell, '{}.tif'.format(gt_file_name))
            
            for level in levels:
                cell_level_input_path = os.path.join(cell_input_path, level)
                input_folders.append(cell_level_input_path)
                if self.is_er:
                    cell_gt_file = os.path.join(self.gt_path, cell, level, '{}.tif'.format(gt_file_name))
                gt_files.append(cell_gt_file)
                       
               
        self.input_folders = input_folders
        self.gt_files = gt_files
        
    
    def __getitem__(self, index):
        input_folder = self.input_folders[index]
        input_files = natsorted(os.listdir(input_folder))    # raw files
        
        # 1. read raw and gt
        input_imgs = []
        for frame in [0,1,2,3,4,5,6,7,8]:
            input_file = input_files[frame]
            input_im = imageio.imread(os.path.join(input_folder, input_file))
            if False:
                input_im = cv2.resize(input_im, (input_im.shape[0]*self.sr_scale, input_im.shape[1]*self.sr_scale),
                                  interpolation = cv2.INTER_CUBIC)
            input_imgs.append(input_im)
        input_imgs = np.array(input_imgs)
        
        if len(self.input_frames) < 9:
            wf = np.mean(input_imgs, axis=0, keepdims=True)
            input_imgs = input_imgs[self.input_frames, :, :]
            input_imgs = np.concatenate([wf,input_imgs], axis=0)
        
                  
        gt_file = self.gt_files[index]
        gt_im = imageio.imread(gt_file)
        
               
        
        # 2. norm ~ [0,1]
        input_imgs = prctile_norm(input_imgs)   
        gt_im = prctile_norm(gt_im)
        
        
        # 3. to tensor
        input_imgs_tensor = torch.Tensor(input_imgs)       
        gt_im_tensor = torch.Tensor(gt_im[np.newaxis,:,:])
        
        
        # 4. padding
        if self.padding != 0:
            #input_imgs_tensor = F.pad(input_imgs_tensor, (0,self.padding,0,self.padding), 'constant', 0)
            input_imgs_tensor = F.pad(input_imgs_tensor.unsqueeze(0), (0,self.padding,0,self.padding), mode='replicate')
            input_imgs_tensor = input_imgs_tensor.squeeze(0)
        
        
        return input_imgs_tensor, gt_im_tensor
    
    def __len__(self):
        return len(self.gt_files)




# ======================================
# SIM Raw data for testing
# ======================================
class BioSR_SIM_test(Dataset):
    def __init__(self, root_dir='D:/code/fluorescence/dataset/BioSR/test/F-actin',
                 snr_level=1,
                 wf_upscale=1,      
                 sr_scale = 2,      
                 padding = 0,       
                 cell_list = [i for i in range(1,5+1,1)],
                 input_frames = [i for i in range(1,9+1)],   
                 ):  
        
        self.root_dir = root_dir
        self.snr_level = snr_level
        self.wf_upscale = wf_upscale
        self.sr_scale = sr_scale
        self.cell_list = natsorted(cell_list)
        #self.input_frames = natsorted(input_frames)
        self.input_frames = natsorted([i-1 for i in input_frames])
        self.padding = padding
        self.is_beads = 'bead' in self.root_dir
        self.is_cos7_lifeact = 'Lifeact' in root_dir
        
        
        self.input_path = os.path.join(self.root_dir,'Raw')
        self.input_sim_folders = []
        
        for cell in self.cell_list:
            if self.is_beads or self.is_cos7_lifeact:
                self.input_sim_folders.append(os.path.join(self.input_path, 'cell_'+str(cell), 'seq_'+str(self.snr_level)))
            else:
                self.input_sim_folders.append(os.path.join(self.input_path, 'cell_'+str(cell), 'level_'+str(self.snr_level)))
        
        
    
    
    def __getitem__(self, index):
        input_folder = self.input_sim_folders[index]
        input_raw_imgs = natsorted(os.listdir(input_folder))     # raw imgs
        
        # read raw
        input_imgs = []
        for frame in [0,1,2,3,4,5,6,7,8]:
            input_raw_name = input_raw_imgs[frame]
            input_raw_img = imageio.imread(os.path.join(input_folder, input_raw_name))
            if self.wf_upscale:
                input_raw_img = cv2.resize(input_raw_img, (input_raw_img.shape[0]*self.sr_scale, input_raw_img.shape[1]*self.sr_scale),
                                  interpolation = cv2.INTER_CUBIC)
            input_imgs.append(input_raw_img)
        input_imgs = np.array(input_imgs)           # [c,502,502] or [c,1004,1004]
        
        if len(self.input_frames) < 9:
            wf = np.mean(input_imgs, axis=0, keepdims=True)
            input_imgs = input_imgs[self.input_frames, :, :]
            input_imgs = np.concatenate([wf,input_imgs], axis=0)
        
                      
          
        # norm ~ [0,1]
        input_imgs = prctile_norm(input_imgs)       # [c,502,502] or [c,1004,1004], c是input_frames的长度
                   
        
        # to tensor
        input_imgs_tensor = torch.Tensor(input_imgs)
                    
        
        if self.padding != 0:
            #input_imgs_tensor = F.pad(input_imgs_tensor, (0,self.padding,0,self.padding), 'constant', 0)
            input_imgs_tensor = F.pad(input_imgs_tensor.unsqueeze(0), (0,self.padding,0,self.padding), mode='replicate')
            input_imgs_tensor = input_imgs_tensor.squeeze(0)
        
        
        return {'img_path': input_folder, 'img': input_imgs_tensor}
    
    def __len__(self):
        return len(self.input_sim_folders)





def create_dataloader(dataset, batch_size, shuffle, num_workers, drop_last, pin_memory):
    
    data_loader = DataLoader(dataset = dataset,
                             batch_size = batch_size,
                             shuffle = shuffle,
                             num_workers = num_workers,
                             drop_last = drop_last,
                             pin_memory = pin_memory)
    
    return data_loader
    
    
    
    
    

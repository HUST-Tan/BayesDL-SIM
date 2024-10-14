# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:27:05 2021

@author: ChefLT
 """

import argparse
import time
import datetime
import torch
import os
import numpy as np
from dataloader import BioSR_SIM_train_online,BioSR_SIM_val,create_dataloader
from model_train import Model_train
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()

# Data
parser.add_argument('--sample', type=str, default='F-actin', help='F-actin / MT / CCPs / beads-1 / beads-2 ')
parser.add_argument('--root_dir', type=str, default=r'D:\LT\项目\5.Bayes-SIM\2.空域不确定性\Nature版本\Demo_BayesDL-SIM\DatasetGenerate\Trainingdata')
parser.add_argument('--SISR_or_SIM', type=str, default='SIM', help='SISR or SIM')
parser.add_argument('--sr_scale', type=int, default=2)
parser.add_argument('--input_frames', type=str, default='1,2,3,4,5,6,7,8,9', help='SIM frames for reconstruction')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--patch_size', type=int, default=128)

# Model
parser.add_argument('--model', type=str, default='BayesDL', help='')
parser.add_argument('--model_D', type=str, default='', help='patchD / unetD')
parser.add_argument('--pretrained_G', type=str, default=r'', help='dir')
parser.add_argument('--pretrained_D', type=str, default='', help='dir')
parser.add_argument('--drop2d', type=float, default=0.0, help='channel dropout before the last conv')
parser.add_argument('--drop', type=float, default=0.0, help='element dropout within model')
parser.add_argument('--n_feat', type=int, default=64, help='')
parser.add_argument('--nGroups', type=int, default=5, help='')
parser.add_argument('--nBlocks', type=int, default=10, help='')
parser.add_argument('--act', type=str, default='relu', help='relu or gelu or silu')
parser.add_argument('--CA_type', type=str, default='CA', help='CA/ECA/1x1Fusion/None')

# Training
parser.add_argument('--train_type', type=str, default='', help='adv / uncertrainty / uncertainty_tail')
parser.add_argument('--likelihood', type=str, default='', help='gauss / laplace / student')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--start_epoch', type=int, default=1, help='')
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--mse_lambda', type=float, default=0)
parser.add_argument('--mae_lambda', type=float, default=1)
parser.add_argument('--fft_lambda', type=float, default=0.01)
parser.add_argument('--ssim_lambda', type=float, default=0.1)
parser.add_argument('--adv_lambda', type=float, default=0.000)
parser.add_argument('--augs', type=str, default='False', help="['none','mixup','cutmix','cutmixup']")
parser.add_argument('--sgld', type=str, default='True', help='using SGLD for posterior inference')
parser.add_argument('--mfvi', type=str, default='False', help='using mean-field-variational-inference for posterior inference')
parser.add_argument('--seed', type=int, default=42)

# Saving
parser.add_argument('--save_model_dir', type=str, default='./checkpoint')
parser.add_argument('--exp_name', type=str, default='BayesDL', help='')

# GPU
parser.add_argument('--gpu_id', type=str, default='0')

args = parser.parse_args()
for arg in vars(args):    # True/False ---> str
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False


# GPU setting
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
args.GPU = [int(i) for i in args.gpu_id.split(',')]


# Args pre-setting
args.is_train = True

sample_list = args.sample.split(',')
root_dir_list = [os.path.join(args.root_dir, i) for i in sample_list]
sample = '+'.join(sample_list)
if args.input_frames == '':
    args.SISR_or_SIM = 'SISR'
save_model_path = os.path.join(args.save_model_dir, sample, args.exp_name+'-'+args.SISR_or_SIM)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

args.augs = ['none','mixup','cutmix','cutmixup'] if args.augs else []

args.weight_decay = 5e-8 if args.sgld else 0
args.param_noise_sigma = 2 if args.sgld else 0
    

# save args to config.txt
open_type = 'a' if os.path.exists(os.path.join(save_model_path, 'config.txt')) else 'w'
with open(os.path.join(save_model_path, 'config.txt'), open_type) as f:
    f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '\n\n')
    for arg in vars(args):
        f.write('{}:{}\n'.format(arg, getattr(args, arg)))
    f.write('\n')

#print('=========================================')
#print(args)
#print('=========================================')


# =====================================================================
#                                 data loader
# =====================================================================
def generate_dataset(root_dir, SISR_or_SIM, patch_size, input_frames, batch_size, 
                     num_workers, cell_index, snr_index, sr_scale=2):
    
    
    train_dataset = []
    val_dataset = []
              
    if SISR_or_SIM == 'SIM':
        ## (1) Training dataset
        for i in range(len(root_dir)):
            train_dataset.append(BioSR_SIM_train_online(root_dir[i], sr_scale=sr_scale, patch_size=patch_size, 
                                               input_frames=input_frames, cell_index = cell_index[i],
                                               snr_index = snr_index[i]))
        ## (2) Validate dataset
        for i in range(len(root_dir)):
            val_dataset.append(BioSR_SIM_val(root_dir[i], sr_scale, input_frames=input_frames, 
                                             cell_index=[1,2,3,4,5], snr_index=snr_index[i]))
    
    # concat dataset
    if len(train_dataset) == 1:
        train_dataset = train_dataset[0]
        val_dataset = val_dataset[0]
    else:
        tmp = train_dataset[0]
        for jj in range(1, len(train_dataset)):
            tmp += train_dataset[jj]
        train_dataset = tmp
        tmp = val_dataset[0]
        for jj in range(1, len(val_dataset)):
            tmp += val_dataset[jj]
        val_dataset = tmp
        
    print('************Training dataset size: %d**************'%(len(train_dataset)))
    print('************Validating dataset size: %d**************'%(len(val_dataset)))
    
    # creat dataloader
    train_data_loader = create_dataloader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                      drop_last=False, pin_memory=False)
    val_data_loader = create_dataloader(val_dataset, 1, shuffle=False, num_workers=num_workers,
                                    drop_last=False, pin_memory=False)
    
    return train_data_loader, val_data_loader


input_frames = [int(i) for i in args.input_frames.split(',')] if args.input_frames else []
cell_index_dict = {'F-actin':[i for i in range(6,51+1)],
                   'MT': [i for i in range(6,55+1)],
                   'CCPs': [i for i in range(6,54+1)],
                   'ER': [i for i in range(6,68+1)],
                   'beads-1': [i for i in range(6,60+1)],
                   'beads-2': [i for i in range(11,150)],
                   'COS7-Lifeact': [i for i in range(6,98+1)],
                   }
snr_index_dict = {'F-actin': [i for i in range(1,9+1)],
                  'MT': [i for i in range(1,9+1)],
                  'CCPs': [i for i in range(1,9+1)],
                  'ER': [i for i in range(1,6+1)],
                  'beads-1': [i for i in range(1,11+1)],
                  'beads-2': [i for i in range(1,15+1)],
                  'COS7-Lifeact': [i for i in range(1,10+1)],
                  }
cell_index_list = [cell_index_dict[k] for k in sample_list]
snr_index_list = [snr_index_dict[k] for k in sample_list]

train_data_loader, val_data_loader = generate_dataset(root_dir_list, args.SISR_or_SIM,
                                                      args.patch_size, input_frames, 
                                                      args.batch_size, args.num_workers, 
                                                      cell_index=cell_index_list,
                                                      snr_index=snr_index_list, 
                                                      sr_scale=args.sr_scale
                                                      )
    
    


# =====================================================================
#                             model / loss / optimizer
# =====================================================================
# model
INPUT_C = 1 if args.SISR_or_SIM == 'SISR' else len(input_frames)+1
if INPUT_C > 9: INPUT_C = 9

model = Model_train(args, INPUT_C)
writer = SummaryWriter(log_dir = save_model_path)

if os.path.exists(args.pretrained_G) and not args.mfvi:
    psnr_val, ssim_val, _, _ = model.model_eval(val_data_loader)
    psnr_epoch_best = psnr_val
    ssim_epoch_best = ssim_val
    print('Pretrained performance --- Validate PSNR: %.3f --- Best validate PSNR: %.3f'%(psnr_val,psnr_epoch_best))
    print('                       --- Validate SSIM: %.3f --- Best validate SSIM: %.3f'%(ssim_val,ssim_epoch_best))
    



# =====================================================================
#                                Train
# =====================================================================
psnr_epoch_best,ssim_epoch_best = 0,0
iteration = 0            
val_psnr_history, training_loss_history= [],[]

for epoch in range(args.start_epoch, args.epoch+1):
       
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_mae_loss, epoch_mse_loss, epoch_fft_loss, epoch_ssim_loss, epoch_adv_loss, epoch_resi_loss, epoch_uncer_reg = 0, 0, 0, 0, 0, 0, 0
    
    # update lr
    model.update_lr(epoch)
    
    for i,data in enumerate(train_data_loader):
        iteration += 1
        
        model.set_input(data)
        model.forward()
        model.optimize_paras(epoch)
        
        
        if model.learn_std and args.likelihood in ['gauss','laplace']:
            epoch_resi_loss += model.resi_loss.item()
            epoch_uncer_reg += model.uncer_reg.item()
        elif not model.learn_std:
            
            epoch_mae_loss += model.mae_loss.item()
            epoch_mse_loss += model.mse_loss.item()
            epoch_fft_loss += model.fft_loss.item()
            epoch_ssim_loss += model.ssim_loss.item()
            epoch_adv_loss += model.adv_loss.item()
        
        epoch_loss += model.g_loss.item()

    

    print('[Epoch {}] \t [Total: {:.3f}]'.format(epoch, epoch_loss))  
    print('Lasting Time %d seconds'%int(time.time()-epoch_start_time))
    print('=========================')


    if not ('uncertainty_tail' in args.train_type):
        if epoch==1 or epoch % 20 == 0:     # validating every 20 epochs
            #model.save_model(save_model_path, epoch)      
            psnr_val,ssim_val = model.model_eval(val_data_loader)
            val_psnr_history.append(psnr_val)
                        
            if psnr_val > psnr_epoch_best:
                psnr_epoch_best = psnr_val
                model.save_model(save_model_path, 'bestPSNR')
            print('Epoch %d --- Validation PSNR: %.3f --- Best validation PSNR: %.3f'%(epoch, psnr_val, psnr_epoch_best))
            model.save_model(save_model_path, 'latest') 
#            model.logger(writer, iteration)
    
    training_loss_history.append(epoch_loss)



# save latest.pth
if not ('uncertainty_tail' in args.train_type):
    model.save_model(save_model_path, 'latest')
    np.savez(os.path.join(save_model_path, 'PSNR_record.npz'), np.array(val_psnr_history))
    np.savez(os.path.join(save_model_path, 'TrainingLoss_record.npz'), np.array(training_loss_history))
    writer.close()

else:
    model.save_model(save_model_path, 'SGLD_{}'.format(args.likelihood))
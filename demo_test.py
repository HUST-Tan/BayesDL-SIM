# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 22:29:53 2021

@author: ChefLT
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tifffile as tiff
import scipy.stats

from model_test import Model_test
import utils
#from skimage.feature import local_binary_pattern


# =====================================================================
#                                data
# =====================================================================
def read_and_process(args):
    raw = tiff.imread(args.test_data)
    raw = utils.prctile_norm(raw)
    raw = torch.Tensor(raw).unsqueeze(0)
    return raw


# =====================================================================
#                                 model
# =====================================================================
def define_model(args):
    input_frames = [int(i) for i in args.input_frames.split(',')] if args.input_frames else []
    INPUT_C = 1 if args.SISR_or_SIM == 'SISR' else len(input_frames)+1
    if INPUT_C > 9: INPUT_C = 9
    
    model = Model_test(args, INPUT_C, args.device)
    
    model.netG.eval()
    if args.sgld:
        model.N_infer = 5
    else:
        model.N_infer = 1
    
    return model
        

    

# ====================================================================
#                               Testing
# ====================================================================
def model_inference(args, raw, model):
    
    ## Deterministic
    if (not model.learn_std) and (not args.sgld):
        with torch.no_grad():
            raw = raw.to(args.device)
            sr = model.netG(raw)
            sr_numpy = utils.tensor2im(sr, norm=False)
        
        save_path = os.path.join(os.path.dirname(args.test_data), 'BayesDL-results')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        utils.save_image(os.path.join(save_path, 'SR.tif'), sr_numpy)
    
    else:
        with torch.no_grad():
            raw = raw.to(args.device)
            sr_list, std_list = [], []
            
            for mc in range(model.N_infer):
                if args.sgld: model.load_model(os.path.join(args.pretrained_G, 'SGLD_gauss_#{}.pth'.format(mc+1)))
                output = model.netG(raw)
                if args.likelihood in ['gauss', 'laplace']:
                    output_sr, output_std = output
                    if args.likelihood == 'gauss': output_std = output_std/2
                elif args.likelihood == 'student':
                    output_sr, log_precision, log_df = output
                    output_std = 0.5*(torch.log(torch.exp(log_df) + 2) - log_precision - log_df)
                else:
                    output_sr = output
                    output_std = None
                
                sr_numpy = utils.tensor2im(output_sr, norm=True)
                sr_list.append(sr_numpy)
                
                if output_std is not None:
                    output_std = np.exp(output_std[0,0,...].detach().cpu().numpy())
                    std_list.append(output_std)
            
            
            # SR
            sr_list = np.array(sr_list)
            sr_numpy = np.mean(sr_list, axis=0, keepdims=False)
            sr_numpy = (65535*sr_numpy).astype('uint16')
            
            # aleatoric
            if len(std_list)>0:
                std_list = np.array(std_list)
                aleatoric = np.mean(std_list, axis=0, keepdims=False)
                #aleatoric[aleatoric>0.333] = 0  # remove outliers
                if aleatoric.max() > 0.333: aleatoric = utils.adaptive_median_filter(aleatoric, 7)
            
            # epistemic
            epistemic = np.std(sr_list, axis=0, keepdims=False)
            
            # Saving
            save_path = os.path.join(os.path.dirname(args.test_data), 'BayesDL-results')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            utils.save_image(os.path.join(save_path, 'SR.tif'), sr_numpy)
            if 'epistemic' in locals() or 'epistemic' in globals():
                imageio.imwrite(os.path.join(save_path, 'Epistemic.tif'), epistemic)
                plt.figure(1)
                plt.imshow(epistemic, cmap=plt.get_cmap('jet'), vmin=0, vmax=0.05)
                plt.axis('off')
                plt.colorbar(shrink=0.99, pad=0.01, ticks=[0,0.05])
                plt.savefig(os.path.join(save_path, 'Epistemic.png'), dpi=600, bbox_inches='tight',pad_inches=-0.001)
                
            if 'aleatoric' in locals() or 'aleatoric' in globals():
                imageio.imwrite(os.path.join(save_path, 'Aleatoric.tif'), aleatoric)
                plt.figure(2)
                plt.imshow(aleatoric, cmap=plt.get_cmap('jet'), vmin=0, vmax=0.1)
                plt.axis('off')
                plt.colorbar(shrink=0.99, pad=0.01, ticks=[0, 0.1])
                plt.savefig(os.path.join(save_path, 'Aleatoric.png'), dpi=600, bbox_inches='tight',pad_inches=-0.001)
                

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--test_data', type=str, default=r'./Data/F-actin/raw.tif', help='')
    parser.add_argument('--SISR_or_SIM', type=str, default='SIM', help='SISR or SIM')
    parser.add_argument('--sr_scale', type=int, default=2)
    parser.add_argument('--input_frames', type=str, default='1,2,3,4,5,6,7,8,9', help='which frames to input when SISR_or_SIM == SIM')

    # model
    parser.add_argument('--model', type=str, default='BayesDL', help='')
    parser.add_argument('--drop', type=float, default=0.0, help='')
    parser.add_argument('--drop2d', type=float, default=0.0, help='')
    parser.add_argument('--likelihood', type=str, default='gauss', help='')
    parser.add_argument('--mfvi', type=bool, default=False, help='')
    parser.add_argument('--sgld', type=bool, default=True, help='')
    parser.add_argument('--n_feat', type=int, default=64, help='')
    parser.add_argument('--act', type=str, default='relu', help='relu or gelu')
    parser.add_argument('--nGroups', type=int, default=5, help='')
    parser.add_argument('--nBlocks', type=int, default=10, help='')
    parser.add_argument('--CA_type', type=str, default='CA', help='')

    parser.add_argument('--save_path', type=str, default='BayesDL-SIM')  # for saving results
    parser.add_argument('--pretrained_G', type=str, default=r'./checkpoint/F-actin/pretrained_G')

    # gpu
    parser.add_argument('--gpu_id', type=str, default='0', help='')

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.GPU = [int(i) for i in args.gpu_id.split(',')]
    
    
    raw = read_and_process(args)
    model = define_model(args)
    model_inference(args, raw, model)
            
            
            
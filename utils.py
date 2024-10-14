# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:35:35 2021

@author: Administrator
"""
import numpy as np
import torch
#import cv2
import imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from math import pi
from scipy import ndimage

def prctile_norm(x, min_prc=0, max_prc=100):  # min_prc=0 max_prc=100时，即min-max normalization
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def mse(x, y):
    return np.mean((x-y)**2)


def nrmse(x, y, nrmse_type='minmax'):
    '''
    y是Ground Truth
    '''
    rmse = (mse(x,y))**0.5
    if nrmse_type == 'minmax':
        return rmse/(np.max(y)-np.min(y))


def psnr(x, y):
    return peak_signal_noise_ratio(x, y)


def ssim(x, y):
    return structural_similarity(x, y)


def tensor2im(im_tensor, norm=True):
    '''
    input: im_tensor ~ [b,c,h,w]
    return: numpy array after clamp
    '''
    im_tensor = torch.clamp(im_tensor, 0, 1)
    if len(im_tensor.shape) == 4:
        im = im_tensor[0,0,:,:].detach().cpu().numpy()
    elif len(im_tensor.shape) == 3:
        im = im_tensor[0,:,:].detach().cpu().numpy()
    
#    im = prctile_norm(im.squeeze(), 0, 100)
    
    if not norm:   # 输出uint16
        im = (65535*im).astype('uint16')
    return im



def save_image(save_path, img_numpy):
    import imageio
    imageio.imwrite(save_path, img_numpy)
   
    
   
def diffxy(img, order=3):
    for _ in range(order):
        img = prctile_norm(img)
        d = np.zeros_like(img)
        dx = (img[1:-1, 0:-2] + img[1:-1, 2:]) / 2
        dy = (img[0:-2, 1:-1] + img[2:, 1:-1]) / 2
        d[1:-1, 1:-1] = img[1:-1, 1:-1] - (dx + dy) / 2  #中心像素减去四邻域均值
        d[d < 0] = 0
        img = d
    return img


def rm_outliers(img, order=3, thresh=0.2):
    img_diff = diffxy(img, order)
    mask = img_diff > thresh
    img_rm_outliers = img
    img_mean = np.zeros_like(img)
    for i in [-1, 1]:
        for a in range(0, 2):
            img_mean = img_mean + np.roll(img, i, axis=a)
    img_mean = img_mean / 4   # img_mean的各像素值 = img对应点的四邻域均值
    img_rm_outliers[mask] = img_mean[mask]     # 用img中的outlier，用其四邻域均值代替原像素
    img_rm_outliers = prctile_norm(img_rm_outliers)
    return img_rm_outliers


# =====================================================================
#                    Average photons count for raw data
# =====================================================================
def average_photons_count(x, Gaussian_sigma=5.0, ratio=0.2, conversion_factor=0.6026):
    
    # (1) substract the camera background (98)
    #     this step has been done in our data processing
    x = x-98
    x[x<0] = 0

    # (2) apply a Gaussian low-pass filter, default sigma=5, ksize is set to 9 as in qiao's paper
    x_blur = cv2.GaussianBlur(x, ksize=(9,9), sigmaX=Gaussian_sigma, sigmaY=Gaussian_sigma, borderType=cv2.BORDER_REPLICATE)

    # (3) percentile-normlization
    #     extracting the feature-only region with threshold 0.2
#    x = utils.prctile_norm(x_blur)
    x = x[x_blur > np.max(x_blur)*ratio]
    
    # (4) calculate the average sCMOS count
    average_sCMOS_count = np.mean(x)
    
    # (5) convert the sCMOS count to the photon count by a conversion factor 0.6026, which is measured by Hamamatsu's protocol
    average_photon_count = average_sCMOS_count*conversion_factor
    return average_photon_count


# ======================
# mask
# ======================
def get_wf_mask(NA=1.41,em_lambda=525, nm_per_pixel=31.2, img_size=1004):
    '''
    NA: numerical aperature
    em_lambda: wave length of emission light
    nm_per_pixel: distance of each pixel
    img_size: pixels of img width or height
    '''
    
    r = round(2*NA*nm_per_pixel*img_size/em_lambda)
    
    mask = np.zeros((img_size, img_size))
    center = (img_size-1)/2
    for i in range(img_size):
        for j in range(img_size):
            if (i-center)**2+(j-center)**2 < r**2:
                mask[i,j] = 1
    return mask


# =======================
# Measuring LF and HF fidelity
# =======================  
def fft(img):                             # numpy version
    #im_fft = np.fft.fft2(img)
    #im_fft_sh = np.fft.fftshift(im_fft)
    #return im_fft_sh   # 返回中心化的频谱
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def Freq_metric(x, y, lf_mask='./masks/WF_1004.tif', hf_mask='./masks/HF_1004.tif'):
    
    lf_mask = imageio.imread(lf_mask)
    hf_mask = imageio.imread(hf_mask)
    assert x.shape == y.shape == lf_mask.shape == hf_mask.shape
    
    assert x.min() >= 0 and x.max() <= 1
    assert y.min() >= 0 and y.max() <= 1
    
    # (1) freq error map
    x_freq = fft(x)
    y_freq = fft(y)
    freq_errorMap = np.abs(x_freq - y_freq)
    
    # (2) LF
    lf_error = np.sum(freq_errorMap*lf_mask)/np.sum(lf_mask)
    LF_metric = 10*np.log(1000/lf_error)
    
    # (3) HF
    hf_error = np.sum(freq_errorMap*hf_mask)/np.sum(hf_mask)
    HF_metric = 10*np.log(200/hf_error)
    
    return LF_metric, HF_metric


def fft_torch(img_tensor):                # torch version
    x = torch.fft.fftshift(img_tensor,dim=[-2,-1])
    x = torch.fft.fft2(x,dim=[-2,-1])
    return torch.fft.fftshift(x,dim=[-2,-1])



def linmap(val, valMin, valMax, mapMin=None, mapMax=None):
    """
    :param val: Input value
    :param valMin: Minimum value of the range of val
    :param valMax: Maximum value of the range of val
    :param mapMin: Minimum value of the new range of val
    :param mapMax: Maximum value of the new range of val
    :return: Rescaled value
    """
    # normalize the data between valMin and valMax
    if mapMin is None and mapMax is None:
        mapMin = valMin
        mapMax = valMax
        valMin = torch.min(val)
        valMax = torch.max(val)
    # convert the input value between 0 and 1
    tempVal = (val - valMin) / (valMax - valMin)
    # clamp the value between 0 and 1
    tempVal[tempVal < 0] = 0
    tempVal[tempVal > 1] = 1
    # rescale and return
    return tempVal * (mapMax - mapMin) + mapMin


def apodImRect(vin, N, device=torch.device('cpu')):
    """
    :param vin: Input image~[b,c,h,w]
    :param N: Number of pixels of the apodization
    :return: [Apodized image, Mask used to apodize the image]
    """
    Nx = vin.shape[-1]   # W
    Ny = vin.shape[-2]   # H
    x = torch.abs(torch.linspace(-Nx / 2, Nx / 2, Nx, device=device)).unsqueeze(0)
    y = torch.abs(torch.linspace(-Ny / 2, Ny / 2, Ny, device=device)).unsqueeze(0)
    mapx = x > (Nx / 2 - N)
    mapy = y > (Ny / 2 - N)
    
    d = (-torch.abs(x) - torch.mean(-torch.abs(x[mapx]))) * mapx  # plt.plot(range(len(d[0])), d[0].cpu().numpy())
    d = linmap(d, -pi / 2, pi / 2)
    d[mapx==0] = pi / 2
    maskx = (torch.sin(d) + 1) / 2  # plt.plot(range(len(maskx[0])), maskx[0].cpu().numpy())
    
    d = (-torch.abs(y) - torch.mean(-torch.abs(y[mapy]))) * mapy
    d = linmap(d, -pi / 2, pi / 2)
    d[mapy==0] = pi / 2
    masky = (torch.sin(d.t())+1)/2
    
    mask = torch.tile(masky, (1,vin.shape[-1])) * torch.tile(maskx, (vin.shape[-2], 1))
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    return vin*mask



def adaptive_median_filter(image, max_window_size):
    filtered_image = image.copy()
    height, width = image.shape[:2]

    for y in range(height):
        for x in range(width):
            neighborhood = image[max(0, y - max_window_size):min(height - 1, y + max_window_size),
                                 max(0, x - max_window_size):min(width - 1, x + max_window_size)]

            median = cv2.medianBlur(neighborhood, 3)

            if image[y, x] > median.mean() * 2:
                filtered_image[y, x] = median.mean()

    return filtered_image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
import imageio
from utils import fft_torch

from torch.autograd import Variable
from math import exp


# ==============================
#       L1 loss (MAE)
# ==============================
class L1Loss(nn.Module):
    """ (L1)"""

    def __init__(self, eps=1e-3):
        super(L1Loss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.abs(diff))
        return loss


# ==============================
#      Charbonnier loss
# ==============================
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


# ==============================
#            Edge loss
# ==============================
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss



# ==============================
# Frequency loss
# ==============================
    
class FFT_loss(nn.Module):
    def __init__(self, loss_type='L1', alpha=0, mask='', device=torch.device('cpu')):
        super(FFT_loss, self).__init__()
        self.loss_type = loss_type    # use L1 or L2 distance for each frequency
        self.alpha = alpha            # how huch the model focus
        self.device = device
#        self.norm = 'backward'
        
        if mask:
            self.mask = torch.tensor(imageio.imread(mask)).to(self.device)
        else:
            self.mask = ''
    
    def compute_w(self, freq_d):
        w = 1 if self.alpha == 0 else (freq_d ** self.alpha).detach()
        
        if self.mask != '':
            w *= self.mask
            
        return w

    def forward(self, x, y):
        x_fft = fft_torch(x)
        y_fft = fft_torch(y)
        #x_fft = torch.fft.rfft2(x, norm=self.norm)
        #y_fft = torch.fft.rfft2(y, norm=self.norm)

        if self.loss_type == 'L1':
            freq_d = torch.abs(x_fft.real-y_fft.real) + torch.abs(x_fft.imag-y_fft.imag)
        elif self.loss_type == 'L2':
            freq_d = (x_fft.real-y_fft.real)**2 + (x_fft.imag-y_fft.imag)**2
        
        w = self.compute_w(freq_d)
        
        return 0.5*torch.mean(w*freq_d)




# ==============================
#          wdct loss
# ==============================
def get_dct_A(H,W):
    A = np.zeros([H, W]).astype("float")
    for i in range(H):
        for j in range(W):
            if i == 0:
                a = 1.0 / np.sqrt(W)
            else:
                a = np.sqrt(2.0) / np.sqrt(W)
            A[i][j] = a * math.cos((j + 0.5) * math.pi * i / H)
    AT = A.T
    return A,AT

def F_mask(threshold):
    highF_mask = np.zeros((256,256))
    lowF_mask = np.zeros((256,256))
    for i in range(256):
        for j in range(256):
            if i**2 + j**2 < threshold:
                highF_mask[i,j] = 0.
                lowF_mask[i,j] = 1.
            else:
                highF_mask[i,j] = 1.
                lowF_mask[i,j] = 0.
    return highF_mask,lowF_mask


class WdctLoss(nn.Module):
    def __init__(self,patch_size = 256,batch_size = 4, loss_type = 'low', threshold = 12100, device = 0):
        super(WdctLoss, self).__init__()
        A,AT = get_dct_A(patch_size, patch_size)
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        A = Tensor(A)
        AT = Tensor(AT)
        self.A = A.expand(batch_size, 3, patch_size, patch_size)     #shape：[B,C,H,W]
        self.AT = AT.expand(batch_size, 3, patch_size, patch_size)   #shape：[B,C,H,W]
        
        highF_W,lowF_W = F_mask(threshold)
        weight = Tensor(lowF_W) if loss_type == 'low' else Tensor(highF_W)
        self.weight = weight.expand(batch_size, 3, patch_size, patch_size)
        
        self.l1loss = L1Loss()
        
        if torch.cuda.is_available():
            self.A = self.A.cuda(device)
            self.AT = self.AT.cuda(device)
            self.weight = self.weight.cuda(device)
    
    def dct_(self,im):
        dct = torch.matmul(self.A,im)
        dct = torch.matmul(dct,self.AT)
        return dct
    
    
    def forward(self,im1,im2):
        dct1 = self.dct_(im1)
        dct2 = self.dct_(im2)
        
        wdct1 = torch.mul(dct1,self.weight)
        wdct2 = torch.mul(dct2,self.weight)
        
        loss = self.l1loss(wdct1,wdct2)
        

        return loss


# ============================
#       Perceptual loss
# ============================
class PerceptualLoss():
	
    def contentFunc(self,layer_idx):
        conv_3_3_layer = layer_idx
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model
		
    def __init__(self,layer_idx = 14):
        self.criterion = L1Loss()
        self.contentFunc = self.contentFunc(layer_idx)
			
    def __call__(self, fakeIm, realIm):
        b,c,h,w = fakeIm.shape
        if c == 1:
            fakeIm = fakeIm.repeat(1,3,1,1)
            realIm = realIm.repeat(1,3,1,1)
        
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss



# ============================
#     SSIM / MS-SSIM loss
# ============================


def type_trans(window,img):
    if img.is_cuda:
        window = window.cuda(img.get_device())
    return window.type_as(img)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    # print(mu1.shape,mu2.shape)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mcs_map  = (2.0 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    # print(ssim_map.shape)
    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        window = create_window(self.window_size,channel)
        window = type_trans(window,img1)
        ssim_map, mcs_map =_ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return ssim_map


class MS_SSIM(torch.nn.Module):
    def __init__(self, window_size = 11,size_average = True):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        # self.channel = 3

    def forward(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))

        msssim = Variable(torch.Tensor(levels,))
        mcs    = Variable(torch.Tensor(levels,))

        if torch.cuda.is_available():
            weight =weight.cuda()
            msssim=msssim.cuda()
            mcs=mcs.cuda()

        _, channel, _, _ = img1.size()
        window = create_window(self.window_size,channel)
        window = type_trans(window,img1)

        for i in range(levels): #5 levels
            ssim_map, mcs_map = _ssim(img1, img2,window,self.window_size, channel, self.size_average)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            # print(img1.shape)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1 #refresh img
            img2 = filtered_im2

        return torch.prod((msssim[levels-1]**weight[levels-1] * mcs[0:levels-1]**weight[0:levels-1]))
        # return torch.prod((msssim[levels-1] * mcs[0:levels-1]))
        #torch.prod: Returns the product of all elements in the input tensor


# =========================
# GAN Loss
# =========================
class CriterionGan(nn.Module):
    '''
    由判别器的输出，计算loss
    '''
    def __init__(self, use_l1=False, real_label=1.0, fake_label=0.0, tensor = torch.FloatTensor, device=torch.device('cuda:0')):
        super(CriterionGan, self).__init__()
        self.real_label = 1.0
        self.fake_label = 0.0
        self.tensor = tensor
        self.device = device
        self.loss = nn.L1Loss() if use_l1 else nn.BCELoss()  # compute the D_output and label
        
        self.real_label_tensor = None
        self.fake_label_tensor = None
    
    def get_label(self, D_output, is_real):
        
        if is_real:
            if (self.real_label_tensor is not None) and (self.real_label_tensor.numel() == D_output.numel()):
                label = self.real_label_tensor
            else:
                self.real_label_tensor = self.tensor(size=D_output.size()).fill_(1.0).to(self.device)
                label = self.real_label_tensor
                
        else:
            if (self.fake_label_tensor is not None) and (self.fake_label_tensor.numel() == D_output.numel()):
                label = self.fake_label_tensor
            else:
                self.fake_label_tensor = self.tensor(size=D_output.size()).fill_(0.0).to(self.device)
                label = self.fake_label_tensor
        
        assert label.requires_grad == False
        return label
    
    def __call__(self, D_output, is_real):
        label = self.get_label(D_output, is_real) # D_output对应的label，0 or 1
        
        return self.loss(D_output, label)         # D_output和label的距离


class DiscLoss():
    def __init__(self, use_l1=False, tensor=torch.FloatTensor, device=torch.device('cuda:0')):
        self.criterion = CriterionGan(use_l1=use_l1, tensor=tensor, device=device)
    
    def get_g_loss(self, netD, fakeB):
        pred_fakeB = netD(fakeB)
        return self.criterion(pred_fakeB, 1)   # 让fakeB的分数接近1
    
    def get_d_loss(self, netD, fakeB, realB):
        pred_fakeB = netD(fakeB.detach())       # detach防止梯度回传到生成器
        
        loss_fakeB = self.criterion(pred_fakeB, 0)
        
        
        pred_realB = netD(realB)
        loss_realB = self.criterion(pred_realB, 1)
        
        return (loss_fakeB + loss_realB)/2
 

class DiscLoss_unetD():
    def __init__(self, use_l1=False, tensor=torch.FloatTensor):
        self.criterion = CriterionGan(use_l1=use_l1, tensor=tensor)
        self.loss_fn = torch.nn.MSELoss()
    
    def rand_bbox(self, size, ratio):
        H = size[0]
        W = size[1]
        
        cut_rat = np.sqrt(1. - ratio)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    
    
    def get_g_loss(self, netD, fakeB, realB):
        enc_fakeB, dec_fakeB, enc_fakeB_feat, dec_fakeB_feat = netD(fakeB)
        enc_realB, dec_realB, enc_realB_feat, dec_realB_feat = netD(realB)
        
        # feature matching loss
        loss_FM = []
        for i in range(len(enc_fakeB_feat)):
            loss_FM += [self.loss_fn(enc_fakeB_feat[i], enc_realB_feat[i])]
            loss_FM += [self.loss_fn(dec_fakeB_feat[i], dec_realB_feat[i])]
        loss_FM = torch.mean(torch.stack(loss_FM))
        
        # GAN loss
        loss_adv = []
        loss_adv += [torch.nn.ReLU()(1.0 - enc_fakeB).mean()]
        loss_adv += [torch.nn.ReLU()(1.0 - dec_fakeB).mean()]
        loss_adv = torch.mean(torch.stack(loss_adv))
        
        return loss_FM + 0.001*loss_adv
        
    
    def get_d_loss(self, netD, fakeB, realB):
        fakeB = fakeB.detach()
        enc_realB, dec_realB, _, _ = netD(realB)
        enc_fakeB, dec_fakeB, _, _ = netD(fakeB)
        
        # 让realB的enc/dec分数接近1
        loss_realB_enc = torch.nn.ReLU()(1.0 - enc_realB).mean()
        loss_realB_dec = torch.nn.ReLU()(1.0 - dec_realB).mean()
        
        # 让fakeB的enc/dec分数接近0
        loss_fakeB_enc = torch.nn.ReLU()(1.0 + enc_fakeB).mean()
        loss_fakeB_dec = torch.nn.ReLU()(1.0 + dec_fakeB).mean()
        
        
        loss_D = loss_realB_enc + loss_realB_dec
        
        
        # CutMix
        fakeB_cutmix = fakeB.clone()
        if torch.rand(1) < 0.5:     # p = 0.5 for cutmix
            r_mix = torch.rand(1)   # fake/real ratio
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(fakeB.shape[2:], r_mix)
            fakeB_cutmix[:,:,bby1:bby2,bbx1:bbx2] = realB[:,:,bby1:bby2,bbx1:bbx2]
            r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (fakeB_cutmix.shape[-1] * fakeB_cutmix.shape[-2]))
            
            # enc/dec of cutmix
            enc_mix, dec_mix, _, _ = netD(fakeB_cutmix)
            
            loss_fakeB_enc = torch.nn.ReLU()(1.0 + enc_mix).mean()
            loss_fakeB_dec = torch.nn.ReLU()(1.0 + dec_mix).mean()
            
            dec_fakeB[:,:,bby1:bby2,bbx1:bbx2] = dec_realB[:,:,bby1:bby2,bbx1:bbx2]
            
            loss_consis = self.loss_fn(dec_fakeB, dec_mix)
            
            loss_D += loss_consis
        
        loss_D += loss_fakeB_enc + loss_fakeB_dec
        
        return loss_D
        
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:57:35 2021

@author: ChefLT
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ========================
#           Conv
# ========================
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride, groups=groups)


# =======================
# Complex Conv
# =======================
class ComplexConv2d_(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, stride=1, groups=1):
        super(ComplexConv2d_, self).__init__()
        self.conv_real = conv(in_channels, out_channels, kernel_size, bias, stride, groups)
        self.conv_imag = conv(in_channels, out_channels, kernel_size, bias, stride, groups)
       
    def forward(self, input):
        '''
        input: [real, imag]
        '''
    #    assert input[0].shape == input[1].shape       
        output_real = self.conv_real(input[0]) - self.conv_imag(input[1])
        output_imag = self.conv_real(input[1]) + self.conv_imag(input[0])
        
        return [output_real, output_imag]



class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, stride=1, groups=1, 
                 in_spatial=False, out_spatial=False, fft_type='fft'):
        super(ComplexConv2d, self).__init__()
        self.in_spatial = in_spatial
        self.out_spatial = out_spatial
        self.fft_type = fft_type
        self.complex_conv = ComplexConv2d_(in_channels, out_channels, kernel_size, bias, stride, groups)
    
    def forward(self, x):
        '''
        x: feature in spatial domain of shape [b,c,h,w] or in the frequency domain of [real, imag]
        '''
        if self.in_spatial:
            if self.fft_type == 'rfft':
                _,_,H,W = x.shape
                y = torch.fft.rfft2(x, norm='backward')
            else:
                y = torch.fft.fft2(x)
        else:
            y = torch.complex(x[0],x[1])
            
        y = self.complex_conv([y.real, y.imag])
        
        if not self.out_spatial:
            return y
        else:
            y = torch.complex(y[0], y[1])
            if self.fft_type == 'rfft':
                assert H and W
                y = torch.fft.irfft2(y, s=(H,W), norm='backward')
            elif self.fft_type == 'fft':
                y = torch.fft.ifft2(y).real
            return y
        

class CRelu(nn.Module):
    def __init__(self, inplace=True):
        super(CRelu, self).__init__()
        self.act = nn.ReLU(inplace=inplace)
    
    def forward(self, input):
        '''
        input: [real, imag]
        '''
    #    assert input[0].shape == input[1].shape
        return [self.act(input[0]), self.act(input[1])]


# ======================================
#         Channal Attention (CA) Layer
# ======================================
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid() )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# =========================================
# ECA layer
# =========================================
class ECAlayer(nn.Module):
    def __init__(self, kernel_size):
        super(ECAlayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):      # [bs,c,h,w]
        y = self.avg_pool(x)  # [bs,c,1,1]
        y = y.squeeze(-1).permute(0,2,1) # [bs,1,c]
        y = self.conv1d(y)    # [bs,1,c]
        y = self.sigmoid(y)   # [bs,1,c]
        y = y.permute(0,2,1).unsqueeze(-1) # [bs,c,1,1]
        return x * y.expand_as(x)



# =========================================
# BasicBlock (conv - (bn) - Relu)
# =========================================
class BasicBlock(nn.Sequential):
    '''
    conv-(bn)-Relu
    '''
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)



# =======================================================
# Residual Block (RB): x + res_scale*conv(relu(conv(x)))
# =======================================================
class RB(nn.Module):
    '''
    conv-ReLu-conv
    '''
    def __init__(self,n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, fft_branch=False):
        super(RB,self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        
        if fft_branch:
            self.fft_branch = FFT_layer(n_feat, kernel_size=1)
        else:
            self.fft_branch = None
    
    def forward(self,x):
        res = self.body(x).mul(self.res_scale)
        res += x
        if self.fft_branch is not None:
            fft_res = self.fft_branch(x)
            res += fft_res
        return res


# =========================================
# Residual channal attention block (RCAB)
# =========================================
# RCAB  ~  CONV-RELU-CONV-CALayer
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction=16, bias=True, bn=False, ln=False, act=nn.ReLU(True), 
        res_scale=1, CA_type='CA', skip=True, fft_branch='FFT_Layer'):

        super(RCAB, self).__init__()
        self.fft_branch = fft_branch
        self.skip = skip
        
        modules_body = []
        if ln:
            modules_body.append(LayerNorm2d(n_feat))
        
#        modules_body.append(conv(n_feat, n_feat, kernel_size=1, bias=bias))
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#        modules_body.append(conv(n_feat, n_feat, kernel_size=1, bias=bias))
        
        if bn: modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#        modules_body.append(DeformableConv2d(n_feat, n_feat, kernel_size, bias=bias))
        if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            
#        for i in range(2):
#            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#            if add_HPF and i == 0: modules_body.append(HPL(n_feat,kernel_size))  # if adding high-pass filter after the first conv
#            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#            if i == 0: modules_body.append(act)
            
        if CA_type == 'CA':
            self.CA = CALayer(n_feat, reduction)
        elif CA_type == 'ECA':
            self.CA = ECAlayer(5)
        elif CA_type == 'None':
            self.CA = nn.Identity()
        elif CA_type == '1x1Fusion':
            self.CA = conv(n_feat, n_feat, 1)
            
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
        
        if self.fft_branch == 'FFT_Layer':
            self.fft_body = FFT_layer(n_feat)
        else:
            self.fft_body = None


    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        
        #res = self.CA(res)
        
        if self.fft_body is not None:
            fft_res = self.fft_body(x)
            res += fft_res
        
        res = self.CA(res)
        
        if self.skip:
            return res + x
        else:
            return res
        


# ===============================
#    Pixel shuffle Upsampler
# ===============================
class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # if scale = 2^n ===> conv+PS
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
        

        


class FFT_layer(nn.Module):
    '''
    using ComplexConv2d and CReLu, reduce the number of parameters of FFT_layer to the half.
    '''
    def __init__(self, n_feat, kernel_size=1, norm='backward', skip=False, in_spatial=True, out_spatial=True):
        super(FFT_layer, self).__init__()
        self.n_feat = n_feat
        self.norm = norm
        self.skip = skip
        self.in_spatial = in_spatial
        self.out_spatial = out_spatial
        
        #self.complex_conv1 = ComplexConv2d(n_feat, n_feat, kernel_size)
        #self.complex_conv2 = ComplexConv2d(n_feat, n_feat, kernel_size)
        #self.act = nn.ReLU(inplace=True)
        self.main_fft = nn.Sequential(ComplexConv2d(n_feat, n_feat, kernel_size), CRelu(True), ComplexConv2d(n_feat, n_feat, kernel_size))

    
    def forward(self, x):
        if self.in_spatial:
            _,_,H,W = x.shape            
            x = torch.fft.rfft2(x, norm=self.norm)
            y = self.main_fft([x.real, x.imag])                # [real part, imaginary part]
        else:
            assert isinstance(x, list) and len(x) == 2
            _,_,H,_ = x.shape
            W = H
            y = self.main_fft([x.real, x.imag])                # [real part, imaginary part]
        
        if self.skip:
            y = [y[0]+x.real, y[1]+x.imag]
        
        if self.out_spatial:
            y = torch.complex(y[0],y[1])
            y = torch.fft.irfft2(y, s=(H,W), norm=self.norm)   # spatial feature
            return y
        else:
            return y
    
        


# ===============================
# Layer Normalization 2d
# ===============================
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)



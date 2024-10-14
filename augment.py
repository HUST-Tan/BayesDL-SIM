# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 00:22:59 2022

@author: Administrator
"""

import numpy as np
import torch
import torch.nn.functional as F

def apply_augment(
    im1, im2,       # im1 is HR, im2 is LR in SR
    augs=['none','mixup','cutmix','cutmixup', 'cutout'],
#    probs=[1.0,1.0,1.0],    # 
#    alphas=[1.0,1.2,0.7,],
#    aux_prob=None, 
#    aux_alpha=None,
    mix_p=None      # 执行每种DA的概率
):
    idx = np.random.choice(len(augs), p=mix_p)
    aug = augs[idx]
#    prob = float(probs[idx])
#    alpha = float(alphas[idx])
    mask = None

    if aug == "none":
        im1_aug, im2_aug = im1.clone(), im2.clone()
    elif aug == "blend":
        im1_aug, im2_aug = blend(
            im1.clone(), im2.clone(),
            prob=1.0, alpha=0.6
        )
    elif aug == "mixup":
        im1_aug, im2_aug, = mixup(
            im1.clone(), im2.clone(),
            prob=1.0, alpha=1.2,
        )
    elif aug == "cutout":
        im1_aug, im2_aug, mask, _ = cutout(
            im1.clone(), im2.clone(),
            prob=1.0, alpha=0.001
        )
    elif aug == "cutmix":
        im1_aug, im2_aug = cutmix(
            im1.clone(), im2.clone(),
            prob=1.0, alpha=0.7,
        )
    elif aug == "cutmixup":
        im1_aug, im2_aug = cutmixup(
            im1.clone(), im2.clone(),
            mixup_prob=1.0, mixup_alpha=1.2,
            cutmix_prob=1.0, cutmix_alpha=0.7,
        )
    elif aug == "cutblur":
        im1_aug, im2_aug = cutblur(
            im1.clone(), im2.clone(),
            prob=1.0, alpha=0.7
        )
    elif aug == "rgb":
        im1_aug, im2_aug = rgb(
            im1.clone(), im2.clone(),
            prob=1.0
        )
    else:
        raise ValueError("{} is not invalid.".format(aug))

    return im1_aug, im2_aug, mask, aug


def blend(im1, im2, prob=1.0, alpha=0.6):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    c = torch.empty((im2.size(0), 3, 1, 1), device=im2.device).uniform_(0, 255)
    rim2 = c.repeat((1, 1, im2.size(2), im2.size(3)))
    rim1 = c.repeat((1, 1, im1.size(2), im1.size(3)))

    v = np.random.uniform(alpha, 1)
    im1 = v * im1 + (1-v) * rim1
    im2 = v * im2 + (1-v) * rim2

    return im1, im2


def mixup(im1, im2, prob=1.0, alpha=1.2):   # prob 执行mixup的概率；alpha控制线性组合系数
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1-v) * im1[r_index, :]
    im2 = v * im2 + (1-v) * im2[r_index, :]
    return im1, im2


def _cutmix(im2, prob=1.0, alpha=0.7):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int_(h*cut_ratio), np.int_(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmix(im1, im2, prob=1.0, alpha=0.7):
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return im1, im2


def cutmixup(
    im1, im2,
    mixup_prob=1.0, mixup_alpha=1.2,
    cutmix_prob=1.0, cutmix_alpha=0.7
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im2_aug = v * im2 + (1-v) * im2[rindex, :]
        im1_aug = v * im1 + (1-v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
        im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2


def cutblur(im1, im2, prob=1.0, alpha=1.0):
    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    return im1, im2


def cutout(im1, im2, prob=1.0, alpha=0.001):
    scale = im1.size(2) // im2.size(2)
    fsize = (im2.size(0), 1)+im2.size()[2:]

    if alpha <= 0 or np.random.rand(1) >= prob:
        fim2 = np.ones(fsize)
        fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
        fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")
        return im1, im2, fim1, fim2

    fim2 = np.random.choice([0.0, 1.0], size=fsize, p=[alpha, 1-alpha])
    fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
    fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")

    im2 *= fim2

    return im1, im2, fim1, fim2


def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(3)
    im1 = im1[:, perm]
    im2 = im2[:, perm]

    return im1, im2
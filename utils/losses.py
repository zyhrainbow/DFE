
import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb
import math
import numpy as np
from typing import Tuple
import cv2
import os
from torch.autograd import Variable
from math import exp


def l1(fake, expert, weight=1):
    return (fake - expert).abs().mean()*weight

def l2(fake, expert, weight=1):
    return (fake - expert).pow(2).mean()*weight

def psnr(fake, expert):
    # pdb.set_trace()
    mse = (fake - expert).pow(2).mean()
    if mse.pow(2) == 0:
        mse += 1e-6
    if torch.max(expert) > 2:
        max_ = 255.
    else:
        max_ = 1.
    return 10 * torch.log10(max_**2 / (mse)) 

def cosine(fake, expert, weight=1):
    return (1 - torch.nn.functional.cosine_similarity(fake, expert, 1)).mean()*weight

def TV(x,y,weight=1):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def ssim(image1, image2,weight=1):
    window_size =11
    K=[0.01,0.03]
    L=255
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5  # default
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    # define constants
    # * L = 255 for constants doesn't produce meaningful results; thus L = 1
    # C1 = (K[0]*L)**2;
    # C2 = (K[1]*L)**2;
    C1 = K[0] ** 2
    C2 = K[1] ** 2

    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()



def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def get_file_paths(folder,suffix):
    file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.split('.')[-1] == suffix:
                file_paths.append(file_path)
        break
    return file_paths

def get_file_name(fp):
    return fp.split('/')[-1].split('.')[0]


import os
import pdb
import math
#import trilinear
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import time
from torch import nn, einsum
from einops import rearrange, repeat
import tqdm
import torch

from inspect import isfunction

class CLUTNet(nn.Module):
    def __init__(self, nsw=3, dim=33, *args, **kwargs):
        super(CLUTNet, self).__init__()
        self.backbone = BackBone()
        self.global_enhance =  CSRNet_global()
        self.local_enhance = CSRNet_local()
        self.fusion = AsyCA()

    def forward(self, img, *args, **kwargs):
        feature_global, feature_local = self.backbone(img)
        map_global = self.global_enhance(img,feature_global.reshape(-1,1,64))#.clamp(0,1)
        map_local = self.local_enhance(img,feature_local)#.clamp(0,1)
        gen = self.fusion(map_local,map_global).clamp(0,1)

        return gen

class CSRNet_local(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, base_nf=64, cond_nf=32):
        super(CSRNet_local, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc
        self.encoder = nn.Sequential(nn.Conv2d(in_nc , base_nf, 1, 1),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(base_nf, base_nf, 1, 1),
                                     nn.LeakyReLU(inplace = True),
                                     nn.Conv2d(base_nf, base_nf, 1, 1),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(base_nf, 3, 1, 1)
                                     )

    def forward(self, x,val):
        x_code = self.encoder(torch.cat((x, val), dim=1))
        return x_code

class CSRNet_global(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_nf=64, cond_nf=32):
        super(CSRNet_global, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1)
        self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1)
        self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
        self.act = nn.LeakyReLU(inplace=True)
        self.scale = nn.Conv2d(8, base_nf, 1)
        self.shift = nn.Conv2d(8, base_nf, 1)
        self.attention = CrossAttention()



    def forward(self, x ,context):
        x_code = self.encoder(x)
        std, mean = torch.std_mean(x_code, dim=[2, 3], keepdim=False)
        coeffs = torch.cat([std, mean], dim=1).unsqueeze(1)
        coeffs = self.attention(coeffs, context).permute(0, 2, 1).unsqueeze(3)
        y_code = coeffs[:,0:64]*x_code + coeffs[:,64:128]
        y_code = self.act(self.mid_conv(y_code))
        y = self.decoder(y_code)
        return y

class CrossAttention(nn.Module):
    def __init__(self, query_dim=128, context_dim=64, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)


        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)


        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))


        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale


        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)



class BackBone(nn.Module):
    def __init__(self, last_channel=64, ):  # org both
        super(BackBone, self).__init__()
        self.resize = nn.Upsample(size=(256, 256), mode='bilinear',align_corners=True)
        ls1 = [
            *discriminator_block(3, 16, normalization=True),  # 128**16
            *discriminator_block(16, 32, normalization=True),  # 64**32
            *discriminator_block(32, 64, normalization=True),  # 32**64
            *discriminator_block(64, 128, normalization=True), ]  # 16**128
        ls2 = [
            *discriminator_block(128, last_channel, normalization=False),  # 8**128
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(1),

        ]

        ls3 = [
            *up_block(128,64),# 32**64
            *up_block(64,32), # 64 **32
            *up_block(32,16), # 128**16
            *up_block(16,1), # 256**8
        ]


        self.model1 = nn.Sequential(*ls1)
        self.model2 = nn.Sequential(*ls2)
        self.model3 = nn.Sequential(*ls3)

    def forward(self, x):

        x1 = self.model1(x)
        x2 = self.model2(x1)
        x3 = self.model3(x1)
        x4 = F.interpolate(x3,size=(x.shape[2:]),mode='bilinear',align_corners=True)

        return x2,x4

def discriminator_block(in_filters, out_filters, kernel_size=3, sp="2_1", normalization=False):
    stride = int(sp.split("_")[0])
    padding = int(sp.split("_")[1])

    layers = [
        nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers
def up_block(in_filters, out_filters, kernel_size=3, sp="1_1", normalization=False):
    stride = int(sp.split("_")[0])
    padding = int(sp.split("_")[1])

    layers = [
        nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers


class Condition(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super(Condition, self).__init__()
        stride = 1
        pad = 1
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, 3, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.InstanceNorm2d(nf,affine=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(x))
        conv2_out = self.act(self.conv2(conv1_out))
        conv3_out = self.act(self.conv3(conv2_out))
        out = self.norm(conv3_out)

        return out


    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None

class AsyCA(nn.Module):

    def __init__(self, num_features=3, ratio=4):
        super(AsyCA, self).__init__()
        self.out_channels = num_features
        self.conv_init = nn.Conv2d(num_features * 2, num_features, kernel_size=1, padding=0, stride=1)
        self.conv_dc = nn.Conv2d(num_features, num_features*2, kernel_size=1, padding=0, stride=1)
        #self.conv_ic = nn.Conv2d(num_features * ratio, num_features * 2, kernel_size=1, padding=0, stride=1)
        self.act = nn.ReLU(inplace=True)
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        batch_size,c,h,w = x1.size()

        feat_init = torch.cat((x1, x2), 1)
        feat_init = self.conv_init(feat_init)
        #feat_avg = self.avg_pool(feat_init)
        feat_ca = self.conv_dc(self.act(feat_init))
        #feat_ca = self.conv_ic(self.act(feat_ca))


        a_b = feat_ca.reshape(batch_size, 2, self.out_channels, -1)

        a_b = self.softmax(a_b)
        # print(a_b[0,0,0,0],)
        
        a_b = list(a_b.chunk(2, dim=1))  # split to a and b
        a_b = list(map(lambda x1: x1.reshape(batch_size, self.out_channels, h, w), a_b))

        V1 = a_b[0] * x1
        V2 = a_b[1] * x2
        V = V1 + V2
        #V = a_b * feat_init
        return V

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def enhance(conn_in, conn_out, device, frame_number,w, h, index):
    if device == 'cpu':
        cpu = True
    else:
        cpu = False
        gpu = 'cuda:%d' % int(device[-1])
        device_ids = [int(device[-1])]
        # print(device_ids)
    torch.set_grad_enabled(False)
    device = torch.device('cpu' if cpu else gpu)
    # print(device)
    PATH = "./model0397pth"
    model = CLUTNet()
    loadnet = torch.load(PATH)
   
    model.load_state_dict(loadnet, strict=True)
    model.eval()
    model.to(device)

    for i in tqdm.trange(frame_number, desc='Enhancement',index ):
        tmp_data = conn_in.recv()
        rgb_data = np.frombuffer(tmp_data, dtype=np.uint8).reshape(h, w, 3)
        # print(f_cnt)
        lr_np = np.ascontiguousarray(rgb_data.transpose((2, 0, 1)))
        lr_np = np.expand_dims(lr_np, axis=0)
        # print(lr_np.shape)

        lr_pytorch = torch.from_numpy(lr_np).float()
        lr_pytorch.mul_(1 / 255)
        lr_pytorch.to(device)

        sr_pytorch = model(lr_pytorch.unsqueeze(0))
        sr_np = sr_pytorch.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
        conn_out.send(sr_np.tobytes())
        
    


             
   

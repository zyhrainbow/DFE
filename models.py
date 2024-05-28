import os
import pdb
import math
#import trilinear
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.LUT import *
import time
from torch import nn, einsum
from einops import rearrange, repeat

from inspect import isfunction

class DFE(nn.Module):
    def __init__(self, nsw=3, dim=33, *args, **kwargs):
        super(DFE, self).__init__()
        self.backbone = BackBone()
        self.global_enhance =  CSRNet_global()
        self.local_enhance = CSRNet_local()
        self.fusion = AsyCA()


    def forward(self, img, *args, **kwargs):
        feature_global, feature_local = self.backbone(img)

        map_global = self.global_enhance(img,feature_global.reshape(-1,1,64))#.clamp(0,1)

        map_local = self.local_enhance(img,feature_local)#.clamp(0,1)

        gen = self.fusion(map_local,map_global).clamp(0,1)





        return gen, {
            "map":feature_local,
            "map_global": map_global,
            "map_local":map_local
        }


class CSRNet_local(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, base_nf=64, cond_nf=32):
        super(CSRNet_local, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc
        self.encoder = nn.Sequential(nn.Conv2d(in_nc , base_nf, 1, 1),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(base_nf, base_nf, 1, 1),
                                     nn.LeakyReLU(inplace = True),
                                     # nn.Conv2d(base_nf, base_nf, 1, 1),
                                     # nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(base_nf, base_nf, 1, 1),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(base_nf, 3, 1, 1)
                                     )

        # self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1)
        # self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
        # self.act = nn.LeakyReLU(inplace=True)
        # self.scale = nn.Conv2d(8, base_nf, 1)
        # self.shift = nn.Conv2d(8, base_nf, 1)


    def forward(self, x,val):
        x_code = self.encoder(torch.cat((x, val), dim=1))
        #x_code = self.encoder(x_code)
        # scale = self.scale(val)
        # shift = self.shift(val)
        # y_code = x_code * (scale +1) + shift
        # y_code = self.act(self.mid_conv(y_code))
        # y = self.decoder(y_code)
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
        # coeffs = torch.mean(x_code, dim=[2, 3], keepdim=False).unsqueeze(1)
        # coeffs_scale = self.attention(coeffs,context).permute(0,2,1).unsqueeze(3)

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

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 64 // factor)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.pre = nn.Conv2d(8, 1, 3, 1, 1)
        self.re = nn.Sigmoid()

    def forward(self, xs):
        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down4(x3)
        print(x4.size())
        print(x3.size())
        print(x2.size())
        print(x1.size())

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.re(self.pre(x))
        return x

class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=bn)
        self.conv2 = ConvBlock(16, 2, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

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

class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, 10, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 10, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class query_SABlock(nn.Module):
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
if __name__ == "__main__":
    import numpy as np
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    img = torch.Tensor(1, 3,1920,1080).cuda()
    img_ = torch.Tensor(8, 3, 400, 500).cuda()

    net = CLUTNet().cuda()

    print('total parameters:', sum(param.numel() for param in net.parameters()))
    # t_list = []
    # for i in range(1, 10):
    #
    #     torch.cuda.synchronize()
    #     t0 = time.time()
    #     for j in range(0, 100):
    #         fake_B= net(img)
    #
    #     torch.cuda.synchronize()
    #     t1 = time.time()
    #     t_list.append(t1 - t0)
    #     print((t1 - t0))
    # print(t_list)
    # torch.cuda.synchronize()
    # pre = time.time()
    # x1= net(img)
    # torch.cuda.synchronize()
    # stage = time.time()-pre
    # print(x1.shape)
    # print(stage)
    # print(x2.shape)
    # print(x3.shape)
    # print(x4.shape)

import os
from os.path import join 
import sys
import numpy as np
import pdb
import torch.nn as nn
from torchvision import utils
from torchvision.utils import save_image
from torchvision.utils import make_grid

from utils.losses import *
from parameter import *
from setting import *
import time

def evaluate(setting, epoch=None, best_psnr=None, do_save_img=True):
    eval_dataloader = setting.eval_dataloader
    opt = setting.opt
    if epoch is not None:
        epoch = "{:0>4}".format(epoch)
        dst = join(opt.save_images_root, epoch) 
    else:
        dst = opt.save_images_root

    os.makedirs(dst, exist_ok=True)
    os.makedirs(join(dst, 'others'), exist_ok=True)
    os.makedirs(join(dst, 'maps'), exist_ok=True)
    psnr_ls = []
    weight_ls = []
    psnr_in, psnr_out, avg_psnr_in, avg_psnr_out = 0, 0, 0, 0
    #ssim_in, ssim_out, avg_ssim_in, avg_ssim_out = 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            targets = batch["target"].type(Tensor)
            imgs = batch["input"].type(Tensor)
            psnr_in = batch.get("psnr_in")
            # ssim_in = batch.get("ssim_in")
            if psnr_in is not None:
                psnr_in = psnr_in.squeeze()
            # if ssim_in is not None:
            #     ssim_in = ssim_in.squeeze()
            fakes, others = setting.evaluate(batch)
            name = os.path.splitext(batch["name"][0])[0]
            if epoch is None:
                sys.stdout.write("\r"+name)
            ########################################## log
            psnr_out = psnr(fakes, targets).item()
            # ssim_out = ssim(fakes, targets).item()
            if psnr_in is None:
                psnr_in = psnr(imgs, targets).item()
            # if ssim_in is None:
            #     ssim_in = psnr(imgs, targets).item()
            change_str = "{:.4f}--{:.4f}".format(psnr_in, psnr_out)
            # change_str_ssim = "{:.4f}--{:.4f}".format(ssim_in, ssim_out)
            avg_psnr_in += psnr_in
            avg_psnr_out += psnr_out
            # avg_ssim_in += ssim_in
            # avg_ssim_out += ssim_out
            #coeffs = [others.get("coeffs").squeeze().data]

            img_ls = [imgs.squeeze().data, fakes.squeeze().data, targets.squeeze().data,others.get("map_global").squeeze().data,
                         others.get("map_local").squeeze().data]
            maps = others.get("map").squeeze().data
            if do_save_img:
                save_image(img_ls, join(dst, "%s %s.jpg" % (name, change_str)), nrow=len(img_ls))
                #save_image(others_ls, join(dst, "other%s %s.jpg" % (name, change_str)), nrow=len(others_ls))
                save_image(fakes.squeeze().data, join(dst, 'others', "%s.jpg" % (name)))
                save_image(maps, join(dst, 'maps', "%s.jpg" % (name)))
       
    isbest = ""
    avg_psnr_in /= len(eval_dataloader)
    avg_psnr_out /= len(eval_dataloader)
    #avg_ssim /= len(eval_dataloader)
    if epoch is not None:
        if avg_psnr_out > best_psnr:
            isbest = "_best"
    change_str = "_%.4f--%.4f" % (avg_psnr_in, avg_psnr_out)
    os.rename(dst, dst + change_str + isbest) 
    torch.cuda.empty_cache()

    return avg_psnr_out

if __name__ == "__main__":
    opt = parser.parse_args()
    setting = Setting(opt, "test")
    print("\n{:.4f}".format(evaluate(setting)))
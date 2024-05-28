import os
from os.path import join
import numpy as np
from thop import profile
from models import *
from parameter import cuda, Tensor, device
from torch.utils.data import DataLoader
from datasets import *


class Setting():
    
    def __init__(self, opt, mode="train"):

        self.opt = opt
        opt.losses = opt.losses.split(" ")

        self.model = DFE(opt.model, dim=opt.dim)
        self.model = self.model.to(device)
        if opt.name is None:
            opt.output_dir = join(opt.save_root, opt.dataset, opt.model)
        else:
            opt.output_dir = join(opt.save_root, opt.dataset, opt.name)
        print("save checkpoints to %s" % opt.output_dir)

        opt.save_models_root = opt.output_dir + "_models"
        self.eval_dataloader = DataLoader(
            eval(opt.dataset)(opt.data_root, mode="test"),
            batch_size=1,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        if mode == "train": # python train.py
            os.makedirs(opt.save_models_root, exist_ok=True)
            opt.save_images_root = opt.output_dir +"_images"
            os.makedirs(opt.save_images_root, exist_ok=True)
            opt.save_logs_root = opt.output_dir
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=opt.lr,betas=(0.9, 0.999), eps=1e-8,weight_decay=opt.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.n_epochs)
            self.train_dataloader = DataLoader(
                eval(opt.dataset)(opt.data_root, mode="train"),
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_cpu,
            )
            ###################################### load ckpt
            if opt.epoch > 1:
                self.optimizer.load_state_dict(torch.load(join(opt.save_models_root, "optimizer_latest.pth")))
                if os.path.exists(join(opt.save_models_root, "model{:0>4}.pth".format(opt.epoch-1))):
                    self.model.load_state_dict(torch.load(join(opt.save_models_root, "model{:0>4}.pth".format(opt.epoch-1))), strict=True)
                    print("ckp loaded from epoch " + str(opt.epoch-1))
                else:
                    self.model.load_state_dict(torch.load(join(opt.save_models_root, "model_latest.pth")), strict=True)
                    print("ckp loaded from the latest epoch")
            else:

                pretrained_dict = torch.load("/media/ps/HDD/zyh/v1/FiveK_zyh_best_models/FiveK_zyh_best_models/model0397.pth")
                   #  "/media/ps/data2/zyh/enhance/zyh_lut_copy/L-pretrain/FiveK_pretrain/03_models/model0070.pth")
                   # "/media/ps/data2/zyh/enhance/zyh_lut_copy/pretrain/FiveK_pretrain/03_models/model0014.pth")#four
                   # "/media/ps/data2/zyh/enhance/zyh_lut_copy/hdrnet_channel_split/FiveK_pretrain/03_models/model0193.pth")
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict  ,strict= True)
            #
        else: # python eval.py
            self.epoch = opt.epoch
            opt.save_images_root = opt.output_dir +"_"+ str(self.epoch)
            if opt.epoch > 1:
                load = torch.load(join(opt.save_models_root, "model{:0>4}.pth".format(opt.epoch)))
                self.model.load_state_dict(load, strict=True)
                print("model loaded from epoch "+str(opt.epoch))
        
        #self.TVMN = TVMN(opt.dim).to(device)
        os.makedirs(opt.save_images_root, exist_ok=True)
        
             
    def train(self, batch):
        self.model.train()
        imgs = batch["input"].type(Tensor)
        experts = batch["target"].type(Tensor)
        # flops, params = profile(self.model, inputs = (imgs, imgs, self.TVMN))
        fakes, others = self.model(imgs)

        
        return fakes, others

    def evaluate(self, batch):
        self.model.eval()
        img = batch["input"].type(Tensor)
        #img_org = batch.get("input_org").type(Tensor)
        fake, others = self.model(img)

        return fake, others

    def save_ckp(self, epoch=None, save_opt=True):
        if epoch is not None:
            torch.save(self.model.state_dict(), "{}/model{:0>4}.pth".format(self.opt.save_models_root, epoch))
        else:
            torch.save(self.model.state_dict(), "{}/model_latest.pth".format(self.opt.save_models_root))
        if save_opt:
            torch.save(self.optimizer.state_dict(), "{}/optimizer_latest.pth".format(self.opt.save_models_root))
   

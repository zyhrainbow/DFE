import os
from os.path import join 
import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.LUT import *
from utils.losses import *
import imageio
import glob
from PIL import Image

def augment(img_input, img_target):
    H,W = img_input.shape[1:]
    crop_h = round(H * np.random.uniform(0.6,1.))
    crop_w = round(W * np.random.uniform(0.6,1.))
    b = np.random.uniform(0.8,1.2)
    s = np.random.uniform(0.8,1.2)
    #img_input = TF.adjust_brightness(img_input,b)
    #img_input = TF.adjust_saturation(img_input,s)
    i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
    img_input = TF.resized_crop(img_input, i, j, h, w, (256, 256))
    img_target = TF.resized_crop(img_target, i, j, h, w, (256, 256))
    if np.random.random() > 0.5:
        img_input = TF.hflip(img_input)
        img_target = TF.hflip(img_target)
    if np.random.random() > 0.5:
        img_input = TF.vflip(img_input)
        img_target = TF.vflip(img_target)
    return img_input, img_target


class FiveK(Dataset):
    def __init__(self, data_root, mode): 
        
        self.mode = mode
        input_dir = join(data_root, "fiveK/input_"+mode)
        target_dir = join(data_root, "fiveK/target_"+mode)
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))
        self.input_files = [join(input_dir, file_name) for file_name in input_files]
        self.target_files = [join(target_dir, file_name) for file_name in target_files]


    def __getitem__(self, index):
        res = {}
        input_path = self.input_files[index]
        img_input = TF.to_tensor(cv2.cvtColor(cv2.imread(input_path, -1), cv2.COLOR_BGR2RGB)/255)
        target_path = self.target_files[index]
        img_target = TF.to_tensor(cv2.cvtColor(cv2.imread(target_path, -1), cv2.COLOR_BGR2RGB)/255) 
        img_name = os.path.split(self.input_files[index])[-1]
        res["name"] = img_name

        if self.mode == "train":
            img_input, img_target = augment(img_input, img_target) 
            res["input"] = img_input
            res["input_org"] = img_input
            res["target"] = img_target
            res["target_org"] = img_target

        else:
            img_input_resize, img_target_resize = TF.resize(img_input, (256, 256)), TF.resize(img_target, (256, 256))
            res["input_org"] = img_input
            res["target_org"] = img_target
            res["input"] = img_input_resize
            res["target"] = img_target_resize
        return res 


    def __len__(self):
        return len(self.input_files)
class FiveK_zyh(Dataset):
    def __init__(self, root, mode,combined=True):

        self.mode = mode

        file = open(os.path.join(root, 'train_input.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        #self.set1_mask_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root, "input", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))
            self.set1_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))
            #self.set1_mask_files.append(os.path.join("/media/ps/data2/zyh/datasets/fiveK/mask", set1_input_files[i][:-1] + ".png"))

        file = open(os.path.join(root, 'train_label.txt'), 'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        #self.set2_mask_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root, "input", "JPG/480p", set2_input_files[i][:-1] + ".jpg"))
            self.set2_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set2_input_files[i][:-1] + ".jpg"))
            #self.set2_mask_files.append(os.path.join("/media/ps/data2/zyh/datasets/fiveK/mask", set1_input_files[i][:-1] + ".png"))

        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        #self.test_mask_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root, "input", "JPG/480p", test_input_files[i][:-1] + ".jpg"))
            self.test_expert_files.append(os.path.join(root, "expertC", "JPG/480p", test_input_files[i][:-1] + ".jpg"))
            #self.test_mask_files.append(os.path.join("/media/ps/data2/zyh/datasets/fiveK/mask", test_input_files[i][:-1] + ".png"))

        if combined:
            self.set1_input_files = self.set1_input_files + self.set2_input_files
            self.set1_expert_files = self.set1_expert_files + self.set2_expert_files

    def __getitem__(self, index):
        res = {}



        if self.mode == "train":
            input_path = self.set1_input_files[index]
            img_input = TF.to_tensor(cv2.cvtColor(cv2.imread(input_path, -1), cv2.COLOR_BGR2RGB) / 255)
            target_path = self.set1_expert_files[index]
            img_target = TF.to_tensor(cv2.cvtColor(cv2.imread(target_path, -1), cv2.COLOR_BGR2RGB) / 255)
            #mask_path = self.set1_mask_files[index]
            #img_mask = TF.to_tensor(cv2.cvtColor(cv2.imread(mask_path, -1), cv2.COLOR_BGR2RGB) / 255)
            img_name = os.path.split(self.set1_input_files[index])[-1]
            res["name"] = img_name
            img_input, img_target = augment(img_input, img_target)
            res["input"] = img_input
            #res["input_org"] = img_input
            res["target"] = img_target
            #res["target_org"] = img_target
            # res["mask"] = img_mask
            # res["mask_org"] = img_mask
        
        else:
            input_path = self.test_input_files[index]
            img_input = TF.to_tensor(cv2.cvtColor(cv2.imread(input_path, -1), cv2.COLOR_BGR2RGB) / 255)
            target_path = self.test_expert_files[index]
            img_target = TF.to_tensor(cv2.cvtColor(cv2.imread(target_path, -1), cv2.COLOR_BGR2RGB) / 255)
            img_name = os.path.split(self.test_input_files[index])[-1]
            #mask_path = self.test_mask_files[index]
            #img_mask = TF.to_tensor(cv2.cvtColor(cv2.imread(mask_path, -1), cv2.COLOR_BGR2RGB) / 255)
            res["name"] = img_name
            #img_input_resize, img_target_resize= TF.resize(img_input, (256, 256)), TF.resize(img_target, (256, 256))
            res["input"] = img_input
            res["target"] = img_target
            #res["mask_org"] = img_mask
            # res["input"] = img_input_resize
            # res["target"] = img_target_resize
            #res["mask"] = img_mask_resize
        return res

    def __len__(self):
        if self.mode == 'train':
            
            return len(self.set1_input_files)
        else:
            return len(self.test_input_files)

# Implement your own DatasetClass according to your data format and dir arrangement.

class FiveK_lite(Dataset):
    def __init__(self, dataroot, mode):
        super(FiveK_lite, self).__init__()

        self.phase = mode
        self.dataroot = "/media/ps/data2/zyh/datasets/fiveK/dataset-lite"

        self.suffix_A = "jpg"
        self.suffix_B = "jpg"


        if self.phase == "train":
            self.path_A = get_file_paths(os.path.join(self.dataroot, "trainA"), self.suffix_A)
            self.path_B = get_file_paths(os.path.join(self.dataroot, "trainB"), self.suffix_B)
        else:
            self.path_A = get_file_paths(os.path.join(self.dataroot, "testA"), self.suffix_A)
            self.path_B = get_file_paths(os.path.join(self.dataroot, "testB"), self.suffix_B)

        assert (len(self.path_A) > 0)
        assert (len(self.path_A) == len(self.path_B))

    def __getitem__(self, index):
        res={}
        path_A, path_B = self.path_A[index], self.path_B[index]
        idx = get_file_name(path_A)

        img_A = TF.to_tensor(cv2.cvtColor(cv2.imread(path_A, -1), cv2.COLOR_BGR2RGB) / 255)
        img_B = TF.to_tensor(cv2.cvtColor(cv2.imread(path_B, -1), cv2.COLOR_BGR2RGB) / 255)


        if self.phase == "train":
            img_A, img_B = augment(img_A, img_B)

       
        res["name"] = idx
        res["input"] = img_A
        res["target"] = img_B
        return res

    def __len__(self):
        return len(self.path_A)

class PPR10K(Dataset):
    def __init__(self, root, mode="train"):
        super(PPR10K, self).__init__()
        self.mode = mode
        self.root = "/media/ps/data2/zyh/datasets/PPR10K/"

        self.retoucher = 'a'
        print('training with target_' + self.retoucher)
        self.train_input_files = sorted(glob.glob(os.path.join(self.root, "train/raw" + "/*.tif")))
        self.train_target_files = sorted(glob.glob(os.path.join(self.root, "train/target_" + self.retoucher + "/*.tif")))


        self.test_input_files = sorted(glob.glob(os.path.join(self.root, "test/raw" + "/*.tif")))
        #print(os.path.join(root, "test/raw" + "/*.tif"))
        #print(glob.glob(os.path.join(root, "test/raw" + "/*.tif")))
        self.test_target_files = sorted(glob.glob(os.path.join(self.root, "test/target_" + self.retoucher + "/*.tif")))


    def __getitem__(self, index):
        res={}
        if self.mode == "train":
            img_name = os.path.split(self.train_input_files[index % len(self.train_input_files)])[-1]
            #print(img_name)
            #img_input = cv2.imread(self.train_input_files[index % len(self.train_input_files)],-1)
            # if len(self.train_input_files) == len(self.train_target_files):
            img_exptC = Image.open(self.train_target_files[index % len(self.train_target_files)])
            img_input = Image.open(self.train_input_files[index % len(self.train_input_files)])
            # else:
            #     split_name = img_name.split('_')
            #     if len(split_name) == 2:
            #         img_exptC = Image.open(os.path.join(self.root, "train/target_" + self.retoucher + '/' + img_name))
            #     else:
            #         img_exptC = Image.open(
            #             os.path.join(self.root, "train/target_" + self.retoucher + '/' + split_name[0] + "_" + split_name[1] + ".tif"))

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            #img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_target_files[index % len(self.test_target_files)])
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])


        W,H = img_exptC._size
        img_input = TF.resize(img_input,(H,W))

        #img_input = img_input[:, :, [2, 1, 0]]

        if self.mode == "train":

            ratio_H = np.random.uniform(0.6, 1.0)
            ratio_W = np.random.uniform(0.6, 1.0)
            W,H = img_exptC._size
            crop_h = round(H * ratio_H)
            crop_w = round(W * ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF.resized_crop(img_input, i, j, h, w, (448, 448))
            img_exptC = TF.resized_crop(img_exptC, i, j, h, w, (448, 448))


            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)


        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        res["name"] = img_name
        res["input"] = img_input
        res["target"] = img_exptC

        return res

    def __len__(self):
        if self.mode == "train":
            return len(self.train_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class FiveK_dark(Dataset):
    def __init__(self, root, mode="train"):

        self.mode = mode
        self.root = "/media/ps/data2/zyh/datasets/fiveK/dataset-dark"

        self.train_input_files = sorted(glob.glob(os.path.join(self.root, "trainA" + "/*.tif")))
        self.train_target_files = sorted(glob.glob(os.path.join(self.root, "trainB" + "/*.jpg")))

        self.test_input_files = sorted(glob.glob(os.path.join(self.root, "testA" + "/*.tif")))
        # print(os.path.join(root, "test/raw" + "/*.tif"))
        # print(glob.glob(os.path.join(root, "test/raw" + "/*.tif")))
        self.test_target_files = sorted(glob.glob(os.path.join(self.root, "testB" + "/*.jpg")))

    def __getitem__(self, index):
        res = {}
        if self.mode == "train":
            img_name = os.path.split(self.train_input_files[index % len(self.train_input_files)])[-1]
            # print(img_name)
            # img_input = cv2.imread(self.train_input_files[index % len(self.train_input_files)],-1)
            # if len(self.train_input_files) == len(self.train_target_files):
            img_exptC = Image.open(self.train_target_files[index % len(self.train_target_files)])
            img_input = Image.open(self.train_input_files[index % len(self.train_input_files)])
            # else:
            #     split_name = img_name.split('_')
            #     if len(split_name) == 2:
            #         img_exptC = Image.open(os.path.join(self.root, "train/target_" + self.retoucher + '/' + img_name))
            #     else:
            #         img_exptC = Image.open(
            #             os.path.join(self.root, "train/target_" + self.retoucher + '/' + split_name[0] + "_" + split_name[1] + ".tif"))

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            # img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_target_files[index % len(self.test_target_files)])
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])

        W, H = img_exptC._size
        img_input = TF.resize(img_input, (H, W))

        # img_input = img_input[:, :, [2, 1, 0]]

        if self.mode == "train":

            ratio_H = np.random.uniform(0.6, 1.0)
            ratio_W = np.random.uniform(0.6, 1.0)
            W, H = img_exptC._size
            crop_h = round(H * ratio_H)
            crop_w = round(W * ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF.resized_crop(img_input, i, j, h, w, (448, 448))
            img_exptC = TF.resized_crop(img_exptC, i, j, h, w, (448, 448))

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        res["name"] = img_name
        res["input"] = img_input
        res["target"] = img_exptC

        return res

    def __len__(self):
        if self.mode == "train":
            return len(self.train_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from parameter import cuda, Tensor, device
    from PIL import Image

    root = "/media/ps/data2/zyh/datasets/fiveK"

    dataset = PPR10K(root=root,mode="test")

    for i, data in enumerate(dataset):
        input = data['input']
        target = data['target']

        input = input.numpy()
        target = target.numpy()



        img = np.concatenate([input,target], axis=2)
        img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0

        image = img.astype(np.uint8)
        image = Image.fromarray(image)
        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.title('Image{}'.format(i + 1))
        plt.show()

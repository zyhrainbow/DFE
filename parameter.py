import argparse
import torch
import numpy as np
import os
import pdb

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--n_cpu", type=int, default=16, help="for dataloader")
parser.add_argument("--optm", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument("--lambda_smooth", type=float, default=0.0001, help="smooth regularization strength")
parser.add_argument("--lambda_mn", type=float, default=10.0, help="monotonicity regularization strength")

# epoch for train:  =1 starts from scratch, >1 load saved checkpoint of <epoch-1>
# epoch for eval:   load the model of <epoch> and evaluate
parser.add_argument("--epoch", type=int, default=1)

parser.add_argument("--n_epochs", type=int, default=400, help="last epoch of training (include)")
parser.add_argument("--dim", type=int, default=33, help="dimension of 3DLUT")
parser.add_argument("--losses", type=str, default="1*l1 0.5*cosine 0.1*TV", help="one or more loss functions (splited by space)")
parser.add_argument("--model", type=str, default="03+-1+-1", help="model configuration, n+s+w")
parser.add_argument("--name", type=str, help="name for this training (if None, use <model> instead)")

parser.add_argument("--save_root", type=str, default="./v1/model", help="root path to save images/models/logs")
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--data_root", type=str, default="/media/ps/data2/zyh/datasets/fiveK/", help="root path of data")

# Dataset Class should be implemented first for different dataset format")
parser.add_argument("--dataset", type=str, default="FiveK_zyh", help="which dateset to use")



cuda =  torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = "cuda"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import models.transformer as transformer
import models.StyTR as StyTR
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from dataset.mydataset import KidneyDataset, KideneyContentDataset, KidneyStyleDataset
import numpy as np
from torch.autograd import Variable


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
'''
ying*************************************************************************************************
change the parset options of code
'''
parser.add_argument('--data_path', default="D:/zy/project/virtual_staining/data/kidney_trans/", type=str,
                    help='Directory path to a batch of content & style images')

# Basic options
# parser.add_argument('--content_dir', default='./datasets/train2014', type=str,
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style_dir', default='./datasets/Images', type=str,  #wikiart dataset crawled from https://www.wikiart.org/
#                     help='Directory path to a batch of style images')

parser.add_argument('--content_dir', default='D:/zy/project/virtual_staining/data/kidney_trans/B/train', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='D:/zy/project/virtual_staining/data/kidney_trans/A/train', type=str,  # wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')

# run the train.py, please download the pretrained vgg checkpoint
parser.add_argument('--vgg', type=str,
                    default='./experiments/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=1600)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()

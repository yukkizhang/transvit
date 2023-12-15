import skimage.io as io
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import util.box_ops as box_ops
# from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from function import normal, normal_style
from function import calc_mean_std
import scipy.stats as stats
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        return x


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# to do --------------------------
# study this class and write the transformer generator by self
class StyTrans(nn.Module):
    """ This is the style transform transformer module """

    def __init__(self, encoder, decoder, PatchEmbed, transformer, args):

        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.decode = decoder
        self.embedding = PatchEmbed

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1] / 1.0))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """
        content_input = samples_c
        style_input = samples_s
        if isinstance(samples_c, (list, torch.Tensor)):
            # support different-sized images padding is used for mask [tensor, mask]
            samples_c = nested_tensor_from_tensor_list(samples_c)
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

        # ### features used to calcate loss      
        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        # content_feats = self.encode_with_intermediate(samples_c)
        # style_feats = self.encode_with_intermediate(samples_s)

        # Linear projection
        style = self.embedding((samples_s.tensors) / 1.0)
        content = self.embedding((samples_c.tensors) / 1.0)

        # postional embedding is calculated in transformer.py
        pos_s = None
        pos_c = None

        mask = None
        # 过transformer的关键部分,hs.shape[8,512,32,32]
        hs = self.transformer(style, mask, content, pos_c, pos_s)
        # Ics.shape[8,3,256,256]
        Ics = self.decode(hs)

        # import matplotlib.pyplot as plt
        # plt.imshow(Ics.numpy())
        # plt.show()

        Ics_feats = self.encode_with_intermediate(Ics)
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(
            content_feats[-1]))+self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        # Style loss
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

        Icc = self.decode(self.transformer(
            content, mask, content, pos_c, pos_c))
        Iss = self.decode(self.transformer(style, mask, style, pos_s, pos_s))

        # Identity losses lambda 1
        loss_lambda1 = self.calc_content_loss(
            Icc, content_input)+self.calc_content_loss(Iss, style_input)

        # Identity losses lambda 2
        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(
            Icc_feats[0], content_feats[0])+self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(
                Icc_feats[i], content_feats[i])+self.calc_content_loss(Iss_feats[i], style_feats[i])
        # Please select and comment out one of the following two sentences

        '''
        ying*************************************************************************************
        add new loss--paired output and input to caltulate loss
        '''
        loss_l1_function = torch.nn.L1Loss()
        loss_l1 = loss_l1_function(samples_s.tensors, Ics)
        # return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2   #train
        return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2, loss_l1

        # return Ics    #test


if __name__ == '__main__':
    import os
    import argparse
    import models.transformer as transformer
    import cv2

    parser = argparse.ArgumentParser()
    '''
    ying*************************************************************************************************
    change the parset options of code
    '''
    parser.add_argument('--data_path', default="D:/zy/project/virtual_staining/data/kidney_trans/", type=str,
                        help='Directory path to a batch of content & style images')
    parser.add_argument('--epochs', default=100, type=int,
                        help='epochs of training')

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

    test_image = io.imread(
        r'D:\zy\project\virtual_staining\data\kidney_trans\A\test/000009.png')

    # transpose test_image from (256,256,3) to (3,256,256)
    test_image = test_image.transpose(2, 0, 1)
    print(test_image.shape)

    vgg = vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = decoder
    embedding = PatchEmbed()

    Trans = transformer.Transformer()

    network = StyTrans(vgg, decoder, embedding, Trans, args)

    # content_images = torch.from_numpy(test_image)
    # style_images = torch.from_numpy(test_image)
    
    content_images = torch.randn(4,3,256,256)
    style_images = torch.randn(4,3,256,256)
    out, loss_c, loss_s, l_identity1, l_identity2, l_l1 = network(
        content_images, style_images)

    print('finished')

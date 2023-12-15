# visualize the loss curve and others using tensorboard 
# https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B8%83%E7%AB%A0/7.3%20%E4%BD%BF%E7%94%A8TensorBoard%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B.html
'''
using tips:
1. visualize the loss curve during training process

2. visualize the loss curve using the saved files after training
 using the logs generated during training--events.out.tfevents format files
 
 enable tensorboard visual events.out.tfevents file service
 --the command format:tensorboard --logdir=日志所在的目录路径 --port=8008
 notes:
    The directory where the log is located refers to the directory folder of the log, not the path of the log itself.
    the port can be changed; Modify the number after --port= to visualize multiple different events.out.tfevents files at the same time
'''

from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', default='./logs', help='Directory to save the log of tensorboard')


args = parser.parse_args()

if __name__ == "__main__":
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # add the loss curve--i:the iter of current iter
    writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    writer.add_scalar('loss_pairedimage', l_l1.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    # visualize the model structure---to do:test the code and results
    '''
    1. define the model
    2.Given an input data, the structure of the model is obtained after forward propagation, 
      and then visualized through TensorBoard, using add_graph
    '''
    import torch.nn as nn
    import torch

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
            self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
            self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
            self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
            self.flatten = nn.Flatten()
            self.linear1 = nn.Linear(64,32)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(32,1)
            self.sigmoid = nn.Sigmoid()

        def forward(self,x):
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.adaptive_pool(x)
            x = self.flatten(x)
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            y = self.sigmoid(x)
            return y

    model = Net()
    print(model)
    
    writer.add_graph(model, input_to_model = torch.rand(1, 3, 224, 224))
    writer.close()
    
    
    # visualize the images
    '''
    using the command--add_image or add_images
    '''
    import torchvision
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform_train = transforms.Compose(
        [transforms.ToTensor()])
    transform_test = transforms.Compose(
        [transforms.ToTensor()])

    train_data = datasets.CIFAR10(".", train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(".", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    images, labels = next(iter(train_loader))
    
    # 仅查看一张图片
    writer = SummaryWriter('./pytorch_tb')
    writer.add_image('images[0]', images[0])
    writer.close()
    
    # 将多张图片拼接成一张图片，中间用黑色网格分割
    # create grid of images
    writer = SummaryWriter('./pytorch_tb')
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('image_grid', img_grid)
    writer.close()
    
    # 将多张图片直接写入
    writer = SummaryWriter('./pytorch_tb')
    writer.add_images("images",images,global_step = 0)

    # visualize the parameter distribution
    '''
    When we need to study changes in parameters (or vectors) or study their distribution, 
    we can easily use TensorBoard to visualize it through add_histogram.
    '''
    import torch
    import numpy as np

    # 创建正态分布的张量模拟参数矩阵
    def norm(mean, std):
        t = std * torch.randn((100, 20)) + mean
        return t
    
    writer = SummaryWriter('./pytorch_tb/')
    for step, mean in enumerate(range(-10, 10, 1)):
        w = norm(mean, 1)
        writer.add_histogram("w", w, step)
        writer.flush()

    
    

    writer.close()


    '''
    Using TensorBoard on the server side:
    https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B8%83%E7%AB%A0/7.3%20%E4%BD%BF%E7%94%A8TensorBoard%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B.html
    
    using mobaxterm or ssh
    1.SSH
    该方法是将服务器的6006端口重定向到自己机器上来，我们可以在本地的终端里输入以下代码：其中16006代表映射到本地的端口，6006代表的是服务器上的端口。
    ssh -L 16006:127.0.0.1:6006 username@remote_server_ip
    在服务上使用默认的6006端口正常启动tensorboard
    tensorboard --logdir=xxx --port=6006
    在本地的浏览器输入地址
    127.0.0.1:16006 或者 localhost:16006
    
    '''



# visualize the loss curve and others using wandb

"""

WITHOUT REPRESENTATION LEARNING
"""

import torch
import imageio
import numpy as np
import torch.nn as nn
import torchvision, torchvision.transforms as transforms
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import Unet, sample_IADB

batchsize = 64
dataset_path = '..'
dataset = torchvision.datasets.MNIST(dataset_path, train=True, download=False,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ]))
dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=4, drop_last=True, shuffle=True)   

D = Unet().to('cuda')
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0005)
num_epochs = 200
#training loop 
for period in range(num_epochs):
    avg_d_loss = 0
    num_items = 0

    for i, batch in tqdm (enumerate(dataloader), "period" +str(period)):

        mnist = -1 + 2*batch[0].to('cuda')
        mnist = torch.nn.functional.interpolate(mnist, size=(32,32),mode='bilinear', align_corners=False)

        x_0 = torch.randn(batchsize, 1, 32, 32, device='cuda')
        x_1 = mnist
        alpha = torch.rand(batchsize, device='cuda')
        x_alpha = (1-alpha[:,None, None, None])*x_0 + alpha[:,None, None, None]*x_1

        loss = torch.sum((D(x_alpha, alpha)- (x_1 - x_0))**2)
        avg_d_loss += loss
        num_items += 1

        optimizer_D.zero_grad()
        loss.backward()
        optimizer_D.step()

    avg_d_loss /= num_items*batchsize

    if (period+1)%20 == 0:
        torch.save(D.state_dict(),f'/home/rajan/Desktop/rajan_thesis/thesis_work/FID_score_evaluation/mnist_without_rep_learning/checkpoints/d_epoch{period}.ckpt')
        sample_IADB(D,period)
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class DiffusionNet(nn.Module):
    def __init__(self, channels=[32,64,128,256]):
        super().__init__()

        self.conv1 = nn.Conv2d(1          ,channels[0],kernel_size=3,stride=1,bias=False)
        self.conv2 = nn.Conv2d(channels[0],channels[1],kernel_size=3, stride=2,bias=False)
        self.conv3 = nn.Conv2d(channels[1],channels[2],kernel_size=3,stride=2,bias=False)
        self.conv4 = nn.Conv2d(channels[2],channels[3],kernel_size=3,stride=2,bias=False)


        self.upconv1 = nn.ConvTranspose2d(channels[3],channels[2],kernel_size=3, stride=2, bias=False)
        self.upconv2 = nn.ConvTranspose2d(channels[2],channels[1],kernel_size=3, stride=2, bias=False)
        self.upconv3 = nn.ConvTranspose2d(channels[1],channels[0],kernel_size=3, stride=2, bias=False) 
        self.upconv4 = nn.ConvTranspose2d(channels[0],1         , kernel_size=3, stride=1, bias=False)

        
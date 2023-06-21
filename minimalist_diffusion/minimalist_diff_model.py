import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Architecture of the model -- MNIST use case
class DiffusionNet(nn.Module):
    def __init__(self, channels=[32,64,128,256]):
        super().__init__()

        #convolutional blocks --> down resolution 
        self.conv1 = nn.Conv2d(1          ,channels[0],kernel_size=3,stride=1,bias=False)
        self.conv2 = nn.Conv2d(channels[0],channels[1],kernel_size=3, stride=2,bias=False)
        self.conv3 = nn.Conv2d(channels[1],channels[2],kernel_size=3,stride=2,bias=False)
        self.conv4 = nn.Conv2d(channels[2],channels[3],kernel_size=3,stride=2,bias=False)

        #convolutional blocks --> up resolution (residual blocks)
        self.upconv1 = nn.ConvTranspose2d(channels[3]            ,channels[2],kernel_size=3, stride=2, bias=False) 
        self.upconv2 = nn.ConvTranspose2d(channels[2]+channels[2],channels[1],kernel_size=3, stride=2, bias=False,output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(channels[1]+channels[1],channels[0],kernel_size=3, stride=2, bias=False,output_padding=1) 
        self.upconv4 = nn.ConvTranspose2d(channels[0]+channels[0],1         , kernel_size=3, stride=1, bias=False)

        
        #group-norm layers for downsampling 
        self.gnorm1 = nn.GroupNorm( 4, channels[0])
        self.gnorm2 = nn.GroupNorm(32, channels[1])
        self.gnorm3 = nn.GroupNorm(32, channels[2])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        #group-norm layers for upsampling
        self.gnorm5 = nn.GroupNorm( 32, channels[2]) 
        self.gnorm6 = nn.GroupNorm( 32, channels[1]) 
        self.gnorm7 = nn.GroupNorm( 32, channels[0]) 

        #activation layers 
        self.act_relu = lambda x : F.relu(x)
        self.act_swish = lambda x : x * torch.sigmoid(x)
        
    def forward(self,x):
        
        #downsampling
        h1 = self.conv1(x)
        h1 = self.gnorm1(h1)
        h1 = self.act_relu(h1)

        h2 = self.conv2(h1)
        h2 = self.gnorm2(h2)
        h2 = self.act_relu(h2)

        h3 = self.conv3(h2)
        h3 = self.gnorm3(h3)
        h3 = self.act_relu(h3)

        h4 = self.conv4(h3)
        h4 = self.gnorm4(h4)
        h4 = self.act_relu(h4)

        #upsampling
        h = self.upconv1(h4)
        h = self.gnorm5(h)
        h = self.act_relu(h)

        h = self.upconv2(torch.cat([h, h3],dim=1))
        h = self.gnorm6(h)
        h = self.act_relu(h)

        h = self.upconv3(torch.cat([h,h2],dim=1))
        h = self.gnorm7(h)
        h = self.act_relu(h)

        h = self.upconv4(torch.cat([h,h1],dim=1))
        h = self.act_swish(h)
    
        return h
    




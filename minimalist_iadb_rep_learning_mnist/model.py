import torch
import imageio
import numpy as np
import torch.nn as nn
import torchvision, torchvision.transforms as transforms
from tqdm import tqdm 
from torch.utils.data import DataLoader


def saveImage(filename, image):
    imageTMP = np.clip(image * 255.0, 0, 255).astype('uint8')
    imageio.imwrite(filename, imageTMP)


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)[..., None, None]


# sampling loop
@torch.no_grad()
def sample_IADB(D,VAE,  period, latent_dims=2):
    # starting points x_alpha = x_0
    batchsize = 64
    x_0 = torch.randn(batchsize, 1, 32, 32, device="cuda")
    x_alpha = x_0
    embedding = torch.randn(batchsize, latent_dims, device='cuda')

    # loop
    T = 128
    for t in tqdm(range(T), "sampling loop"):

        # current alpha value
        alpha = t / T * torch.ones(batchsize, device="cuda")
        embed = VAE.decoder(embedding)
        
        # update 
        x_alpha = x_alpha + 1/T * D(x_alpha, alpha,embed)

        # create result image
        result = np.zeros((8*32, 8*32, 3))         
        for i in range(8):
            for j in range(8):
                tmp = 0.5+0.5*x_alpha[(i+8*j)%batchsize, ...].repeat(3,1,1).detach().cpu().clone().numpy()
                tmp = np.swapaxes(tmp, 0, 2)
                tmp = np.swapaxes(tmp, 0, 1)
                result[32*i:32*i+32, 32*j:32*j+32, :] = tmp          
    saveImage('/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_iadb_rep_learning_mnist/samples/generated_mnist_'+str(t)+'_'+str(period)+'_.png', result)

class Unet(torch.nn.Module):
    def __init__(self,embedding_dim=1024):
        super(Unet, self).__init__()

        self.dense_rep1 = Dense(embedding_dim, 64) 
        self.dense_rep2 = Dense(embedding_dim, 64) 
        self.dense_rep3 = Dense(embedding_dim, 64) 
        self.dense_rep4 = Dense(embedding_dim, 64)
        self.dense_rep5 = Dense(embedding_dim, 64)
        self.dense_rep6 = Dense(embedding_dim, 64) 

        # block down 1
        self.block1_conv1 = torch.nn.Conv2d( 2, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block1_conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2)
        # block down 2
        self.block2_conv1 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block2_conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2)
        # block down 3
        self.block3_conv1 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block3_conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block3_conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block3_conv4 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2)
        # block up 3
        self.block3_up1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2, output_padding=1)
        self.block3_up2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        # block up 2
        self.block2_up1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2, output_padding=1)
        self.block2_up2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        # block up 1
        self.block1_up1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2, output_padding=1)
        self.block1_up2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        # output
        self.conv_output = torch.nn.Conv2d(64, 1, kernel_size=(1,1), padding=(0,0), padding_mode='zeros', stride=1)
        #
        self.relu = torch.nn.ReLU()

    def forward(self, x, alpha, embed):

        b0 = torch.cat([x, alpha[:,None,None,None].repeat(1, 1, 32, 32)], dim=1)

        b1_c1 = self.relu(self.block1_conv1(b0))
        b1_c2 = self.relu(self.block1_conv2(b1_c1)+self.dense_rep1(embed))

        b2_c1 = self.relu(self.block2_conv1(b1_c2))
        b2_c2 = self.relu(self.block2_conv2(b2_c1)+self.dense_rep2(embed))

        b3_c1 = self.relu(self.block3_conv1(b2_c2))
        b3_c2 = self.relu(self.block3_conv2(b3_c1))
        b3_c3 = self.relu(self.block3_conv3(b3_c2)+self.dense_rep3(embed)) + b3_c1
        b3_c4 = self.relu(self.block3_conv4(b3_c3))

        u2_c1 = self.relu(self.block3_up1(b3_c4)) + b3_c3
        u2_c2 = self.relu(self.block3_up2(u2_c1)+self.dense_rep4(embed)) + b2_c2

        u1_c1 = self.relu(self.block2_up1(u2_c2)) + b1_c2
        u1_c2 = self.relu(self.block2_up2(u1_c1)+self.dense_rep5(embed))

        u0_c1 = self.relu(self.block1_up1(u1_c2)) + b1_c1
        u0_c2 = self.relu(self.block1_up2(u0_c1)+self.dense_rep6(embed))

        output = self.conv_output(u0_c2)

        return output


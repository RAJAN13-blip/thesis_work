
import torch
import imageio
import numpy as np
import torch.nn as nn
import torchvision, torchvision.transforms as transforms
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import Unet, sample_IADB
from augment import VariationalEncoder

batchsize = 64
dataset = torchvision.datasets.MNIST('.', train=True, download=False,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ]))
dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=4, drop_last=True, shuffle=True)   

latent_dims = 5
D = Unet().to('cuda')
VAE = VariationalEncoder(latent_dims=latent_dims).to('cuda')
optimizer_DVAE = torch.optim.Adam(list(D.parameters())+ list(VAE.parameters()), lr=0.0005)
gamma = 0
loss_kl_list = []
loss_d_list = []
num_epochs = 50
# training loop
for period in range(num_epochs):
    avg_kl_loss = 0
    avg_d_loss = 0
    num_items = 0
    for batch in tqdm(dataloader, "period " + str(period)):

        # get data      
        mnist = -1 + 2*batch[0].to("cuda")
        mnist = torch.nn.functional.interpolate(mnist, size=(32,32), mode='bilinear', align_corners=False)
        
        # 
        x_0 = torch.randn(batchsize, 1, 32, 32, device="cuda")
        x_1 = mnist            
        embed = VAE(x_1)
        alpha = torch.rand(batchsize, device="cuda")
        x_alpha = (1-alpha[:,None,None,None]) * x_0 + alpha[:,None,None,None] * x_1

        #
        loss = torch.sum( (D(x_alpha, alpha,embed) - (x_1-x_0))**2 ) + (gamma)*VAE.encoder.kl
        loss_kl = VAE.encoder.kl
        loss_d = loss.item() - loss_kl

        avg_kl_loss += loss_kl
        avg_d_loss += loss_d
        num_items +=1 

        optimizer_DVAE.zero_grad()
        loss.backward()
        optimizer_DVAE.step()

    avg_kl_loss /= num_items
    avg_d_loss /= num_items
    loss_kl_list.append(avg_kl_loss.item())
    loss_d_list.append(avg_d_loss.item())

    if period%10 == 0 or period==num_epochs-1:
        torch.save(VAE.state_dict(), f'/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_iadb_rep_learning_mnist/checkpoints/vae_epoch{period}_gamma{gamma}_ld{latent_dims}.ckpt')
        torch.save(D.state_dict(),f'/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_iadb_rep_learning_mnist/checkpoints/d_epoch{period}_gamma{gamma}_ld{latent_dims}.ckpt')
        sample_IADB(D,VAE, period, latent_dims=latent_dims)


plt.plot(loss_kl_list)
plt.plot(loss_d_list)
plt.show()
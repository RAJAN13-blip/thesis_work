from model import Unet, saveImage
from augment import VariationalEncoder
import torch
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader

D = Unet()
VAE = VariationalEncoder()
epoch = 100
ckpt_d = D.load_state_dict(torch.load(f'/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_iadb_rep_learning_mnist/checkpoints/d_{epoch}.ckpt'))
ckpt_vae = VAE.load_state_dict(torch.load(f'/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_iadb_rep_learning_mnist/checkpoints/vae_{epoch}.ckpt'))
D = D.to('cuda')
VAE = VAE.to('cuda')

# def plot_latent(autoencoder, data, num_batches=100):
#     for i, (x, y) in enumerate(data):
#         z = autoencoder.encoder(x.to('cuda'))
#         z = z.to('cpu').detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             plt.show()
#             break

# batchsize = 64
# dataset = torchvision.datasets.MNIST('..', train=True, download=False,
#                             transform=torchvision.transforms.Compose([
#                             torchvision.transforms.ToTensor(),
#                             torchvision.transforms.Normalize(
#                                 (0.1307,), (0.3081,))
#                             ]))
# dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=4, drop_last=True, shuffle=True)   


# plot_latent(VAE, dataloader)

# sampling loop
@torch.no_grad()
def sample_IADB(D,VAE:VariationalEncoder,  period=50):
    # starting points x_alpha = x_0
    batchsize = 64
    x_0 = torch.randn(batchsize, 1, 32, 32, device="cuda")
    x_alpha = x_0
    z = torch.tensor([1,0.3506]).repeat(batchsize,1)
    embedding = z.to(device='cuda')
    # embedding = (0.5588)*torch.ones(batchsize, 2, device='cuda')

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
        saveImage('/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_iadb_rep_learning_mnist/sampler_test/generated_mnist_'+str(t)+'_'+str(period)+'_.png', result)


sample_IADB(D, VAE, period=epoch)
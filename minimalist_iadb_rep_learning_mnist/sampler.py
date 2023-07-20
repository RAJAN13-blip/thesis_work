from model import Unet, saveImage
from augment import VariationalEncoder
import torch
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pandas as pd
import plotly.express as px
import plotly.io as pio
from tsnecuda import TSNE

D = Unet()
latent_dims = 5
VAE = VariationalEncoder(latent_dims=latent_dims)
epoch = 49
gamma = 0
ckpt_d = D.load_state_dict(torch.load(f'/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_iadb_rep_learning_mnist/checkpoints/d_epoch{epoch}_gamma{gamma}_ld{latent_dims}.ckpt'))
ckpt_vae = VAE.load_state_dict(torch.load(f'/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_iadb_rep_learning_mnist/checkpoints/vae_epoch{epoch}_gamma{gamma}_ld{latent_dims}.ckpt'))
D = D.to('cuda')
VAE = VAE.to('cuda')

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to('cuda'))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            plt.show()
            break

batchsize = 64
dataset = torchvision.datasets.MNIST('..', train=True, download=False,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ]))
dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=4, drop_last=True, shuffle=True)   


def plot_tsne_latent(autoencoder, data, num_batches = 100):
    latent_mnist = []
    target = []
    for i , (x, y) in enumerate(data):
        z_means = autoencoder.encoder(x.to('cuda'))
        latent_mnist.extend(z_means.to('cpu').detach().numpy())
        target.extend(y.numpy())

    latent = np.array(latent_mnist)
    target = np.array(target)
    X = TSNE(n_components=2, perplexity = 50, learning_rate=20).fit_transform(latent)

    
    data = np.vstack((X.T, target)).T
    df = pd.DataFrame(data=data, columns=["z1", "z2", "label"])
    df["label"] = df["label"].astype(str)

    fig = px.scatter(df, x="z1", y="z2", color="label")
    # plt.show()

    pio.write_html(fig, file="vis3.html", auto_open=True)


# plot_latent(VAE, dataloader)

plot_tsne_latent(VAE, dataloader)

# sampling loop
@torch.no_grad()
def sample_IADB(D,VAE:VariationalEncoder,  period=50):
    # starting points x_alpha = x_0
    batchsize = 64
    x_0 = torch.randn(batchsize, 1, 32, 32, device="cuda")
    x_alpha = x_0
    # z0 = torch.tensor([0.0]).repeat(batchsize,1)
    # z1 = torch.linspace(-0.78004, -0.6233, batchsize).reshape(-1,1)
    # z = torch.concatenate([z0, z1], dim=1)

    # # z = torch.tensor([1,0.3506]).repeat(batchsize,1)
    # embedding = z.to(device='cuda')
    embedding = (0.5588)*torch.ones(batchsize, 5, device='cuda')

    # loop
    T = 128
    for t in tqdm(range(T), "sampling loop"):

        # current alpha value
        alpha = t / T * torch.ones(batchsize, device="cuda")
        embed = VAE.decoder(embedding)
        
        # update 
        x_alpha = x_alpha + 1/T * D(x_alpha, alpha,embed)

        # create result image
        # result = np.zeros((10*32, 10*32, 3))         
        # for i in range(10):
        #     for j in range(10):
        #         tmp = 0.5+0.5*x_alpha[(i+10*j)%batchsize, ...].repeat(3,1,1).detach().cpu().clone().numpy()
        #         tmp = np.swapaxes(tmp, 0, 2)
        #         tmp = np.swapaxes(tmp, 0, 1)
        #         result[32*i:32*i+32, 32*j:32*j+32, :] = tmp          
        # saveImage('/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_iadb_rep_learning_mnist/sampler_test/generated_mnist_'+str(t)+'_'+str(period)+'_.png', result)
    samples = x_alpha.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow = batchsize)

    plt.figure(figsize=(200,200))
    plt.imshow(sample_grid.permute(1,2,0).cpu(), vmin=0., vmax=1.)
    plt.show()

sample_IADB(D, VAE, period=epoch)
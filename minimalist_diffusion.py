import torch 
import torch.nn as nn
import torch.nn.functional as F
from  torchvision.datasets import MNIST
import numpy as np

from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm 

from minimalist_diff_model import DiffusionNet,device

#Setting up the parameters
batch_size = 32
num_epochs = 50
learning_rate = 1e-4
model = torch.nn.DataParallel(DiffusionNet())
model = model.to(device=device)
datasets = MNIST('.',train=True,transform=transforms.ToTensor(),download=True)
data_loader = DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True,num_workers=4)
optimizer = Adam(model.parameters(),lr=learning_rate)
time_steps = 100


#define an alpha schedule 
def alpha_linear_schedule(steps):
    """steps : number of steps
    """
    return torch.linspace(0,1,steps)

#defining the loss function
def loss_fn(model:DiffusionNet,
            x1:torch.tensor):
    """model : the minimialistic diffusion model
          x1 : tensor for manipulating data (alpha deblending)
    """
    x0 = torch.randn_like(x1)
    alpha = torch.randn(1,).uniform_(0,1)
    
    blended_x = (1. - alpha)*x0 + (alpha)*x1
    model_output = model(blended_x)
    loss = torch.mean(torch.square(model_output - (x1-x0)))
    return loss


#Training loop 
for epoch in range(num_epochs):
    loss = 0
    num_items = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        loss = loss_fn(model,x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_items += x.shape[0]
    print('Average Loss: {:5f}'.format(loss / num_items))
  # Update the checkpoint after each epoch of training.
    torch.save(model.state_dict(), 'ckpt.pth')


#for sampling procedure
alpha_schedule = alpha_linear_schedule(time_steps)


#Sampling steps 
def sampler():
    pass

def improved_sampler():
    pass
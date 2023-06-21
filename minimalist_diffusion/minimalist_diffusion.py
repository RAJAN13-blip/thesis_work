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
batch_size = 128
num_epochs = 30
learning_rate = 1e-4
model = torch.nn.DataParallel(DiffusionNet())
model = model.to(device=device)
datasets = MNIST('.',train=True,transform=transforms.ToTensor(),download=True)
data_loader = DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True,num_workers=4)
optimizer = Adam(model.parameters(),lr=learning_rate,weight_decay=0.01,betas=(0.9,0.999))



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
    avg_loss = 0
    num_items = 0
    for x, y in data_loader:
        x = x.to(device)
        loss = loss_fn(model,x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_items += x.shape[0]
        avg_loss += loss.item() * x.shape[0]
    print(f'Epoch number : {epoch}')
    print('Average Loss: {:5f}'.format(avg_loss / num_items))
  # Update the checkpoint after each epoch of training.
    torch.save(model.state_dict(), f'mode_{epoch}_ckpt.pth')


from torchvision.utils import make_grid

time_steps = 300

#define an alpha schedule 
def alpha_linear_schedule(steps):
    """steps : number of steps
    """
    return torch.linspace(0,1,steps)

def cosine_schedule(alpha_schedule):
    """alpha_schedule : a linear alpha schedule of t/T where T : max non of steps
    """
    return 1. - torch.cos(alpha_schedule * (torch.pi/2.))
    
#for sampling procedure
alpha_schedule = alpha_linear_schedule(time_steps)


#Sampling steps 
def sampler(alpha_schedule,model):

    x_alpha = torch.randn(32,1,28,28)
    for i  in range(1,len(alpha_schedule)):
        x_alpha = x_alpha + (alpha_schedule[i] - alpha_schedule[i-1])*model(x_alpha)

    return x_alpha

def cosine_function(t: int, T: int):
    """ t: unit time
        T: total time steps 
    """
    return 1. - torch.cos(torch.tensor((t/T)* (torch.pi/2)))


def improved_sampler(model:DiffusionNet, steps):
    """model : 
       steps : total time steps T
    """
    x_alpha = torch.randn(32,1,28,28)
    for t in range(steps):
        x_alpha_half  =  x_alpha + (cosine_function(t+0.5,steps) - cosine_function(t,steps))* model(x_alpha)
        x_alpha       =  x_alpha + (cosine_function(t+1,steps)   - cosine_function(t,steps))* model(x_alpha_half)  

    return x_alpha


model = torch.nn.DataParallel(DiffusionNet())
ckpt = torch.load('mode_28_ckpt.pth', map_location=device)
model.load_state_dict(ckpt)

samples = sampler(alpha_schedule, model)


samples = samples.clamp(0.0, 1.0)
import matplotlib.pyplot as plt
sample_grid = make_grid(samples, nrow=int(np.sqrt(32)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1,2,0).cpu(), vmin=0., vmax=1.)
plt.show()
from minimalist_diff_rl_model import minimalDiffRl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from collections import OrderedDict
import torchvision.transforms as transforms
import numpy as np

#Setting up the parameters
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
num_epochs = 20 
learning_rate = 1e-4
model = torch.nn.DataParallel(minimalDiffRl())
model = model.to(device = device)

#load the pretrained  DiffusionNet from the checkpoint --> Note that models are being saved in the form of Datparallel objects
#so --> need to bring in a new dictionary -> 
#approach taken from : https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4

new_state_dict = OrderedDict()

"""loading parameters for the diffusion net"""

# ckpt = torch.load('./minimalist_diffusion_ckpts/mode_28_ckpt.pth')
# for k, v in ckpt.items():
#     name = k[7:] # remove module.
#     new_state_dict[name] = v

# model.module.encoder_drl.load_state_dict(new_state_dict)
    
optimizer = Adam(model.parameters(), learning_rate, weight_decay=0.01, betas=(0.9, 0.999))

#defining the loss function 
def loss_fn(model: minimalDiffRl, x1:torch.tensor, alpha_tensor:torch.tensor):
    """model : minimalDiffRl object wtih alpha embeddings
       x     : input image
    """
    x0 = torch.randn_like(x1).to(device)
    alpha = torch.randn(batch_size,).uniform_(0,1).to(device)
    blended_x  = (1.-alpha.view(-1,1,1,1))*x0 + alpha.view(-1,1,1,1)*x1
    model_output = model(blended_x, alpha)

    loss = torch.mean(torch.square(model_output - (x1-x0)))
    return loss


datasets = MNIST('.',train=True,transform=transforms.ToTensor(),download=False)
data_loader = DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True,num_workers=4)
t = torch.ones(batch_size).to(device=device)
#Training loop 
for epoch in range(num_epochs):
    avg_loss = 0
    num_items = 0
    for i , (x, y) in enumerate(data_loader):
        if x.shape[0] != batch_size:
            continue
        x = x.to(device)
        loss = loss_fn(model,x,t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_items += x.shape[0]
        avg_loss += loss.item() * x.shape[0]
    print(f'Epoch number : {epoch}')
    print('Average Loss: {:5f}'.format(avg_loss / num_items))
  # Update the checkpoint after each epoch of training.
    torch.save(model.state_dict(), f'./minimalist_drl_ckpts/model_{epoch}_ckpt.pth')


from torchvision.utils import make_grid

time_steps = 300

#define an alpha schedule 
def alpha_linear_schedule(steps):
    """steps : number of steps
    """
    return torch.linspace(0,1,steps).to(device=device)

def cosine_schedule(alpha_schedule):
    """alpha_schedule : a linear alpha schedule of t/T where T : max non of steps
    """
    return 1. - torch.cos(alpha_schedule * (torch.pi/2.)).to(device=device)
    
#for sampling procedure
alpha_schedule = alpha_linear_schedule(time_steps)


#Sampling steps 
def sampler(alpha_schedule,model):

    x_alpha = torch.randn(32,1,28,28).to(device=device)
    for i  in range(1,len(alpha_schedule)):
        alpha_tensor = torch.ones(x_alpha.shape[0])*alpha_schedule[i-1].to(device=device)
        x_alpha = x_alpha + (alpha_schedule[i] - alpha_schedule[i-1])*model(x_alpha,alpha_tensor)

    return x_alpha

def cosine_function(t: int, T: int):
    """ t: unit time
        T: total time steps 
    """
    return 1. - torch.cos(torch.tensor((t/T)* (torch.pi/2))).to(device=device)


def improved_sampler(model:minimalDiffRl, steps):
    """model : 
       steps : total time steps T
    """
    x_alpha = torch.randn(32,1,28,28).to(device=device)
    for t in range(steps):
        alpha_half_tensor = torch.ones(x_alpha.shape[0]).to(device=device)*cosine_function(t+0.5,steps)
        alpha_tensor = torch.ones(x_alpha.shape[0]).to(device=device)*cosine_function(t,steps)
        x_alpha_half  =  x_alpha + (cosine_function(t+0.5,steps) - cosine_function(t,steps))* model(x_alpha,alpha_tensor)
        x_alpha       =  x_alpha + (cosine_function(t+1,steps)   - cosine_function(t,steps))* model(x_alpha_half,alpha_half_tensor)  

    return x_alpha

#sampling procedure
model = torch.nn.DataParallel(minimalDiffRl())
ckpt = torch.load('./minimalist_drl_ckpts/model_18_ckpt.pth', map_location=device)
model.load_state_dict(ckpt)

samples = sampler(alpha_schedule, model)


samples = samples.clamp(0.0, 1.0)
import matplotlib.pyplot as plt
sample_grid = make_grid(samples, nrow=int(np.sqrt(32)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1,2,0).cpu(), vmin=0., vmax=1.)
plt.show()
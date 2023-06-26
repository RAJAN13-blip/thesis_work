import torchvision
from minimalist_diff_rl_model import minimalDiffRl
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from collections import OrderedDict
import torchvision.transforms as transforms
import numpy as np


device  = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 128
last_epoch = 1347

num_epochs = 9000
learning_rate = 1e-5
dataset_path = '/home/rajan/Desktop/thesis/thesis_work'

model_preloading_path = f'/home/rajan/Desktop/thesis/thesis_work/minimalist_diffusion_ckpts/mode_{last_epoch}_ckpt.pth'


from hugging_face_model import IADB

model = torch.nn.DataParallel(IADB(1)).to(device=device)


# new_state_dict = OrderedDict()

# """loading parameters for the diffusion net"""

# ckpt = torch.load(model_preloading_path)
# for k, v in ckpt.items():
#     name = k[7:] # remove module.
#     new_state_dict[name] = v

# model.module.pretrained_model.load_state_dict(new_state_dict)

"""loading params"""
ckpt = torch.load(model_preloading_path)
model.load_state_dict(ckpt)



# """loading parameters for continual learning"""
# ckpt_last = torch.load(model_last_trained_path)
# model.load_state_dict(ckpt_last)
    
optimizer = Adam(model.parameters(), learning_rate)

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


""" cosine functions and improved sampler support for inference during training
"""

def cosine_function(t: int, T: int):
    """ t: unit time
        T: total time steps 
    """
    return 1. - torch.cos(torch.tensor((t/T)* (torch.pi/2))).to(device=device)


def improved_sampler(model, steps):
    """model : 
       steps : total time steps T
    """
    x_alpha = torch.randn(32,1,28,28).to(device=device)
    ones = torch.ones(x_alpha.shape[0]).to(device=device)
    for t in range(steps):
        alpha_half_tensor = ones*cosine_function(t+0.5,steps)
        alpha_tensor = ones*cosine_function(t,steps)
        x_alpha_half  =  x_alpha + (cosine_function(t+0.5,steps) - cosine_function(t,steps))* model(x_alpha,alpha_tensor)
        x_alpha       =  x_alpha + (cosine_function(t+1,steps)   - cosine_function(t,steps))* model(x_alpha_half,alpha_half_tensor)  

    return torch.relu(x_alpha)

datasets = MNIST(dataset_path,train=True,transform=transforms.ToTensor(),download=False)
data_loader = DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True,num_workers=4)
t = torch.ones(batch_size).to(device=device)
#Training loop 
for epoch in range(last_epoch, last_epoch+num_epochs):
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
    if epoch%100 == 0 :
        torch.save(model.state_dict(),f'/home/rajan/Desktop/thesis/thesis_work/minimalist_diffusion_ckpts/mode_{epoch}_ckpt.pth')


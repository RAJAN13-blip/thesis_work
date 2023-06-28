from model import MINIMAL
from data import generate_data, PROB
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
p0, p1 = generate_data() #generates a list of 2D arrays 

num_epochs = 1000
batch_size = 1
last_epoch = 0
dataset = PROB(p0[0], p1[0])
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
model = torch.nn.DataParallel(MINIMAL()).to(device=device)
learning_rate = 1e-4
optimizer = Adam(model.parameters(), learning_rate)

# model_pre_loading_path = f'/home/rajan/Desktop/thesis/thesis_work/jax_based_codes/minimalist_diffusion/ckpts/model_{last_epoch}_ckpt.pth'
# ckpt = torch.load(ckpt)
# model.load_state_dict(ckpt)

for n, weights in model.named_parameters():
    if len(weights.shape) != 1:
        nn.init.xavier_normal(weights.data)
        print(n)


def cosine_function(t: int, T: int):
    """ t: unit time
        T: total time steps 
    """
    return 1. - torch.cos(torch.tensor((t/T)* (torch.pi/2))).to(device=device)


def improved_sampler(model, steps, prob_density_0):
    """model : 
       steps : total time steps T
    """
    elem = []
    for x, _ in prob_density_0:
        elem.append(x[0].numpy())

    arr = np.array(elem)
    x_alpha = torch.from_numpy(arr).to(device=device)

    for t in range(steps):
        alpha_half = cosine_function(t+0.5, steps).reshape(1,)
        alpha = cosine_function(t,steps).reshape(1,)
        x_alpha_half  =  x_alpha + (cosine_function(t+0.5,steps) - cosine_function(t,steps))* model(x_alpha,alpha)
        x_alpha       =  x_alpha + (cosine_function(t+1,steps)   - cosine_function(t,steps))* model(x_alpha_half,alpha_half)  

    return x_alpha

def loss_function(model, X0, X1):
    alpha = torch.randn(1).uniform_(0,1).to(device)
    X_alpha = (1. - alpha)*X0 + (alpha)*X1
    D_theta = model(X_alpha, alpha)
    loss = torch.mean(torch.square(D_theta - (X1 - X0)))
    return loss


for epoch in range(num_epochs):
    avg_loss = 0
    num_items = 0
    for i, (X0, X1) in enumerate(dataloader):
        X0 = X0.to(device)
        X1 = X1.to(device)

        loss = loss_function(model, X0, X1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_items += 1
        avg_loss += loss.item()* batch_size
    print(f'Epoch number : {epoch}')
    print('Average Loss: {:7f}'.format(avg_loss / num_items))

    if epoch % 100 == 0:
        torch.save(model.state_dict(),f'/home/rajan/Desktop/thesis/thesis_work/jax_based_codes/minimalist_diffusion/ckpts/model_{epoch}_ckpt.pth')
        with torch.no_grad():
            p1 = improved_sampler(model, 128, dataloader)
            p1_arr = p1.cpu().numpy()

            plt.scatter(p1_arr[:,0],p1_arr[:,1],s=1)
            plt.savefig(f'/home/rajan/Desktop/thesis/thesis_work/jax_based_codes/minimalist_diffusion/p1_images/p1_trained_image{epoch}.png')

            


    

    

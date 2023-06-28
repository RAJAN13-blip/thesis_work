import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class alpha_step_embedding_sinusoidal(nn.Module):
    """For encoding time steps"""
    def __init__(self, embed_dim):
        super().__init__()
        # Non trainable parameters --> weights are fixed during optimization
        self.w = nn.Parameter(torch.randn(embed_dim//2), requires_grad=False)
    def forward(self,x):
        x_proj = x[:,None]* self.w[None,:]*2*np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class alpha_step_embedding(nn.Module):
   def __init__(self,embed_dim):
      super().__init__()
      """A non trainable parameter --> just to extend 
      """
      self.w = nn.Parameter(torch.randn(embed_dim),requires_grad=False)
   
   def forward(self,x):
      x_proj = x[:,None]*self.w[None,:]
      return x_proj
    
class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)
    



class MINIMAL(nn.Module):
    def __init__(self, channels=[64,64,64,64,64],embed_dim=256):
        super().__init__()
        self.embed = nn.Sequential(alpha_step_embedding(embed_dim=embed_dim),
           nn.Linear(embed_dim,embed_dim)
        )
        self.dense_1 = nn.Linear(2, channels[0])
        self.dense_2 = nn.Linear(channels[0],channels[1])
        self.dense_3 = nn.Linear(channels[1],channels[2])
        self.dense_4 = nn.Linear(channels[2],channels[3])
        self.dense_5 = nn.Linear(channels[3],channels[4])
        self.dense_6 = nn.Linear(channels[4], 2)
        self.act_relu = lambda x: F.relu(x)
        self.act_sigmoid = lambda x : x* torch.sigmoid(x)

        self.dense_embed = Dense(embed_dim, channels[0])

    def forward(self,x,alpha):
        embed = self.embed(alpha)
        h1 = self.dense_1(x)
        h1+= self.dense_embed(embed)
        h1 = self.act_sigmoid(h1)
        h2 = self.dense_2(h1)
        h2+= self.dense_embed(embed)
        h2 = self.act_sigmoid(h2)
        h3 = self.dense_3(h2)
        h3+= self.dense_embed(embed)
        h3 = self.act_sigmoid(h3)
        h4 = self.dense_4(h3)
        h4+= self.dense_embed(embed)
        h4 = self.act_relu(h4)
        h5 = self.dense_5(h4)
        h5+= self.dense_embed(embed)
        h5 = self.act_relu(h5)
        h6 = self.dense_6(h5)
        h6 = self.act_relu(h6)
        return h6

# o = torch.randn(1,2).to(device='cuda:0')
# alpha = torch.randn(1).to(device='cuda:0')
# model = torch.nn.DataParallel(MINIMAL()).to(device='cuda:0')
# a = model(o, alpha)
import torch 
import torch.nn.functional as F
import torch.nn as nn
from minimalist_diff_model import DiffusionNet
# from ..diff_rep_learning.augment import Encoder

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
    return self.dense(x)[..., None, None]
    


class minimalDiffRl(nn.Module):
    def __init__(self,embed_dim=256,channels=[32,64,128,256]):
        super().__init__()
        self.embed = nn.Sequential(alpha_step_embedding(embed_dim=embed_dim),
           nn.Linear(embed_dim,embed_dim)
        )
        self.dense_embed1 = Dense(embed_dim, channels[0])
        self.dense_embed2 = Dense(embed_dim, channels[1])
        self.dense_embed3 = Dense(embed_dim, channels[2])
        self.dense_embed4 = Dense(embed_dim, channels[3])
        self.dense_embed5 = Dense(embed_dim, channels[2])
        self.dense_embed6 = Dense(embed_dim, channels[1])
        self.dense_embed7 = Dense(embed_dim, channels[0])

        self.encoder_drl = DiffusionNet()


    def forward(self, x, alpha):
        #embedding 
        embed = self.embed(alpha)

        #downsampling
        h1 = self.encoder_drl.conv1(x)
        h1+= self.dense_embed1(embed)
        h1 = self.encoder_drl.gnorm1(h1)
        h1 = self.encoder_drl.act_relu(h1)

        h2 = self.encoder_drl.conv2(h1)
        h2+= self.dense_embed2(embed)
        h2 = self.encoder_drl.gnorm2(h2)
        h2 = self.encoder_drl.act_relu(h2)

        h3 = self.encoder_drl.conv3(h2)
        h3+= self.dense_embed3(embed)
        h3 = self.encoder_drl.gnorm3(h3)
        h3 = self.encoder_drl.act_relu(h3)

        h4 = self.encoder_drl.conv4(h3)
        h4+= self.dense_embed4(embed)
        h4 = self.encoder_drl.gnorm4(h4)
        h4 = self.encoder_drl.act_relu(h4)

        #upsampling
        h = self.encoder_drl.upconv1(h4)
        h+= self.dense_embed5(embed)
        h = self.encoder_drl.gnorm5(h)
        h = self.encoder_drl.act_relu(h)


        h = self.encoder_drl.upconv2(torch.cat([h,h3],dim=1))
        h+= self.dense_embed6(embed)
        h = self.encoder_drl.gnorm6(h)
        h = self.encoder_drl.act_relu(h)

        h = self.encoder_drl.upconv3(torch.cat([h,h2],dim=1))
        h+= self.dense_embed7(embed)
        h = self.encoder_drl.gnorm7(h)
        h = self.encoder_drl.act_relu(h)

        h = self.encoder_drl.upconv4(torch.cat([h,h1],dim=1))
        h = self.encoder_drl.act_swish(h)

        return h


